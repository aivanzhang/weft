"""Test one fixed source-derived crop on every remaining unusable pose, locally."""

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2

from weft.benchmark import seal_suite
from weft.exercise_vision import _detector, _draw_pose, _save_previews, model_provenance, summarize_result
from weft.io import canonical_hash, digest, local_time, write_json
from weft.limb_fidelity import grade_limb_projection, load_annotations
from weft.pipeline import _implementation_files, verify_artifacts
from weft.pose_crop import map_crop_pose
from weft.pose_envelope import CONTRACT, apply_envelope, assess_envelope, plan_envelope
from weft.root_motion import build_root_reference
from check_crop_diversity import CLIPS, LIMB_LABELS, MODEL, PAIR_LABELS, deltas, limb_predictions
from check_selective_crop_videos import decoded_images, finish_evidence
from grade_pose_identity import grade as grade_identity


ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'artifacts/selective-crop-videos-03'
CLIP='jumping-jacks-burpees'
VIDEO=ROOT/'data/exercise/jumping-jacks-burpees.webm'


def read(path): return json.loads(path.read_text())


def worker(out):
    pre=read(out/'pre-run.json'); inputs=out/'inputs'
    frozen=verify_artifacts(inputs)
    current,context,plan=[read(inputs/name) for name in ('current.json','context.json','plan.json')]
    if plan_envelope(current,context)!=plan or digest(VIDEO)!=current['source_sha256'] or model_provenance(MODEL)!=pre['model']:
        raise ValueError('Frozen inference inputs changed')
    output=out/'candidate'; output.mkdir()
    indexed={row['frame']:row for row in current['frames']}
    requested=set(plan['targets'])|set(plan['controls'])
    previews,preview_times=[],[]
    preview_frames=set(list(indexed)[::max(1,math.ceil(len(indexed)/48))])
    proposals,decisions,controls={},{},[]
    begin=time.monotonic(); detector=None
    try:
        mp,detector=_detector(MODEL,'image')
        with (out/'crop-observations.jsonl').open('x') as journal:
            for frame,stamp,bgr in decoded_images(VIDEO,current,begin):
                if time.monotonic()-begin>85: raise TimeoutError('Envelope inference exceeded 85 seconds')
                if frame not in indexed: continue
                original=indexed[frame]
                if abs(stamp-original['decoder_timestamp_s'])>1e-6: raise ValueError('Source decoder clock changed')
                if frame==0 and not cv2.imwrite(str(output/'background.jpg'),bgr): raise OSError('Cannot save source preview')
                if frame in requested:
                    x1,y1,x2,y2=plan['box_xyxy']; pixels=bgr[y1:y2,x1:x2]
                    tick=time.monotonic()
                    raw=summarize_result(detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(pixels,cv2.COLOR_BGR2RGB))))
                    crop={'frame':frame,'box_xyxy':plan['box_xyxy']}
                    proposal=map_crop_pose(raw,crop,current,original,time.monotonic()-tick)
                    if frame in plan['targets']:
                        proposals[frame]=proposal
                        decisions[frame]=assess_envelope(current,original,proposal,plan)
                        kind='target'; decision=decisions[frame]
                    else:
                        kind='unchanged_control'; decision=None
                        controls.append({'frame':frame,'original_status':original['status'],
                            'crop_status':proposal['status'],'crop_pose_count':proposal['pose_count'],
                            'adopted':False,'original':original,'proposed':proposal})
                    journal.write(json.dumps({'frame':frame,'kind':kind,'raw_crop_result':raw,
                        'proposed':proposal,'decision':decision},allow_nan=False)+'\n'); journal.flush()
                    _draw_pose(bgr,original).save(out/f'before-{frame:04d}.png')
                    _draw_pose(bgr,proposal).save(out/f'proposal-{frame:04d}.png')
                if frame in preview_frames:
                    selected=proposals[frame] if decisions.get(frame,{}).get('accepted') else original
                    previews.append(_draw_pose(bgr,selected));preview_times.append(original['time_s'])
        candidate=apply_envelope(current,context,plan,proposals,decisions)
        quality_error=finish_evidence(candidate,output,context['frames'])
        _save_previews(previews,preview_times,output)
        write_json(out/'decisions.json',list(decisions.values()));write_json(out/'controls.json',controls)
        changed=candidate['pose_reinference']['changed_frames']
        if any(a!=b for a,b in zip(current['frames'],candidate['frames'],strict=True) if a['frame'] not in changed):
            raise ValueError('An unselected original observation changed')
        if verify_artifacts(inputs)!=frozen or digest(VIDEO)!=current['source_sha256']:
            raise ValueError('Original source or inputs changed')
        write_json(out/'construction.json',{'created_at':local_time(),'status':'candidate_frozen_before_comparison',
            'verification':seal_suite(output),'changed_frames':changed,'unselected_rows_unchanged':len(current['frames'])-len(changed),
            'quality_error':quality_error,'source_frames_decoded':current['decoded_frames'],'sampled_frames':len(current['frames']),
            'local_inference_calls':len(requested),'control_count':len(controls),
            'control_statuses':dict(Counter(row['crop_status'] for row in controls)),
            'decision_reasons':dict(Counter(reason for row in decisions.values() for reason in row['reasons'])),
            'wall_seconds':time.monotonic()-begin,'paid_requests':0})
    finally:
        if detector is not None: detector.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--worker',action='store_true');args=parser.parse_args();out=args.out.resolve()
    if args.worker: worker(out);return
    if out.is_relative_to(SOURCE): raise ValueError('Use a fresh output outside the original studies')
    verified=verify_artifacts(SOURCE)
    current=read(SOURCE/CLIP/'selected/evidence.json');context=read(SOURCE/CLIP/'whole_image/evidence.json')
    plan=plan_envelope(current,context)
    if digest(VIDEO)!=current['source_sha256']: raise ValueError('Source bytes changed')
    implementation={name:digest(path) for name,path in _implementation_files().items()}
    out.mkdir(parents=True,exist_ok=False);inputs=out/'inputs';inputs.mkdir()
    for name,value in (('current',current),('context',context),('plan',plan)):write_json(inputs/(name+'.json'),value)
    write_json(inputs/'report.json',{'status':'source_inputs_frozen','created_at':local_time(),'source_verification':verified})
    seal_suite(inputs)
    for name in ('check_jacks_pose_envelope.py','check_crop_diversity.py','check_selective_crop_videos.py','grade_pose_identity.py'):
        shutil.copyfile(Path(__file__).with_name(name),out/name)
    for name,path in _implementation_files().items():
        target=out/'implementation-source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    write_json(out/'pre-run.json',{'created_at':local_time(),'source_verification':verified,
        'model':model_provenance(MODEL),'implementation':implementation,'contract':dict(CONTRACT),
        'maximum_local_calls':len(plan['targets'])+len(plan['controls']),'process_timeout_seconds':95,
        'labels':{str(path):digest(path) for path in (PAIR_LABELS,LIMB_LABELS)},
        'scope':'Fixed source-pose envelope over the complete interval; selected evidence already contains prior crop repairs',
        'selection_exclusions':['manual labels','body/floor scores','Meta outputs'],
        'new_external_requests':0,'meta_upload':'Not executed: automatic review requires specific-media approval'})
    with (out/'worker.log').open('x') as log:
        result=subprocess.run([sys.executable,str(Path(__file__)),'--out',str(out),'--worker'],cwd=ROOT,
            env=os.environ|{'MPLCONFIGDIR':str(ROOT/'.local/matplotlib')},stdout=log,stderr=subprocess.STDOUT,timeout=95)
    if result.returncode:
        write_json(out/'report.json',{'status':'inference_failed','returncode':result.returncode,'paid_requests':0})
        seal_suite(out);raise RuntimeError('Preserved failed local study; inspect worker.log')
    construction=read(out/'construction.json'); candidate=read(out/'candidate/evidence.json')
    if any(digest(Path(path))!=value for path,value in read(out/'pre-run.json')['labels'].items()):
        raise ValueError('Comparison labels changed')
    labels={'identity':read(PAIR_LABELS),'limbs':load_annotations(LIMB_LABELS,project_root=ROOT)}
    scores={}
    for mode,replace in (('before',current),('after',candidate)):
        observations={clip:read(SOURCE/clip/'selected/evidence.json') for clip in CLIPS};observations[CLIP]=replace
        predicted,matching=limb_predictions(labels['limbs'],observations)
        scores[mode]={'identity':grade_identity(labels['identity'],observations),
            'limbs':grade_limb_projection(labels['limbs'],predicted)}
        write_json(out/(mode+'-limb-predictions.json'),{'predictions':predicted,'matching':matching})
    changes={kind:deltas(scores['before'][kind]['observations'],scores['after'][kind]['observations'],
        'passes' if kind=='identity' else 'within_tolerance') for kind in ('identity','limbs')}
    write_json(out/'scores.json',{'scores':scores,'changes':changes})
    roots={}
    for mode,evidence in (('before',current),('after',candidate)):
        audit=build_root_reference(evidence);write_json(out/(mode+'-root-motion.json'),audit)
        roots[mode]={'status':audit['status'],'errors':audit['quality']['errors'],
            'long_intervals':len(audit['quality']['long_reference_intervals']),
            'maximum_gap_s':max((row['seconds'] for row in audit['quality']['long_reference_intervals']),default=0),
            'unfilled_gaps':sum(row['status']=='unfilled' for row in audit['quality']['gaps'])}
    if verify_artifacts(SOURCE)!=verified or implementation!={name:digest(path) for name,path in _implementation_files().items()}:
        raise ValueError('Frozen source or runtime changed')
    report={'status':'whole_interval_envelope_compared','created_at':local_time(),'construction':construction,
        'before_usable':current['quality']['tracked_frames'],'after_usable':candidate['quality']['tracked_frames'],
        'root_comparison':roots,'summaries':{mode:{name:group['summary'] for name,group in groups.items()} for mode,groups in scores.items()},
        'changes':{kind:{key:value for key,value in rows.items() if key!='paired_rows'} for kind,rows in changes.items()},
        'paid_requests':0,'default_changed':False,'scope':'New inferred image hypotheses; original observations preserved, anatomy/contact unverified'}
    write_json(out/'report.json',report)
    print(json.dumps({'verification':seal_suite(out),'before_usable':report['before_usable'],
        'after_usable':report['after_usable'],'construction':construction,'root_comparison':roots,
        'changes':report['changes']},indent=2))


if __name__=='__main__':main()
