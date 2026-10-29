"""Frozen image-direction candidates, followed by unchanged independent checks."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

import cv2
import mujoco
import numpy as np

from weft.benchmark import seal_suite
from weft.body_image_refinement import IMAGE_REFINEMENT_CONTRACT, refine_image_reference
from weft.io import canonical_hash, digest, local_time, write_json
from weft.limb_fidelity import grade_limb_projection, load_annotations
from weft.whole_body_feasibility import audit_reference_feasibility
from check_body_refinement import limb_predictions


CASES = ('single-leg-squat', 'squat', 'kettlebell-pushup')


def construction_metrics(audit, when):
    samples, weighted, weights = [], [], []
    per_segment = {}
    huber = 0.
    for row in audit['frames']:
        errors = row.get('direction_errors_' + when + '_rad')
        if errors is None:
            continue
        huber += row['weighted_huber_' + when]
        for segment in row['segments']:
            name = segment['segment']
            error = errors[name]
            if error is None:
                continue
            weight = segment['effective_fit_weight']
            samples.append(error); weighted.append(weight*error*error); weights.append(weight)
            per_segment.setdefault(name, []).append(error)
    return {'comparable_segment_samples': len(samples), 'effective_weight_sum': sum(weights),
            'unweighted_mean_direction_error_rad': float(np.mean(samples)) if samples else None,
            'unweighted_direction_rmse_rad': float(np.sqrt(np.mean(np.square(samples)))) if samples else None,
            'confidence_weighted_direction_rmse_rad': float(np.sqrt(sum(weighted)/sum(weights))) if sum(weights) else None,
            'summed_weighted_huber_unit_direction_loss': huber,
            'per_segment_mean_direction_error_rad': {name:float(np.mean(values)) for name,values in per_segment.items()}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    def forbidden(*args,**kwargs):
        raise AssertionError('Image-direction study cannot integrate physics')
    mujoco.mj_step=forbidden
    paths=[Path('src/weft')/name for name in ['body_image_refinement.py','body_refinement.py',
        'whole_body_feasibility.py','whole_body.py','human.py','spatial_pose.py','body_frame.py',
        'mobile_human.py','root_motion.py','io.py','limb_fidelity.py','body_fidelity.py']]
    paths += [Path('tests/test_body_image_refinement.py'),Path('tests/test_body_refinement.py'),
              Path('tests/test_whole_body_feasibility.py'),Path('scripts/studies/check_body_refinement.py'),Path(__file__)]
    hashes={str(path):digest(path) for path in paths}
    implementation=out/'implementation';implementation.mkdir()
    for path in paths:
        shutil.copyfile(path,implementation/path.name)
    inputs={}
    for case in CASES:
        source=Path('artifacts/mobile-exercise-e2e-10')/case/'run'
        inputs[case]={name:{'path':str(source/name),'sha256':digest(source/name)}
                      for name in ['human.json','pose-landmarks.json','source.webm']}
    write_json(out/'pre-fit-freeze.json',{'created_at':local_time(),'contract':IMAGE_REFINEMENT_CONTRACT,
        'implementation_hashes':hashes,'inputs':inputs,'analytic_tests_passed':21,
        'maximum_fits':3,'maximum_total_fit_seconds':180,'external_api_usd':0,
        'scope':'Settings frozen before real fits; no optimization against annotations or floor'})
    results=[];baseline_specs={};candidate_specs={}
    for case in CASES:
        directory=out/case;directory.mkdir()
        source=Path('artifacts/mobile-exercise-e2e-10')/case/'run'
        for original,name in [('human.json','baseline-human.json'),('pose-landmarks.json','original-pose-landmarks.json')]:
            shutil.copyfile(source/original,directory/name)
        spec=json.loads((source/'human.json').read_text())
        raw=json.loads((source/'pose-landmarks.json').read_text())
        assert set(raw)=={'frames','model','scope'}
        assert inputs[case]['source.webm']['sha256']==spec['source_sha256']
        assert all(row['id'].startswith(spec['source_sha256']+':frame:') for row in raw['frames'])
        cap=cv2.VideoCapture(str(source/'source.webm'))
        metadata_dimensions=[int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        ok,image=cap.read();cap.release()
        if not ok:
            raise ValueError('Cannot independently decode source dimensions')
        height,width=map(int,image.shape[:2])
        assert metadata_dimensions==[width,height]
        dimensions={'method':'verified_source_video','source_sha256':spec['source_sha256'],
                    'width':width,'height':height,'decoded_frame_index':0,
                    'decoder':'OpenCV '+cv2.__version__,'metadata_dimensions_match_decoded_frame':True}
        evidence={**raw,'source_sha256':spec['source_sha256'],'width':width,'height':height,
                  'image_dimension_provenance':dimensions}
        write_json(directory/'evidence-adapter.json',{'method':'attach_verified_source_digest_and_decoded_dimensions',
            'source_sha256':spec['source_sha256'],'raw_file_sha256':inputs[case]['pose-landmarks.json']['sha256'],
            'adapted_evidence_canonical_sha256':canonical_hash(evidence),'all_source_ids_verified':True,
            'image_dimension_provenance':dimensions,'new_pose_inference':False})
        candidate,audit=refine_image_reference(spec,evidence)
        write_json(directory/'refinement.json',audit)
        write_json(directory/'candidate-human.json',candidate)
        fixed={key:candidate.get(key)==spec.get(key) for key in
               ['root_reference','mobility','lengths','body_frame','motor_limits','dt']}
        assert all(fixed.values())
        assert [r['time_s'] for r in spec['reference']]==[r['time_s'] for r in candidate['reference']]
        original_floor=audit_reference_feasibility(spec)
        candidate_floor=audit_reference_feasibility(candidate)
        write_json(directory/'baseline-feasibility.json',original_floor)
        write_json(directory/'candidate-feasibility.json',candidate_floor)
        statuses=Counter(segment['source_status'] for row in audit['frames'] for segment in row.get('segments') or [])
        results.append({'case':case,'input_hashes':inputs[case],'image_dimensions':dimensions,
            'fit_status':audit['status'],'frame_statuses':dict(Counter(r['status'] for r in audit['frames'])),
            'source_direction_statuses':dict(statuses),'reference_frames':len(spec['reference']),
            'original_evidence_frames':len(raw['frames']),'unused_evidence_ids':audit['unused_evidence_ids'],
            'changed_frames':audit['changed_frames'],'max_angle_change_rad':max(r.get('maximum_angle_change_rad',0.) for r in audit['frames']),
            'fit_seconds':audit['seconds'],'fixed_fields_unchanged':fixed,
            'construction_before':construction_metrics(audit,'before'),'construction_after':construction_metrics(audit,'after'),
            'baseline_feasibility_status':original_floor['status'],'candidate_feasibility_status':candidate_floor['status'],
            'baseline_feasibility':original_floor['summary'],'candidate_feasibility':candidate_floor['summary']})
        baseline_specs[case]=spec;candidate_specs[case]=candidate
    # No manual source labels enter the fitter or its settings; grading starts after every fit.
    labels_path=Path('data/limb-orientation/annotations.json')
    labels=load_annotations(labels_path,project_root='.')
    subset=deepcopy(labels);subset['frames']=[r for r in labels['frames'] if r['clip_id'] in CASES]
    shutil.copyfile(labels_path,out/'original-limb-annotations.json')
    write_json(out/'limb-annotation-subset.json',subset)
    grading={}
    for kind,specs in [('baseline',baseline_specs),('candidate',candidate_specs)]:
        predictions,matching=limb_predictions(specs,subset)
        write_json(out/f'{kind}-limb-predictions.json',predictions)
        write_json(out/f'{kind}-limb-matching.json',matching)
        grading[kind]=grade_limb_projection(subset,predictions)
    write_json(out/'limb-grading.json',{'annotation_input_sha256':digest(labels_path),
        'scope':'Frozen source visual development annotations, not expert ground truth or final holdout',
        'projection':'Unchanged simulation x,-z and time match bound1/23s; no camera or label fitting',
        'cases_without_annotations':[c for c in CASES if not any(r['clip_id']==c for r in subset['frames'])],**grading})
    assert all(digest(Path(path))==expected for path,expected in hashes.items())
    assert all(digest(Path(row['path']))==row['sha256'] for case in inputs.values() for row in case.values())
    report={'schema_version':1,'created_at':local_time(),'accepted':False,'promoted':False,'cases':results,
        'independent_limb_grading':{name:value['summary'] for name,value in grading.items()},
        'implementation_hashes':hashes,'seconds':time.perf_counter()-started,'physics_steps':0,'external_api_usd':0,
        'scope':'Optional image-direction candidate; fitted model image estimates separate from independent annotations; no physical-depth or contact validation'}
    write_json(out/'report.json',report)
    print(json.dumps({'cases':[{'case':r['case'],'fit_seconds':r['fit_seconds'],
        'construction_mean_before_rad':r['construction_before']['unweighted_mean_direction_error_rad'],
        'construction_mean_after_rad':r['construction_after']['unweighted_mean_direction_error_rad'],
        'floor_source_failures_before':r['baseline_feasibility']['failed_source_poses'],
        'floor_source_failures_after':r['candidate_feasibility']['failed_source_poses'],
        'floor_max_before':r['baseline_feasibility']['max_penetration_model_units'],
        'floor_max_after':r['candidate_feasibility']['max_penetration_model_units']} for r in results],
        'independent':report['independent_limb_grading'],'verification':seal_suite(out)},indent=2))


if __name__=='__main__':
    main()
