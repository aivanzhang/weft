"""Evaluate source-based crop selection using frozen proposals without new inference."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json
from weft.limb_fidelity import grade_limb_projection, load_annotations
from weft.pipeline import _implementation_files, verify_artifacts
from weft.pose_crop_selection import CONTRACT, assess_selective_crop, plan_selective_crop
from check_crop_diversity import CLIPS, LIMB_LABELS, PAIR_LABELS, deltas, limb_predictions
from grade_pose_identity import grade as grade_identity


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'artifacts/crop-diversity-screen-01'


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    if out.is_relative_to(SOURCE): raise ValueError('Use a new output outside the frozen proposals')
    verified = verify_artifacts(SOURCE)
    implementation = {name: digest(path) for name, path in _implementation_files().items()}
    out.mkdir(parents=True, exist_ok=False)
    for path in (Path(__file__), Path(__file__).with_name('check_crop_diversity.py'), Path(__file__).with_name('grade_pose_identity.py')):
        shutil.copyfile(path, out/path.name)
    for name, path in _implementation_files().items():
        target = out/'implementation-source'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'source_verification': verified,
        'contract': dict(CONTRACT), 'implementation': implementation, 'new_inference_calls': 0, 'paid_requests': 0,
        'scope': 'Development selector designed after the mixed crop screen; not independent validation',
        'construction_context': 'Original historical raw pose neighbors; same fixed crops as the frozen screen',
        'excluded_from_selection': ['comparison labels', 'body/floor scores', 'best-score frame selection']})
    original, selected, decisions = {}, {}, {}
    for clip in CLIPS:
        context = read(SOURCE/'inputs'/(clip+'.json'))
        whole, crop = [read(SOURCE/clip/(mode+'.json')) for mode in ('whole_image', 'crop_image')]
        crops = {row['frame']: row for row in crop['frames']}
        original[clip], selected[clip], decisions[clip] = whole, deepcopy(whole), []
        for index, row in enumerate(whole['frames']):
            plan = plan_selective_crop(context, row)
            decision = {'frame': row['frame'], 'accepted': False, 'reasons': [plan['reason']]}
            if plan['eligible']:
                proposal = crops[row['frame']]
                if proposal.get('crop_proposal') != plan['crop']:
                    raise ValueError('The selector needs a different crop; do not reuse unmatched inference')
                decision = assess_selective_crop(context, row, proposal, plan)
                if decision['accepted']: selected[clip]['frames'][index] = deepcopy(proposal)
            decisions[clip].append({'plan': plan, 'decision': decision})
        write_json(out/(clip+'-selected.json'), selected[clip])
    write_json(out/'decisions.json', decisions)
    write_json(out/'construction.json', {'created_at': local_time(), 'status': 'selection_frozen_before_scoring',
        'selected_frames': {clip: [row['decision']['frame'] for row in rows if row['decision']['accepted']] for clip, rows in decisions.items()},
        'no_labels_used': True})
    labels = {'identity': read(PAIR_LABELS), 'limbs': load_annotations(LIMB_LABELS, project_root=ROOT)}
    if any(digest(Path(path)) != value for path, value in read(SOURCE/'pre-run.json')['labels'].items()):
        raise ValueError('The frozen comparison labels changed')
    scores = {}
    for mode, observations in (('whole_image', original), ('selected', selected)):
        predicted, matching = limb_predictions(labels['limbs'], observations)
        scores[mode] = {'identity': grade_identity(labels['identity'], observations),
                        'limbs': grade_limb_projection(labels['limbs'], predicted)}
        write_json(out/(mode+'-limb-predictions.json'), {'predictions': predicted, 'time_matching': matching})
    changes = {kind: deltas(scores['whole_image'][kind]['observations'], scores['selected'][kind]['observations'],
                           'passes' if kind == 'identity' else 'within_tolerance') for kind in ('identity', 'limbs')}
    write_json(out/'scores.json', {'scores': scores, 'changes': changes})
    if verify_artifacts(SOURCE) != verified or implementation != {name: digest(path) for name, path in _implementation_files().items()}:
        raise ValueError('A frozen input or implementation changed')
    report = {'status': 'cached_selective_crop_screen_complete', 'created_at': local_time(),
        'selected_frames': read(out/'construction.json')['selected_frames'],
        'summaries': {mode: {name: group['summary'] for name, group in kinds.items()} for mode, kinds in scores.items()},
        'changes': {name: {key: value for key, value in rows.items() if key != 'paired_rows'} for name, rows in changes.items()},
        'new_inference_calls': 0, 'paid_requests': 0, 'default_changed': False,
        'scope': 'Source-only selector on exposed development proposals; complete-video/new-context qualification remains required'}
    write_json(out/'report.json', report)
    print(json.dumps({'selected_frames': report['selected_frames'], 'summaries': report['summaries'],
        'regressions': {name: value['regressions'] for name, value in report['changes'].items()},
        'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__': main()
