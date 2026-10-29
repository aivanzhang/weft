"""Frozen three-case Cartesian candidate study; never promotes or advances dynamics."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np

from weft.benchmark import seal_suite
from weft.body_refinement import JOINT_LANDMARKS, REFINEMENT_CONTRACT, refine_reference
from weft.human import HumanEnv
from weft.io import canonical_hash, digest, local_time, write_json
from weft.limb_fidelity import grade_limb_projection, load_annotations
from weft.whole_body_feasibility import audit_reference_feasibility


CASES = ('single-leg-squat', 'squat', 'kettlebell-pushup')


def geometry_metrics(spec, frames):
    env = HumanEnv(spec)
    squared, weighted, weight_sum = [], [], []
    per_joint = {name: [] for name in JOINT_LANDMARKS}
    try:
        data = mujoco.MjData(env.model)
        for row in frames:
            if 'target_joints_hip_centered' not in row:
                continue
            index = row['reference_index']
            data.qpos[env._q_array] = [spec['reference'][index]['joints'][name] for name in env.joint_names]
            data.qpos[env.mobility.q] = spec['root_reference'][index]['translation']
            mujoco.mj_forward(env.model, data)
            center = data.site_xpos[env._site_ids['pelvis']]
            for name in JOINT_LANDMARKS:
                residual = data.site_xpos[env._site_ids[name]] - center - row['target_joints_hip_centered'][name]
                error = float(residual @ residual)
                weight = row['landmark_weights'][name]
                squared.append(error); weighted.append(weight * error); weight_sum.append(weight)
                per_joint[name].append(error)
    finally:
        env.close()
    return {'endpoint_count': len(squared), 'confidence_weight_sum': sum(weight_sum),
            'unweighted_endpoint_rmse': float(np.sqrt(np.mean(squared))) if squared else None,
            'confidence_weighted_endpoint_rmse': float(np.sqrt(sum(weighted) / sum(weight_sum))) if sum(weight_sum) else None,
            'per_joint_unweighted_rmse': {name:float(np.sqrt(np.mean(values))) if values else None for name,values in per_joint.items()}}


def limb_predictions(specs, annotations):
    predictions, matching = [], []
    for case, spec in specs.items():
        env = HumanEnv(spec)
        try:
            times = np.array([row['time_s'] for row in spec['reference']])
            for label in (row for row in annotations['frames'] if row['clip_id'] == case):
                assert label['source_sha256'] == spec['source_sha256']
                nearest = int(np.argmin(abs(times - label['timestamp_seconds'])))
                offset = float(times[nearest] - label['timestamp_seconds'])
                matched = abs(offset) <= 1 / 23 + 1e-8
                matching.append({'id': label['id'], 'reference_index': nearest,
                                 'offset_seconds': offset, 'matched': matched})
                segments = {}
                if matched:
                    state = env.reference_snapshot(nearest)
                    for segment in state['human']['segments']:
                        if segment['id'] in label['segments']:
                            axis = np.subtract(segment['b'], segment['a'])
                            segments[segment['id']] = {'axis_xy': [float(axis[0]), float(-axis[2])]}
                predictions.append({'id':label['id'],'source_sha256':spec['source_sha256'],'segments':segments})
        finally:
            env.close()
    return predictions, matching


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    out=args.out
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    def forbidden(*args,**kwargs):
        raise AssertionError('Candidate construction study cannot advance dynamics')
    mujoco.mj_step=forbidden
    paths=[Path('src/weft')/name for name in ['body_refinement.py','whole_body_feasibility.py','whole_body.py',
        'human.py','spatial_pose.py','body_frame.py','mobile_human.py','root_motion.py','io.py','limb_fidelity.py','body_fidelity.py']]
    paths += [Path('tests/test_body_refinement.py'),Path('tests/test_whole_body_feasibility.py'),Path(__file__)]
    hashes={str(path):digest(path) for path in paths}
    implementation=out/'implementation';implementation.mkdir()
    for path in paths:
        shutil.copyfile(path,implementation/path.name)
    inputs={}
    for case in CASES:
        source=Path('artifacts/mobile-exercise-e2e-10')/case/'run'
        inputs[case]={name:{'path':str(source/name),'sha256':digest(source/name)}
                      for name in ['human.json','pose-landmarks.json','source.webm']}
    write_json(out/'pre-fit-freeze.json',{'created_at':local_time(),'contract':REFINEMENT_CONTRACT,
        'implementation_hashes':hashes,'inputs':inputs,'analytic_tests_passed':16,
        'maximum_fits':3,'maximum_total_fit_seconds':180,'external_api_usd':0,
        'scope':'Constants fixed before these candidate fits; no tuning to independent labels or floor outcomes'})
    results=[]; baseline_specs={}; candidate_specs={}
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
        evidence={'source_sha256':spec['source_sha256'],**raw}
        write_json(directory/'evidence-adapter.json',{'method':'attach_verified_source_digest_only',
            'source_sha256':spec['source_sha256'],'raw_file_sha256':inputs[case]['pose-landmarks.json']['sha256'],
            'adapted_evidence_canonical_sha256':canonical_hash(evidence),'all_source_ids_verified':True,
            'new_pose_inference':False})
        candidate,audit=refine_reference(spec,evidence)
        write_json(directory/'refinement.json',audit)
        write_json(directory/'candidate-human.json',candidate)
        fixed={key: candidate.get(key)==spec.get(key) for key in
               ['root_reference','mobility','lengths','body_frame','motor_limits','dt']}
        assert all(fixed.values())
        assert [row['time_s'] for row in candidate['reference']]==[row['time_s'] for row in spec['reference']]
        baseline_floor=audit_reference_feasibility(spec)
        candidate_floor=audit_reference_feasibility(candidate)
        write_json(directory/'baseline-feasibility.json',baseline_floor)
        write_json(directory/'candidate-feasibility.json',candidate_floor)
        baseline_geometry=geometry_metrics(spec,audit['frames'])
        candidate_geometry=geometry_metrics(candidate,audit['frames'])
        directions={}
        for row in audit['frames']:
            if row['direction_errors_before_rad'] is None:
                continue
            for name,before in row['direction_errors_before_rad'].items():
                after=row['direction_errors_after_rad'][name]
                if before is not None and after is not None:
                    directions.setdefault(name,[]).append((before,after))
        direction_summary={name:{'count':len(values),
            'mean_before_rad':float(np.mean([a for a,b in values])),
            'mean_after_rad':float(np.mean([b for a,b in values])),
            'maximum_after_rad':max(b for a,b in values),
            'worsened_samples':sum(b>a+1e-8 for a,b in values),
            'improved_samples':sum(b<a-1e-8 for a,b in values)} for name,values in directions.items()}
        results.append({'case':case,'input_hashes':inputs[case],'fit_status':audit['status'],
            'statuses':dict(Counter(row['status'] for row in audit['frames'])),
            'original_evidence_frames':len(raw['frames']),'reference_frames':len(spec['reference']),
            'unused_evidence_ids':audit['unused_evidence_ids'],'changed_frames':audit['changed_frames'],
            'max_angle_change_rad':max(row.get('maximum_angle_change_rad',0.) for row in audit['frames']),
            'fit_seconds':audit['seconds'],'fixed_fields_unchanged':fixed,
            'baseline_geometry':baseline_geometry,'candidate_geometry':candidate_geometry,
            'direction_tradeoffs':direction_summary,
            'baseline_feasibility':baseline_floor['summary'],'candidate_feasibility':candidate_floor['summary'],
            'baseline_feasibility_status':baseline_floor['status'],'candidate_feasibility_status':candidate_floor['status']})
        baseline_specs[case]=spec;candidate_specs[case]=candidate
    # Labels are read only after all candidate fits finish; neither optimizer nor its settings receives them.
    labels_path=Path('data/limb-orientation/annotations.json')
    labels=load_annotations(labels_path,project_root='.')
    subset=deepcopy(labels)
    subset['frames']=[row for row in labels['frames'] if row['clip_id'] in CASES]
    shutil.copyfile(labels_path,out/'original-limb-annotations.json')
    write_json(out/'limb-annotation-subset.json',subset)
    grading={}
    for kind,specs in [('baseline',baseline_specs),('candidate',candidate_specs)]:
        predictions,matching=limb_predictions(specs,subset)
        write_json(out/f'{kind}-limb-predictions.json',predictions)
        write_json(out/f'{kind}-limb-matching.json',matching)
        grading[kind]=grade_limb_projection(subset,predictions)
    write_json(out/'limb-grading.json',{'annotation_input_sha256':digest(labels_path),
        'scope':'Frozen source-only development estimates, not expert labels or a final holdout',
        'projection':'Simulation x and -z; unchanged time-match bound1/23s; no camera or label fit',
        'cases_without_annotations':[case for case in CASES if not any(row['clip_id']==case for row in subset['frames'])],
        **grading})
    assert all(digest(Path(path))==expected for path,expected in hashes.items())
    assert all(digest(Path(value['path']))==value['sha256'] for case in inputs.values() for value in case.values())
    report={'schema_version':1,'created_at':local_time(),'accepted':False,'promoted':False,'cases':results,
        'independent_limb_grading':{key:value['summary'] for key,value in grading.items()},
        'implementation_hashes':hashes,'seconds':time.perf_counter()-started,'physics_steps':0,'external_api_usd':0,
        'scope':'Frozen optional Cartesian angle candidate; construction fit and unchanged independent checks, not physical validation or default promotion'}
    write_json(out/'report.json',report)
    print(json.dumps({'cases':[{'case':row['case'],'seconds':row['fit_seconds'],
        'rmse_before':row['baseline_geometry']['unweighted_endpoint_rmse'],
        'rmse_after':row['candidate_geometry']['unweighted_endpoint_rmse'],
        'floor_before':row['baseline_feasibility']['max_penetration_model_units'],
        'floor_after':row['candidate_feasibility']['max_penetration_model_units']} for row in results],
        'limb_grading':report['independent_limb_grading'],'verification':seal_suite(out)},indent=2))


if __name__=='__main__':
    main()
