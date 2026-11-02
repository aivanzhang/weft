"""Camera labs expose saved evidence without enabling controllers or loading physics."""
from hashlib import sha256
import json

import pytest

from test_camera_shared_policy_receipt import package, raw, saved_only
from test_lab_evidence import card_rows, digest, load, metrics, put, seal
from weft.io import canonical_hash
from weft.lab_inspection import _timeline


def camera_run(root, *, roots=True, learned=True, evaluation=True):
    data = package(roots=roots, evaluation=evaluation)
    root.mkdir()
    scene, task, contract, admission = (data[name] for name in ('scene', 'task', 'contract', 'admission'))
    put(root, 'source.json', {'sha256': scene['human_spec']['source_sha256']})
    put(root, 'scene.json', scene); put(root, 'exercise-task.json', task)
    contract.update(schema_version=1, profile='shared-world', control_profile='shared_exercise',
                    scene_sha256=canonical_hash(scene), task_sha256=canonical_hash(task))
    if learned:
        policy_raw = data['envelope']['learned_policy']['document_utf8'].encode()
        put(root, 'policy.json', policy_raw)
        put(root, 'policy-evidence-run/policy.json', policy_raw)
        put(root, 'policy-evidence-run/manifest.json', b'opaque archived metadata; not parsed by inspection')
        evidence_pin = digest(root/'policy-evidence-run/manifest.json')
        kind = 'evaluation' if evaluation else 'training'
        admission[kind+'_manifest_sha256'] = evidence_pin
        contract['learning_manifest_sha256'] = admission['training_manifest_sha256']
        contract['camera_policy_evidence_manifest_sha256'] = evidence_pin
        if evaluation:
            put(root, 'policy-evaluation.json', b'opaque trial arrays; not parsed by inspection')
            contract.update(policy_evaluation_manifest_sha256=evidence_pin,
                            policy_evaluation_sha256=digest(root/'policy-evaluation.json'))
        put(root, 'camera-policy-admission.json', admission)
        contract['camera_policy_admission_sha256'] = canonical_hash(admission)
        contract['exercise_policy'] = {'profile': data['policy']['profile'], 'policy_sha256': sha256(policy_raw).hexdigest(),
            'admission_profile': admission['profile'], 'admission_sha256': canonical_hash(admission),
            'privileged_simulator_features': True, **admission['summary'],
            'evaluation_status': 'evaluated' if evaluation else 'not_attached',
            'evaluation_successes': {name: {'successes': n, 'episodes': 3} for name,n in
                [('learned',3),('authored',3),('coast',0),('random',0)]} if evaluation else None}
    else:
        contract = {k: contract[k] for k in ('schema_version','profile','control_profile','scene_sha256','task_sha256')}
        contract['learned_policy_available'] = False
    put(root, 'exercise-lab.json', contract)
    put(root, 'reference-feasibility.json', b'geometry only; must not invent a force check')
    put(root, 'two-view-body.json', b'large source fit; linked, not parsed here')
    seal(root)
    return root


@pytest.mark.parametrize('roots', [False, True])
@pytest.mark.parametrize('evaluation', [False, True])
def test_camera_learning_keeps_saved_counts_admission_and_unknowns(tmp_path, roots, evaluation):
    root = camera_run(tmp_path/'run', roots=roots, evaluation=evaluation)
    before = {p.relative_to(root): digest(p) for p in root.rglob('*') if p.is_file()}
    result = load(root)
    assert result['status'] == 'available', result
    rows = card_rows(result); assert len(rows) == 14
    assert rows['physical_relationships']['status'] == 'unavailable'
    assert rows['parameter_validity']['status'] == 'unavailable'
    learning = metrics(rows['task_learning_utility'])
    assert learning['physical_transfer_trials']['value'] is None
    if evaluation:
        for name,n in [('learned',3),('authored',3),('coast',0),('random',0)]:
            assert learning[name+'_successes']['numerator'] == n
            assert learning[name+'_successes']['denominator'] == 3
            assert learning[name+'_successes']['partition'] == 'exposed_development'
        assert 'camera-policy-admission.json' in {p['source_name'] for p in learning['learned_successes']['evidence']}
    else:
        assert rows['task_learning_utility']['status'] == 'unavailable'
    verified = {r['source_name'] for r in result['verified_metadata']}
    assert 'camera-policy-admission.json' in verified
    assert not verified & {'reference-feasibility.json','policy-evaluation.json','two-view-body.json','policy-evidence-run/manifest.json'}
    assert 'two-view-body.json' in {r['name'] for r in result['artifact_links']}
    assert before == {p.relative_to(root): digest(p) for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('roots', [False, True])
def test_untrained_camera_lab_retains_all_categories_without_a_policy(tmp_path, roots):
    result = load(camera_run(tmp_path/'run', roots=roots, learned=False))
    assert result['status'] == 'available', result
    assert len(card_rows(result)) == 14
    assert card_rows(result)['task_learning_utility']['status'] == 'unavailable'
    assert 'policy.json' not in {r['source_name'] for r in result['verified_metadata']}


@pytest.mark.parametrize('fault', ['missing_admission','oversized_admission','admission_identity','policy_bytes',
    'policy_scope','policy_runtime','execution_runtime','root_label','research_label','selection_label',
    'evidence_manifest','evaluation_manifest','evaluation_alias','inventory','counts','scene_profile'])
def test_camera_summary_rejects_resealed_conflicts(tmp_path, fault):
    root = camera_run(tmp_path/'run'); path = root/'exercise-lab.json'
    contract = json.loads(path.read_text()); compact = contract['exercise_policy']
    if fault == 'missing_admission':
        (root/'camera-policy-admission.json').unlink()
    elif fault == 'oversized_admission': put(root,'camera-policy-admission.json',{'extra':'x'*(128*1024)})
    elif fault == 'admission_identity': contract['camera_policy_admission_sha256'] = '0'*64
    elif fault == 'policy_bytes':
        policy=json.loads((root/'policy.json').read_text());policy['root_assistance']='disabled';put(root,'policy.json',policy)
    elif fault == 'policy_scope': compact['privileged_simulator_features'] = False
    elif fault == 'policy_runtime': contract['policy_runtime_sha256']['human.py'] = '0'*64
    elif fault == 'execution_runtime': contract['runtime_sha256']['human.py'] = '0'*64
    elif fault == 'root_label': compact['root_assistance'] = 'disabled'
    elif fault == 'research_label': compact['learning_qualification'] = 'physical_validation'
    elif fault == 'selection_label': compact['selected_feedback_kind'] = 'authored_unchanged'
    elif fault == 'evidence_manifest': contract['camera_policy_evidence_manifest_sha256'] = '0'*64
    elif fault == 'evaluation_manifest': contract['policy_evaluation_manifest_sha256'] = '0'*64
    elif fault == 'evaluation_alias': contract['camera_policy_evidence_kind'] = 'training'
    elif fault == 'inventory': contract['camera_policy_evidence_files_verified'] = 2
    elif fault == 'counts': compact['evaluation_successes']['learned']['successes'] = 4
    else:
        scene=json.loads((root/'scene.json').read_text());scene['profile']='shared-world-v7';put(root,'scene.json',scene)
    put(root,'exercise-lab.json',contract);seal(root)
    result=load(root)
    assert result['status'] == 'unavailable', result
    assert 'evidence_scorecard' not in result


def test_camera_task_timeline_does_not_turn_unsampled_frames_into_measurements():
    source='a'*64;frames=list(range(0,185,4))
    human={'source_sha256':source,'reference':[{'time_s':f/50} for f in frames],
        'reference_evidence':[{'time_s':f/50,'evidence_id':source+f':frame:{f}'} for f in frames]}
    result=_timeline({'profile':'shared-world-v6','dt':.02},human,185,
                     {'interval':{'start_offset_s':0.,'horizon_steps':184}})
    assert result['task_steps']==184 and result['task_window_s']==[0.,3.68]
    assert result['source_slots']['with_reference']==47 and result['source_slots']['without_reference']==138
    assert result['native_source_clock']=='not_read_here' and result['media_sync_verified'] is False
