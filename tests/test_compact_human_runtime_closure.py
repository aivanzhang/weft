"""Projection validation is pinned by every supported actor without changing legacy closures."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_shared_human_projection import full_human, raw_human
from weft import shared_exercise_lab as lab, shared_exercise_learning as gains
from weft import shared_exercise_residual_learning as residual
from weft import shared_exercise_feasibility_learning as feasibility
from weft import shared_exercise_feedback_residual as feedback
from weft import shared_exercise_equipment_feedback_policy as equipment
from weft.io import canonical_hash, digest
from weft.shared_human_projection import create_projection
from weft.shared_world import validate_shared_world
from weft.shared_world_lab import authored_shared_world_scene
from weft.shared_exercise_task import build_shared_exercise_task


HELPER = 'shared_human_projection.py'
RUNTIMES = [gains._runtime_hashes, residual._scene_runtime_hashes,
            feasibility._scene_runtime_hashes, feedback.runtime_hashes, equipment.runtime_hashes]


@pytest.mark.parametrize('runtime', RUNTIMES)
@pytest.mark.parametrize('profile', ['shared-world-v1', 'shared-world-v2', 'shared-world-v3'])
def test_actor_projection_adds_only_helper_to_existing_native_closure(runtime, profile):
    scene = {'profile': profile, 'human_spec': {'native_clock': {}}}
    before = deepcopy(scene); legacy = runtime(scene)
    projected = {**scene, 'human_provenance': {}}
    assert HELPER not in legacy
    assert runtime(projected) == {**legacy, HELPER: digest(Path(gains.__file__).with_name(HELPER))}
    assert scene == before and runtime(scene) == legacy


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('profile', ['shared-world-v1', 'shared-world-v2', 'shared-world-v3'])
def test_lab_projection_closure_preserves_existing_order_and_default(native, profile):
    scene = {'profile': profile, 'human_spec': {'native_clock': {}} if native else {}}
    legacy = lab.runtime_files(scene)
    projected = lab.runtime_files({**scene, 'human_provenance': {}})
    assert HELPER not in legacy and projected.count(HELPER) == 1
    assert tuple(name for name in projected if name != HELPER) == legacy


def projected_scene():
    scene = authored_shared_world_scene(); human = full_human()
    scene.pop('action_contract')
    for owner,key in ((human,'reference'), (human,'root_reference'), (human,'reference_evidence'), (human['body_frame'],'reference')):
        rows = [deepcopy(owner[key][0]) for _ in range(21)]
        owner[key] = rows
        for index,row in enumerate(rows):
            row['time_s'] = index / 10
            if 'evidence_id' in row: row['evidence_id'] = f"{human['source_sha256']}:frame:{index*3}"
            if 'id' in row: row['id'] = f"{human['source_sha256']}:frame:{index*3}"
    human['body_frame']['evidence_ids'] = [r['evidence_id'] for r in human['reference_evidence']]
    human['native_clock'].update(metadata_frames=90, decoded_frames=90)
    scene['human_spec'],scene['human_provenance'] = create_projection(raw_human(human))
    scene['coordinate_frame']['clock_binding'].update(native_clock=human['native_clock'],
        alignment_method='native_pts_elapsed_time_v1', source_fps=30., source_time_offset_s=0.)
    ids = ['human', *(obj['id'] for obj in scene['objects'])]
    scene['source_binding'] = {'mode':'source_qualified', 'source_sha256':human['source_sha256'],
        'total_source_frames':90, 'evidence_documents':{'e'*64:'analytical.json'},
        'entity_bindings':[{'world_entity_id':name, 'entity_kind':'articulated_human' if name=='human' else 'rigid_object',
            'observation_entity_id':name, 'observation_document_sha256':'e'*64, 'provenance_mode':'observed_track',
            'observed_slots':90, 'missing_slots':0, 'pre_init_slots':0} for name in ids]}
    # Metadata-only analytic fixture: no source package or physical qualification is claimed.
    return validate_shared_world(scene)


@pytest.mark.parametrize('kind', ['gains', 'residual', 'feasibility', 'feedback'])
@pytest.mark.parametrize('damage', ['omit', 'replace'])
def test_policy_loaders_reject_missing_or_changed_projection_runtime(kind, damage):
    scene = projected_scene(); task = build_shared_exercise_task(scene)
    runtime = {'gains':gains._runtime_hashes, 'residual':residual._scene_runtime_hashes,
               'feasibility':feasibility._scene_runtime_hashes, 'feedback':feedback.runtime_hashes}[kind]
    hashes = runtime(scene)
    env = SimpleNamespace(scene=scene, task=task, scene_sha256=canonical_hash(scene), task_sha256=canonical_hash(task),
        evaluator_sha256=hashes['shared_exercise_task.py'], action_names=scene['action_contract']['names'][:19], action_count=19)
    if kind == 'gains':
        policy = gains._policy(scene, task, env, np.ones((3,3)), 1729, .5, hashes)
        validate = gains.validate_shared_exercise_policy
    elif kind == 'residual':
        policy = residual._policy(env, np.zeros((2,19)), 1729, .5, hashes)
        validate = residual.validate_shared_exercise_residual_policy
    else:
        rank = {'profile':feasibility.RANK_PROFILE, 'complete':True, 'feasible':True,
            'max_normalized_excess':0., 'mean_normalized_excess':0., 'violated_terms':0,
            'gate_terms':2*(len(feasibility.BODY_GATES)+len(feasibility.EQUIPMENT_GATES)*len(task['equipment_ids'])
                            +3*len(task['contact_requirements'])), 'reward_score':.5}
        if kind == 'feedback':
            policy = feedback.make_policy(env, np.zeros((3,2)), 1729, .5, rank, hashes)
            validate = feedback.validate_shared_exercise_feedback_policy
        else:
            selected = {'residual_coefficients':np.zeros((2,19)).tolist(), 'score':.5, 'optimization_rank':rank}
            policy = feasibility._policy(env, selected, 1729, hashes, {'original_training_manifest_sha256':'a'*64})
            validate = feasibility.validate_shared_exercise_feasibility_policy
    assert validate(policy,scene,task) == policy
    if damage == 'omit': del policy['runtime_sha256'][HELPER]
    else: policy['runtime_sha256'][HELPER] = '0'*64
    with pytest.raises(ValueError, match='runtime'): validate(policy,scene,task)
