"""Saved fitted-controller custody only; no controller import or native replay."""

from hashlib import sha256
import json
from pathlib import PurePosixPath
import re

from weft.camera_shared_receipt import (
    IDENTITIES, PROFILE as CAMERA_PROFILE, RUNTIME_REQUIRED, saved_camera_bindings,
)


PROFILE = 'camera-shared-reference-recording-v1'
PROTOCOL = 'camera-shared-reference-full-state-replay-v1'
IMPLEMENTATIONS = frozenset({'camera_reference_receipt.py', 'camera_reference_recording.py',
    'camera_reference_controller.py', 'camera_reference_lab.py', 'shared_exercise_reference_recording.py',
    'shared_exercise_reference_controller.py'})
REQUIRED_RUNTIME = RUNTIME_REQUIRED | IMPLEMENTATIONS | frozenset({
    'camera_reference.py', 'shared_world_packaging.py', 'shared_exercise_lab.py', 'lab.py', 'lab_console.py',
    'cli.py', 'capabilities.py', 'camera_shared_capabilities.py', 'recording_receipt.py'})
ORIGINAL_RUNTIME = RUNTIME_REQUIRED | frozenset({'camera_reference_controller.py', 'camera_reference_recording.py',
    'shared_exercise_reference_controller.py', 'shared_exercise_reference_recording.py',
    'camera_reference.py', 'shared_world_packaging.py'})


def reference_runtime_hashes(value):
    from weft.recording_receipt import _hash, _require

    _require(type(value) is dict and 1 <= len(value) <= 128
             and all(type(name) is str and re.fullmatch(r'[A-Za-z0-9_]+\.py', name) for name in value),
             'Invalid bounded fitted runtime identity map')
    return {name: _hash(pin) for name, pin in value.items()}


def saved_reference_bindings(scene, task, envelope, contract, native, members, *, recipe, recipe_file_sha256,
                             custody, trial_manifest, trial_report):
    from weft.recording_receipt import _canonical, _hash, _require

    _require(type(envelope) is dict and set(envelope) == {
        'schema_version', 'profile', 'engine_scene', 'exercise_task', 'dt', 'objects', 'reference_controller'}
        and envelope.get('profile') == PROFILE, 'Fitted receipt requires its distinct recording envelope')
    # Reuse only common camera metadata joins after checking the distinct wrapper; no state is reprofiled.
    base = {name: value for name, value in envelope.items() if name != 'reference_controller'}
    base['profile'] = CAMERA_PROFILE
    identities = saved_camera_bindings(scene, task, base, contract, native, members)
    _require(REQUIRED_RUNTIME <= native.keys(), 'Fitted lab runtime closure is incomplete')
    reference = envelope['reference_controller']
    _require(type(reference) is dict and set(reference) == {'document_utf8', 'sha256', 'implementation_sha256'}
             and type(reference['document_utf8']) is str and len(reference['document_utf8'].encode()) <= 1 << 20,
             'Fitted recipe envelope is malformed or oversized')
    pin = _hash(reference['sha256'])
    _require(sha256(reference['document_utf8'].encode()).hexdigest() == pin == recipe_file_sha256
             == members.get('reference-controller.json') == contract.get('reference_controller_sha256'),
             'Fitted recipe bytes differ from the recording/lab identity')
    multiview = recipe.get('profile') == 'camera-multiview-source-fit-controller-recipe-v1'
    _require(json.loads(reference['document_utf8']) == recipe
             and recipe.get('profile') in {'camera-source-fit-controller-recipe-v1',
                                          'camera-multiview-source-fit-controller-recipe-v1'}
             and recipe.get('interpretation') == 'archived_execution_only_not_source_fit_admission'
             and recipe.get('scene_sha256') == _canonical(scene) and recipe.get('task_sha256') == _canonical(task)
             and recipe.get('gates_sha256') == task.get('gates_sha256')
             and all(recipe.get(key) == task[key] for key in IDENTITIES),
             'Fitted recipe changes its original camera task or authority')
    original_runtime = reference_runtime_hashes(recipe.get('runtime_sha256'))
    _require(ORIGINAL_RUNTIME <= original_runtime.keys() and reference['implementation_sha256'] == original_runtime
             and all(native.get(name) == digest for name, digest in original_runtime.items()),
             'Original recipe runtime must remain exact within the new lab')
    _require(custody.get('profile') == 'camera-reference-lab-attachment-v1'
             and custody.get('scene_sha256') == _canonical(scene) and custody.get('task_sha256') == _canonical(task)
             and custody.get('summary') == contract.get('exercise_reference'), 'Fitted attachment summary differs')
    for key in ('reference_controller_sha256', 'reference_trial_manifest_sha256', 'reference_proposal_manifest_sha256'):
        _require(_hash(custody.get(key)) == contract.get(key), 'Fitted attachment identity differs')
    _require(members.get('reference-trial/manifest.json') == contract['reference_trial_manifest_sha256']
             and members.get('reference-trial/inputs/proposal-manifest.json') == contract['reference_proposal_manifest_sha256']
             and trial_manifest.get('profile') == 'shared-exercise-reference-trial-v1'
             and trial_manifest.get('status') == trial_report.get('status') == 'completed',
             'Original completed fitted trial is not retained')
    selected = custody.get('selected_trial_members')
    _require(type(selected) is dict and 1 <= len(selected) <= 64
             and all(type(name) is str and 0 < len(name) <= 512 and not PurePosixPath(name).is_absolute()
                     and all(part not in ('', '.', '..') for part in name.split('/'))
                     and '\\' not in name and ':' not in name and _hash(pin) for name, pin in selected.items()),
             'Invalid selected fitted proof inventory')
    aliases = trial_report.get('proposal_input_aliases')
    _require(type(aliases) is dict and 1 <= len(aliases) <= 32 and all(type(name) is str for name in aliases.values()),
             'Original fitted proposal aliases are unavailable')
    required_selected = {'plan.json', 'report.json', 'inputs/proposal-manifest.json', 'inputs/scene-manifest.json',
        'inputs/exercise-task.json', 'run/exercise-task.json', 'inputs/scene.json', 'run/scene.json', 'run/reference-controller.json',
        'proofs/authored-action-replay.json', 'proofs/reference_proposal-action-replay.json',
        'recordings/authored/manifest.json', 'recordings/reference_proposal/manifest.json', *aliases.values()}
    _require(set(selected) == required_selected, 'Selected fitted proof inventory is incomplete')
    _require(all(members.get('reference-trial/' + name) == digest == trial_manifest.get('files', {}).get(name)
                 for name, digest in selected.items()), 'Selected fitted proof inventory differs from its original seal')
    source = custody.get('source_identity', {})
    _require(recipe.get('input_human_sha256') == source.get('original_human_sha256') == members.get('source-run/human.json')
             and recipe.get('input_evidence_sha256') == source.get('input_evidence_sha256') == members.get('source-run/evidence.json')
             and recipe.get('angle_proposal_sha256') == source.get('angle_proposal_sha256')
             and recipe.get('proposal_report_sha256') == source.get('proposal_report_sha256')
             and trial_report.get('source_identities') == source, 'Fitted recipe and retained source identities differ')
    if multiview:
        overlay = recipe.get('angle_proposal')
        _require(trial_report.get('controller_family') == 'camera_multiview_angle_overlay'
                 and type(overlay) is dict and overlay.get('profile') == 'camera-body-multiview-angle-proposal-v1'
                 and type(overlay.get('additional_view')) is dict and type(recipe.get('multiview_images')) is dict,
                 'Multiview recipe requires its distinct overlay and qualification')
        extension = overlay['additional_view']
        binding = extension.get('binding')
        _require(type(binding) is dict and binding.get('profile') == 'camera-multiview-evidence-extension-v1'
                 and extension.get('binding_canonical_sha256') == _canonical(binding)
                 and recipe.get('additional_view_sha256') == source.get('additional_view_sha256') == _canonical(extension)
                 and recipe.get('multiview_qualification_sha256') == source.get('multiview_qualification_sha256')
                     == _canonical(recipe['multiview_images'])
                 and recipe['multiview_images'].get('binding_canonical_sha256') == extension['binding_canonical_sha256'],
                 'Multiview recipe qualification differs from retained source identity')
        secondary = binding.get('secondary_input_bindings')
        views = binding.get('views')
        _require(type(secondary) is dict and type(views) is list and len(views) == 2
                 and type(views[1]) is dict and views[1].get('source_sha256') == source.get('secondary_source_sha256'),
                 'Multiview secondary source identity is unavailable')
        for name, key, original in (
                ('inputs/secondary-camera.json', 'camera', 'camera.json'),
                ('inputs/secondary-evidence.json', 'evidence', 'evidence.json')):
            expected = _hash(source.get('secondary_' + key + '_sha256'))
            _require(secondary.get(key + '_raw_sha256') == expected
                     == members.get('source-run/inputs/two-view-root/' + original)
                     == members.get('reference-trial/' + aliases.get(name, '')),
                     'Multiview secondary raw input differs from original source or trial')
        expected = _hash(source.get('secondary_view_config_sha256'))
        _require(members.get('reference-trial/' + aliases.get('inputs/camera-secondary-config.json', '')) == expected,
                 'Multiview config is absent from retained trial')
        expected = _hash(source.get('secondary_world_alignment_sha256'))
        _require(secondary.get('alignment_raw_sha256') == expected
                 == members.get('reference-trial/' + aliases.get('inputs/camera-secondary-alignment.json', '')),
                 'Multiview explicit world/clock alignment is absent from retained trial')
        observation_pins = source.get('secondary_observation_sha256s')
        _require(type(observation_pins) is list and 1 <= len(observation_pins) <= 8
                 and len(set(observation_pins)) == len(observation_pins)
                 and all(_hash(value) for value in observation_pins)
                 and secondary.get('observation_document_sha256s') == sorted(observation_pins)
                 and all(members.get('reference-trial/' + aliases.get(f'inputs/camera-secondary-observation-{index:02d}.json', '')) == pin
                         for index, pin in enumerate(observation_pins)),
                 'Multiview observation inventory differs from retained trial')
    else:
        _require(not {'additional_view_sha256', 'multiview_qualification_sha256', 'multiview_images'} & recipe.keys(),
                 'Single-view recipe cannot carry unversioned multiview qualification')
    outcomes = {row['mode']: {'steps': row['steps'], 'success': row['metrics']['success'],
                            'mean_reward': row['metrics'].get('mean_reward')} for row in trial_report.get('trials', [])}
    _require(set(outcomes) == {'authored', 'reference_proposal'}
             and custody['summary'].get('saved_trial_outcomes') == outcomes
             and custody['summary'].get('learned') is False, 'Saved fitted outcomes differ or claim learning')
    return {**identities, 'reference_controller_sha256': pin}
