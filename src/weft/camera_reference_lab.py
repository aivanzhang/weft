"""Attach exact saved fitted controls without replacing the camera body or task."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
import time

from weft.io import canonical_hash
from weft.shared_world_packaging import read_package_file, validate_package_layout


PROFILE = 'camera-reference-lab-attachment-v1'
TRIAL_PROFILE = 'shared-exercise-reference-trial-v1'
PREFIX = 'reference-trial/'
ATTACHMENT = 'reference-attachment.json'
RECIPE = 'reference-controller.json'
MAX_SELECTED_BYTES = 32 << 20
MAX_SELECTED_FILES = 64
EXECUTION_FILES = frozenset({'camera_reference_lab.py', 'shared_exercise_lab.py', 'lab.py', 'lab_console.py',
    'cli.py', 'capabilities.py', 'camera_shared_capabilities.py', 'recording_receipt.py', 'camera_reference_receipt.py'})
SCOPE = ('Fitted joint targets, not a learned policy or necessarily better control. Original body, camera, '
         'root assistance, floor, task and scoring stay unchanged. Contact and physical accuracy are unverified.')
_SHA = re.compile(r'[0-9a-f]{64}\Z')
_ALIASES = {'angle-proposal.json': 'inputs/angle-proposal.json', 'inputs/human.json': 'inputs/human.json',
            'report.json': 'inputs/proposal-report.json', 'inputs/evidence-derived.json': 'inputs/evidence-derived.json'}


def _require(value, message):
    if not value:
        raise ValueError('Camera fitted lab: ' + message)


def _check(deadline):
    _require(time.monotonic() < deadline, 'preparation deadline exceeded')


def _json(raw):
    from weft.whole_body_reference_report import _json as strict_json
    return strict_json(raw)


def _bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def _seal(raw):
    value = _json(raw)
    hashes = value.get('files')
    _require(type(value.get('schema_version')) is int and value['schema_version'] == 1
             and value.get('profile') == TRIAL_PROFILE and value.get('status') == 'completed'
             and type(hashes) is dict and 1 <= len(hashes) <= 2048
             and all(type(name) is str and type(pin) is str and _SHA.fullmatch(pin) for name, pin in hashes.items()),
             'a completed sealed matched trial is required')
    validate_package_layout(hashes)
    _require(not {'failure.json', 'finalization-error.json', 'manifest.json'} & hashes.keys(),
             'failed or recursive trial package')
    return value, hashes


def execution_runtime(recipe_runtime, *, deadline):
    """Lab adapters are additional pins; original recipe pins are never replaced."""
    result = dict(recipe_runtime)
    package, total = Path(__file__).parent, 0
    for name in sorted(EXECUTION_FILES):
        _check(deadline)
        raw = read_package_file(package, name, max_bytes=min(1 << 20, (4 << 20) - total))
        total += len(raw)
        pin = sha256(raw).hexdigest()
        _require(name not in result or result[name] == pin, 'adapter conflicts with original recipe runtime')
        result[name] = pin
    return result


def _attachment(manifest_raw, read, scene, task, source_manifest_raw, source_files, task_raw, *, deadline):
    from weft.camera_reference_inputs import _load_camera
    from weft.camera_reference_recording import _prepare
    from weft.camera_shared_receipt import IDENTITIES
    from weft.shared_exercise_reference_inputs import _manifest
    from weft.shared_exercise_reference_trial import _runtime_versions

    manifest, hashes = _seal(manifest_raw)
    scene_pin, task_pin = canonical_hash(scene), canonical_hash(task)
    _require(scene.get('profile') == 'shared-world-v6' and task.get('profile') == 'camera-shared-task-v1'
             and scene.get('learned_policy_available') is False and scene.get('objects') == []
             and manifest.get('scene_sha256') == scene_pin and manifest.get('task_sha256') == task_pin,
             'trial must preserve the exact camera scene and task')
    selected = {}

    def member(name, maximum=8 << 20):
        _check(deadline)
        _require(name in hashes, 'missing sealed trial member: ' + name)
        if name not in selected:
            _require(len(selected) < MAX_SELECTED_FILES, 'selected trial file bound exceeded')
            remaining = MAX_SELECTED_BYTES - len(manifest_raw) - sum(map(len, selected.values()))
            _require(remaining > 0, 'selected trial byte bound exhausted')
            value = read(name, min(maximum, remaining), hashes[name])
            _require(0 < len(value) <= min(maximum, remaining) and sha256(value).hexdigest() == hashes[name],
                     'trial member identity or byte bound differs')
            selected[name] = value
        _require(len(selected[name]) <= maximum, 'trial member exceeds its document bound')
        return selected[name]

    plan, report = _json(member('plan.json', 256 << 10)), _json(member('report.json', 1 << 20))
    _require(plan.get('profile') == TRIAL_PROFILE + '-plan' and plan.get('status') == 'frozen'
             and report.get('profile') == TRIAL_PROFILE and report.get('status') == 'completed'
             and report.get('error') is None and report.get('same_initial_state') is True
             and plan.get('controller_family') == report.get('controller_family')
             and report['controller_family'] in {'camera_angle_overlay', 'camera_multiview_angle_overlay'}
             and plan.get('modes') == ['authored', 'reference_proposal']
             and type(plan.get('seed')) is int and 0 <= plan['seed'] < 2 ** 31 and plan['seed'] == report.get('seed')
             and plan.get('reset_options') == report.get('reset_options') == {'jitter': False, 'jitter_scale': 0.},
             'completed original-start camera comparison required')
    _require(type(plan.get('preserved_files')) is dict
             and all(hashes.get(name) == pin for name, pin in plan['preserved_files'].items()),
             'trial preserved input/runtime pins differ from its seal')
    _require(plan.get('runtime_versions') == _runtime_versions(), 'original trial runtime versions differ')
    recipe_raw = member('run/reference-controller.json', 1 << 20)
    recipe = _json(recipe_raw)
    recipe_sha = sha256(recipe_raw).hexdigest()
    _require(recipe_sha == plan.get('reference_controller_sha256')
             == report.get('reference_controller', {}).get('sha256')
             and report['reference_controller'].get('document') == 'run/reference-controller.json',
             'fitted recipe differs from the saved trial')
    _require(type(recipe.get('runtime_sha256')) is dict
             and all(plan.get('runtime_sha256', {}).get(name) == pin
                     and hashes.get('runtime/weft/' + name) == pin for name, pin in recipe['runtime_sha256'].items()),
             'recipe runtime differs from the retained trial implementation')
    envelope, _ = _prepare(recipe_raw, scene, task)
    _check(deadline)
    proposal_raw = member('inputs/proposal-manifest.json', 128 << 10)
    proposal_manifest, proposal_hashes = _manifest(proposal_raw)
    _require(proposal_manifest.get('profile') == 'whole-body-reference-candidate-study-v1'
             and proposal_manifest.get('status') == 'completed', 'completed original proposal required')
    aliases = report.get('proposal_input_aliases')
    _require(type(aliases) is dict and 1 <= len(aliases) <= 32
             and all(type(name) is str and target == _ALIASES.get(name, 'inputs/proposal/' + name)
                     and hashes.get(target) == proposal_hashes.get(name)
                     and target in plan['preserved_files'] for name, target in aliases.items()),
             'trial proposal aliases differ from original sealed members')
    _require(member('inputs/scene-manifest.json', 128 << 10) == source_manifest_raw
             and member('inputs/exercise-task.json', 64 << 10) == task_raw
             and member('run/exercise-task.json', 64 << 10) == task_raw
             and member('inputs/scene.json') == member('run/scene.json') == source_files.get('scene.json'),
             'trial source/task bytes differ from the independently admitted original run')
    source_hashes = _json(source_manifest_raw).get('files', {})
    proposal, source, task_path = Path('proposal'), Path('original-source'), Path('original-task/exercise-task.json')

    def preserved_read(directory, name, expected=None, maximum=8 << 20):
        _check(deadline)
        if directory == proposal:
            _require(name in aliases, 'trial omitted a required original proposal member')
            value = member(aliases[name], maximum)
        elif directory == source:
            value = source_manifest_raw if name == 'manifest.json' else source_files.get(name)
            _require(type(value) is bytes and (name == 'manifest.json' or source_hashes.get(name) == sha256(value).hexdigest()),
                     'original source view is absent or changed')
        else:
            _require(directory == task_path.parent and name == task_path.name, 'unexpected original input request')
            value = task_raw
        _require(0 < len(value) <= maximum and (expected is None or sha256(value).hexdigest() == expected),
                 'original input pin or member bound differs')
        return value

    inputs = _load_camera(proposal, source, task_path, proposal_raw, proposal_hashes,
        read=preserved_read, check=lambda: _check(deadline), finish=lambda: {},
        pins=(sha256(proposal_raw).hexdigest(), sha256(source_manifest_raw).hexdigest(), sha256(task_raw).hexdigest()))
    _require(set(aliases) == set(inputs.proposal_members)
             and inputs.identities == plan.get('source_identities') == report.get('source_identities')
             and inputs.identities['scene_sha256'] == scene_pin and inputs.identities['task_sha256'] == task_pin,
             'trial proposal/source identities or complete selected inventory differ')
    multiview = inputs.receipt().get('profile') == 'camera-multiview-reference-inputs-v1'
    _require(report['controller_family'] == ('camera_multiview_angle_overlay' if multiview else 'camera_angle_overlay')
             and recipe.get('profile') == ('camera-multiview-source-fit-controller-recipe-v1' if multiview
                                           else 'camera-source-fit-controller-recipe-v1'),
             'trial controller family differs from its admitted proposal and recipe')
    _require(recipe.get('input_human_sha256') == inputs.identities['original_human_sha256']
             and recipe.get('input_evidence_sha256') == inputs.identities['input_evidence_sha256']
             and recipe.get('angle_proposal_sha256') == inputs.identities['angle_proposal_sha256']
             and recipe.get('proposal_report_sha256') == inputs.identities['proposal_report_sha256'],
             'fitted recipe differs from the exact original proposal')
    if multiview:
        _require(all(recipe.get(key) == inputs.identities[key] for key in
                     ('additional_view_sha256', 'multiview_qualification_sha256')),
                 'multiview recipe differs from the exact retained qualification')
    rows = report.get('trials')
    _require(type(rows) is list and len(rows) == 2 and [row.get('mode') for row in rows] == plan['modes'],
             'both matched saved controller outcomes are required')
    _require(type(rows[0].get('initial_state_sha256')) is str and _SHA.fullmatch(rows[0]['initial_state_sha256'])
             and rows[0]['initial_state_sha256'] == rows[1].get('initial_state_sha256'), 'saved matched starts differ')
    outcomes = {}
    for row in rows:
        mode, steps = row['mode'], row.get('steps')
        recording = row.get('recording')
        recording_path = 'recordings/' + mode + '/'
        _require(type(recording) is dict and recording.get('directory') == recording_path[:-1]
                 and recording.get('manifest_sha256') == hashes.get(recording_path + 'manifest.json'),
                 'saved comparison recording seal is missing or different')
        recording_manifest = _json(member(recording_path + 'manifest.json', 64 << 10))
        recording_files = recording_manifest.get('files')
        _require(type(recording_files) is dict and set(recording_files) == {
            'scene.json', 'recording.json', 'events.jsonl', 'video.mp4', 'preview.gif', 'replay.html'}
            and all(hashes.get(recording_path + name) == pin for name, pin in recording_files.items()),
            'saved comparison recording inventory differs from the trial seal')
        path = 'proofs/' + mode + '-action-replay.json'
        replay = _json(member(path, 64 << 10))
        _require(row.get('status') == 'verified' and type(steps) is int and 1 <= steps <= task['interval']['horizon_steps']
                 and replay.get('exact_state_replay') is True and replay.get('steps') == steps
                 and replay.get('events') == steps + 1 and replay.get('scene_sha256') == scene_pin
                 and replay.get('task_sha256') == task_pin
                 and replay.get('protocol') == ('camera-shared-reference-full-state-replay-v1'
                     if mode == 'reference_proposal' else 'camera-shared-full-state-replay-v1')
                 and replay.get('policy_sha256') is None and replay.get('learned_actions_verified') is None
                 and all(replay.get(key) == task[key] for key in IDENTITIES)
                 and row.get('action_replay') == {**replay, 'path': path, 'status': 'passed'},
                 'saved replay receipt differs from its matched trial')
        if mode == 'reference_proposal':
            _require(replay.get('reference_controller_sha256') == recipe_sha
                     and replay.get('reference_actions_verified') == steps, 'saved fitted-action proof is incomplete')
        metrics = row.get('metrics')
        _require(type(metrics) is dict and type(metrics.get('success')) is bool, 'saved trial outcome is missing')
        outcomes[mode] = {'steps': steps, 'success': metrics['success'], 'mean_reward': metrics.get('mean_reward')}
    identity = {'reference_controller_sha256': recipe_sha,
        'reference_trial_manifest_sha256': sha256(manifest_raw).hexdigest(),
        'reference_proposal_manifest_sha256': sha256(proposal_raw).hexdigest()}
    summary = {'profile': PROFILE, **identity, 'kind': 'source_fit_authored_controller', 'learned': False,
        'root_assistance': task['control']['root_assistance'], 'saved_trial_outcomes': outcomes,
        'saved_replay_scope': 'Retained matched-trial receipts; attachment does not rerun or independently prove replay',
        'physical_validation': 'unverified', 'scope': SCOPE}
    custody = {'schema_version': 1, 'profile': PROFILE, **identity, 'scene_sha256': scene_pin, 'task_sha256': task_pin,
        'selected_trial_members': {name: hashes[name] for name in selected},
        'runtime_versions': deepcopy(plan['runtime_versions']),
        'source_identity': dict(inputs.identities), 'summary': summary,
        'scope': 'Selected sealed metadata only; omitted trial media/journals are not copied or freshly verified'}
    files = {PREFIX + 'manifest.json': manifest_raw, **{PREFIX + name: raw for name, raw in selected.items()},
             RECIPE: recipe_raw, ATTACHMENT: _bytes(custody)}
    return {'recipe_document': recipe_raw, 'envelope': envelope, 'identity': identity, 'summary': summary,
            'custody': custody, 'files': files,
            'runtime_sha256': execution_runtime(envelope['implementation_sha256'], deadline=deadline)}


def prepare_reference_attachment(directory, scene, task, source_manifest_raw, source_files, task_raw, *, deadline,
                                 expected_manifest_sha256=None):
    directory = Path(directory).absolute()
    _require('..' not in directory.parts and len(directory.parts) <= 64
             and not any(path.is_symlink() for path in (directory, *directory.parents)), 'unsafe trial path')
    _check(deadline)
    manifest_raw = read_package_file(directory, 'manifest.json', max_bytes=256 << 10,
                                     expected_sha256=expected_manifest_sha256)
    result = _attachment(manifest_raw,
        lambda name, maximum, pin: read_package_file(directory, name, max_bytes=maximum, expected_sha256=pin),
        scene, task, source_manifest_raw, source_files, task_raw, deadline=deadline)
    for name, raw in result['files'].items():
        if name.startswith(PREFIX):
            _check(deadline)
            read_package_file(directory, name[len(PREFIX):], max_bytes=len(raw), expected_sha256=sha256(raw).hexdigest())
    return result


def preflight_reference_attachment(contract, hashes, *, recipe_raw=None, scene=None, task=None, deadline=None):
    _require(contract.get('learned_policy_available') is False
             and {RECIPE, ATTACHMENT, PREFIX + 'manifest.json'} <= hashes.keys()
             and not any(name in {'policy.json', 'policy-evaluation.json', 'camera-policy-admission.json'}
                         or name.startswith(('learning-run/', 'policy-evaluation-run/', 'policy-evidence-run/')) for name in hashes),
             'fitted attachment cannot mix a learned policy or omit its original proof')
    _require(all(type(contract.get(key)) is str and _SHA.fullmatch(contract[key]) for key in (
        'reference_controller_sha256', 'reference_trial_manifest_sha256', 'reference_proposal_manifest_sha256')),
        'fitted attachment requires exact distinct identities')
    recipe_pin = sha256(hashes[RECIPE]).hexdigest() if type(hashes.get(RECIPE)) is bytes else hashes.get(RECIPE)
    _require(recipe_pin == contract['reference_controller_sha256'], 'fitted recipe hash differs from the lab contract')
    if recipe_raw is not None:
        from weft.camera_reference_recording import _runtime_hashes
        deadline = time.monotonic() + 10 if deadline is None else deadline
        _check(deadline)
        _require(sha256(recipe_raw).hexdigest() == contract['reference_controller_sha256']
                 and canonical_hash(scene) == contract.get('scene_sha256')
                 and canonical_hash(task) == contract.get('task_sha256'), 'fitted preflight identities differ')
        runtime = _runtime_hashes(_json(recipe_raw).get('runtime_sha256'))
        _check(deadline)
        current = execution_runtime(runtime, deadline=deadline)
        _require(contract.get('runtime_sha256') == current
                 and all(contract.get('implementation_sha256', {}).get(name) == pin for name, pin in current.items()),
                 'fitted lab execution runtime differs before native admission')


def load_reference_attachment(files, scene, task, source_manifest_raw, source_files, task_raw, contract, *, deadline):
    preflight_reference_attachment(contract, files)
    def read(name, maximum, pin):
        value = files.get(PREFIX + name)
        _require(type(value) is bytes and len(value) <= maximum and sha256(value).hexdigest() == pin,
                 'retained selected trial member is missing or changed')
        return value
    result = _attachment(files[PREFIX + 'manifest.json'], read, scene, task, source_manifest_raw, source_files,
                         task_raw, deadline=deadline)
    _require(all(files.get(name) == raw for name, raw in result['files'].items())
             and {name for name in files if name.startswith(PREFIX)} == {name for name in result['files'] if name.startswith(PREFIX)}
             and all(contract.get(key) == value for key, value in result['identity'].items())
             and contract.get('exercise_reference') == result['summary']
             and contract.get('runtime_sha256') == result['runtime_sha256'], 'retained fitted attachment differs')
    return result


def reference_capabilities(capabilities, summary):
    result = deepcopy(capabilities)
    result['controllers']['fitted_reference_controller'] = True
    result['fitted_reference'] = deepcopy(summary)
    return result
