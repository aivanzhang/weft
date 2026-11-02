"""Prepare a manual camera-aware lab from sealed source evidence and explicit calibration."""

import json
from hashlib import sha256
from pathlib import Path
import shutil
import time

from weft.camera_body import make_camera_body
from weft.human import HumanEnv, human_xml
from weft.io import digest, local_time, write_json
from weft.pipeline import _implementation_files, verify_artifacts
from weft.whole_body_feasibility import audit_reference_feasibility


def _root_view_inputs(config_path, out, *, additional=False):
    inputs = {}

    def read(path, name, limit):
        path = Path(path).resolve()
        if (path.parent/'manifest.json').exists() and out.is_relative_to(path.parent):
            raise ValueError('Keep the new lab outside sealed second-view inputs')
        if path.stat().st_size > limit:
            raise ValueError('Second-view input exceeds its byte bound')
        payload = path.read_bytes()
        if len(payload) > limit:
            raise ValueError('Second-view input changed beyond its byte bound')
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError('Second-view inputs must be JSON objects')
        inputs[name] = (path, payload)
        return value

    config_path = Path(config_path).resolve()
    config = read(config_path, 'config.json', 64*1024)
    fields = ({'schema_version', 'name', 'evidence', 'camera', 'clock_provenance'} if additional else
              {'schema_version', 'first_name', 'second_name', 'second_evidence', 'second_camera',
               'clock_provenance', 'world_provenance'})
    if (set(config) != fields or type(config['schema_version']) is not int or config['schema_version'] != 1
            or any(not isinstance(config[name], str) or not 1 <= len(config[name].strip()) <= 2000
                   for name in fields-{'schema_version'})):
        raise ValueError('Second-view config must explicitly identify inputs, shared world and aligned clocks')
    second = {'name': config['name' if additional else 'second_name'], 'clock_provenance': config['clock_provenance'],
              'evidence': read(config_path.parent/config['evidence' if additional else 'second_evidence'], 'evidence.json', 32*1024*1024),
              'camera': read(config_path.parent/config['camera' if additional else 'second_camera'], 'camera.json', 8*1024*1024)}
    return config, second, inputs


def prepare_camera_body_lab(source_run, camera_path, out, *, ground_path=None, root_view_path=None, body_view=None, partial_segments=False, additional_view_paths=()):
    started = time.monotonic()
    source_run, camera_path, out = (Path(value).resolve() for value in (source_run, camera_path, out))
    if (not isinstance(additional_view_paths, (list, tuple)) or len(additional_view_paths) > 2
            or additional_view_paths and (root_view_path is None or partial_segments)):
        raise ValueError('At most two additional views require --root-view; partial-segment transport is not qualified')
    if body_view is not None and (body_view not in ('directions', 'dimensions', 'both') or root_view_path is None):
        raise ValueError('Partial two-view body geometry requires --root-view and an explicit supported mode')
    if type(partial_segments) is not bool or partial_segments and (body_view not in ('directions', 'both') or root_view_path is None):
        raise ValueError('Partial segments require --root-view and --body-view directions or both')
    if out.exists() or out.is_relative_to(source_run):
        raise ValueError('Choose a new camera-body lab outside the immutable source run')
    if (camera_path.parent/'manifest.json').exists() and out.is_relative_to(camera_path.parent):
        raise ValueError('Keep the new lab outside the sealed camera-input bundle')
    verified = verify_artifacts(source_run)
    manifest_bytes = (source_run/'manifest.json').read_bytes()
    if sha256(manifest_bytes).hexdigest() != verified['manifest_sha256']:
        raise ValueError('Source manifest changed after verification')
    manifest = json.loads(manifest_bytes)
    total_bytes = 0
    for name in manifest['files']:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or path.as_posix() != name or name == 'manifest.json':
            raise ValueError('Source manifest paths must be canonical relative names')
        total_bytes += (source_run/name).stat().st_size
    if total_bytes > 400*1024*1024:
        raise ValueError('Source bundle exceeds the 400-MiB copy bound')

    def read_input(name):
        path = source_run/name
        if name not in manifest['files'] or path.stat().st_size > 32*1024*1024:
            raise ValueError('Required camera-body input is not sealed or exceeds its byte bound')
        payload = path.read_bytes()
        if len(payload) > 32*1024*1024 or sha256(payload).hexdigest() != manifest['files'][name]:
            raise ValueError('Source input changed after verification')
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError('Camera-body inputs must be JSON objects')
        return value

    original, evidence, source = (read_input(name) for name in ('human.json', 'evidence.json', 'source.json'))
    selector_file = 'policy.json' if 'policy.json' in manifest['files'] else 'task-proposal.json'
    selector = read_input(selector_file)
    if not isinstance(selector.get('selected_joint'), str):
        raise ValueError('Camera-body source must identify an explicit control joint')
    artifact = source.get('artifact')
    if (not isinstance(artifact, str) or Path(artifact).name != artifact
            or manifest['files'].get(artifact) != original.get('source_sha256')
            or source.get('sha256') != original.get('source_sha256')):
        raise ValueError('The sealed source video must match the original body')
    if camera_path.stat().st_size > 8*1024*1024:
        raise ValueError('Camera JSON exceeds the 8-MiB input bound')
    camera_bytes = camera_path.read_bytes()
    if len(camera_bytes) > 8*1024*1024:
        raise ValueError('Camera JSON changed beyond its byte bound')
    camera = json.loads(camera_bytes)
    ground_bytes = None
    if ground_path is not None:
        ground_path = Path(ground_path).resolve()
        if (ground_path.parent/'manifest.json').exists() and out.is_relative_to(ground_path.parent):
            raise ValueError('Keep the new lab outside the sealed ground-input bundle')
        if ground_path.stat().st_size > 64*1024:
            raise ValueError('Ground JSON exceeds the 64-KiB input bound')
        ground_bytes = ground_path.read_bytes()
        if len(ground_bytes) > 64*1024:
            raise ValueError('Ground JSON changed beyond its byte bound')
    root_inputs, additional_views = {}, []
    if root_view_path is not None:
        if ground_bytes is None:
            raise ValueError('Two-view root requires explicit supplied ground, not a first-pose floor fit')
        root_config, second, root_inputs = _root_view_inputs(root_view_path, out)
        for index, path in enumerate(additional_view_paths):
            _, view, inputs = _root_view_inputs(path, out, additional=True)
            additional_views.append(view)
            root_inputs.update({f'extra-{index}/{name}': item for name, item in inputs.items()})
    implementation = _implementation_files()
    implementation_hashes = {name: digest(path) for name, path in implementation.items()}
    spec, estimate = make_camera_body(evidence, camera, original)
    ground_audit = root_report = triangulation = body_report = None
    if ground_bytes is not None:
        from weft.camera_ground import apply_camera_ground, audit_ground_clearance
        spec = apply_camera_ground(spec, json.loads(ground_bytes))
    original_camera_spec = spec
    if root_inputs:
        from weft.multiview_root import make_two_view_root
        first = {'name': root_config['first_name'], 'evidence': evidence, 'camera': camera,
                 'clock_provenance': root_config['clock_provenance']}
        spec, root_report, triangulation = make_two_view_root(spec, first, second,
            world_provenance=root_config['world_provenance'], additional_views=additional_views)
    root_only_spec = spec
    if body_view is not None:
        from weft.multiview_body import make_two_view_body
        spec, body_report = make_two_view_body(spec, triangulation, evidence, second['evidence'],
            mode=body_view, partial_segments=partial_segments, additional_evidence=[view['evidence'] for view in additional_views])
    if ground_bytes is not None:
        ground_audit = audit_ground_clearance(spec)
    feasibility = audit_reference_feasibility(spec)
    env = HumanEnv(spec)
    try:
        env.select_joint(selector['selected_joint'])
        env.scene['joint_names'] = list(env.joint_names)
        references = [env.reference_snapshot(index) for index in range(len(spec['reference']))]
        scene = env.scene
    finally:
        env.close()
    out.mkdir(parents=True)
    run = out/'run'; run.mkdir()
    shared = {artifact, 'source.json', 'evidence.json', 'pose-landmarks.json', 'background.jpg',
              'tracks-contact-sheet.png', 'tracks.gif', 'joint-angles.json', 'identity-audit.json',
              'model.json', 'pose-quality.json'}
    copied = {}
    for name, expected in manifest['files'].items():
        destination = name if name in shared else 'inputs/original/'+name
        target = run/destination; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_run/name, target)
        if digest(target) != expected:
            raise ValueError('Source changed during copying; candidate remains unsealed')
        copied[name] = destination
    (run/'source-input-manifest.json').write_bytes(manifest_bytes)
    (run/'camera.json').write_bytes(camera_bytes)
    if ground_bytes is not None:
        (run/'ground.json').write_bytes(ground_bytes)
        write_json(run/'ground-clearance.json', ground_audit)
    write_json(run/'human.json', spec)
    (run/'human.xml').write_text(human_xml(spec))
    write_json(run/'camera-reference.json', estimate)
    if root_inputs:
        for name, (_, payload) in root_inputs.items():
            target = run/'inputs/two-view-root'/name; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        write_json(run/'inputs/monocular-camera-body.json', original_camera_spec)
        write_json(run/'two-view-root.json', root_report)
        write_json(run/'two-view-triangulation.json', triangulation)
    if body_report is not None:
        write_json(run/'inputs/two-view-root-body.json', root_only_spec)
        write_json(run/'two-view-body.json', body_report)
    write_json(run/'body-frame.json', spec['body_frame'])
    write_json(run/'scene.json', scene)
    write_json(run/'task.json', scene['task'])
    write_json(run/'human-reference.json', {'frames': references, 'scope': 'Kinematic reference inspection, not dynamic simulation'})
    write_json(run/'reference-feasibility.json', feasibility)
    write_json(run/'repair-ledger.json', {'proposals': [], 'scope': 'Unpromoted additional-input candidate; no physical repair accepted'})
    write_json(run/'reconstruction.json', {'scope': 'No new dynamic fidelity or physical-accuracy claim',
               'reference_feasibility': {key: feasibility[key] for key in ('status', 'summary')},
               'camera_pose_coverage': estimate['coverage']})
    provenance = {'created_at': local_time(), 'source_run': str(source_run), 'source_verification': verified,
                  'camera_file_sha256': sha256(camera_bytes).hexdigest(), 'copied_source_files': copied,
                  'selected_joint_source': ('Original policy metadata only; no weights or performance transferred'
                                            if selector_file == 'policy.json' else
                                            'Source task proposal only; no learned policy exists'),
                  'new_training': False, 'new_inference': False, 'reference_joints_used_for_fitting': False}
    if ground_bytes is not None:
        provenance['ground_file_sha256'] = sha256(ground_bytes).hexdigest()
    if root_inputs:
        provenance['two_view_inputs'] = {name: {'path': str(path), 'sha256': sha256(payload).hexdigest()}
                                        for name, (path, payload) in root_inputs.items()}
        provenance['monocular_camera_reference_unchanged'] = True
    write_json(run/'camera-provenance.json', provenance)
    for name, path in implementation.items():
        target = run/'implementation-source'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if digest(target) != implementation_hashes[name]:
            raise RuntimeError('Implementation changed during preparation; candidate remains unsealed')
    if _implementation_files() != implementation:
        raise RuntimeError('Implementation file set changed during preparation; candidate remains unsealed')
    write_json(run/'implementation.json', implementation_hashes)
    report = {'created_at': local_time(), 'profile': 'exercise-camera-candidate-v1', 'status': 'manual_candidate',
              'physical_validation': 'unverified', 'accepted': False, 'learned_policy_available': False,
              'source_frames': estimate['source_frames'], 'camera_pose_coverage': estimate['coverage'],
              'reference_feasibility': {key: feasibility[key] for key in ('status', 'summary')},
              'paid_requests': 0, 'runtime_seconds': time.monotonic()-started,
              'scope': 'Executable manual/reference-control candidate; fixed-scale 3D estimates and authored forces, not recovered anatomy'}
    if ground_audit is not None:
        report['ground_clearance'] = {key: ground_audit[key] for key in ('status', 'scope', 'summary')}
    if root_report is not None:
        report['two_view_root'] = {key: root_report[key] for key in ('method', 'coverage', 'scope')}
        report['scope'] = ('Manual multi-view-root candidate; monocular shape and authored forces, not recovered anatomy' if additional_views
                           else 'Manual two-view-root candidate; monocular shape and authored forces, not recovered anatomy')
    if body_report is not None:
        report['two_view_body'] = {key: body_report[key] for key in ('contract', 'mode', 'group_coverage', 'scope')}
        report['scope'] = ('Manual partial multi-view body candidate; explicit monocular priors and authored dynamics, not recovered anatomy' if additional_views
                           else 'Manual partial two-view body candidate; explicit monocular priors and authored dynamics, not recovered anatomy')
    write_json(run/'report.json', report)
    write_json(run/'manifest.json', {'schema': 1, 'created_at': local_time(),
               'files': {str(path.relative_to(run)): digest(path) for path in sorted(run.rglob('*')) if path.is_file()}})
    if verify_artifacts(source_run) != verified or digest(camera_path) != sha256(camera_bytes).hexdigest():
        raise RuntimeError('An input changed during preparation; no ready lab published')
    if ground_bytes is not None and digest(ground_path) != sha256(ground_bytes).hexdigest():
        raise RuntimeError('Ground input changed during preparation; no ready lab published')
    if any(digest(path) != sha256(payload).hexdigest() for path, payload in root_inputs.values()):
        raise RuntimeError('Second-view input changed during preparation; no ready lab published')
    write_json(out/'lab-config.json', {'profile': 'exercise', 'control_profile': 'camera_body',
               'created_at': local_time(), 'source_run': str(source_run), 'episodes': 0, 'eval_episodes': 0,
               'reference_feasibility': report['reference_feasibility'],
               **({'ground_clearance': report['ground_clearance']} if ground_audit is not None else {})})
    events = [{'index': i, 'stage': stage, 'time': local_time(), 'message': message} for i, (stage, message) in enumerate([
        ('source', 'Reused sealed video and raw poses; no new inference'),
        ('construct', 'Applied supplied cameras, retained full travel and every unavailable pose'),
        ('evaluate', 'Independent floor preflight: '+feasibility['status']),
        ('ready', 'Manual camera-aware candidate ready; no learned policy or physical-accuracy approval')])]
    write_json(out/'status.json', {'status': 'ready', 'events': events, 'result': report})
    return {'directory': str(out), 'verification': verify_artifacts(run),
            'reference_feasibility': feasibility['status'], 'camera_pose_coverage': estimate['coverage'],
            'learned_policy_available': False, 'runtime_seconds': report['runtime_seconds'],
            'inspect': f'.venv-lab/bin/python -m weft lab --resume {out}'}
