"""Bounded saved-run metadata for the lab's five inspection stages.

``load_inspection(run, profile=...)`` reads only a manifest and selected metadata;
it never decodes media, reads observation arrays, builds an environment or scores
evidence. Metadata hashes must match the saved manifest. Linked media/documents
are checked for confined regular-file paths, not independently rehashed here.
The live lab verifies the full run separately before enabling simulation.
"""

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat

from weft.io import canonical_hash
from weft.lab_evidence import build_lab_evidence


MAX_BASE_SUMMARY_BYTES = 64 * 1024
MAX_SUMMARY_BYTES = 128 * 1024
MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_MANIFEST_FILES = 4096
MAX_ACTORS = 128
_SHA = re.compile(r'[0-9a-f]{64}')
_VIDEO = re.compile(r'source\.(mp4|webm|mov|m4v|avi|mkv)')
_SESSION = re.compile(r'session-[0-9a-f]{12}')
_STAGES = (('original', 'Original', 'evidence'), ('tracking', 'Tracks', 'evidence'),
           ('geometry', 'Geometry', 'world'), ('sandbox', 'Sandbox', 'play'),
           ('recordings', 'Recordings', 'play'))


def _path(name):
    if (not isinstance(name, str) or not name or len(name) > 512 or '\\' in name
            or any(ord(c) < 32 for c in name) or name.startswith('/')
            or any(part in ('', '.', '..') for part in name.split('/'))):
        raise ValueError('Unsafe inspection artifact path')
    return PurePosixPath(name).parts


def _open(root_fd, name):
    parts = _path(name)
    parent = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise ValueError('Inspection artifacts must be regular files')
        return fd
    finally:
        os.close(parent)


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate inspection metadata key')
        result[key] = value
    return result


def _nonfinite(_):
    raise ValueError('Nonfinite inspection metadata')


def _float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('Nonfinite inspection metadata')
    return number


def _count(value):
    if type(value) is not int or not 0 <= value <= 100000:
        raise ValueError('Invalid saved inspection count')
    return value


def _text(value, limit=400):
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError('Oversized or invalid inspection text')
    return value


def _timeline(scene, human, decoded, exercise_task):
    result = {'status': 'unavailable', 'reference_span_s': None, 'reference_poses': None,
              'task_window_s': None, 'task_duration_s': None, 'task_steps': None,
              'source_slots': None, 'alignment_method': None, 'native_source_clock': 'not_read_here',
              'media_sync_verified': False,
              'scope': 'Declared reference times are not native source timestamps or synchronized video playback.'}
    reference = human.get('reference')
    if not isinstance(reference, list) or not 1 <= len(reference) <= 100000:
        return result
    finite = lambda value: type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 3600
    times = [row.get('time_s') if isinstance(row, dict) else None for row in reference]
    if any(not finite(value) for value in times) or any(a >= b for a, b in zip(times, times[1:])):
        return result
    result.update(status='available', reference_span_s=[times[0], times[-1]], reference_poses=len(times))
    frame = scene.get('coordinate_frame', {})
    clock = frame.get('clock_binding', {}) if isinstance(frame, dict) else {}
    method = clock.get('alignment_method') if isinstance(clock, dict) else None
    if isinstance(method, str) and len(method) <= 200:
        result['alignment_method'] = method
    metadata, source_sha = human.get('reference_evidence'), human.get('source_sha256')
    if isinstance(metadata, list) and len(metadata) == len(times) and isinstance(source_sha, str) and _SHA.fullmatch(source_sha):
        frames = []
        for row, timestamp in zip(metadata, times):
            evidence_id = row.get('evidence_id') if isinstance(row, dict) else None
            match = re.fullmatch(re.escape(source_sha) + r':frame:(\d+)(?::.*)?', evidence_id) if isinstance(evidence_id, str) else None
            if match is None or row.get('time_s') != timestamp:
                break
            frames.append(int(match[1]))
        if (decoded is not None and len(frames) == len(times) and len(set(frames)) == len(frames)
                and all(0 <= index < decoded for index in frames) and frames == sorted(frames)):
            result['source_slots'] = {'with_reference': len(frames), 'without_reference': decoded - len(frames),
                                      'leading_without_reference': frames[0],
                                      'trailing_without_reference': decoded - frames[-1] - 1}
    # The shared runtime starts at first retained reference + its declared task offset.
    task = exercise_task.get('interval') if exercise_task else scene.get('task')
    dt = scene.get('dt')
    if scene.get('profile') in ('shared-world-v1', 'shared-world-v2', 'shared-world-v3', 'shared-world-v4', 'shared-world-v6') and isinstance(task, dict) and finite(dt) and dt > 0:
        offset, steps = task.get('start_offset_s'), task.get('horizon_steps')
        if finite(offset) and type(steps) is int and 1 <= steps <= 100000:
            start, duration = times[0] + offset, steps * dt
            if 'native_interval' in task:
                from weft.native_interval import validate_interval
                try:
                    marker = validate_interval(scene)
                except (KeyError, TypeError, ValueError):
                    return result
                if marker != task['native_interval']:
                    return result
                duration = marker['duration_s']
                result['terminal_control_duration_s'] = marker['terminal_duration_s']
                result['native_source_clock'] = 'declared_native_presentation_binding'
                result['scope'] = 'Retained native presentation interval; camera capture timing and synchronized video playback remain unverified.'
            if duration <= 3600 and start + duration <= times[-1] + 1e-9:
                result.update(task_window_s=[start, start + duration], task_duration_s=duration, task_steps=steps)
    return result


def _fullspan_summary(receipt, scene, task, files, timeline):
    if (receipt.get('profile') != 'shared-exercise-fullspan-preparation-v1'
            or receipt.get('status') != 'ready' or scene.get('profile') != 'shared-world-v2'
            or receipt.get('runnable_scene_published') is not True
            or receipt.get('old_policy_or_evaluation_attached') is not False
            or any(type(receipt.get(key)) is not int or receipt[key] != 0
                   for key in ('physics_steps', 'resets', 'training_episodes', 'inference_calls', 'api_calls', 'video_decodes'))):
        raise ValueError('Invalid full-reference preparation summary')
    if (task.get('profile') != 'shared-exercise-task-v1'
            or receipt.get('scene_sha256') != canonical_hash(scene) or receipt.get('task_sha256') != canonical_hash(task)
            or task.get('scene_sha256') != canonical_hash(scene)
            or task.get('interval') != {key: scene.get('task', {}).get(key) for key in ('start_offset_s', 'horizon_steps')}
            or receipt.get('gates_sha256') != task.get('gates_sha256') or timeline['task_window_s'] is None
            or any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-9)
                   for a, b in zip(timeline['task_window_s'], timeline['reference_span_s']))):
        raise ValueError('Full-reference preparation scene, task or interval differs')
    expected_interval = {'first_time_s': timeline['reference_span_s'][0],
                         'last_time_s': timeline['reference_span_s'][1], 'horizon_steps': timeline['task_steps'],
                         'dt': scene.get('dt'), 'substeps': scene.get('substeps'),
                         'reference_poses': timeline['reference_poses']}
    if receipt.get('reference_interval') != expected_interval:
        raise ValueError('Full-reference preparation clocks or pose count differ')
    # The ordinary lab adds one source-run layer around the unchanged preparation package.
    nested = files.get('source-run/fullspan.json') == files['fullspan.json']
    archive = 'source-run/source-run/manifest.json' if nested else 'source-run/manifest.json'
    source_manifest = receipt.get('input_manifest_sha256')
    if (not isinstance(source_manifest, str) or not _SHA.fullmatch(source_manifest)
            or receipt.get('original_manifest_path') != 'source-run/manifest.json'
            or files.get(archive) != receipt.get('input_manifest_sha256')):
        raise ValueError('Full-reference preparation original archive differs')
    return {'status': 'preparation_only', 'scene_sha256': receipt['scene_sha256'],
            'task_sha256': receipt['task_sha256'], 'input_manifest_sha256': receipt['input_manifest_sha256'],
            'scope': 'The task spans the retained reference without new geometry, observations or physics checks. '
                     'Missing source frames remain missing; preparation alone does not establish task success.',
            'artifact': 'fullspan.json'}


def _summary(run, profile):
    root_fd = os.open(Path(run), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    consumed = 0
    files = {}
    links = {}
    verified = {}
    retained = {}
    try:
        def read(name, limit=256 * 1024):
            nonlocal consumed
            if name != 'manifest.json' and name not in files:
                return {}
            with os.fdopen(_open(root_fd, name), 'rb') as stream:
                size = os.fstat(stream.fileno()).st_size
                if size > limit or consumed + size > MAX_METADATA_BYTES:
                    raise ValueError('Inspection metadata exceeds its byte limit')
                raw = stream.read(min(limit, MAX_METADATA_BYTES - consumed) + 1)
            consumed += len(raw)
            if len(raw) != size or len(raw) > limit or consumed > MAX_METADATA_BYTES:
                raise ValueError('Inspection metadata changed or exceeds its byte limit')
            digest = hashlib.sha256(raw).hexdigest()
            if name != 'manifest.json' and digest != files[name]:
                raise ValueError('Inspection metadata hash does not match the saved manifest')
            result = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_nonfinite, parse_float=_float)
            if not isinstance(result, dict):
                raise ValueError('Inspection metadata must be an object')
            verified[name] = {'source_name': name, 'path': 'run/' + name, 'sha256': digest, 'bytes': len(raw)}
            if name == 'policy.json':
                retained[name] = raw
            return result

        manifest = read('manifest.json', MAX_MANIFEST_BYTES)
        files = manifest.get('files')
        if not isinstance(files, dict) or not 1 <= len(files) <= MAX_MANIFEST_FILES:
            raise ValueError('Missing or oversized inspection manifest')
        for name, digest in files.items():
            _path(name)
            if not isinstance(digest, str) or not _SHA.fullmatch(digest):
                raise ValueError('Invalid inspection manifest hash')

        def artifact(name, label):
            if name not in files:
                return None
            if name not in links:
                if len(links) >= 32:
                    raise ValueError('Too many inspection links')
                with os.fdopen(_open(root_fd, name), 'rb') as stream:
                    size = os.fstat(stream.fileno()).st_size
                links[name] = {'name': name, 'label': label, 'bytes': size, 'sha256': files[name]}
            return links[name]

        source_meta = read('source.json')
        scene = read('scene.json', 4 * 1024 * 1024)
        human = scene.get('human_spec', {})
        if not human and profile == 'exercise':
            human = read('human.json', 4 * 1024 * 1024)
        if not isinstance(human, dict):
            raise ValueError('Invalid saved body metadata')
        binding = scene.get('source_binding') or {}
        if not isinstance(binding, dict):
            raise ValueError('Invalid saved source binding')
        source_names = [name for name in files if _VIDEO.fullmatch(name)]
        if len(source_names) > 1:
            raise ValueError('Ambiguous original source video')
        video = artifact(source_names[0], 'Original video') if source_names else None
        hashes = [value for value in (source_meta.get('sha256'), binding.get('source_sha256'),
                                      video['sha256'] if video else None) if value is not None]
        if any(not isinstance(v, str) or not _SHA.fullmatch(v) for v in hashes) or len(set(hashes)) > 1:
            raise ValueError('Saved source identities disagree')
        counts = [value for value in (source_meta.get('decoded_frames'), binding.get('total_source_frames'))
                  if value is not None]
        if len({_count(value) for value in counts}) > 1:
            raise ValueError('Saved source frame counts disagree')
        decoded = counts[0] if counts else None
        source = {'sha256': hashes[0] if hashes else None, 'decoded_frames': decoded,
                  'video': video['name'] if video else None,
                  'scope': 'Saved source identity and decoded count; not physical calibration.'}

        bound = {}
        rows = binding.get('entity_bindings', [])
        documents = binding.get('evidence_documents', {})
        if not isinstance(rows, list) or len(rows) > MAX_ACTORS or not isinstance(documents, dict) or len(documents) > 8:
            raise ValueError('Oversized saved source binding')
        for digest, name in documents.items():
            if not isinstance(digest, str) or not _SHA.fullmatch(digest) or files.get(name) != digest:
                raise ValueError('Observation link is not pinned by the run manifest')
            artifact(name, 'Source observation document')
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('Invalid saved entity binding')
            entity = _text(row.get('world_entity_id'), 128)
            if entity in bound:
                raise ValueError('Duplicate saved entity binding')
            counts = {key: _count(row.get(key)) for key in ('observed_slots', 'missing_slots', 'pre_init_slots')}
            if decoded is None or sum(counts.values()) != decoded:
                raise ValueError('Saved entity slots do not cover the source')
            bound[entity] = {'kind': row.get('provenance_mode'), 'counts': counts,
                             'document_sha256': row.get('observation_document_sha256'),
                             'observation_entity_id': row.get('observation_entity_id')}
            if bound[entity]['kind'] in ('observed_track', 'inferred_prior'):
                if bound[entity]['document_sha256'] not in documents:
                    raise ValueError('Entity observation document is unavailable')
                _text(bound[entity]['observation_entity_id'], 128)
            elif bound[entity]['kind'] != 'authored_addition':
                raise ValueError('Unknown saved entity provenance mode')
            elif (counts['observed_slots'] or counts['pre_init_slots']
                  or bound[entity]['document_sha256'] is not None or bound[entity]['observation_entity_id'] is not None):
                raise ValueError('Authored additions must not claim source measurements')

        objects = scene.get('objects', [])
        if not isinstance(objects, list) or len(objects) + bool(human) > MAX_ACTORS:
            raise ValueError('Too many inspection actors')
        if profile == 'exercise' and human:
            # Standalone body scene objects are anatomical segments, not separate actors.
            objects = []
        actors = []
        reference = human.get('reference')
        reference_poses = len(reference) if isinstance(reference, list) else None
        if reference_poses is not None:
            _count(reference_poses)
        entities = ([{'id': 'human', 'label': 'Articulated human', 'shape': 'simplified body', 'role': 'human'}]
                    if human else []) + objects
        for index, obj in enumerate(entities):
            if not isinstance(obj, dict):
                raise ValueError('Invalid inspection actor')
            identifier = _text(obj.get('id'), 128)
            if any(actor['id'] == identifier for actor in actors):
                raise ValueError('Duplicate inspection actor')
            evidence = bound.get(identifier)
            label = _text(obj.get('label') or identifier, 120)
            generic = (label == identifier or re.match(r'^(sam-\d+|Model object\b)', label)
                       or 'source-linked solid-box proxy' in label)
            if generic:
                label = (f'Tracked object {index + 1 - bool(human)}'
                         if evidence and evidence['kind'] == 'observed_track' else f'Object {index + 1 - bool(human)}')
            label = _text(label, 120)
            actors.append({'id': identifier, 'label': label,
                           'label_origin': 'generic display label' if generic else 'saved display label; not verified identity',
                           'role': _text(obj.get('role', 'unknown'), 40),
                           'geometry': _text(obj.get('shape', 'unknown'), 40),
                           'source_binding': evidence,
                           'scope': ('Source-linked model estimate; shape and physics are assumptions.'
                                     if evidence and evidence['kind'] == 'observed_track'
                                     else 'Source-linked prior, not an observed track.' if evidence and evidence['kind'] == 'inferred_prior'
                                     else 'Authored model; no source-qualified observation binding available.')})

        study = read('geometry-study.json')
        proposal = read('geometry-proposal.json')
        hypothesis = read('geometry-hypothesis.json')
        exercise_task = read('exercise-task.json', 64 * 1024)
        timeline = _timeline(scene, human, decoded, exercise_task)
        fullspan = None
        if 'fullspan.json' in files:
            fullspan = _fullspan_summary(read('fullspan.json', 64 * 1024), scene, exercise_task, files, timeline)
            if study or proposal or hypothesis:
                raise ValueError('Full-reference preparation cannot relabel an earlier geometry comparison')
            artifact('fullspan.json', 'Full-reference preparation and preserved limitations')
        config = {}
        if 'geometry-hypothesis.json' in files:
            if (hypothesis.get('profile') != 'shared-world-geometry-hypothesis-v1'
                    or hypothesis.get('status') != 'ready' or hypothesis.get('source_fit') != 'unverified'
                    or hypothesis.get('physical_validation') != 'unverified'):
                raise ValueError('Invalid authored geometry hypothesis summary')
            if proposal or study:
                raise ValueError('An authored geometry alternative cannot inherit an earlier geometry comparison')
            config = read('input-config.json', 64 * 1024)
            if (config.get('profile') != 'shared-world-geometry-hypothesis-config-v1'
                    or hypothesis.get('candidate_scene_sha256') != canonical_hash(scene)
                    or hypothesis.get('input_config_sha256') != files.get('input-config.json')
                    or hypothesis.get('base_scene_sha256') != config.get('base_scene_sha256')):
                raise ValueError('Geometry hypothesis configuration or scene identity differs')
        pose_counts = study.get('source_pose_status_counts')
        if pose_counts is not None:
            if not isinstance(pose_counts, dict) or len(pose_counts) > 16:
                raise ValueError('Invalid saved pose-status summary')
            pose_counts = {_text(key, 64): _count(value) for key, value in pose_counts.items()}
            if decoded is None or sum(pose_counts.values()) != decoded:
                raise ValueError('Saved pose-status counts do not cover the source')
        panels = study.get('panels', [])
        if not isinstance(panels, list) or len(panels) > 8:
            raise ValueError('Too many geometry panels')
        previews = []
        for name in panels:
            if len(_path(name)) != 1 or not name.endswith(('.png', '.jpg', '.jpeg')):
                raise ValueError('Invalid geometry panel path')
            panel = artifact('geometry-inspection/' + name, 'Initial geometry comparison')
            if panel is None:
                raise ValueError('Declared geometry panel is not sealed with this run')
            previews.append(panel)
        assumptions = config.get('assumptions', []) if hypothesis else proposal.get('assumptions', [])
        if not isinstance(assumptions, list) or len(assumptions) > 32:
            raise ValueError('Invalid geometry assumption summary')
        geometry = {'status': 'initial_hypothesis' if proposal or previews or hypothesis else 'no_saved_comparison',
                    'panels': previews, 'assumptions': [_text(value, 1000) for value in assumptions],
                    'scope': ('Authored compound-shape alternative: only initial clearance was checked. Source-image fit, full-motion clearance, grasp and physical accuracy are unverified.' if hypothesis else
                              'Initial authored geometry hypotheses, not recovered shape, grasp, sustained support or physical accuracy.')}
        for name, label in [('source.json', 'Source metadata'), ('geometry-proposal.json', 'Geometry assumptions'),
                            ('geometry-study.json', 'Geometry comparison summary'), ('evidence.json', 'Frame measurements'),
                            ('observations.json', 'Frame observations'), ('coverage-summary.json', 'Tracking coverage'),
                            ('pose-quality.json', 'Pose-quality summary')]:
            artifact(name, label)
        if scene.get('profile') == 'shared-world-v3':
            for name, label in [('inventory.json', 'Declared person-only scope; absence is not detected'),
                                ('source-clock.json', 'Original source clocks and unknown timestamps'),
                                ('reference-feasibility.json', 'Full reference geometry audit; not task success'),
                                ('reconstruction.json', 'Person-only construction and remaining assumptions')]:
                artifact(name, label)
        if scene.get('profile') == 'shared-world-v4':
            for name, label in [('support-config.json', 'Authored support geometry and load requirements'),
                                ('exercise-task.json', 'Frozen support task; simulated loads, not measured forces'),
                                ('source-clock.json', 'Original source clocks and unknown timestamps'),
                                ('reference-feasibility.json', 'Full reference geometry audit; not task success'),
                                ('reconstruction.json', 'Source body and authored static-support assumptions')]:
                artifact(name, label)
        if scene.get('profile') == 'shared-world-v6':
            for name, label in [('exercise-task.json', 'Original camera-body task and unchanged gates'),
                                ('reference-feasibility.json', 'Sampled geometry checks; not force feasibility'),
                                ('ground-clearance.json', 'Body–ground gaps; not observed support'),
                                ('two-view-body.json', 'Camera-based reconstruction assumptions and image errors'),
                                ('two-view-triangulation.json', 'Available and missing multi-view joint estimates')]:
                artifact(name, label)
        if hypothesis:
            for name, label in [('input-config.json', 'Authored shape configuration'),
                                ('geometry-hypothesis.json', 'Geometry alternative and limitations'),
                                ('previous-geometry.json', 'Previous geometry and provenance'),
                                ('initial-geometry-check.json', 'Initial-clearance check'),
                                ('reconstruction.json', 'Current reconstruction scope')]:
                if artifact(name, label) is None:
                    raise ValueError('Authored geometry hypothesis is missing a declared inspection artifact')
        stages = []
        states = ['available' if video else 'unavailable', 'available' if documents or 'evidence.json' in files or 'observations.json' in files else 'unavailable',
                  'initial_hypothesis' if geometry['status'] == 'initial_hypothesis' else 'model_only', 'simulation', 'local_recording']
        for (key, title, panel), state in zip(_STAGES, states):
            stages.append({'id': key, 'title': title, 'panel': panel, 'state': state})
        contract = read('exercise-lab.json', 128 * 1024)
        policy = read('policy.json', 64 * 1024) if contract.get('learned_policy_available') is True else {}
        camera_admission = None
        if scene.get('profile') == 'shared-world-v6' and contract.get('learned_policy_available') is True:
            camera_admission = read('camera-policy-admission.json', 128 * 1024)
            artifact('camera-policy-admission.json', 'Saved research-policy admission; not physical validation')
        rigid_task, rigid_evaluation = None, None
        if (scene.get('profile') == 'spatial-rigid-v1' and scene.get('schema_version') == 1
                and isinstance(scene.get('task'), dict) and isinstance(scene.get('controlled_id'), str)
                and {'policy.json', 'rl-evaluation.json'} <= files.keys()):
            policy = read('policy.json', 64 * 1024)
            rigid_task = read('task.json', 64 * 1024) if 'task.json' in files else None
            rigid_evaluation = read('rl-evaluation.json', 1024 * 1024)
            for name, label in [('policy.json', 'Saved rigid controller'), ('task.json', 'Authored rigid task'),
                                ('rl-evaluation.json', 'Saved nominal and changed-start evaluation')]:
                artifact(name, label)
        result = {'schema_version': 1, 'status': 'available', 'profile': 'lab-inspection-v1',
                'lab_profile': profile, 'source': source, 'stages': stages, 'actors': actors,
                'tracking': {'reference_poses': reference_poses, 'pose_status_counts': pose_counts,
                             'scope': 'A construction observation is input retained to build the model; it may still be too uncertain for a usable pose. These counts do not verify identity, true visibility or accuracy.'},
                'timeline': timeline, 'fullspan_preparation': fullspan,
                'geometry': geometry, 'artifact_links': list(links.values()),
                'source_manifest_sha256': verified['manifest.json']['sha256'],
                'verified_metadata': list(verified.values()),
                'scope': 'Saved manifest-linked inspection only. Simulation, controller availability and recordings use the live lab contract.'}
        if len(json.dumps(result, allow_nan=False).encode('utf-8')) > MAX_BASE_SUMMARY_BYTES:
            raise ValueError('Inspection base summary exceeds 64 KiB')
        result['evidence_scorecard'] = build_lab_evidence(source=source, scene=scene, task=exercise_task,
            contract=contract, policy=policy, files=files, verified_metadata=result['verified_metadata'],
            rigid_task=rigid_task, rigid_evaluation=rigid_evaluation,
            camera_admission=camera_admission, policy_document=retained.get('policy.json'))
        return result
    finally:
        os.close(root_fd)


def load_inspection(run, *, profile):
    """Return at most 128 KiB of metadata and scorecard, or explicit unavailability.

    Run once in the lab executor for a completed, immutable run. This helper does
    not enable controllers or assert full media/source-binding verification.
    """
    try:
        result = _summary(run, _text(profile, 64))
        if len(json.dumps(result, allow_nan=False).encode('utf-8')) > MAX_SUMMARY_BYTES:
            raise ValueError('Inspection and evidence summary exceed 128 KiB')
        return result
    except (ValueError, OSError, KeyError, TypeError, RecursionError) as error:
        return {'status': 'unavailable', 'reason': str(error)[:300],
                'scope': 'No inspection completeness claim. Saved artifacts and simulation checks are unchanged.'}


def recording_session_name(name):
    if not isinstance(name, str) or not _SESSION.fullmatch(name):
        raise ValueError('Choose one completed recording from this lab')
    return name


def load_recording_inspection(root, name, *, scene_sha256, task_sha256):
    """Read one sealed trial off-thread; never replay physics or alter the active trial."""
    try:
        from weft.shared_exercise_diagnostics import inspect_recording_failure

        name = recording_session_name(name)
        if any(not isinstance(value, str) or not _SHA.fullmatch(value) for value in (scene_sha256, task_sha256)):
            raise ValueError('Current lab scene/task identity is unavailable')
        result = inspect_recording_failure(Path(root) / 'sessions' / name, max_seconds=10)
        if result.get('scene_sha256') != scene_sha256 or result.get('task_sha256') != task_sha256:
            raise ValueError('Saved trial belongs to a different scene or task')
        return {'status': 'available', 'session': name, 'timeline': result,
                'scope': 'Sealed saved trial only; separate from the current live state and independent evaluation.'}
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, RecursionError) as error:
        return {'status': 'unavailable', 'session': name if isinstance(name, str) else None,
                'reason': str(error)[:300], 'scope': 'No failure timing or task success is inferred.'}
