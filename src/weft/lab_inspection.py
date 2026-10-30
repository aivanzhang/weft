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


MAX_SUMMARY_BYTES = 64 * 1024
MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_MANIFEST_FILES = 4096
MAX_ACTORS = 128
_SHA = re.compile(r'[0-9a-f]{64}')
_VIDEO = re.compile(r'source\.(mp4|webm|mov|m4v|avi|mkv)')
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


def _summary(run, profile):
    root_fd = os.open(Path(run), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    consumed = 0
    files = {}
    links = {}
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
            if name != 'manifest.json' and hashlib.sha256(raw).hexdigest() != files[name]:
                raise ValueError('Inspection metadata hash does not match the saved manifest')
            result = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_nonfinite, parse_float=_float)
            if not isinstance(result, dict):
                raise ValueError('Inspection metadata must be an object')
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
        assumptions = proposal.get('assumptions', [])
        if not isinstance(assumptions, list) or len(assumptions) > 32:
            raise ValueError('Invalid geometry assumption summary')
        geometry = {'status': 'initial_hypothesis' if proposal or previews else 'no_saved_comparison',
                    'panels': previews, 'assumptions': [_text(value, 1000) for value in assumptions],
                    'scope': 'Initial authored geometry hypotheses, not recovered shape, grasp, sustained support or physical accuracy.'}
        for name, label in [('source.json', 'Source metadata'), ('geometry-proposal.json', 'Geometry assumptions'),
                            ('geometry-study.json', 'Geometry comparison summary'), ('evidence.json', 'Frame measurements'),
                            ('observations.json', 'Frame observations'), ('coverage-summary.json', 'Tracking coverage'),
                            ('pose-quality.json', 'Pose-quality summary')]:
            artifact(name, label)
        stages = []
        states = ['available' if video else 'unavailable', 'available' if documents or 'evidence.json' in files or 'observations.json' in files else 'unavailable',
                  'initial_hypothesis' if geometry['status'] == 'initial_hypothesis' else 'model_only', 'simulation', 'local_recording']
        for (key, title, panel), state in zip(_STAGES, states):
            stages.append({'id': key, 'title': title, 'panel': panel, 'state': state})
        return {'schema_version': 1, 'status': 'available', 'profile': 'lab-inspection-v1',
                'lab_profile': profile, 'source': source, 'stages': stages, 'actors': actors,
                'tracking': {'reference_poses': reference_poses, 'pose_status_counts': pose_counts,
                             'scope': 'A construction observation is input retained to build the model; it may still be too uncertain for a usable pose. These counts do not verify identity, true visibility or accuracy.'},
                'geometry': geometry, 'artifact_links': list(links.values()),
                'scope': 'Saved manifest-linked inspection only. Simulation, controller availability and recordings use the live lab contract.'}
    finally:
        os.close(root_fd)


def load_inspection(run, *, profile):
    """Return at most 64 KiB of metadata, or an explicit unavailable result.

    Run once in the lab executor for a completed, immutable run. This helper does
    not enable controllers or assert full media/source-binding verification.
    """
    try:
        result = _summary(run, _text(profile, 64))
        if len(json.dumps(result, allow_nan=False).encode('utf-8')) > MAX_SUMMARY_BYTES:
            raise ValueError('Inspection summary exceeds 64 KiB')
        return result
    except (ValueError, OSError, KeyError, TypeError, RecursionError) as error:
        return {'status': 'unavailable', 'reason': str(error)[:300],
                'scope': 'No inspection completeness claim. Saved artifacts and simulation checks are unchanged.'}
