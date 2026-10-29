"""Image-registration hypotheses retain original proposals; no physical calibration."""

from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np

from weft.io import canonical_hash, digest, local_time, write_json


PROFILE = 'stabilized-segmentation-v1'
SCOPE = ('Source-camera motion remains unresolved physically. Similarity registration maps model '
         'proposals into one reference image; it does not establish metric scale, common depth, '
         'physical identities, observed forces, or a fixed physical camera.')


def _validated_inputs(evidence, registration):
    from weft.camera_stabilization import registration_evidence_hash, validate_registration
    from weft.segmentation_scene import _normalize

    if not isinstance(evidence, dict) or evidence.get('profile') != 'segmentation-evidence-v1':
        raise ValueError('Stabilization requires original segmentation evidence, not a stacked correction')
    source, _, _, times, _, initial, _ = _normalize(evidence)
    report = validate_registration(registration)
    if report['ready'] is not True:
        raise ValueError('Complete source-supported registration is unavailable')
    if (report['source_sha256'] != source['submitted_video_sha256']
            or [report['width'], report['height']] != [source['width_px'], source['height_px']]
            or report['timestamps_s'] != times
            or report['reference_frame'] != max(initial.values())
            or report['evidence_sha256'] != registration_evidence_hash(evidence)):
        raise ValueError('Registration source, clock, exclusions or initialization differs from the evidence')
    return report


def stabilized_evidence(evidence, registration):
    report = _validated_inputs(evidence, registration)
    result = {'schema_version': 1, 'profile': PROFILE, 'physics_ready': False,
              'source_evidence': deepcopy(evidence), 'registration': report,
              'source_evidence_sha256': canonical_hash(evidence),
              'registration_sha256': canonical_hash(report), 'scope': SCOPE}
    # Reject invalid transformed geometry now, not after a candidate has been published.
    materialize_stabilized(result)
    return result


def materialize_stabilized(document):
    fields = {'schema_version', 'profile', 'physics_ready', 'source_evidence', 'registration',
              'source_evidence_sha256', 'registration_sha256', 'scope'}
    if (not isinstance(document, dict) or set(document) != fields or document['profile'] != PROFILE
            or type(document['schema_version']) is not int or document['schema_version'] != 1
            or document['physics_ready'] is not False or document['scope'] != SCOPE):
        raise ValueError('Invalid stabilized segmentation contract')
    evidence, registration = document['source_evidence'], document['registration']
    if (canonical_hash(evidence) != document['source_evidence_sha256']
            or canonical_hash(registration) != document['registration_sha256']):
        raise ValueError('Stabilization inputs changed')
    report = _validated_inputs(evidence, registration)
    source = deepcopy(evidence['source'])
    source.update(coordinate_system='registered_reference_image_pixels', reference_frame=report['reference_frame'],
                  camera_registration_sha256=document['registration_sha256'])
    rows = []
    for original in evidence['observations']:
        row = deepcopy(original)
        geometry = row['geometry']
        if geometry is not None:
            frame = report['frames'][row['frame']]
            if frame['status'] != 'registered':
                raise ValueError('Returned geometry has no supported transform; no missing-position substitution')
            matrix = np.asarray(frame['current_to_reference'], dtype=float)
            left, top, right, bottom = geometry['box_xyxy_halfopen']
            corners = np.array([[left, top, 1], [right, top, 1], [right, bottom, 1], [left, bottom, 1]])
            points = (corners @ matrix.T)[:, :2]
            lower, upper = points.min(axis=0), points.max(axis=0)
            if (not np.isfinite(points).all() or np.any(lower < 0)
                    or np.any(upper > [source['width_px'], source['height_px']]) or np.any(upper <= lower)):
                raise ValueError(f'Transformed geometry leaves reference canvas at frame {row["frame"]}; no clipping')
            row['geometry'] = {
                'box_xyxy_halfopen': [*lower.tolist(), *upper.tolist()],
                'box_xywh': [*lower.tolist(), *(upper-lower).tolist()],
                'center_px': ((lower+upper)/2).tolist(), 'status': 'inferred',
                'coordinate_system': 'registered_reference_image_pixels', 'reference_frame': report['reference_frame'],
                'original_box_xyxy_halfopen': deepcopy(geometry['box_xyxy_halfopen']),
                'transformed_corners_px': points.tolist(),
                'scope': 'Envelope of transformed model box; not decoded mask or measured collision shape'}
        rows.append(row)
    # This private view feeds the existing planar validator; the public wrapper keeps raw inputs.
    return {'schema_version': 1, 'profile': 'segmentation-evidence-v1', 'physics_ready': False,
            'source': source, 'time': deepcopy(evidence['time']), 'objects': deepcopy(evidence['objects']),
            'observations': rows, 'identity_groups': deepcopy(evidence['identity_groups'])}


def stabilize_tracking(evidence_run, video, out, *, reference_frame):
    from weft.benchmark import seal_suite
    from weft.camera_stabilization import register_video
    from weft.segmentation_evidence import _Bundle
    from weft.segmentation_scene import _normalize

    bundle = _Bundle(evidence_run, ('report.json', 'segmentation-evidence.json'))
    evidence = bundle.read('segmentation-evidence.json')
    source, _, _, _, _, initial, _ = _normalize(evidence)
    if type(reference_frame) is not int or reference_frame != max(initial.values()):
        raise ValueError('Reference frame must match the latest object initialization')
    video, out = Path(video).resolve(), Path(out).resolve()
    if out.is_relative_to(bundle.root):
        raise ValueError('Keep stabilization outside immutable tracking evidence')
    if not video.is_file() or video.stat().st_size > 37_000_000 or digest(video) != source['submitted_video_sha256']:
        raise ValueError('Use the exact bounded submitted video')
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(video, out/'source.mp4')
    shutil.copyfile(bundle.root/'segmentation-evidence.json', out/'original-evidence.json')
    shutil.copyfile(bundle.root/'manifest.json', out/'source-evidence-manifest.json')
    if digest(out/'original-evidence.json') != bundle.files['segmentation-evidence.json']:
        raise ValueError('Evidence changed during stabilization preparation')
    if digest(out/'source.mp4') != source['submitted_video_sha256']:
        raise ValueError('Video changed during stabilization preparation')
    implementation = out/'implementation'
    implementation.mkdir()
    paths = [Path(__file__), Path(__file__).with_name('camera_stabilization.py')]
    hashes = {path.name: digest(path) for path in paths}
    for path in paths:
        shutil.copyfile(path, implementation/path.name)
    report = {'created_at': local_time(), 'status': 'prepared', 'physics_ready': False,
              'physical_validation': 'unverified', 'new_api_calls': 0, 'new_cost_usd': 0,
              'input_verification': bundle.provenance(), 'source_sha256': source['submitted_video_sha256'],
              'implementation_hashes': hashes, 'scope': SCOPE}
    write_json(out/'pre-run.json', report)
    try:
        registration = register_video(out/'source.mp4', evidence, reference_frame=reference_frame)
        write_json(out/'registration.json', registration)
        if registration['ready']:
            wrapped = stabilized_evidence(evidence, registration)
            write_json(out/'stabilized-evidence.json', wrapped)
            report.update(status='registered_image_hypothesis', registration_sha256=canonical_hash(registration))
        else:
            report.update(status='registration_unavailable', reasons=registration['reasons'])
        if any(digest(path) != hashes[path.name] for path in paths):
            raise RuntimeError('Registration implementation changed during execution')
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, reason=str(error))
        raise
    finally:
        write_json(out/'report.json', report)
        seal_suite(out)
    return report
