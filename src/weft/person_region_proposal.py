"""A fixed attention region retains all confidence-qualified pose hypotheses."""

from hashlib import sha256
import math

from weft.pose_identity_windows import _document, MAX_INPUT_BYTES

PROFILE = 'all-saved-pose-hypotheses-fixed-region-v1'
SCOPE = ('Inferred fixed attention region, not verified person identity, absence, equipment extent or physical geometry. '
         'Outside-region people and coverage between sampled frames remain unknown. Low-confidence estimates stay in the original evidence.')


def propose_person_region(histories):
    """Accept original evidence/raw byte pairs; never select a person or inference mode."""
    if (not isinstance(histories, list) or not 1 <= len(histories) <= 4
            or any(not isinstance(pair, (list, tuple)) or len(pair) != 2 for pair in histories)
            or any(not isinstance(raw, bytes) for pair in histories for raw in pair)
            or sum(len(raw) for pair in histories for raw in pair) > MAX_INPUT_BYTES):
        raise ValueError('Use one to four complete histories within the aggregate 16-MiB bound')
    points, summaries, source, decoder_times = [], [], None, None
    for evidence_raw, poses_raw in histories:
        evidence, raw = _document(evidence_raw), _document(poses_raw)
        key = tuple(evidence.get(k) for k in ('source_sha256', 'width', 'height', 'fps', 'frame_count', 'decoded_frames', 'inference_stride'))
        pin, width, height, fps, count, decoded, stride = key
        identity = evidence.get('pose_identity', {})
        if (not isinstance(pin, str) or len(pin) != 64 or any(c not in '0123456789abcdef' for c in pin)
                or any(type(v) is not int for v in (width, height, count, decoded, stride))
                or not 64 <= width <= 4096 or not 64 <= height <= 2160 or width*height > 8_847_360
                or type(fps) not in (int, float) or not math.isfinite(fps) or not 1 <= fps <= 120
                or not 12 <= count <= 500 or decoded != count or count/fps > 60 or not 1 <= stride <= count
                or evidence.get('profile') != 'exercise-pose-v1' or evidence.get('schema_version') != 1
                or not isinstance(identity, dict) or identity.get('mode') != 'raw' or identity.get('repaired_frames') != 0
                or identity.get('raw_artifact_sha256') != sha256(poses_raw).hexdigest()
                or any(k in evidence for k in ('person_region', 'source_clock', 'pose_reinference', 'pose_mode_recovery'))):
            raise ValueError('Original full-frame raw pose source and hash are required')
        if source is not None and key != source:
            raise ValueError('Histories must share exact source, clock and sampling inventory')
        source = key
        rows, model = raw.get('frames'), raw.get('model')
        if (not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows) or rows != evidence.get('frames')
                or not isinstance(model, dict) or model != evidence.get('model')
                or model.get('maximum_pose_hypotheses') != 2 or model.get('network_inference') is not False
                or model.get('running_mode') not in {'VIDEO', 'IMAGE'}
                or model.get('temporal_tracking') is not (model.get('running_mode') == 'VIDEO')
                or 'person_region' in model
                or [r.get('frame') for r in rows] != list(range(0, count, stride))):
            raise ValueError('All ordered source rows and both-hypothesis model provenance are required')
        times = [row.get('decoder_timestamp_s') for row in rows]
        if (any(t is not None and (type(t) not in (int, float) or not math.isfinite(t) or not 0 <= t <= 60) for t in times)
                or decoder_times is not None and times != decoder_times):
            raise ValueError('Observed decoder timestamps differ or are invalid')
        decoder_times = times
        summary = {'evidence_sha256': sha256(evidence_raw).hexdigest(), 'raw_pose_sha256': sha256(poses_raw).hexdigest(),
                   'model': model, 'rows': len(rows), 'hypotheses': 0, 'credible_points': 0, 'samples': []}
        for row in rows:
            n = row.get('pose_count')
            if (type(row.get('frame')) is not int or type(n) is not int or not 0 <= n <= 2
                    or row.get('id') != f"{pin}:frame:{row['frame']}"
                    or type(row.get('time_s')) not in (int, float) or not math.isfinite(row['time_s'])
                    or abs(row['time_s']-row['frame']/fps) > 1e-9):
                raise ValueError('Malformed or retimed original pose row')
            alternatives = [row] if n == 1 else row.get('poses', [])
            if (not isinstance(alternatives, list) or len(alternatives) != n
                    or n == 2 and (row.get('status') != 'multiple_people' or row.get('landmarks') != []
                                   or row.get('identity_status') != 'unresolved_multiple_pose_hypotheses')
                    or n == 0 and row.get('status') != 'no_pose'
                    or n == 1 and (row.get('status') not in {'tracked', 'tracked_partial', 'low_confidence', 'invalid_geometry', 'invalid_landmark_count'}
                                   or row.get('poses'))):
                raise ValueError('All competing alternatives must remain explicit')
            sample = {'frame': row['frame'], 'id': row['id'], 'status': row['status'], 'hypotheses': []}
            for index, pose in enumerate(alternatives):
                landmarks = pose.get('landmarks') if isinstance(pose, dict) else None
                if not isinstance(landmarks, list) or len(landmarks) != 33:
                    raise ValueError('Every hypothesis must retain all 33 landmarks')
                indices = []
                for j, point in enumerate(landmarks):
                    values = [point.get(k) for k in ('x', 'y', 'visibility', 'presence')] if isinstance(point, dict) else []
                    if (len(values) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                            or any(not 0 <= v <= 1 for v in values[2:])):
                        raise ValueError('Malformed original landmark')
                    if min(values[2:]) >= .5:
                        pixel = (values[0]*width, values[1]*height)
                        if not all(math.isfinite(v) for v in pixel):
                            raise ValueError('Landmark pixel coordinates overflow the source domain')
                        points.append(pixel); indices.append(j)
                sample['hypotheses'].append({'index': index, 'credible_landmark_indices': indices})
                summary['credible_points'] += len(indices)
            summary['hypotheses'] += n; summary['samples'].append(sample)
        summaries.append(summary)
    if not points:
        raise ValueError('No confidence-qualified source points')
    x0, y0 = min(p[0] for p in points), min(p[1] for p in points)
    x1, y1 = max(p[0] for p in points), max(p[1] for p in points)
    padding = max(24., .25*max(x1-x0, y1-y0))
    if not all(math.isfinite(v) for v in (padding, x0-padding, y0-padding, x1+padding, y1+padding)):
        raise ValueError('Landmark envelope overflows the source domain')
    region = [max(0, math.floor(x0-padding)), max(0, math.floor(y0-padding)),
              min(width, math.ceil(x1+padding)), min(height, math.ceil(y1+padding))]
    if region[2]-region[0] < 64 or region[3]-region[1] < 64:
        raise ValueError('Envelope is below the ordinary fixed-region minimum')
    return {'profile': PROFILE, 'source_sha256': pin, 'width': width, 'height': height, 'region_xyxy': region,
            'credible_bounds_xyxy': [x0, y0, x1, y1], 'padding_pixels': padding, 'minimum_confidence': .5,
            'padding_rule': 'max(24px,25%longestextent), outward rounding and source-frame clipping',
            'histories': summaries, 'source_outside_frame_points': sum(not (0 <= x < width and 0 <= y < height) for x,y in points),
            'inference_executed': False, 'simulation_ready': False, 'native_clock_verified': False,
            'source_video_verified': False, 'scope': SCOPE}
