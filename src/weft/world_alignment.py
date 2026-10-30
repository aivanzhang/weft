"""Explicit coordinate/clock hypotheses; conversion is not physical calibration.

Frames name right-handed bases. Positions use s*R*p+t, clocks r*(time-origin)+target.
Mass, contacts, actuator laws, reward tolerances and engine migration are not converted.
"""

from dataclasses import dataclass
from collections.abc import Mapping
import json
import math
import re

import numpy as np


PROFILE = 'world-alignment-v1'
MAX_BYTES = 16_384
RIGHT_HANDED_AXES = frozenset({'right_handed_xyz', 'x_right_y_away_z_up',
    'x_right_y_depth_z_up', 'x_right_y_down_z_forward'})


def _fields(value, names, label):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(f'{label} fields differ from the alignment contract')


def _text(value, label, limit=160):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f'{label} must be bounded nonempty text')


def _number(value, label, low=-1e6, high=1e6):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{label} must be finite in [{low}, {high}]')
    return float(value)


def _vector(value, label, length=3):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f'{label} needs {length} coordinates')
    return np.array([_number(v, label) for v in value])


def _quaternion(value):
    q = _vector(value, 'rotation_wxyz', 4)
    if abs(float(np.linalg.norm(q))-1.) > 1e-10:
        raise ValueError('rotation_wxyz must be unit length; no fitted rotation or reflection')
    return q / np.linalg.norm(q)


def _multiply(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return np.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
        aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])


def _matrix(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _frame(value):
    _fields(value, ('id', 'length_unit', 'axes'), 'frame')
    for name in value:
        _text(value[name], name)
    if value['axes'] not in RIGHT_HANDED_AXES:
        raise ValueError('Unsupported axes; both frames must declare a supported right-handed basis')


def _registry(value):
    if value is None:return {}
    if not isinstance(value, Mapping) or len(value) > 8:
        raise ValueError('Evidence registry must map at most eight document hashes to complete IDs')
    result = {}
    for sha, identifiers in value.items():
        if not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{64}', sha):
            raise ValueError('Evidence registry needs exact document hashes')
        if not isinstance(identifiers, (list, tuple, set, frozenset)) or len(identifiers) > 40_000:
            raise ValueError('Evidence IDs must be a bounded collection, not a string or mapping')
        for identifier in identifiers:
            _text(identifier, 'evidence_id', 240)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError('Evidence registry contains duplicate IDs')
        result[sha] = frozenset(identifiers)
    return result


def _claim(claim, value, evidence_ids_by_document):
    _fields(claim, ('status', 'method', 'uncertainty', 'evidence_refs'), 'claim')
    if claim['status'] not in {'observed', 'inferred', 'assumed', 'unknown'}:
        raise ValueError('Unknown evidence status')
    if (claim['status'] == 'unknown') != (value is None):
        raise ValueError('Unknown alignment must remain null; numeric alignment needs an explicit claim')
    _text(claim['method'], 'method', 1000); _text(claim['uncertainty'], 'uncertainty', 2000)
    refs = claim['evidence_refs']
    if not isinstance(refs, list) or len(refs) > 64:
        raise ValueError('Evidence references must be a bounded list')
    if claim['status'] in {'observed', 'inferred'} and not refs:
        raise ValueError('Observed/inferred alignment needs qualified evidence')
    seen = set()
    for ref in refs:
        _fields(ref, ('document_sha256', 'evidence_id'), 'evidence reference')
        sha, identifier = ref['document_sha256'], ref['evidence_id']
        if not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{64}', sha):
            raise ValueError('Evidence needs an exact observation-document SHA256')
        _text(identifier, 'evidence_id', 240)
        if (sha, identifier) in seen or identifier not in evidence_ids_by_document.get(sha, ()):
            raise ValueError('Duplicate, unavailable or excluded evidence reference')
        seen.add((sha, identifier))


@dataclass(frozen=True)
class WorldAlignment:
    """Immutable validated hypothesis; each inspection or mapped state is a fresh value."""

    _encoded: bytes
    _from_frame: tuple
    _to_frame: tuple
    _spatial: tuple | None
    _clock: tuple | None
    _rotation: tuple | None

    def document(self):
        return json.loads(self._encoded)

    @property
    def ready(self):
        return self._spatial is not None and self._clock is not None

    def map_state(self, *, from_frame, to_frame, time_s, position, velocity=None,
                  angular_velocity=None, acceleration=None, orientation_wxyz=None):
        if from_frame != dict(self._from_frame) or to_frame != dict(self._to_frame):
            raise ValueError('Coordinate direction, units or axes differ from the declared frames')
        if not self.ready:
            raise ValueError('Unknown coordinate/clock alignment cannot transform a state')
        scale, q, translation = self._spatial
        rate, source_origin, target_origin = self._clock
        rotation = np.asarray(self._rotation); source_time = _number(time_s, 'source time', -3600, 3600)
        result = {'source_time_s': source_time, 'time_s': rate*(source_time-source_origin)+target_origin,
            'frame': dict(to_frame), 'position': (scale*rotation @ _vector(position, 'position')+translation).tolist(),
            'physical_validation': 'unverified', 'scope': 'coordinate_conversion_only'}
        for name, vector, factor in (('velocity', velocity, scale/rate),
                ('acceleration', acceleration, scale/rate**2),
                ('angular_velocity', angular_velocity, 1/rate)):
            if vector is not None:
                result[name] = (factor*rotation @ _vector(vector, name)).tolist()
        if orientation_wxyz is not None:
            result['orientation_wxyz'] = _multiply(q, _quaternion(orientation_wxyz)).tolist()
        return result


def load_world_alignment(document, *, evidence_ids_by_document=None):
    """Validate a proposal using only the caller's construction-authorized evidence IDs."""
    try:
        encoded = json.dumps(document, allow_nan=False, sort_keys=True, separators=(',', ':')).encode()
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('Alignment must be finite JSON') from error
    if len(encoded) > MAX_BYTES:
        raise ValueError('Alignment exceeds its 16 KiB envelope')
    _fields(document, ('schema_version', 'profile', 'from_frame', 'to_frame', 'spatial', 'clock', 'claims'), 'alignment')
    if type(document['schema_version']) is not int or document['schema_version'] != 1 or document['profile'] != PROFILE:
        raise ValueError('Unsupported world alignment profile/version')
    _frame(document['from_frame']); _frame(document['to_frame'])
    if document['from_frame']['id'] == document['to_frame']['id']:
        raise ValueError('Distinct frame IDs are required for an explicit source-to-target mapping')
    _fields(document['claims'], ('spatial', 'clock'), 'claims')
    registry = _registry(evidence_ids_by_document)
    for name in ('spatial', 'clock'):
        _claim(document['claims'][name], document[name], registry)
    spatial = clock = None
    if document['spatial'] is not None:
        value = document['spatial']
        _fields(value, ('scale', 'rotation_wxyz', 'translation'), 'spatial')
        spatial = (_number(value['scale'], 'scale', 1e-6, 1e6), tuple(_quaternion(value['rotation_wxyz'])),
            tuple(_vector(value['translation'], 'translation')))
    if document['clock'] is not None:
        value = document['clock']
        _fields(value, ('rate', 'source_origin_s', 'target_origin_s'), 'clock')
        clock = (_number(value['rate'], 'clock rate', 1e-6, 1e6),
            _number(value['source_origin_s'], 'source clock origin', -3600, 3600),
            _number(value['target_origin_s'], 'target clock origin', -3600, 3600))
    rotation = tuple(tuple(row) for row in _matrix(spatial[1])) if spatial else None
    return WorldAlignment(encoded, tuple(document['from_frame'].items()), tuple(document['to_frame'].items()),
        spatial, clock, rotation)
