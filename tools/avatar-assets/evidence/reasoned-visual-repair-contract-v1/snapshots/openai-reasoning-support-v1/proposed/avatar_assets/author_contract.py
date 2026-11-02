"""Typed author transport; deterministic conversion never repairs or invents a design."""

from copy import deepcopy
import json

from avatar_assets.agent_runner import AUTHORED_BONE_PATTERN, _validate
from avatar_assets.safety import require

MAX_DESIGN_BYTES = 128 * 1024


def _object(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def _nullable(schema):
    return {'anyOf': [schema, {'type': 'null'}]}


def _array(items, maximum, minimum=0):
    return {'type': 'array', 'items': items, 'minItems': minimum, 'maxItems': maximum}


def _number(low, high, *, integer=False):
    return {'type': 'integer' if integer else 'number', 'minimum': low, 'maximum': high}


def _string(maximum=64, minimum=1):
    return {'type': 'string', 'minLength': minimum, 'maxLength': maximum}


def _enum(*values):
    return {'type': 'string', 'enum': list(values)}


def _vector(low, high, size=3):
    return _array(_number(low, high), size, size)


def _ref(name):
    return {'$ref': '#/$defs/' + name}


_SLUG = {**_string(), 'pattern': '^[a-z][a-z0-9_-]{0,63}$'}
# Existing imported skeleton names are data; only newly authored bone names have a restricted grammar.
_BONE = _string()
_BONES = _array(_BONE, 64, 1)
_AUTHORED_BONE = {**_string(), 'pattern': AUTHORED_BONE_PATTERN}
_AUTHORED_BONES = _array(_AUTHORED_BONE, 64, 1)
_LAYER = _object({'bones': _AUTHORED_BONES, 'min': _vector(-1, 2), 'max': _vector(-1, 2),
                  'feather': _number(0, .3), 'min_vertices': _number(1, 100000, integer=True)})
_REGION = _object({'center': _vector(-1, 3), 'radius': _vector(.001, 3)})
_DEFS = {
    'region': _REGION,
    'layer': _LAYER,
    'rig': _object({
        'space': _enum('normalized_bounds_blender_z_up'), 'height_meters': _number(.05, 10),
        'bones': _array(_object({'name': _AUTHORED_BONE, 'parent': _nullable(_AUTHORED_BONE),
                                  'head': _vector(-1, 2), 'tail': _vector(-1, 2)}), 64, 1),
        'base_bones': _AUTHORED_BONES, 'layers': _array(_ref('layer'), 32),
    }),
    'rotation': _object({'bone': _BONE, 'xyz': _vector(-60, 60)}),
    'translation': _object({'bone': _BONE, 'xyz': _vector(-2, 2)}),
    'key': _object({'second': _number(0, 10), 'rotations': _array(_ref('rotation'), 64),
                    'translations_m': _nullable(_array(_ref('translation'), 64))}),
    'action': _object({'id': _SLUG, 'keys': _array(_ref('key'), 32, 2),
                       'interpolation': _nullable(_enum('LINEAR', 'BEZIER'))}),
    'binding': _object({'state_id': _SLUG, 'roles': _array(_object({
        'id': _SLUG, 'root_bone': _BONE, 'bones': _BONES}), 4, 1)}),
    'part_rig': _object({'name': _SLUG, 'materials': _array(_string(256), 8, 1), 'root_bone': _AUTHORED_BONE,
                         'layers': _array(_ref('layer'), 32),
                         'min_vertices': _number(1, 100000, integer=True),
                         'max_vertices': _number(1, 100000, integer=True)}),
    'normal_continuity': _object({'name': _SLUG, 'materials': _array(_string(128), 64, 1),
        'reference_materials': _array(_string(128), 64, 1), 'max_distance_m': _number(0, .03),
        'min_vertices': _number(1, 10000, integer=True), 'max_vertices': _number(1, 10000, integer=True)}),
    'surface_profile': _object({'name': _SLUG, 'regions': _array(_ref('region'), 8),
        'brightness': _vector(0, 1.01, 2), 'max_chroma': _number(0, 1), 'roughness': _number(.04, 1),
        'metallic': _number(0, 1), 'normal_strength': _number(0, 1),
        'min_faces': _number(1, 100000, integer=True), 'max_faces': _number(1, 100000, integer=True)}),
    'surface_smoothing': _object({'iterations': _number(1, 20, integer=True), 'strength': _number(.01, .5),
        'max_displacement_ratio': _number(.001, .05), 'feather': _number(.05, 1),
        'exclusions': _array(_ref('region'), 16), 'method': _nullable(_enum('taubin', 'laplacian'))}),
    'weight_constraints': _object({'fallback_bone': _BONE, 'regions': _array(_object({
        'bone': _BONE, 'region': _ref('region')}), 64)}),
    'attachment': _object({'bone': _BONE, 'parent_bone': _BONE, 'anchor_from_bone': _BONE,
        'dominant_channel': _number(0, 2, integer=True), 'dominant_ratio': _number(1, 5),
        'min_vertices': _number(1, 30000, integer=True), 'max_vertices': _number(1, 30000, integer=True),
        'regions': _array(_ref('region'), 8, 1), 'expand_meters': _number(0, .05)}),
    'groom': _object({'parents': _number(16, 10000, integer=True), 'children': _number(0, 30, integer=True),
        'length': _number(.001, .04), 'radius': _number(.00001, .001), 'roughness': _number(0, .01),
        'seed': _number(0, 2**31-1, integer=True), 'direction': _nullable(_vector(-1, 1)),
        'comb': _nullable(_number(0, .95))}),
    'groom_flow': _object({'regions': _array(_object({'name': _SLUG, 'center': _vector(-1, 3),
        'radius': _vector(.001, 3), 'feather': _number(.05, 1), 'direction': _vector(-1, 1),
        'comb': _number(0, .95), 'length_scale': _number(.25, 2)}), 12, 1)}),
    'portable_settings': _object({'seed': _number(0, 2**31-5, integer=True), 'length': _number(.001, .04),
        'width': _number(.0003, .02), 'direction': _vector(-1, 1),
        'profile': _nullable(_enum('legacy', 'combed-v1', 'tuft-v1')),
        'sample_root_color': _nullable({'type': 'boolean'}),
        'root_placement': _nullable(_enum('skin-consistent-v1'))}),
    'portable_fur': _object({'settings': _ref('portable_settings'), 'bands': _array(_object({
        'brightness': _vector(0, 1.01, 2), 'color': _vector(0, 1, 4),
        'count': _number(1, 5000, integer=True)}), 4, 1)}),
    'undercoat': _object({'profile': _enum('short-pile-v1'),
        'resolution': {'type': 'integer', 'enum': [1024, 2048]}, 'seed': _number(0, 2**31-1, integer=True),
        'pitch_m': _number(.002, .02), 'strength': _number(0, .5), 'roughness': _number(.5, 1),
        'min_faces': _number(1, 100000, integer=True), 'max_faces': _number(1, 100000, integer=True),
        'uv_policy': _nullable(_enum('material-adaptive-v1'))}),
    'color_correction': _object({'name': _SLUG, 'regions': _array(_ref('region'), 4, 1),
        'dominant_channel': _number(0, 2, integer=True), 'dominant_ratio': _number(1.05, 5),
        'color_srgb': _vector(0, 1), 'roughness': _number(.04, 1),
        'min_faces': _number(1, 100000, integer=True), 'max_faces': _number(1, 100000, integer=True)}),
    # The native backend still rejects this experimental profile; typed transport does not enable it.
    'fringe': _object({'profile': _enum('edge-fringe-v1'), 'settings': _object({
        'seed': _number(0, 2**31-1, integer=True), 'min_fins': _number(1, 4800, integer=True),
        'max_fins': _number(1, 4800, integer=True), 'length_m': _number(.001, .04),
        'min_spacing_m': _number(.0005, .05), 'min_edge_m': _number(.0085, .020),
        'max_edge_m': _number(.0085, .020), 'comb': _number(0, .35)}),
        'design_views': _array(_vector(-1, 1), 16, 1), 'excluded_materials': _array(_string(128), 32)}),
}
_DESIGN_PROPERTIES = {
    'authored_rig': _ref('rig'), 'authored_actions': _array(_ref('action'), 8, 1),
    'action_bindings': _array(_ref('binding'), 8), 'part_rigs': _array(_ref('part_rig'), 4, 1),
    'normal_continuity': _array(_ref('normal_continuity'), 4),
    'smoothing': _object({'iterations': _number(1, 30, integer=True), 'strength': _number(.01, .8)}),
    'weight_constraints': _ref('weight_constraints'), 'attachment': _ref('attachment'),
    'groom': _ref('groom'), 'groom_exclusions': _array(_ref('region'), 16),
    'groom_excluded_parts': _array(_SLUG, 4, 1), 'fur_max_chroma': _number(0, 1), 'fur_clump': _number(0, 1),
    'portable_fur': _ref('portable_fur'), 'portable_fringe': _ref('fringe'),
    'camera_view': _enum('front', 'three_quarter', 'three_quarter_left'), 'camera_fit': _enum('rest', 'motion'),
    'surface_profiles': _array(_ref('surface_profile'), 8, 1), 'surface_smoothing': _ref('surface_smoothing'),
    'groom_shader': _object({'roughness': _number(.04, 1), 'radial_roughness': _number(.04, 1)}),
    'groom_flow': _ref('groom_flow'), 'color_corrections': _array(_ref('color_correction'), 4, 1),
    'pose_resolution': _number(128, 1024, integer=True), 'review_views': {'type': 'boolean'},
    'fur_undercoat': _ref('undercoat'),
    'material_fur': _object({'version': {'type': 'integer', 'const': 1}, 'materials': _array(_string(128), 32, 1)}),
}
_DEFS['design'] = _object({name: _nullable(schema) for name, schema in _DESIGN_PROPERTIES.items()})
AUTHOR_SCHEMA_V2 = _object({'schema_version': {'type': 'integer', 'const': 2},
    'status': _enum('ready', 'unsupported'), 'reason': _string(4000, 0), 'design': _nullable(_ref('design'))})
AUTHOR_SCHEMA_V2['$defs'] = _DEFS
AUTHOR_FIELDS_V2 = frozenset(_DESIGN_PROPERTIES)
# New API jobs enforce policy v2 without changing archived transport or adding fur to other subjects.
AUTHOR_SCHEMA_V2_SKIN_CONSISTENT = deepcopy(AUTHOR_SCHEMA_V2)
AUTHOR_SCHEMA_V2_SKIN_CONSISTENT['$defs']['portable_settings']['properties'].update({
    'profile': {'type': 'string', 'const': 'tuft-v1'},
    'root_placement': {'type': 'string', 'const': 'skin-consistent-v1'},
    'sample_root_color': {'type': 'boolean', 'const': True},
})

# Only these declared optional fields use null as omission; a root bone's null parent is retained.
_OPTIONAL_NESTED = {
    ('groom',): ('direction', 'comb'),
    ('surface_smoothing',): ('method',),
    ('portable_fur', 'settings'): ('profile', 'sample_root_color', 'root_placement'),
    ('fur_undercoat',): ('uv_policy',),
}


def _named_rows(rows, field):
    result = {}
    for row in rows:
        name = row['bone']
        require(name not in result, 'Duplicate bone in typed '+field)
        result[name] = row['region' if field == 'weight regions' else 'xyz']
    return result


def decode_typed_author(response):
    """Return detached canonical fields; existing finishing validators still own semantics."""
    _validate(response, AUTHOR_SCHEMA_V2)
    try:
        size = 0
        for chunk in json.JSONEncoder(allow_nan=False, ensure_ascii=False, separators=(',', ':')).iterencode(response):
            size += len(chunk.encode())
            require(size <= MAX_DESIGN_BYTES, 'Typed author response exceeds128KiB')
    except (TypeError, RecursionError) as error:
        raise ValueError('Invalid typed author response') from error
    if response['status'] == 'unsupported':
        require(response['reason'].strip() and response['design'] is None,
                'Unsupported typed author needs a reason and no design')
        return None
    require(response['design'] is not None, 'Ready typed author needs a design')
    design = {key: deepcopy(value) for key, value in response['design'].items() if value is not None}
    require(design.get('authored_actions'), 'Ready typed author needs real authored actions')
    for path, optional in _OPTIONAL_NESTED.items():
        node = design
        for part in path:
            node = node.get(part) if node is not None else None
        if node is not None:
            for key in optional:
                if node.get(key) is None:
                    node.pop(key, None)
    for action in design['authored_actions']:
        if action['interpolation'] is None:
            del action['interpolation']
        for key in action['keys']:
            key['rotations'] = _named_rows(key['rotations'], 'rotations')
            if key['translations_m'] is None:
                del key['translations_m']
            else:
                key['translations_m'] = _named_rows(key['translations_m'], 'translations')
    if 'weight_constraints' in design:
        design['weight_constraints']['regions'] = _named_rows(design['weight_constraints']['regions'], 'weight regions')
    return design


TYPED_AUTHOR_GUIDE = """
TRANSPORT CONTRACT v2: return schema_version:2, status, reason, and design as a JSON OBJECT, never a JSON-encoded string.
Return every schema property. For an unused optional design field use null; for unsupported return design:null and explain why.
Only declared optional nulls are omitted by the trusted adapter. A root bone's parent:null remains meaningful and is preserved.
In every motion key, rotations is [{bone,xyz:[x,y,z]}] and translations_m is null or [{bone,xyz:[x,y,z]}].
Use each bone at most once per list; angles remain degrees and translations remain meters. Empty rotations means no keyed rotation.
weight_constraints.regions is [{bone,region:{center:[x,y,z],radius:[x,y,z]}}], with unique bone names.
All other fields retain their documented meaning. Do not put motions, grooming or camera settings inside authored_rig.
Typed validity is not rig, motion, fur or appearance acceptance; all independent finishing and quality checks still apply.
"""
