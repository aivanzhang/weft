"""Model-free validation of explicit, source-pinned equipment choices."""

from copy import deepcopy
import math
import re


CONFIG_PROFILE = 'source-equipment-proposal-config-v1'
MAX_CONFIG_BYTES = 64 * 1024
MAX_EQUIPMENT = 12
DEPTH_RATIO_PRIOR = 'floor_box_explicit_depth_ratio'
# These are consequences of the constructor's unchanged half-extent limits.
MIN_DEPTH_TO_WIDTH_RATIO = .005 / .6
MAX_DEPTH_TO_WIDTH_RATIO = .6 / .005
MAX_DEPTH_PROVENANCE_CHARS = 2000
CONSENSUS_METHOD = 'construction_box_consensus_v1'


class EquipmentProposalError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def validate_frame_selection(value):
    if not isinstance(value, dict) or value != {'method': CONSENSUS_METHOD}:
        raise EquipmentProposalError('invalid_frame_selection', 'Use exactly the explicit construction_box_consensus_v1 frame-selection method')
    return deepcopy(value)


def validate_depth_prior(value):
    if not isinstance(value, dict) or set(value) != {'depth_to_width_ratio', 'provenance'}:
        raise EquipmentProposalError('invalid_depth_prior', 'An explicit depth prior requires depth_to_width_ratio and provenance only')
    ratio = value['depth_to_width_ratio']
    if (type(ratio) not in (int, float) or not MIN_DEPTH_TO_WIDTH_RATIO <= ratio <= MAX_DEPTH_TO_WIDTH_RATIO
            or not math.isfinite(ratio)):
        raise EquipmentProposalError('invalid_depth_ratio', 'Depth-to-width ratio must be finite and between 1/120 and 120')
    note = value['provenance']
    if not isinstance(note, str) or not note.strip() or len(note) > MAX_DEPTH_PROVENANCE_CHARS:
        raise EquipmentProposalError('invalid_depth_provenance', 'Explain the supplied depth assumption in 1–2000 characters; it is not verified measurement or contact')
    return deepcopy(value)


def validate_proposal_config(config):
    required = {'schema_version', 'profile', 'source_sha256', 'camera', 'geometry_prior', 'equipment', 'target_object_id'}
    if (not isinstance(config, dict) or not required <= set(config) or set(config) - required - {'frame_translation_xy'} or
            type(config['schema_version']) is not int or config['schema_version'] != 1 or config['profile'] != CONFIG_PROFILE):
        raise EquipmentProposalError('unsupported_config', 'Use exactly the versioned source-equipment proposal configuration')
    if not isinstance(config['source_sha256'], str) or not re.fullmatch('[0-9a-f]{64}', config['source_sha256']):
        raise EquipmentProposalError('invalid_source_identity', 'Configuration must pin the original video SHA256')
    camera = config['camera']
    if (not isinstance(camera, dict) or set(camera) != {'mode', 'distance_model_units'} or
            camera['mode'] != 'fixed_pelvis_perspective_prior'):
        raise EquipmentProposalError('unsupported_camera', 'Only an explicitly authored fixed-pelvis perspective prior is supported')
    distance = camera['distance_model_units']
    if type(distance) not in (int, float) or not math.isfinite(distance) or not .1 <= distance <= 50:
        raise EquipmentProposalError('invalid_camera_distance', 'Camera distance must be between 0.1 and 50 original body-model units')
    ratio_prior = config['geometry_prior'] == DEPTH_RATIO_PRIOR
    if not ratio_prior and config['geometry_prior'] != 'floor_box_depth_equals_width':
        raise EquipmentProposalError('unsupported_geometry_prior', 'Only unchanged-floor solid boxes with depth equal to width are supported')
    objects = config['equipment']
    if not isinstance(objects, list) or not 1 <= len(objects) <= MAX_EQUIPMENT:
        raise EquipmentProposalError('invalid_equipment_selection', 'Select one to twelve equipment observation entities')
    ids = set()
    for obj in objects:
        automatic = isinstance(obj, dict) and 'frame_selection' in obj
        selection_key = 'frame_selection' if automatic else 'source_frame'
        if not isinstance(obj, dict) or set(obj) != {'object_id', selection_key} | ({'depth_prior'} if ratio_prior else set()):
            message = ('Each explicit-depth selection requires object_id, source_frame and depth_prior only' if ratio_prior
                       else 'Each equipment selection requires object_id and source_frame only')
            raise EquipmentProposalError('invalid_equipment_selection', message)
        if ratio_prior:
            validate_depth_prior(obj['depth_prior'])
        if automatic:
            validate_frame_selection(obj['frame_selection'])
        name, frame = obj['object_id'], obj.get('source_frame')
        if (not isinstance(name, str) or not 1 <= len(name) <= 80 or name in ids or name == 'human' or
                not automatic and (type(frame) is not int or not 0 <= frame < 1800)):
            raise EquipmentProposalError('invalid_equipment_selection', 'Equipment identities must be unique and source frames explicit bounded integers')
        ids.add(name)
    if config['target_object_id'] not in ids:
        raise EquipmentProposalError('invalid_target', 'The authored task target must be one of the selected equipment IDs')
    translation = config.get('frame_translation_xy')
    if translation is not None:
        if (not isinstance(translation, list) or len(translation) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) or not 0. <= v <= 6. for v in translation)):
            raise ValueError('Explicit positive XY frame translation requires 2 finite bounded coordinates')
        if not any(v > 0 for v in translation):
            raise EquipmentProposalError('invalid_frame_translation', 'Omit frame translation or supply a nonzero positive XY vector')
    return deepcopy(config)
