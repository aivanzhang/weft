"""Model-free validation of explicit, source-pinned equipment choices."""

from copy import deepcopy
import math
import re


CONFIG_PROFILE = 'source-equipment-proposal-config-v1'
MAX_CONFIG_BYTES = 64 * 1024
MAX_EQUIPMENT = 12


class EquipmentProposalError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


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
    if config['geometry_prior'] != 'floor_box_depth_equals_width':
        raise EquipmentProposalError('unsupported_geometry_prior', 'Only unchanged-floor solid boxes with depth equal to width are supported')
    objects = config['equipment']
    if not isinstance(objects, list) or not 1 <= len(objects) <= MAX_EQUIPMENT:
        raise EquipmentProposalError('invalid_equipment_selection', 'Select one to twelve equipment observation entities')
    ids = set()
    for obj in objects:
        if not isinstance(obj, dict) or set(obj) != {'object_id', 'source_frame'}:
            raise EquipmentProposalError('invalid_equipment_selection', 'Each equipment selection requires object_id and source_frame only')
        name, frame = obj['object_id'], obj['source_frame']
        if (not isinstance(name, str) or not 1 <= len(name) <= 80 or name in ids or name == 'human' or
                type(frame) is not int or not 0 <= frame < 1800):
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
