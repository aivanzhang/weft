"""Carry a supplied camera body's fixed coordinates into conditional surface inputs."""

from dataclasses import dataclass
from hashlib import sha256
import json
import time

from weft.camera_pose import validate_camera
from weft.camera_shared_contract import build_camera_coupling
from weft.io import canonical_hash
from weft.source_surface_anchors import _document, _fields
from weft.source_surface_envelopes import SourceSurfaceEnvelopes
from weft.source_surface_interval_construction import (
    BINDING_PROFILE, MAX_BINDING_BYTES, MAX_BODY_BYTES, MAX_CHANNELS, MAX_FRAMES,
    MAX_REFERENCES, _encode, prepare_surface_interval_objective,
)
from weft.world_alignment import MAX_BYTES as MAX_DECLARATION_BYTES, load_world_alignment


PROFILE = 'camera-surface-binding-v1'
MAX_SECONDS, MAX_CAMERA_BYTES, MAX_METADATA_BYTES = 10., 4 << 20, 256 << 10
MAX_ENVELOPE_CACHE_BYTES, MAX_ENVELOPE_ROW_BYTES = 8 << 20, 128 << 10


def _require(value, message):
    if not value:
        raise ValueError('Camera surface binding: '+message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Camera surface binding cooperative deadline exceeded')


def _optional(raw, limit, deadline):
    if raw is None:
        return None
    _require(type(raw) is bytes and 0 < len(raw) <= limit, 'bounded exact input bytes required')
    return _document(raw, deadline)


def _selections(entity, correspondences, channels, deadline):
    if entity is not None:
        _fields(entity, ('document_sha256','object_id'), 'explicit human entity')
        _require(type(entity['document_sha256']) is str and len(entity['document_sha256']) == 64
                 and type(entity['object_id']) is str and 0 < len(entity['object_id']) <= 160,
                 'bounded explicit human entity required')
    if correspondences is not None:
        _require(type(correspondences) is list and len(correspondences) <= MAX_CHANNELS,
                 'at most twelve explicit endpoint correspondences')
        for row in correspondences:
            _check(deadline)
            _fields(row, ('channel_id','body_point','status','method','uncertainty'), 'endpoint correspondence')
            _require(all(type(row[key]) is str and 0 < len(row[key]) <= limit for key,limit in
                         (('channel_id',80),('status',32),('method',2000),('uncertainty',2000)))
                     and (row['body_point'] is None or type(row['body_point']) is str and len(row['body_point']) <= 80),
                     'bounded explicit correspondence fields required')
        _require(len({row['channel_id'] for row in correspondences}) == len(correspondences)
                 and {row['channel_id'] for row in correspondences} == {row['id'] for row in channels},
                 'each original channel needs its explicit correspondence or unknown row')


def _world_declaration(raw, metadata, camera_hash, deadline):
    declaration = _optional(raw, MAX_DECLARATION_BYTES, deadline)
    if declaration is None:
        return None, 'missing_explicit_calibration_world_declaration'
    # This adapter carries priors; measurement claims need their own evidence path.
    _fields(declaration, ('schema_version','profile','from_frame','to_frame','spatial','clock','claims'), 'calibration world declaration')
    _fields(declaration['claims'], ('spatial','clock'), 'calibration world claims')
    _require(all(type(claim) is dict and claim.get('status') in ('assumed','unknown')
                 and claim.get('evidence_refs') == [] for claim in declaration['claims'].values()),
             'conditional assumed/unknown world claims without invented measurement receipts required')
    alignment = load_world_alignment(declaration)
    expected = {'id':'calibration-world:'+camera_hash,'length_unit':'m','axes':'x_right_y_away_z_up'}
    if declaration['from_frame'] != expected or declaration['to_frame'] != metadata['world_frame']:
        return declaration, 'calibration_or_surface_world_identity_differs'
    if not alignment.ready:
        return declaration, 'unknown_calibration_world_alignment'
    spatial, clock = declaration['spatial'], declaration['clock']
    if (spatial['scale'] != 1. or spatial['rotation_wxyz'] not in ([1.,0.,0.,0.],[-1.,0.,0.,0.])
            or spatial['translation'] != [0.,0.,0.]):
        return declaration, 'nonidentity_calibration_world_mapping_unsupported'
    if clock != {'rate':1.,'source_origin_s':0.,'target_origin_s':0.}:
        return declaration, 'nonidentity_original_source_clock_unsupported'
    _check(deadline)
    return declaration, None


@dataclass(frozen=True)
class CameraSurfaceBinding:
    """Exact inputs and one shared prepared consumer; queries never repeat qualification."""

    envelopes: SourceSurfaceEnvelopes
    body_document: bytes | None
    camera_document: bytes | None
    calibration_world_declaration: bytes | None
    objective: object | None
    _binding: bytes | None
    _metadata: bytes

    def binding_document(self):
        return self._binding

    def describe(self):
        return json.loads(self._metadata)

    def frame(self, index):
        if self.objective is None:
            return {'source_frame':index,'source_envelopes':self.envelopes.frame(index),'body_join':None}
        joined = self.objective.frame(index)
        return {'source_frame':index,'source_envelopes':joined.pop('source_envelopes'),'body_join':joined}


def prepare_camera_surface_binding(envelopes, *, body_document, camera_document,
        calibration_world_declaration, human_entity_ref, channel_correspondences):
    """Build the existing binding from explicit inputs, never ground/contact evidence.

    The declaration uses world-alignment-v1 and must explicitly preserve the
    supplied calibration world and original clock. Null inputs stay unavailable.
    No source/body points, intervals, camera parameters or physics are evaluated.
    """
    deadline = time.monotonic()+MAX_SECONDS
    for raw, limit in ((body_document,MAX_BODY_BYTES),(camera_document,MAX_CAMERA_BYTES),
                       (calibration_world_declaration,MAX_DECLARATION_BYTES)):
        _require(raw is None or type(raw) is bytes and 0 < len(raw) <= limit, 'bounded exact input bytes required')
    _require(type(envelopes) is SourceSurfaceEnvelopes and 1 <= envelopes.frame_count <= MAX_FRAMES,
             'bounded prepared source envelopes required')
    _require(type(envelopes._metadata) is bytes and len(envelopes._metadata) <= MAX_ENVELOPE_ROW_BYTES,
             'bounded envelope metadata required')
    envelope_bytes = len(envelopes._metadata)
    for row in envelopes.iter_frame_bytes():
        _check(deadline)
        _require(type(row) is bytes and 0 < len(row) <= MAX_ENVELOPE_ROW_BYTES, 'bounded complete envelope rows required')
        envelope_bytes += len(row)
        _require(envelope_bytes <= MAX_ENVELOPE_CACHE_BYTES, 'complete shared envelope cache exceeds eight MiB')
    metadata = envelopes.describe()
    _require(metadata.get('profile') == 'source-surface-envelopes-v1' and metadata.get('schema_version') == 1
             and metadata.get('counts',{}).get('retained_source_frames') == envelopes.frame_count
             and type(metadata.get('channels')) is list and len(metadata['channels']) <= MAX_CHANNELS,
             'existing envelope profile or complete source slots differ')
    body = _optional(body_document, MAX_BODY_BYTES, deadline)
    camera = _optional(camera_document, MAX_CAMERA_BYTES, deadline)
    _selections(human_entity_ref, channel_correspondences, metadata['channels'], deadline)
    body_pin = None if body is None else sha256(body_document).hexdigest()
    camera_pin = None if camera is None else sha256(camera_document).hexdigest()
    camera_model_pin = None if camera is None else canonical_hash(camera)
    reasons = []
    coupling = None
    if body is None:
        reasons.append('missing_exact_original_body')
    else:
        coupling = build_camera_coupling(body)
        _require(body['source_sha256'] == metadata['source_sha256'], 'body and envelope source identities differ')
    coupling_pin = None if coupling is None else canonical_hash(coupling)
    if camera is None:
        reasons.append('missing_exact_original_camera')
    else:
        _require(type(envelopes.additional_inputs) is tuple and len(envelopes.additional_inputs) == 3
                 and envelopes.additional_inputs[0] == camera_document
                 and metadata['bindings']['camera_sha256'] == camera_pin,
                 'camera bytes must exactly match the prepared envelope camera')
        _require(camera['source_sha256'] == metadata['source_sha256'], 'camera and envelope source identities differ')
    if coupling is not None and camera is not None:
        _require(camera_model_pin == body['camera_reference']['calibration_sha256'],
                 'camera differs from the original canonical calibration')
        validate_camera(camera, {'source_sha256':metadata['source_sha256'], 'width':camera['width'], 'height':camera['height'],
            'frame_count':metadata['counts']['metadata_frames'], 'fps':coupling['clock']['source_fps']})
        _require(camera['frame_count'] == coupling['clock']['total_source_frames'], 'camera/body original frame denominators differ')
        for row in body['camera_reference']['root_refinement']['roots']:
            _check(deadline)
            _require(camera['frames'][row['frame']]['time_s'] == row['time_s'], 'exact body/camera original source clock differs')
    declaration, world_reason = (None,'missing_exact_original_camera') if camera_pin is None else _world_declaration(
        calibration_world_declaration, metadata, camera_pin, deadline)
    if camera_pin is None:
        _optional(calibration_world_declaration, MAX_DECLARATION_BYTES, deadline)
    elif world_reason:
        reasons.append(world_reason)
    if human_entity_ref is None:
        reasons.append('missing_explicit_human_entity')
    if channel_correspondences is None:
        reasons.append('missing_explicit_endpoint_correspondences')
    mapping = None
    if coupling is not None and camera_pin is not None and world_reason is None:
        declaration_pin = sha256(calibration_world_declaration).hexdigest()
        claim = {'status':'assumed',
            'method':'Exact inverse of supplied camera-body coupling '+coupling_pin+'; explicit calibration-world declaration '+declaration_pin,
            'uncertainty':'Supplied camera metre scale, ground and body shape remain priors; no independently measured anatomy, calibration or contact',
            'evidence_refs':[]}
        mapping = {'schema_version':1,'profile':'world-alignment-v1',
            'from_frame':{'id':'body:'+coupling['body_sha256'],'length_unit':'human_model_unit','axes':'x_right_y_away_z_up'},
            'to_frame':declaration['to_frame'],
            'spatial':{'scale':1.,'rotation_wxyz':[1.,0.,0.,0.],
                       'translation':[-value for value in coupling['world_to_local']['translation']]},
            'clock':declaration['clock'],'claims':{'spatial':claim,'clock':dict(claim)}}
        load_world_alignment(mapping)
    binding = None
    if body is not None and human_entity_ref is not None and channel_correspondences is not None:
        binding = _encode({'schema_version':1,'profile':BINDING_PROFILE,'source_sha256':metadata['source_sha256'],
            'body_sha256':body_pin,'envelope_bindings':metadata['bindings'],'human_entity_ref':human_entity_ref,
            'body_to_world':mapping,'channels':channel_correspondences},MAX_BINDING_BYTES,deadline)
    objective = None
    if body is not None:
        objective = prepare_surface_interval_objective(envelopes,body_bytes=body_document,binding_bytes=binding,include_image_admission=True)
    _check(deadline)
    joined = None if objective is None else objective.describe()
    if joined is not None and joined['binding_unavailable_reason'] is not None:
        reasons.append(joined['binding_unavailable_reason'])
    reasons = list(dict.fromkeys(reasons))
    cache = {'shared_envelope_bytes':envelope_bytes,'binding_bytes':0 if binding is None else len(binding),
        'objective_metadata_bytes':0 if objective is None else len(objective._metadata),
        'objective_join_bytes':0 if objective is None else sum(map(len,objective._rows)),
        'objective_image_admission_bytes':0 if objective is None else sum(map(len,objective._image_admission))}
    description = {'schema_version':1,'profile':PROFILE,'status':'unavailable' if reasons else 'conditional_binding_available',
        'unavailable_reasons':reasons,'mapping_status':'unavailable' if mapping is None else 'fixed_supplied_inverse_available',
        'input_bindings':{'body_raw_sha256':body_pin,'body_canonical_sha256':None if coupling is None else coupling['body_sha256'],
            'camera_raw_sha256':camera_pin,'camera_canonical_sha256':camera_model_pin,
            'declaration_raw_sha256':None if calibration_world_declaration is None else sha256(calibration_world_declaration).hexdigest(),
            'source_sha256':metadata['source_sha256'],'envelope_bindings':metadata['bindings']},
        'binding_sha256':None if binding is None else sha256(binding).hexdigest(),
        'camera_coupling':coupling,'camera_coupling_sha256':coupling_pin,
        'original_body_dt_s':None if body is None else body['dt'],'source_clocks':metadata['source_clocks'],
        'source_counts':metadata['counts'],'objective_status':None if joined is None else joined['status'],
        'objective_counts':None if joined is None else joined['counts'],
        'original_start_frame':None if joined is None else joined['original_start_frame'],
        'cache_bytes':cache,'contact':'unknown','load':'unknown','grasp':'unknown','physical_validation':'unverified',
        'body_or_floor_installed':False,'geometry_modified':False,'geometry_evaluations':0,'fit_or_simulator_admission':False,
        'scope':'Fixed supplied camera-body coordinate prior plus explicit source/entity/correspondence joins; not recovered ground or contact',
        'bounds':{'cooperative_seconds':MAX_SECONDS,'body_bytes':MAX_BODY_BYTES,'camera_bytes':MAX_CAMERA_BYTES,
            'declaration_bytes':MAX_DECLARATION_BYTES,'binding_bytes':MAX_BINDING_BYTES,'metadata_bytes':MAX_METADATA_BYTES,
            'source_frames':MAX_FRAMES,'body_references':MAX_REFERENCES,'channels':MAX_CHANNELS}}
    encoded = _encode(description,MAX_METADATA_BYTES,deadline)
    _check(deadline)
    return CameraSurfaceBinding(envelopes,body_document,camera_document,calibration_world_declaration,objective,binding,encoded)
