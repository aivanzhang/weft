"""Source wrist ranges versus saved model silhouettes; never physical contact evidence."""

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import PurePosixPath
import time

from weft.exercise_support_evidence import _equipment_mapping, _equipment_point
from weft.io import canonical_hash
from weft.pose_surface_annotations import PoseSurfaceAnnotations
from weft.source_surface_anchors import _document
from weft.source_surface_interval_construction import _encode
from weft.world_evidence import load_world_evidence


PROFILE = 'equipment-silhouette-evidence-v1'
MAX_SECONDS, MAX_MASK_BYTES, MAX_EQUIPMENT_BYTES = 10., 8 << 20, 16 << 20
MAX_CACHE_BYTES, MAX_RASTER_BYTES, MAX_LOADS, MAX_RUNS = 8 << 20, 4 << 20, 512, 100_000
CELL_MAPPING = 'Raster cell edges span the original half-open model box; centroid uses cell centers'
WRISTS = {'left_wrist':15,'right_wrist':16}


def _require(value, message):
    if not value:
        raise ValueError('Equipment silhouette evidence: '+message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Equipment silhouette evidence cooperative deadline exceeded')


def _number(value, maximum=4096):
    return type(value) in (int,float) and math.isfinite(value) and 0 <= value <= maximum


def _ratio(value):
    return Fraction(value)


def _interval(value):
    converted = float(value)
    represented = Fraction(converted)
    return [math.nextafter(converted,-math.inf) if represented > value else converted,
            math.nextafter(converted,math.inf) if represented < value else converted]


def _rectangle(point, width, height):
    pixel, uncertainty = point['pixel'], point['authored_uncertainty_px']
    _require(type(pixel) is list and len(pixel) == 2 and all(_number(v) for v in pixel)
             and _number(uncertainty,64), 'bounded supplied wrist pixel and explicit authored range required')
    u = _ratio(uncertainty)
    x, y = _ratio(pixel[0]), _ratio(pixel[1])
    if x-u < 0 or x+u > width or y-u < 0 or y+u > height:
        return None
    return (x-u,x+u,y-u,y+u)


def _runs(raw, dimensions, box, deadline, work):
    height, width = dimensions
    _require(type(raw) is bytes and len(raw) == height*width, 'exact bounded binary raster bytes required')
    _require(raw.count(b'\0')+raw.count(b'\1') == len(raw), 'saved raster must be binary')
    left, top, right, bottom = map(Fraction,box)
    sx, sy = (right-left)/width, (bottom-top)/height
    runs = []
    for row in range(height):
        _check(deadline)
        end, cursor = (row+1)*width, row*width
        while cursor < end:
            _check(deadline)
            first = raw.find(b'\1',cursor,end)
            if first < 0: break
            last = raw.find(b'\0',first,end)
            if last < 0: last = end
            work['foreground_runs'] += 1
            _require(work['foreground_runs'] <= MAX_RUNS, 'foreground-run work bound exceeded; no partial result')
            runs.append((left+(first-row*width)*sx,left+(last-row*width)*sx,top+row*sy,top+(row+1)*sy))
            cursor = last+1
    return tuple(runs),raw.count(b'\1')


def _relation(rectangle, runs, deadline):
    if not runs:
        return {'status':'unavailable','reason':'empty_model_silhouette','intersects':None,'distance_infimum_squared_px2':None}
    x0,x1,y0,y1 = rectangle
    best = None
    intersects = False
    for left,right,top,bottom in runs:
        _check(deadline)
        if x1 >= left and x0 < right and y1 >= top and y0 < bottom:
            intersects, best = True,Fraction(0)
            break
        dx,dy = max(left-x1,x0-right,0),max(top-y1,y0-bottom,0)
        distance = dx*dx+dy*dy
        best = distance if best is None else min(best,distance)
    return {'status':'conditional_image_intersection' if intersects else 'conditional_image_exclusion','reason':None,
        'intersects':intersects,'touches_excluded_edge':not intersects and best == 0,
        'distance_infimum_squared_px2':{'numerator':best.numerator,'denominator':best.denominator,
                                      'outward_interval':_interval(best)}}


def _identity_fields(row):
    identity=row.get('identity')
    if type(identity) is not dict or identity.get('status') not in ('assumed_model_identity','unresolved_association'):
        return None
    group=identity.get('unresolved_group_id')
    if (type(identity.get('gap_seen')) is not bool or identity.get('physical_identity_verified') is not False
            or (group is not None and not (type(group) is str and 0 < len(group) <= 160))):
        return None
    # Raw proposal dictionaries may contain coordinates, including in withheld slots.
    return {key:identity.get(key) for key in ('status','unresolved_group_id','gap_seen','physical_identity_verified')}


def _mask_index(document, equipment, deadline):
    _require(document.get('profile') == 'decoded-segmentation-masks-v1' and document.get('physics_ready') is False
             and type(document.get('contract')) is dict and document['contract'].get('method') == 'saved-sam-mask-raster-v1'
             and document['contract'].get('mapping') == CELL_MAPPING, 'existing decoded-mask contract required')
    source, frames, rows = document.get('source'),document.get('frames'),document.get('observations')
    _require(type(source) is dict and source.get('submitted_video_sha256') == equipment['source']['sha256']
             and source.get('width_px') == equipment['image']['width_px'] and source.get('height_px') == equipment['image']['height_px']
             and type(source.get('frame_count')) is int and source['frame_count'] == equipment['source']['decoded_frames']
             and 1 <= source['frame_count'] <= 500, 'complete mask source/image identity differs')
    _require(equipment['provenance'].get('adapter') == 'saved_segmentation'
             and document.get('original_evidence_sha256') == equipment['provenance'].get('input_evidence_sha256'),
             'mask and canonical equipment producer pins differ')
    _require(type(frames) is list and len(frames) == source['frame_count']
             and type(rows) is list and 1 <= len(rows) <= 4000, 'bounded complete decoded-mask slots required')
    indexed, identities = {},set()
    for index,row in enumerate(rows):
        _check(deadline)
        _require(type(row) is dict and type(row.get('frame')) is int and 0 <= row['frame'] < len(frames)
                 and type(row.get('object_id')) is str and 0 < len(row['object_id']) <= 80
                 and type(row.get('id')) is str and 0 < len(row['id']) <= 240,
                 'bounded mask row identity required')
        _require(all(type(row.get(key)) is str and 0 < len(row[key]) <= 80 for key in ('phase','mask_status'))
                 and type(row.get('time_s')) in (int,float) and math.isfinite(row['time_s']),
                 'bounded scalar mask phase/status/clock metadata required')
        key = (row['frame'],row['object_id'])
        _require(key not in indexed and row['id'] not in identities, 'duplicate mask identity')
        indexed[key] = row;identities.add(row['id'])
    objects = {key[1] for key in indexed}
    _require(1 <= len(objects) <= 8 and len(rows) == len(frames)*len(objects), 'all model-object slots must survive')
    for index,frame in enumerate(frames):
        _check(deadline)
        _require(type(frame) is dict and type(frame.get('frame')) is int and frame['frame'] == index
                 and type(frame.get('observation_indices')) is list and len(frame['observation_indices']) == len(objects),
                 'complete original mask frame/index rows required')
        slots = frame['observation_indices']
        _require(all(type(i) is int and 0 <= i < len(rows) for i in slots) and len(set(slots)) == len(slots)
                 and {rows[i]['object_id'] for i in slots} == objects and all(rows[i]['frame'] == index for i in slots),
                 'mask frame references differ')
        expected = equipment['source']['frame_times_s'][index] if equipment['schema_version'] == 3 else index/equipment['source']['fps']
        _require(frame.get('time_s') == expected and all(rows[i]['time_s'] == expected for i in slots),
                 'mask frame clock differs from canonical equipment')
    return indexed


def _qualified_mask(row, observed, records, box, native_mapping, source, equivalence, deadline):
    if row is None: return 'mask_slot_unavailable'
    if row.get('phase') == 'pre_initialization': return 'mask_pre_initialization'
    if row.get('phase') not in ('initialization','post_initialization'): return 'mask_phase_unqualified'
    if row.get('proposal_status') != 'returned': return 'mask_proposal_unavailable'
    if row.get('mask_status') != 'decoded': return 'mask_not_decoded'
    record = records.get(row['id'])
    _require(record is not None and record['id'] in observed['evidence_ids']
             and record['object_id'] == row['object_id'] and record['frame'] == row['frame']
             and record['time_s'] == row['time_s'] == observed['time_s']
             and record['kind'] == 'model_segmentation_proposal' and record['status'] == 'inferred',
             'mask does not match qualified equipment observation/evidence')
    payload = record['payload']
    _require(canonical_hash(payload) == row.get('raw_observation_sha256')
             and all(row.get(key) == payload.get(key) for key in
                 ('frame','time_s','object_id','raw_model_id','phase','proposal_status','identity','confidence','original_source')),
             'decoded mask lost its exact raw proposal identity')
    _require(row.get('box') == payload.get('geometry') and type(row['box']) is dict
             and row['box'].get('box_xyxy_halfopen') == box
             and row.get('encoded_sha256') == payload.get('mask_provenance',{}).get('encoded_sha256'),
             'mask pixel-cell box or encoded provenance differs')
    _require(type(row.get('identity')) is dict and row['identity'].get('physical_identity_verified') is False,
             'model identity cannot become physical authentication')
    if _identity_fields(row) is None:
        return 'mask_identity_status_unqualified'
    if native_mapping is not None:
        original = row.get('original_source')
        _require(type(original) is dict and original.get('sha256') == source
                 and original.get('frame') == native_mapping['original_frame']
                 and original.get('time_s') == float(native_mapping['original_native_pts']*Fraction(equivalence.time_base)),
                 'mask original-source identity or native clock differs from verified mapping')
    else:
        _require(row.get('original_source') is None, 'same-source mask must not attach an undeclared original-source transform')
    _check(deadline)
    return None


@dataclass(frozen=True)
class EquipmentSilhouetteEvidence:
    pose_annotations: PoseSurfaceAnnotations
    equipment_document: bytes
    mask_document: bytes
    source_equivalence: object | None
    _metadata: bytes
    _rows: tuple[bytes,...]

    def describe(self):
        return json.loads(self._metadata)

    def frame(self,index):
        _require(type(index) is int and 0 <= index < len(self._rows), 'original source frame outside range')
        return json.loads(self._rows[index])


def prepare_equipment_silhouette_evidence(pose_annotations, *, equipment_document, mask_document,
        equipment_ids, wrist_channel_ids, raster_loader, source_equivalence=None, construction_frames=None):
    """Use canonical construction slots; any additional frame selection must be explicit.

    raster_loader receives an exact raster request with dimensions, binary SHA,
    byte/work limits and remaining cooperative time before it may load anything.
    The adapter never decodes an image, chooses pixel uncertainty or fits geometry.
    """
    deadline = time.monotonic()+MAX_SECONDS
    _require(type(pose_annotations) is PoseSurfaceAnnotations, 'prepared source pose annotations required')
    _require(type(equipment_document) is bytes and 0 < len(equipment_document) <= MAX_EQUIPMENT_BYTES
             and type(mask_document) is bytes and 0 < len(mask_document) <= MAX_MASK_BYTES, 'bounded exact source metadata required')
    _require(type(equipment_ids) in (tuple,list) and 1 <= len(equipment_ids) <= 8
             and all(type(v) is str and 0 < len(v) <= 80 for v in equipment_ids) and len(set(equipment_ids)) == len(equipment_ids),
             'explicit unique equipment IDs required')
    _require(type(wrist_channel_ids) in (tuple,list) and 1 <= len(wrist_channel_ids) <= 2
             and all(type(v) is str for v in wrist_channel_ids) and len(set(wrist_channel_ids)) == len(wrist_channel_ids)
             and callable(raster_loader), 'explicit wrist channels and bounded raster loader required')
    meta = pose_annotations.describe()
    _require(type(pose_annotations.pose_document) is bytes and len(pose_annotations.pose_document) <= 8<<20
             and sha256(pose_annotations.pose_document).hexdigest() == meta['provenance']['pose_raw_sha256'], 'original pose bytes differ')
    pose = _document(pose_annotations.pose_document,deadline)
    source = pose['source_sha256']; frame_count = max(pose['decoded_frames'],pose['frame_count'])
    _require(frame_count <= 1800 and len(pose['frames']) <= 900, 'complete source/pose slot bound exceeded')
    selected = None
    if construction_frames is not None:
        _require(type(construction_frames) in (tuple,list) and len(construction_frames) <= frame_count
                 and all(type(i) is int and 0 <= i < frame_count for i in construction_frames)
                 and list(construction_frames) == sorted(set(construction_frames)), 'explicit bounded ordered original-frame selection required')
        selected = frozenset(construction_frames)
    annotation = _document(pose_annotations.annotation_document(),deadline)
    channels = annotation['channels']
    _require(len(channels) == len(wrist_channel_ids) and {c['id'] for c in channels} == set(wrist_channel_ids)
             and all(c['point_role'] in WRISTS for c in channels)
             and len({c['point_role'] for c in channels}) == len(channels), 'prepare only the explicitly selected named wrist channels')
    channel_map = {c['id']:c for c in channels}; channels = [channel_map[i] for i in wrist_channel_ids]
    equipment_pin = sha256(equipment_document).hexdigest(); mask_pin = sha256(mask_document).hexdigest()
    inputs = dict(pose_annotations.input_documents)
    _require(len(inputs) < 8 and equipment_pin not in inputs, 'combined canonical source registry must fit eight distinct documents')
    registry = load_world_evidence(inputs|{equipment_pin:equipment_document})
    equipment = _document(equipment_document,deadline)
    _require(equipment['image'] == {'width_px':pose['width'],'height_px':pose['height']}
             and equipment['source']['decoded_frames'] == pose['decoded_frames'], 'equipment image/decoded source coverage differs')
    mapping,proof = _equipment_mapping(equipment,pose,source_equivalence,deadline)
    aliases = None if mapping is None else {equipment['source']['sha256']:source}
    exported = json.loads(registry.construction_bytes(source_aliases=aliases))
    sources = {s['document_sha256']:s for s in exported['sources']}
    human_rows = {(s['document_sha256'],r['object_id'],r['frame']):r for s in exported['sources'] for r in s['observations']}
    _require(set(equipment_ids) <= {o['id'] for o in equipment['objects']}, 'equipment IDs absent from exact document')
    equip_rows = {(r['frame'],r['object_id']):r for r in equipment['observations']}
    records = {r['id']:r for r in equipment['evidence']}
    masks = _document(mask_document,deadline); mask_index = _mask_index(masks,equipment,deadline)
    _require(masks['source'].get('original_source_sha256') in (None,source)
             and (mapping is None or masks['source'].get('original_source_sha256') == source), 'undeclared mask derivative transform')
    poses = {r['frame']:{key:r.get(key) for key in ('id','frame','time_s','decoder_timestamp_s')} for r in pose['frames']}
    work = {'pose_frame_queries':0,'raster_loads':0,'binary_raster_bytes':0,'foreground_runs':0,'pair_relations':0}
    output,cache_bytes,statuses = [],0,Counter()
    for frame in range(frame_count):
        _check(deadline)
        supplied = poses.get(frame); native = None if mapping is None else mapping.get(frame)
        derivative_frame = frame if mapping is None else None if native is None else native['derivative_frame']
        common = ('metadata_only_not_decoded' if frame >= pose['decoded_frames'] else
                  'excluded_by_explicit_frame_selection' if selected is not None and frame not in selected else
                  'source_pose_not_sampled' if supplied is None else None)
        human = {}
        for channel in channels:
            ref = channel['entity_ref']; observed = human_rows.get((ref['document_sha256'],ref['object_id'],frame))
            reason = common
            if reason is None:
                if observed is None: reason = 'human_observation_unavailable'
                elif observed['partition'] != 'construction' or observed['admission'] != 'included': reason = 'human_partition_excluded'
                elif observed['visibility'] not in ('detected','partial'): reason = 'human_source_unavailable'
            human[channel['id']] = (observed,reason)
        equipment_candidates = {}
        for identity in equipment_ids:
            observed = equip_rows.get((derivative_frame,identity)); row = mask_index.get((derivative_frame,identity))
            reason = common
            if reason is None and not any(why is None for _,why in human.values()): reason = 'no_qualified_wrist_source'
            if reason is None:
                if observed is None: reason = 'equipment_slot_unavailable'
                elif observed['partition'] != 'construction': reason = 'equipment_partition_excluded'
                elif row is not None and row.get('phase') == 'pre_initialization': reason = 'mask_pre_initialization'
            box = None
            if reason is None:
                box,reason = _equipment_point(observed,supplied,equipment['schema_version'] == 3,native)
            if reason is None:
                reason = _qualified_mask(row,observed,records,box,native,source,source_equivalence,deadline)
            equipment_candidates[identity] = (observed,row,box,reason)
        points = {}
        if any(v[3] is None for v in equipment_candidates.values()):
            work['pose_frame_queries'] += 1
            queried = pose_annotations.frame(frame)
            points = {p['channel_id']:p for p in queried['channels']}
        pairs = []
        for identity,(observed,mask,box,equipment_reason) in equipment_candidates.items():
            pending = []
            for channel in channels:
                human_row,human_reason = human[channel['id']]
                point = points.get(channel['id']); reason = human_reason or equipment_reason
                rectangle = None
                if reason is None:
                    _require(point['landmark_name'] == channel['point_role'] and point['landmark_index'] == WRISTS[channel['point_role']], 'named wrist identity differs')
                    if point['status'] != 'copied_source_image_estimate': reason = point['reason'] or 'source_wrist_unavailable'
                    else:
                        rectangle = _rectangle(point,pose['width'],pose['height'])
                        if rectangle is None: reason = 'authored_pixel_range_outside_source_image'
                pair = {'wrist_channel_id':channel['id'],'wrist_name':channel['point_role'],'equipment_id':identity,
                    'human_entity_ref':channel['entity_ref'],'human_observation_id':None if human_row is None else human_row['id'],
                    'human_partition':None if human_row is None else human_row['partition'],
                    'equipment_observation_id':None if observed is None else observed['id'],
                    'equipment_time_s':None if observed is None else observed['time_s'],
                    'equipment_partition':None if observed is None else observed['partition'],
                    'mask_evidence_id':None if mask is None else mask['id'],'mask_phase':None if mask is None else mask.get('phase'),
                    'mask_status':None if mask is None else mask.get('mask_status'),
                    'mask_time_s':None if mask is None else mask.get('time_s'),
                    'model_identity':None if reason is not None else _identity_fields(mask),
                    'status':'unavailable','reason':reason,'pixel':None if point is None or human_reason else point['pixel'],
                    'source_confidence':None if point is None or human_reason else point['source_confidence'],
                    'source_visibility_score':None if point is None or human_reason else point['source_visibility_score'],
                    'source_presence_score':None if point is None or human_reason else point['source_presence_score'],
                    'representation':None if point is None or human_reason else point['representation'],
                    'authored_uncertainty_px':None if point is None else point['authored_uncertainty_px'],
                    'intersects':None,'distance_infimum_squared_px2':None,'contact':'unknown'}
                pairs.append(pair)
                if reason is None: pending.append((pair,rectangle))
            if pending:
                dimensions = mask.get('raster_dimensions'); path = mask.get('raster_path'); raster_pin = mask.get('raster_sha256')
                _require(type(dimensions) is list and len(dimensions) == 2 and all(type(v) is int and 1 <= v <= 4096 for v in dimensions)
                         and type(path) is str and 0 < len(path) <= 160 and not PurePosixPath(path).is_absolute()
                         and '..' not in PurePosixPath(path).parts and '\\' not in path
                         and type(raster_pin) is str and len(raster_pin) == 64
                         and all(c in '0123456789abcdef' for c in raster_pin), 'bounded exact raster descriptor required')
                size = dimensions[0]*dimensions[1]
                _require(work['raster_loads'] < MAX_LOADS and work['binary_raster_bytes']+size <= MAX_RASTER_BYTES,
                         'raster work budget exhausted before loader call')
                _require(type(box) is list and len(box) == 4 and all(type(v) is int for v in box)
                         and 0 <= box[0] < box[2] <= pose['width'] and 0 <= box[1] < box[3] <= pose['height'], 'half-open original image-cell box required')
                request = {'path':path,'raster_sha256':raster_pin,'dimensions':list(dimensions),'maximum_bytes':size,
                    'remaining_total_bytes':MAX_RASTER_BYTES-work['binary_raster_bytes'],'deadline_monotonic':deadline,
                    'maximum_seconds':max(0.,deadline-time.monotonic()),'mask_document_sha256':mask_pin,'mask_evidence_id':mask['id']}
                _check(deadline);work['raster_loads'] += 1
                raster = raster_loader(request);_check(deadline)
                _require(type(raster) is bytes and len(raster) == size and sha256(raster).hexdigest() == raster_pin, 'returned raster bytes differ from exact descriptor')
                work['binary_raster_bytes'] += size
                runs,foreground = _runs(raster,dimensions,box,deadline,work)
                _require(type(mask.get('silhouette')) is dict and type(mask['silhouette'].get('foreground_pixels')) is int
                         and mask['silhouette']['foreground_pixels'] == foreground, 'decoded foreground count differs')
                for pair,rectangle in pending:
                    pair.update(_relation(rectangle,runs,deadline));pair.update(raster_sha256=raster_pin,raster_dimensions=list(dimensions),image_box_px=box)
                    if pair['status']=='unavailable': pair['model_identity']=None
                    work['pair_relations'] += 1
        statuses.update(pair['reason'] or pair['status'] for pair in pairs)
        row = {'source_frame':frame,'source_status':'decoded' if frame < pose['decoded_frames'] else 'metadata_only_not_decoded',
            'pose_evidence_id':None if supplied is None else supplied['id'],'pose_time_s':None if supplied is None else supplied['time_s'],
            'pose_decoder_timestamp_s':None if supplied is None else supplied['decoder_timestamp_s'],
            'equipment_source_frame':derivative_frame,'source_mapping':native,'pairs':pairs}
        encoded = _encode(row,128<<10,deadline);cache_bytes += len(encoded)
        _require(cache_bytes <= MAX_CACHE_BYTES, 'complete result cache exceeds eight MiB; no slots dropped')
        output.append(encoded)
    description = {'schema_version':1,'profile':PROFILE,'status':'conditional_image_evidence_only',
        'source_sha256':source,'equipment_source_sha256':equipment['source']['sha256'],
        'equipment_document_sha256':equipment_pin,'mask_document_sha256':mask_pin,'pose_provenance':meta['provenance'],
        'source_equivalence':proof,'source_clocks':{key:value['time'] for key,value in sources.items()},
        'construction_frames':None if selected is None else list(construction_frames),
        'selection_policy':'Canonical construction eligibility intersected with the explicit original-frame selection; no automatic prefix',
        'counts':{'source_frames':frame_count,'decoded_frames':pose['decoded_frames'],'metadata_frames':pose['frame_count'],
            'pose_slots':len(poses),'pair_slots':frame_count*len(channels)*len(equipment_ids),'status_counts':dict(statuses)},
        'work':work,'cached_row_bytes':cache_bytes,'pixel_cell_mapping':CELL_MAPPING,
        'geometry_convention':'Authored closed wrist rectangle versus half-open foreground cells; exact rational intersection and distance infimum. Zero distance alone does not imply intersection.',
        'scope':'Model silhouette/image-estimate consistency or exclusion only; a wrist is not a hand-contact point. Occlusion, segmentation error and uncertain anatomy remain.',
        'contact':'unknown','physical_noncontact':'unknown','load':'unknown','grasp':'unknown','physical_validation':'unverified',
        'body_or_floor_installed':False,'native_calls':0,'image_decodes_in_adapter':0,
        'bounds':{'cooperative_seconds':MAX_SECONDS,'mask_metadata_bytes':MAX_MASK_BYTES,'equipment_metadata_bytes':MAX_EQUIPMENT_BYTES,
            'binary_raster_bytes':MAX_RASTER_BYTES,'raster_loads':MAX_LOADS,'foreground_runs':MAX_RUNS,'result_cache_bytes':MAX_CACHE_BYTES}}
    encoded = _encode(description,256<<10,deadline)
    _require(cache_bytes+len(encoded) <= MAX_CACHE_BYTES, 'complete result/report cache exceeds eight MiB')
    _check(deadline)
    return EquipmentSilhouetteEvidence(pose_annotations,equipment_document,mask_document,source_equivalence,encoded,tuple(output))
