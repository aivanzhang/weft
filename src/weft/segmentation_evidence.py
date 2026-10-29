"""Sealed SAM proposals with exact decoder times, not verified physical identities."""

from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
import re

from weft.io import digest


CONTRACT = {
    'method': 'saved-sam-spatial-proposals-v1',
    'maximum_frames': 500, 'maximum_objects': 8, 'maximum_pair_checks': 14000,
    'maximum_json_bytes': 64_000_000, 'maximum_bundle_bytes': 256_000_000,
    'maximum_bundle_files': 128, 'maximum_mask_characters': 250_000,
    'proximity_gap_fraction': .5,
    'proximity_rule': 'Euclidean box edge gap <= 0.5 times the smaller box diagonal',
    'identity_rule': 'Groups persist after proximity and merge conservatively; no appearance analysis or physical identity verification',
    'gap_rule': 'A missing post-initialization slot permanently marks continuity unverified',
    'scope': 'Possible association ambiguity, not physical contact, collision, identity error, or simulator readiness',
}
_HASH = re.compile(r'[0-9a-f]{64}\Z')


def _hash(value):
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError('Expected a SHA256 identity')
    return value


def _number(value, low, high):
    if type(value) not in (int,float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError('Expected a bounded finite number')
    return value


def _integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError('Expected a bounded integer')
    return value


def _json_bytes(raw):
    if len(raw) > CONTRACT['maximum_json_bytes']:
        raise ValueError('Evidence JSON exceeds the byte bound')
    def invalid(value):
        raise ValueError('Nonfinite JSON is not evidence')
    return json.loads(raw, parse_constant=invalid)


class _Bundle:
    def __init__(self, directory, required):
        self.root = Path(directory).resolve()
        path = self.root/'manifest.json'
        if path.is_symlink() or path.stat().st_size > CONTRACT['maximum_json_bytes']:
            raise ValueError('Invalid bounded artifact manifest')
        raw = path.read_bytes()
        self.manifest_sha256 = sha256(raw).hexdigest()
        manifest = _json_bytes(raw)
        files = manifest.get('files')
        if not isinstance(files,dict) or not 1 <= len(files) <= CONTRACT['maximum_bundle_files'] or not set(required) <= files.keys():
            raise ValueError('Required inputs must belong to a bounded sealed bundle')
        total = 0
        self.files = {}
        for name, expected in files.items():
            if not isinstance(name,str) or Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Invalid artifact path')
            target = self.root/name
            if target.is_symlink() or not target.resolve().is_relative_to(self.root) or not target.is_file():
                raise ValueError('Artifact escapes the sealed bundle')
            total += target.stat().st_size
            if total > CONTRACT['maximum_bundle_bytes']:
                raise ValueError('Artifact bundle exceeds the byte bound')
            if digest(target) != _hash(expected):
                raise ValueError('Sealed artifact integrity failed: ' + name)
            self.files[name] = expected
        self.total_bytes = total

    def read(self,name):
        if name not in self.files:
            raise ValueError('Input is not sealed: ' + name)
        path = self.root/name
        if path.stat().st_size > CONTRACT['maximum_json_bytes']:
            raise ValueError('Evidence JSON exceeds the byte bound')
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != self.files[name]:
            raise ValueError('Evidence changed after verification')
        return _json_bytes(raw)

    def provenance(self):
        return {'manifest_sha256':self.manifest_sha256,'files':dict(self.files),
                'verified_files':len(self.files),'verified_bytes':self.total_bytes}


def _initializations(candidates, ids, count, width, height):
    present = [value for value in candidates if value is not None]
    if not present:
        raise ValueError('Initialization provenance is missing; do not infer it from filenames')
    normalized = []
    for values in present:
        if not isinstance(values,list) or len(values) != len(ids):
            raise ValueError('Initialization must cover every model ID')
        by_id = {}
        for value in values:
            if not isinstance(value,dict) or set(value) != {'object_id','frame','roi'}:
                raise ValueError('Malformed initialization provenance')
            identifier = str(_integer(value['object_id'],1,8))
            if identifier not in ids or identifier in by_id:
                raise ValueError('Initialization identity conflict')
            frame = _integer(value['frame'],0,count-1)
            roi = value['roi']
            if not isinstance(roi,list) or len(roi)!=4:
                raise ValueError('Initialization ROI requires four coordinates')
            x,y,w,h = [_integer(v,0,4096) for v in roi]
            if not w or not h or x+w>width or y+h>height:
                raise ValueError('Initialization ROI lies outside source image')
            by_id[identifier] = {'frame':frame,'roi':list(roi)}
        normalized.append(by_id)
    if any(value != normalized[0] for value in normalized[1:]):
        raise ValueError('Initialization sources disagree')
    return normalized[0]


def _geometry(raw, width, height):
    if raw.get('source_dimensions') != [width,height]:
        raise ValueError('Proposal source dimensions conflict')
    status = raw.get('status')
    if status not in ('returned','missing'):
        raise ValueError('Unknown raw proposal status')
    keys = ('box_xyxy_halfopen','box_xywh','mask_encoded','mask_dimensions','mask_encoding','confidence','box_confidence','mask_confidence')
    if status == 'missing':
        if any(raw.get(key) is not None for key in keys):
            raise ValueError('Missing proposal contains fabricated geometry/confidence')
        return None, {'box':None,'mask':None}, None
    box, xywh = raw.get('box_xyxy_halfopen'), raw.get('box_xywh')
    if not isinstance(box,list) or len(box)!=4:
        raise ValueError('Returned proposal needs a source box')
    x1,y1,x2,y2 = [_integer(value,0,4096) for value in box]
    if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height or xywh != [x1,y1,x2-x1,y2-y1]:
        raise ValueError('Returned box geometry conflicts')
    confidences = {'box':raw.get('box_confidence'),'mask':raw.get('mask_confidence')}
    for value in confidences.values():
        if value is not None:
            _number(value,0,1)
    if raw.get('confidence') != confidences['box']:
        raise ValueError('Box confidence aliases conflict')
    encoded, dimensions, encoding = raw.get('mask_encoded'), raw.get('mask_dimensions'), raw.get('mask_encoding')
    if (not isinstance(encoded,str) or not encoded.isascii() or not 6 <= len(encoded) <= CONTRACT['maximum_mask_characters']
            or encoding not in ('one_bit','lossless') or encoded[0] != ('!' if encoding=='one_bit' else '~')
            or raw.get('mask_status') != 'opaque_not_decoded_or_codec_validated'):
        raise ValueError('Invalid opaque mask provenance')
    if not isinstance(dimensions,list) or len(dimensions)!=2:
        raise ValueError('Invalid mask dimensions')
    for value in dimensions:
        _integer(value,1,4096)
    if math.prod(dimensions)>8_847_360:
        raise ValueError('Opaque mask dimensions exceed bound')
    geometry = {'box_xyxy_halfopen':list(box),'box_xywh':list(xywh),'center_px':[(x1+x2)/2,(y1+y2)/2],
                'coordinate_system':'submitted_image_pixels','status':'inferred'}
    mask = {'encoding':encoding,'dimensions':list(dimensions),'encoded_sha256':sha256(encoded.encode()).hexdigest(),
            'status':'opaque_not_decoded','scope':'Encoded provider payload retained unchanged in raw_model_output; no raster/area/centroid claim'}
    return geometry, confidences, mask


def _proximity(left,right):
    a,b = left['box_xyxy_halfopen'],right['box_xyxy_halfopen']
    gap = math.hypot(max(a[0]-b[2],b[0]-a[2],0),max(a[1]-b[3],b[1]-a[3],0))
    scale = min(math.hypot(a[2]-a[0],a[3]-a[1]),math.hypot(b[2]-b[0],b[3]-b[1]))
    threshold = CONTRACT['proximity_gap_fraction'] * scale
    return gap <= threshold, gap, threshold


def _mapping(bundle, video_hash, times, width, height):
    report = bundle.read('report.json')
    choices = [name for name in ('frame-mapping-and-pixels.json','frame-mapping.json') if name in bundle.files]
    if len(choices)!=1:
        raise ValueError('Expected exactly one supported sealed frame mapping format')
    mapping = bundle.read(choices[0])
    original_hash = _hash(report.get('source_sha256'))
    if (report.get('derivative_sha256') != video_hash or original_hash not in bundle.files.values()
            or video_hash not in bundle.files.values()
            or report.get('dimensions',report.get('frame_dimensions')) != [width,height]
            or report.get('all_frame_mapping_verified') is not True):
        raise ValueError('Mapping bundle does not verify this submitted video and original source')
    try:
        scale = Fraction(mapping['time_base'])
    except (TypeError,ValueError,ZeroDivisionError):
        raise ValueError('Invalid source mapping time base') from None
    if scale.numerator!=1 or not 1 <= scale.denominator <= 1_000_000_000:
        raise ValueError('Unsupported bounded source mapping clock')
    rows=mapping.get('frames')
    if not isinstance(rows,list) or len(rows)!=len(times):
        raise ValueError('Source mapping must cover every submitted frame')
    derivative_format=choices[0]=='frame-mapping.json'
    frame_key,pts_key=('derivative_frame','derivative_pts') if derivative_format else ('relative_frame','relative_pts')
    offset=0 if derivative_format else _integer(mapping.get('source_pts_offset'),0,3_600_000_000_000)
    previous,previous_time=-1,-1
    for index,row in enumerate(rows):
        frame=_integer(row.get('source_frame'),0,216000)
        if frame<=previous or type(row.get(frame_key)) is not int or row[frame_key]!=index:
            raise ValueError('Source mapping frame identity conflicts')
        source_pts=_integer(row.get('source_pts'),0,3_600_000_000_000)
        relative_pts=_integer(row.get(pts_key),0,3_600_000_000_000)
        original_time=_number(float(source_pts*scale),0,3600)
        relative_time=_number(float(relative_pts*scale),0,3600)
        if original_time<=previous_time or not math.isclose(relative_time,times[index],rel_tol=0,abs_tol=1e-6):
            raise ValueError('Mapping clock differs from decoded presentation timestamps')
        if source_pts-offset!=relative_pts:
            raise ValueError('Source mapping offset conflicts')
        previous,previous_time=frame,original_time
    return {'original_source_sha256':original_hash,'status':'verified_sealed_mapping',
            'raw_mapping':deepcopy(mapping),'mapping_format':choices[0],'verification':bundle.provenance()}


def _legacy_initialization(bundle, video_hash, ids, width, height):
    if bundle is None or ids!=['1'] or 'request-template.json' not in bundle.files:
        return None
    report=bundle.read('report.json')
    roi=report.get('source_frame0_roi')
    if roi is None:
        return None
    if report.get('derivative_sha256')!=video_hash or not isinstance(roi,list) or len(roi)!=4:
        raise ValueError('Legacy initialization is not pinned to this submitted video')
    x,y,w,h=(_integer(value,0,4096) for value in roi)
    prompt=f'<0f>1<|box;x1={x};y1={y};x2={x+w-1};y2={y+h-1};w={width};h={height}|>'
    template=bundle.read('request-template.json')
    inputs=template.get('input')
    if not isinstance(inputs,list) or len(inputs)!=1:
        raise ValueError('Legacy prompt template is malformed')
    content=inputs[0].get('content')
    if (not isinstance(content,list) or len(content)!=2 or content[0]!={'type':'input_text','text':prompt}
            or content[1].get('type')!='input_video' or template.get('model')!='sam-3.1'
            or template.get('metadata',{}).get('propagation_direction')!='forward'):
        raise ValueError('Legacy prompt template disagrees with the frozen initialization')
    return [{'object_id':1,'frame':0,'roi':list(roi)}]


def adapt_tracking_run(directory, *, source_mapping_bundle=None):
    """Read a sealed tracking run; no inference, decoding, API calls, or input mutation."""
    required=('source.mp4','video.json','report.json','call/call.json','call/segmentation.json')
    bundle=_Bundle(directory,required)
    video,call,result,report=(bundle.read(name) for name in ('video.json','call/call.json','call/segmentation.json','report.json'))
    video_hash=_hash(video.get('source_sha256'))
    if (video_hash!=bundle.files['source.mp4'] or call.get('video_sha256')!=video_hash
            or call.get('status')!='completed' or result.get('status')!='completed' or call.get('model')!='sam-3.1'):
        raise ValueError('Completed model output must match the sealed submitted video')
    if (not isinstance(call.get('response_id'),str) or not 1 <= len(call['response_id']) <= 256
            or call.get('response_id')!=result.get('response_id')):
        raise ValueError('Provider response identities conflict')
    _hash(call.get('request_sha256'))
    count=_integer(video.get('frame_count'),1,CONTRACT['maximum_frames'])
    width,height=(_integer(video.get(key),1,4096) for key in ('width','height'))
    if width*height>8_847_360:
        raise ValueError('Source image exceeds the area bound')
    times=video.get('decoder_timestamps_s')
    if not isinstance(times,list) or len(times)!=count:
        raise ValueError('Exact decoder timestamps are required for every frame')
    previous=-1
    for stamp in times:
        _number(stamp,0,3600)
        if stamp<=previous:
            raise ValueError('Decoder timestamps must strictly increase')
        previous=stamp
    raw_frames=result.get('frames')
    if not isinstance(raw_frames,list) or len(raw_frames)!=count:
        raise ValueError('Every submitted frame slot is required')
    ids=result.get('object_ids')
    if ids is None and raw_frames:
        ids=[raw_frames[0].get('object_id')]
    if (not isinstance(ids,list) or not 1 <= len(ids) <= CONTRACT['maximum_objects']
            or any(value not in tuple(str(i) for i in range(1,9)) for value in ids) or len(set(ids))!=len(ids)):
        raise ValueError('Expected 1..8 unique raw model IDs')
    mapping_bundle=_Bundle(source_mapping_bundle,('report.json',)) if source_mapping_bundle else None
    mapping=_mapping(mapping_bundle,video_hash,times,width,height) if mapping_bundle else None
    initialization=_initializations((call.get('initialization'),result.get('initialization'),report.get('objects'),
        _legacy_initialization(mapping_bundle,video_hash,ids,width,height)),ids,count,width,height)
    groups,events,observations,frames=[],[],[],[]
    components={identifier:{identifier} for identifier in ids}
    current_groups={identifier:None for identifier in ids}
    gaps={identifier:False for identifier in ids}
    checks=0
    for index,raw_frame in enumerate(raw_frames):
        if raw_frame.get('frame')!=index or type(raw_frame.get('frame')) is not int:
            raise ValueError('Frame slots must preserve their exact submitted indexes')
        raw_objects=raw_frame.get('objects',[raw_frame])
        if not isinstance(raw_objects,list) or len(raw_objects)!=len(ids) or {row.get('object_id') for row in raw_objects}!=set(ids):
            raise ValueError('Each frame needs every configured model object exactly once')
        by_id={row['object_id']:row for row in raw_objects}
        rows=[]
        for identifier in ids:
            raw=by_id[identifier]
            if raw.get('frame')!=index or type(raw.get('frame')) is not int:
                raise ValueError('Object frame differs from enclosing frame')
            geometry,confidence,mask=_geometry(raw,width,height)
            initial=initialization[identifier]['frame']
            phase='pre_initialization' if index<initial else 'initialization' if index==initial else 'post_initialization'
            if phase=='post_initialization' and raw['status']=='missing':
                gaps[identifier]=True
            rows.append({'id':f'{video_hash}:frame:{index}:model-object:{identifier}',
                'frame':index,'time_s':times[index],'object_id':'sam-'+identifier,'raw_model_id':identifier,
                'phase':phase,'proposal_status':raw['status'],'geometry':geometry,'confidence':confidence,
                'mask_provenance':mask,'raw_model_output':deepcopy(raw)})
        for offset,left in enumerate(rows):
            for right in rows[offset+1:]:
                checks+=1
                if checks>CONTRACT['maximum_pair_checks']:
                    raise ValueError('Pair checks exceed the resource bound')
                if left['geometry'] is None or right['geometry'] is None:
                    continue
                near,gap,threshold=_proximity(left['geometry'],right['geometry'])
                if not near:
                    continue
                a,b=left['raw_model_id'],right['raw_model_id']
                events.append({'frame':index,'time_s':times[index],'raw_model_ids':[a,b],
                    'evidence_ids':[left['id'],right['id']],'type':'possible_association_ambiguity',
                    'status':'inferred','edge_gap_px':gap,'threshold_px':threshold,'physical_contact_verified':False})
                if components[a]==components[b]:
                    continue
                members=components[a]|components[b]
                group_id=f'{video_hash}:identity-group:{index}:'+','.join(sorted(members))
                predecessors=sorted({current_groups[v] for v in members if current_groups[v] is not None})
                groups.append({'id':group_id,'raw_model_ids':sorted(members),'object_ids':['sam-'+v for v in sorted(members)],
                    'onset_frame':index,'onset_time_s':times[index],'evidence_ids':[left['id'],right['id']],
                    'predecessor_group_ids':predecessors,'status':'unresolved_association','physical_identity_verified':False})
                for member in members:
                    components[member]=members
                    current_groups[member]=group_id
        for row in rows:
            identifier=row['raw_model_id']
            row['identity']={'status':'unresolved_association' if current_groups[identifier] else 'assumed_model_identity',
                'unresolved_group_id':current_groups[identifier],'gap_seen':gaps[identifier],'physical_identity_verified':False}
            if mapping:
                original=mapping['raw_mapping']['frames'][index]
                row['original_source']={'sha256':mapping['original_source_sha256'],'frame':original['source_frame'],
                    'time_s':float(original['source_pts']*Fraction(mapping['raw_mapping']['time_base']))}
        observations.extend(rows)
        frames.append({'frame':index,'time_s':times[index],'observation_ids':[row['id'] for row in rows],
            'anonymous_location_set':[{'proposal_id':row['id'],'box_xyxy_halfopen':row['geometry']['box_xyxy_halfopen'].copy(),
                                      'center_px':row['geometry']['center_px'].copy()} for row in rows if row['geometry'] is not None],
            'identity_groups':sorted({current_groups[v] for v in ids if current_groups[v] is not None})})
    document={'schema_version':1,'profile':'segmentation-evidence-v1','contract':deepcopy(CONTRACT),
        'source':{'submitted_video_sha256':video_hash,'width_px':width,'height_px':height,'frame_count':count,
                  'original_source_sha256':mapping['original_source_sha256'] if mapping else None},
        'time':{'basis':'decoder_presentation_timestamps','unit':'seconds','origin':'submitted_video_start','timestamps_s':times.copy(),
                'status':'decoder_reported_not_physical_clock'},
        'objects':[{'id':'sam-'+identifier,'raw_model_id':identifier,'initialization':initialization[identifier],
                    'kind':None,'role':None,'identity_assumption':'Provider model ID continuity is unverified physical identity'} for identifier in ids],
        'observations':observations,'frames':frames,'identity_groups':groups,'proximity_events':events,
        'provenance':{'adapter':'saved_sam','model':call['model'],'response_id':call.get('response_id'),
            'request_sha256':call.get('request_sha256'),'tracking_verification':bundle.provenance(),
            'source_mapping':mapping,'raw_text_sha256':sha256(result.get('raw_text','').encode()).hexdigest(),
            'parser_metadata':{key:deepcopy(value) for key,value in result.items() if key not in ('frames','raw_text')},
            'pair_checks':checks,'raw_model_id_policy':'Unchanged; no manual-label use, nearest-label relabeling, or confidence threshold'},
        'physics_ready':False,
        'limits':['Model boxes and centers are inferred spatial proposals, not verified physical observations.',
            'Anonymous location sets have no temporal identity claim; raw IDs remain separately available.',
            'Proximity groups are conservative and persistent, even for visually distinct objects; no appearance analysis was performed.',
            'No proximity event proves a collision, contact, identity error, common depth, or common support plane.',
            'Missing proposals do not establish absence; initialization/pre-initialization are separate phases.',
            'Opaque masks are not decoded; box confidence is not calibrated physical identity confidence.',
            'Camera motion/calibration, shape, material, mass, force and mechanics remain unknown; no simulator fit or learning.']}
    if len(json.dumps(document,allow_nan=False).encode())>CONTRACT['maximum_json_bytes']:
        raise ValueError('Adapted evidence exceeds the JSON byte bound')
    return document
