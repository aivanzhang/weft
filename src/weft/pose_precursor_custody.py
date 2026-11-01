"""Bind saved original pose arrays without relabeling temporal processing as raw."""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import time

from weft.io import canonical_hash
from weft.source_surface_anchors import _document
from weft.source_surface_interval_construction import _encode


PROFILE='saved-pose-precursor-custody-v1'
MAX_SECONDS, MAX_INPUT_BYTES, MAX_CACHE_BYTES=5.,16<<20,2<<20
MEMBERS=('evidence.json','pose-landmarks.json','identity-audit.json','manifest.json')
RAW_STATUS='single_model_hypothesis_not_verified_person_count'
DERIVED=('identity_raw','identity_derivation_id','identity_recompute_required','image_refinement','image_refinement_raw')


def _require(value,message):
    if not value: raise ValueError('Pose precursor custody: '+message)


def _check(deadline):
    if time.monotonic()>=deadline: raise TimeoutError('Pose precursor custody cooperative deadline exceeded')


def _digest(value):
    return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _number(value,low,high):
    return type(value) in (int,float) and math.isfinite(value) and low<=value<=high


def _metadata_rows(document,source,count,deadline):
    rows=document.get('frames')
    _require(type(rows) is list and len(rows)<=900,'at most900 original pose rows required')
    result={}; previous=-1
    for row in rows:
        _check(deadline)
        _require(type(row) is dict and type(row.get('frame')) is int and previous<row['frame']<count
                 and row.get('id')==f"{source}:frame:{row['frame']}"
                 and _number(row.get('time_s'),0,60)
                 and (row.get('decoder_timestamp_s') is None or _number(row['decoder_timestamp_s'],0,60)),
                 'ordered original source/frame/clock metadata required')
        previous=row['frame']
        _require(all(row.get(key) is None or type(row[key]) is str and 0<len(row[key])<=80
                     for key in ('status','identity_status')),
                 'bounded scalar original pose labels required')
        # Index metadata only; excluded landmark payloads are not compared or normalized.
        result[row['frame']]={key:row.get(key) for key in ('id','frame','time_s','decoder_timestamp_s')}
        result[row['frame']].update(status=row.get('status'),identity_status=row.get('identity_status'),
            pose_count=row.get('pose_count') if type(row.get('pose_count')) in (int,float,bool) and _number(row['pose_count'],-1e6,1e6) else None,
            pose_count_type=type(row.get('pose_count')).__name__,derivation_markers=[key for key in DERIVED if key in row])
    return result


@dataclass(frozen=True)
class PosePrecursorCustody:
    input_documents: tuple[tuple[str,bytes],...]
    _metadata: bytes
    _source_rows: tuple[bytes,...]
    _preparation_seconds: float

    def describe(self):
        return json.loads(self._metadata)

    def frame(self,index):
        _require(type(index) is int and 0<=index<len(self._source_rows),'original frame outside range')
        return json.loads(self._source_rows[index])

    def _begin(self,context_document):
        _require(type(context_document) is bytes and context_document==self.input_documents[0][1],
                 'annotation context bytes differ from sealed precursor context')
        return _Qualification(self)


class _Qualification:
    def __init__(self,custody):
        self.custody=custody; self.started=time.monotonic()
        self.deadline=self.started+MAX_SECONDS-custody._preparation_seconds
        _check(self.deadline)
        context=_document(custody.input_documents[0][1],self.deadline)
        raw=_document(custody.input_documents[1][1],self.deadline)
        self.context={row['frame']:row for row in context['frames']}
        self.raw={row['frame']:row for row in raw['frames']}
        self.cache={}; self.cache_bytes=len(custody._metadata)+sum(map(len,custody._source_rows))
        self.image_comparisons=0; self.finished=False

    def qualify(self,frame):
        _require(not self.finished,'qualification context is closed')
        _check(self.deadline)
        if frame in self.cache: return json.loads(self.cache[frame])
        result=self.custody.frame(frame); current=self.context.get(frame); raw=self.raw.get(frame)
        reason=result['reason']; representation='unqualified_precursor'
        if reason is None:
            if any(key in current or key in raw for key in DERIVED): reason='derived_or_crop_pose_requires_other_proof'
            elif current.get('identity_status')!=RAW_STATUS or raw.get('identity_status')!=RAW_STATUS:
                reason='unsupported_precursor_identity_status'
            elif (type(current.get('pose_count')) is not int or current['pose_count']!=1
                  or type(raw.get('pose_count')) is not int or raw['pose_count']!=1): reason='unavailable_or_ambiguous_model_hypothesis'
            elif current.get('status') not in ('tracked','tracked_partial') or raw.get('status') not in ('tracked','tracked_partial'):
                reason='unavailable_precursor_pose_status'
            elif current.get('status')!=raw.get('status'): reason='precursor_pose_status_differs'
            else:
                images=[raw.get('landmarks'),current.get('landmarks')]
                if any(type(points) is not list or len(points)!=33 for points in images): reason='missing_33_precursor_image_landmarks'
                elif any(type(point) is not dict or set(point)!={'x','y','z','visibility','presence'}
                         or any(value is not None and not _number(value,-1e6,1e6) for value in point.values())
                         for points in images for point in points): reason='unsupported_precursor_image_payload'
                else:
                    hashes=[sha256(_encode(points,16<<10,self.deadline)).hexdigest() for points in images]
                    self.image_comparisons+=1
                    result.update(raw_image_payload_sha256=hashes[0],context_image_payload_sha256=hashes[1])
                    if hashes[0]!=hashes[1]: reason='original_image_coordinates_or_scores_differ'
                    else: representation='matched_saved_precursor_model_estimate_identity_unverified'
        result.update(status='matched_saved_precursor' if reason is None else 'unavailable',reason=reason,representation=representation)
        encoded=_encode(result,4096,self.deadline); self.cache_bytes+=len(encoded)
        _require(self.cache_bytes<=MAX_CACHE_BYTES,'complete precursor cache exceeds two MiB; no frames dropped')
        self.cache[frame]=encoded
        return json.loads(encoded)

    def finish(self):
        _require(not self.finished,'qualification context is closed'); _check(self.deadline)
        rows=[]; counts=Counter()
        for index in range(len(self.custody._source_rows)):
            _check(self.deadline)
            row=self.diagnostic(index)
            counts[row['reason'] or row['status']]+=1
            rows.append(_encode(row,4096,self.deadline))
        summary=self.custody.describe()
        summary.update(qualification={'status':'construction_only_image_custody_not_identity_or_contact',
            'requested_frames':len(self.cache),'image_payload_comparisons':self.image_comparisons,
            'status_counts':dict(counts),'combined_processing_seconds':self.custody._preparation_seconds+time.monotonic()-self.started})
        encoded=_encode(summary,64<<10,self.deadline)
        _require(sum(map(len,rows))+len(encoded)<=MAX_CACHE_BYTES,'complete precursor qualification cache exceeds two MiB')
        _check(self.deadline); self.finished=True
        self.context.clear(); self.raw.clear(); self.cache.clear()
        return encoded,tuple(rows)

    def diagnostic(self,index):
        _check(self.deadline)
        if index in self.cache: return json.loads(self.cache[index])
        row=self.custody.frame(index)
        row.update(status='not_queried',reason=row['reason'] or 'not_queried_canonical_unqualified_or_unsampled',
                   representation='not_read_unqualified_source')
        return row


def prepare_pose_precursor_custody(*,context_pose_document,raw_pose_document,identity_audit_document,manifest_document):
    """Verify four supplied members; image comparisons happen only after canonical admission."""
    started=time.monotonic(); deadline=started+MAX_SECONDS
    supplied=(context_pose_document,raw_pose_document,identity_audit_document,manifest_document)
    _require(all(type(value) is bytes and 0<len(value)<=8<<20 for value in supplied)
             and sum(map(len,supplied))<=MAX_INPUT_BYTES,'four exact documents must fit sixteen MiB')
    documents=tuple(zip(MEMBERS,supplied,strict=True)); pins={name:sha256(value).hexdigest() for name,value in documents}
    manifest=_document(manifest_document,deadline)
    _require(type(manifest.get('schema')) is int and manifest['schema']==1 and type(manifest.get('files')) is dict
             and len(manifest['files'])<=4096,'existing bounded hash-only run manifest required')
    _require(all(manifest['files'].get(name)==pins[name] for name in MEMBERS[:3]),'fixed source member hashes differ')
    context,raw,audit=(_document(value,deadline) for value in supplied[:3])
    source=context.get('source_sha256')
    _require(context.get('profile')=='exercise-pose-v1' and type(context.get('schema_version')) is int
             and context['schema_version']==1 and _digest(source),'source-bound exercise pose context required')
    _require(all(type(context.get(key)) is int and 1<=context[key]<=1800 for key in ('frame_count','decoded_frames'))
             and all(type(context.get(key)) is int and 1<=context[key]<=4096 for key in ('width','height'))
             and _number(context.get('fps'),1,240),'explicit source dimensions/counts/fps required')
    count=max(context['frame_count'],context['decoded_frames'])
    _require(type(context.get('model')) is dict and type(raw.get('model')) is dict
             and _encode(context['model'],64<<10,deadline)==_encode(raw['model'],64<<10,deadline),
             'exact original model metadata differs')
    _require(type(raw.get('scope')) is str and 0<len(raw['scope'])<=2000 and raw['scope']==context.get('scope'),
             'original precursor scope differs')
    context_rows=_metadata_rows(context,source,count,deadline); raw_rows=_metadata_rows(raw,source,count,deadline)
    declaration=context.get('pose_identity')
    _require(type(declaration) is dict and declaration.get('mode') in ('raw','temporal')
             and declaration.get('audit_artifact')=='identity-audit.json'
             and declaration.get('raw_artifact_sha256')==pins['pose-landmarks.json']
             and type(declaration.get('repaired_frames')) is int and 0<=declaration['repaired_frames']<=900,
             'exact original context identity declaration required')
    _require(type(audit.get('schema_version')) is int and audit['schema_version']==1
             and audit.get('source_sha256')==source and audit.get('mode')==declaration['mode']
             and audit.get('raw_artifact')=='pose-landmarks.json' and audit.get('raw_artifact_sha256')==pins['pose-landmarks.json'],
             'audit raw/source/mode binding differs')
    method=audit.get('method')
    _require(method is None or type(method) is str and 0<len(method.strip())<=160 and len(method)<=160,
             'bounded scalar audit method must be null or a nonempty label')
    if audit['mode']=='temporal':
        _require(method=='short-island-bilateral-continuity-v1' and _digest(audit.get('implementation_sha256')),
                 'known recorded temporal audit method required')
    repaired=audit.get('repaired_evidence_ids')
    ids={row['id'] for row in context_rows.values()}
    _require(type(repaired) is list and len(repaired)==declaration['repaired_frames']
             and all(type(value) is str and value in ids for value in repaired)
             and len(set(repaired))==len(repaired) and (audit['mode']!='raw' or not repaired),
             'explicit repaired identity denominator differs')
    repaired_ids=frozenset(repaired)
    unavailable=audit.get('unavailable_frames',[]); decisions=audit.get('decisions',[])
    _require(type(unavailable) is list and len(unavailable)<=900 and type(decisions) is list and len(decisions)<=900,
             'bounded retained audit records required')
    reasons={}
    for item in unavailable:
        _check(deadline)
        _require(type(item) is dict and set(item)=={'evidence_id','reason'} and item['evidence_id'] in ids
                 and item['evidence_id'] not in reasons and type(item['reason']) is str and 0<len(item['reason'])<=240,
                 'exact source-bound audit unavailable record required')
        reasons[item['evidence_id']]=item['reason']
    rows=[]
    for frame in range(count):
        _check(deadline); current=context_rows.get(frame); original=raw_rows.get(frame)
        reason=('metadata_only_not_decoded' if frame>=context['decoded_frames'] else
                'context_pose_not_sampled' if current is None else 'raw_precursor_slot_missing' if original is None else
                'precursor_decoder_clock_missing' if current['decoder_timestamp_s'] is None or original['decoder_timestamp_s'] is None else
                'original_frame_or_clock_differs' if any(current[key]!=original[key] for key in ('id','frame','time_s','decoder_timestamp_s')) else
                'recorded_identity_repair_not_original_coordinates' if current['id'] in repaired_ids else None)
        row={'source_frame':frame,'source_status':'decoded' if frame<context['decoded_frames'] else 'metadata_only_not_decoded',
             'context_metadata':current,'raw_metadata':original,'status':'not_queried','reason':reason,
             'audit_mode':audit['mode'],'audit_identity_unavailable_reason':None if current is None else reasons.get(current['id']),
             'identity_authenticated':False,'contact':'unknown'}
        rows.append(_encode(row,4096,deadline))
    metadata={'schema_version':1,'profile':PROFILE,'status':'member_custody_only_pending_construction_image_comparison',
        'source_sha256':source,'input_raw_sha256s':pins,'source':{key:context[key] for key in ('width','height','fps','frame_count','decoded_frames')},
        'model_canonical_sha256':canonical_hash(context['model']),'identity_declaration_canonical_sha256':canonical_hash(declaration),
        'mode':audit['mode'],'recorded_repaired_frames':len(repaired),'audit_unavailable_frames':len(unavailable),'audit_decisions':len(decisions),
        'audit_method':method,'counts':{'original_source_slots':count,'context_pose_rows':len(context_rows),'raw_pose_rows':len(raw_rows)},
        'manifest_semantics':'supplied_hash_only_member_custody_not_authenticated_history_or_physical_identity',
        'clock_semantics':'exact_supplied_context_and_precursor_clocks_not_independent_decoder_verification',
        'identity_authenticated':False,'contact':'unknown','scope':'exact_saved_model_array_custody_not_anatomy_or_contact',
        'input_bytes':sum(map(len,supplied)),'bounds':{'cooperative_processing_seconds':MAX_SECONDS,'input_bytes':MAX_INPUT_BYTES,
            'source_slots':1800,'pose_rows':900,'cache_bytes':MAX_CACHE_BYTES}}
    encoded=_encode(metadata,64<<10,deadline)
    _require(len(encoded)+sum(map(len,rows))<=MAX_CACHE_BYTES,'complete metadata cache exceeds two MiB')
    _check(deadline)
    return PosePrecursorCustody(documents,encoded,tuple(rows),time.monotonic()-started)
