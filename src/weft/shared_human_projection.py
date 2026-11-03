"""Externalize audit duplication, never runtime references or physical model values."""
from copy import deepcopy
from hashlib import sha256
import json
import math
import re

PROFILE = 'shared-human-provenance-v1'
RUNTIME_PROFILE = 'shared-human-runtime-projection-v1'
MAX_HUMAN_BYTES = 8 << 20
SCOPE = 'Runtime projection only; complete original inferred-body provenance remains in the exact sealed human.json sidecar'
FIELDS = {'schema_version', 'profile', 'artifact', 'source_sha256', 'full_human_sha256',
          'full_human_canonical_sha256', 'runtime_human_sha256', 'reference_count', 'scope'}


def _canonical(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _tree(value, depth=0, budget=None):
    if budget is None:budget=[300000]
    budget[0]-=1
    if depth>24 or budget[0]<0:raise ValueError('Human provenance exceeds bounded JSON complexity')
    if isinstance(value,dict):
        if len(value)>128 or any(not isinstance(k,str) or len(k)>512 for k in value):raise ValueError('Human provenance mapping is unbounded')
        for item in value.values():_tree(item,depth+1,budget)
    elif isinstance(value,list):
        if len(value)>1800:raise ValueError('Human provenance list exceeds source bounds')
        for item in value:_tree(item,depth+1,budget)
    elif isinstance(value,float):
        if not math.isfinite(value):raise ValueError('Human provenance must be finite')
    elif isinstance(value,str):
        if len(value)>8000:raise ValueError('Human provenance text is unbounded')
    elif value is not None and type(value) not in (int,bool):raise ValueError('Human provenance must be JSON')


def _raw(raw):
    if not isinstance(raw,bytes) or not 1<=len(raw)<=MAX_HUMAN_BYTES:raise ValueError('Full human sidecar requires bounded original bytes')
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise ValueError('Human provenance contains duplicate keys')
            result[k]=v
        return result
    try:value=json.loads(raw,object_pairs_hook=unique)
    except (UnicodeError,json.JSONDecodeError,RecursionError) as error:raise ValueError('Human sidecar is not bounded JSON') from error
    _tree(value)
    if not isinstance(value,dict):raise ValueError('Full human sidecar must be an object')
    return value


def _project(full):
    if (full.get('profile')!='exercise-v3' or type(full.get('schema_version')) is not int or full['schema_version']!=3
            or 'native_clock' not in full or 'camera_reference' in full or 'provenance_projection' in full
            or not isinstance(full.get('source_sha256'),str) or not re.fullmatch('[0-9a-f]{64}',full['source_sha256'])
            or not isinstance(full.get('reconstruction_coverage'),dict) or not isinstance(full.get('body_frame'),dict)
            or not isinstance(full['body_frame'].get('reference'),list)):
        raise ValueError('Projection requires complete native mobile-body provenance without camera/refinement projection')
    refs, roots, evidence = (full.get(k) for k in ('reference','root_reference','reference_evidence'))
    if (any(not isinstance(v,list) for v in (refs,roots,evidence)) or not 3<=len(refs)<=900
            or len(refs)!=len(roots) or len(refs)!=len(evidence)):
        raise ValueError('Projection must retain all aligned reference/root/evidence rows')
    identities=[];previous=-1
    for ref,root,origin in zip(refs,roots,evidence,strict=True):
        if not all(isinstance(row,dict) for row in (ref,root,origin)):raise ValueError('Projection reference rows must be objects')
        stamp=ref.get('time_s');identifier=origin.get('evidence_id')
        match=re.fullmatch(re.escape(full['source_sha256'])+r':frame:(\d+)',identifier) if isinstance(identifier,str) else None
        if (type(stamp) not in (int,float) or not math.isfinite(stamp) or not 0<=stamp<=60
                or root.get('time_s')!=stamp or origin.get('time_s')!=stamp or match is None
                or not previous<int(match[1])<1800 or not isinstance(origin.get('joint_confidence'),dict)):
            raise ValueError('Projection cannot change or invent source reference identities/confidence')
        previous=int(match[1]);identities.append({'time_s':stamp,'evidence_id':identifier})
    compact=deepcopy(full)
    compact.pop('reconstruction_coverage');compact['body_frame'].pop('reference')
    compact['reference_evidence']=identities
    compact['provenance_projection']={'profile':RUNTIME_PROFILE,'full_human_canonical_sha256':_canonical(full)}
    return compact


def create_projection(full_human_raw):
    """Project exact bytes after caller's ordinary full-body validation; no qualification occurs here."""
    full=_raw(full_human_raw);compact=_project(full)
    descriptor={'schema_version':1,'profile':PROFILE,'artifact':'human.json','source_sha256':full['source_sha256'],
        'full_human_sha256':sha256(full_human_raw).hexdigest(),'full_human_canonical_sha256':_canonical(full),
        'runtime_human_sha256':_canonical(compact),'reference_count':len(full['reference']),'scope':SCOPE}
    return compact,descriptor


def validate_projection(compact, descriptor, full_human_raw=None):
    """Check model-only binding, or exact sidecar join when bytes are supplied; never executes producer code."""
    _tree(compact);_tree(descriptor)
    if (not isinstance(compact,dict) or not isinstance(descriptor,dict) or set(descriptor)!=FIELDS
            or type(descriptor.get('schema_version')) is not int or descriptor['schema_version']!=1
            or descriptor.get('profile')!=PROFILE or descriptor.get('artifact')!='human.json'
            or descriptor.get('scope')!=SCOPE or type(descriptor.get('reference_count')) is not int
            or not 3<=descriptor['reference_count']<=900
            or any(not isinstance(descriptor.get(k),str) or not re.fullmatch('[0-9a-f]{64}',descriptor[k])
                   for k in ('source_sha256','full_human_sha256','full_human_canonical_sha256','runtime_human_sha256'))):
        raise ValueError('Shared human provenance descriptor differs from closed contract')
    marker={'profile':RUNTIME_PROFILE,'full_human_canonical_sha256':descriptor['full_human_canonical_sha256']}
    if (compact.get('provenance_projection')!=marker or compact.get('source_sha256')!=descriptor['source_sha256']
            or compact.get('profile')!='exercise-v3' or 'native_clock' not in compact or 'camera_reference' in compact
            or 'reconstruction_coverage' in compact or not isinstance(compact.get('body_frame'),dict)
            or 'reference' in compact['body_frame']
            or _canonical(compact)!=descriptor['runtime_human_sha256']
            or any(not isinstance(compact.get(k),list) or len(compact[k])!=descriptor['reference_count']
                   for k in ('reference','root_reference','reference_evidence'))
            or any(not isinstance(row,dict) or set(row)!={'time_s','evidence_id'} for row in compact['reference_evidence'])):
        raise ValueError('Runtime human differs from its explicit provenance projection')
    if full_human_raw is not None:
        expected, binding=create_projection(full_human_raw)
        if descriptor!=binding or compact!=expected:raise ValueError('Complete sidecar and runtime projection are not the exact same body')
    return deepcopy(compact)


def validate_runtime_human(human):
    """Shared model-only validation; direct full-body/source readers reject the projection marker."""
    from weft.human import validate_human
    if not isinstance(human,dict):raise ValueError('Runtime human must be an object')
    if 'provenance_projection' not in human:return validate_human(human)
    marker=human.get('provenance_projection')
    if (not isinstance(marker,dict) or set(marker)!={'profile','full_human_canonical_sha256'}
            or marker.get('profile')!=RUNTIME_PROFILE or not isinstance(marker.get('full_human_canonical_sha256'),str)
            or not re.fullmatch('[0-9a-f]{64}',marker['full_human_canonical_sha256'])
            or human.get('profile')!='exercise-v3' or 'native_clock' not in human or 'camera_reference' in human):
        raise ValueError('Unsupported runtime human projection')
    original=deepcopy(human);original.pop('provenance_projection')
    result=validate_human(original)
    if result!=original:raise ValueError('Projected physical model must already be canonically validated')
    result['provenance_projection']=deepcopy(marker)
    return result


def compact_scene_if_needed(scene, full_human_raw):
    """Only oversized, ordinarily validated source bodies gain the explicit sidecar contract."""
    if len(json.dumps(scene,separators=(',',':'),allow_nan=False).encode()) <= 500_000:
        return scene
    if _canonical(_raw(full_human_raw)) != _canonical(scene['human_spec']):
        raise ValueError('Original full human bytes differ from the validated scene body')
    compact,descriptor=create_projection(full_human_raw)
    result=deepcopy(scene);result['human_spec']=compact;result['human_provenance']=descriptor
    return result
