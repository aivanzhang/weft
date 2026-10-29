"""Bounded fixed-reference background registration, never physical calibration."""

from copy import deepcopy
import math
import multiprocessing
from pathlib import Path
import re
import time

import cv2
import numpy as np

from weft.io import canonical_hash, digest


CONTRACT = {
    'method':'fixed-reference-background-similarity-v4','maximum_frames':500,'maximum_dimension':4096,
    'partition_rule':'deduplicated-reference-pool-before-matching-cell-stratified-v1',
    'seed_rule':'orb-training-only-bounded-similarity-no-composition-v1',
    'appearance_rule':'prefit-normalized-patch-compatibility-and-local-uniqueness-v1',
    'patch_radius_analysis_px':5,'patch_search_radius_analysis_px':8,
    'patch_candidate_radius_analysis_px':1,'patch_competitor_exclusion_analysis_px':2,
    'minimum_patch_std_intensity':5.,'minimum_patch_correlation':.8,'minimum_patch_margin':.1,
    'maximum_area':8_847_360,'maximum_video_bytes':37_000_000,'maximum_features':1000,
    'maximum_analysis_dimension':960,'maximum_correspondences':120,'maximum_workers':1,
    'deadline_seconds':30.,'cleanup_seconds':2.,'maximum_report_bytes':32_000_000,'maximum_ipc_event_bytes':250_000,
    'correspondence_retention':'Spatial sample after image-only appearance filtering; pre_appearance histograms retain unfiltered pair counts',
    'foreground_padding_analysis_px':21,'foreground_padding_box_diagonal_fraction':.75,
    'minimum_laplacian_variance':10.,'minimum_laplacian_variance_ratio':.2,
    'orb_ratio':.75,'orb_maximum_hamming':64,'lk_forward_backward_px':1.,
    'fit_ransac_source_px':1.5,'minimum_fit_count':24,'minimum_fit_inliers':20,'minimum_fit_inlier_fraction':.7,
    'minimum_withheld_count':12,'withheld_stride':3,'minimum_covariance_eigenratio':.02,
    'grid_size':4,'minimum_grid_cells':8,'minimum_bbox_width_fraction':.5,'minimum_bbox_height_fraction':.5,
    'minimum_bbox_area_fraction':.3,'minimum_hull_area_fraction':.2,
    'maximum_withheld_rmse_source_px':1.5,'maximum_withheld_median_source_px':1.,
    'maximum_withheld_p95_source_px':2.5,'maximum_withheld_error_source_px':6.,
    'minimum_scale':.95,'maximum_scale':1.05,'maximum_rotation_degrees':5.,
    'maximum_center_shift_diagonal_fraction':.05,
    'method_order':['orb','descriptor_seeded_lk'],'frame_clock_tolerance_seconds':1e-6,
    'scope':'Candidate background similarity registration; no physical calibration, camera pose, common-depth or collision proof',
}


def _finite(value,low,high):
    if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:
        raise ValueError('Invalid bounded finite registration value')
    return value


def _int(value,low,high):
    if type(value) is not int or not low<=value<=high:
        raise ValueError('Invalid bounded registration integer')
    return value


def _sha(value):
    if not isinstance(value,str) or not re.fullmatch(r'[0-9a-f]{64}',value):
        raise ValueError('Registration requires SHA256 source identities')
    return value


def _compact(evidence):
    if evidence.get('profile')!='segmentation-evidence-v1' or evidence.get('physics_ready') is not False:
        raise ValueError('Registration requires unpromoted segmentation evidence')
    source=evidence['source']; width,height=source['width_px'],source['height_px']
    _int(width,16,CONTRACT['maximum_dimension']);_int(height,16,CONTRACT['maximum_dimension'])
    if width*height>CONTRACT['maximum_area']:
        raise ValueError('Registration image area exceeds bound')
    count=_int(source['frame_count'],1,CONTRACT['maximum_frames'])
    times=evidence['time']['timestamps_s']
    if len(times)!=count:
        raise ValueError('Every source timestamp must be retained')
    previous=-1
    for value in times:
        _finite(value,0,3600)
        if value<=previous:
            raise ValueError('Source times must strictly increase')
        previous=value
    objects=evidence['objects']
    if not isinstance(objects,list) or not 1<=len(objects)<=8:
        raise ValueError('Expected at most eight explicit foreground objects')
    ids=[obj['id'] for obj in objects]
    if any(not isinstance(value,str) for value in ids) or len(set(ids))!=len(ids):
        raise ValueError('Foreground object identities conflict')
    initial=[]
    for obj in objects:
        frame=_int(obj['initialization']['frame'],0,count-1)
        roi=obj['initialization']['roi']
        if not isinstance(roi,list) or len(roi)!=4:
            raise ValueError('Initialization ROI is required for exclusions')
        x,y,w,h=(_finite(value,0,4096) for value in roi)
        if not w or not h or x+w>width or y+h>height:
            raise ValueError('Foreground initialization ROI lies outside the image')
        initial.append({'object_id':obj['id'],'frame':frame,'box':[x,y,x+w,y+h]})
    rows={}
    for row in evidence['observations']:
        frame=_int(row['frame'],0,count-1);identifier=row['object_id']
        if identifier not in ids or (frame,identifier) in rows or row['time_s']!=times[frame]:
            raise ValueError('Foreground observation identity/time conflict')
        box=None
        if row['proposal_status']=='returned':
            box=row['geometry']['box_xyxy_halfopen']
            if not isinstance(box,list) or len(box)!=4:
                raise ValueError('Foreground box is malformed')
            x1,y1,x2,y2=(_finite(value,0,4096) for value in box)
            if not x1<x2<=width or not y1<y2<=height:
                raise ValueError('Foreground box lies outside the submitted image')
        elif row['proposal_status']!='missing' or row['geometry'] is not None:
            raise ValueError('Missing foreground slots cannot invent positions')
        rows[(frame,identifier)]=deepcopy(box)
    if len(rows)!=count*len(ids):
        raise ValueError('Every foreground slot, including missing ones, is required')
    return {'source_sha256':_sha(source['submitted_video_sha256']),'width':width,'height':height,
        'timestamps_s':list(times),'initialization':initial,
        'foreground_boxes':[[rows[(frame,identifier)] for identifier in ids] for frame in range(count)],
        'object_ids':ids}


def registration_evidence_hash(evidence):
    """Hash exactly the source, times, authored ROIs and model boxes used for masks."""
    return canonical_hash(_compact(evidence))


def _analysis_shape(width,height):
    scale=min(1.,CONTRACT['maximum_analysis_dimension']/max(width,height))
    small=(max(16,round(width*scale)),max(16,round(height*scale)))
    return small,np.array([small[0]/width,small[1]/height])


def _exclusions(compact,frame):
    width,height=compact['width'],compact['height']
    _,scale=_analysis_shape(width,height)
    result=[]
    boxes=compact['foreground_boxes'][frame]+[obj['box'] for obj in compact['initialization']]
    for box in boxes:
        if box is None:
            continue
        x1,y1,x2,y2=box
        padding=max(CONTRACT['foreground_padding_analysis_px']/min(scale),
                    CONTRACT['foreground_padding_box_diagonal_fraction']*math.hypot(x2-x1,y2-y1))
        expanded=[max(0,x1-padding),max(0,y1-padding),min(width,x2+padding),min(height,y2+padding)]
        if expanded not in result:
            result.append(expanded)
    return result


def _mask(size,scale,boxes):
    mask=np.full((size[1],size[0]),255,np.uint8)
    mask[:4]=mask[-4:]=0;mask[:,:4]=mask[:,-4:]=0
    for x1,y1,x2,y2 in boxes:
        a,b=np.floor(np.array([x1,y1])*scale).astype(int)
        c,d=np.ceil(np.array([x2,y2])*scale).astype(int)
        mask[b:d,a:c]=0
    return mask


def _spread(points,width,height):
    points=np.asarray(points,dtype=float).reshape(-1,2)
    normalized=points/np.array([width,height])
    if len(points)<3:
        return {'cells':0,'quadrants':0,'bbox_width':0.,'bbox_height':0.,'bbox_area':0.,'hull_area':0.,'eigenratio':0.}
    span=np.ptp(normalized,axis=0)
    cells=np.minimum(3,np.maximum(0,np.floor(normalized*4).astype(int)))
    quadrants=np.minimum(1,np.maximum(0,np.floor(normalized*2).astype(int)))
    eigen=np.linalg.eigvalsh(np.cov(normalized.T))
    return {'cells':len(set(map(tuple,cells))),'quadrants':len(set(map(tuple,quadrants))),
        'bbox_width':float(span[0]),'bbox_height':float(span[1]),'bbox_area':float(np.prod(span)),
        'hull_area':float(cv2.contourArea(cv2.convexHull(normalized.astype(np.float32)))),
        'eigenratio':float(max(0,eigen[0])/max(eigen[-1],1e-15))}


def _spread_ok(value):
    return (value['cells']>=CONTRACT['minimum_grid_cells'] and value['quadrants']==4
        and value['bbox_width']>=CONTRACT['minimum_bbox_width_fraction']
        and value['bbox_height']>=CONTRACT['minimum_bbox_height_fraction']
        and value['bbox_area']>=CONTRACT['minimum_bbox_area_fraction']
        and value['hull_area']>=CONTRACT['minimum_hull_area_fraction']
        and value['eigenratio']>=CONTRACT['minimum_covariance_eigenratio'])


def _matrix_bounds(matrix,width,height):
    matrix=np.asarray(matrix,dtype=float)
    if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or not np.allclose(matrix[2],[0,0,1],atol=1e-12,rtol=0):
        return False
    a=matrix[:2,:2]
    scale=math.hypot(a[0,0],a[1,0])
    if not np.allclose(a,[[a[0,0],-a[1,0]],[a[1,0],a[0,0]]],atol=1e-9,rtol=0):
        return False
    rotation=abs(math.degrees(math.atan2(a[1,0],a[0,0])))
    center=np.array([width/2,height/2])
    shift=np.linalg.norm(a@center+matrix[:2,2]-center)/math.hypot(width,height)
    return (CONTRACT['minimum_scale']<=scale<=CONTRACT['maximum_scale']
            and rotation<=CONTRACT['maximum_rotation_degrees']
            and shift<=CONTRACT['maximum_center_shift_diagonal_fraction'])


def _subset_indices(reference,current,width,height,*,with_unique=False):
    # Spatial round-robin sampling precedes fitting and does not inspect residuals.
    buckets={};seen_reference=set();seen_current=set()
    for index,point in enumerate(reference):
        a,b=tuple(np.round(point,4)),tuple(np.round(current[index],4))
        if a in seen_reference or b in seen_current:
            continue
        seen_reference.add(a);seen_current.add(b)
        cell=tuple(np.minimum(3,np.maximum(0,np.floor(point/[width,height]*4).astype(int))))
        buckets.setdefault(cell,[]).append(index)
    unique=[index for values in buckets.values() for index in values]
    selected=[]
    while buckets and len(selected)<CONTRACT['maximum_correspondences']:
        for cell in sorted(list(buckets)):
            selected.append(buckets[cell].pop(0))
            if not buckets[cell]:
                del buckets[cell]
            if len(selected)==CONTRACT['maximum_correspondences']:
                break
    result=np.asarray(selected,dtype=int)
    return (result,np.asarray(unique,dtype=int)) if with_unique else result


def _subset(reference,current,width,height):
    selected=_subset_indices(reference,current,width,height)
    return np.asarray(reference)[selected],np.asarray(current)[selected]


def _fit_partition(reference,width,height):
    """Partition the fixed reference pool without matching results or residuals."""
    cells={}
    for index,point in enumerate(reference):
        cell=tuple(np.minimum(3,np.maximum(0,np.floor(point/[width,height]*4).astype(int))))
        cells.setdefault(cell,[]).append(index)
    fit=np.ones(len(reference),dtype=bool)
    for (x,y),indices in sorted(cells.items()):
        ordered=sorted(indices,key=lambda index:tuple(reference[index]))
        if len(ordered)==1:
            # A singleton cannot support both sets; balance these cells deterministically.
            fit[ordered[0]]=(y*4+x)%CONTRACT['withheld_stride']!=0
            continue
        held_count=max(1,(len(ordered)+1)//CONTRACT['withheld_stride'])
        for rank in range(held_count):
            fit[ordered[1+rank*CONTRACT['withheld_stride']]]=False
    return fit


def _fit(reference,current,width,height,method,*,identity=False,pool=None,feature_ids=None):
    matched=len(reference)
    selected,unique=_subset_indices(reference,current,width,height,with_unique=True)
    stages={'deduplicated_reference':_histogram(np.asarray(reference)[unique],width,height),
            'deduplicated_current':_histogram(np.asarray(current)[unique],width,height)}
    reference,current=np.asarray(reference)[selected],np.asarray(current)[selected]
    ids=None if feature_ids is None else np.asarray(feature_ids,dtype=int)[selected]
    fit=_fit_partition(reference,width,height) if pool is None else _fit_partition(pool,width,height)[ids]
    held=~fit
    audit={'method':method,'stages':stages,'status':'unavailable','reason':'insufficient_correspondences',
        'candidate_current_to_reference':None,'matched_count':matched,'fit_count':int(sum(fit)),
        'withheld_count':int(sum(held)),'fit_inlier_count':0,'fit_outlier_count':int(sum(fit)),
        'heldout_rmse_px':None,'heldout_median_px':None,'heldout_p95_px':None,'heldout_max_px':None,
        'fit_spread':_spread(current[fit],width,height),
        'withheld_source_spread':_spread(current[held],width,height),
        'withheld_reference_spread':_spread(reference[held],width,height),'correspondences':[]}
    matrix=None;inliers=np.zeros(sum(fit),dtype=bool)
    if sum(fit)>=CONTRACT['minimum_fit_count'] and sum(held)>=CONTRACT['minimum_withheld_count']:
        if audit['fit_spread']['eigenratio']<CONTRACT['minimum_covariance_eigenratio']:
            audit['reason']='rank_deficient_fit'
        else:
            affine,flags=cv2.estimateAffinePartial2D(current[fit].astype(np.float32),reference[fit].astype(np.float32),
                method=cv2.RANSAC,ransacReprojThreshold=CONTRACT['fit_ransac_source_px'],maxIters=1000,confidence=.99,refineIters=10)
            if affine is not None and flags is not None:
                matrix=np.eye(3) if identity else np.vstack([affine,[0,0,1]])
                inliers=flags.ravel().astype(bool)
    errors=np.full(len(reference),np.nan)
    if matrix is not None:
        audit['candidate_current_to_reference']=matrix.tolist()
        errors=np.linalg.norm(current@matrix[:2,:2].T+matrix[:2,2]-reference,axis=1)
        residual=errors[held]
        audit.update(fit_inlier_count=int(sum(inliers)),fit_outlier_count=int(sum(~inliers)),
            heldout_rmse_px=float(np.sqrt(np.mean(residual**2))),heldout_median_px=float(np.median(residual)),
            heldout_p95_px=float(np.percentile(residual,95)),heldout_max_px=float(max(residual)))
        positive=(sum(inliers)>=CONTRACT['minimum_fit_inliers'] and sum(inliers)/sum(fit)>=CONTRACT['minimum_fit_inlier_fraction'])
        spread=all(_spread_ok(audit[key]) for key in ('fit_spread','withheld_source_spread','withheld_reference_spread'))
        errors_ok=(audit['heldout_rmse_px']<=CONTRACT['maximum_withheld_rmse_source_px']
            and audit['heldout_median_px']<=CONTRACT['maximum_withheld_median_source_px']
            and audit['heldout_p95_px']<=CONTRACT['maximum_withheld_p95_source_px']
            and audit['heldout_max_px']<=CONTRACT['maximum_withheld_error_source_px'])
        audit['reason']=('transform_out_of_bounds' if not _matrix_bounds(matrix,width,height) else
                         'insufficient_fit_inliers' if not positive else 'insufficient_spatial_support' if not spread else
                         'withheld_residual_failed' if not errors_ok else None)
        audit['status']='registered' if audit['reason'] is None else 'unavailable'
    cursor=0
    for index,(target,point) in enumerate(zip(reference,current)):
        flag=bool(inliers[cursor]) if fit[index] else None
        cursor+=bool(fit[index])
        audit['correspondences'].append({'reference_xy':target.tolist(),'current_xy':point.tolist(),
            'partition':'fit' if fit[index] else 'withheld','fit_inlier':flag,
            'error_px':float(errors[index]) if matrix is not None else None,
            'reference_feature_id':None if ids is None else int(ids[index])})
    return audit


def _histogram(points,width,height):
    counts=[0]*16
    for point in points:
        x,y=np.minimum(3,np.maximum(0,np.floor(np.asarray(point)/[width,height]*4).astype(int)))
        counts[int(y)*4+int(x)]+=1
    return {'count':len(points),'cells':counts}


def _prepare(reference,mask):
    orb=cv2.ORB_create(nfeatures=CONTRACT['maximum_features'],edgeThreshold=21,patchSize=31,fastThreshold=10)
    keypoints,descriptors=orb.detectAndCompute(reference,mask)
    # One immutable pool makes seed-training and validation identities disjoint across methods.
    selected=[];seen=set()
    for index,point in enumerate(keypoints):
        location=tuple(np.round(point.pt,4))
        if location not in seen:
            selected.append(index);seen.add(location)
    points=np.array([keypoints[index].pt for index in selected],dtype=float).reshape(-1,2)
    radius=CONTRACT['patch_radius_analysis_px'];side=2*radius+1
    patches=[cv2.getRectSubPix(reference,(side,side),tuple(map(float,p))) for p in points]
    patch_support=[bool(np.min(cv2.getRectSubPix(mask,(side,side),tuple(map(float,p))))==255) for p in points]
    return {'image':reference,'patches':patches,'patch_support':patch_support,'orb':orb,'keypoints':[keypoints[index] for index in selected],
            'descriptors':None if descriptors is None else descriptors[selected],
            'pool':points,'detected_points':np.array([p.pt for p in keypoints],dtype=float).reshape(-1,2),
            'laplacian_variance':float(cv2.Laplacian(reference,cv2.CV_32F).var())}


def _appearance_filter(prepared,current,current_mask,ids,current_points,scale,linear=None):
    # Image compatibility precedes every fit; neither memberships nor residuals enter this gate.
    radius=CONTRACT['patch_radius_analysis_px'];search=CONTRACT['patch_search_radius_analysis_px']
    extent=radius+search;side=2*radius+1
    yy,xx=np.mgrid[-extent:extent+1,-extent:extent+1].astype(np.float32)
    offsets=np.stack([xx,yy],axis=-1)
    if linear is not None:
        offsets=offsets@np.asarray(linear,dtype=np.float32).T
    candidate_radius=CONTRACT['patch_candidate_radius_analysis_px']
    exclusion=CONTRACT['patch_competitor_exclusion_analysis_px']
    gy,gx=np.mgrid[-search:search+1,-search:search+1]
    candidate=(np.maximum(abs(gx),abs(gy))<=candidate_radius)
    competitor=(np.maximum(abs(gx),abs(gy))>exclusion)
    keep=[];metrics={};rejected={'support':0,'texture':0,'correlation':0,'ambiguous':0}
    for index,(identifier,point) in enumerate(zip(ids,current_points)):
        patch=prepared['patches'][identifier]
        if not prepared['patch_support'][identifier]:
            rejected['support']+=1;continue
        if float(np.std(patch))<CONTRACT['minimum_patch_std_intensity']:
            rejected['texture']+=1;continue
        maps=offsets+np.asarray(point)*scale
        mapx,mapy=maps[:,:,0].astype(np.float32),maps[:,:,1].astype(np.float32)
        mask=cv2.remap(current_mask,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        if np.min(mask)!=255:
            rejected['support']+=1;continue
        window=cv2.remap(current,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        scores=cv2.matchTemplate(window,patch,cv2.TM_CCOEFF_NORMED)
        values=window.astype(np.float32)
        mean=cv2.boxFilter(values,-1,(side,side))[radius:-radius,radius:-radius]
        square=cv2.boxFilter(values*values,-1,(side,side))[radius:-radius,radius:-radius]
        textured=np.maximum(0,square-mean*mean)>=CONTRACT['minimum_patch_std_intensity']**2
        valid=np.isfinite(scores)&textured
        if not np.any(valid&candidate):
            rejected['texture']+=1;continue
        score=float(np.max(scores[valid&candidate]))
        if score<CONTRACT['minimum_patch_correlation']:
            rejected['correlation']+=1;continue
        alternative=float(np.max(scores[valid&competitor])) if np.any(valid&competitor) else -1.
        margin=score-alternative
        if margin<CONTRACT['minimum_patch_margin']:
            rejected['ambiguous']+=1;continue
        keep.append(index);metrics[int(identifier)]=[round(score,6),round(margin,6)]
    return np.asarray(keep,dtype=int),metrics,rejected


def _seed_supported(audit,width,height):
    return (audit['candidate_current_to_reference'] is not None
            and _matrix_bounds(audit['candidate_current_to_reference'],width,height)
            and audit['fit_count']>=CONTRACT['minimum_fit_count']
            and audit['fit_inlier_count']>=CONTRACT['minimum_fit_inliers']
            and audit['fit_inlier_count']/audit['fit_count']>=CONTRACT['minimum_fit_inlier_fraction']
            and audit['fit_spread']['eigenratio']>=CONTRACT['minimum_covariance_eigenratio'])



def _allowed(points,mask):
    result=[]
    for point in points:
        if not np.isfinite(point).all():
            result.append(False);continue
        x,y=np.rint(point).astype(int)
        result.append(0<=x<mask.shape[1] and 0<=y<mask.shape[0] and mask[y,x]!=0)
    return np.asarray(result,dtype=bool)


def _pair(prepared,current,current_mask,scale,width,height,*,identity=False):
    reference=prepared['image'];attempts=[];pool=prepared['pool']/scale
    quality={'reference_laplacian_variance':prepared['laplacian_variance'],
             'current_laplacian_variance':float(cv2.Laplacian(current,cv2.CV_32F).var())}
    stages={'reference_detected':_histogram(prepared['detected_points']/scale,width,height),
            'reference_pool':_histogram(pool,width,height)}
    def fitted(ref,cur,ids,method,stage,*,linear=None):
        stage={**stage,'pre_appearance_reference':_histogram(ref,width,height),
               'pre_appearance_current':_histogram(cur,width,height)}
        selected,metrics,rejected=_appearance_filter(prepared,current,current_mask,ids,cur,scale,linear)
        ref,cur=np.asarray(ref)[selected],np.asarray(cur)[selected]
        ids=np.asarray(ids,dtype=int)[selected]
        audit=_fit(ref,cur,width,height,method,identity=identity,pool=pool,feature_ids=ids)
        audit['image_quality']=quality
        audit['appearance_rejections']=rejected
        for point in audit['correspondences']:
            point['appearance']=metrics[point['reference_feature_id']]
        audit['stages']={**audit['stages'],**stages,**stage,
            'matched_reference':_histogram(ref,width,height),'matched_current':_histogram(cur,width,height),
            'retained_reference':_histogram([p['reference_xy'] for p in audit['correspondences']],width,height),
            'retained_current':_histogram([p['current_xy'] for p in audit['correspondences']],width,height)}
        return audit
    empty=np.empty((0,2))
    if (min(quality.values())<CONTRACT['minimum_laplacian_variance']
            or quality['current_laplacian_variance']/quality['reference_laplacian_variance']<CONTRACT['minimum_laplacian_variance_ratio']):
        audit=fitted(empty,empty,[],'orb',{})
        audit['reason']='blur_or_low_texture'
        return [audit]
    keypoints,descriptors=prepared['orb'].detectAndCompute(current,current_mask)
    current_detected=np.array([p.pt for p in keypoints],dtype=float).reshape(-1,2)/scale
    first,second=prepared['descriptors'],descriptors
    reference_points=empty;current_points=empty;ids=[]
    stage={'current_detected':_histogram(current_detected,width,height)}
    if first is not None and second is not None and len(first)>=2 and len(second)>=2:
        matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
        forward=matcher.knnMatch(first,second,k=2);reverse=matcher.knnMatch(second,first,k=2)
        def accepted(pairs):
            return {pair[0].queryIdx:pair[0] for pair in pairs if len(pair)==2
                    and pair[0].distance<CONTRACT['orb_ratio']*pair[1].distance
                    and pair[0].distance<=CONTRACT['orb_maximum_hamming']}
        f,r=accepted(forward),accepted(reverse)
        stage['forward_descriptor_reference']=_histogram(pool[list(f)],width,height)
        matches=[match for index,match in sorted(f.items()) if match.trainIdx in r and r[match.trainIdx].trainIdx==index]
        ids=[m.queryIdx for m in matches]
        reference_points=pool[ids]
        current_points=np.array([keypoints[m.trainIdx].pt for m in matches],dtype=float).reshape(-1,2)/scale
    orb=fitted(reference_points,current_points,ids,'orb',stage);attempts.append(orb)
    if orb['status']=='registered':
        return attempts
    if not _seed_supported(orb,width,height):
        audit=fitted(empty,empty,[],'descriptor_seeded_lk',{})
        audit.update(reason='descriptor_training_seed_unavailable',seed=None)
        attempts.append(audit)
        return attempts
    seed=np.asarray(orb['candidate_current_to_reference']);forward=np.linalg.inv(seed)
    seed_audit={'current_to_reference':seed.tolist(),
                'training_reference_feature_ids':[p['reference_feature_id'] for p in orb['correspondences'] if p['partition']=='fit']}
    points=prepared['pool'].astype(np.float32).reshape(-1,1,2)
    initial=((pool@forward[:2,:2].T+forward[:2,2])*scale).astype(np.float32).reshape(-1,1,2)
    options={'winSize':(21,21),'maxLevel':3,'criteria':(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01),
             'flags':cv2.OPTFLOW_USE_INITIAL_FLOW}
    moved,status,_=cv2.calcOpticalFlowPyrLK(reference,current,points,initial,**options)
    reference_points=empty;current_points=empty;ids=[];stage={}
    if moved is not None and status is not None:
        valid=status.ravel().astype(bool)&_allowed(moved.reshape(-1,2),current_mask)
        original_ids=np.flatnonzero(valid);originals=points[valid];moved=moved[valid]
        stage['forward_flow_reference']=_histogram(pool[original_ids],width,height)
        if len(moved):
            backward,back_status,_=cv2.calcOpticalFlowPyrLK(current,reference,moved,originals.copy(),**options)
            if backward is not None and back_status is not None:
                good=(back_status.ravel().astype(bool)&np.isfinite(backward.reshape(-1,2)).all(axis=1)
                      &(np.linalg.norm(backward.reshape(-1,2)-originals.reshape(-1,2),axis=1)<=CONTRACT['lk_forward_backward_px']))
                ids=original_ids[good];reference_points=pool[ids];current_points=moved.reshape(-1,2)[good]/scale
    audit=fitted(reference_points,current_points,ids,'descriptor_seeded_lk',stage,
                 linear=np.diag(scale)@forward[:2,:2]@np.diag(1/scale))
    audit['seed']=seed_audit;attempts.append(audit)
    return attempts



def _empty(frame,compact,reason):
    return {'frame':frame,'time_s':compact['timestamps_s'][frame],'status':'unavailable',
        'current_to_reference':None,'method':None,'attempts':[],'exclusion_boxes':_exclusions(compact,frame),'reason':reason}


def _worker(pipe,video,compact,reference_frame):
    capture=None
    try:
        cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);cv2.setRNGSeed(1729)
        width,height=compact['width'],compact['height'];size,scale=_analysis_shape(width,height)
        capture=cv2.VideoCapture(video)
        reference=None
        for frame in range(reference_frame+1):
            ok,image=capture.read()
            if not ok:
                raise ValueError('Could not decode the fixed reference frame')
            if image.shape[:2]!=(height,width):
                raise ValueError('Decoded source dimensions changed')
            if frame==reference_frame:
                reference=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),size,interpolation=cv2.INTER_AREA)
        capture.release();capture=cv2.VideoCapture(video)
        prepared=_prepare(reference,_mask(size,scale,_exclusions(compact,reference_frame)))
        pipe.send(('reference_pool',(prepared['pool']/scale).tolist()))
        count=len(compact['timestamps_s'])
        for frame in range(count):
            ok,image=capture.read()
            if not ok or image.shape[:2]!=(height,width):
                raise ValueError('Decoded source frame count/dimensions changed')
            stamp=capture.get(cv2.CAP_PROP_POS_MSEC)/1000
            if not math.isfinite(stamp) or not math.isclose(stamp,compact['timestamps_s'][frame],rel_tol=0,abs_tol=CONTRACT['frame_clock_tolerance_seconds']):
                raise ValueError('Decoded source timestamps differ from saved evidence')
            current=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),size,interpolation=cv2.INTER_AREA)
            row=_empty(frame,compact,'background_registration_failed')
            cv2.setRNGSeed(1729+frame)
            row['attempts']=_pair(prepared,current,_mask(size,scale,row['exclusion_boxes']),scale,width,height,identity=frame==reference_frame)
            passed=next((audit for audit in row['attempts'] if audit['status']=='registered'),None)
            if passed:
                row.update(status='registered',current_to_reference=passed['candidate_current_to_reference'],
                           method='reference' if frame==reference_frame else passed['method'],reason=None)
            pipe.send(('frame',row))
        if capture.read()[0]:
            raise ValueError('Decoded source has unreported suffix frames')
        after=digest(video)
        if after!=compact['source_sha256']:
            raise ValueError('Source video changed during registration')
        pipe.send(('done',{'worker_complete':True,'decoded_frames':count,'source_hash_after':after,
                           'analysis_size':list(size),'analysis_scale_xy':scale.tolist()}))
    except Exception as error:
        pipe.send(('error',type(error).__name__,str(error)[:300]))
    finally:
        if capture is not None:
            capture.release()
        pipe.close()


def register_video(video,evidence,*,reference_frame=40):
    import json
    compact=_compact(evidence);count=len(compact['timestamps_s'])
    _int(reference_frame,0,count-1)
    if reference_frame!=max(obj['frame'] for obj in compact['initialization']):
        raise ValueError('Reference must equal the latest explicitly authored initialization frame')
    video=Path(video).resolve()
    if not video.is_file() or video.stat().st_size>CONTRACT['maximum_video_bytes'] or digest(video)!=compact['source_sha256']:
        raise ValueError('Source video must match the bounded submitted-video hash')
    report={'schema_version':1,'profile':'fixed-reference-registration-v1','source_sha256':compact['source_sha256'],
        'evidence_sha256':canonical_hash(compact),'width':compact['width'],'height':compact['height'],
        'timestamps_s':compact['timestamps_s'],'reference_frame':reference_frame,'contract':deepcopy(CONTRACT),
        'contract_sha256':canonical_hash(CONTRACT),'reference_pool':[],'frames':[_empty(i,compact,'worker_not_completed') for i in range(count)],
        'ready':False,'status':'unavailable','reasons':[],'provenance':{'worker_complete':False,'decoded_frames':0,
            'source_hash_after':None,'fixed_reference_no_composition':True,'manual_labels_used':False,
            'scope':'Candidate background outside supplied foreground exclusions; staticness/depth and calibration are not independently established'}}
    context=multiprocessing.get_context('spawn');receive,send=context.Pipe(duplex=False)
    process=context.Process(target=_worker,args=(send,str(video),compact,reference_frame),daemon=True)
    started=time.monotonic()
    retained_bytes=len(json.dumps(report).encode())
    try:
        process.start();send.close()
        while True:
            remaining=CONTRACT['deadline_seconds']-(time.monotonic()-started)
            if remaining<=0 or not receive.poll(remaining):
                report['reasons'].append('hard_worker_deadline_exceeded');break
            try:
                event=receive.recv()
            except EOFError:
                report['reasons'].append('worker_eof_without_completion');break
            if event[0]=='reference_pool':
                report['reference_pool']=event[1]
                retained_bytes+=len(json.dumps(event[1]).encode())
            elif event[0]=='frame':
                row=event[1];size=len(json.dumps(row).encode())
                previous_size=len(json.dumps(report['frames'][row['frame']]).encode())
                if size>CONTRACT['maximum_ipc_event_bytes'] or retained_bytes+size-previous_size>CONTRACT['maximum_report_bytes']-10000:
                    report['reasons'].append('bounded_registration_output_exceeded');break
                retained_bytes+=size-previous_size
                report['frames'][row['frame']]=row
            elif event[0]=='done':
                report['provenance'].update(event[1]);break
            elif event[0]=='error':
                report['reasons'].append(event[1]+': '+event[2]);break
            else:
                report['reasons'].append('unknown_worker_message');break
    finally:
        receive.close();send.close()
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill();process.join(timeout=1)
            process.close()
    report['provenance']['seconds']=time.monotonic()-started
    failures=[row['frame'] for row in report['frames'][reference_frame:] if row['status']!='registered']
    if failures:
        report['reasons'].append('initialized_suffix_has_unavailable_frames')
    report['ready']=report['provenance']['worker_complete'] and not report['reasons'] and not failures
    report['status']='ready' if report['ready'] else 'unavailable'
    return validate_registration(report)


def _validate_attempts(attempts,pool,width,height):
    methods=[audit.get('method') for audit in attempts]
    if methods not in ([],['orb'],CONTRACT['method_order']):
        raise ValueError('Registration method order conflicts')
    if len(attempts)==2 and attempts[0]['status']=='registered':
        raise ValueError('Fallback cannot follow an accepted first method')
    partitions=_fit_partition(pool,width,height)
    for audit in attempts:
        if audit.get('status') not in ('registered','unavailable'):
            raise ValueError('Unknown attempt status')
        points=audit['correspondences']
        _int(len(points),0,CONTRACT['maximum_correspondences'])
        _int(audit['matched_count'],len(points),CONTRACT['maximum_features'])
        if audit['status']=='registered' and audit.get('reason') is not None:
            raise ValueError('Accepted attempt has a rejection reason')
        ids=[_int(p['reference_feature_id'],0,len(pool)-1) for p in points]
        if len(ids)!=len(set(ids)):
            raise ValueError('Reference feature identities must be unique')
        ref=np.array([p['reference_xy'] for p in points],dtype=float).reshape(-1,2)
        cur=np.array([p['current_xy'] for p in points],dtype=float).reshape(-1,2)
        if (not np.isfinite(ref).all() or not np.isfinite(cur).all() or np.any(cur<0)
                or np.any(cur>np.array([width,height])) or not np.array_equal(ref,pool[ids])):
            raise ValueError('Correspondences conflict with immutable reference pool')
        if len(set(map(tuple,np.round(cur,4))))!=len(points):
            raise ValueError('Duplicate current correspondences')
        fit=partitions[ids];held=~fit
        if any(p['partition']!=('fit' if fit[i] else 'withheld') for i,p in enumerate(points)):
            raise ValueError('Fit/withheld split changed after matching')
        inliers=[p['fit_inlier'] for p in points if p['partition']=='fit']
        if any(type(v) is not bool for v in inliers) or any(p['fit_inlier'] is not None for p in points if p['partition']=='withheld'):
            raise ValueError('Invalid fit inlier flags')
        expected={'fit_count':int(sum(fit)),'withheld_count':int(sum(held)),
                  'fit_inlier_count':sum(inliers),'fit_outlier_count':len(inliers)-sum(inliers)}
        if any(type(audit[key]) is not int or audit[key]!=value for key,value in expected.items()):
            raise ValueError('Attempt support count conflicts')
        spreads={'fit_spread':_spread(cur[fit],width,height),'withheld_source_spread':_spread(cur[held],width,height),
                 'withheld_reference_spread':_spread(ref[held],width,height)}
        if any(audit[key]!=value for key,value in spreads.items()):
            raise ValueError('Attempt spatial support conflicts')
        stages=audit['stages']
        allowed={'pre_appearance_reference','pre_appearance_current','reference_detected','deduplicated_reference','deduplicated_current','reference_pool','current_detected','forward_descriptor_reference',
                 'forward_flow_reference','matched_reference','matched_current','retained_reference','retained_current'}
        if set(stages)-allowed:
            raise ValueError('Unknown stage histogram')
        _int(stages['reference_detected']['count'],len(pool),CONTRACT['maximum_features'])
        for key,value in stages.items():
            if set(value)!={'count','cells'} or len(value['cells'])!=16:
                raise ValueError('Malformed bounded stage histogram')
            _int(value['count'],0,CONTRACT['maximum_features'])
            for cell in value['cells']:
                _int(cell,0,CONTRACT['maximum_features'])
            if sum(value['cells'])!=value['count']:
                raise ValueError('Stage histogram count conflicts')
        expected_stages={'reference_pool':_histogram(pool,width,height),
                         'retained_reference':_histogram(ref,width,height),'retained_current':_histogram(cur,width,height)}
        if any(stages.get(key)!=value for key,value in expected_stages.items()):
            raise ValueError('Retained or reference histogram conflicts')
        if any(stages[key]['count']!=audit['matched_count'] for key in ('matched_reference','matched_current')):
            raise ValueError('Matched stage count conflicts')
        rejected=audit.get('appearance_rejections')
        if not isinstance(rejected,dict) or set(rejected)!={'support','texture','correlation','ambiguous'}:
            raise ValueError('Malformed appearance rejection audit')
        for value in rejected.values():
            _int(value,0,CONTRACT['maximum_features'])
        if (stages['pre_appearance_reference']['count']!=stages['pre_appearance_current']['count']
                or sum(rejected.values())!=stages['pre_appearance_reference']['count']-audit['matched_count']):
            raise ValueError('Appearance filter counts conflict')
        for point in points:
            values=point.get('appearance')
            if not isinstance(values,list) or len(values)!=2:
                raise ValueError('Missing per-point appearance evidence')
            _finite(values[0],CONTRACT['minimum_patch_correlation'],1.)
            _finite(values[1],CONTRACT['minimum_patch_margin'],2.)
        for name in ('reference','current'):
            for earlier,later in (('pre_appearance_','matched_'),('matched_','deduplicated_'),('deduplicated_','retained_')):
                if any(a>b for a,b in zip(stages[later+name]['cells'],stages[earlier+name]['cells'])):
                    raise ValueError('Retained histogram exceeds preceding support')
        if stages['deduplicated_reference']['count']!=stages['deduplicated_current']['count']:
            raise ValueError('Deduplicated pair counts conflict')
        matrix=audit['candidate_current_to_reference']
        residual_keys=('heldout_rmse_px','heldout_median_px','heldout_p95_px','heldout_max_px')
        if matrix is None:
            if any(p['error_px'] is not None for p in points) or any(audit[key] is not None for key in residual_keys):
                raise ValueError('Missing transform cannot have residuals')
        else:
            matrix=np.asarray(matrix,dtype=float)
            if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or not sum(held):
                raise ValueError('Malformed candidate transform')
            errors=np.linalg.norm(cur@matrix[:2,:2].T+matrix[:2,2]-ref,axis=1)
            if any(type(p['error_px']) not in (int,float) or not math.isclose(p['error_px'],error,rel_tol=1e-9,abs_tol=1e-9)
                   for p,error in zip(points,errors)):
                raise ValueError('Per-correspondence residual audit conflicts')
            expected_values=(np.sqrt(np.mean(errors[held]**2)),np.median(errors[held]),np.percentile(errors[held],95),max(errors[held]))
            if any(type(audit[key]) not in (int,float) or not math.isclose(audit[key],value,rel_tol=1e-9,abs_tol=1e-9)
                   for key,value in zip(residual_keys,expected_values)):
                raise ValueError('Attempt residual summary conflicts')
        if audit['method']=='descriptor_seeded_lk':
            seed=audit.get('seed');orb=attempts[0]
            if seed is None:
                if points or audit['status']!='unavailable' or audit['reason']!='descriptor_training_seed_unavailable' or _seed_supported(orb,width,height):
                    raise ValueError('Missing descriptor seed conflicts')
            elif (not _seed_supported(orb,width,height) or seed.get('current_to_reference')!=orb['candidate_current_to_reference']
                    or seed.get('training_reference_feature_ids')!=[p['reference_feature_id'] for p in orb['correspondences'] if p['partition']=='fit']):
                raise ValueError('Seed must use only the preceding ORB training fit')


def validate_registration(report):
    """Recheck source clocks, similarity bounds and independent validation evidence."""
    import json
    if (not isinstance(report,dict) or type(report.get('schema_version')) is not int or report.get('schema_version')!=1
            or report.get('profile')!='fixed-reference-registration-v1' or report.get('contract')!=CONTRACT
            or report.get('contract_sha256')!=canonical_hash(CONTRACT)):
        raise ValueError('Unsupported fixed-reference registration contract')
    if len(json.dumps(report,allow_nan=False).encode())>CONTRACT['maximum_report_bytes']:
        raise ValueError('Registration report exceeds its byte bound')
    _sha(report['source_sha256']);_sha(report['evidence_sha256'])
    width,height=report['width'],report['height'];_int(width,16,4096);_int(height,16,4096)
    if width*height>CONTRACT['maximum_area']:
        raise ValueError('Registration image area exceeds bound')
    if report['provenance'].get('fixed_reference_no_composition') is not True or report['provenance'].get('manual_labels_used') is not False:
        raise ValueError('Registration provenance must preserve the fixed-reference/independent-label boundary')
    raw_pool=report.get('reference_pool',[])
    if not isinstance(raw_pool,list) or len(raw_pool)>CONTRACT['maximum_features']:
        raise ValueError('Invalid immutable reference feature pool')
    for point in raw_pool:
        if not isinstance(point,list) or len(point)!=2:
            raise ValueError('Invalid immutable reference feature pool coordinates')
        _finite(point[0],0,width);_finite(point[1],0,height)
    pool=np.asarray(raw_pool,dtype=float).reshape(-1,2)
    if (len(pool)>CONTRACT['maximum_features'] or not np.isfinite(pool).all() or np.any(pool<0)
            or np.any(pool>np.array([width,height])) or len(set(map(tuple,np.round(pool,4))))!=len(pool)):
        raise ValueError('Invalid immutable reference feature pool')
    times=report['timestamps_s'];count=len(times);_int(count,1,500)
    reference=_int(report['reference_frame'],0,count-1)
    if len(report['frames'])!=count:
        raise ValueError('Registration must retain every source frame')
    previous=-1
    for frame,(stamp,row) in enumerate(zip(times,report['frames'])):
        _finite(stamp,0,3600)
        if stamp<=previous or type(row['frame']) is not int or row['frame']!=frame or row['time_s']!=stamp:
            raise ValueError('Registration frame/time mismatch')
        previous=stamp
        if row['status'] not in ('registered','unavailable') or len(row['attempts'])>2:
            raise ValueError('Unknown registration status/attempt count')
        _validate_attempts(row['attempts'],pool,width,height)
        if row['status']=='unavailable':
            if any(a['status']=='registered' for a in row['attempts']):
                raise ValueError('Unavailable row has accepted audit')
            if row['current_to_reference'] is not None or row['method'] is not None:
                raise ValueError('Rejected registration cannot publish a transform')
            continue
        if not _matrix_bounds(row['current_to_reference'],width,height):
            raise ValueError('Accepted transform is not a bounded similarity')
        if frame==reference and row['current_to_reference']!=np.eye(3).tolist():
            raise ValueError('Reference frame must preserve exact identity coordinates')
        accepted=[audit for audit in row['attempts'] if audit['status']=='registered']
        if len(accepted)!=1 or accepted[0]['candidate_current_to_reference']!=row['current_to_reference']:
            raise ValueError('Accepted transform lacks a unique supporting audit')
        audit=accepted[0]
        if audit['method'] not in CONTRACT['method_order'] or row['method']!=('reference' if frame==reference else audit['method']):
            raise ValueError('Registration method audit conflicts')
        quality=audit['image_quality']
        reference_quality=_finite(quality['reference_laplacian_variance'],CONTRACT['minimum_laplacian_variance'],1e9)
        current_quality=_finite(quality['current_laplacian_variance'],CONTRACT['minimum_laplacian_variance'],1e9)
        if current_quality/reference_quality<CONTRACT['minimum_laplacian_variance_ratio']:
            raise ValueError('Accepted registration fails blur/texture quality')
        points=audit['correspondences']
        if len(points)>CONTRACT['maximum_correspondences']:
            raise ValueError('Correspondence bound exceeded')
        ref=np.array([point['reference_xy'] for point in points]);current=np.array([point['current_xy'] for point in points])
        if ref.shape!=(len(points),2) or current.shape!=ref.shape or not np.isfinite(ref).all() or not np.isfinite(current).all():
            raise ValueError('Invalid correspondence coordinates')
        if np.any(ref<0) or np.any(current<0) or np.any(ref>np.array([width,height])) or np.any(current>np.array([width,height])):
            raise ValueError('Correspondence lies outside source image')
        if len(set(map(tuple,np.round(ref,4))))!=len(points) or len(set(map(tuple,np.round(current,4))))!=len(points):
            raise ValueError('Duplicate correspondences cannot count as independent support')
        fit=np.array([point['partition']=='fit' for point in points]);held=~fit
        if any(point['partition'] not in ('fit','withheld') for point in points) or not np.array_equal(fit,_fit_partition(pool,width,height)[[p['reference_feature_id'] for p in points]]):
            raise ValueError('Fit/withheld split changed after fitting')
        inliers=[point['fit_inlier'] for point in points if point['partition']=='fit']
        if (any(type(value) is not bool for value in inliers) or any(point['fit_inlier'] is not None for point in points if point['partition']=='withheld')
                or sum(fit)<CONTRACT['minimum_fit_count'] or sum(held)<CONTRACT['minimum_withheld_count']
                or sum(inliers)<CONTRACT['minimum_fit_inliers'] or sum(inliers)/sum(fit)<CONTRACT['minimum_fit_inlier_fraction']):
            raise ValueError('Registration lacks positive independent support')
        spreads={'fit_spread':_spread(current[fit],width,height),'withheld_source_spread':_spread(current[held],width,height),
                 'withheld_reference_spread':_spread(ref[held],width,height)}
        if not all(_spread_ok(value) for value in spreads.values()):
            raise ValueError('Independent correspondence spread is insufficient')
        if any(audit[key]!=value for key,value in spreads.items()):
            raise ValueError('Reported spatial support conflicts with retained matches')
        matrix=np.asarray(row['current_to_reference']);errors=np.linalg.norm(current@matrix[:2,:2].T+matrix[:2,2]-ref,axis=1)
        if any(type(point['error_px']) not in (int,float) or not math.isclose(point['error_px'],error,rel_tol=1e-9,abs_tol=1e-9)
               for point,error in zip(points,errors)):
            raise ValueError('Per-correspondence residual audit conflicts')
        expected={'fit_count':int(sum(fit)),'withheld_count':int(sum(held)),'fit_inlier_count':sum(inliers),'fit_outlier_count':len(inliers)-sum(inliers),
            'heldout_rmse_px':float(np.sqrt(np.mean(errors[held]**2))),'heldout_median_px':float(np.median(errors[held])),
            'heldout_p95_px':float(np.percentile(errors[held],95)),'heldout_max_px':float(max(errors[held]))}
        for key,value in expected.items():
            if type(audit.get(key)) not in (int,float) or not math.isclose(audit[key],value,rel_tol=1e-9,abs_tol=1e-9):
                raise ValueError('Registration residual/support audit conflicts')
        for key,limit in (('heldout_rmse_px','maximum_withheld_rmse_source_px'),('heldout_median_px','maximum_withheld_median_source_px'),
                          ('heldout_p95_px','maximum_withheld_p95_source_px'),('heldout_max_px','maximum_withheld_error_source_px')):
            if expected[key]>CONTRACT[limit]:
                raise ValueError('Independent withheld residual gate failed')
    completed=(report['provenance'].get('worker_complete') is True and report['provenance'].get('decoded_frames')==count
               and report['provenance'].get('source_hash_after')==report['source_sha256'])
    ready=completed and not report['reasons'] and all(row['status']=='registered' for row in report['frames'][reference:])
    if report['ready'] is not ready or report['status']!=('ready' if ready else 'unavailable'):
        raise ValueError('Registration readiness conflicts with complete suffix support')
    return deepcopy(report)
