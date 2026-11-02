"""Bounded UV diagnostics and authored normal-only atlases; never rewrite color UV0.

Regions are numeric source primitive/material groups, not anatomy labels. Normal
UV1 and its tangent frames are authored HD data and are not Weft-compatible.
"""

import hashlib
import math
import time

import numpy as np


MAX_VERTICES = 100000
MAX_FACES = 100000
MAX_REGIONS = 4
MAX_ATLAS_FACES = 2048
MAX_PIXELS = 32000000
MAX_SECONDS = 20.0


def _array(value, shape, name, kinds='fiu'):
    if not isinstance(value,np.ndarray) or value.shape!=shape or value.dtype.kind not in kinds:
        raise ValueError('Invalid normal-atlas '+name+' array')
    if value.dtype.kind=='f' and not np.isfinite(value).all():
        raise ValueError('Nonfinite normal-atlas '+name)
    return value


def _inputs(points,faces,uv0,selected,regions):
    if not isinstance(points,np.ndarray) or points.ndim!=2 or not 1<=len(points)<=MAX_VERTICES:
        raise ValueError('Invalid or oversized normal-atlas points')
    if not isinstance(faces,np.ndarray) or faces.ndim!=2 or not 1<=len(faces)<=MAX_FACES:
        raise ValueError('Invalid or oversized normal-atlas faces')
    nv,nf=len(points),len(faces)
    points=_array(points,(nv,3),'points')
    faces=_array(faces,(nf,3),'faces','iu')
    uv0=_array(uv0,(nf,3,2),'UV0')
    selected=_array(selected,(nf,),'selection','b')
    regions=_array(regions,(nf,),'regions','iu')
    if (faces.min()<0 or faces.max()>=nv or np.any(np.abs(points)>10000) or
            np.any(uv0 < -1e-6) or np.any(uv0 > 1+1e-6) or regions.min()<0 or regions.max()>31):
        raise ValueError('Normal-atlas coordinates, indices or region IDs exceed bounds')
    ids=np.unique(regions[selected])
    if not 1<=len(ids)<=MAX_REGIONS:
        raise ValueError('Normal-atlas requires one to four selected material regions')
    return points,faces,uv0,selected,regions,ids


def _area(points):
    return np.linalg.norm(np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0]),axis=1)*.5


def _coverage(points,faces,uvs,ids,size,started,remaining):
    owner=np.full((size,size),-1,dtype=np.int32)
    bary=np.zeros((size,size,2),dtype=np.float32)
    conflicts=np.zeros((size,size),dtype=bool)
    work,overlaps,subpixel,degenerate=0,0,0,0
    for sequence,index in enumerate(ids):
        if sequence%64==0 and time.monotonic()-started>MAX_SECONDS:
            raise RuntimeError('Normal-atlas analysis exceeded its20-second deadline')
        uv=uvs[index].astype(np.float64)
        a,b=uv[1]-uv[0],uv[2]-uv[0]
        determinant=a[0]*b[1]-a[1]*b[0]
        if abs(determinant)<1e-12:
            degenerate+=1
            continue
        low=np.maximum(np.ceil(uv.min(axis=0)*size-.5).astype(int),0)
        high=np.minimum(np.floor(uv.max(axis=0)*size-.5).astype(int),size-1)
        if np.any(high<low):
            subpixel+=1
            continue
        work+=int(np.prod(high-low+1))
        if work>remaining:
            raise ValueError('Normal-atlas aggregate raster work exceeds32000000 pixels')
        hit=False
        for start in range(int(low[1]),int(high[1])+1,32):
            end=min(start+32,int(high[1])+1)
            x=(np.arange(low[0],high[0]+1)+.5)/size-uv[0,0]
            y=(np.arange(start,end)+.5)/size-uv[0,1]
            b1=(x[None,:]*b[1]-y[:,None]*b[0])/determinant
            b2=(y[:,None]*a[0]-x[None,:]*a[1])/determinant
            inside=(b1>=-1e-8)&(b2>=-1e-8)&(b1+b2<=1+1e-8)
            if not inside.any(): continue
            hit=True
            target=owner[start:end,low[0]:high[0]+1]
            old_weights=bary[start:end,low[0]:high[0]+1]
            collision=inside&(target>=0)
            if collision.any():
                old_ids,weights=target[collision],old_weights[collision]
                old=points[faces[old_ids,0]]*(1-weights.sum(axis=1))[:,None]
                old+=points[faces[old_ids,1]]*weights[:,:1]+points[faces[old_ids,2]]*weights[:,1:]
                new=points[faces[index,0]]*(1-b1[collision]-b2[collision])[:,None]
                new+=points[faces[index,1]]*b1[collision,None]+points[faces[index,2]]*b2[collision,None]
                bad=np.max(np.abs(old-new),axis=1)>1e-5
                overlaps+=int(bad.sum())
                cy,cx=np.where(collision)
                conflicts[start+cy[bad],low[0]+cx[bad]]=True
            write=inside&(target<0)
            target[write]=index
            old_weights[write,0],old_weights[write,1]=b1[write],b2[write]
        subpixel+=not hit
    return {'raster_candidates':work,'covered_texels':int(np.count_nonzero(owner>=0)),
        'conflicting_surface_samples':overlaps,'unique_conflicting_texels':int(conflicts.sum()),
        'subpixel_faces':int(subpixel),'degenerate_uv_faces':degenerate}


def analyze_uv_regions(points,faces,uv0,selected,regions,*,resolution=2048,pitch_m=.004):
    """Measure aliasing independently per material; separate normal images remove cross-material aliasing."""
    if type(resolution) is not int or resolution not in (1024,2048):
        raise ValueError('Normal-atlas analysis resolution must be1024 or2048')
    if type(pitch_m) not in (int,float) or not math.isfinite(pitch_m) or not .002<=pitch_m<=.02:
        raise ValueError('Normal-atlas pitch must be2–20mm')
    points,faces,uv0,selected,regions,ids=_inputs(points,faces,uv0,selected,regions)
    started=time.monotonic();work=0;rows=[]
    for region in ids:
        chosen=np.flatnonzero(selected&(regions==region))
        result=_coverage(points,faces,uv0,chosen,resolution,started,MAX_PIXELS-work)
        work+=result['raster_candidates']
        area=float(_area(points[faces[chosen]].astype(np.float64)).sum())
        meters=math.sqrt(area/result['covered_texels']) if result['covered_texels'] and area else None
        pitch_texels=pitch_m/meters if meters else 0
        reasons=[]
        if result['conflicting_surface_samples']:reasons.append('different_surface_uv_aliases')
        if result['degenerate_uv_faces']:reasons.append('degenerate_source_uv')
        if result['subpixel_faces']/len(chosen)>.05:reasons.append('over_five_percent_subpixel_faces')
        if pitch_texels<2:reasons.append('under_two_effective_texels_per_pile_pitch')
        rows.append({'region_id':int(region),'selected_faces':len(chosen),**result,
            'surface_area_m2':area,'effective_meters_per_unique_texel':meters,
            'effective_texels_per_pitch':pitch_texels,'method':'authored_uv1' if reasons else 'keep_uv0',
            'reasons':reasons,'selection_sha256':hashlib.sha256(chosen.astype('<i4').tobytes()).hexdigest()})
    return {'schema_version':1,'resolution':resolution,'pitch_m':pitch_m,'regions':rows,
        'elapsed_seconds':round(time.monotonic()-started,6),'raster_candidates':work,
        'scope':'Per-material alias/density measurements; not visual quality or recovered fur',
        'normal_texture_policy':'One normal image per selected material region',
        'original_uv0_unchanged':True}


def triangle_atlas(points,faces,face_ids,*,resolution=1024,padding_px=2):
    """Pack bounded face-local projections into distinct padded cells without moving geometry."""
    if type(resolution) is not int or resolution not in (1024,2048):
        raise ValueError('Authored atlas resolution must be1024 or2048')
    if type(padding_px) is not int or not 2<=padding_px<=16:
        raise ValueError('Authored atlas needs2–16pixel padding')
    if not isinstance(points,np.ndarray) or points.ndim!=2 or not 1<=len(points)<=MAX_VERTICES:
        raise ValueError('Invalid atlas points')
    if not isinstance(faces,np.ndarray) or faces.ndim!=2 or not 1<=len(faces)<=MAX_FACES:
        raise ValueError('Invalid atlas faces')
    points=_array(points,(len(points),3),'points');faces=_array(faces,(len(faces),3),'faces','iu')
    if not isinstance(face_ids,np.ndarray) or face_ids.ndim!=1 or not 1<=len(face_ids)<=MAX_ATLAS_FACES:
        raise ValueError('Authored atlas exceeds2048 selected faces')
    face_ids=_array(face_ids,(len(face_ids),),'face IDs','iu')
    if np.any(np.abs(points)>10000) or faces.min()<0 or faces.max()>=len(points) or face_ids.min()<0 or face_ids.max()>=len(faces) or not np.all(np.diff(face_ids.astype(np.int64))>0):
        raise ValueError('Atlas needs valid ordered unique face IDs')
    count=len(face_ids);columns=math.ceil(math.sqrt(count));cell=resolution//columns
    usable=cell-2*padding_px-1
    if usable<4:raise ValueError('Authored atlas cannot resolve padded charts')
    result=np.empty((count,3,2),dtype=np.float32);densities=[]
    for ordinal,index in enumerate(face_ids):
        triangle=points[faces[index]].astype(np.float64)
        edges=np.roll(triangle,-1,axis=0)-triangle
        lengths=np.linalg.norm(edges,axis=1)
        normal=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]);area2=float(np.linalg.norm(normal))
        if not math.isfinite(area2) or area2<=1e-12:raise ValueError('Cannot atlas a degenerate geometric triangle')
        basis=edges[int(np.argmax(lengths))]/lengths.max()
        normal/=area2
        flat=np.column_stack(((triangle-triangle[0])@basis,(triangle-triangle[0])@np.cross(normal,basis)))
        extent=float(np.ptp(flat,axis=0).max())
        origin=np.array([ordinal%columns,ordinal//columns])*cell+padding_px+.5
        result[ordinal]=(origin+(flat-flat.min(axis=0))/extent*usable)/resolution
        densities.append(extent/usable)
    return result,{'resolution':resolution,'padding_px':padding_px,'columns':columns,'cell_pixels':cell,
        'faces':count,'max_meters_per_texel':max(densities),'median_meters_per_texel':float(np.median(densities)),
        'uv1_float32_sha256':hashlib.sha256(result.tobytes()).hexdigest(),
        'layout':'One ordered longest-edge local triangle per padded grid cell; no shared chart texels',
        'authored_not_recovered':True,'geometry_changed':False,'uv0_changed':False}


def atlas_tangents(points,faces,face_ids,uv1,corner_normals):
    """Construct finite unit tangent frames for the authored, unshared triangle charts."""
    if not isinstance(points,np.ndarray) or points.ndim!=2 or not 1<=len(points)<=MAX_VERTICES:
        raise ValueError('Invalid authored tangent points')
    if not isinstance(faces,np.ndarray) or faces.ndim!=2 or not 1<=len(faces)<=MAX_FACES:
        raise ValueError('Invalid authored tangent faces')
    if not isinstance(face_ids,np.ndarray) or face_ids.ndim!=1:
        raise ValueError('Invalid authored tangent face IDs')
    points=_array(points,(len(points),3),'points');faces=_array(faces,(len(faces),3),'faces','iu')
    face_ids=_array(face_ids,(len(face_ids),),'face IDs','iu')
    if np.any(np.abs(points)>10000) or faces.min()<0 or faces.max()>=len(points) or not len(face_ids) or face_ids.min()<0 or face_ids.max()>=len(faces):
        raise ValueError('Authored tangent coordinates or indices exceed bounds')
    count=len(face_ids)
    if not 1<=count<=MAX_ATLAS_FACES:raise ValueError('Authored tangent face limit exceeded')
    uv1=_array(uv1,(count,3,2),'UV1')
    normals=_array(corner_normals,(count,3,3),'corner normals').astype(np.float64)
    magnitude=np.linalg.norm(normals,axis=2)
    if np.any(np.abs(magnitude-1)>.01):raise ValueError('Authored tangents require unit source normals')
    normals/=magnitude[:,:,None]
    xyz=points[faces[face_ids]].astype(np.float64)
    a,b=uv1[:,1]-uv1[:,0],uv1[:,2]-uv1[:,0]
    determinant=a[:,0]*b[:,1]-a[:,1]*b[:,0]
    if np.any(np.abs(determinant)<1e-12):raise ValueError('Authored UV1 tangent chart is degenerate')
    e1,e2=xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]
    du=(e1*b[:,1,None]-e2*a[:,1,None])/determinant[:,None]
    dv=(e2*a[:,0,None]-e1*b[:,0,None])/determinant[:,None]
    tangent=np.repeat(du[:,None,:],3,axis=1)
    tangent-=normals*np.sum(tangent*normals,axis=2)[:,:,None]
    size=np.linalg.norm(tangent,axis=2)
    if np.any(size<1e-8):raise ValueError('Source normals cannot support authored tangent frames')
    tangent/=size[:,:,None]
    sign=np.sign(np.sum(np.cross(normals,tangent)*dv[:,None,:],axis=2))
    if np.any(sign==0):raise ValueError('Authored tangent handedness is undefined')
    return np.concatenate((tangent,sign[:,:,None]),axis=2).astype(np.float32)


def plan_normal_atlases(points,faces,uv0,selected,regions,*,source_export_vertices,resolution=2048,
                        atlas_resolution=1024,pitch_m=.004,reserve_vertices=4096):
    """Plan without mutating inputs; fail before geometry can exceed the100k exported limit."""
    if type(source_export_vertices) is not int or not isinstance(points,np.ndarray) or not len(points)<=source_export_vertices<=MAX_VERTICES or type(reserve_vertices) is not int or not 0<=reserve_vertices<=10000:
        raise ValueError('Invalid normal-atlas export budget')
    analysis=analyze_uv_regions(points,faces,uv0,selected,regions,resolution=resolution,pitch_m=pitch_m)
    extra=0;payloads={};pending=[]
    for row in analysis['regions']:
        region=row['region_id'];yes=np.flatnonzero(selected&(regions==region));no=np.flatnonzero(~selected&(regions==region))
        retained=np.unique(faces[yes]);other=np.unique(faces[no])
        material_split=len(np.intersect1d(retained,other,assume_unique=True));extra+=material_split
        atlas_split=0
        if row['method']=='authored_uv1':
            if len(yes)>MAX_ATLAS_FACES:raise ValueError('Aliased material exceeds2048face UV1 repair bound')
            atlas_split=3*len(yes)-len(retained);extra+=atlas_split;pending.append((region,yes))
        row.update(material_split_extra_vertices=material_split,uv1_extra_vertices=atlas_split)
    projected=source_export_vertices+extra+reserve_vertices
    if projected>MAX_VERTICES:raise ValueError('Normal-only atlas exceeds the100000vertex aggregate export budget')
    for region,ids in pending:
        uvs,report=triangle_atlas(points,faces,ids,resolution=atlas_resolution)
        payloads[region]={'face_ids':ids,'uvs':uvs,'report':report}
    analysis.update(atlas_regions=[region for region,_ in pending],source_export_vertices=source_export_vertices,
        additional_export_vertices=extra,reserve_vertices=reserve_vertices,projected_export_vertex_ceiling=projected,
        hd_only=True,weft_appearance_compatible=False,
        limitations=['Normal UV1 and selected tangents are authored, not recovered.',
                    'Weft rejects normal maps and TEXCOORD_1; no importer change is implied.',
                    'This plan establishes no visible softness or animation quality.'])
    return analysis,payloads
