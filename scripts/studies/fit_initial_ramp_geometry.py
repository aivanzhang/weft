"""Fit initial first-trial silhouettes without using later motion or repeat evidence."""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json
from weft.pipeline import _implementation_files
from weft.primitive_geometry import conditional_box_depth, fit_initial_width, silhouette_overlap
from weft.spatial_rigid import corners, validate_scene


ROOT=Path(__file__).resolve().parents[2]
MATERIAL=ROOT/'artifacts/spatial-material-fit-01'
MASKS=ROOT/'artifacts/meta-phys101-ramp-masks-01'
SOURCE=ROOT/'artifacts/phys101-ramp-source-01/member-01-Camera_1.mp4'
OBJECTS={'sam-1':'wood-block','sam-2':'green-box'}
CONTRACT={
    'scope':'Initial-frame construction fit, not independent accuracy or recovered physical geometry',
    'fit_source_frame':8,'source_frames_decoded':9,'width_factors':np.linspace(.6,1.2,31).tolist(),
    'maximum_local_fit_seconds':10,
    'fixed':['camera','object centers','height','depth during width fit','orientation','mass','materials','gravity'],
    'objective':'Maximum discrete silhouette intersection-over-union at original mask-cell centers',
    'tie_break':'Least change from original width, then smaller factor',
    'depth_hypothesis':'Separately derive wood cuboid depth from provider volume and assumed 9.81 physical gravity',
    'future_observations_used':False,'repeat_observations_used':False,
    'new_model_inference':False,'new_paid_calls':0,'simulation_steps':0,'training':False,
}


def read(path):
    return json.loads(path.read_text())


def verified_input(root,relative):
    path=root/relative
    expected=read(root/'manifest.json')['files'][relative]
    if digest(path)!=expected:
        raise ValueError(f'Input changed: {relative}')
    return path


def projected(obj,matrix):
    return np.column_stack([corners(obj),np.ones(8)])@matrix.T


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    out=parser.parse_args().out.resolve()
    if out.parent!=ROOT/'artifacts':raise ValueError('Use a new top-level artifacts directory')
    frozen=read(verified_input(MATERIAL,'construction-frozen.json'))
    scene_path=verified_input(MATERIAL,f'candidates/{frozen["selected"]}/scene.json')
    if digest(scene_path)!=frozen['scene_sha256']:raise ValueError('Frozen selected scene changed')
    baseline=read(scene_path)
    projection=read(verified_input(MATERIAL,'construction-inputs.json'))['projection']
    if digest(SOURCE)!=projection['source_sha256']:raise ValueError('Source bytes changed')
    matrix=np.asarray(projection['matrix_px'])
    if not np.array_equal(matrix[:,1],np.zeros(2)):
        raise ValueError('This conditional-depth diagnostic requires the declared unobserved depth axis')
    evidence=read(verified_input(MASKS,'mask-evidence.json'))
    rows=[row for row in evidence['observations'] if row['frame']==8]
    if len(rows)!=2 or {row['object_id'] for row in rows}!=OBJECTS.keys():
        raise ValueError('Both complete frame-eight mask slots are required')
    metadata_path=ROOT/'data/general/phys101-ramps.source.json'
    metadata=read(metadata_path)
    if metadata['object']['provider_id']!='w_block_01':raise ValueError('Provider object mismatch')
    out.mkdir(exist_ok=False);(out/'inputs').mkdir();(out/'implementation-source').mkdir()
    shutil.copyfile(__file__,out/Path(__file__).name)
    implementation=_implementation_files()
    for name in ('primitive_geometry.py','spatial_rigid.py'):
        shutil.copyfile(implementation[name],out/'implementation-source'/name)
    write_json(out/'inputs/scene.json',baseline)
    write_json(out/'inputs/projection.json',projection)
    write_json(out/'inputs/source-metadata.json',metadata)
    write_json(out/'inputs/mask-observations.json',rows)
    write_json(out/'pre-run.json',{'created_at':local_time(),'contract':CONTRACT,
        'source_sha256':digest(SOURCE),'selected_scene_sha256':digest(scene_path),
        'mask_manifest_sha256':digest(MASKS/'manifest.json'),
        'material_manifest_sha256':digest(MATERIAL/'manifest.json'),
        'metadata_sha256':digest(metadata_path),
        'implementation':{name:digest(path) for name,path in implementation.items()}})
    started=time.perf_counter();candidate=deepcopy(baseline);fits={};raster_data={}
    for row in rows:
        identifier=OBJECTS[row['object_id']]
        raster_path=verified_input(MASKS,row['raster_path'])
        shutil.copyfile(raster_path,out/'inputs'/f'{identifier}-mask.png')
        with Image.open(raster_path) as image:raster=np.asarray(image).copy()
        if sha256(raster.tobytes()).hexdigest()!=row['raster_sha256']:raise ValueError('Decoded mask identity changed')
        if list(raster.shape)!=row['raster_dimensions']:raise ValueError('Mask dimensions changed')
        box=row['box']['box_xyxy_halfopen'];old=next(obj for obj in baseline['objects'] if obj['id']==identifier)
        before=silhouette_overlap(raster,box,projected(old,matrix))
        candidate,fit=fit_initial_width(candidate,identifier,projection,raster,box,width_factors=CONTRACT['width_factors'])
        fit.update(source_id=row['id'],source_frame=8,source_time_s=row['time_s'],original_overlap=before)
        fits[identifier]=fit;raster_data[identifier]=(raster,box)
    fit_seconds=time.perf_counter()-started
    if fit_seconds>CONTRACT['maximum_local_fit_seconds']:
        raise TimeoutError('Initial silhouette fit exceeded its work envelope')
    (out/'width-only').mkdir();write_json(out/'width-only/scene.json',candidate)
    write_json(out/'width-fits.json',fits)
    conditional=deepcopy(candidate)
    wood=next(obj for obj in conditional['objects'] if obj['id']=='wood-block')
    depth=conditional_box_depth(wood['size'][0],wood['size'][2],
        reported_volume_ml=metadata['object']['source_reported_volume_ml'],
        normalized_gravity=float(np.linalg.norm(conditional['gravity'])))
    old_depth=wood['size'][1];wood['size'][1]=depth['depth_scene_units']
    wood['provenance']['conditional_depth']=depth
    conditional['assumptions'].append('Wood depth is conditional on provider volume, a uniform cuboid and assumed gravity/scale; it is not independently measured.')
    conditional=validate_scene(conditional)
    (out/'conditional-depth').mkdir();write_json(out/'conditional-depth/scene.json',conditional)
    depth.update(original_depth_scene_units=old_depth)
    capture=cv2.VideoCapture(str(SOURCE))
    try:
        for frame in range(9):
            ok,bgr=capture.read()
            if not ok or bgr.shape[:2]!=(1080,1920):raise ValueError('Original frame unavailable')
            if abs(capture.get(cv2.CAP_PROP_POS_MSEC)/1000-frame*1001/30000)>1e-7:
                raise ValueError('Original frame clock changed')
    finally:
        capture.release()
    source=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));source.save(out/'inputs/source-frame-0008.png')
    panel=Image.new('RGB',(1440,740),'#F5EADC');draw=ImageDraw.Draw(panel)
    for row_number,identifier in enumerate(OBJECTS.values()):
        raster,box=raster_data[identifier];left,top,right,bottom=box
        center=[(left+right)/2,(top+bottom)/2];extent=(center[0]-90,center[1]-70,center[0]+90,center[1]+70)
        crop=source.crop(extent).resize((360,280),Image.Resampling.NEAREST)
        origin=(int(round(extent[0])),int(round(extent[1])))
        mask=np.zeros((1080,1920),dtype=np.uint8);mask[top:bottom,left:right]=(raster!=0).astype(np.uint8)*255
        mask_crop=Image.fromarray(mask).crop(extent).resize((360,280),Image.Resampling.NEAREST)
        tinted=Image.blend(crop,Image.new('RGB',crop.size,'#00D7FF'),.4)
        columns=[('Source frame 8',crop),('Saved model mask',Image.composite(tinted,crop,mask_crop))]
        for label,scene in (('Original cuboid',baseline),('Width-fitted cuboid',candidate)):
            image=crop.copy();overlay=ImageDraw.Draw(image)
            obj=next(obj for obj in scene['objects'] if obj['id']==identifier)
            pixels=(projected(obj,matrix)-origin)*2
            hull=cv2.convexHull(pixels.astype(np.float32)).reshape(-1,2).tolist()
            overlay.line([tuple(p) for p in hull+[hull[0]]],fill='#00D7FF',width=3)
            columns.append((label,image))
        for column,(label,image) in enumerate(columns):
            x=360*column;y=30+340*row_number
            draw.text((x+6,y),f'{identifier}: {label}',fill='#214F50');panel.paste(image,(x,y+25))
        fit=fits[identifier]
        draw.text((6,340+340*row_number),f"Mask overlap {fit['original_overlap']['iou']:.3f} -> {fit['selected']['overlap']['iou']:.3f}; width factor {fit['selected']['factor']:.2f}. Depth is unobserved.",fill='#214F50')
    draw.text((6,715),'Initial-frame fit only; no dynamic accuracy or measured 3D geometry. Source: Wu et al., Physics101 (2016).',fill='#214F50')
    panel.save(out/'initial-comparison.png')
    report={'created_at':local_time(),'status':'initial_silhouette_fit_complete_unverified','contract':CONTRACT,
        'fit_seconds':fit_seconds,'fit_results':fits,'conditional_wood_depth':depth,
        'green_depth_identified':False,'source_image_dimensions':[1920,1080],
        'source_frames_decoded':9,'new_paid_calls':0,'simulation_steps':0,'promoted':False,
        'limitations':['A fixed cuboid cannot exactly reproduce a source silhouette under uncertain camera geometry.',
            'Initial projection has a zero depth column; depth alternatives project identically here.',
            'Later rotation, impact, contacts, changed actions and complete trajectories remain unchecked for these new shapes.',
            'Mask overlap is agreement with model output, not independently measured geometry.',
            'No future frame or repeat observation drives this fit; repeat two remains exposed from its prior evaluation.']}
    write_json(out/'report.json',report)
    (out/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Weft · initial shape comparison</title><style>body{font:17px system-ui;line-height:1.5;background:#F5EADC;color:#214F50;max-width:1440px;margin:30px auto;padding:0 20px}a{color:#965644}img{max-width:100%}</style>
<h1>Fit the opening silhouette</h1><p>Only each cuboid's width changes. Camera, height, position, orientation and physics stay fixed. The outlines are fitted to saved model masks from one frame, not measured physical shapes.</p>
<img src="initial-comparison.png" alt="Wood and green box: source image, saved mask, original cuboid and fitted cuboid">
<p><strong>This camera model does not reveal depth.</strong> A separate wood-depth candidate uses the reported volume and an assumed gravity scale. Neither candidate has been validated against motion or physical interactions.</p>
<p><a href="report.json">Results and limits</a> · <a href="width-fits.json">All 62 width alternatives</a> · <a href="width-only/scene.json">Width-only candidate</a> · <a href="conditional-depth/scene.json">Conditional-depth candidate</a> · <a href="../spatial-ramp-repeat-01/index.html">Existing interactive simulator and failed accuracy check</a></p>
<p>Source: Jiajun Wu et al., Physics101 (2016), local/private research use. No new API call or simulator run.</p></html>''')
    verification=seal_suite(out)
    print(json.dumps({'status':report['status'],'fit_seconds':fit_seconds,
        'objects':{key:{'before':fit['original_overlap']['iou'],'after':fit['selected']['overlap']['iou'],
                       'factor':fit['selected']['factor']} for key,fit in fits.items()},
        'conditional_wood_depth':depth,'verification':verification}))


if __name__=='__main__':
    main()
