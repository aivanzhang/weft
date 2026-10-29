"""Inspect the two declared construction cameras; never decode the reserved third view."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import shutil
import time
import tomllib

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json
from weft.pipeline import verify_artifacts


ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'artifacts/freemocap-test-source-01'
FRAMES=[0,55,110,165,221]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--reuse-inspection',type=Path)
    args=parser.parse_args();out=args.out.resolve()
    if out.parent!=ROOT/'artifacts':raise ValueError('Use a new top-level artifacts directory')
    original=json.loads((SOURCE/'report.json').read_text())
    if original['status']!='public_sample_acquired_uninspected':raise ValueError('Complete pinned acquisition required')
    rows=[r for r in original['extracted'] if r['role']=='construction']
    if len(rows)!=2:raise ValueError('Exactly two declared construction views required')
    partition=json.loads((SOURCE/'view-partition-before-pixels.json').read_text())
    if any(partition['roles'][r['member']]!='construction' for r in rows):raise ValueError('View split changed')
    for row in rows:
        if digest(SOURCE/row['file'])!=row['sha256']:raise ValueError('Original construction video changed')
    reused=None
    if args.reuse_inspection:
        prior=args.reuse_inspection.resolve();reused=verify_artifacts(prior)
        old_plan=json.loads((prior/'pre-run.json').read_text())
        if old_plan['views']!=rows or old_plan['snapshot_frames']!=FRAMES or old_plan['reserved_view_opened']:
            raise ValueError('Saved inspection inputs differ')
    out.mkdir(exist_ok=False);shutil.copyfile(__file__,out/Path(__file__).name)
    write_json(out/'pre-run.json',{'created_at':local_time(),'source_manifest_sha256':digest(SOURCE/'manifest.json'),
        'partition_sha256':digest(SOURCE/'view-partition-before-pixels.json'),'views':rows,
        'snapshot_frames':FRAMES,'maximum_decoded_frames_per_view':300,'maximum_workers':2,
        'maximum_case_seconds':30,'reserved_view_opened':False,'pose_inference':False,'paid_calls':0,
        'reused_inspection':str(args.reuse_inspection) if reused else None,'reuse_verification':reused})
    cv2.setNumThreads(1);started=time.perf_counter()
    def inspect(item):
        index,row=item;cap=cv2.VideoCapture(str(SOURCE/row['file']));times=[];saved=[];tick=time.perf_counter()
        metadata={'fps':cap.get(cv2.CAP_PROP_FPS),'frame_count_hint':cap.get(cv2.CAP_PROP_FRAME_COUNT),
            'width':int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),'height':int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
        try:
            for frame in range(301):
                ok,bgr=cap.read()
                if not ok:break
                if frame==300 or time.perf_counter()-tick>30:raise ValueError('Construction inspection exceeds its envelope')
                if list(bgr.shape[:2])!=[metadata['height'],metadata['width']]:raise ValueError('Image dimensions changed')
                times.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
                if frame in FRAMES:
                    image=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));name=f'view-{index+1}-frame-{frame:04d}.png';image.save(out/name)
                    saved.append({'frame':frame,'time_s':times[-1],'file':name,'sha256':digest(out/name),'rgb_sha256':sha256(np.asarray(image).tobytes()).hexdigest()})
            if len(times)!=222 or len(saved)!=len(FRAMES):raise ValueError('Complete test video differs from provider frame count')
            if any(b<=a for a,b in zip(times,times[1:])):raise ValueError('Video clock is not strictly increasing')
        finally:cap.release()
        return {'view':index+1,'source':row,'metadata':metadata,'decoded_frames':len(times),'times_s':times,
            'snapshots':saved,'decode_export_seconds':time.perf_counter()-tick}
    if reused:
        views=json.loads((prior/'report.json').read_text())['views']
        for view in views:
            for snapshot in view['snapshots']:shutil.copyfile(prior/snapshot['file'],out/snapshot['file'])
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:views=list(pool.map(inspect,enumerate(rows)))
    panel=Image.new('RGB',(1800,1340),'#F5EADC');draw=ImageDraw.Draw(panel)
    for row_number,view in enumerate(views):
        for i,snapshot in enumerate(view['snapshots']):
            x=360*i;y=670*row_number
            draw.text((x+7,y+7),f"View {row_number+1} | frame {snapshot['frame']} | {snapshot['time_s']:.3f}s",fill='#214F50')
            with Image.open(out/snapshot['file']) as image:
                thumbnail=ImageOps.contain(image,(360,640))
                panel.paste(thumbnail,(x+(360-thumbnail.width)//2,y+25+(640-thumbnail.height)//2))
    panel.save(out/'construction-contact-sheet.png')
    calibration=tomllib.loads((SOURCE/'member-04-freemocap_test_data_camera_calibration.toml').read_text())
    report={'created_at':local_time(),'status':'construction_views_inspected','views':views,
        'board':json.loads((SOURCE/'member-03-charuco_board_info.json').read_text()),'calibration_sections':list(calibration),
        'encoded_clocks_equal':views[0]['times_s']==views[1]['times_s'],
        'reserved_view_opened':False,'reserved_view_decoded_frames':0,'pose_inference':False,'paid_calls':0,
        'runtime_seconds':time.perf_counter()-started,
        'newly_decoded_frames':0 if reused else 444,'reused_decoded_frames':444 if reused else 0,
        'display':'Aspect-preserving portrait thumbnails; supersedes the earlier stretched contact sheet only, not source pixels or decoding.',
        'timing_scope':'Encoded clocks only. Provider says the test sequence is downsampled; physical capture timing is not yet verified.',
        'calibration_scope':'Provider parameters are available but not validated against visible board corners; ground orientation is not assumed.'}
    write_json(out/'report.json',report)
    print(json.dumps({'status':report['status'],'views':[{k:v[k] for k in ('view','metadata','decoded_frames')} for v in views],
        'encoded_clocks_equal':report['encoded_clocks_equal'],'runtime_seconds':report['runtime_seconds'],'verification':seal_suite(out)}))


if __name__=='__main__':main()
