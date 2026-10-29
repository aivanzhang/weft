"""Inspect both construction streams once, preserving full-rate corners and clock evidence."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json


ROOT=Path(__file__).resolve().parents[2]
FULL=ROOT/'artifacts/freemocap-full-source-01'
TEST=ROOT/'artifacts/freemocap-test-source-01'
OFFSETS=[-2,-1,0,1,2]
SNAPSHOTS=[0,165,275,550,825,1107]


def read(path):return json.loads(path.read_text())


def construction(root):
    report=read(root/'report.json');manifest=read(root/'manifest.json')
    if digest(root/'report.json')!=manifest['files']['report.json']:raise ValueError('Source report changed')
    rows=[row for row in report['extracted'] if row['role']=='construction']
    if len(rows)!=2:raise ValueError('Keep exactly two construction cameras')
    for row in rows:
        if digest(root/row['file'])!=manifest['files'][row['file']]:raise ValueError('Original camera bytes changed')
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    out=parser.parse_args().out.resolve()
    if out.parent!=ROOT/'artifacts':raise ValueError('Use a new top-level artifact directory')
    full,test=construction(FULL),construction(TEST)
    if any(Path(a['member']).name!=Path(b['member']).name for a,b in zip(full,test)):
        raise ValueError('Named source cameras differ')
    out.mkdir(exist_ok=False);shutil.copyfile(__file__,out/Path(__file__).name)
    contract={'fps':30,'full_frames':1108,'test_fps':6,'test_frames':222,'prefix_frames':list(range(500)),
        'board_squares':[7,5],'square_length_m':.058,'marker_length_ratio':.8,'dictionary':'DICT_4X4_250',
        'phase_offsets':OFFSETS,'phase_reference':'Full frame 5*test_frame + offset; no source retiming.',
        'phase_common_cohort':list(range(1,221)),'thumbnail_size':[180,320],
        'maximum_workers':2,'maximum_seconds_per_view':60,'maximum_prefix_corner_detections_per_view':500,
        'snapshots':SNAPSHOTS,'reserved_camera_opened':False,'human_pose_inference':False,'paid_calls':0}
    write_json(out/'pre-run.json',{'created_at':local_time(),'contract':contract,
        'source_manifest_sha256':digest(FULL/'manifest.json'),'test_manifest_sha256':digest(TEST/'manifest.json'),
        'source_rows':full,'test_rows':test,
        'subsequent_calibration_limits_from':'artifacts/freemocap-board-check-01/pre-run.json',
        'original_calibration_check_sha256':digest(ROOT/'artifacts/freemocap-board-check-01/pre-run.json')})
    cv2.setNumThreads(1);started=time.perf_counter()
    def prepare(index):
        tick=time.perf_counter();cache=[];test_times=[];cap=cv2.VideoCapture(str(TEST/test[index]['file']))
        try:
            if cap.get(cv2.CAP_PROP_FPS)!=6:raise ValueError('Test frame rate changed')
            for frame in range(223):
                ok,bgr=cap.read()
                if not ok:break
                if frame==222 or bgr.shape[:2]!=(1280,720):raise ValueError('Test source extent changed')
                stamp=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                if abs(stamp-frame/6)>1e-7:raise ValueError('Test source clock changed')
                cache.append(cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),(180,320),interpolation=cv2.INTER_AREA));test_times.append(stamp)
            if len(cache)!=222:raise ValueError('Test source coverage changed')
        finally:cap.release()
        cache=np.asarray(cache,dtype=np.int16);scores=np.full((222,5),np.nan);times=[];detections=[];snapshots=[]
        board=cv2.aruco.CharucoBoard((7,5),.058,.0464,cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250))
        detector=cv2.aruco.CharucoDetector(board);cap=cv2.VideoCapture(str(FULL/full[index]['file']))
        try:
            if cap.get(cv2.CAP_PROP_FPS)!=30:raise ValueError('Full source frame rate differs from its plan')
            for frame in range(1109):
                ok,bgr=cap.read()
                if not ok:break
                if frame==1108 or bgr.shape[:2]!=(1280,720):raise ValueError('Full source extent differs from its plan')
                stamp=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                if abs(stamp-frame/30)>1e-7:raise ValueError('Full source clock changed')
                times.append(stamp);gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
                thumbnail=cv2.resize(gray,(180,320),interpolation=cv2.INTER_AREA).astype(np.int16)
                for column,offset in enumerate(OFFSETS):
                    if (frame-offset)%5==0 and 0<=(frame-offset)//5<222:
                        sample=(frame-offset)//5;scores[sample,column]=float(np.abs(thumbnail-cache[sample]).mean())
                corners=ids=None
                if frame<500:
                    corners,ids,_,marker_ids=detector.detectBoard(gray)
                    points=[] if ids is None else [{'id':int(identifier),'pixel':point.tolist()} for identifier,point in zip(ids.ravel(),corners.reshape(-1,2))]
                    if len({p['id'] for p in points})!=len(points) or any(not 0<=p['id']<24 for p in points):
                        raise ValueError('Detected board IDs exceed the declared board')
                    detections.append({'frame':frame,'time_s':stamp,'source_sha256':full[index]['sha256'],
                        'corner_count':len(points),'corners':points,'marker_ids':[] if marker_ids is None else marker_ids.ravel().tolist()})
                    if frame==499:write_json(out/f'view-{index+1}-detections.json',detections)
                if frame in SNAPSHOTS:
                    name=f'view-{index+1}-source-{frame:04d}.png'
                    if not cv2.imwrite(str(out/name),bgr):raise RuntimeError('Source PNG encoder unavailable')
                    snapshots.append({'frame':frame,'time_s':stamp,'file':name,'sha256':digest(out/name)})
                    if ids is not None:
                        cv2.aruco.drawDetectedCornersCharuco(bgr,corners,ids)
                        if not cv2.imwrite(str(out/f'view-{index+1}-board-{frame:04d}.png'),bgr):raise RuntimeError('Board overlay encoder unavailable')
                if time.perf_counter()-tick>60:raise TimeoutError('Construction evidence preparation exceeds its view envelope')
            if len(times)!=1108 or len(detections)!=500:raise ValueError('Incomplete full construction source')
        finally:cap.release()
        common=scores[1:221]
        if not np.isfinite(common).all():raise ValueError('Phase comparison loses part of its common cohort')
        summaries=[{'full_frame_offset':offset,'available_test_slots':int(np.isfinite(scores[:,i]).sum()),
            'common_cohort':220,'common_mean_absolute_gray_error':float(common[:,i].mean()),
            'common_median_absolute_gray_error':float(np.median(common[:,i])),
            'common_lowest_error_count':int((np.argmin(common,axis=1)==i).sum())} for i,offset in enumerate(OFFSETS)]
        phase_rows=[{'test_frame':i,'test_time_s':test_times[i],'candidates':[
            {'full_frame':5*i+offset,'offset':offset,'gray_mae':float(scores[i,c]) if np.isfinite(scores[i,c]) else None} for c,offset in enumerate(OFFSETS)]} for i in range(222)]
        write_json(out/f'view-{index+1}-all-phase-slots.json',phase_rows)
        result={'view':index+1,'source':full[index],'source_frames':1108,'source_times_s':times,'fps':30,'width':720,'height':1280,
            'test_source':test[index],'test_frames':222,'corner_detection_slots':500,'snapshots':snapshots,
            'phase_summaries':summaries,'runtime_seconds':time.perf_counter()-tick,'reserved_camera_opened':False}
        write_json(out/f'view-{index+1}-report.json',result);return result
    with ThreadPoolExecutor(max_workers=2) as pool:views=list(pool.map(prepare,(0,1)))
    report={'created_at':local_time(),'status':'full_construction_evidence_prepared','contract':contract,'views':views,
        'encoded_clocks_equal':views[0]['source_times_s']==views[1]['source_times_s'],
        'runtime_seconds':time.perf_counter()-started,'newly_decoded_full_frames':2216,'newly_decoded_test_frames':444,
        'new_corner_detection_slots':1000,'reserved_camera_opened':False,'human_pose_inference':False,'paid_calls':0,
        'limitations':['Photometric correspondence is not proof of physical synchronization between cameras.',
            'Both releases are the same named recording; increased sampling is not independent video diversity.',
            'No time correction, calibration adjustment, ground inference or body construction is applied.']}
    write_json(out/'report.json',report)
    print(json.dumps({'status':report['status'],'runtime_seconds':report['runtime_seconds'],
        'phase_summaries':{view['view']:view['phase_summaries'] for view in views},'verification':seal_suite(out)}))


if __name__=='__main__':main()
