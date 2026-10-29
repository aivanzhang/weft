"""Check supplied camera geometry on a fixed calibration prefix, not human poses."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import shutil
import time
import tomllib

import cv2
import numpy as np

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json


ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'artifacts/freemocap-test-source-01'
INSPECTION=ROOT/'artifacts/freemocap-test-inspection-02'
DEFINITION=ROOT/'artifacts/freemocap-board-definition-01'
CONTRACT={
    'scope':'Source-board consistency check of supplied camera geometry; not anatomical or ground validation',
    'construction_views':[0,1],'prefix_frames':list(range(100)),'board_squares':[7,5],
    'square_length_m':.058,'marker_length_ratio':.8,'dictionary':'DICT_4X4_250',
    'minimum_common_corners':8,'minimum_compared_poses':10,
    'camera0_fit_p95_limit_px':2.,'cross_view_p95_limit_px':3.,'cross_view_max_limit_px':10.,
    'edge_length_absolute_error_p95_limit_m':.005,'camera_depth_bounds_m':[.2,10.],
    'maximum_workers':2,'maximum_case_seconds':30,'maximum_total_seconds':60,
    'snapshots':[0,25,50,55,75,99],'camera_refit':False,'temporal_alignment_fit':False,
    'human_pose_inference':False,'reserved_camera_opened':False,'paid_calls':0,
}


def read(path):
    return json.loads(path.read_text())


def verified(root,relative):
    path=root/relative
    if digest(path)!=read(root/'manifest.json')['files'][relative]:raise ValueError(f'Pinned input changed: {relative}')
    return path


def describe(values):
    return {'count':len(values),'mean':float(np.mean(values)) if values else None,
        'p95':float(np.percentile(values,95)) if values else None,'maximum':max(values) if values else None}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    out=parser.parse_args().out.resolve()
    if out.parent!=ROOT/'artifacts':raise ValueError('Use a new top-level artifacts directory')
    source=read(verified(SOURCE,'report.json'));inspection=read(verified(INSPECTION,'report.json'))
    partition=read(verified(SOURCE,'view-partition-before-pixels.json'))
    rows=[r for r in source['extracted'] if r['role']=='construction']
    if len(rows)!=2 or inspection['reserved_view_opened']:raise ValueError('Construction split changed')
    calibration=tomllib.loads(verified(SOURCE,'member-04-freemocap_test_data_camera_calibration.toml').read_text())
    board_info=read(verified(SOURCE,'member-03-charuco_board_info.json'))
    if board_info!={'square_size_mm':58,'num_squares_height':5,'num_squares_width':7}:
        raise ValueError('Board dimensions differ from the frozen source contract')
    definitions=verified(DEFINITION,'charuco_board_definition.py.txt')
    cameras=[]
    for i,row in enumerate(rows):
        path=verified(SOURCE,row['file']);camera=calibration[f'cam_{i}']
        if camera['name']+'.mp4'!=Path(row['member']).name or partition['roles'][row['member']]!='construction':
            raise ValueError('Video and calibration camera IDs differ')
        if camera['size']!=[720,1280]:raise ValueError('Camera dimensions changed')
        cameras.append({'path':path,'K':np.asarray(camera['matrix'],float),'dist':np.asarray(camera['distortions'],float),
            'rvec':np.asarray(camera['rotation'],float),'R':cv2.Rodrigues(np.asarray(camera['rotation'],float))[0],
            't':np.asarray(camera['translation'],float)*.001})
    out.mkdir(exist_ok=False);shutil.copyfile(__file__,out/Path(__file__).name)
    shutil.copyfile(definitions,out/'provider-board-definition.py.txt')
    write_json(out/'pre-run.json',{'created_at':local_time(),'contract':CONTRACT,'opencv_version':cv2.__version__,
        'source_manifest_sha256':digest(SOURCE/'manifest.json'),'inspection_manifest_sha256':digest(INSPECTION/'manifest.json'),
        'definition_manifest_sha256':digest(DEFINITION/'manifest.json'),
        'camera_convention':'Supplied Rodrigues vectors/translation map camera-0 world coordinates to each camera; mm converted to m. No z-up or ground claim.'})
    started=time.perf_counter();cv2.setNumThreads(1)
    def detect(item):
        index,camera=item
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        board=cv2.aruco.CharucoBoard((7,5),.058,.058*.8,dictionary)
        detector=cv2.aruco.CharucoDetector(board)
        cap=cv2.VideoCapture(str(camera['path']));detections=[];tick=time.perf_counter()
        try:
            for frame in CONTRACT['prefix_frames']:
                ok,bgr=cap.read()
                if not ok or bgr.shape[:2]!=(1280,720):raise ValueError('Construction prefix frame unavailable')
                stamp=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                if abs(stamp-frame/6)>1e-7:raise ValueError('Construction source clock changed')
                corners,ids,markers,marker_ids=detector.detectBoard(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY))
                points=[] if ids is None else [{'id':int(identifier),'pixel':p.tolist()} for identifier,p in zip(ids.ravel(),corners.reshape(-1,2))]
                if len({p['id'] for p in points})!=len(points) or any(not 0<=p['id']<24 for p in points):
                    raise ValueError('Detected board IDs violate the source pattern')
                detections.append({'frame':frame,'time_s':stamp,'source_sha256':rows[index]['sha256'],
                    'corner_count':len(points),'corners':points,'marker_ids':[] if marker_ids is None else marker_ids.ravel().tolist()})
                if frame in CONTRACT['snapshots']:
                    original=f'view-{index+1}-source-{frame:04d}.png'
                    if not cv2.imwrite(str(out/original),bgr):raise RuntimeError('Source PNG writer unavailable')
                    if ids is not None:cv2.aruco.drawDetectedCornersCharuco(bgr,corners,ids)
                    if not cv2.imwrite(str(out/f'view-{index+1}-board-{frame:04d}.png'),bgr):raise RuntimeError('Overlay PNG writer unavailable')
                if time.perf_counter()-tick>30:raise TimeoutError('Board detection exceeded its view envelope')
        finally:cap.release()
        write_json(out/f'view-{index+1}-detections.json',detections)
        return detections
    with ThreadPoolExecutor(max_workers=2) as pool:detections=list(pool.map(detect,enumerate(cameras)))
    board=cv2.aruco.CharucoBoard((7,5),.058,.058*.8,cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250))
    model=board.getChessboardCorners().astype(float);comparisons=[];cross_errors=[];fit_errors=[];edge_errors=[]
    a,b=cameras;matrices=[np.column_stack([c['R'],c['t']]) for c in cameras]
    for left,right in zip(*detections):
        if left['frame']!=right['frame'] or left['time_s']!=right['time_s']:raise ValueError('Construction frame pairing changed')
        l={p['id']:p['pixel'] for p in left['corners']};r={p['id']:p['pixel'] for p in right['corners']};shared=sorted(l.keys() & r.keys())
        row={'frame':left['frame'],'time_s':left['time_s'],'common_corners':len(shared),'status':'insufficient_common_board_corners'}
        if len(shared)>=CONTRACT['minimum_common_corners']:
            ids=sorted(l);success,rvec,tvec=cv2.solvePnP(model[ids],np.asarray([l[i] for i in ids]),a['K'],a['dist'],flags=cv2.SOLVEPNP_ITERATIVE)
            if not success:raise ValueError('Planar board pose solve failed')
            projected=cv2.projectPoints(model[ids],rvec,tvec,a['K'],a['dist'])[0].reshape(-1,2)
            residuals=np.linalg.norm(projected-np.asarray([l[i] for i in ids]),axis=1)
            board_camera0=model[shared]@cv2.Rodrigues(rvec)[0].T+tvec.ravel()
            board_world=(board_camera0-a['t'])@a['R']
            other=cv2.projectPoints(board_world,b['rvec'],b['t'],b['K'],b['dist'])[0].reshape(-1,2)
            cross=np.linalg.norm(other-np.asarray([r[i] for i in shared]),axis=1)
            undistorted=[cv2.undistortPoints(np.asarray([data[i] for i in shared],float).reshape(-1,1,2),camera['K'],camera['dist']).reshape(-1,2).T for data,camera in ((l,a),(r,b))]
            homogeneous=cv2.triangulatePoints(*matrices,*undistorted)
            points=(homogeneous[:3]/homogeneous[3]).T
            depths=[(points@c['R'].T+c['t'])[:,2] for c in cameras]
            valid_depth=bool(np.isfinite(points).all() and all((d>=.2).all() and (d<=10).all() for d in depths))
            by_id={identifier:point for identifier,point in zip(shared,points)};edges=[]
            for identifier in shared:
                neighbors=([identifier+1] if identifier%6<5 else [])+[identifier+6]
                for neighbor in neighbors:
                    if neighbor in by_id:
                        length=float(np.linalg.norm(by_id[identifier]-by_id[neighbor]))
                        edges.append({'ids':[identifier,neighbor],'length_m':length,'absolute_error_m':abs(length-.058)})
            row.update(status='compared',positive_bounded_depth=valid_depth,source_ids=shared,
                camera0_fit_errors_px=residuals.tolist(),cross_view_errors_px=cross.tolist(),
                triangulated_corners_m=points.tolist(),edges=edges)
            fit_errors.extend(residuals.tolist());cross_errors.extend(cross.tolist());edge_errors.extend(e['absolute_error_m'] for e in edges)
        comparisons.append(row)
    write_json(out/'all-prefix-comparisons.json',comparisons)
    fit,cross,edges=describe(fit_errors),describe(cross_errors),describe(edge_errors)
    compared=[r for r in comparisons if r['status']=='compared']
    checks={'enough_board_poses':len(compared)>=10,'all_positive_bounded_depth':bool(compared) and all(r['positive_bounded_depth'] for r in compared),
        'camera0_fit_p95':fit['p95'] is not None and fit['p95']<=2,
        'cross_view_p95':cross['p95'] is not None and cross['p95']<=3,
        'cross_view_maximum':cross['maximum'] is not None and cross['maximum']<=10,
        'board_edge_p95':edges['p95'] is not None and edges['p95']<=.005}
    if time.perf_counter()-started>60:raise TimeoutError('Calibration check exceeded its total envelope')
    report={'created_at':local_time(),'status':'board_consistency_passed' if all(checks.values()) else 'board_consistency_failed',
        'checks':checks,'contract':CONTRACT,'compared_poses':len(compared),'unavailable_poses':100-len(compared),
        'source_frames_per_view':100,'camera0_fit_px':fit,'cross_view_error_px':cross,'edge_length_absolute_error_m':edges,
        'runtime_seconds':time.perf_counter()-started,'human_pose_inference':False,'reserved_camera_opened':False,'paid_calls':0,
        'limitations':['Calibration consistency is checked only on visible construction-view board poses.',
            'Shared board/camera estimation is not independent laboratory ground truth or complete synchronization proof.',
            'No camera parameters, distortion, scale, time offset, body pose, floor or thresholds are refitted.',
            'Camera-0 axes are not gravity-aligned; ground orientation must come from additional source evidence.',
            'All unsupported board frames remain in the comparison; the third camera is still pixel-unread.']}
    write_json(out/'report.json',report)
    print(json.dumps({'status':report['status'],'compared_poses':len(compared),'checks':checks,'cross_view_error_px':cross,
        'edge_error_m':edges,'runtime_seconds':report['runtime_seconds'],'verification':seal_suite(out)}))


if __name__=='__main__':main()
