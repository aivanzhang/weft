"""Apply the unchanged board limits to both supplied calibrations on full-rate evidence."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import time
import tomllib

import cv2
import numpy as np

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json


ROOT=Path(__file__).resolve().parents[2]
EVIDENCE=ROOT/'artifacts/freemocap-full-evidence-01'
ORIGINAL=ROOT/'artifacts/freemocap-board-check-01'
SOURCES={'full-supplied':ROOT/'artifacts/freemocap-full-source-01','test-supplied':ROOT/'artifacts/freemocap-test-source-01'}


def read(path):return json.loads(path.read_text())


def verified(root,name):
    path=root/name
    if digest(path)!=read(root/'manifest.json')['files'][name]:raise ValueError('Pinned source changed: '+name)
    return path


def describe(values):
    return {'count':len(values),'mean':float(np.mean(values)) if values else None,
        'p95':float(np.percentile(values,95)) if values else None,'maximum':max(values) if values else None}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    out=parser.parse_args().out.resolve()
    if out.parent!=ROOT/'artifacts':raise ValueError('Use a new top-level artifact directory')
    limits=read(verified(ORIGINAL,'pre-run.json'))['contract']
    evidence=read(verified(EVIDENCE,'report.json'));detections=[read(verified(EVIDENCE,f'view-{i}-detections.json')) for i in (1,2)]
    if evidence['reserved_camera_opened'] or evidence['new_corner_detection_slots']!=1000:
        raise ValueError('Full construction evidence does not match the declared source split')
    for index,view in enumerate(detections):
        if len(view)!=500 or [r['frame'] for r in view]!=list(range(500)):raise ValueError('Full source prefix coverage changed')
        source=evidence['views'][index]['source']['sha256']
        if any(r['source_sha256']!=source or abs(r['time_s']-r['frame']/30)>1e-7 for r in view):
            raise ValueError('Source identity or clock changed')
    board=cv2.aruco.CharucoBoard((7,5),.058,.0464,cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250))
    model=board.getChessboardCorners().astype(float);calibrations={};paths={}
    for name,root in SOURCES.items():
        source=read(verified(root,'report.json'))
        files=[row['file'] for row in source['extracted'] if row['file'].endswith('camera_calibration.toml')]
        if len(files)!=1:raise ValueError('One explicit provider calibration required')
        path=verified(root,files[0]);paths[name]=path
        calibration=tomllib.loads(path.read_text());cameras=[]
        for i in range(2):
            c=calibration[f'cam_{i}']
            if c['name']+'.mp4'!=Path(evidence['views'][i]['source']['member']).name or c['size']!=[720,1280]:
                raise ValueError('Calibration and source camera IDs/dimensions differ')
            cameras.append({'K':np.asarray(c['matrix'],float),'dist':np.asarray(c['distortions'],float),
                'rvec':np.asarray(c['rotation'],float),'R':cv2.Rodrigues(np.asarray(c['rotation'],float))[0],
                't':np.asarray(c['translation'],float)*.001})
        calibrations[name]=cameras
    out.mkdir(exist_ok=False);shutil.copyfile(__file__,out/Path(__file__).name)
    write_json(out/'pre-run.json',{'created_at':local_time(),'unchanged_limits':limits,
        'limits_source_sha256':digest(ORIGINAL/'pre-run.json'),'evidence_manifest_sha256':digest(EVIDENCE/'manifest.json'),
        'calibration_sources':{name:{'path':str(path.relative_to(ROOT)),'sha256':digest(path)} for name,path in paths.items()},
        'source_frames':list(range(500)),'fps':30,'maximum_workers':2,'maximum_seconds':30,
        'scope':'Full-rate construction comparison of two provider-supplied calibrations; no parameter selection, refit or third-view evidence.',
        'new_detection_calls':0,'new_video_frames_decoded':0,'reserved_camera_opened':False,'paid_calls':0})
    started=time.perf_counter()
    def evaluate(item):
        name,cameras=item;a,b=cameras
        matrices=[np.column_stack([c['R'],c['t']]) for c in cameras]
        comparisons=[];fit_errors=[];cross_errors=[];edge_errors=[]
        for left,right in zip(*detections):
            if left['frame']!=right['frame'] or left['time_s']!=right['time_s']:raise ValueError('Paired source clock mismatch')
            l={p['id']:p['pixel'] for p in left['corners']};r={p['id']:p['pixel'] for p in right['corners']};shared=sorted(l.keys() & r.keys())
            row={'frame':left['frame'],'time_s':left['time_s'],'common_corners':len(shared),'status':'insufficient_common_board_corners'}
            if len(shared)>=limits['minimum_common_corners']:
                ids=sorted(l);success,rvec,tvec=cv2.solvePnP(model[ids],np.asarray([l[i] for i in ids]),a['K'],a['dist'],flags=cv2.SOLVEPNP_ITERATIVE)
                if not success:raise ValueError('Board pose solve failed')
                projected=cv2.projectPoints(model[ids],rvec,tvec,a['K'],a['dist'])[0].reshape(-1,2)
                residuals=np.linalg.norm(projected-np.asarray([l[i] for i in ids]),axis=1)
                world=(model[shared]@cv2.Rodrigues(rvec)[0].T+tvec.ravel()-a['t'])@a['R']
                other=cv2.projectPoints(world,b['rvec'],b['t'],b['K'],b['dist'])[0].reshape(-1,2)
                cross=np.linalg.norm(other-np.asarray([r[i] for i in shared]),axis=1)
                rays=[cv2.undistortPoints(np.asarray([data[i] for i in shared],float).reshape(-1,1,2),c['K'],c['dist']).reshape(-1,2).T for data,c in ((l,a),(r,b))]
                homogeneous=cv2.triangulatePoints(*matrices,*rays);points=(homogeneous[:3]/homogeneous[3]).T
                depths=[(points@c['R'].T+c['t'])[:,2] for c in cameras];low,high=limits['camera_depth_bounds_m']
                valid=bool(np.isfinite(points).all() and all((d>=low).all() and (d<=high).all() for d in depths))
                by_id=dict(zip(shared,points));edges=[]
                for identifier in shared:
                    for neighbor in ([identifier+1] if identifier%6<5 else [])+[identifier+6]:
                        if neighbor in by_id:
                            length=float(np.linalg.norm(by_id[identifier]-by_id[neighbor]));error=abs(length-limits['square_length_m'])
                            edges.append({'ids':[identifier,neighbor],'length_m':length,'absolute_error_m':error});edge_errors.append(error)
                row.update(status='compared',positive_bounded_depth=valid,source_ids=shared,board_rotation=rvec.ravel().tolist(),
                    board_translation_m=tvec.ravel().tolist(),camera0_fit_errors_px=residuals.tolist(),
                    cross_view_errors_px=cross.tolist(),triangulated_corners_m=points.tolist(),edges=edges)
                fit_errors.extend(residuals.tolist());cross_errors.extend(cross.tolist())
            comparisons.append(row)
        write_json(out/f'{name}-all-prefix-comparisons.json',comparisons)
        fit,cross,edges=describe(fit_errors),describe(cross_errors),describe(edge_errors);compared=[r for r in comparisons if r['status']=='compared']
        checks={'enough_board_poses':len(compared)>=limits['minimum_compared_poses'],
            'all_positive_bounded_depth':bool(compared) and all(r['positive_bounded_depth'] for r in compared),
            'camera0_fit_p95':fit['p95'] is not None and fit['p95']<=limits['camera0_fit_p95_limit_px'],
            'cross_view_p95':cross['p95'] is not None and cross['p95']<=limits['cross_view_p95_limit_px'],
            'cross_view_maximum':cross['maximum'] is not None and cross['maximum']<=limits['cross_view_max_limit_px'],
            'board_edge_p95':edges['p95'] is not None and edges['p95']<=limits['edge_length_absolute_error_p95_limit_m']}
        return name,{'status':'board_consistency_passed' if all(checks.values()) else 'board_consistency_failed','checks':checks,
            'compared_poses':len(compared),'unavailable_poses':500-len(compared),'compared_frames':[r['frame'] for r in compared],
            'camera0_fit_px':fit,'cross_view_error_px':cross,'edge_length_absolute_error_m':edges}
    with ThreadPoolExecutor(max_workers=2) as pool:results=dict(pool.map(evaluate,calibrations.items()))
    seconds=time.perf_counter()-started
    if seconds>30:raise TimeoutError('Calibration comparison exceeds the declared work envelope')
    report={'created_at':local_time(),'status':'full_rate_board_comparison_complete','results':results,'runtime_seconds':seconds,
        'thresholds_changed':False,'camera_refit':False,'retiming_applied':False,'reserved_camera_opened':False,
        'human_pose_inference':False,'new_detection_calls':0,'new_video_frames_decoded':0,'paid_calls':0,
        'limitations':['Same underlying recording at a higher frame rate, not new scene diversity or an independent validation dataset.',
            'More adjacent board frames are correlated; they are not independent subjects or camera placements.',
            'Both cameras are supplied estimates. Ground, gravity alignment, physical synchronization and body accuracy remain unverified.']}
    write_json(out/'report.json',report)
    print(json.dumps({'results':results,'runtime_seconds':seconds,'verification':seal_suite(out)}))


if __name__=='__main__':main()
