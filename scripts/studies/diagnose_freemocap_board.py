"""Inspect planar-pose alternatives without changing the failed calibration check."""

import argparse
import json
from pathlib import Path
import shutil
import tomllib

import cv2
import numpy as np

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json


ROOT=Path(__file__).resolve().parents[2]
PARENT=ROOT/'artifacts/freemocap-board-check-01'
SOURCE=ROOT/'artifacts/freemocap-test-source-01'


def read(path):return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    out=parser.parse_args().out.resolve()
    if out.parent!=ROOT/'artifacts':raise ValueError('Use a new top-level artifact directory')
    out.mkdir(exist_ok=False);shutil.copyfile(__file__,out/Path(__file__).name)
    names=['view-1-detections.json','view-2-detections.json','all-prefix-comparisons.json']
    manifest=read(PARENT/'manifest.json')
    for name in names:
        if digest(PARENT/name)!=manifest['files'][name]:raise ValueError('Frozen detections/comparison changed')
    a,b,rows=[read(PARENT/name) for name in names]
    calibration=tomllib.loads((SOURCE/'member-04-freemocap_test_data_camera_calibration.toml').read_text())
    cams=[{'K':np.asarray(calibration[f'cam_{i}']['matrix']), 'd':np.asarray(calibration[f'cam_{i}']['distortions']),
        'r':np.asarray(calibration[f'cam_{i}']['rotation']), 't':np.asarray(calibration[f'cam_{i}']['translation'])*.001} for i in (0,1)]
    if not np.array_equal(cams[0]['r'],np.zeros(3)) or not np.array_equal(cams[0]['t'],np.zeros(3)):
        raise ValueError('Diagnostic requires the declared camera-0 origin')
    board=cv2.aruco.CharucoBoard((7,5),.058,.058*.8,cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250))
    model=board.getChessboardCorners().astype(float);results=[]
    write_json(out/'pre-run.json',{'created_at':local_time(),'parent_manifest_sha256':digest(PARENT/'manifest.json'),
        'source_calibration_sha256':digest(SOURCE/'member-04-freemocap_test_data_camera_calibration.toml'),
        'scope':'Report both planar IPPE solutions and stereo consistency; no solution selection, camera/time fit, changed gate or promotion.',
        'new_video_frames_decoded':0,'new_corner_detection_calls':0,'reserved_camera_opened':False})
    def project(points,camera):return cv2.projectPoints(points,camera['r'],camera['t'],camera['K'],camera['d'])[0].reshape(-1,2)
    def errors(prediction,observed):
        values=np.linalg.norm(prediction-np.asarray(observed),axis=1)
        return {'rmse_px':float(np.sqrt(np.mean(values**2))),'maximum_px':float(values.max()),'values_px':values.tolist()}
    for original in rows:
        if original['status']!='compared':continue
        frame=original['frame'];left={p['id']:p['pixel'] for p in a[frame]['corners']};right={p['id']:p['pixel'] for p in b[frame]['corners']}
        shared=original['source_ids'];ids=sorted(left);points=np.asarray(original['triangulated_corners_m'])
        singular=np.linalg.svd(points-points.mean(axis=0),compute_uv=False)
        result={'frame':frame,'stereo_reprojection':[
            errors(project(points,camera),[data[i] for i in shared]) for camera,data in zip(cams,(left,right))],
            'stereo_plane_rms_m':float(singular[-1]/np.sqrt(len(points))),'original_check_unchanged':True,'planar_solutions':[]}
        solved=cv2.solvePnPGeneric(model[ids],np.asarray([left[i] for i in ids]),cams[0]['K'],cams[0]['d'],flags=cv2.SOLVEPNP_IPPE)
        for rvec,tvec in zip(solved[1],solved[2]):
            world=model@cv2.Rodrigues(rvec)[0].T+tvec.ravel()
            result['planar_solutions'].append({'positive_camera0_depth':bool((world[:,2]>0).all()),
                'camera0':errors(project(world[ids],cams[0]),[left[i] for i in ids]),
                'camera1':errors(project(world[shared],cams[1]),[right[i] for i in shared]),
                'rotation':rvec.ravel().tolist(),'translation_m':tvec.ravel().tolist()})
        results.append(result)
    report={'created_at':local_time(),'status':'diagnostic_complete_no_repair','poses':results,
        'failed_check_preserved':True,'thresholds_changed':False,'physical_validation':'unverified',
        'new_video_frames_decoded':0,'new_corner_detection_calls':0,'reserved_camera_opened':False,'paid_calls':0}
    write_json(out/'report.json',report)
    print(json.dumps({'poses':[{'frame':r['frame'],'stereo_rmse':[v['rmse_px'] for v in r['stereo_reprojection']],
        'planar_solutions':[[s['camera0']['rmse_px'],s['camera1']['rmse_px']] for s in r['planar_solutions']]} for r in results],
        'verification':seal_suite(out)}))


if __name__=='__main__':main()
