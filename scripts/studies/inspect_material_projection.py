"""Compare saved cuboid projections with source boxes without refitting or simulation."""

import argparse
import json
import math
from pathlib import Path
import shutil

import numpy as np

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json
from weft.pipeline import verify_artifacts
from weft.recording import load_recording
from weft.spatial_rigid import corners, rotation


ROOT=Path(__file__).resolve().parents[2]
CASES={
    'construction':('spatial-material-fit-01/selected-coast','meta-phys101-ramp-evidence-01'),
    'repeat':('spatial-ramp-repeat-01/candidate-coast','meta-phys101-repeat-evidence-01'),
}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    out=parser.parse_args().out.resolve()
    if out.parent!=ROOT/'artifacts':raise ValueError('Use a new top-level artifacts directory')
    out.mkdir(exist_ok=False);shutil.copyfile(__file__,out/Path(__file__).name)
    projection=json.loads((ROOT/'artifacts/spatial-material-fit-01/construction-inputs.json').read_text())['projection']
    matrix=np.asarray(projection['matrix_px']);results={};all_rows={};verification={}
    for name,(recording_name,evidence_name) in CASES.items():
        recording=ROOT/'artifacts'/recording_name;evidence=ROOT/'artifacts'/evidence_name
        verification[recording_name]=verify_artifacts(recording)
        verification[evidence_name]=verify_artifacts(evidence)
        saved=load_recording(recording)
        states={8+i//2:event['snapshot'] for i,event in enumerate(saved['events']) if i%2==0}
        rows=[]
        for original in json.loads((evidence/'segmentation-evidence.json').read_text())['observations']:
            identifier={'sam-1':'wood-block','sam-2':'green-box'}[original['object_id']]
            row={'frame':original['frame'],'time_s':original['time_s'],'source_id':original['id'],
                'object_id':identifier,'identity':original['identity'],'source_box_xyxy':None,
                'projected_box_xyxy':None,'iou':None,'width_ratio':None,'height_ratio':None,'simulated_world_yaw_deg':None}
            if original['geometry'] is not None:row['source_box_xyxy']=original['geometry']['box_xyxy_halfopen']
            if original['frame'] in states:
                obj=next(obj for obj in states[original['frame']]['objects'] if obj['id']==identifier)
                projected=np.column_stack([corners(obj),np.ones(8)])@matrix.T
                low,high=projected.min(axis=0),projected.max(axis=0)
                row['projected_box_xyxy']=[*low.tolist(),*high.tolist()]
                orient=rotation(obj['quaternion'])
                row['simulated_world_yaw_deg']=math.degrees(math.atan2(orient[1,0],orient[0,0]))
                if row['source_box_xyxy'] is not None:
                    reference=np.asarray(row['source_box_xyxy']);size=reference[2:]-reference[:2]
                    if (size<=0).any():raise ValueError('Source boxes must have positive area')
                    intersect=np.maximum(0,np.minimum(high,reference[2:])-np.maximum(low,reference[:2])).prod()
                    union=(high-low).prod()+size.prod()-intersect
                    row.update(iou=float(intersect/union),width_ratio=float((high[0]-low[0])/size[0]),height_ratio=float((high[1]-low[1])/size[1]))
            rows.append(row)
        all_rows[name]=rows
        results[name]={}
        for identifier in ('wood-block','green-box'):
            available=[row for row in rows if row['object_id']==identifier and row['iou'] is not None]
            final=available[-1]
            results[name][identifier]={'slots':len([row for row in rows if row['object_id']==identifier]),
                'compared':len(available),'mean_box_iou':float(np.mean([row['iou'] for row in available])),
                'final':{key:final[key] for key in ('frame','source_box_xyxy','projected_box_xyxy','iou','width_ratio','height_ratio','simulated_world_yaw_deg')}}
    write_json(out/'all-slots.json',all_rows)
    report={'created_at':local_time(),'status':'projection_diagnostic_complete','results':results,'source_verification':verification,
        'scope':'Additional diagnostic after the frozen center check; not a changed success threshold',
        'limitations':['Source boxes are inferred, not measured geometry','2D overlap cannot validate depth or physical 3D yaw',
            'Original ambiguous identities remain unchanged','No hypothesis is refitted or promoted'],
        'projection_sha256':digest(ROOT/'artifacts/spatial-material-fit-01/construction-inputs.json'),
        'simulation_steps':0,'model_inference':False,'paid_calls':0}
    write_json(out/'report.json',report)
    print(json.dumps({'results':results,'verification':seal_suite(out)}))


if __name__=='__main__':
    main()
