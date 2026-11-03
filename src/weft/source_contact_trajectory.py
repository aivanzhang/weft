"""Source-constrained authored target corrections; observations and scored reference stay immutable."""
from hashlib import sha256
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np

from weft.constrained_qp import solve_feasible_qp
from weft.human import joint_contract
from weft.io import canonical_hash
from weft.shared_world import validate_shared_world
from weft.shared_world_clearance import _Kinematics, _source, _schedule
from weft.shared_world_packaging import read_package_file, read_package_manifest
from weft.whole_body_reference_refinement import POINTS

PROFILE = 'source-contact-authored-target-v1'
VARIABLES = ('right_shoulder','right_shoulder_lateral','right_shoulder_twist','right_elbow','root_x','root_y','root_z')
CONTRACT = {'maximum_references':128,'maximum_samples':512,'maximum_seconds':30.,'audit_seconds':8.,
    'maximum_evaluations':12000,'maximum_backtracks':6,'passes':1,'angle_bound_rad':.15,'angle_update_rad':.10,
    'root_norm_bound_model_units':.08,'angle_rate_rad_s':1.5,'root_rate_model_units_s':.3,
    'minimum_confidence':.5,'minimum_camera_depth':.05,'maximum_depth_change_model_units':.08,
    'floor_limit_model_units':.025,'tolerance':1e-8,'minimum_gap_improvement':1e-9,
    'scope':'One uninstalled authored target hypothesis, not recovered observations, learned policy or physical contact.',
    'source_constraint':'Every confident original endpoint error must not increase; fixed original authored camera.',
    'gap_constraint':'No new/worse negative human-floor or human-equipment gap at source and30Hz samples.',
    'dynamics':'Original scene, scored reference, task, mass, friction and force limits are unchanged.'}


def project(points, camera, origin, scale):
    if (type(scale) not in (int,float) or not math.isfinite(scale) or scale<=0
            or not math.isfinite(float(camera['focal_px'])) or camera['focal_px']<=0
            or not math.isfinite(float(camera['distance_model_units'])) or camera['distance_model_units']<=0):
        raise ValueError('Fixed source camera requires positive finite scale, focal length and distance')
    local=(np.asarray(points)-origin)/scale-np.asarray(camera['world_anchor_model_units'])
    depth=local[:,1]+camera['distance_model_units'];f=float(camera['focal_px'])
    if not np.isfinite(local).all() or np.any(depth<CONTRACT['minimum_camera_depth']):
        raise ValueError('Fixed source camera depth is unsupported')
    pixels=np.c_[f*local[:,0]/depth,-f*local[:,2]/depth]+camera['pixel_origin']
    jac=np.zeros((len(local),2,3));jac[:,0,0]=f/depth/scale;jac[:,0,1]=-f*local[:,0]/depth**2/scale
    jac[:,1,2]=-f/depth/scale;jac[:,1,1]=f*local[:,2]/depth**2/scale
    return pixels,jac,depth


def bounded(delta,times):
    delta,times=np.asarray(delta,dtype=float),np.asarray(times,dtype=float)
    if (delta.shape!=(len(times),7) or len(times)<3 or not np.isfinite(delta).all()
            or not np.isfinite(times).all() or np.any(np.diff(times)<=0)):
        return False
    tol=CONTRACT['tolerance']
    return bool(np.array_equal(delta[0],np.zeros(7)) and np.array_equal(delta[-1],np.zeros(7))
        and np.max(abs(delta[:,:4]))<=.15+tol and np.max(np.linalg.norm(delta[:,4:],axis=1))<=.08+tol
        and np.all(abs(np.diff(delta[:,:4],axis=0))<=1.5*np.diff(times)[:,None]+tol)
        and np.all(np.linalg.norm(np.diff(delta[:,4:],axis=0),axis=1)<=.3*np.diff(times)+tol))


class ContactGeometry(_Kinematics):
    def __init__(self,scene):
        super().__init__(scene)
        self.names=tuple(joint_contract(scene['human_spec'])[0])+('root_x','root_y','root_z')
        if any(name not in self.names for name in VARIABLES):raise ValueError('Exact seven authored target variables unavailable')
        self.indices=np.array([self.names.index(name) for name in VARIABLES])
        self.dofs=np.array([int(self.model.jnt_dofadr[self.model.joint(name).id]) for name in VARIABLES])
        self.scale=scene['coordinate_frame']['scale_bridge']['body_height_to_world_scale']
        self.unit=np.array([1.]*4+[self.scale]*3)
        self.original=self.reference.copy()
        self.sites=[self.model.site(name+'_site').id for name in POINTS]
        self.selected=np.array([i for i,c in enumerate(self.columns) if c['kind'] in ('human_rigid','human_floor')])
        self.equipment=np.array([i for i,c in enumerate(self.columns) if c['kind']=='human_rigid'])
        self.floors=np.array([i for i,c in enumerate(self.columns) if c['kind']=='human_floor'])

    def evaluate(self,weights,stamp,delta,deadline,derivatives=False):
        self.reference=self.original.copy();self.reference[:,self.indices]+=delta*self.unit
        gaps=self.measure(weights,stamp,deadline,np.empty(len(self.pairs)))
        points=self.data.site_xpos[self.sites].copy()
        result={'gaps':gaps,'points':points}
        if derivatives:
            point_jac=[];jac=np.zeros((3,self.model.nv))
            for sid in self.sites:
                mujoco.mj_jacSite(self.model,self.data,jac,None,sid);point_jac.append(jac[:,self.dofs]*self.unit)
            gap_jac=np.zeros((len(gaps),7));a=np.zeros((3,self.model.nv));b=np.zeros_like(a)
            for i in self.selected:
                first,second=self.pairs[i];segment=np.zeros(6)
                gap=mujoco.mj_geomDistance(self.model,self.data,first,second,100.,segment)
                if abs(gap)<1e-9:continue  # Nonsmooth touching points are independently checked, not assigned a derivative.
                normal=(segment[:3]-segment[3:])/gap
                mujoco.mj_jac(self.model,self.data,a,None,segment[:3],self.model.geom_bodyid[first])
                mujoco.mj_jac(self.model,self.data,b,None,segment[3:],self.model.geom_bodyid[second])
                gap_jac[i]=(normal@(a-b)[:,self.dofs])*self.unit
            result.update(point_jac=np.array(point_jac),gap_jac=gap_jac)
        return result


def propose_contact_targets(source_run,*,max_seconds=30.):
    """Fit one time-ordered trajectory and independently audit it; never edit a source or scorer."""
    if type(max_seconds) not in (float,int) or not math.isfinite(max_seconds) or not 1<=max_seconds<=30:
        raise ValueError('Contact target fitting requires a1–30second work bound')
    started=time.monotonic();deadline=started+max_seconds
    root=Path(source_run);manifest,pins=read_package_manifest(root)
    if not {'scene.json','exercise-task.json'}<=pins.keys():
        raise ValueError('Original scene and task must both appear in the sealed manifest')
    scene_raw=read_package_file(root,'scene.json',max_bytes=16<<20,expected_sha256=pins['scene.json'])
    scene=validate_shared_world(json.loads(scene_raw));human=scene['human_spec'];refs=human['reference']
    if not 3<=len(refs)<=128 or human.get('profile')!='exercise-v3':raise ValueError('Complete spatial source references required')
    source=_source(scene,root,deadline)
    if source['status']!='verified_source_binding':raise ValueError('Contact targets require source-qualified original observations')
    task_raw=read_package_file(root,'exercise-task.json',max_bytes=64<<10,expected_sha256=pins['exercise-task.json'])
    from weft.shared_exercise_task import validate_shared_exercise_task
    task=validate_shared_exercise_task(json.loads(task_raw),scene)
    camera=scene['objects'][0]['provenance']['camera_hypothesis']
    if any(o['provenance'].get('camera_hypothesis')!=camera for o in scene['objects']):raise ValueError('All equipment must retain one exact original camera')
    observation_name=scene['source_binding']['evidence_documents'][source['document_sha256']]
    document=json.loads(read_package_file(root,observation_name,max_bytes=32<<20,expected_sha256=source['document_sha256']))
    by_frame={r['frame']:r for r in document['observations'] if r['object_id']==source['entity_id']}
    targets,confidence=[],[]
    for frame in source['reference_frames']:
        row=by_frame[frame];measure=row['measurements'].get('landmarks',{})
        if [measure.get(k) for k in ('coordinate_system','unit','status')]!=['image','pixels','inferred']:
            raise ValueError('Source-pixel landmarks required at every reference')
        points={p['index']:p for p in measure['value']}
        targets.append([points[i]['position'][:2] for i in POINTS.values()])
        confidence.append([min(points[i]['visibility'],points[i]['presence'])>=.5 for i in POINTS.values()])
    targets,confidence=np.array(targets),np.array(confidence)
    if not np.isfinite(targets).all() or np.any(confidence.sum(axis=1)<6):raise ValueError('Six confident source endpoints per reference required')
    rows,clocks=_schedule(scene,source);unique={}
    for row in rows:
        if row['weights']:unique.setdefault(float(row['reference_time_s']),row['weights'])
    stamps=sorted(unique);samples=[unique[t] for t in stamps]
    if len(samples)>512:raise ValueError('Complete source/control schedule exceeds512samples; no truncation')
    times=np.array([r['time_s'] for r in refs]);lookup={t:i for i,t in enumerate(stamps)};source_indices=[lookup[float(t)] for t in times]
    affected=[[j for j,w in enumerate(samples) if any(k==i for k,_ in w)] for i in range(len(times))]
    origin=np.array(scene['coordinate_frame']['scale_bridge']['origin_world']);scale=scene['coordinate_frame']['scale_bridge']['body_height_to_world_scale']
    delta=np.zeros((len(times),7));calls=0;events=[];auditing=False
    engine=ContactGeometry(scene)
    def evaluate(engine,j,change,derivatives=False):
        nonlocal calls
        limit=CONTRACT['maximum_evaluations']-(0 if auditing else len(samples))
        if time.monotonic()>=deadline or calls>=limit:raise TimeoutError('Contact trajectory native work bound')
        calls+=1;return engine.evaluate(samples[j],stamps[j],change,deadline,derivatives)
    def pixels(state):return project(state['points'],camera,origin,scale)
    def error(state,i):return np.linalg.norm(pixels(state)[0]-targets[i],axis=1)
    original=[evaluate(engine,j,delta) for j in range(len(samples))];current=list(original)
    errors=np.array([error(original[j],i) for i,j in enumerate(source_indices)])
    depths=np.array([pixels(original[j])[2] for j in source_indices])
    gap_before=np.array([s['gaps'] for s in original]);lower=np.minimum(gap_before,0.)
    overlap=[i for i,j in enumerate(source_indices) if np.min(gap_before[j,engine.equipment])<0]
    stop='complete_single_pass'
    for i in overlap:
        if i in (0,len(times)-1):continue
        try:
            states={j:evaluate(engine,j,delta,True) for j in affected[i]}
            H=np.eye(7)*.001;g=np.zeros(7);A=[];B=[]
            for j,state in states.items():
                weight=next(w for k,w in samples[j] if k==i)
                J=state['gap_jac']*weight
                active=engine.equipment[state['gaps'][engine.equipment]<0]
                H+=J[active].T@J[active];g+=J[active].T@state['gaps'][active]
                # All affected pairs retain their original negative-gap envelope.
                A.extend(J[engine.selected]);B.extend(lower[j,engine.selected]-state['gaps'][engine.selected])
            j=source_indices[i];state=states[j];pix,jproj,_=pixels(state)
            jp=np.einsum('pab,pbc->pac',jproj,state['point_jac'])
            residual=pix-targets[i];norm=np.linalg.norm(residual,axis=1)
            for k in np.flatnonzero(confidence[i]):
                grad=(residual[k]@jp[k])/max(norm[k],1e-12)
                A.append(-grad);B.append(norm[k]-errors[i,k])
            bound=np.array([.15]*4+[.08/math.sqrt(3)]*3)
            update=np.array([.10]*4+[.04/math.sqrt(3)]*3)
            low=np.maximum(-bound-delta[i],-update);high=np.minimum(bound-delta[i],update)
            for neighbor in (i-1,i+1):
                rate=np.array([1.5]*4+[.3/math.sqrt(3)]*3)*abs(times[i]-times[neighbor])
                low=np.maximum(low,delta[neighbor]-delta[i]-rate);high=np.minimum(high,delta[neighbor]-delta[i]+rate)
            for k,name in enumerate(VARIABLES[:4]):
                jid=engine.model.joint(name).id;q=refs[i]['joints'][name]+delta[i,k]
                low[k]=max(low[k],engine.model.jnt_range[jid,0]-q);high[k]=min(high[k],engine.model.jnt_range[jid,1]-q)
            A.extend(np.eye(7));B.extend(low);A.extend(-np.eye(7));B.extend(-high)
            solved=solve_feasible_qp(H,g,np.array(A),np.array(B),settings={'maximum_seconds':.1})
            event={'reference_index':i,'source_frame':source['reference_frames'][i],'qp':solved,'accepted':False,'rejections':[]};events.append(event)
            if solved['step'] is None:continue
            step=np.array(solved['step']);old_cost=sum(float(np.minimum(current[j]['gaps'][engine.equipment],0)@np.minimum(current[j]['gaps'][engine.equipment],0)) for j in affected[i])
            for backtrack in range(6):
                candidate=delta.copy();candidate[i]+=step*(.5**backtrack)
                trial={j:evaluate(engine,j,candidate) for j in affected[i]};reason=None
                if not bounded(candidate,times):reason='parameter_or_rate_bound'
                if reason is None and any(np.any(s['gaps'][engine.selected]<lower[j,engine.selected]-1e-8) for j,s in trial.items()):reason='original_gap_nonregression'
                if reason is None and np.any(error(trial[source_indices[i]],i)[confidence[i]]>errors[i,confidence[i]]+1e-8):reason='original_source_pixel_nonregression'
                if reason is None and np.any(abs(pixels(trial[source_indices[i]])[2]-depths[i])>.08+1e-8):reason='original_depth_bound'
                cost=sum(float(np.minimum(s['gaps'][engine.equipment],0)@np.minimum(s['gaps'][engine.equipment],0)) for s in trial.values())
                if reason is None and cost>=old_cost-1e-12:reason='no_contact_improvement'
                if reason is None:
                    delta=candidate
                    for j,s in trial.items():current[j]=s
                    event.update(accepted=True,backtracks=backtrack,cost_before=old_cost,cost_after=cost);break
                event['rejections'].append(reason)
        except TimeoutError:
            stop='work_bound';break
    # Fresh native geometry audits all rows, never the fitter's cached states.
    auditing=True;deadline=time.monotonic()+8;auditor=ContactGeometry(scene)
    final=[evaluate(auditor,j,delta) for j in range(len(samples))]
    gaps=np.array([s['gaps'] for s in final]);final_errors=np.array([error(final[j],i) for i,j in enumerate(source_indices)])
    final_depth=np.array([pixels(final[j])[2] for j in source_indices])
    checks={'parameter_bounds':bounded(delta,times),'source_pixels':bool(np.all(final_errors[confidence]<=errors[confidence]+1e-8)),
        'fixed_camera_depth':bool(np.all(abs(final_depth-depths)<=.08+1e-8)),
        'gap_nonregression':bool(np.all(gaps[:,auditor.selected]>=lower[:,auditor.selected]-1e-8)),
        'floor_limit':bool(np.min(gaps[:,auditor.floors])/scale>=-.025-1e-8)}
    before_cost=float(np.sum(np.minimum(gap_before[:,auditor.equipment],0)**2));after_cost=float(np.sum(np.minimum(gaps[:,auditor.equipment],0)**2))
    checks['contact_improved']=after_cost<before_cost-1e-9
    report={'schema_version':1,'profile':PROFILE,'contract':CONTRACT,'status':'source_constrained_target_proposed' if all(checks.values()) else 'rejected',
        'checks':checks,'stop':stop,'source':source,'source_manifest_sha256':sha256(manifest).hexdigest(),
        'scene_sha256':canonical_hash(scene),'task_sha256':canonical_hash(task),'task_bytes_sha256':sha256(task_raw).hexdigest(),
        'variables':list(VARIABLES),'reference_times_s':times.tolist(),'corrections':delta.tolist(),'original_source_slots':rows,'clocks':clocks,
        'samples':[{'time_s':t,'weights':w,'original_gaps_world':a.tolist(),'candidate_gaps_world':b.tolist()} for t,w,a,b in zip(stamps,samples,gap_before,gaps)],
        'pair_columns':engine.columns,'source_endpoint_error_px':{'original':errors.tolist(),'candidate':final_errors.tolist(),'confidence_mask':confidence.tolist()},
        'summary':{'squared_penetration_before':before_cost,'squared_penetration_after':after_cost,
            'worst_equipment_penetration_before':max(0.,-float(gap_before[:,auditor.equipment].min())),
            'worst_equipment_penetration_after':max(0.,-float(gaps[:,auditor.equipment].min())),
            'accepted_blocks':sum(e['accepted'] for e in events),'geometry_evaluations':calls,'seconds':time.monotonic()-started},
        'events':events,'physics_steps':0,'source_changed':False,'task_changed':False,'learned_policy':False}
    if (read_package_file(root,'manifest.json',max_bytes=1<<20)!=manifest
            or _source(scene,root,deadline)!=source):
        raise ValueError('Original source proof changed during target fitting')
    if len(json.dumps(report,allow_nan=False).encode())>16<<20:raise ValueError('Contact target report exceeds16MiB')
    return report
