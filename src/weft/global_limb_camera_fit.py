"""Shared leg-shape/camera source fitting with fixed authored inertia and independent floor audit."""

from copy import deepcopy
from hashlib import sha256
import json
import math
import time

import numpy as np

from weft import root_leg_source_fit as source_fit
from weft.body_camera_root_refinement import _rotation, project_with_jacobians
from weft.limb_position_fit import limb_geometry
from weft.spatial_pose import SPATIAL_JOINT_LIMITS
from weft.whole_body_reference_refinement import _NativeGeometry, _schedule

PROFILE = "native-global-limb-camera-source-fit-v1"
CONTRACT = {"maximum_references": 64, "maximum_iterations": 8, "maximum_backtracks": 8,
    "maximum_geometry_calls": 12000, "maximum_seconds": 30., "finalization_seconds": 8.,
    "maximum_output_bytes": 16 << 20, "shape_fraction": .05, "shape_update_fraction": .02,
    "angle_correction_rad": .15, "angle_update_rad": .10, "angle_correction_speed_rad_s": 1.5,
    "camera_rotation_rad": .15, "camera_update_rad": .10, "endpoint_depth_change": .08,
    "minimum_depth": .05, "floor_limit": .025, "numerical_tolerance": 1e-8,
    "huber_delta": .05, "joint_prior": .001, "shape_prior": .01, "camera_prior": .001,
    "temporal_prior": .0005, "directions_per_iteration": ["joint_shape_camera", "joint_shape_fixed_camera"],
    "gauge": "Original source-width focal, center, scale, roots, first-pose base height and floor remain fixed",
    "shape": "Two symmetric thigh/shin multipliers shared by both legs and every original native knot",
    "inertia": "Original compiled body masses, local inertias/centers, motors and force settings remain exact",
    "scope": "Uninstalled kinematic development proposal; no anatomy, calibrated camera or dynamics claim"}
LEGS = sorted(set(i for block in source_fit.LEGS for i in block))
_DYNAMIC_ARRAYS = ("body_mass", "body_inertia", "body_ipos", "body_iquat", "dof_damping", "dof_armature",
                   "jnt_stiffness", "jnt_range", "actuator_ctrlrange", "actuator_forcerange", "actuator_gear",
                   "geom_friction", "geom_solref", "geom_solimp", "geom_contype", "geom_conaffinity")


def _require(value, message):
    if not value: raise ValueError("Global limb/camera fit: " + message)


class SharedShapeGeometry(_NativeGeometry):
    """Move only authored geometric lengths; density-based model rebuilding would change mass."""
    def __init__(self, human):
        super().__init__(human)
        self.original_lengths = {key: human["lengths"][key] for key in ("thigh", "shin")}
        self.frozen = {key: getattr(self.env.model, key).copy() for key in _DYNAMIC_ARRAYS}
        self.constants = (self.env.model.opt.timestep, self.env.mobility.mass, self.env.mobility.limit,
                          self.env.mobility.kp, self.env.mobility.kd)
        self.gravity = self.env.model.opt.gravity.copy()
        self.shape = np.ones(2)

    def invariant(self):
        model = self.env.model
        return (all(np.array_equal(getattr(model, key), value) for key, value in self.frozen.items())
                and np.array_equal(model.opt.gravity, self.gravity)
                and self.constants == (model.opt.timestep, self.env.mobility.mass, self.env.mobility.limit,
                                       self.env.mobility.kp, self.env.mobility.kd))

    def set_shape(self, shape):
        value = np.asarray(shape, dtype=float)
        _require(value.shape == (2,) and np.isfinite(value).all() and np.max(abs(value - 1)) <= .05 + 1e-12,
                 "one bounded global thigh/shin multiplier pair required")
        if np.array_equal(self.shape, value): return
        model = self.env.model
        thigh, shin = (self.original_lengths[key] * factor for key, factor in zip(("thigh", "shin"), value))
        for side in ("left", "right"):
            model.body_pos[model.body(side + "_knee_body").id, 2] = -thigh
            for name, length in (("hip", thigh), ("knee", shin)):
                geom = model.geom(side + "_" + name + "_geom").id
                model.geom_pos[geom, 2] = -length / 2
                model.geom_size[geom, 1] = length / 2
            model.site_pos[model.site(side + "_ankle_site").id, 2] = -shin
        self.mujoco.mj_setConst(model, self.data)
        _require(self.invariant(), "shape changed original mass/inertia, force or model settings")
        self.shape = value.copy()

    def source_state(self, angles, root, derivatives=False):
        state = source_fit._state(self, angles, root, derivatives)
        if derivatives:
            jac = np.zeros((12, 3, 2))
            rotation = self.data.xmat[self.env.model.body("torso").id].reshape(3, 3)
            for side, block, knee, ankle in (("left", source_fit.LEGS[0], 8, 10),
                                            ("right", source_fit.LEGS[1], 9, 11)):
                for parameter, lengths in enumerate(((self.original_lengths["thigh"], 0.),
                                                      (0., self.original_lengths["shin"]))):
                    points, _ = limb_geometry(np.asarray(angles)[list(block)], lengths, -1)
                    jac[[knee, ankle], :, parameter] = points @ rotation.T
            state["shape_point_jacobian"] = jac
            normal=self.data.geom_xmat[self.floor].reshape(3,3)[:,2]
            support_jac=np.zeros((len(self.supports),2))
            for index,(name,geom,end) in enumerate(self.supports):
                for side in ('left','right'):
                    proximal=self.data.xmat[self.env.model.body(side+'_hip_body').id].reshape(3,3)
                    distal=self.data.xmat[self.env.model.body(side+'_knee_body').id].reshape(3,3)
                    axis=self.data.geom_xmat[geom].reshape(3,3)[:,2]
                    if name==side+'_hip_geom':
                        length=self.original_lengths['thigh']
                        support_jac[index,0]=normal@(proximal@[0.,0.,-length/2]+end*length/2*axis)
                    elif name==side+'_knee_geom':
                        support_jac[index,0]=normal@(proximal@[0.,0.,-self.original_lengths['thigh']])
                        length=self.original_lengths['shin']
                        support_jac[index,1]=normal@(distal@[0.,0.,-length/2]+end*length/2*axis)
            state['shape_gap_jacobian']=support_jac
        return state


def _camera_vector(rotation, original):
    relative = rotation @ original.T
    vector = np.array([relative[2,1]-relative[1,2], relative[0,2]-relative[2,0], relative[1,0]-relative[0,1]]) / 2
    norm = float(np.linalg.norm(vector)); angle = math.atan2(norm, float((np.trace(relative)-1)/2))
    return vector * angle / norm if norm > 1e-15 else vector


def _schur_step(global_hessian, local_hessian, cross, global_gradient, local_gradient):
    """Eliminate all local angles together, retaining their native-order temporal coupling."""
    inverse = np.linalg.solve(local_hessian, np.c_[cross.T, local_gradient])
    reduced = global_hessian - cross @ inverse[:, :-1]
    gradient = global_gradient - cross @ inverse[:, -1]
    global_step = -np.linalg.solve(reduced, gradient)
    local_step = -inverse[:, -1] - inverse[:, :-1] @ global_step
    return global_step, local_step


def audit_global_parameters(shape, original_angles, correction, rotation, original_rotation, times):
    """Independently check a detached candidate's declared parameter envelope without model calls."""
    shape, original_angles, correction, rotation, original_rotation, times = (
        np.asarray(value, dtype=float) for value in (shape, original_angles, correction, rotation, original_rotation, times))
    _require(times.ndim == 1 and 3 <= len(times) <= 64 and shape.shape == (2,)
             and original_angles.shape == (len(times),19) and correction.shape == (len(times),8)
             and rotation.shape == original_rotation.shape == (3,3), "parameter audit shape differs")
    _require(all(np.isfinite(value).all() for value in
                 (shape,original_angles,correction,rotation,original_rotation,times)), "nonfinite parameter audit input")
    clock_ok=bool(times[0]>=0 and times[-1]<=60 and np.all(np.diff(times)>0) and np.all(np.diff(times)<=.25+1e-9))
    limits=np.array([SPATIAL_JOINT_LIMITS[source_fit.NAMES[i]] for i in LEGS])
    rotation_ok=all(np.allclose(value@value.T,np.eye(3),atol=1e-10,rtol=0)
                    and abs(np.linalg.det(value)-1)<1e-10 for value in (rotation,original_rotation))
    relative=rotation@original_rotation.T
    skew=np.array([relative[2,1]-relative[1,2],relative[0,2]-relative[2,0],relative[1,0]-relative[0,1]])/2
    camera_angle=math.atan2(float(np.linalg.norm(skew)),float(np.clip((np.trace(relative)-1)/2,-1.,1.)))
    checks={"one_shared_shape_pair":True,"shape_fraction":bool(np.max(abs(shape-1))<=.05+1e-12),
        "angle_correction":bool(np.max(abs(correction))<=.15+1e-8),
        "authored_joint_limits":bool(np.all(original_angles[:,LEGS]+correction>=limits[:,0]-1e-8)
                                     and np.all(original_angles[:,LEGS]+correction<=limits[:,1]+1e-8)),
        "native_clock":clock_ok,
        "native_correction_speed":bool(clock_ok and np.all(abs(np.diff(correction,axis=0))<=1.5*np.diff(times)[:,None]+1e-8)),
        "proper_camera_rotation":bool(rotation_ok),"camera_rotation_bound":bool(camera_angle<=.15+1e-8)}
    return {"passed":all(checks.values()),"checks":checks,"maximum_shape_fraction":float(np.max(abs(shape-1))),
        "maximum_angle_correction_rad":float(np.max(abs(correction))),
        "maximum_native_correction_rate_rad_s":float(np.max(abs(np.diff(correction,axis=0))/np.diff(times)[:,None])) if clock_ok else None,
        "camera_rotation_rad":camera_angle,"scope":"Detached parameter audit only; source identity and geometry are checked separately"}


def fit_global_limb_camera(human_bytes, original_evidence_bytes, derived_evidence_bytes, dimensions_bytes,
                           *, maximum_seconds=30., maximum_iterations=8, _strategy=None):
    _require(type(maximum_seconds) in (int, float) and .1 <= maximum_seconds <= 30., "invalid tighter time bound")
    _require(type(maximum_iterations) is int and 1 <= maximum_iterations <= 8, "invalid tighter iteration bound")
    started = time.monotonic(); deadline = started + maximum_seconds
    args = (human_bytes, original_evidence_bytes, derived_evidence_bytes, dimensions_bytes)
    human, times, angles, roots, target, confidence, scale, rotation0, center, K = source_fit._prepare(*args)
    _require(len(times) <= 64, "complete source exceeds64reference bound")
    rows, samples, lookup = _schedule(times)
    source_indices = [lookup[float(t)] for t in times]
    shape, rotation, delta = np.ones(2), rotation0.copy(), np.zeros((len(times), 8))
    joint_limits = np.array([SPATIAL_JOINT_LIMITS[source_fit.NAMES[i]] for i in LEGS])
    calls, finalizing, journal, stop = 0, False, [], "iteration_limit"

    def check():
        reserve = 0 if finalizing else len(samples)
        if time.monotonic() >= deadline or calls >= 12000 - reserve: raise TimeoutError("global source-fit work bound")

    def at(engine, index, correction, derivatives=False):
        nonlocal calls
        check(); calls += 1
        weights = samples[index]["weights"]
        q = sum(w * angles[i] for i, w in weights)
        q[LEGS] += sum(w * correction[i] for i, w in weights)
        return engine.source_state(q, sum(w * roots[i] for i, w in weights), derivatives)

    def projection(states, rot):
        return [project_with_jacobians(states[j]["absolute_points"], rot, center, K) for j in source_indices]

    def errors(projected):
        return np.linalg.norm(np.array([p["pixels"] for p in projected]) - target, axis=2) * scale

    def valid(projected):
        return all(p["valid"].all() and np.isfinite(p["pixels"]).all() and p["camera_points"][:, 2].min() >= .05 for p in projected)

    def objective(projected, factors, correction, rot):
        norm = errors(projected)
        cost = np.sum(confidence * np.where(norm <= .05, .5 * norm ** 2, .05 * (norm - .025)))
        return float(cost + .0005*np.sum(correction**2) + .00025*np.sum(np.diff(correction,axis=0)**2)
                     + .005*np.sum((factors-1)**2) + .0005*np.sum(_camera_vector(rot,rotation0)**2))

    engine = SharedShapeGeometry(human)
    try:
        states = [at(engine, j, delta) for j in range(len(samples))]
        baseline = states.copy(); projected = projection(states, rotation)
        _require(valid(projected), "original authored gauge projection invalid")
        initial_errors = errors(projected); initial_depth = np.array([p["camera_points"][:, 2] for p in projected])
        initial_cost = objective(projected, shape, delta, rotation)
        if _strategy is not None:
            seed=_strategy.seed(args,deepcopy(human),deepcopy({'K':K,'center':center,'rotation0':rotation0}))
            seed_shape=np.asarray(seed['shape'],dtype=float).copy();seed_delta=np.asarray(seed['delta'],dtype=float).copy()
            seed_rotation=np.asarray(seed['rotation'],dtype=float).copy()
            _require(audit_global_parameters(seed_shape,angles,seed_delta,seed_rotation,rotation0,times)['passed'],
                     'strategy seed exceeds original parameter bounds')
            engine.set_shape(seed_shape)
            seed_states=[at(engine,j,seed_delta) for j in range(len(samples))]
            seed_projection=projection(seed_states,seed_rotation)
            _require(valid(seed_projection)
                and not np.any((errors(seed_projection)>initial_errors+1e-8)&(confidence>0))
                and np.max(abs(np.array([p['camera_points'][:,2] for p in seed_projection])-initial_depth))<=.08+1e-8
                and all(np.all(a['gaps']>=np.minimum(0.,b['gaps'])-1e-8) for a,b in zip(seed_states,baseline)),
                'strategy seed regresses original source/floor evidence')
            shape,delta,rotation,states,projected=seed_shape,seed_delta,seed_rotation,seed_states,seed_projection
        try:
            for iteration in range(maximum_iterations):
                check(); engine.set_shape(shape)
                count = len(times)*8
                Hglobal = np.diag([.01,.01,.001,.001,.001]); Hlocal = np.eye(count)*.001
                cross = np.zeros((5,count)); gg = np.r_[.01*(shape-1), .001*_camera_vector(rotation,rotation0)]
                gl = .001*delta.flatten()
                for knot in range(len(times)-1):
                    a,b=slice(knot*8,(knot+1)*8),slice((knot+1)*8,(knot+2)*8)
                    Hlocal[a,a] += np.eye(8)*.0005;Hlocal[b,b] += np.eye(8)*.0005
                    Hlocal[a,b] -= np.eye(8)*.0005;Hlocal[b,a] -= np.eye(8)*.0005
                    diff=.0005*(delta[knot]-delta[knot+1]);gl[a]+=diff;gl[b]-=diff
                derivatives=[]
                for knot,index in enumerate(source_indices):
                    state=at(engine,index,delta,True);p=project_with_jacobians(state["absolute_points"],rotation,center,K)
                    if _strategy is not None:derivatives.append(state)
                    residual=(p["pixels"]-target[knot])*scale
                    global_jac=np.concatenate((p["root_jacobian"]@state["shape_point_jacobian"],p["rotation_jacobian"]),axis=2)*scale
                    local_jac=(p["root_jacobian"]@state["absolute_jacobian"][:,:,LEGS])*scale
                    norm=np.linalg.norm(residual,axis=1);weight=np.sqrt(confidence[knot]*np.minimum(1.,.05/np.maximum(norm,1e-15)))
                    A=(global_jac*weight[:,None,None]).reshape(-1,5)
                    B=(local_jac*weight[:,None,None]).reshape(-1,8);r=(residual*weight[:,None]).reshape(-1)
                    slot=slice(knot*8,(knot+1)*8)
                    Hglobal+=A.T@A;Hlocal[slot,slot]+=B.T@B;cross[:,slot]+=A.T@B;gg+=A.T@r;gl[slot]+=B.T@r
                accepted=False
                if _strategy is None:
                    proposals=[]
                    for camera_free in (True,False):
                        size=5 if camera_free else 2
                        gs,ls=_schur_step(Hglobal[:size,:size],Hlocal,cross[:size],gg[:size],gl)
                        proposals.append({'global_step':np.r_[gs,np.zeros(5-size)],'local_step':ls.reshape(-1,8),
                            'name':'joint_shape_camera' if camera_free else 'joint_shape_fixed_camera'})
                else:
                    support_derivatives=[at(engine,j,delta,True) for j in range(len(samples))]
                    context={'Hglobal':Hglobal,'Hlocal':Hlocal,'cross':cross,'global_gradient':gg,'local_gradient':gl,
                        'shape':shape,'delta':delta,'rotation':rotation,'rotation0':rotation0,'center':center,'K':K,
                        'angles':angles,'times':times,'target':target,'confidence':confidence,'scale':scale,
                        'initial_depth':initial_depth,'initial_errors':initial_errors,'source_states':derivatives,'support_states':support_derivatives,
                        'baseline_states':baseline,'samples':samples,'iteration':iteration,'remaining_seconds':max(0.,deadline-time.monotonic())}
                    context['deadline']=deadline
                    proposals=_strategy.proposals(deepcopy(context))
                    _require(isinstance(proposals,list) and len(proposals)<=2,'bounded strategy proposal list required')
                    check()
                for proposed in proposals:
                    global_step=np.asarray(proposed['global_step'],dtype=float).copy()
                    local_step=np.asarray(proposed['local_step'],dtype=float).copy()
                    _require(global_step.shape==(5,) and local_step.shape==delta.shape
                        and np.isfinite(global_step).all() and np.isfinite(local_step).all(),'finite bounded strategy step required')
                    factor=min(1.,.02/max(np.max(abs(global_step[:2])),1e-15),.10/max(np.max(abs(local_step)),1e-15),
                               .10/max(np.linalg.norm(global_step[2:]),1e-15))
                    global_step*=factor;local_step*=factor
                    # Projection bounds the proposal, never the source observations or acceptance gates.
                    global_step[:2] = np.clip(shape+global_step[:2], .95, 1.05)-shape
                    lower = np.maximum(-.15, joint_limits[:,0]-angles[:,LEGS])
                    upper = np.minimum(.15, joint_limits[:,1]-angles[:,LEGS])
                    local_step = np.clip(delta+local_step, lower, upper)-delta
                    record={"iteration":iteration,"direction":proposed['name'],
                            "global_step":global_step.tolist(),"local_step":local_step.tolist(),"trials":[],"accepted":False}
                    if _strategy is not None:
                        record['proposal_solver']=deepcopy(proposed.get('solver_report'))
                        record['restoration_requested']=proposed.get('restoration') is True
                    journal.append(record)
                    for backtrack in range(8):
                        check();alpha=2.**-backtrack
                        trial_shape=shape+alpha*global_step[:2];trial_delta=delta+alpha*local_step
                        trial_rotation=_rotation(alpha*global_step[2:])@rotation
                        rejection=None
                        if (np.max(abs(trial_shape-1))>.05+1e-12 or np.max(abs(trial_delta))>.15+1e-8
                            or np.any(abs(np.diff(trial_delta,axis=0))>1.5*np.diff(times)[:,None]+1e-8)
                            or np.linalg.norm(_camera_vector(trial_rotation,rotation0))>.15+1e-8
                            or np.any(angles[:,LEGS]+trial_delta<joint_limits[:,0]-1e-8)
                            or np.any(angles[:,LEGS]+trial_delta>joint_limits[:,1]+1e-8)):
                            rejection="parameter_bounds"
                        trial_states=None
                        if rejection is None:
                            engine.set_shape(trial_shape)
                            trial_states=[at(engine,j,trial_delta) for j in range(len(samples))]
                            if any(np.any(v["gaps"]<np.maximum(np.minimum(0.,b["gaps"]),np.minimum(0.,c["gaps"]))-1e-8)
                                   for v,b,c in zip(trial_states,baseline,states)):
                                rejection="floor_nonregression"
                        trial_projection=projection(trial_states,trial_rotation) if trial_states is not None else None
                        if rejection is None and not valid(trial_projection):rejection="invalid_projection"
                        if rejection is None and np.max(abs(np.array([p["camera_points"][:,2] for p in trial_projection])-initial_depth))>.08+1e-8:
                            rejection="endpoint_depth"
                        if rejection is None and np.any((errors(trial_projection)>np.minimum(initial_errors,errors(projected))+1e-8)&(confidence>0)):
                            rejection="endpoint_image_nonregression"
                        cost=objective(trial_projection,trial_shape,trial_delta,trial_rotation) if rejection is None else None
                        if rejection is None and cost>=objective(projected,shape,delta,rotation)-1e-12:
                            restoring=_strategy is not None and proposed.get('restoration') is True
                            before_error=errors(projected);after_error=errors(trial_projection)
                            image_before=float(np.sum(confidence*np.where(before_error<=.05,.5*before_error**2,.05*(before_error-.025))))
                            image_after=float(np.sum(confidence*np.where(after_error<=.05,.5*after_error**2,.05*(after_error-.025))))
                            floor_before=sum(float(np.maximum(-s['gaps']-.025,0).sum()) for s in states)
                            floor_after=sum(float(np.maximum(-s['gaps']-.025,0).sum()) for s in trial_states)
                            if not (restoring and image_after<=image_before+1e-12 and floor_after<floor_before-1e-8):
                                rejection="no_image_objective_descent"
                        comparison = ({"weighted_mean_pixel_error":float(np.sum(confidence*errors(trial_projection))/np.sum(confidence)/scale),
                            "maximum_confident_endpoint_error_increase":float(np.max(np.where(confidence>0,errors(trial_projection)-initial_errors,0.))),
                            "floor":source_fit._floor_summary(trial_states,rows,lookup)}
                            if trial_projection is not None and valid(trial_projection) else None)
                        record["trials"].append({"backtrack":backtrack,"shape":trial_shape.tolist(),"rejection":rejection,"objective":cost,
                                                 "source_comparison":comparison})
                        if rejection is None:
                            shape,delta,rotation,states,projected=trial_shape,trial_delta,trial_rotation,trial_states,trial_projection
                            record["accepted"]=accepted=True;break
                    if accepted:break
                if not accepted:stop="no_admitted_joint_direction";break
        except TimeoutError:
            stop="work_bound"
        invariant=engine.invariant()
        inertial_hash=sha256(b"".join(v.tobytes() for v in engine.frozen.values())).hexdigest()
    finally:
        engine.close()
    finalizing=True;deadline=time.monotonic()+8.
    audit=SharedShapeGeometry(human)
    try:
        audit.set_shape(shape);final=[at(audit,j,delta) for j in range(len(samples))]
        invariant=invariant and audit.invariant();supports=list(audit.support_names)
        _require(inertial_hash==sha256(b"".join(v.tobytes() for v in audit.frozen.values())).hexdigest(),"independent inertial model mismatch")
    finally:
        audit.close()
    final_projection=projection(final,rotation);final_errors=errors(final_projection)
    parameter_audit=audit_global_parameters(shape,angles,delta,rotation,rotation0,times)
    nonregression=(parameter_audit["passed"] and invariant and valid(final_projection)
        and not np.any((final_errors>initial_errors+1e-8)&(confidence>0))
        and np.max(abs(np.array([p["camera_points"][:,2] for p in final_projection])-initial_depth))<=.08+1e-8
        and all(np.all(a["gaps"]>=np.minimum(0.,b["gaps"])-1e-8) for a,b in zip(final,baseline)))
    before_floor=source_fit._floor_summary(baseline,rows,lookup);after_floor=source_fit._floor_summary(final,rows,lookup)
    image_improved=float(np.sum(confidence*(initial_errors-final_errors)))>1e-8
    floor_pass=all(v["failed"]==0 for v in after_floor.values())
    result={"schema_version":1,"profile":PROFILE,"contract":CONTRACT.copy(),
        "status":"source_qualified_uninstalled" if nonregression and image_improved and floor_pass else "rejected_unpromoted",
        "bindings":dict(zip(("human","original_evidence","derived_evidence","dimensions"),(sha256(raw).hexdigest() for raw in args))),
        "camera":{"K":K.tolist(),"center":center.tolist(),"rotation_original":rotation0.tolist(),"rotation":rotation.tolist()},
        "candidate":{"global_shape_factors":dict(zip(("thigh","shin"),map(float,shape))),
            "lengths":dict(human["lengths"],**{name:human["lengths"][name]*float(value) for name,value in zip(("thigh","shin"),shape)}),
            "reference":[{"time_s":float(t),"joints":dict(zip(source_fit.NAMES,map(float,q)))}
                         for t,q in zip(times,angles+np.array([[d[LEGS.index(i)] if i in LEGS else 0. for i in range(19)] for d in delta]))],
            "root_reference":deepcopy(human["root_reference"]),"inertial_arrays_sha256":inertial_hash,
            "requires_fixed_inertia_shape_runtime":True},
        "audit":{"source_image_improved":bool(image_improved),"nonregression":bool(nonregression),
            "parameter_bounds":parameter_audit,
            "complete_original_floor_gate":floor_pass,"inertial_force_settings_unchanged":bool(invariant),
            "floor_before":before_floor,"floor_after":after_floor,
            "weighted_mean_pixel_error_before":float(np.sum(confidence*initial_errors)/np.sum(confidence)/scale),
            "weighted_mean_pixel_error_after":float(np.sum(confidence*final_errors)/np.sum(confidence)/scale),
            "objective_before":initial_cost,"objective_after":objective(final_projection,shape,delta,rotation)},
        "samples":[{"time_s":s["time_s"],"before_gaps":b["gaps"].tolist(),"after_gaps":a["gaps"].tolist()}
                   for s,b,a in zip(samples,baseline,final)],"support_names":supports,
        "source_endpoints":[{"time_s":float(t),"weights":w.tolist(),"before_error_px":(b/scale).tolist(),"after_error_px":(a/scale).tolist()}
                            for t,w,b,a in zip(times,confidence,initial_errors,final_errors)],
        "journal":journal,"stop_reason":stop,"geometry_calls":calls,"seconds":time.monotonic()-started,
        "physical_validation":"unverified","partition":"exposed_development","rollouts":0,"inference_calls":0,"installed":False}
    if _strategy is not None:
        result['profile']=_strategy.profile
        result['strategy']=deepcopy(_strategy.receipt())
    _require(len(json.dumps(result,allow_nan=False).encode())<=16<<20,"output exceeds bound")
    return result
