"""Constraint-aware source proposals; the original global fitter owns every acceptance gate."""

from copy import deepcopy
from hashlib import sha256
import itertools
import json
import time

import numpy as np

from weft import global_limb_camera_fit as shared
from weft.body_camera_root_refinement import project_with_jacobians
from weft.spatial_pose import SPATIAL_JOINT_LIMITS

PROFILE='native-constrained-global-limb-camera-fit-v1'
CONTRACT={'maximum_iterations':8,'solver_seconds_per_phase':2.,'maximum_solver_sweeps':512,
    'maximum_rows':8192,'maximum_variables':518,'seed':'Exact prior candidate; every bound measured from original source body',
    'full_floor_phase':'Original .025 gate plus all original negative-gap nonregression constraints',
    'restoration_phase':'Minimize one nonnegative worst-floor slack; original image/rate/no-worse-support inequalities remain hard',
    'restoration_slack_linear_cost':1.,'restoration_slack_quadratic_cost':.01,
    'camera_search_region':'Conservative L1 inner region of original rotation/update norm bounds; exact geodesic guard remains independent',
    'acceptance':'Shared source fitter independently enforces all original nonlinear gates and fresh audit; only full gate can qualify'}


SOLVER_CONTRACT = {"maximum_variables": 518, "maximum_rows": 8192, "maximum_active_rows": 1024,
    "maximum_sweeps": 512, "maximum_seconds": 2., "primal_tolerance": 1e-8,
    "relative_kkt_tolerance": 1e-8, "maximum_coefficient": 1e8,
    "scope": "Convex quadratic proposal with dependent inequalities; every returned point verifies original-unit primal feasibility",
    "memory": "Whitened rows and bounded active set only; no rows-squared Gram matrix"}


def _require(condition, message):
    if not condition: raise ValueError("Constrained source fit: " + message)


def solve_source_qp(hessian, gradient, matrix, lower, *, maximum_seconds=2., maximum_sweeps=512, feasible_initial=None):
    """Project the whitened unconstrained minimizer using nonnegative dual coordinate updates."""
    _require(type(maximum_seconds) in (int,float) and .001 <= maximum_seconds <= 2., "invalid solver time bound")
    _require(type(maximum_sweeps) is int and 1 <= maximum_sweeps <= 512, "invalid solver sweep bound")
    started=time.monotonic();deadline=started+maximum_seconds
    H,g,A,b=(np.asarray(value,dtype=float) for value in (hessian,gradient,matrix,lower))
    n=len(g) if g.ndim==1 else 0
    _require(1 <= n <= 518 and H.shape==(n,n) and A.ndim==2 and A.shape[1]==n
             and len(A)<=8192 and b.shape==(len(A),), "bounded QP dimensions required")
    _require(all(np.isfinite(value).all() and np.max(abs(value),initial=0)<=1e8 for value in (H,g,A,b)),
             "bounded finite QP coefficients required")
    _require(np.allclose(H,H.T,atol=1e-12,rtol=0),"symmetric Hessian required")
    initial=None if feasible_initial is None else np.asarray(feasible_initial,dtype=float)
    if initial is not None:
        _require(initial.shape==(n,) and np.isfinite(initial).all() and np.max(abs(initial),initial=0)<=1e6
                 and np.min(A@initial-b,initial=0)>=-1e-10,'independently feasible initial point required')
    eigenvalues=np.linalg.eigvalsh(H)
    _require(eigenvalues[0]>=1e-12 and eigenvalues[-1]/eigenvalues[0]<=1e12,
             'positive well-conditioned Hessian required without hidden regularization')
    try:factor=np.linalg.cholesky(H)
    except np.linalg.LinAlgError as error:raise ValueError("Constrained source fit: positive-definite Hessian required") from error
    # Shape is rows×variables, never rows×rows, including the native linear solve.
    transformed=np.linalg.solve(factor,A.T).T
    norms=np.linalg.norm(transformed,axis=1);nonzero=norms>0
    C=np.zeros_like(transformed);C[nonzero]=transformed[nonzero]/norms[nonzero,None]
    rhs=np.zeros(len(A));rhs[nonzero]=b[nonzero]/norms[nonzero]
    target=-np.linalg.solve(factor,g);value=target.copy();dual=np.zeros(len(A))
    active=[];active_mask=np.zeros(len(A),dtype=bool);sweeps=0;status="sweep_limit"
    if np.any((~nonzero)&(b>1e-8)):status="inconsistent_constant_linear_row"
    else:
        for sweeps in range(1,maximum_sweeps+1):
            if time.monotonic()>=deadline:status="time_limit";break
            violations=(rhs-C@value)*norms
            violated=np.flatnonzero((violations>1e-8)&nonzero&~active_mask)
            if len(violated):
                priorities=violations[violated]/norms[violated]
                additions=violated[np.argsort(-priorities,kind='stable')[:32]]
                if len(active)+len(additions)>1024:status="active_row_limit";break
                active.extend(map(int,additions));active_mask[additions]=True
            for index in active:
                previous=dual[index]
                dual[index]=max(0.,previous+rhs[index]-float(C[index]@value))
                value+=(dual[index]-previous)*C[index]
            if not np.isfinite(value).all() or np.max(abs(value),initial=0)>1e12:
                status="numeric_bound";break
            slack=C@value-rhs
            primal=max(0.,float(np.max(-slack*norms,initial=0)))
            complement=float(np.max(abs(dual*slack),initial=0))
            if primal<=1e-8 and complement<=1e-8*(1+abs(float(.5*value@value-target@value))):
                status="optimal_verified";break
    point=np.linalg.solve(factor.T,value)
    slack=A@point-b
    backoff=None
    if initial is not None and np.isfinite(point).all() and np.min(slack,initial=0)<-1e-8:
        displacement=point-initial;initial_slack=A@initial-b;direction=A@displacement
        decreasing=direction<0
        alpha=min(1.,float(np.min((initial_slack[decreasing]+5e-9)/-direction[decreasing],initial=1.)))
        candidate=initial+max(0.,alpha)*displacement
        old_cost=float(.5*initial@H@initial+g@initial);new_cost=float(.5*candidate@H@candidate+g@candidate)
        if np.max(abs(candidate),initial=0)<=1e6 and np.min(A@candidate-b,initial=0)>=-1e-8 and new_cost<old_cost-1e-12:
            point=candidate;slack=A@point-b
            backoff={'alpha':max(0.,alpha),'initial_objective':old_cost,'returned_objective':new_cost,
                     'original_constraints_reverified':True}
            status='feasible_progress_backoff'
    primal=max(0.,float(np.max(-slack,initial=0)))
    multipliers=np.zeros(len(A));multipliers[nonzero]=dual[nonzero]/norms[nonzero]
    stationarity=H@point+g-A.T@multipliers
    stationarity_relative=float(np.max(abs(stationarity),initial=0))/(1+float(np.max(abs(H@point),initial=0))+float(np.max(abs(g),initial=0)))
    complement=float(np.max(abs(multipliers*slack),initial=0))
    objective=float(.5*point@H@point+g@point)
    feasible=bool(np.isfinite(point).all() and np.max(abs(point),initial=0)<=1e6 and primal<=1e-8)
    optimal=feasible and stationarity_relative<=1e-8 and complement<=1e-8*(1+abs(objective))
    if status=="optimal_verified" and not optimal:status="feasible_unconverged" if feasible else "verification_failed"
    elif feasible and status not in {"optimal_verified","inconsistent_constant_linear_row","feasible_progress_backoff"}:status="feasible_unconverged"
    return {"status":status,"step":point.tolist() if feasible else None,
        "report":{"variables":n,"constraints":len(A),"active_rows":active,"sweeps":sweeps,
            "seconds":time.monotonic()-started,"primal_maximum_violation":primal,
            "stationarity_relative":stationarity_relative,"complementarity_maximum_abs":complement,
            "objective":objective,"optimality_verified":bool(optimal),"global_nonlinear_infeasibility_claim":False,
            "feasible_progress_backoff":backoff,
            "workspace_numeric_bytes":sum(value.nbytes for value in (H,g,A,b,factor,transformed,C,rhs,dual)),
            "rows_squared_allocation":False}}


def _constraint_rows(context):
    count=len(context['times']);n=5+8*count
    A=[];hard=[];full=[];labels=[];floor_indices=[]
    delta=context['delta'];shape=context['shape'];angles=context['angles'];rotation=context['rotation']
    def add(row,lower,label,*,floor_target=None):
        _require(len(A)<8190,'complete constraint inventory exceeds row bound')
        A.append(row);hard.append(float(lower));full.append(float(lower if floor_target is None else floor_target));labels.append(label)
        if floor_target is not None:floor_indices.append(len(A)-1)
    def local(knot,axis):
        row=np.zeros(n);row[5+8*knot+axis]=1.;return row
    for sample,before,after in zip(context['samples'],context['baseline_states'],context['support_states'],strict=True):
        for support,gap in enumerate(after['gaps']):
            row=np.zeros(n);row[:2]=after['shape_gap_jacobian'][support]
            for knot,weight in sample['weights']:
                row[5+8*knot:5+8*(knot+1)]+=weight*after['gap_jac'][support,shared.LEGS]
            threshold=max(min(0.,before['gaps'][support]),min(0.,gap))
            add(row,threshold-gap,{'kind':'support','time_s':sample['time_s'],'support_index':support},
                floor_target=max(threshold,-.025)-gap)
    for knot,state in enumerate(context['source_states']):
        p=project_with_jacobians(state['absolute_points'],rotation,context['center'],context['K'])
        residual=(p['pixels']-context['target'][knot])*context['scale'];norm=np.linalg.norm(residual,axis=1)
        J=np.zeros((12,2,n));J[:,:,:2]=p['root_jacobian']@state['shape_point_jacobian'];J[:,:,2:5]=p['rotation_jacobian']
        J[:,:,5+8*knot:5+8*(knot+1)]=p['root_jacobian']@state['absolute_jacobian'][:,:,shared.LEGS]
        depth=np.zeros((12,n));depth[:,:2]=rotation[2]@state['shape_point_jacobian']
        depth[:,2:5]=np.c_[p['camera_points'][:,1],-p['camera_points'][:,0],np.zeros(12)]
        depth[:,5+8*knot:5+8*(knot+1)]=rotation[2]@state['absolute_jacobian'][:,:,shared.LEGS]
        for point in range(12):
            if context['confidence'][knot,point]>0:
                jac=J[point]*context['scale']
                limit=min(norm[point],context['initial_errors'][knot,point])
                if norm[point]>1e-12:
                    add(-residual[point]@jac/norm[point],norm[point]-limit-1e-8,
                        {'kind':'source_image','source_index':knot,'point':point})
                else:
                    for axis in range(2):
                        for sign in (-1.,1.):add(sign*jac[axis],-5e-9,{'kind':'zero_source_image','source_index':knot,'point':point})
            current=p['camera_points'][point,2];original=context['initial_depth'][knot,point]
            add(depth[point],max(.05,original-.08)-current,{'kind':'depth_lower','source_index':knot,'point':point})
            add(-depth[point],current-original-.08,{'kind':'depth_upper','source_index':knot,'point':point})
    for axis in range(2):
        row=np.eye(n)[axis]
        add(row,max(-.02,.95-shape[axis]),{'kind':'shape_lower','axis':axis})
        add(-row,-min(.02,1.05-shape[axis]),{'kind':'shape_upper','axis':axis})
    camera=shared._camera_vector(rotation,context['rotation0'])
    for signs in itertools.product((-1.,1.),repeat=3):
        row=np.zeros(n);row[2:5]=-np.array(signs)
        add(row,-min(.10,.15-float(np.array(signs)@camera)),{'kind':'camera_inner_bound','signs':list(signs)})
    limits=np.array([SPATIAL_JOINT_LIMITS[shared.source_fit.NAMES[i]] for i in shared.LEGS])
    for knot in range(count):
        for axis in range(8):
            row=local(knot,axis);current=angles[knot,shared.LEGS[axis]]+delta[knot,axis]
            add(row,max(-.10,-.15-delta[knot,axis],limits[axis,0]-current),{'kind':'joint_lower','source_index':knot,'axis':axis})
            add(-row,-min(.10,.15-delta[knot,axis],limits[axis,1]-current),{'kind':'joint_upper','source_index':knot,'axis':axis})
            if knot:
                difference=delta[knot,axis]-delta[knot-1,axis];bound=1.5*(context['times'][knot]-context['times'][knot-1])
                row=row-local(knot-1,axis)
                add(row,-bound-difference,{'kind':'native_rate_lower','source_index':knot,'axis':axis})
                add(-row,-bound+difference,{'kind':'native_rate_upper','source_index':knot,'axis':axis})
    return np.array(A),np.array(hard),np.array(full),labels,np.array(floor_indices,dtype=int)


class _ConstrainedStrategy:
    profile=PROFILE
    def __init__(self,candidate_bytes,expected_candidate_sha256):
        _require(type(candidate_bytes) is bytes and 0<len(candidate_bytes)<=16<<20,'bounded exact seed bytes required')
        self.seed_sha=sha256(candidate_bytes).hexdigest()
        _require(self.seed_sha==expected_candidate_sha256,'seed digest differs')
        self.document=json.loads(candidate_bytes);self.phases=[]

    def seed(self,args,human,gauge):
        value=self.document
        _require(value.get('profile')==shared.PROFILE and value.get('contract')==shared.CONTRACT,'unknown seed source-fit contract')
        _require(value.get('bindings')==dict(zip(('human','original_evidence','derived_evidence','dimensions'),
                                               (sha256(raw).hexdigest() for raw in args))),'seed original inputs differ')
        candidate=value['candidate'];refs=candidate['reference'];original=human['reference']
        _require(len(refs)==len(original) and [r['time_s'] for r in refs]==[r['time_s'] for r in original]
                 and candidate['root_reference']==human['root_reference'],'seed changed original clock or roots')
        _require(value['camera']['K']==gauge['K'].tolist() and value['camera']['center']==gauge['center'].tolist()
                 and value['camera']['rotation_original']==gauge['rotation0'].tolist(),'seed original camera gauge differs')
        names=shared.source_fit.NAMES
        _require(all(set(r['joints'])==set(names) for r in refs),'seed joint chart differs')
        before=np.array([[r['joints'][name] for name in names] for r in original])
        after=np.array([[r['joints'][name] for name in names] for r in refs])
        protected=sorted(set(range(19))-set(shared.LEGS))
        _require(np.array_equal(before[:,protected],after[:,protected]),'seed changed protected upper-body angles')
        factors=candidate['global_shape_factors']
        _require(set(factors)=={'thigh','shin'} and candidate['lengths']==dict(human['lengths'],
            **{key:human['lengths'][key]*factors[key] for key in factors}),'seed has nonshared shape changes')
        return {'shape':[factors['thigh'],factors['shin']],'delta':(after-before)[:,shared.LEGS],
                'rotation':value['camera']['rotation']}

    def proposals(self,context):
        A,hard,full,labels,floor_rows=_constraint_rows(context)
        H=np.block([[context['Hglobal'],context['cross']],[context['cross'].T,context['Hlocal']]])
        g=np.r_[context['global_gradient'],context['local_gradient']]
        def solve(hessian,gradient,matrix,lower,initial=None):
            remaining=context['deadline']-time.monotonic()
            if remaining<.01:raise TimeoutError('constrained proposal deadline')
            return solve_source_qp(hessian,gradient,matrix,lower,maximum_seconds=min(2.,remaining),feasible_initial=initial)
        solved=solve(H,g,A,full)
        phase={'iteration':context['iteration'],'phase':'full_original_floor','solver':solved,'constraint_labels':labels}
        self.phases.append(phase)
        restoration=False
        if solved['step'] is None:
            # A bounded slack restores feasibility without weakening any source/nonregression constraint.
            n=len(g);rows=len(A)
            _require(rows+len(floor_rows)+2<=8192,'complete restoration constraints exceed row bound')
            restore_H=np.zeros((n+1,n+1));restore_H[:n,:n]=H;restore_H[n,n]=.01
            restore_g=np.r_[g,1.]
            restore_A=np.c_[A,np.zeros(rows)]
            floor_A=np.c_[A[floor_rows],np.ones(len(floor_rows))]
            slack_row=np.zeros(n+1);slack_row[-1]=1.
            restore_A=np.r_[restore_A,floor_A,slack_row[None],-slack_row[None]]
            restore_b=np.r_[hard,full[floor_rows],0.,-max(0.,float(np.max(full-hard)))]
            initial=np.zeros(n+1);initial[-1]=max(0.,float(np.max(full-hard)))
            solved=solve(restore_H,restore_g,restore_A,restore_b,initial)
            self.phases.append({'iteration':context['iteration'],'phase':'bounded_support_restoration','solver':solved,
                                'base_constraint_count':rows,'floor_target_rows':floor_rows.tolist(),
                                'full_floor_feasibility_not_verified':True})
            restoration=True
        if solved['step'] is None:return []
        point=np.array(solved['step'])
        return [{'name':'constrained_support_restoration' if restoration else 'constrained_full_floor',
                 'global_step':point[:5],'local_step':point[5:5+8*len(context['times'])].reshape(-1,8),
                 'restoration':restoration,'solver_report':solved['report']}]

    def receipt(self):
        return {'profile':PROFILE,'contract':deepcopy(CONTRACT),'solver_contract':deepcopy(SOLVER_CONTRACT),
                'seed_sha256':self.seed_sha,'phases':deepcopy(self.phases),'source_or_task_gates_changed':False}


def fit_constrained_limb_camera(human_bytes,original_evidence_bytes,derived_evidence_bytes,dimensions_bytes,
                               *,initial_candidate_bytes,expected_candidate_sha256,maximum_seconds=30.,maximum_iterations=8):
    """Reuse the shared immutable-source/geometry/acceptance pipeline with a constrained proposal strategy."""
    strategy=_ConstrainedStrategy(initial_candidate_bytes,expected_candidate_sha256)
    return shared.fit_global_limb_camera(human_bytes,original_evidence_bytes,derived_evidence_bytes,dimensions_bytes,
        maximum_seconds=maximum_seconds,maximum_iterations=maximum_iterations,_strategy=strategy)
