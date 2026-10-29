"""Explicit planar sandbox hypotheses from saved proposals; never physics promotion."""

from copy import deepcopy
import itertools
import math
import re

import numpy as np

from weft.io import canonical_hash
from weft.scene import MAX_SPEED, SceneEnv, validate_scene


CONTRACT = {
    'method':'segmentation-authored-planar-candidate-v1', 'maximum_frames':500, 'maximum_objects':8,
    'fit_samples':5, 'maximum_fit_gap_seconds':.1, 'maximum_replay_gap_seconds':.25,
    'maximum_duration_seconds':6., 'maximum_fit_residual_image_widths':.01,
    'maximum_static_displacement_image_widths':.005, 'maximum_identity_hypotheses':6,
    'maximum_physics_steps':1800, 'maximum_anonymous_assignment_states':256,
    'scope':'One authored physical world with alternative identity correspondences; no model selection or physical validation',
}


class UnsupportedSegmentationScene(ValueError):
    """The explicit assumptions or evidence cannot support this bounded sandbox."""


def _reject(message):
    raise UnsupportedSegmentationScene(message)


def _number(value, low, high, name):
    if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:
        _reject('Invalid bounded '+name)
    return float(value)


def _normalize(evidence):
    if evidence.get('profile') == 'stabilized-segmentation-v1':
        from weft.stabilized_segmentation import materialize_stabilized
        return _normalize(materialize_stabilized(evidence))
    if evidence.get('profile')=='scene-observations-v1':
        from weft.observations import validate_observations
        shared=validate_observations(evidence)
        if shared['schema_version']!=3 or shared['provenance'].get('physics_ready') is not False:
            _reject('Only unpromoted decoder-timed segmentation interchange is supported')
        if shared['camera']['motion']['value'] in ('moving','moving_background'):
            _reject('Shared evidence declares camera motion; a fixed-camera scene is unsupported')
        rows=[deepcopy(row['payload']) for row in shared['evidence'] if row['kind']=='model_segmentation_proposal']
        groups=[{'id':row['id'],'object_ids':row['objects'], **deepcopy(row['attributes'])}
                for row in shared['relationships'] if row['type']=='possible_association_ambiguity']
        evidence={'profile':'segmentation-evidence-v1','physics_ready':False,
            'source':{'submitted_video_sha256':shared['source']['sha256'],'frame_count':shared['source']['decoded_frames'],
                      'width_px':shared['image']['width_px'],'height_px':shared['image']['height_px']},
            'time':{'timestamps_s':shared['source']['frame_times_s']},'observations':rows,
            'objects':[{'id':obj['id'],'raw_model_id':next((row['raw_model_id'] for row in rows if row['object_id']==obj['id']),None)}
                       for obj in shared['objects']], 'identity_groups':groups}
    if evidence.get('profile')!='segmentation-evidence-v1' or evidence.get('physics_ready') is not False:
        _reject('Expected unpromoted saved segmentation evidence')
    source=evidence['source']
    width,height=source['width_px'],source['height_px']
    if type(width) is not int or type(height) is not int or not 1<=width<=4096 or not 1<=height<=4096:
        _reject('Unsupported source image bounds')
    ids=[obj['id'] for obj in evidence['objects']]
    if (not 1<=len(ids)<=CONTRACT['maximum_objects'] or any(not isinstance(value,str) or not re.fullmatch(r'sam-[1-8]',value) for value in ids)
            or len(set(ids))!=len(ids)):
        _reject('Object identities must be unique and bounded')
    raw_ids={obj['id']:obj['raw_model_id'] for obj in evidence['objects']}
    if (any(not isinstance(value,str) or value not in tuple(str(i) for i in range(1,9)) or key!='sam-'+value for key,value in raw_ids.items())
            or len(set(raw_ids.values()))!=len(ids)):
        _reject('Raw model identities conflict')
    times=evidence['time']['timestamps_s']
    if not isinstance(times,list) or not 1<=len(times)<=CONTRACT['maximum_frames'] or len(times)!=source['frame_count']:
        _reject('Every source timestamp is required')
    previous=-1
    for stamp in times:
        _number(stamp,0,3600,'source time')
        if stamp<=previous:
            _reject('Source times must strictly increase')
        previous=stamp
    rows={}
    for row in evidence['observations']:
        frame,identifier=row['frame'],row['object_id']
        if type(frame) is not int or not 0<=frame<len(times) or identifier not in ids or (frame,identifier) in rows:
            _reject('Observation identity/frame conflict')
        if row['time_s']!=times[frame] or row['raw_model_id']!=raw_ids[identifier]:
            _reject('Observation source time/raw identity conflict')
        if row['phase'] not in ('pre_initialization','initialization','post_initialization'):
            _reject('Missing initialization phase provenance')
        if row['proposal_status']=='missing':
            if row['geometry'] is not None:
                _reject('Missing proposals cannot have geometry')
        elif row['proposal_status']=='returned':
            geometry=row['geometry']
            if not isinstance(geometry,dict) or geometry.get('status')!='inferred':
                _reject('Geometry must remain an inferred model proposal')
            box,center=geometry['box_xyxy_halfopen'],geometry['center_px']
            if not isinstance(box,list) or len(box)!=4 or not isinstance(center,list) or len(center)!=2:
                _reject('Invalid proposal coordinate shape')
            for value in box+center:
                _number(value,0,4096,'source coordinate')
            if not 0<=box[0]<box[2]<=width or not 0<=box[1]<box[3]<=height or center!=[(box[0]+box[2])/2,(box[1]+box[3])/2]:
                _reject('Source box and center conflict')
        else:
            _reject('Unknown proposal status')
        rows[(frame,identifier)]=row
    if len(rows)!=len(times)*len(ids):
        _reject('All frame/object slots, including missing ones, are required')
    initial={}
    for identifier in ids:
        found=[frame for frame in range(len(times)) if rows[(frame,identifier)]['phase']=='initialization']
        if len(found)!=1:
            _reject('Each object requires one explicit initialization frame')
        initial[identifier]=found[0]
        for frame in range(len(times)):
            expected='pre_initialization' if frame<found[0] else 'initialization' if frame==found[0] else 'post_initialization'
            if rows[(frame,identifier)]['phase']!=expected:
                _reject('Initialization phases are temporally inconsistent')
    groups=evidence.get('identity_groups',[])
    if not isinstance(groups,list) or len(groups)>len(ids)-1:
        _reject('Identity group count exceeds the conservative union envelope')
    for group in groups:
        if (type(group.get('onset_frame')) is not int or not 0<=group['onset_frame']<len(times)
                or not isinstance(group.get('object_ids'),list) or len(group['object_ids'])<2
                or len(set(group['object_ids']))!=len(group['object_ids']) or not set(group['object_ids'])<=set(ids)):
            _reject('Identity group provenance conflicts')
    return source,ids,raw_ids,times,rows,initial,groups


def _identity_hypotheses(ids,groups,end):
    components={identifier:{identifier} for identifier in ids}
    for group in groups:
        if group['onset_frame']>end:
            continue
        merged=set().union(*(components[value] for value in group['object_ids']))
        for value in merged:
            components[value]=merged
    unique=sorted({tuple(sorted(value)) for value in components.values()})
    total=math.prod(math.factorial(len(group)) for group in unique)
    if total>CONTRACT['maximum_identity_hypotheses']:
        _reject('Ambiguity exceeds the six-correspondence envelope; no truncation or winner selection')
    alternatives=[]
    for product in itertools.product(*(itertools.permutations(group) for group in unique)):
        alternatives.append({key:value for group,permutation in zip(unique,product) for key,value in zip(group,permutation)})
    return {'initial_anchor':'Raw model IDs anchor scene objects by an authored identity assumption',
        'generation_only_alternatives':alternatives, 'selected_alternative':None,
        'scope':'Same authored physical world, not independently recovered dynamical hypotheses; differing roles/masses/shapes remain authored and do not establish which physical body follows a model ID'}


def generate_candidate(evidence, assumptions):
    """Construct one sandbox using generation-prefix geometry only, never development fit."""
    source,ids,raw_ids,times,rows,initial,groups=_normalize(evidence)
    required={'schema_version','camera_model','known_camera_motion','plane','units','boundaries','shape_friction',
        'fit_motion','generation_end_frame','dt','gravity','damping','restitution','objects','controlled_id','target_px','task'}
    if not isinstance(assumptions,dict) or set(assumptions)!=required or type(assumptions['schema_version']) is not int or assumptions['schema_version']!=1:
        _reject('All physical, task and partition assumptions must be explicitly authored')
    stabilized = evidence.get('profile') == 'stabilized-segmentation-v1'
    if ('camera_registration_sha256' in source) != stabilized:
        _reject('Registered coordinates require their validated original-evidence wrapper')
    supported={'camera_model':'similarity_stabilized_orthographic_assumed' if stabilized else 'fixed_orthographic_assumed','plane':'common_2d_plane_assumed',
        'units':'image_widths_not_meters','boundaries':'closed_image_edges_assumed','fit_motion':'unforced_prefix_assumed'}
    if any(assumptions[key]!=value for key,value in supported.items()):
        _reject('Unsupported camera, plane, units, walls or prefix-motion assumption')
    if assumptions['known_camera_motion'] not in (('unknown','not_detected','moving') if stabilized else ('unknown','not_detected')):
        _reject('Known moving camera is unsupported; no camera correction or override')
    if _number(assumptions['shape_friction'],0,1,'shape friction')!=.5:
        _reject('Existing scene engine requires explicitly authored per-shape friction 0.5')
    dt=_number(assumptions['dt'],1/240,1/20,'dt')
    damping=_number(assumptions['damping'],.01,1,'damping')
    restitution=_number(assumptions['restitution'],0,1,'restitution')
    gravity=assumptions['gravity']
    if not isinstance(gravity,list) or len(gravity)!=2:
        _reject('Gravity must be explicitly authored in image-width units/s²')
    gravity=np.array([_number(v,-20,20,'gravity') for v in gravity])
    start=max(initial.values())
    end=assumptions['generation_end_frame']
    if type(end) is not int or not start+CONTRACT['fit_samples']-1<=end<len(times)-1:
        _reject('Generation must include five prefix samples and leave a development suffix')
    if times[-1]-times[start]>CONTRACT['maximum_duration_seconds']:
        _reject('Full initialized interval exceeds six seconds; do not search a shorter interval')
    if any(b-a>CONTRACT['maximum_replay_gap_seconds'] for a,b in zip(times[start:],times[start+1:])):
        _reject('Source gaps exceed the supported replay bound')
    steps=sum(math.ceil((b-a)/dt) for a,b in zip(times[start:],times[start+1:]))
    if steps>CONTRACT['maximum_physics_steps']:
        _reject('Exact-timestamp rollout exceeds the bounded physics step count')
    authored=assumptions['objects']
    if not isinstance(authored,list) or len(authored)!=len(ids) or {obj.get('id') for obj in authored}!=set(ids):
        _reject('Physical assumptions must cover every proposed object without dropping any')
    profiles={obj['id']:obj for obj in authored}
    identity=_identity_hypotheses(ids,groups,end)
    fit_frames=list(range(start,start+CONTRACT['fit_samples']))
    fit_times=np.array([times[frame]-times[start] for frame in fit_frames])
    if any(b-a>CONTRACT['maximum_fit_gap_seconds'] for a,b in zip(fit_times,fit_times[1:])):
        _reject('Initial fit has an unsupported time gap')
    k=math.log(damping)
    a=fit_times if abs(k)<1e-10 else np.expm1(k*fit_times)/k
    b=fit_times**2/2 if abs(k)<1e-10 else (a-fit_times)/k
    objects,fit_report=[],[]
    for identifier in ids:
        profile=profiles[identifier]
        shape=profile.get('shape')
        geometry_key='radius_px' if shape=='circle' else 'size_px'
        if shape not in ('circle','box') or set(profile)!={'id','role','shape','mass','color',geometry_key}:
            _reject('Every collision shape/size, mass, role and display color must be authored')
        observed=[rows[(frame,identifier)] for frame in fit_frames]
        if any(row['geometry'] is None for row in observed):
            _reject('Initial generation prefix contains missing geometry; no interpolation or frame search')
        positions=np.array([row['geometry']['center_px'] for row in observed])/source['width_px']
        origin=positions[0]
        velocity=np.sum(a[:,None]*(positions-origin-b[:,None]*gravity),axis=0)/np.dot(a,a)
        if profile['role']=='static':
            if max(np.linalg.norm(positions-origin,axis=1))>CONTRACT['maximum_static_displacement_image_widths']:
                _reject('Authored static body moves in the generation prefix')
            velocity=np.zeros(2)
        if np.linalg.norm(velocity)>MAX_SPEED:
            _reject('Estimated initial speed exceeds engine bounds; no clamp')
        fitted=np.broadcast_to(origin,positions.shape) if profile['role']=='static' else origin+a[:,None]*velocity+b[:,None]*gravity
        residual=float(max(np.linalg.norm(fitted-positions,axis=1)))
        if residual>CONTRACT['maximum_fit_residual_image_widths']:
            _reject('Generation prefix contradicts the authored unforced-motion hypothesis')
        obj={'id':identifier,'label':'Model object '+raw_ids[identifier],'role':profile['role'],'shape':shape,
            'position':origin.tolist(),'velocity':velocity.tolist(),'mass':profile['mass'],'color':profile['color'],
            'provenance':{'position':'First generation model box midpoint; inferred, not calibrated',
                'velocity':'Five exact-time generation samples under authored unforced gravity/damping hypothesis',
                'geometry_mass_role':'Authored physical assumptions, not segmented or recovered mechanics',
                'identity':'Raw model ID anchor is an assumption; correspondence alternatives remain unresolved',
                'evidence_ids':[row['id'] for row in observed]}}
        if shape=='circle':
            obj['radius']=_number(profile['radius_px'],.001,4096,'radius')/source['width_px']
        else:
            size=profile['size_px']
            if not isinstance(size,list) or len(size)!=2:
                _reject('Authored box dimensions require two values')
            obj['size']=[_number(value,.001,4096,'box size')/source['width_px'] for value in size]
        objects.append(obj)
        fit_report.append({'object_id':identifier,'frames':fit_frames,'maximum_residual_px':residual*source['width_px'],
                           'initial_velocity_image_widths_per_s':velocity.tolist()})
    target=assumptions['target_px']
    if not isinstance(target,list) or len(target)!=2:
        _reject('Target must be an authored source-image point')
    scene={'schema_version':1,'profile':'planar-rigid-v1','width':1.,'height':source['height_px']/source['width_px'],
        'dt':dt,'gravity':gravity.tolist(),'damping':damping,'restitution':restitution,'objects':objects,
        'controlled_id':assumptions['controlled_id'],'target':[_number(v,0,4096,'target')/source['width_px'] for v in target],
        'task':deepcopy(assumptions['task']),'relationships':[],
        'assumptions':['Counterfactual planar sandbox, not validated reconstruction; camera/plane/closed walls are authored.',
            'Per-shape friction=0.5, circular/box collision geometry, mass, gravity, damping and restitution are authored.',
            'No physical ID, contact, force, material, scale or task-control recovery is claimed.',
            'Initial velocity assumes an unforced generation prefix; same-image fit does not validate changed actions.',
            'Raw model identities remain hypotheses; anonymous replay agreement never promotes this candidate.']}
    if stabilized:
        scene['assumptions'].append('Coordinates are image-registration estimates in the initialization frame; the original camera motion is not reclassified as fixed. Shared depth, physical scale and perspective remain uncalibrated.')
    try:
        scene=validate_scene(scene)
    except ValueError as error:
        _reject('Authored scene rejected without geometry repair: '+str(error))
    pre=[rows[(frame,identifier)] for frame in range(start) for identifier in ids]
    generation=[rows[(frame,identifier)] for frame in range(start,end+1) for identifier in ids]
    return {'schema_version':1,'profile':'segmentation-planar-candidate-v1','status':'executable_hypothesis',
        'physics_ready':False,'physical_validation':'unverified','scene':scene,'contract':deepcopy(CONTRACT),
        'authored_assumptions':deepcopy(assumptions),'source':deepcopy(source),
        'source_timestamps_s':times.copy(),'initialization_frame':start,
        'partitions':{'generation_frames':[start,end],'development_frames':[end+1,len(times)-1],
                      'pre_initialization_frames':[0,start-1] if start else None},
        'coverage':{'pre_initialization_slots':len(pre),'pre_initialization_returned':sum(row['geometry'] is not None for row in pre),
                    'full_source_frames':len(times),'selected_frames':len(times)-start,'discarded_postinit_frames':0},
        'generation_fit':fit_report,'identity_hypotheses':identity,
        'generation_evidence_sha256':canonical_hash(generation),'input_evidence_sha256':canonical_hash(evidence)}


def _anonymous(observed,predicted):
    states={0:(0.,[])}
    for _,point in observed:
        next_states={}
        for used,(cost,pairs) in states.items():
            for index,(identifier,target) in enumerate(predicted):
                if used&(1<<index):
                    continue
                error=math.dist(point,target)
                proposal=(cost+error**2,pairs+[(identifier,error)])
                mask=used|(1<<index)
                if mask not in next_states or proposal[0]<next_states[mask][0]:
                    next_states[mask]=proposal
        states=next_states
        if len(states)>CONTRACT['maximum_anonymous_assignment_states']:
            _reject('Anonymous matching exceeded its bounded state count')
    return min(states.values(),key=lambda item:item[0])[1] if observed else []


def evaluate_candidate(candidate,evidence):
    """Coast replay on all fixed timestamps; scores are diagnostics, never promotion."""
    source,ids,_,times,rows,_,_=_normalize(evidence)
    if (candidate.get('profile')!='segmentation-planar-candidate-v1' or candidate.get('physics_ready') is not False
            or canonical_hash(evidence)!=candidate['input_evidence_sha256'] or source!=candidate['source']
            or times!=candidate['source_timestamps_s']):
        _reject('Replay evidence differs from the fixed candidate source')
    if canonical_hash(candidate)!=canonical_hash(generate_candidate(evidence,candidate['authored_assumptions'])):
        _reject('Candidate differs from its deterministic generation-only construction')
    scene=validate_scene(candidate['scene'])
    if [obj['id'] for obj in scene['objects']]!=ids:
        _reject('Scene object identity order differs')
    start,end=candidate['partitions']['generation_frames']
    if (start!=candidate['initialization_frame'] or candidate['partitions']['development_frames']!=[end+1,len(times)-1]
            or scene['dt']!=candidate['authored_assumptions']['dt']):
        _reject('Candidate partition/time contract changed')
    environment=SceneEnv(scene)
    environment.reset(seed=0,options={'jitter':False})
    snapshots,failures=[],[]
    width=source['width_px']
    previous=times[start]
    try:
        for frame in range(start,len(times)):
            try:
                delta=times[frame]-previous
                steps=math.ceil(delta/scene['dt']) if delta else 0
                contacts=set()
                for _ in range(steps):
                    environment.scene['dt']=delta/steps
                    state=environment.advance_physics(0)
                    contacts.update(tuple(pair) for pair in state['contacts'])
                    if state['speed_limited'] or state['terminated']:
                        raise RuntimeError('Speed limit or escaped state invalidates replay')
                state=environment.snapshot()
                positions={obj['id']:[value*width for value in obj['position']] for obj in state['objects']}
                snapshots.append({'frame':frame,'source_time_s':times[frame],'elapsed_time_s':times[frame]-times[start],
                                  'positions_px':positions,'simulation_contacts':[list(pair) for pair in sorted(contacts)],
                                  'contact_scope':'Simulated under authored geometry, not observed contact'})
                previous=times[frame]
            except (ValueError,RuntimeError,OverflowError) as error:
                failures.append({'frame':frame,'error_type':type(error).__name__,'reason':str(error),
                                 'remaining_frames_not_simulated':len(times)-frame})
                break
    finally:
        environment.close()
    by_frame={row['frame']:row for row in snapshots}
    score_rows=[]
    summaries={}
    for partition,bounds in candidate['partitions'].items():
        if partition not in ('generation_frames','development_frames'):
            continue
        first,last=bounds
        fixed,anonymous=[],[]
        returned=missing=unmatched=failed=0
        for frame in range(max(first,start+1),last+1):
            observed=[(identifier,rows[(frame,identifier)]['geometry']['center_px']) for identifier in ids if rows[(frame,identifier)]['geometry'] is not None]
            returned+=len(observed); missing+=len(ids)-len(observed)
            predicted=by_frame.get(frame)
            if predicted is None:
                failed+=1
                score_rows.append({'frame':frame,'partition':partition,'status':'simulation_unavailable','observed_slots':len(observed),
                                   'fixed_id_errors_px':None,'anonymous_errors_px':None})
                continue
            positions=predicted['positions_px']
            fixed_errors=[{'object_id':identifier,'error_px':math.dist(point,positions[identifier])} for identifier,point in observed]
            assignment=_anonymous(observed,[(identifier,positions[identifier]) for identifier in ids])
            fixed.extend(row['error_px'] for row in fixed_errors)
            anonymous.extend(pair[1] for pair in assignment)
            unmatched+=len(ids)-len(assignment)
            score_rows.append({'frame':frame,'partition':partition,'status':'evaluated','observed_slots':len(observed),
                'missing_observed_slots':len(ids)-len(observed),'unmatched_simulated_slots':len(ids)-len(assignment),
                'fixed_id_errors_px':fixed_errors,
                'anonymous_errors_px':[{'observation_object_id':observed[i][0],'simulation_object_id':pair[0],'error_px':pair[1]}
                                       for i,pair in enumerate(assignment)]})
        def metrics(values):
            return {'count':len(values),'conditional_rmse_px':math.sqrt(sum(v*v for v in values)/len(values)) if values else None,
                    'conditional_max_error_px':max(values) if values else None}
        summaries[partition]={'observed_returned_slots':returned,'missing_observed_slots':missing,
            'unmatched_simulated_slots':unmatched,'simulation_unavailable_frames':failed,
            'fixed_raw_id_hypothesis':metrics(fixed),'anonymous_location_sets':metrics(anonymous)}
    return {'status':'failed_replay' if failures else 'completed_diagnostic_replay','physics_ready':False,
        'physical_validation':'unverified','candidate_sha256':canonical_hash(candidate),'action':'coast_only',
        'snapshots':snapshots,'failures':failures,'rows':score_rows,'partitions':summaries,
        'initialization_frame_unscored':start,'simulation_steps':environment.steps,
        'coverage':deepcopy(candidate['coverage']),
        'scope':'Model-output replay, not independent physical truth. Anonymous assignment is diagnostic only; no raw IDs changed, no candidate selected, no passing score or physics promotion.'}


def export_candidate(evidence_run, assumptions, out):
    """Seal inputs before bounded construction/replay; preserve unsupported outcomes."""
    from pathlib import Path
    import shutil
    from weft.benchmark import seal_suite
    from weft.io import digest, local_time, write_json
    from weft.segmentation_evidence import _Bundle

    bundle=_Bundle(evidence_run,('report.json',))
    source_name=next((name for name in ('stabilized-evidence.json','segmentation-evidence.json','scene-observations.json') if name in bundle.files),None)
    if source_name is None:
        _reject('Sealed run contains no supported spatial observation export')
    evidence=bundle.read(source_name)
    out=Path(out).resolve()
    if out.is_relative_to(bundle.root):
        _reject('Candidate output cannot be inside its immutable evidence input')
    out.mkdir(parents=True,exist_ok=False)
    inputs=out/'inputs'
    inputs.mkdir()
    shutil.copyfile(bundle.root/source_name,inputs/source_name)
    shutil.copyfile(bundle.root/'manifest.json',inputs/'source-manifest.json')
    if digest(inputs/source_name)!=bundle.files[source_name]:
        _reject('Source evidence changed during the input freeze')
    write_json(inputs/'authored-assumptions.json',assumptions)
    implementations=[Path(__file__),Path(__file__).with_name('scene.py'),Path(__file__).with_name('observations.py')]
    if evidence.get('profile') == 'stabilized-segmentation-v1':
        implementations.extend(Path(__file__).with_name(name) for name in ('stabilized_segmentation.py','camera_stabilization.py'))
    for path in implementations:
        shutil.copyfile(path,inputs/path.name)
    write_json(inputs/'report.json',{'status':'frozen_before_candidate','created_at':local_time(),
        'source_verification':bundle.provenance(),'implementation_hashes':{path.name:digest(path) for path in implementations},
        'new_api_calls':0,'new_cost_usd':0,'scope':'Authored sandbox assumptions, not source facts'})
    seal_suite(inputs)
    report={'created_at':local_time(),'status':'prepared','new_api_calls':0,'new_cost_usd':0,
        'physics_ready':False,'physical_validation':'unverified','source_verification':bundle.provenance(),
        'contract':deepcopy(CONTRACT),'assumptions_sha256':canonical_hash(assumptions)}
    try:
        candidate=generate_candidate(evidence,assumptions)
        write_json(out/'candidate.json',candidate)
        write_json(out/'scene.json',candidate['scene'])
        replay=evaluate_candidate(candidate,evidence)
        write_json(out/'replay.json',replay)
        report.update(status=replay['status'],replay_partitions=replay['partitions'],
                      identity_hypotheses=candidate['identity_hypotheses'],coverage=candidate['coverage'],
                      candidate_sha256=canonical_hash(candidate),simulation_steps=replay['simulation_steps'])
    except UnsupportedSegmentationScene as error:
        report.update(status='unsupported_retained',reason=str(error),candidate_created=(out/'candidate.json').exists())
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__)
        raise
    finally:
        write_json(out/'report.json',report)
        seal_suite(out)
    return report
