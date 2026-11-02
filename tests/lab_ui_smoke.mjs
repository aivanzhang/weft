/** Deterministic UI lifecycle checks without a browser or a development server. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {pathToFileURL} from 'node:url';

const html = fs.readFileSync(new URL('../src/weft/assets/lab.html', import.meta.url), 'utf8');
const script = fs.readFileSync(new URL('../src/weft/assets/lab.js', import.meta.url), 'utf8');
const clone = value => JSON.parse(JSON.stringify(value));

class Element {
  constructor(tag = 'div', attributes = {}) {
    this.tagName = tag.toUpperCase();
    this.attributes = {...attributes};
    this.id = attributes.id || '';
    this.dataset = {};
    for (const [key, value] of Object.entries(attributes)) if (key.startsWith('data-')) {
      this.dataset[key.slice(5).replace(/-([a-z])/g, (_, char) => char.toUpperCase())] = value;
    }
    this.hidden = 'hidden' in attributes;
    this.disabled = 'disabled' in attributes;
    this.checked = 'checked' in attributes;
    this.value = attributes.value || '0';
    this.width = Number(attributes.width || 960);
    this.height = Number(attributes.height || 540);
    this.children = [];
    this.listeners = new Map();
    this.textContent = '';
    this.classes = new Set((attributes.class || '').split(' ').filter(Boolean));
    this.classList = {toggle: (name, value) => value ? this.classes.add(name) : this.classes.delete(name),
      add: name => this.classes.add(name), remove: name => this.classes.delete(name), contains: name => this.classes.has(name)};
  }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return this.attributes[name] ?? null; }
  removeAttribute(name) { delete this.attributes[name]; }
  append(...elements) { this.children.push(...elements); }
  replaceChildren(...elements) { this.children = [...elements]; }
  addEventListener(type, listener) { this.listeners.set(type, [...(this.listeners.get(type) || []), listener]); }
  dispatch(type, event = {}) {
    event.preventDefault ??= () => {};
    event.target ??= this;
    for (const listener of this.listeners.get(type) || []) listener(event);
    this[`on${type}`]?.(event);
  }
  click() { if (!this.disabled) this.dispatch('click'); }
  getContext() {
    this.drawCalls ??= [];
    return new Proxy({}, {get: (object, name) => object[name] || ((...args) => {
      if (name === 'fillRect') this.drawCalls.length = 0;
      this.drawCalls.push({name, args, strokeStyle: object.strokeStyle, fillStyle: object.fillStyle, lineWidth: object.lineWidth});
    }), set: (object, name, value) => (object[name] = value, true)});
  }
  setPointerCapture() {}
  releasePointerCapture() {}
  scrollIntoView() { this.scrolled = true; }
  focus() {}
}

function fixture(profile = 'planar', {deferArtifacts = false, spatial = false, mobile = false, wholeBody = false, cameraBody = false, spatialTrained = false, sharedTrained = false, multiObject = false, suppliedGround = false, geometryMode = null, partialSegments = false, multiView = false, limbFit = false, partialPixels = false, floorNormal = null, initialFailure = false, legacyPackageEvent = false, feasibilityStatus = "infeasible", temporalIdentity = false, sourceInterval = false, coverage = null, poseCoverage = null, coverageLoading = 0, generationFailed = false, inspection = null, inspectionLoading = 0, sharedExercise = false, rootAssistance = false, exerciseInitialMetrics = {}, exerciseEngineStatus = 'current', exerciseGeometryTime = 0, missingExerciseState = false, missingExerciseContract = false, exercisePolicy = null, exercisePolicyAvailable = false, rigidObjects = null, sharedProfile = "shared-world-v1", exerciseInterval = null, savedRecordings = [], recordingInspection = null, recordingInspectionLoading = 0, deferRecordingInspection = false, trainingDisabled = false} = {}) {
  const shared = profile === 'shared-world';
  const elements = [];
  const ids = new Map();
  for (const match of html.matchAll(/<([a-z][a-z0-9-]*)\b([^>]*)>/gi)) {
    const attributes = {};
    for (const attr of match[2].matchAll(/([\w-]+)(?:="([^"]*)")?/g)) attributes[attr[1]] = attr[2] ?? '';
    const element = new Element(match[1], attributes);
    elements.push(element);
    if (element.id) ids.set(element.id, element);
  }
  const select = selector => {
    if (selector === '.panel') return elements.filter(element => element.classes.has('panel'));
    if (selector === '[data-action]') return elements.filter(element => 'action' in element.dataset);
    if (selector === '[data-json]') return elements.filter(element => 'json' in element.dataset);
    if (selector === '[data-action][aria-pressed=true]') return select('[data-action]').filter(element => element.getAttribute('aria-pressed') === 'true');
    if (selector.startsWith('#')) return [ids.get(selector.slice(1))].filter(Boolean);
    throw new Error(`Unsupported mock selector: ${selector}`);
  };
  ids.get('stages').querySelectorAll = () => elements.filter(element => 'panel' in element.dataset);
  const documentListeners = new Map(), windowListeners = new Map();
  const listen = (map, type, handler) => map.set(type, [...(map.get(type) || []), handler]);
  const document = {hidden: false, getElementById: id => {
    assert(ids.has(id), `UI references missing HTML element #${id}`);
    return ids.get(id);
  }, querySelectorAll: select, querySelector: selector => select(selector)[0] || null,
  createElement: tag => new Element(tag), addEventListener: (type, handler) => listen(documentListeners, type, handler)};
  const window = {addEventListener: (type, handler) => listen(windowListeners, type, handler)};
  let now = 0, timerId = 0;
  const timers = new Map();
  const setTimeout = (callback, delay = 0) => { const id = ++timerId; timers.set(id, {callback, at: now + delay}); return id; };
  const clearTimeout = id => timers.delete(id);
  const scene = {profile: 'planar-rigid-v1', width: 1, height: .6, dt: 1/30, controlled_id: 'actor',
    target: [.8, .3], task: {position_tolerance: .06}, assumptions: ['Synthetic UI fixture'],
    objects: [{id: 'actor', label: 'Synthetic actor', role: 'active', shape: 'circle', radius: .035, color: '#E0B620'}]};
  let state = {objects: [{id: 'actor', role: 'active', shape: 'circle', position: [.2,.3], velocity: [0,0], angle: 0}], controlled_id: 'actor', target: scene.target,
    steps: 0, time_s: 0, terminated: false, truncated: false, success: false};
  if (profile === 'planar' && multiObject) {
    scene.objects.push(
      {id: 'companion', label: 'Companion puck', role: 'passive', shape: 'circle', radius: .03, color: '#C7795D'},
      {id: 'divider', label: 'Static divider', role: 'static', shape: 'box', size: [.1, .03], color: '#AAA399'},
    );
    state.objects.push(
      {id: 'companion', role: 'passive', shape: 'circle', position: [.45, .3], velocity: [0, 0], angle: 0},
      {id: 'divider', role: 'static', shape: 'box', position: [.5, .52], velocity: [0, 0], angle: 0},
    );
    state.contacts = [['actor', 'companion']];
  }
  if (profile === 'spatial-rigid' || shared) {
    Object.assign(scene, {profile: 'spatial-rigid-v1', bounds: [1, .8, .8], dt: 1/120, controlled_id: 'block',
      target: [.78, .56, .065], task: {position_tolerance: .035, speed_tolerance: .06, acceleration: 12, max_steps: 360},
      objects: [
        {id: 'block', label: 'Controlled block', role: 'active', shape: 'box', size: [.06, .05, .05], color: '#965644'},
        {id: 'companion', label: 'Companion sphere', role: 'passive', shape: 'sphere', radius: .035, color: '#C7795D'},
        {id: 'floor', label: 'Support table', role: 'static', shape: 'box', size: [.96, .76, .04], color: '#D8C9BA'},
      ]});
    delete scene.width; delete scene.height;
    state = {controlled_id: 'block', target: scene.target, steps: 0, time_s: 0, substeps: 4,
      virtual_force: [0, 0, 0], contacts: [['block', 'floor']], contact_count: 1,
      contact_details: [{objects: ['block', 'floor'], position: [.24, .36, .04], distance: -.0002, normal_force: 9.81}],
      objects: [
        {id: 'block', role: 'active', shape: 'box', size: [.06, .05, .05], color: '#965644', position: [.24, .36, .065], quaternion: [1, 0, 0, 0], velocity: [0, 0, 0], angular_velocity: [0, 0, 0]},
        {id: 'companion', role: 'passive', shape: 'sphere', radius: .035, color: '#C7795D', position: [.5, .42, .075], quaternion: [1, 0, 0, 0], velocity: [0, 0, 0], angular_velocity: [0, 0, 0]},
        {id: 'floor', role: 'static', shape: 'box', size: [.96, .76, .04], color: '#D8C9BA', position: [.5, .4, .02], quaternion: [1, 0, 0, 0], velocity: [0, 0, 0], angular_velocity: [0, 0, 0]},
      ], terminated: false, truncated: false, success: false};
  }
  if (profile === 'exercise' || shared) {
    scene.profile=mobile?'exercise-v3':spatial?'exercise-v2':'exercise-v1';scene.joint_names=['left_knee','left_elbow'];
    state.human={height:1.8,segments:[{id:'torso',a:[0,0,.8],b:[0,0,1.5],radius:.1}],
      muscles:[{name:'left_knee_flexor',a:[0,0,.7],b:[0,.1,.3],activation:.4,status:'illustrative'}],
      angles:{left_knee:.2,left_elbow:.3},selected_joint:'left_knee',target_angle:.8};
    if(spatial||mobile)Object.assign(state.human,{view_height:2.5,view_origin:[0,0,.8]});
    if(mobile)state.human.mobility={support_enabled:true,translation:[0,0,0],contact_count:0,contacts:[]};
  }
  if (wholeBody || shared) {
    scene.profile = 'exercise-imitation-v1';
    scene.joint_names = ['torso_twist','torso_lateral','torso_flex', ...['left','right'].flatMap(side =>
      ['shoulder_twist','shoulder_lateral','shoulder','elbow','hip_twist','hip_lateral','hip','knee'].map(joint => `${side}_${joint}`))];
    scene.action_contract = {names:[...scene.joint_names,'root_x','root_y','root_z'],physical_limits:[...Array(19).fill(30),100,100,100]};
    Object.assign(state.human,{control_mode:'direct_motor_actions',angles:Object.fromEntries(scene.joint_names.map(name=>[name,0.])),
      muscles:scene.joint_names.flatMap(name=>['flexor','extensor'].map(kind=>({name:`${name}_${kind}`,a:[0,0,.7],b:[0,.1,.3],activation:0,status:'illustrative'}))),
      mobility:{support_enabled:false,translation:[0,0,0],support_force:[0,0,50],floor_height:0,contact_count:2,contacts:[]}});
    delete state.human.selected_joint; delete state.human.target_angle;
    state.whole_body_metrics={angular_rmse_rad:.12,root_rmse_model_units:.03,max_floor_penetration_model_units:.001,failure:null};
  }
  if (shared) {
    Object.assign(scene, {profile:sharedProfile, bounds:[3,2,2.5], controlled_id:'human',
      task:{object_goal:{target_object_id:'block',position_tolerance:.035,require_human_contact:true}}});
    scene.action_contract.names.push(...['block','companion'].flatMap(id => ['fx','fy','fz'].map(axis => `${id}_${axis}`)));
    scene.action_contract.physical_limits.push(...Array(6).fill(10));
    state.controlled_id='human';
    for (const obj of [...scene.objects,...state.objects]) if (obj.role!=='static') obj.role='passive';
    state.contact_count=2; state.contact_summary={human_object:1,object_object:1,human_floor:0};
    state.contact_details.push({objects:['human:left_hand','block'],kind:'human_object',position:[.24,.36,.11],distance:-.0001,normal_force:2});
    state.contact_details[0].kind='object_object';
    state.coupled_metrics={angular_rmse_rad:.12,root_rmse_model_units:.03,object_distance_world:.15,human_object_contacts_total:1,failure:null};
    delete state.whole_body_metrics;
  }
  if (rigidObjects) {scene.objects = clone(rigidObjects); state.objects = clone(rigidObjects);}
  if (floorNormal) state.human.mobility.floor_normal = floorNormal;
  if (cameraBody) {
    scene.learned_policy_available = false;
    Object.assign(state.human, {view_mode: 'body_follow', view_origin: [18, 4.5, .8]});
    state.human.mobility.translation = [18, 4.5, 0];
    for (const part of [...state.human.segments, ...state.human.muscles]) for (const endpoint of ['a', 'b']) {
      part[endpoint][0] += 18; part[endpoint][1] += 4.5;
    }
  }
  if (geometryMode) scene.human = {root_method:'two-view-body-v1',geometry_mode:geometryMode,geometry_coverage:{
    left_shoulder:{two_view_estimate:59,monocular_prior_missing_points:53},right_shoulder:{monocular_prior_missing_points:112},
    left_hip:{two_view_estimate:90,monocular_prior_missing_points:22},right_hip:{two_view_estimate:80,monocular_prior_missing_points:32}}};
  if (multiView) {
    scene.human.root_method = 'multi-view-body-v1';
    scene.human.camera_view_count = 3;
    scene.human.geometry_coverage.left_shoulder = {multi_view_estimate:104,monocular_prior_missing_points:8};
  }
  if (limbFit) {
    scene.human.root_method = 'camera-body-length-fit-v1';
    scene.human.limb_position_fit = {method:'fixed-length-limb-position-v1',supported_limbs:330,changed_limbs:120};
  }
  if (partialSegments) {
    scene.human.root_method=limbFit?'camera-body-length-fit-v2':multiView?'multi-view-body-v2':'two-view-body-v2';
    scene.human.partial_segments=true;
    scene.human.geometry_coverage.left_shoulder={two_view_estimate:59,two_view_proximal_monocular_distal_and_twist:42,monocular_prior_missing_points:11};
    scene.human.geometry_coverage.left_hip={two_view_estimate:90,two_view_proximal_monocular_distal_and_twist:22};
    scene.human.geometry_coverage.right_hip={two_view_estimate:80,two_view_proximal_monocular_distal_and_twist:15,monocular_prior_missing_points:17};
    if (multiView) for (const values of Object.values(scene.human.geometry_coverage)) {
      for (const key of Object.keys(values)) if (key.startsWith('two_view')) {
        values[key.replace('two_view','multi_view')]=values[key]; delete values[key];
      }
    }
  }
  if (partialPixels) {
    scene.human.root_method='camera-body-partial-pixels-v1';
    scene.human.partial_pixels={fitted_group_poses:10,changed_group_poses:10,depth_measured:false};
    const prefix=multiView?'multi_view':'two_view';
    scene.human.geometry_coverage.left_shoulder[prefix+'_proximal_monocular_distal_and_twist']=32;
    scene.human.geometry_coverage.left_shoulder[prefix+'_proximal_image_constrained_distal']=10;
  }
  if (initialFailure) {state.terminated=true;state.simulation_failure={reason:'Synthetic nonfinite dynamics',scope:'Last valid state shown'};}
  const exerciseTask = sharedExercise ? {profile:'shared-exercise-task-v1',control:{root_assistance:rootAssistance?'explicit_3_axis':'disabled',object_forces:'forbidden'},
    interval:exerciseInterval || {horizon_steps:6,start_offset_s:0},equipment_ids:['block','companion'],contact_requirements:[]} : null;
  const exerciseActionContract = sharedExercise ? {names:scene.action_contract.names.slice(0,rootAssistance?22:19),
    root_assistance:exerciseTask.control.root_assistance,object_force_channels:[]} : null;
  const freshExerciseMetrics = () => ({profile:'shared-exercise-task-v1',simulated_duration_s:0,declared_duration_s:exerciseTask?exerciseTask.interval.horizon_steps/30:.2,clock_status:'observed',
    observed_substeps:0,valid_substeps:0,completed_interval:false,success:false,failure:null,failed_gates:[],
    angular_rmse_rad:null,peak_angular_rmse_rad:0,root_rmse_model_units:null,peak_root_rmse_model_units:0,max_floor_penetration_model_units:0,
    equipment:['block','companion'].map(object_id=>({object_id,displacement_model_units:0,orientation_rad:0,speed_model_units_s:0,angular_speed_rad_s:0,within_limits:true})),
    contact_requirements:[],gates_sha256:'f'.repeat(64)});
  let exerciseMetrics = {...freshExerciseMetrics(),...exerciseInitialMetrics};
  let lastExerciseAction = Array(rootAssistance?22:19).fill(0);
  if(sharedExercise){scene.dt=1/30;scene.learned_policy_available=false;state.substeps=8;state.success=!exerciseMetrics.success;state.time_s=exerciseGeometryTime;state.human.mobility.support_force=[0,0,0];}
  const wireState = () => sharedExercise ? {profile:'shared-exercise-state-v1',engine_state:state,engine_state_status:exerciseEngineStatus,
    ...(missingExerciseState?{}:{task_metrics:exerciseMetrics}),last_action:lastExerciseAction,
    terminated:Boolean(exerciseMetrics.failure||exerciseMetrics.success),truncated:Boolean(exerciseMetrics.completed_interval&&!exerciseMetrics.success&&!exerciseMetrics.failure)} : state;
  let backendReady = false, backendRecording = false, exportStatus = null;
  let savedSessions = clone(savedRecordings);
  const isSpatialUntrained = profile === 'spatial-rigid' && !spatialTrained;
  const noPolicy = sharedExercise ? !exercisePolicyAvailable : cameraBody || isSpatialUntrained || (shared && !sharedTrained) || (trainingDisabled && ['planar','appearance'].includes(profile));
  const trainedActor = profile === 'spatial-rigid' ? (spatialTrained ? 'block' : null) : profile === 'exercise' ? null : 'actor';
  const buildCapabilities = () => ({
    schema_version: 1,
    profile: 'environment-capabilities-v1',
    scene_profile: scene.profile,
    lab_profile: sharedExercise ? 'shared_exercise' : wholeBody ? 'whole_body' : cameraBody ? 'camera_body' : profile,
    entities: {
      articulated_body_present: profile === 'exercise' || shared,
      active_control_target: state.human?.selected_joint || scene.controlled_id,
    },
    controllers: {
      learned_policy_available: !noPolicy,
      trained_target_id: profile === 'exercise' ? (cameraBody ? null : wholeBody ? 'all_direct_motors' : 'left_knee') : trainedActor,
    },
    evidence_separation: {physical_validation: 'unverified'},
  });
  const buildEvidenceProvenance = () => ({
    schema_version: 1,
    physical_validation: 'unverified',
    source_evidence: {
      source_video_present: profile !== 'spatial-rigid',
      source_sha256: profile !== 'spatial-rigid' ? 'a'.repeat(64) : null,
      frame_count: profile !== 'spatial-rigid' ? 90 : null,
      observations_artifact_present: false,
      missing_or_uncertain_summary: profile === 'exercise' ? '82 usable, 5 partial, 3 unusable sampled frames' : null,
    },
    layers: {
      recorded_source: profile !== 'spatial-rigid'
        ? 'Sealed source video and per-frame measurements preserved in run/'
        : 'Analytic or imported scene without sealed source video',
      kinematic_projection: profile === 'exercise' || shared
        ? 'Source-derived joint poses projected onto simplified body geometry (no physics solver)'
        : 'Not applicable (rigid-object simulation)',
      controllers: [
        'manual',
        ...(profile === 'exercise' || shared ? ['authored_reference_controller'] : []),
        ...(!noPolicy ? ['learned_policy'] : []),
      ],
      physical_simulation: `${shared ? 'mujoco_3d_coupled' : profile === 'exercise' ? 'mujoco_3d_articulated' : profile === 'spatial-rigid' ? 'mujoco_3d_rigid' : 'pymunk_2d'} forward dynamics under authored geometry, contact, and actuator assumptions (physical_validation: unverified)`,
      muscle_display: profile === 'exercise' || shared ? 'illustrative_motor_command_not_measured_activation' : 'not_applicable',
    },
  });
  const status = () => ({status: backendReady ? generationFailed ? 'failed' : 'ready' : 'running', profile, directory: '/synthetic/lab',
    ...(sharedExercise?{control_profile:'shared_exercise',exercise_task:exerciseTask,action_contract:missingExerciseContract?null:exerciseActionContract,exercise_policy:exercisePolicy}:{}),
    learned_policy_available: sharedExercise ? exercisePolicyAvailable : !noPolicy,
    ...(backendReady && !generationFailed ? {capabilities: buildCapabilities(), evidence_provenance: buildEvidenceProvenance()} : {}),
    ...(generationFailed ? {error:{message:'Synthetic missing source poses'}} : {}),
    ...(suppliedGround ? {ground_clearance:{status:'descriptive_no_contact_evidence',summary:{minimum:.076,median:.356,maximum:.630,positive_gap_poses:112,reference_poses:112}}} : {}),
    ...(sourceInterval ? {requested_source_interval:{first_frame:2580,last_frame:2760,initialization_frame:2620}} : {}),
    ...(wholeBody || cameraBody || shared ? {reference_feasibility:{status:feasibilityStatus,summary:{max_penetration_model_units:.42,
      failed_source_poses:3,source_pose_checks:10,failed_control_grid_poses:7,control_grid_checks:30}}} : {}),
    ...(temporalIdentity ? {result:{pose_identity:{mode:'temporal',repaired_frames:2}}} : {}),
    events: legacyPackageEvent && backendReady ? [{index:1,stage:'whole_body',message:'Existing packaged run without elapsed time'}] : [{index: 0, stage: backendReady ? 'ready' : 'source', elapsed_seconds: .1, message: 'Synthetic UI fixture'}],
    scene, snapshot: wireState(), recording: backendRecording, sessions: savedSessions,
    ...(exportStatus ? {export:{status:exportStatus}} : {}),
    artifacts: [...new Set([...(backendReady ? [...(profile === 'spatial-rigid' ? ['scene.json', 'interventions.json'] : ['source.mp4', ...(!generationFailed ? ['tracks.gif'] : [])]), ...Object.keys(artifacts)] : []), ...(coverage ? ['coverage-summary.json', 'reconstruction-coverage.json'] : [])])].map(name => ({name,bytes:500}))});
  const artifacts = {'training.json': {completed_episodes: 10, steps: 200, seconds: .3,
    history: [{episode: 10,recent_training_success_rate: .4}]},
    'rl-evaluation.json': {policies: {q_learning: {successes: 3,episodes: 5}, random: {successes: 1,episodes: 5}}},
    'repair-ledger.json': [{proposal: {damping:.8},accepted:true,reason:'Synthetic test'}],
    'reconstruction.json': {before:{rmse_px:10},after:{rmse_px:4}},
    'policy.json': {selected_joint:'left_knee'}};
  if (coverage) artifacts['coverage-summary.json'] = coverage;
  if (sharedExercise) artifacts['exercise-task.json'] = exerciseTask;
  if (noPolicy || sharedExercise) {
    delete artifacts['policy.json']; delete artifacts['training.json']; delete artifacts['rl-evaluation.json'];
    artifacts['reconstruction.json'] = {scope: isSpatialUntrained ? 'Unvalidated 3D rigid scene' : 'Unvalidated camera-aware candidate'};
    artifacts['repair-ledger.json'] = {proposals: []};
  }
  if (sharedExercise && exercisePolicy) artifacts['policy.json'] = {profile:'shared-exercise-policy-v1',scope:'Synthetic attached-policy UI fixture'};
  if (profile === 'spatial-rigid' && spatialTrained) {
    artifacts['policy.json'] = {profile: 'spatial-policy-v1', controlled_id: 'block'};
    artifacts['rl-evaluation.json'] = {policies: {learned: {successes: 8, episodes: 10, mean_return: .5, mean_final_distance_m: .01}, random: {successes: 1, episodes: 10}, coast: {successes: 0, episodes: 10}, brake: {successes: 0, episodes: 10}}};
    artifacts['reconstruction.json'] = {scope: 'Trained 3D rigid scene; physical validation remains unverified'};
    artifacts['repair-ledger.json'] = {proposals: []};
  }
  if (wholeBody) {
    artifacts['reference-feasibility.json'] = {status:feasibilityStatus,scope:'Synthetic floor-feasibility UI test'};
    artifacts['policy.json'] = {algorithm:'whole_body_cem_v1',action_names:scene.action_contract.names};
    artifacts['training.json'] = {completed_generations:3,seconds:.4,history:[.2,.4,.6].map((best_training_return,generation)=>({generation,best_training_return}))};
    artifacts['rl-evaluation.json'] = {policies:Object.fromEntries(['learned','zero','random','authored_feedback'].map((name,index)=>
      [name,{trials:[0,1,2].map(trial=>({success:trial<[2,0,1,3][index]})),mean_return:[.6,.1,.2,.7][index]}]))};
  }
  if (shared && sharedTrained) {
    artifacts['policy.json']={profile:'shared-world-policy-v1',action_names:scene.action_contract.names};
    artifacts['training.json']={completed_generations:3,seconds:.4,history:[.2,.4,.6].map((best_training_return,generation)=>({generation,best_training_return}))};
    artifacts['rl-evaluation.json']={policies:Object.fromEntries(['learned','zero','random','authored_feedback'].map((name,index)=>
      [name,{successes:[2,0,1,3][index],episodes:3,mean_return:[.6,.1,.2,.7][index],mean_final_object_distance_world:.15}]))};
  }
  if(profile==='exercise'||shared)artifacts['human-reference.json']={frames:[0,1.5,3,4.5].map(source_time_s=>({...clone(state),source_time_s,
    ...(shared?{contact_details:[],contact_count:0,contact_summary:{human_object:0},reference_projection:true}:{}),
  }))};
  const requests = [], pending = [];
  let activeSteps = 0, maxActiveSteps = 0;
  function response(value,ok=true) { return {ok, status: ok?200:400, headers: {get: () => '500'}, json: async () => clone(value)}; }
  function fetch(path, options = {}) {
    const request = {path, data: options.body ? JSON.parse(options.body) : null, options};
    requests.push(request);
    if (path === '/api/status') return Promise.resolve(response(status()));
    if (path === '/api/exercise-coverage') return Promise.resolve(response(coverageLoading-- > 0 ? {status:'loading'} : poseCoverage || {status:'unavailable'}));
    if (path === '/api/inspection') return Promise.resolve(response(inspectionLoading-- > 0 ? {status:'loading'} : inspection || {status:'unavailable',reason:'Synthetic run has no compact inspection metadata'}));
    if (path.startsWith('/api/recording-inspection?')) {
      if (deferRecordingInspection) return new Promise(resolve => pending.push({...request, resolve}));
      return Promise.resolve(response(recordingInspectionLoading-- > 0 ? {status:'loading'} : recordingInspection || {status:'unavailable',reason:'Synthetic recording has no failure summary'}));
    }
    if (path.startsWith('/files/')) {
      const filename = decodeURIComponent(path.split('/').at(-1).split('?')[0]);
      assert(filename in artifacts, `Unexpected artifact fetch: ${filename}`);
      if(deferArtifacts)return new Promise(resolve=>pending.push({...request,resolve,artifactValue:artifacts[filename]}));
      return Promise.resolve(response(artifacts[filename]));
    }
    if (path === '/api/step') { activeSteps++; maxActiveSteps = Math.max(maxActiveSteps, activeSteps); }
    return new Promise(resolve => pending.push({...request, resolve}));
  }
  function resolveNext(path) {
    const index = pending.findIndex(request => request.path === path);
    assert(index !== -1, `No pending ${path}; pending: ${pending.map(request => request.path)}`);
    const request = pending.splice(index, 1)[0];
    if('artifactValue' in request){request.resolve(response(request.artifactValue));return;}
    if(path==='/api/joint'){
      assert(!wholeBody, 'Whole-body selector must not call the legacy joint endpoint');
      if(backendRecording){request.resolve(response({error:'Stop recording before changing the controlled joint'},false));return;}
      state=clone(state);state.human.selected_joint=request.data.joint;
    }
    if(path==='/api/actor'){
      assert(profile !== 'exercise', 'Human runs must not call /api/actor');
      if(backendRecording){request.resolve(response({error:'Stop and export the recording before changing its controlled object'},false));return;}
      const target = scene.objects.find(obj => obj.id === request.data.actor);
      if((!target && !(shared && request.data.actor==='human')) || target?.role === 'static'){request.resolve(response({error:'Static or unknown object cannot be controlled'},false));return;}
      for(const obj of scene.objects) if(obj.role !== 'static') obj.role = obj.id === request.data.actor ? 'active' : 'passive';
      scene.controlled_id = request.data.actor;
      state = clone(state); state.controlled_id = request.data.actor;
      for(const obj of state.objects) if(obj.role && obj.role !== 'static') obj.role = obj.id === request.data.actor ? 'active' : 'passive';
    }
    if (path === '/api/step') {
      activeSteps--; state = clone(state); state.steps += request.data.steps; state.time_s = state.steps / 30;
      state.objects[0].position[0] += .01 * request.data.steps;
      if (wholeBody) {
        if (request.data.action !== undefined) {
          assert.equal(request.data.action.length,22);
          assert(request.data.action.every(value=>Number.isFinite(value)&&value>=-1&&value<=1));
        } else assert(request.data.policy===true||request.data.controller==='reference');
        state.human.normalized_action=request.data.action||Array(22).fill(request.data.policy?.1:.2);
        state.human.mobility.support_force=state.human.normalized_action.slice(19).map(value=>value*100);
      }
      if (shared && request.data.action !== undefined) {
        if(Array.isArray(request.data.action)) {
          assert.equal(request.data.action.length,sharedExercise?(rootAssistance?22:19):28);
          assert(request.data.action.every(value=>Number.isFinite(value)&&value>=-1&&value<=1));
          state.human.normalized_action=request.data.action.slice(0,22);
        } else assert(Number.isInteger(request.data.action)&&request.data.action>=0&&request.data.action<=7);
      }
      if(sharedExercise) {
        if(request.data.action===undefined)assert(request.data.policy===true&&exercisePolicyAvailable||['coast','reference'].includes(request.data.controller));
        else assert(Array.isArray(request.data.action),'Exercise never uses object-force integer actions');
        lastExerciseAction=request.data.action||Array(rootAssistance?22:19).fill(request.data.policy?.1:request.data.controller==='coast'?0:.2);
        state.human.normalized_action=[...lastExerciseAction,...(rootAssistance?[]:[0,0,0])];
        state.human.mobility.support_force=rootAssistance?lastExerciseAction.slice(19).map(value=>value*100):[0,0,0];
        exerciseMetrics={...exerciseMetrics,simulated_duration_s:state.time_s,observed_substeps:state.steps*8,valid_substeps:state.steps*8,
          angular_rmse_rad:.12,root_rmse_model_units:.03};
      }
    } else if (path === '/api/reset' || path === '/api/record/start') {
      state = clone(state); state.steps = 0; state.time_s = 0;
      state.objects[0].position = profile === 'spatial-rigid' || shared ? [.24, .36, .065] : [.2,.3];
      state.terminated = state.truncated = state.success = false;
      delete state.simulation_failure;
      if(sharedExercise){exerciseMetrics=freshExerciseMetrics();exerciseEngineStatus='current';lastExerciseAction=Array(rootAssistance?22:19).fill(0);}
    }
    if (path === '/api/record/start') backendRecording = true;
    if (path === '/api/record/stop') {backendRecording = false;exportStatus = 'running';savedSessions = [{name:'trial-001',complete:false}];}
    request.resolve(response({snapshot: wireState(), ...((path==='/api/joint'||path==='/api/actor')?{scene, capabilities: buildCapabilities()}:{}), ...(path.includes('/record/') ? {recording:backendRecording} : {}), ...(path==='/api/record/stop'?{export:'running'}:{})}));
  }
  const storage = new Map();
  vm.runInNewContext(script, {document, window, location: {hash:'#token=synthetic-token',pathname:'/'},
    sessionStorage:{getItem:key=>storage.get(key),setItem:(key,value)=>storage.set(key,value)},
    history:{replaceState(){}}, URLSearchParams, AbortSignal, fetch, setTimeout, clearTimeout,
    performance:{now:()=>now}, console}, {filename:'lab.js'});
  return {ids, pending, requests, timers, resolveNext, document,
    rejectNext(path,message) {const index=pending.findIndex(request=>request.path===path);assert(index>=0);pending.splice(index,1)[0].resolve(response({error:message},false));},
    resolveArtifacts(){for(const request of [...pending])if('artifactValue' in request)resolveNext(request.path);},
    resolveRecording(value) {const index=pending.findIndex(request=>request.path.startsWith('/api/recording-inspection?'));assert(index>=0);pending.splice(index,1)[0].resolve(response(value));},
    backendState:()=>clone(state),
    backendTaskState:()=>clone(wireState()),
    finish(){state=clone(state);state.terminated=true;state.success=true;},
    finishExport(){exportStatus='ready';savedSessions[0].complete=true;},
    setReady() { backendReady = true; },
    maxActiveSteps:()=>maxActiveSteps,
    dispatchWindow(type) { for (const listener of windowListeners.get(type) || []) listener(); },
    visibility(hidden) { document.hidden = hidden; for (const listener of documentListeners.get('visibilitychange') || []) listener(); },
    stage(name) { elements.find(element => element.dataset.panel === name).click(); },
    action(value) { (select('[data-action]').find(element => !element.hidden && element.dataset.action === String(value)) || select('[data-action]').find(element => element.dataset.action === String(value))).click(); },
    async nextTimer() { const entry = [...timers].sort((a,b)=>a[1].at-b[1].at)[0]; assert(entry, 'Expected scheduled timer'); timers.delete(entry[0]); now=entry[1].at; entry[1].callback(); await drain(); },
    async ready() { await drain(); assert.match(ids.get('status').textContent,/running/); backendReady=true; await this.nextTimer(); await drain();
      assert.match(ids.get('status').textContent,/ready/i); assert.equal(ids.get('toggle').disabled,missingExerciseState||missingExerciseContract); assert.equal(ids.get('objects').children.length,rigidObjects?.length ?? ((profile==='spatial-rigid'||shared||multiObject)?3:1));
      if(!sharedExercise)assert.equal(ids.get('scores').children.length,noPolicy?0:(wholeBody||spatialTrained||sharedTrained)?4:2); this.stage('play'); }
  };
}

async function drain() { for(let i=0;i<25;i++) await Promise.resolve(); }

async function basicLifecycle() {
  const ui=fixture(); await ui.ready();
  assert(ui.requests.filter(request=>request.path.startsWith('/files/')).length>=4,'Ready state loads artifacts');
  ui.action(2);await drain();
  assert.equal(ui.pending[0].data.action,2);assert.equal(ui.pending[0].data.steps,3,'A paused direction click makes a visible nudge');
  ui.resolveNext('/api/step');await drain();
  assert.match(ui.ids.get('readout').textContent,/step 3/);
  ui.ids.get('step').click();await drain();assert.equal(ui.pending[0].data.steps,1);
  ui.resolveNext('/api/step');await drain();
  ui.ids.get('record').click();await drain();ui.resolveNext('/api/record/start');await drain();
  assert.match(ui.ids.get('record-status').textContent,/recording/i);assert.equal(ui.ids.get('stop-record').disabled,false);
  assert(ui.pending.some(request=>request.path==='/api/step'),'Record & start begins movement');
  ui.ids.get('reset').click();await drain();ui.resolveNext('/api/step');await drain();ui.resolveNext('/api/reset');await drain();
  assert.match(ui.ids.get('readout').textContent,/step 0/);
  ui.ids.get('stop-record').click();await drain();ui.resolveNext('/api/record/stop');await drain();
  assert.equal(ui.ids.get('stop-record').disabled,true);
  ui.ids.get('toggle').click();await drain();assert(ui.pending.some(request=>request.path==='/api/step'));
  ui.visibility(true);ui.resolveNext('/api/step');await drain();
  assert.match(ui.ids.get('toggle').textContent,/start|run/i);assert.equal(ui.timers.size,0,'Hidden page stops polling and playback');
  ui.visibility(false);await ui.nextTimer();assert.match(ui.ids.get('toggle').textContent,/start|run/i,'Visibility return must not restart movement');
  assert.equal(ui.maxActiveSteps(),1);
}

async function mutationsWaitForPendingStep() {
  const ui=fixture();await ui.ready();ui.ids.get('toggle').click();await drain();
  ui.ids.get('record').click();await drain();
  assert.equal(ui.pending.filter(request=>request.path==='/api/step').length,1);
  ui.resolveNext('/api/step');await drain();
  assert(ui.pending.some(request=>request.path==='/api/record/start'),'Record click during a pending step must be queued, not dropped');
  ui.resolveNext('/api/record/start');await drain();
  ui.ids.get('reset').click();await drain();ui.resolveNext('/api/step');await drain();
  assert(ui.pending.some(request=>request.path==='/api/reset'),'Reset click during a pending step must be queued');
  ui.resolveNext('/api/reset');await drain();
  ui.ids.get('toggle').click();await drain();ui.ids.get('stop-record').click();await drain();
  ui.resolveNext('/api/step');await drain();
  assert(ui.pending.some(request=>request.path==='/api/record/stop'),'Stop recording during a pending step must be queued');
  ui.resolveNext('/api/record/stop');await drain();
  assert.equal(ui.maxActiveSteps(),1);
}

async function restartKeepsOnePlaybackChain() {
  const ui=fixture();await ui.ready();ui.ids.get('toggle').click();await drain();
  ui.ids.get('toggle').click();ui.ids.get('toggle').click();await drain();
  assert.equal(ui.pending.filter(request=>request.path==='/api/step').length,1,'No overlapping step requests');
  ui.resolveNext('/api/step');await drain();
  assert(ui.timers.size<=1,'Pause/restart during an awaited step must leave at most one playback timer');
  ui.dispatchWindow('blur');assert.match(ui.ids.get('toggle').textContent,/start|run/i);
  assert.equal(ui.timers.size,0,'Blur cancels playback timers');
  assert.equal(ui.maxActiveSteps(),1);
}

async function firstReadyDoesNotOverwriteRecordingChange() {
  const ui=fixture('planar',{deferArtifacts:true});await drain();ui.setReady();await ui.nextTimer();
  assert.equal(ui.ids.get('record').disabled,false);
  ui.ids.get('record').click();await drain();ui.resolveNext('/api/record/start');await drain();
  assert.match(ui.ids.get('record-status').textContent,/recording/i);
  ui.resolveArtifacts();await drain();
  assert.match(ui.ids.get('record-status').textContent,/recording/i,'Slow initial artifact load must not overwrite a newer recording response');
}

async function humanViewControlsAndLearning() {
  const ui=fixture('exercise');await ui.ready();
  assert.equal(ui.ids.get('human-controls').hidden,false);
  assert.equal(ui.ids.get('human-reference-controls').hidden,false);
  assert.equal(ui.ids.get('joint').children.length,2);
  assert.equal(ui.ids.get('reference-frame').max,3);
  const before=ui.backendState(),requests=ui.requests.length;
  ui.ids.get('camera-yaw').value='1.2';ui.ids.get('camera-yaw').dispatch('input');
  ui.ids.get('show-muscles').checked=false;ui.ids.get('show-muscles').dispatch('input');
  ui.ids.get('reference-frame').value='1';ui.ids.get('reference-frame').dispatch('input');
  assert.match(ui.ids.get('reference-time').textContent,/1.50 seconds/);
  assert.equal(ui.requests.length,requests,'Camera/reference inspection must not send physics commands');
  assert.deepEqual(ui.backendState(),before,'Render controls must not mutate physical state');
  ui.ids.get('joint').value='left_elbow';ui.ids.get('joint').dispatch('change');await drain();
  ui.resolveNext('/api/joint');await drain();assert.match(ui.ids.get('readout').textContent,/left elbow/);
  ui.ids.get('joint').value='left_knee';ui.ids.get('joint').dispatch('change');await drain();ui.resolveNext('/api/joint');await drain();
  ui.ids.get('reference-servo').click();await drain();
  assert(!ui.pending.some(request=>request.path==='/api/step'),'Selecting a mode does not start movement');
  ui.ids.get('toggle').click();await drain();
  assert.equal(ui.pending.find(request=>request.path==='/api/step').data.action,5,'Reference playback is explicit action 5');
  assert.equal(ui.pending.find(request=>request.path==='/api/step').data.policy,false,'Reference playback is not a learned policy');
  ui.resolveNext('/api/step');await drain();ui.ids.get('toggle').click();
  ui.ids.get('policy').click();await drain();
  assert(!ui.pending.some(request=>request.path==='/api/step'),'Selecting learned mode is separate from starting');
  ui.ids.get('toggle').click();await drain();
  assert.equal(ui.pending.find(request=>request.path==='/api/step').data.policy,true);
  ui.resolveNext('/api/step');await drain();ui.ids.get('toggle').click();
  assert.equal(ui.maxActiveSteps(),1);
}

async function spatialBodyExplainsRotationAndAnchoring() {
  const ui=fixture('exercise',{spatial:true,temporalIdentity:true});await ui.ready();
  assert.match(ui.ids.get('world-description').textContent,/19 rotation axes/);
  assert.match(ui.ids.get('world-description').textContent,/pelvis stays fixed/);
  assert.match(ui.ids.get('control-help').textContent,/Rotate/);
  assert.match(ui.ids.get('evidence-boundary').textContent,/2 frames had left\/right labels reassigned/);
  ui.action(1);await drain();
  assert.equal(ui.pending.find(request=>request.path==='/api/step').data.action,1);
}

async function mobileControlsStayExplicitAndProfileScoped() {
  const ui=fixture('exercise',{mobile:true});await ui.ready();
  assert.equal(ui.ids.get('mobile-controls').hidden,false);
  assert.match(ui.ids.get('world-description').textContent,/three translation axes/);
  assert.match(ui.ids.get('play-boundary').textContent,/not recovered balance/);
  assert.match(ui.ids.get('mobile-status').textContent,/support on/);
  for(const action of [6,7,8,9]) {
    ui.action(action);await drain();
    assert.equal(ui.pending.find(request=>request.path==='/api/step').data.action,action);
    ui.resolveNext('/api/step');await drain();
  }
  ui.ids.get('reference-servo').click();await drain();
  for(const button of ui.document.querySelectorAll('[data-action]')) assert.equal(button.disabled,true);
  const anchored=fixture('exercise',{spatial:true});await anchored.ready();
  assert.equal(anchored.ids.get('mobile-controls').hidden,true);
  for(const button of anchored.document.querySelectorAll('[data-action]').filter(button=>Number(button.dataset.action)>=6)) {
    assert.equal(button.disabled,true);assert.equal(button.hidden,true);
  }
}

async function humanJointSelectionDoesNotConflictWithRecording() {
  const ui=fixture('exercise');await ui.ready();ui.ids.get('record').click();await drain();
  ui.resolveNext('/api/record/start');await drain();
  if(!ui.ids.get('joint').disabled){
    ui.ids.get('joint').value='left_elbow';ui.ids.get('joint').dispatch('change');await drain();
    ui.resolveNext('/api/joint');await drain();
    assert.equal(ui.ids.get('joint').value,'left_knee','Rejected joint change must restore the selected joint in the UI');
    assert.match(ui.ids.get('error').textContent,/Stop recording/);
  }
  assert.equal(ui.backendState().human.selected_joint,'left_knee');
  assert.match(ui.ids.get('record-status').textContent,/recording/i);
}

async function humanHeldKeysKeepExerciseMappingOnRelease() {
  const ui=fixture('exercise');await ui.ready();ui.ids.get('toggle').click();await drain();
  const canvas=ui.ids.get('live');canvas.dispatch('keydown',{key:'ArrowUp'});canvas.dispatch('keydown',{key:'ArrowRight'});
  canvas.dispatch('keyup',{key:'ArrowRight'});ui.resolveNext('/api/step');await drain();await ui.nextTimer();
  assert.equal(ui.pending.find(request=>request.path==='/api/step').data.action,1,'Remaining Up key must flex, not revert to planar brake mapping');
  ui.visibility(true);ui.resolveNext('/api/step');await drain();
}

async function learnedShortcutSelectsTrainedJointBeforeStarting() {
  const ui=fixture('exercise');await ui.ready();
  ui.ids.get('joint').value='left_elbow';ui.ids.get('joint').dispatch('change');await drain();
  ui.resolveNext('/api/joint');await drain();
  ui.stage('evidence');ui.ids.get('start-learned').click();await drain();
  assert.equal(ui.ids.get('play').hidden,false);
  assert.equal(ui.pending.find(request=>request.path==='/api/joint').data.joint,'left_knee');
  assert(!ui.pending.some(request=>request.path==='/api/step'),'Do not run a knee policy against the elbow');
  ui.resolveNext('/api/joint');await drain();ui.resolveNext('/api/reset');await drain();
  const step=ui.pending.find(request=>request.path==='/api/step');
  assert.equal(step.data.policy,true);assert.equal(ui.backendState().human.selected_joint,'left_knee');
  assert.equal(ui.ids.get('policy').getAttribute('aria-pressed'),'true');
  ui.visibility(true);ui.resolveNext('/api/step');await drain();assert.equal(ui.maxActiveSteps(),1);
}

async function nonmanualModesIgnoreManualInput() {
  const ui=fixture('exercise');await ui.ready();ui.ids.get('policy').click();await drain();
  const requests=ui.requests.length;
  ui.action(1);ui.ids.get('live').dispatch('keydown',{key:'ArrowUp'});await drain();
  assert.equal(ui.requests.length,requests,'Manual input cannot silently run or replace the chosen controller');
  ui.ids.get('step').click();await drain();
  assert.equal(ui.pending.find(request=>request.path==='/api/step').data.policy,true);
  ui.resolveNext('/api/step');await drain();ui.ids.get('mode-manual').click();ui.action(1);await drain();
  const manual=ui.pending.find(request=>request.path==='/api/step');
  assert.equal(manual.data.policy,false);assert.equal(manual.data.action,1);
  ui.resolveNext('/api/step');await drain();
}

async function navigationCancelsAutomaticDemoStart() {
  const ui=fixture('exercise');await ui.ready();ui.ids.get('start-learned').click();await drain();
  assert(ui.pending.some(request=>request.path==='/api/reset'));
  ui.stage('world');ui.resolveNext('/api/reset');await drain();
  assert(!ui.pending.some(request=>request.path==='/api/step'),'Finishing setup on another tab must not start hidden movement');
  assert.equal(ui.ids.get('play').hidden,true);
  assert.match(ui.ids.get('toggle').textContent,/start|run/i);
}

async function referencePreviewStaysLocalAndStopsWhenHidden() {
  const ui=fixture('exercise');await ui.ready();ui.ids.get('start-reference').click();await drain();
  assert.equal(ui.ids.get('world').hidden,false);
  const requests=ui.requests.length,before=ui.backendState();
  ui.ids.get('reference-play').click();await drain();await ui.nextTimer();
  assert.equal(ui.requests.length,requests,'Pose playback must not send simulator commands');
  assert.deepEqual(ui.backendState(),before);
  assert(ui.timers.size>0,'Exercise fixture must still be playing when visibility changes');
  ui.visibility(true);assert.equal(ui.timers.size,0,'Hidden page stops pose replay timers');
}

async function finishedTrialRestartsInSelectedMode() {
  const ui=fixture('exercise');await ui.ready();ui.ids.get('policy').click();
  ui.ids.get('step').click();await drain();ui.finish();ui.resolveNext('/api/step');await drain();
  ui.ids.get('toggle').click();await drain();
  assert(ui.pending.some(request=>request.path==='/api/reset'),'Starting a finished trial first resets it');
  ui.resolveNext('/api/reset');await drain();
  assert.equal(ui.pending.find(request=>request.path==='/api/step').data.policy,true);
  ui.visibility(true);ui.resolveNext('/api/step');await drain();
}

async function recordingStartsSelectedControllerAfterJointSetup() {
  const ui=fixture('exercise');await ui.ready();
  ui.ids.get('joint').value='left_elbow';ui.ids.get('joint').dispatch('change');await drain();ui.resolveNext('/api/joint');await drain();
  ui.ids.get('policy').click();ui.ids.get('record').click();await drain();
  assert.equal(ui.pending.find(request=>request.path==='/api/joint').data.joint,'left_knee');
  assert(!ui.pending.some(request=>request.path==='/api/record/start'),'Joint setup must finish before recording locks the joint');
  ui.resolveNext('/api/joint');await drain();ui.resolveNext('/api/record/start');await drain();
  assert.equal(ui.ids.get('joint').disabled,true);
  assert.equal(ui.pending.find(request=>request.path==='/api/step').data.policy,true);
  ui.ids.get('stop-record').click();await drain();ui.resolveNext('/api/step');await drain();
  ui.resolveNext('/api/record/stop');await drain();
  assert.equal(ui.ids.get('stop-record').disabled,true);
  assert.equal(ui.maxActiveSteps(),1);
}

async function savedRecordingExposesVerifiedReplayLinks() {
  const ui=fixture();await ui.ready();ui.ids.get('record').click();await drain();
  ui.resolveNext('/api/record/start');await drain();ui.ids.get('stop-record').click();await drain();
  ui.resolveNext('/api/step');await drain();ui.resolveNext('/api/record/stop');await drain();
  assert.equal(ui.ids.get('record').disabled,true,'Do not start another recording while its predecessor is exporting');
  await ui.nextTimer();
  let links=ui.ids.get('recordings').children[0].children;
  assert.equal(links.length,1,'Incomplete export offers only preserved action data');
  assert.match(links[0].href,/sessions\/trial-001\/events.jsonl/);
  ui.finishExport();await ui.nextTimer();
  links=ui.ids.get('recordings').children[0].children;
  assert.deepEqual(links.map(link=>link.textContent),['Action data','Interactive replay','Video','Preview']);
  assert(links.every(link=>link.href.includes('token=synthetic-token')),'Saved artifacts retain local authorization');
  assert.equal(ui.ids.get('record').disabled,false);
  assert.equal(ui.timers.size,0,'Completed export stops status polling');
}

async function selectedSourceRemainsExplicitBeforeReady() {
  const ui=fixture('appearance',{sourceInterval:true});await drain();
  assert.equal(ui.ids.get('source-selection').hidden,false);
  assert.match(ui.ids.get('source-selection').textContent,/Requested section: original frames 2580–2760/);
  assert.match(ui.ids.get('source-selection').textContent,/Tracking starts at frame 2620/);
  assert.match(ui.ids.get('source-selection').textContent,/check the decode audit/);
  assert.equal(ui.ids.get('toggle').disabled,true);
  await ui.ready();
  assert.match(ui.ids.get('source-selection').textContent,/full source is preserved/);
}


async function fullIntervalCoverageIsVisibleBeforeReadyAndLoadedOnce() {
  const coverage = {status: 'planned', requested_interval: [80, 119], requested_frames: 40,
    model_interval: [85, 119], pre_model_frames: 5, measured_observation_slots: 35,
    observation_slots: 40, missing_observation_slots: 5};
  const ui = fixture('appearance', {coverage}); await drain();
  assert.equal(ui.ids.get('coverage-details').hidden, false);
  const label = ui.ids.get('coverage-text').textContent;
  assert.match(label, /frames 80–119 \(40 frames\)/);
  assert.match(label, /planned for frames 85–119, with no restart at gaps/);
  assert.match(label, /35\/40; 5 unavailable/);
  assert.match(label, /does not mean the reconstruction passed/);
  await ui.ready();
  assert.equal(ui.ids.get('coverage-text').textContent, label);
  assert.equal(ui.requests.filter(request => request.path.includes('/coverage-summary.json')).length, 1);
  const unavailable = fixture('planar', {coverage: {...coverage, status: 'unavailable'}}); await drain();
  assert.match(unavailable.ids.get('coverage-text').textContent, /No supported simulation interval/);
  const legacy = fixture('planar'); await legacy.ready();
  assert.match(legacy.ids.get('coverage-text').textContent, /no full-interval coverage report/);
  assert.equal(legacy.ids.get('coverage-inspect').hidden, true);
}

async function existingWholeBodyPackageEventsRenderWithoutInventedTime() {
  const ui=fixture('exercise',{wholeBody:true,legacyPackageEvent:true});await ui.ready();
  assert.equal(ui.ids.get('progress').value,7);
  assert.equal(ui.ids.get('events').children[0].textContent,'whole_body · Existing packaged run without elapsed time');
  assert(!ui.ids.get('events').children[0].textContent.includes('NaN'));
}

async function reloadedSimulationFailureShowsLastValidStateAndResetClearsOnlyItsMessage() {
  const ui=fixture('exercise',{wholeBody:true,initialFailure:true});await ui.ready();
  assert.equal(ui.ids.get('error').hidden,false);
  assert.match(ui.ids.get('error').textContent,/Synthetic nonfinite dynamics/);
  assert.match(ui.ids.get('error').textContent,/Showing the last valid state/);
  ui.ids.get('reset').click();await drain();ui.resolveNext('/api/reset');await drain();
  assert.equal(ui.ids.get('error').hidden,true);
  assert.equal(ui.backendState().simulation_failure,undefined);
  const other=fixture('exercise',{wholeBody:true,initialFailure:true});await other.ready();
  other.ids.get('record').click();await drain();
  other.rejectNext('/api/record/start','Unrelated recording storage failure');await drain();
  assert.match(other.ids.get('error').textContent,/Unrelated recording storage failure/);
  other.ids.get('reset').click();await drain();other.resolveNext('/api/reset');await drain();
  assert.equal(other.ids.get('error').hidden,false);
  assert.match(other.ids.get('error').textContent,/Unrelated recording storage failure/);
}

async function wholeBodyPreflightWarningDoesNotOverrideScoresOrBlockRecording() {
  const ui=fixture('exercise',{wholeBody:true});await ui.ready();
  assert.equal(ui.ids.get('reference-feasibility').hidden,false);
  assert.match(ui.ids.get('feasibility-notice').textContent,/Reference has floor conflicts; this sandbox is a diagnostic simulation/);
  assert.match(ui.ids.get('feasibility-notice').textContent,/Maximum reference penetration: 0.420 model units/);
  assert.match(ui.ids.get('feasibility-notice').textContent,/Conflicting source poses: 3\/10/);
  assert.match(ui.ids.get('feasibility-notice').textContent,/Selected RL window results remain separate/);
  assert.equal(ui.ids.get('record').disabled,false);
  assert.equal(ui.ids.get('scores').children[0].children[0].textContent,'2/3');
  const inspect=ui.document.querySelectorAll('[data-json]').find(button=>button.dataset.json==='reference-feasibility.json');
  inspect.click();await drain();
  assert.equal(ui.ids.get('inspector').hidden,false);
  assert.equal(JSON.parse(ui.ids.get('json').textContent).status,'infeasible');
  const checked=fixture('exercise',{wholeBody:true,feasibilityStatus:'feasible'});await checked.ready();
  assert.match(checked.ids.get('feasibility-notice').textContent,/not validated physical reconstruction/);
  const legacy=fixture('exercise');await legacy.ready();
  assert.equal(legacy.ids.get('reference-feasibility').hidden,true);
}

async function wholeBodyLearnedControlsWaitForPolicyContract() {
  const ui=fixture('exercise',{wholeBody:true,deferArtifacts:true});
  await drain();ui.setReady();await ui.nextTimer();await drain();
  assert.equal(ui.ids.get('policy').disabled,true);
  assert.equal(ui.ids.get('start-learned').disabled,true);
  const before=ui.requests.length;ui.ids.get('start-learned').click();await drain();
  assert.equal(ui.requests.length,before,'Do not start learned control before its policy contract loads');
  for(let i=0;i<4;i++){ui.resolveArtifacts();await drain();}
  assert.equal(ui.ids.get('policy').disabled,false);
  assert.equal(ui.ids.get('start-learned').disabled,false);
}

async function wholeBodyModesUseExplicitContractAndShowHonestMetrics() {
  const ui=fixture('exercise',{wholeBody:true});await ui.ready();
  assert.equal(ui.ids.get('joint').children.length,22);
  assert.equal(ui.ids.get('joint-label').textContent,'Motor to control');
  assert.equal(ui.ids.get('policy').disabled,false,'Whole-body policy does not require a trained joint');
  assert.equal(ui.ids.get('mobile-controls').hidden,true);
  assert.equal(ui.ids.get('whole-body-status').hidden,false);
  assert.match(ui.ids.get('readout').textContent,/angular RMSE 0.120 rad/);
  assert.match(ui.ids.get('readout').textContent,/root RMSE 0.030 model units/);
  assert.match(ui.ids.get('readout').textContent,/floor penetration 0.001/);
  assert(!ui.ids.get('readout').textContent.includes('undefined'));
  assert.match(ui.ids.get('whole-body-status').textContent,/Virtual root forces \[0.0, 0.0, 50.0\] N/);
  for(const button of ui.document.querySelectorAll('[data-action]')) {
    const incompatible=Number(button.dataset.action)>2;
    assert.equal(button.hidden,incompatible);
    if(incompatible)assert.equal(button.disabled,true);
  }
  const labels=ui.ids.get('scores').children.map(metric=>metric.children[1].textContent);
  assert.deepEqual(ui.ids.get('scores').children.map(metric=>metric.children[0].textContent),['2/3','0/3','1/3','3/3']);
  assert(labels.some(label=>label.includes('Learned full-body controller')));
  assert(labels.some(label=>label.includes('Zero motor force')));
  assert(labels.some(label=>label.includes('Random actions')));
  assert(labels.some(label=>label.includes('Authored feedback (not learned)')));
  assert.match(ui.ids.get('training-caption').textContent,/3 completed generations/);
  assert(!ui.ids.get('training-caption').textContent.includes('undefined'));
  ui.ids.get('policy').click();ui.ids.get('step').click();await drain();
  assert.deepEqual(ui.pending.find(request=>request.path==='/api/step').data,{steps:1,policy:true});
  ui.resolveNext('/api/step');await drain();
  ui.ids.get('reference-servo').click();ui.ids.get('step').click();await drain();
  assert.deepEqual(ui.pending.find(request=>request.path==='/api/step').data,{steps:1,controller:'reference'});
  assert.match(ui.ids.get('mode-description').textContent,/baseline is not learned/);
  ui.resolveNext('/api/step');await drain();
  assert(!ui.requests.some(request=>request.path==='/api/joint'));
}

async function wholeBodyManualSelectorAndReleaseNeverImplyOtherForces() {
  const ui=fixture('exercise',{wholeBody:true});await ui.ready();
  const before=ui.requests.length;
  ui.ids.get('joint').value='root_z';ui.ids.get('joint').dispatch('change');await drain();
  assert.equal(ui.requests.length,before,'Motor selection is client-only');
  ui.action(1);await drain();
  let pending=ui.pending.find(request=>request.path==='/api/step');
  assert.deepEqual(pending.data,{steps:3,action:[...Array(21).fill(0),.25]});
  ui.resolveNext('/api/step');await drain();
  ui.action(2);await drain();
  assert.deepEqual(ui.pending.find(request=>request.path==='/api/step').data.action,[...Array(21).fill(0),-.25]);
  ui.resolveNext('/api/step');await drain();
  ui.action(0);await drain();
  assert.deepEqual(ui.pending.find(request=>request.path==='/api/step').data.action,Array(22).fill(0));
  ui.resolveNext('/api/step');await drain();
  ui.ids.get('live').dispatch('keydown',{key:' '});await drain();
  assert.deepEqual(ui.pending.find(request=>request.path==='/api/step').data.action,Array(22).fill(0));
  ui.resolveNext('/api/step');await drain();
  ui.ids.get('joint').value='left_elbow';ui.ids.get('joint').dispatch('change');
  ui.action(1);await drain();
  pending=ui.pending.find(request=>request.path==='/api/step');
  assert.equal(pending.data.action.filter(value=>value!==0).length,1);
  assert.equal(pending.data.action[6],.25);
  ui.resolveNext('/api/step');await drain();
  ui.ids.get('policy').click();
  const count=ui.requests.length;ui.action(1);ui.ids.get('live').dispatch('keydown',{key:'ArrowUp'});await drain();
  assert.equal(ui.requests.length,count);
}

async function wholeBodyRecordingAndSourcePreviewPreserveLifecycle() {
  const ui=fixture('exercise',{wholeBody:true});await ui.ready();
  ui.stage('evidence');ui.ids.get('start-learned').click();await drain();
  assert(ui.pending.some(request=>request.path==='/api/reset'));
  assert(!ui.requests.some(request=>request.path==='/api/joint'));
  ui.resolveNext('/api/reset');await drain();
  assert.deepEqual(ui.pending.find(request=>request.path==='/api/step').data,{steps:3,policy:true});
  ui.ids.get('record').click();await drain();ui.resolveNext('/api/step');await drain();
  ui.resolveNext('/api/record/start');await drain();
  assert.deepEqual(ui.pending.find(request=>request.path==='/api/step').data,{steps:3,policy:true});
  ui.resolveNext('/api/step');await drain();ui.ids.get('toggle').click();
  ui.ids.get('mode-manual').click();
  assert.equal(ui.ids.get('joint').disabled,false,'Client-only motor selection stays available while recording');
  const count=ui.requests.length;ui.ids.get('joint').value='root_x';ui.ids.get('joint').dispatch('change');await drain();
  assert.equal(ui.requests.length,count);
  ui.action(2);await drain();
  assert.deepEqual(ui.pending.find(request=>request.path==='/api/step').data.action,[...Array(19).fill(0),-.25,0,0]);
  ui.ids.get('stop-record').click();await drain();ui.resolveNext('/api/step');await drain();
  ui.resolveNext('/api/record/stop');await drain();
  assert.equal(ui.ids.get('stop-record').disabled,true);
  ui.finishExport();await ui.nextTimer();
  assert.equal(ui.ids.get('recordings').children[0].children.length,4);
  const before=ui.backendState(),requests=ui.requests.length;
  ui.stage('world');ui.ids.get('reference-frame').value='1';ui.ids.get('reference-frame').dispatch('input');
  assert.deepEqual(ui.backendState(),before);
  assert.equal(ui.requests.length,requests);
  assert.equal(ui.maxActiveSteps(),1);
}

async function groundGridMatchesSimulatedPlaneWithoutChangingBody() {
  for (const floorNormal of [null, [0, .6, .8]]) {
    const ui = fixture('exercise', {mobile: true, floorNormal}); await ui.ready();
    const before = ui.backendState(), requests = ui.requests.length;
    const yaw = Number(ui.ids.get('camera-yaw').value), scale = 640 / 3;
    const calls = ui.ids.get('live').drawCalls.filter(call => ['moveTo', 'lineTo'].includes(call.name));
    for (let endpoint = 0; endpoint < 2; endpoint++) {
      const x = -1, y = endpoint ? 1 : -1, z = floorNormal ? -.75 * y : 0;
      const expected = [480 + (x * Math.cos(yaw) - y * Math.sin(yaw)) * scale,
        320 - ((z - .8) * Math.cos(.2) - (x * Math.sin(yaw) + y * Math.cos(yaw)) * Math.sin(.2)) * scale];
      calls[endpoint].args.forEach((value, axis) => assert(Math.abs(value - expected[axis]) < 1e-9));
    }
    assert.deepEqual(ui.backendState(), before);
    assert.equal(ui.requests.length, requests);
  }
}

const checks=[fullIntervalCoverageIsVisibleBeforeReadyAndLoadedOnce,groundGridMatchesSimulatedPlaneWithoutChangingBody,existingWholeBodyPackageEventsRenderWithoutInventedTime,reloadedSimulationFailureShowsLastValidStateAndResetClearsOnlyItsMessage,wholeBodyPreflightWarningDoesNotOverrideScoresOrBlockRecording,wholeBodyLearnedControlsWaitForPolicyContract,wholeBodyModesUseExplicitContractAndShowHonestMetrics,wholeBodyManualSelectorAndReleaseNeverImplyOtherForces,wholeBodyRecordingAndSourcePreviewPreserveLifecycle,selectedSourceRemainsExplicitBeforeReady,basicLifecycle,mutationsWaitForPendingStep,restartKeepsOnePlaybackChain,
  firstReadyDoesNotOverwriteRecordingChange,humanViewControlsAndLearning,spatialBodyExplainsRotationAndAnchoring,mobileControlsStayExplicitAndProfileScoped,
  humanJointSelectionDoesNotConflictWithRecording,humanHeldKeysKeepExerciseMappingOnRelease,
  learnedShortcutSelectsTrainedJointBeforeStarting,nonmanualModesIgnoreManualInput,
  navigationCancelsAutomaticDemoStart,referencePreviewStaysLocalAndStopsWhenHidden,
  finishedTrialRestartsInSelectedMode,recordingStartsSelectedControllerAfterJointSetup,
  savedRecordingExposesVerifiedReplayLinks];
async function cameraAwareCandidateKeepsManualControlsAndMissingLearningExplicit() {
  const ui = fixture('exercise', {mobile: true, cameraBody: true}); await ui.ready();
  assert.equal(ui.ids.get('policy').disabled, true);
  assert.equal(ui.ids.get('start-learned').disabled, true);
  assert.equal(ui.ids.get('reference-feasibility').hidden, false);
  assert.match(ui.ids.get('feasibility-notice').textContent, /floor conflicts/);
  assert.match(ui.ids.get('training-caption').textContent, /Not trained/);
  assert.match(ui.ids.get('world-description').textContent, /view follows the body/);
  assert(!ui.requests.some(request => /\/(policy|training|rl-evaluation)\.json/.test(request.path)));
  const before = ui.requests.length;
  ui.ids.get('start-learned').click(); ui.ids.get('policy').click(); await drain();
  assert.equal(ui.requests.length, before);
  const lines = ui.ids.get('live').drawCalls.filter(call => ['moveTo', 'lineTo'].includes(call.name));
  assert(lines.slice(0, 2).every(call => call.args.every(value => value > 0 && value < 960)), 'World grid remains near the traveling body');
  const state = ui.backendState();
  ui.ids.get('camera-yaw').value = '1'; ui.ids.get('camera-yaw').dispatch('input');
  assert.deepEqual(ui.backendState(), state, 'Changing the view never changes global recorded state');
  ui.action(2); await drain();
  assert.deepEqual(ui.pending[0].data, {action: 2, steps: 3, policy: false});
  ui.resolveNext('/api/step'); await drain();
  ui.ids.get('record').click(); await drain();
  assert(ui.pending.some(request => request.path === '/api/record/start'));
  ui.resolveNext('/api/record/start'); await drain();
  assert(ui.pending.some(request => request.path === '/api/step'));
  assert.equal(ui.maxActiveSteps(), 1);
}

checks.push(cameraAwareCandidateKeepsManualControlsAndMissingLearningExplicit);
async function suppliedGroundDoesNotTurnCollisionClearanceIntoContactValidation() {
  const ui = fixture('exercise', {mobile:true,cameraBody:true,suppliedGround:true,feasibilityStatus:'feasible'}); await ui.ready();
  assert.match(ui.ids.get('feasibility-notice').textContent,/not validated physical reconstruction/);
  assert.match(ui.ids.get('feasibility-notice').textContent,/minimum 0.076, median 0.356, maximum 0.630/);
  assert.match(ui.ids.get('feasibility-notice').textContent,/Contact timing has not been inferred/);
  assert.match(ui.ids.get('world-description').textContent,/Gaps above the ground remain visible/);
  assert.equal(ui.ids.get('policy').disabled,true);
  assert(!ui.requests.some(request=>/ground-clearance/.test(request.path)),'Compact ready metadata avoids another artifact fetch');
}
checks.push(suppliedGroundDoesNotTurnCollisionClearanceIntoContactValidation);
async function partialTwoViewBodyDisclosesMissingLimbsWithoutExtraRequests() {
  const ui = fixture('exercise', {mobile:true,cameraBody:true,suppliedGround:true,geometryMode:'both'}); await ui.ready();
  assert.match(ui.ids.get('world-description').textContent,/body lengths and available joint directions/);
  assert.match(ui.ids.get('world-description').textContent,/right arm 0\/112/);
  assert.match(ui.ids.get('world-description').textContent,/single-view estimates/);
  assert.match(ui.ids.get('training-caption').textContent,/two-view-body.json/);
  assert.equal(ui.ids.get('policy').disabled,true);
  assert(!ui.requests.some(request=>/two-view-(body|triangulation)/.test(request.path)));
}
checks.push(partialTwoViewBodyDisclosesMissingLimbsWithoutExtraRequests);

async function multiViewBodyCountsSupportedLimbsWithoutExtraRequests() {
  const ui = fixture('exercise', {mobile:true,cameraBody:true,suppliedGround:true,geometryMode:'both',multiView:true}); await ui.ready();
  assert.match(ui.ids.get('world-description').textContent,/3 views set position/);
  assert.match(ui.ids.get('world-description').textContent,/left arm 104\/112/);
  assert.match(ui.ids.get('world-description').textContent,/right arm 0\/112/);
  assert.match(ui.ids.get('world-description').textContent,/single-view estimates/);
  assert.match(ui.ids.get('training-caption').textContent,/multiple views/);
  assert.equal(ui.ids.get('policy').disabled,true);
  assert(!ui.requests.some(request=>/two-view-(body|triangulation)/.test(request.path)));
}
checks.push(multiViewBodyCountsSupportedLimbsWithoutExtraRequests);

async function fixedLengthFitRemainsAnUntrainedSourceProposal() {
  for (const multiView of [false,true]) {
    const ui = fixture('exercise', {mobile:true,cameraBody:true,suppliedGround:true,geometryMode:'both',multiView,limbFit:true}); await ui.ready();
    assert.match(ui.ids.get('world-description').textContent,/bounded angle fit/);
    assert.match(ui.ids.get('world-description').textContent,/fixed limb lengths/);
    assert.match(ui.ids.get('world-description').textContent,/does not establish contact/);
    assert.match(ui.ids.get('training-caption').textContent,/Source observations are unchanged/);
    assert.equal(ui.ids.get('policy').disabled,true);
    assert(!ui.requests.some(request=>/two-view-(body|triangulation)/.test(request.path)));
  }
}
checks.push(fixedLengthFitRemainsAnUntrainedSourceProposal);

async function fittedPartialLimbsKeepMissingEndpointsVisible() {
  for (const multiView of [false,true]) for (const limbFit of [false,true]) {
    const ui = fixture('exercise',{mobile:true,cameraBody:true,suppliedGround:true,geometryMode:'both',partialSegments:true,multiView,limbFit}); await ui.ready();
    assert.match(ui.ids.get('world-description').textContent,/left arm 59 complete \+ 42 upper-only \/ 112/);
    assert.match(ui.ids.get('world-description').textContent,/missing endpoint, bend and twist/);
    assert.equal(ui.ids.get('world-description').textContent.includes('bounded angle fit'),limbFit);
    if (limbFit) assert.match(ui.ids.get('world-description').textContent,/For complete limbs, a bounded angle fit/);
    assert.equal(ui.ids.get('policy').disabled,true);
    assert(!ui.requests.some(request=>/two-view-(body|triangulation)/.test(request.path)));
  }
}
checks.push(fittedPartialLimbsKeepMissingEndpointsVisible);

async function partialPixelFitNeverClaimsObservedDepthOrContact() {
  for (const multiView of [false,true]) {
    const ui = fixture('exercise',{mobile:true,cameraBody:true,suppliedGround:true,geometryMode:'both',partialSegments:true,multiView,limbFit:true,partialPixels:true}); await ui.ready();
    assert.match(ui.ids.get('world-description').textContent,/left arm 59 complete \+ 32 upper-only \+ 10 image-constrained \/ 112/);
    assert.match(ui.ids.get('world-description').textContent,/upper-bone direction stays fixed; depth is still inferred/);
    assert.match(ui.ids.get('training-caption').textContent,/excluded view and rejected proposal/);
    assert.match(ui.ids.get('training-caption').textContent,/not an observed 3D joint or verified contact/);
    assert.equal(ui.ids.get('policy').disabled,true);
    assert(!ui.requests.some(request=>/two-view-(body|triangulation)/.test(request.path)));
  }
}
checks.push(partialPixelFitNeverClaimsObservedDepthOrContact);


async function upperSegmentCorrectionDoesNotClaimObservedWristsOrAnkles() {
  const ui = fixture('exercise', {mobile:true,cameraBody:true,suppliedGround:true,geometryMode:'both',partialSegments:true}); await ui.ready();
  assert.match(ui.ids.get('world-description').textContent,/left arm 59 complete \+ 42 upper-only \/ 112/);
  assert.match(ui.ids.get('world-description').textContent,/right arm 0 complete \+ 0 upper-only \/ 112/);
  assert.match(ui.ids.get('world-description').textContent,/endpoint, bend and twist around the bone remain single-view estimates/);
  assert.equal(ui.ids.get('policy').disabled,true);
  assert(!ui.requests.some(request=>/two-view-(body|triangulation)/.test(request.path)));
}
checks.push(upperSegmentCorrectionDoesNotClaimObservedWristsOrAnkles);

const poseCoverageFixture = {
  status:'available',decoded_frames:12,sampled_frames:6,unsampled_frames:6,
  usable_samples:3,partial_samples:1,unusable_samples:3,competing_samples:1,
  markers:[[0,0,'tracked'],[2,.103,'tracked_partial'],[4,.203,'unusable'],[6,.303,'competing'],[8,.403,'unusable'],[10,.503,'tracked']],
  movement:{status:'insufficient_reference',long_gaps:1,unfilled_gaps:0},
  issues:[{kind:'long_movement_gap',first_frame:2,last_frame:10,seek_s:.103},
    {kind:'unusable_pose',first_frame:4,last_frame:8,seek_s:.203}]
};

async function poseCoverageSeeksSourceWithoutChangingSimulationOrFetchingRawPoses() {
  const ui=fixture('exercise',{poseCoverage:poseCoverageFixture});await ui.ready();
  assert.match(ui.ids.get('pose-coverage-text').textContent,/3\/6 analyzed frames/);
  assert.match(ui.ids.get('pose-coverage-text').textContent,/6 were not analyzed/);
  assert.match(ui.ids.get('pose-coverage-text').textContent,/1 long gaps/);
  assert.equal(ui.ids.get('pose-coverage').hidden,false);
  assert(!ui.requests.some(request=>/\/(evidence|root-motion)\.json/.test(request.path)));
  const before=ui.backendState(), requests=ui.requests.length;
  ui.ids.get('source').readyState=1;
  ui.ids.get('pose-next-issue').click();
  assert.equal(ui.ids.get('source').currentTime,.103);
  assert.match(ui.ids.get('pose-issue').textContent,/1\/2.*frames 2–10/);
  ui.ids.get('pose-next-issue').click();
  assert.equal(ui.ids.get('source').currentTime,.203);
  assert.deepEqual(ui.backendState(),before);
  assert.equal(ui.requests.length,requests);
  ui.visibility(true);ui.visibility(false);await ui.nextTimer();
  assert.equal(ui.requests.filter(request=>request.path==='/api/exercise-coverage').length,1);
}

async function failedPoseRunRemainsInspectableAndSeeksAfterMetadataLoads() {
  const ui=fixture('exercise',{poseCoverage:poseCoverageFixture,generationFailed:true});
  await drain();ui.setReady();await ui.nextTimer();
  assert.match(ui.ids.get('build-status').textContent,/stopped/);
  assert.match(ui.ids.get('start-help').textContent,/Generation stopped/);
  assert.match(ui.ids.get('tracks-empty').textContent,/No tracking preview was saved/);
  assert.equal(ui.ids.get('toggle').disabled,true);
  assert.equal(ui.ids.get('pose-next-issue').disabled,false);
  ui.ids.get('pose-next-issue').click();
  assert.equal(ui.ids.get('source').currentTime,undefined);
  ui.ids.get('source').readyState=1;ui.ids.get('source').dispatch('loadedmetadata');
  assert.equal(ui.ids.get('source').currentTime,.103);
  assert.equal(ui.timers.size,0);
}

async function coverageBackgroundLoadDoesNotRestartOrPollAfterCompletion() {
  const ui=fixture('exercise',{poseCoverage:poseCoverageFixture,coverageLoading:1});
  await ui.ready();
  assert.match(ui.ids.get('pose-coverage-text').textContent,/Reading the saved/);
  assert.equal(ui.ids.get('toggle').disabled,false);
  await ui.nextTimer();
  assert.match(ui.ids.get('pose-coverage-text').textContent,/3\/6 analyzed/);
  assert.equal(ui.requests.filter(request=>request.path==='/api/exercise-coverage').length,2);
  assert.equal(ui.timers.size,0);
  const missing=fixture('exercise');await missing.ready();
  assert.match(missing.ids.get('pose-coverage-text').textContent,/No bounded pose-coverage/);
  assert.equal(missing.ids.get('pose-timeline').hidden,true);
}

async function spatialRigidExposesEightControlsThreeViewsAndHonestUntrainedState() {
  const ui = fixture('spatial-rigid'); await ui.ready();
  assert.match(ui.ids.get('readout').textContent, /scene_width/);
  assert.doesNotMatch(ui.ids.get('readout').textContent, / m\/s/);
  assert.doesNotMatch(ui.ids.get('spatial-readout').textContent, /\dN\b/);
  assert.equal(ui.ids.get('world-tab').textContent, '3D scene');
  assert.equal(ui.ids.get('spatial-controls').hidden, false);
  assert.equal(ui.ids.get('spatial-world-controls').hidden, false);
  assert.equal(ui.ids.get('human-controls').hidden, true);
  assert.equal(ui.ids.get('mobile-controls').hidden, true);
  assert.equal(ui.ids.get('reference-feasibility').hidden, true);
  assert.equal(ui.ids.get('reference-servo').hidden, true);
  assert.equal(ui.ids.get('policy').disabled, true);
  assert.equal(ui.ids.get('start-learned').disabled, true);
  assert.match(ui.ids.get('tracks-empty').textContent, /No source video or tracking preview was sealed/);
  assert.match(ui.ids.get('training-caption').textContent, /Not trained/);
  assert(!ui.requests.some(request => /\/(policy|training|rl-evaluation)\.json/.test(request.path)));
  const visibleActions = ui.document.querySelectorAll('[data-action]').filter(button => !button.hidden);
  assert.deepEqual(visibleActions.map(button => Number(button.dataset.action)).sort((a, b) => a - b), [0, 1, 2, 3, 4, 5, 6, 7]);
  assert(visibleActions.every(button => !button.disabled));
  const hiddenMobile = ui.document.querySelectorAll('[data-action]').filter(button => button.hidden);
  assert.equal(hiddenMobile.length, 4, 'Mobile-body support actions 6..9 stay hidden in 3D rigid scenes');
  assert(hiddenMobile.every(button => button.disabled));
  const viewLabels = ui.ids.get('live').drawCalls.filter(call => call.name === 'fillText').map(call => call.args[0]);
  assert.deepEqual(viewLabels, ['Front: x / height', 'Top: x / depth', '3D: same state']);
  const beforeView = ui.backendState(), requestsBeforeView = ui.requests.length;
  ui.ids.get('spatial-yaw').value = '-1.1'; ui.ids.get('spatial-yaw').dispatch('input');
  ui.ids.get('spatial-zoom').value = '2.2'; ui.ids.get('spatial-zoom').dispatch('input');
  ui.ids.get('spatial-follow').checked = true; ui.ids.get('spatial-follow').dispatch('change');
  assert.equal(ui.ids.get('spatial-world-yaw').value, '-1.1');
  assert.equal(ui.ids.get('spatial-world-zoom').value, '2.2');
  assert.equal(ui.ids.get('spatial-world-follow').checked, true);
  assert.deepEqual(ui.backendState(), beforeView, '3D camera yaw/zoom/follow must not advance simulation');
  assert.equal(ui.requests.length, requestsBeforeView);
  for (const code of [0, 1, 2, 3, 4, 5, 6, 7]) {
    ui.action(code); await drain();
    assert.deepEqual(ui.pending[0].data, {action: code, steps: 3, policy: false});
    ui.resolveNext('/api/step'); await drain();
  }
  assert.match(ui.ids.get('readout').textContent, /quat \[1\.000, 0\.000, 0\.000, 0\.000\]/);
  assert.match(ui.ids.get('readout').textContent, /contacts 1 · substeps 4/);
  assert.equal(ui.ids.get('spatial-readout').hidden, false);
  assert.match(ui.ids.get('spatial-readout').textContent, /companion \(passive, sphere\)/);
  assert.match(ui.ids.get('spatial-readout').textContent, /contact block↔floor.*normal force 9\.81 relative_mass\*scene_width\/second\^2/);
  for (const [key, expected] of [['q', 5], ['e', 6], [' ', 7], ['w', 3], ['s', 4], ['a', 1], ['d', 2]]) {
    ui.ids.get('live').dispatch('keydown', {key}); await drain();
    assert.equal(ui.pending[0].data.action, expected);
    ui.resolveNext('/api/step'); await drain();
    ui.ids.get('live').dispatch('keyup', {key});
  }
  ui.ids.get('record').click(); await drain();
  ui.resolveNext('/api/record/start'); await drain();
  ui.resolveNext('/api/step'); await drain();
  ui.ids.get('stop-record').click(); await drain();
  ui.resolveNext('/api/record/stop'); await drain();
  assert.equal(ui.ids.get('stop-record').disabled, true);
}

async function spatialRigidLearnedModeUnlocksOnlyWhenPolicyIsSealed() {
  const ui = fixture('spatial-rigid', {spatialTrained: true}); await ui.ready();
  assert.equal(ui.ids.get('policy').disabled, false);
  assert.equal(ui.ids.get('start-learned').disabled, false);
  assert.match(ui.ids.get('scores').children[0].children[1].textContent, /Learned controller/);
  assert.match(ui.ids.get('scores').children[3].children[1].textContent, /Braking baseline/);
  assert.match(ui.ids.get('world-description').textContent, /relative scene-width/);
  assert.doesNotMatch(ui.ids.get('world-description').textContent, /meter-scale/);
  assert.match(ui.ids.get('scores').children[0].children[1].textContent, /0\.010 scene-width units/);
  ui.ids.get('policy').click(); ui.ids.get('step').click(); await drain();
  assert.deepEqual(ui.pending[0].data, {action: 0, steps: 1, policy: true});
  ui.resolveNext('/api/step'); await drain();
}

async function manualObjectVideosDoNotClaimTrainingOrFetchMissingPolicies() {
  for (const profile of ['planar', 'appearance']) {
    const ui = fixture(profile, {trainingDisabled: true}); await ui.ready();
    assert.equal(ui.ids.get('policy').disabled, true);
    assert.equal(ui.ids.get('start-learned').disabled, true);
    assert.match(ui.ids.get('start-help').textContent, /2D scene ready/);
    assert.match(ui.ids.get('learning-description').textContent, /No learned controller or policy evaluation/);
    assert.match(ui.ids.get('training-caption').textContent, /Not trained/);
    assert.equal(ui.ids.get('learning-curve').hidden, true);
    assert.equal(ui.ids.get('scores').children.length, 0);
    assert(!ui.requests.some(request => /\/(policy|training|rl-evaluation)\.json/.test(request.path)));
    ui.action(2); await drain();
    assert.deepEqual(ui.pending[0].data, {action: 2, steps: 3, policy: false});
    ui.resolveNext('/api/step'); await drain();
    ui.ids.get('reset').click(); await drain();
    ui.resolveNext('/api/reset'); await drain();
    ui.ids.get('record').click(); await drain();
    ui.resolveNext('/api/record/start'); await drain();
    assert.equal(ui.pending[0].data.policy, false);
    ui.resolveNext('/api/step'); await drain();
    ui.ids.get('stop-record').click(); await drain();
    ui.resolveNext('/api/record/stop'); await drain();
    assert.equal(ui.ids.get('stop-record').disabled, true);
    assert.equal(ui.maxActiveSteps(), 1);
  }
}

async function multiObjectActorSelectionLocksDuringRecordingAndRestoresTrainedActor() {
  const single = fixture('planar'); await single.ready();
  assert.equal(single.ids.get('actor-controls').hidden, true, 'Single-object planar run hides actor selector');
  assert.equal(single.ids.get('spatial-readout').hidden, true);

  const ui = fixture('planar', {multiObject: true}); await ui.ready();
  assert.equal(ui.ids.get('actor-controls').hidden, false);
  assert.deepEqual(ui.ids.get('actor').children.map(option => option.value), ['actor', 'companion']);
  assert.equal(ui.ids.get('actor').value, 'actor');
  assert.equal(ui.ids.get('spatial-readout').hidden, false, 'Multi-object 2D planar scene shows per-object and contact readout');
  assert.match(ui.ids.get('spatial-readout').textContent, /contacts: actor↔companion/);
  assert.match(ui.ids.get('capabilities-json').textContent, /"active_control_target": "actor"/);

  ui.ids.get('actor').value = 'companion';
  ui.ids.get('actor').dispatch('change'); await drain();
  assert.deepEqual(ui.pending[0].data, {actor: 'companion'});
  ui.resolveNext('/api/actor'); await drain();
  assert.equal(ui.ids.get('actor').value, 'companion');
  assert.match(ui.ids.get('goal-summary').textContent, /Move Companion puck into the dashed target/);
  assert.match(ui.ids.get('capabilities-json').textContent, /"active_control_target": "companion"/);

  ui.ids.get('actor').value = 'divider';
  ui.ids.get('actor').dispatch('change'); await drain();
  ui.resolveNext('/api/actor'); await drain();
  assert.equal(ui.ids.get('actor').value, 'companion', 'Rejected static object restores active selector value');
  assert.equal(ui.ids.get('error').hidden, false);

  ui.ids.get('record').click(); await drain();
  ui.resolveNext('/api/record/start'); await drain();
  ui.resolveNext('/api/step'); await drain();
  assert.equal(ui.ids.get('actor').disabled, true, 'Actor selector locks while recording');
  ui.ids.get('stop-record').click(); await drain();
  ui.resolveNext('/api/record/stop'); await drain();

  ui.ids.get('policy').click();
  ui.ids.get('step').click(); await drain();
  assert.equal(ui.pending[0].path, '/api/actor', 'Learned mode restores trained actor before stepping');
  assert.deepEqual(ui.pending[0].data, {actor: 'actor'});
  ui.resolveNext('/api/actor'); await drain();
  ui.resolveNext('/api/reset'); await drain();
  ui.resolveNext('/api/step'); await drain();
  assert.equal(ui.ids.get('actor').value, 'actor');
}

async function spatialRigidActorSelectionSwitchesDynamicObjectsOnly() {
  const ui = fixture('spatial-rigid', {spatialTrained: true}); await ui.ready();
  assert.equal(ui.ids.get('actor-controls').hidden, false);
  assert.deepEqual(ui.ids.get('actor').children.map(option => option.value), ['block', 'companion']);
  ui.ids.get('actor').value = 'companion';
  ui.ids.get('actor').dispatch('change'); await drain();
  ui.resolveNext('/api/actor'); await drain();
  assert.match(ui.ids.get('goal-summary').textContent, /Move Companion sphere into the 3D dashed target/);
  ui.ids.get('start-learned').click(); await drain();
  assert.equal(ui.pending[0].path, '/api/actor');
  assert.deepEqual(ui.pending[0].data, {actor: 'block'});
  ui.resolveNext('/api/actor'); await drain();
  ui.resolveNext('/api/reset'); await drain();
  ui.resolveNext('/api/step'); await drain();
  assert.equal(ui.ids.get('actor').value, 'block');
}

async function evidenceProvenanceSeparatesSourceKinematicsControllersAndSimulation() {
  const ex = fixture('exercise'); await ex.ready();
  assert.equal(ex.ids.get('evidence-layers').hidden, false);
  const exText = ex.ids.get('evidence-layers-text').textContent;
  assert.match(exText, /Sealed source video and per-frame measurements preserved in run\/ \(90 frames\)/);
  assert.match(exText, /3 unusable sampled frames/);
  assert.match(exText, /Kinematic projection: Source-derived joint poses/);
  assert.match(exText, /Controllers: manual, authored_reference_controller, learned_policy/);
  assert.match(exText, /illustrative motor command not measured activation/);

  const sp = fixture('spatial-rigid'); await sp.ready();
  assert.equal(sp.ids.get('evidence-layers').hidden, false);
  const spText = sp.ids.get('evidence-layers-text').textContent;
  assert.match(spText, /Analytic or imported scene without sealed source video/);
  assert.match(spText, /Kinematic projection: Not applicable \(rigid-object simulation\)/);
  assert.match(spText, /Controllers: manual\./);
}

async function sharedWorldShowsBodyObjectsAndContactsInAllThreeViews() {
  const ui=fixture('shared-world'); await ui.ready();
  assert.equal(ui.ids.get('world-tab').textContent,'Coupled 3D world');
  assert.equal(ui.ids.get('human-controls').hidden,false);
  assert.equal(ui.ids.get('spatial-controls').hidden,false);
  assert.match(ui.ids.get('world-description').textContent,/one MuJoCo model, coordinate frame, clock, and contact solver/);
  assert.match(ui.ids.get('muscle-readout').textContent,/38 illustrative muscle bands.*not measured activation/);
  const calls=ui.ids.get('live').drawCalls;
  assert.deepEqual(calls.filter(c=>c.name==='fillText').map(c=>c.args[0]),['Front: x / height','Top: x / depth','3D: coupled world']);
  assert.equal(calls.filter(c=>c.name==='clip').length,3);
  assert.equal(calls.filter(c=>c.name==='stroke'&&c.strokeStyle==='#C6A58A').length,3,'Human segment must appear in each view');
  assert.equal(calls.filter(c=>c.name==='fill'&&c.fillStyle==='#965644').length,21,'Six box faces and one ordinary contact appear in each view');
  assert.equal(calls.filter(c=>c.name==='arc'&&c.args[2]===4.5).length,3,'Human-object contacts are visually distinct in each view');
  for(const c of calls.filter(c=>['moveTo','lineTo','arc'].includes(c.name))) assert(c.args.every(Number.isFinite));
  assert.match(ui.ids.get('spatial-readout').textContent,/contact \[human_object\] human:left_hand↔block/);
  assert.doesNotMatch(ui.ids.get('spatial-readout').textContent,/\dN\b/);
  assert.doesNotMatch(ui.ids.get('whole-body-status').textContent,/\] N\b/);
  assert(!JSON.stringify(ui.ids.get('objects').children.map(row=>row.children.map(cell=>cell.textContent))).includes('Measured geometry'));
  const before=ui.backendState(), requests=ui.requests.length;
  ui.ids.get('spatial-yaw').value='-1.1';ui.ids.get('spatial-yaw').dispatch('input');
  ui.ids.get('spatial-zoom').value='2';ui.ids.get('spatial-zoom').dispatch('input');
  ui.ids.get('show-muscles').checked=false;ui.ids.get('show-muscles').dispatch('input');
  assert(!ui.ids.get('live').drawCalls.some(c=>c.name==='stroke'&&c.strokeStyle?.startsWith('rgb(')));
  assert.deepEqual(ui.backendState(),before);assert.equal(ui.requests.length,requests);
}

async function sharedWorldRoutesHumanMotorVectorsAndSelectedObjectActions() {
  const ui=fixture('shared-world');await ui.ready();
  assert.deepEqual(ui.ids.get('actor').children.map(option=>option.value),['human','block','companion']);
  assert.equal(ui.ids.get('joint').children.length,22);
  const visible=()=>ui.document.querySelectorAll('[data-action]').filter(button=>!button.hidden).map(button=>Number(button.dataset.action));
  assert.deepEqual(visible().sort((a,b)=>a-b),[0,1,2]);
  ui.ids.get('joint').value='root_y';ui.ids.get('joint').dispatch('change');await drain();
  assert(!ui.requests.some(request=>request.path==='/api/joint'));
  ui.action(1);await drain();
  let data=ui.pending[0].data;
  assert.equal(data.action.length,28);assert.equal(data.action[20],.25);
  assert.equal(data.action.filter(Boolean).length,1);assert(data.action.slice(22).every(value=>value===0));
  ui.resolveNext('/api/step');await drain();
  ui.action(2);await drain();assert.equal(ui.pending[0].data.action[20],-.25);
  ui.resolveNext('/api/step');await drain();
  ui.ids.get('actor').value='companion';ui.ids.get('actor').dispatch('change');await drain();
  ui.resolveNext('/api/actor');await drain();
  assert.equal(ui.ids.get('joint').disabled,true);
  assert.deepEqual(visible().sort((a,b)=>a-b),[0,1,2,3,4,5,6,7]);
  for(const action of [0,1,2,3,4,5,6,7]) {
    ui.action(action);await drain();assert.deepEqual(ui.pending[0].data,{steps:3,action});
    ui.resolveNext('/api/step');await drain();
  }
  ui.ids.get('actor').value='human';ui.ids.get('actor').dispatch('change');await drain();
  ui.resolveNext('/api/actor');await drain();
  ui.ids.get('step').click();await drain();
  assert(ui.pending[0].data.action.every(value=>value===0),'Changing actor must not leak a held object action into a human motor');
  ui.resolveNext('/api/step');await drain();
  assert.equal(ui.maxActiveSteps(),1);
}

async function sharedWorldPolicyAvailabilityAndControllerPayloadsStayExplicit() {
  const untrained=fixture('shared-world');await untrained.ready();
  assert.equal(untrained.ids.get('policy').disabled,true);assert.equal(untrained.ids.get('start-learned').disabled,true);
  assert.match(untrained.ids.get('learning-description').textContent,/No learned coupled policy/);
  assert(!untrained.requests.some(request=>/\/(policy|training|rl-evaluation)\.json/.test(request.path)));
  untrained.ids.get('policy').click();untrained.ids.get('step').click();await drain();
  assert(Array.isArray(untrained.pending[0].data.action));assert(!untrained.pending[0].data.policy);
  untrained.resolveNext('/api/step');await drain();
  const ui=fixture('shared-world',{sharedTrained:true});await ui.ready();
  assert.equal(ui.ids.get('policy').disabled,false);assert.equal(ui.ids.get('start-learned').disabled,false);
  assert.match(ui.ids.get('scores').children[0].children[1].textContent,/Learned coupled controller/);
  assert.match(ui.ids.get('scores').children[3].children[1].textContent,/Authored human feedback/);
  ui.ids.get('policy').click();ui.ids.get('step').click();await drain();
  assert.deepEqual(ui.pending[0].data,{steps:1,policy:true});ui.resolveNext('/api/step');await drain();
  ui.ids.get('reference-servo').click();ui.ids.get('step').click();await drain();
  assert.deepEqual(ui.pending[0].data,{steps:1,controller:'reference'});ui.resolveNext('/api/step');await drain();
  assert.match(ui.ids.get('mode-description').textContent,/Authored human feedback/);
}

async function sharedWorldRecordingLocksActorAndSerializesReset() {
  const ui=fixture('shared-world');await ui.ready();
  ui.ids.get('record').click();await drain();ui.resolveNext('/api/record/start');await drain();
  assert.equal(ui.ids.get('actor').disabled,true);
  assert.equal(ui.ids.get('joint').disabled,false,'Selected motor can change without changing the recorded actor binding');
  ui.ids.get('reset').click();await drain();
  assert.equal(ui.pending.filter(request=>request.path==='/api/reset').length,0);
  ui.resolveNext('/api/step');await drain();ui.resolveNext('/api/reset');await drain();
  assert.equal(ui.backendState().steps,0);assert.match(ui.ids.get('record-status').textContent,/Recording/);
  assert.equal(ui.ids.get('actor').disabled,true);
  ui.ids.get('stop-record').click();await drain();ui.resolveNext('/api/record/stop');await drain();
  assert.equal(ui.ids.get('actor').disabled,false);assert.equal(ui.ids.get('stop-record').disabled,true);
  assert.equal(ui.maxActiveSteps(),1);
}

async function sharedWorldKinematicScrubbingCannotAdvanceLivePhysics() {
  const ui=fixture('shared-world',{feasibilityStatus:'feasible'});await ui.ready();
  assert.match(ui.ids.get('feasibility-notice').textContent,/diagnostic simulation, not validated physical reconstruction/);
  assert.match(ui.ids.get('evidence-layers-text').textContent,/Kinematic projection: Source-derived joint poses/);
  assert.match(ui.ids.get('evidence-layers-text').textContent,/mujoco_3d_coupled forward dynamics/);
  const state=ui.backendState(),requests=ui.requests.length;
  ui.stage('world');ui.ids.get('reference-frame').value='2';ui.ids.get('reference-frame').dispatch('input');
  assert.match(ui.ids.get('reference-time').textContent,/3\.00 seconds · estimated pose/);
  assert.deepEqual(ui.backendState(),state);assert.equal(ui.requests.length,requests);
  assert(!ui.ids.get('world-canvas').drawCalls.some(call=>call.name==='arc'&&call.args[2]===4.5),'Kinematic projection must not show live collision forces');
  assert.equal(ui.ids.get('live').drawCalls.filter(call=>call.name==='arc'&&call.args[2]===4.5).length,3);
  ui.ids.get('reference-play').click();assert(ui.timers.size>0);
  ui.visibility(true);assert.equal(ui.timers.size,0);
}

function inspectionFixture() {
  const panel = name => ({name:`geometry-inspection/${name}`,label:'Initial geometry comparison',bytes:400000,sha256:'b'.repeat(64)});
  return {status:'available',profile:'lab-inspection-v1',source:{sha256:'a'.repeat(64),decoded_frames:10,video:'source.mp4',scope:'Saved count, not physical calibration.'},
    stages:[['original','available'],['tracking','available'],['geometry','initial_hypothesis'],['sandbox','simulation'],['recordings','local_recording']].map(([id,state])=>({id,state})),
    actors:[{id:'human',label:'Articulated human',geometry:'simplified body',label_origin:'generic label',scope:'Estimated body; not recovered anatomy.',source_binding:{kind:'observed_track',counts:{observed_slots:6,missing_slots:4,pre_init_slots:0}}},
      {id:'block',label:'Tracked object 1',geometry:'box',label_origin:'generic display label',scope:'Source-linked model estimate; shape and physics are assumptions.',source_binding:{kind:'observed_track',counts:{observed_slots:8,missing_slots:2,pre_init_slots:0}}}],
    tracking:{reference_poses:4,pose_status_counts:{tracked:4,tracked_partial:2,no_pose:3,not_sampled:1},scope:'Construction slots are not true visibility or accuracy.'},
    geometry:{status:'initial_hypothesis',scope:'Initial authored geometry hypotheses, not recovered shape or grasp.',panels:[panel('first.png'),panel('later.png')],assumptions:['Depth is authored.']},
    artifact_links:[{name:'evidence/observations.json',label:'Source observation document',bytes:3000000,sha256:'c'.repeat(64)},panel('first.png'),panel('later.png')]};
}

function windowInspection(full = false) {
  const data = inspectionFixture();
  data.timeline = {status:'available',reference_span_s:[.2,8.666666666666666],reference_poses:4,
    task_window_s:full?[.2,8.666666666666666]:[.3,.6],task_duration_s:full?254/30:.3,task_steps:full?254:9,
    source_slots:{with_reference:4,without_reference:6,leading_without_reference:2,trailing_without_reference:0},
    alignment_method:'pose_frame_index_over_fps_prior_v1',native_source_clock:'not_read_here',media_sync_verified:false};
  return data;
}

async function sourceAndSimulationWindowsKeepPlaybackIndependent() {
  const ui=fixture('shared-world',{sharedExercise:true,inspection:windowInspection(),exerciseInterval:{start_offset_s:.1,horizon_steps:9}});await ui.ready();
  assert.equal(ui.ids.get('source-simulation-window').hidden,false);
  assert.match(ui.ids.get('window-task').textContent,/0\.300–0\.600s.*0\.300s, 9 control steps.*Only part/);
  assert.match(ui.ids.get('window-coverage').textContent,/4 with.*6 without.*2 leading.*not detection or visibility/);
  assert.match(ui.ids.get('window-clocks').textContent,/frame-index \/ FPS prior.*Native source timestamps.*not synchronized/);
  assert.match(ui.ids.get('window-media').textContent,/unavailable until media metadata/);
  const before=ui.backendTaskState(),requests=ui.requests.length;
  const video=ui.ids.get('source');video.currentTime=7.25;video.duration=8.7;video.dispatch('loadedmetadata');video.dispatch('timeupdate');
  assert.match(ui.ids.get('window-media').textContent,/7\.250s \/ 8\.700s media duration.*independent/);
  assert.match(ui.ids.get('window-live').textContent,/0\.000s elapsed.*reference-clock position unavailable/);
  assert.equal(ui.requests.length,requests);assert.deepEqual(ui.backendTaskState(),before);
  video.dispatch('error');assert.match(ui.ids.get('window-media').textContent,/unavailable/);
}

async function sourceWindowPreservesUnknownAndFailedGeometryClocks() {
  const missing=fixture('shared-world',{sharedExercise:true,inspection:inspectionFixture(),missingExerciseState:true});await missing.ready();
  assert.match(missing.ids.get('window-reference').textContent,/span unavailable/);
  assert.match(missing.ids.get('window-task').textContent,/window unavailable/);
  assert.match(missing.ids.get('window-live').textContent,/position unavailable.*No zero time/);
  const failed=fixture('shared-world',{sharedExercise:true,inspection:windowInspection(),exerciseEngineStatus:'last_valid_control_boundary',
    exerciseGeometryTime:.125,exerciseInitialMetrics:{failure:'Saved failure',simulated_duration_s:.15}});await failed.ready();
  assert.match(failed.ids.get('window-live').textContent,/Last valid simulation geometry: 0\.125s.*not a new terminal pose/);
  assert.match(failed.ids.get('exercise-duration').textContent,/0\.150/);
}

async function sharedWorldV2FullReferenceWindowDoesNotClipOrClaimFullVideo() {
  const inspection = windowInspection(true);
  inspection.fullspan_preparation = {status:'preparation_only',artifact:'fullspan.json'};
  inspection.artifact_links.push({name:'fullspan.json',label:'Full-reference preparation and preserved limitations',bytes:2000,sha256:'d'.repeat(64)});
  inspection.geometry = {status:'no_saved_comparison',panels:[],assumptions:[]};
  const ui=fixture('shared-world',{sharedProfile:'shared-world-v2',sharedExercise:true,rootAssistance:true,
    exerciseInterval:{start_offset_s:0,horizon_steps:254},inspection});await ui.ready();
  assert.equal(ui.ids.get('joint').children.length,22);assert.equal(ui.ids.get('actor-controls').hidden,true);
  assert.match(ui.ids.get('evidence-boundary').textContent,/shared-world-v2/);
  assert.match(ui.ids.get('window-task').textContent,/254 control steps.*does not establish complete source-video coverage/);
  assert.doesNotMatch(ui.ids.get('window-task').textContent,/180|Only part/);
  assert.match(ui.ids.get('exercise-task-scope').textContent,/254 control steps/);
  assert.match(ui.ids.get('window-task').textContent,/changes the task interval, not the recovered geometry.*fullspan.json in Files/);
  assert.equal(ui.ids.get('geometry-preview').src,undefined);
  assert.match(ui.ids.get('inspection-geometry').textContent,/No saved source-to-geometry comparison/);
  ui.ids.get('mode-coast').click();ui.ids.get('step').click();await drain();
  assert.deepEqual(ui.pending[0].data,{steps:1,controller:'coast'});ui.resolveNext('/api/step');await drain();
  assert.match(ui.ids.get('window-live').textContent,/Displayed simulation geometry: 0\.033s/);
  assert(ui.ids.get('live').drawCalls.length>0);
}

async function fiveStagesExposeCountsAndLazyGeometryWithoutChangingPhysics() {
  const ui=fixture('shared-world',{inspection:inspectionFixture()});await ui.ready();
  assert.match(ui.ids.get('inspection-source').textContent,/10 decoded source frames/);
  assert.match(ui.ids.get('inspection-tracking').textContent,/6 usable, including 2 partial/);
  assert.match(ui.ids.get('inspection-tracking').textContent,/4 body poses retained/);
  assert.match(ui.ids.get('inspection-actors').children[0].children[2].textContent,/6 construction-observation slots/);
  assert.match(ui.ids.get('inspection-actors').children[1].children[2].textContent,/8 construction-observation slots/);
  assert.equal(ui.ids.get('geometry-preview').src,undefined,'Do not fetch geometry images until their stage is opened');
  assert.match(ui.ids.get('inspection-geometry').textContent,/not recovered shape/);
  const state=ui.backendState(),requests=ui.requests.length;
  for(const [id,panel,target] of [['original','evidence','source'],['tracking','evidence','tracking-summary'],
    ['geometry','world','geometry-summary'],['sandbox','play','live'],['recordings','play','record-box']]) {
    assert.equal(ui.ids.get(`inspect-${id}`).disabled,false);ui.ids.get(`inspect-${id}`).click();
    assert.equal(ui.ids.get(panel).hidden,false);assert.equal(ui.ids.get(target).scrolled,true);
  }
  assert.match(ui.ids.get('geometry-preview').src,/geometry-inspection\/first\.png\?token=/);
  ui.ids.get('inspect-geometry').click();ui.ids.get('geometry-panel').value='1';ui.ids.get('geometry-panel').dispatch('change');
  assert.match(ui.ids.get('geometry-preview').src,/later\.png/);
  assert.match(ui.ids.get('geometry-caption').textContent,/not synchronized to video playback/);
  assert.deepEqual(ui.backendState(),state);assert.equal(ui.requests.length,requests);
  assert(!ui.requests.some(request=>request.path.includes('/evidence/')),'Raw observation arrays must stay download-only');
}

async function inspectionNamesKeepControlIdsAndUnavailablePolicyDistinct() {
  const ui=fixture('shared-world',{inspection:inspectionFixture()});await ui.ready();
  assert.deepEqual(ui.ids.get('actor').children.map(option=>option.value),['human','block','companion']);
  assert.equal(ui.ids.get('actor').children[1].textContent,'Tracked object 1 (block)');
  assert.match(ui.ids.get('goal-summary').textContent,/stop Tracked object 1/);
  assert.match(ui.ids.get('goal-summary').textContent,/terminal task/);
  assert.match(ui.ids.get('goal-summary').textContent,/not exercise or grasp success/);
  assert.equal(ui.ids.get('policy').disabled,true);assert.equal(ui.ids.get('start-learned').disabled,true);
  assert.equal(ui.ids.get('measurements-inspect').disabled,true,'An absent evidence.json must not produce a dead inspector link');
  const row=ui.ids.get('files').children.find(row=>row.children[0].children[0].textContent==='evidence/observations.json');
  assert(row);assert.match(row.children[2].textContent,/Large document/);assert.equal(row.children[2].children.length,0);
  ui.ids.get('actor').value='block';ui.ids.get('actor').dispatch('change');await drain();
  assert.deepEqual(ui.pending[0].data,{actor:'block'});ui.resolveNext('/api/actor');await drain();
  assert.match(ui.ids.get('mode-description').textContent,/Tracked object 1 selected/);
}

async function inspectionPollingIsBoundedCachedAndIndependentOfRecording() {
  const ui=fixture('shared-world',{inspection:inspectionFixture(),inspectionLoading:1});await ui.ready();
  assert.match(ui.ids.get('inspection-status').textContent,/Reading a compact summary/);
  assert.equal(ui.ids.get('geometry-preview').src,undefined);
  await ui.nextTimer();assert.match(ui.ids.get('inspection-status').textContent,/prepared simulation candidate/);
  assert.equal(ui.timers.size,0);
  assert.equal(ui.requests.filter(request=>request.path==='/api/inspection').length,2);
  const count=ui.requests.length;ui.visibility(true);ui.visibility(false);await ui.nextTimer();
  assert.equal(ui.requests.length,count+1,'A refresh reads status, not already cached inspection metadata');
  ui.ids.get('inspect-recordings').click();ui.ids.get('record').click();await drain();
  assert.equal(ui.pending[0].path,'/api/record/start');ui.resolveNext('/api/record/start');await drain();
  assert.equal(ui.ids.get('actor').disabled,true);
  ui.ids.get('inspect-original').click();await drain();ui.resolveNext('/api/step');await drain();
  assert.equal(ui.ids.get('stop-record').disabled,false,'Navigation pauses movement but must not silently stop or discard the journal');
  assert(!ui.pending.some(request=>request.path==='/api/record/stop'));
}

async function inspectionTimeoutOffersRetryWithoutAnUnboundedPollLoop() {
  const ui=fixture('shared-world',{inspection:inspectionFixture(),inspectionLoading:10});await ui.ready();
  for(let i=1;i<10;i++)await ui.nextTimer();
  assert.equal(ui.timers.size,0);assert.equal(ui.ids.get('inspection-retry').hidden,false);
  assert.equal(ui.requests.filter(request=>request.path==='/api/inspection').length,10);
  ui.ids.get('inspection-retry').click();await ui.nextTimer();
  assert.equal(ui.ids.get('inspection-retry').hidden,true);assert.equal(ui.timers.size,0);
  assert.match(ui.ids.get('inspection-status').textContent,/prepared simulation candidate/);
}

async function sharedExerciseShowsImmutableTaskPassiveEquipmentAndExplicitAssistance() {
  for(const rootAssistance of [false,true]) {
    const ui=fixture('shared-world',{sharedExercise:true,rootAssistance,sharedTrained:true});await ui.ready();
    assert.equal(ui.ids.get('exercise-task-card').hidden,false);assert.equal(ui.ids.get('exercise-results').hidden,false);
    assert.match(ui.ids.get('exercise-task-scope').textContent,/0\.20 seconds \(6 control steps\)/);
    assert.match(ui.ids.get('exercise-assistance').textContent,rootAssistance?/3 explicit virtual controls/:/OFF · 19 joint motors/);
    assert.match(ui.ids.get('exercise-contact-scope').textContent,/No wrist-region contact is required/);
    assert.equal(ui.ids.get('joint').children.length,rootAssistance?22:19);
    assert.equal(ui.ids.get('actor-controls').hidden,true);assert.equal(ui.ids.get('actor').disabled,true);
    assert.deepEqual(ui.ids.get('actor').children.map(option=>option.value),['human']);
    assert.equal(ui.ids.get('mode-coast').hidden,false);assert.equal(ui.ids.get('policy').disabled,true);assert.equal(ui.ids.get('start-learned').disabled,true);
    assert(!ui.requests.some(request=>/\/(policy|training|rl-evaluation)\.json/.test(request.path)),'Legacy learned artifacts must not activate the human-only task');
    assert(!ui.ids.get('live').drawCalls.some(call=>call.name==='setLineDash'&&call.args[0]?.length),'Exercise view has no object-target marker');
    assert.match(ui.ids.get('goal-summary').textContent,/not a target-object task/);
    assert.doesNotMatch(ui.ids.get('readout').textContent,/object.*dist|target:|TASK SUCCESS/);
    assert.match(ui.ids.get('exercise-outcome').textContent,/In progress/);
    assert.notEqual(ui.ids.get('play-state').textContent,'Goal reached','Legacy engine success must not decide exercise success');
    const before=ui.requests.length;ui.ids.get('actor').value='companion';ui.ids.get('actor').dispatch('change');await drain();
    assert.equal(ui.requests.length,before,'Hidden object selector cannot submit an exercise actor switch');
  }
}

async function sharedExerciseManualCoastAndAuthoredRequestsStayDistinct() {
  for(const rootAssistance of [false,true]) {
    const ui=fixture('shared-world',{sharedExercise:true,rootAssistance});await ui.ready();
    const motor=rootAssistance?'root_x':'left_elbow';
    ui.ids.get('joint').value=motor;ui.ids.get('joint').dispatch('change');ui.action(1);await drain();
    let data=ui.pending[0].data;
    assert.equal(data.action.length,rootAssistance?22:19);assert.equal(data.action.filter(Boolean).length,1);
    const index=ui.ids.get('joint').children.findIndex(option=>option.value===motor);
    assert.equal(data.action[index],.25);assert(!('controller' in data));ui.resolveNext('/api/step');await drain();
    ui.ids.get('mode-coast').click();ui.ids.get('step').click();await drain();
    assert.deepEqual(ui.pending[0].data,{steps:1,controller:'coast'});ui.resolveNext('/api/step');await drain();
    assert(ui.backendTaskState().last_action.every(value=>value===0));assert.match(ui.ids.get('mode-description').textContent,/every joint and root-force command is zero/);
    ui.ids.get('reference-servo').click();ui.ids.get('step').click();await drain();
    assert.deepEqual(ui.pending[0].data,{steps:1,controller:'reference'});ui.resolveNext('/api/step');await drain();
    assert.match(ui.ids.get('mode-description').textContent,/programmed control, not learned behavior/);
    const before=ui.requests.length;ui.action(1);ui.ids.get('live').dispatch('keydown',{key:'ArrowUp'});await drain();
    assert.equal(ui.requests.length,before,'Manual input cannot replace the authored controller');
    assert.equal(ui.maxActiveSteps(),1);
  }
}

async function sharedExerciseReportsWholeIntervalMetricsAndExactOutcomeStates() {
  const equipment=[{object_id:'block',displacement_model_units:.18,orientation_rad:.21,speed_model_units_s:.22,angular_speed_rad_s:.81,within_limits:false},
    {object_id:'companion',displacement_model_units:.02,orientation_rad:.03,speed_model_units_s:.04,angular_speed_rad_s:.05,within_limits:true}];
  const contacts=[{object_id:'block',wrist:'left',kind:'upward_support',start_step:0,end_step:6,qualifying_fraction:.5,
    observed_duration_s:.1,required_duration_s:.2,max_gap_s:.11,satisfied:false}];
  const ui=fixture('shared-world',{sharedExercise:true,exerciseInitialMetrics:{simulated_duration_s:.1,observed_substeps:24,valid_substeps:24,
    angular_rmse_rad:.2,peak_angular_rmse_rad:.5,root_rmse_model_units:.1,peak_root_rmse_model_units:.2,max_floor_penetration_model_units:.03,
    equipment,contact_requirements:contacts,failed_gates:['equipment_stability:block']}});await ui.ready();
  assert.match(ui.ids.get('exercise-body-errors').textContent,/mean 0\.200 rad \/ peak 0\.500 rad/);
  assert.match(ui.ids.get('exercise-equipment-summary').textContent,/0\.180 model units/);
  assert.match(ui.ids.get('exercise-duration').textContent,/24 valid \/ 24 observed substeps/);
  assert.equal(ui.ids.get('exercise-equipment').children.length,0,'Detailed tables stay lazy until opened');
  ui.ids.get('exercise-metrics-details').open=true;ui.ids.get('exercise-metrics-details').dispatch('toggle');
  assert.equal(ui.ids.get('exercise-equipment').children.length,2);
  assert.equal(ui.ids.get('exercise-equipment').children[0].children[1].textContent,'0.180 model units');
  assert.match(ui.ids.get('exercise-contacts').children[0].children[2].textContent,/50\.0% qualifying/);
  assert.match(ui.ids.get('exercise-gates').textContent,/equipment_stability:block/);
  assert.match(ui.ids.get('exercise-outcome').textContent,/In progress/);
  for(const success of [true,false]) {
    const finished=fixture('shared-world',{sharedExercise:true,exerciseInitialMetrics:{completed_interval:true,success,simulated_duration_s:.2,observed_substeps:48,valid_substeps:48}});
    await finished.ready();
    assert.match(finished.ids.get('exercise-outcome').textContent,success?/^Passed/:/^Not passed/);
    assert.equal(finished.ids.get('play-state').textContent,success?'Goal reached':'Trial finished');
    assert.equal(finished.ids.get('step').disabled,true);
  }
}

async function sharedExerciseMissingAndFailureStatesNeverInventTimeOrSuccess() {
  const missing=fixture('shared-world',{sharedExercise:true,missingExerciseState:true});await missing.ready();
  assert.match(missing.ids.get('exercise-outcome').textContent,/unavailable/);
  assert.match(missing.ids.get('exercise-duration').textContent,/no zero values or success/);
  for(const id of ['toggle','step','record','mode-manual','mode-coast','reference-servo'])assert.equal(missing.ids.get(id).disabled,true);
  assert.equal(missing.ids.get('reset').disabled,false);assert.equal(missing.ids.get('exercise-equipment').children.length,0);
  const before=missing.requests.length;missing.action(1);await drain();assert.equal(missing.requests.length,before);
  for(const stale of [false,true]) {
    const failed=fixture('shared-world',{sharedExercise:true,exerciseEngineStatus:stale?'last_valid_control_boundary':'current',exerciseGeometryTime:stale?.1:.12,
      exerciseInitialMetrics:{failure:'engine_failure:partial action',simulated_duration_s:.12,observed_substeps:29,valid_substeps:28}});
    await failed.ready();
    assert.match(failed.ids.get('exercise-outcome').textContent,/Not passed · simulation failed/);
    assert.match(failed.ids.get('exercise-duration').textContent,/28 valid \/ 29 observed/);
    assert.match(failed.ids.get('readout').textContent,stale?/task 0\.120s · geometry 0\.100s/:/task 0\.120s · geometry 0\.120s/);
    assert.match(failed.ids.get('exercise-failure').textContent,stale?/last valid geometry.*No terminal pose was fabricated/:/current geometry time/);
    assert.equal(failed.ids.get('step').disabled,true);assert.equal(failed.ids.get('stop-record').disabled,true);
    failed.ids.get('reset').click();await drain();failed.resolveNext('/api/reset');await drain();
    assert.equal(failed.ids.get('exercise-failure').hidden,true);assert.equal(failed.ids.get('error').hidden,true);
    assert.match(failed.ids.get('exercise-outcome').textContent,/In progress/);
  }
}

async function sharedExerciseRecordingAndInspectionPreserveTheTaskWrapper() {
  const ui=fixture('shared-world',{sharedExercise:true,rootAssistance:true});await ui.ready();
  const before=ui.backendTaskState(),requestCount=ui.requests.length;
  ui.stage('world');ui.ids.get('reference-frame').value='2';ui.ids.get('reference-frame').dispatch('input');
  ui.ids.get('spatial-world-yaw').value='1.2';ui.ids.get('spatial-world-yaw').dispatch('input');
  assert.deepEqual(ui.backendTaskState(),before);assert.equal(ui.requests.length,requestCount);
  assert(!ui.ids.get('world-canvas').drawCalls.some(call=>call.name==='setLineDash'&&call.args[0]?.length));
  ui.stage('play');ui.ids.get('mode-coast').click();ui.ids.get('record').click();await drain();
  assert.equal(ui.pending[0].path,'/api/record/start');ui.resolveNext('/api/record/start');await drain();
  assert.equal(ui.ids.get('actor').disabled,true);assert.equal(ui.ids.get('joint').disabled,true);
  assert.deepEqual(ui.pending[0].data,{steps:3,controller:'coast'});
  ui.ids.get('reset').click();await drain();assert(!ui.pending.some(request=>request.path==='/api/reset'));
  ui.resolveNext('/api/step');await drain();ui.resolveNext('/api/reset');await drain();
  assert.equal(ui.backendTaskState().profile,'shared-exercise-state-v1');assert.equal(ui.backendTaskState().task_metrics.observed_substeps,0);
  assert.equal(ui.backendTaskState().engine_state.time_s,0);assert.match(ui.ids.get('record-status').textContent,/Recording/);
  ui.ids.get('mode-manual').click();assert.equal(ui.ids.get('joint').disabled,false,'A different human motor does not change the immutable task');
  ui.ids.get('stop-record').click();await drain();ui.resolveNext('/api/record/stop');await drain();
  assert.equal(ui.ids.get('stop-record').disabled,true);assert.equal(ui.maxActiveSteps(),1);
}

async function sharedExerciseMissingContractAndContactFractionStayUnavailable() {
  const missing=fixture('shared-world',{sharedExercise:true,missingExerciseContract:true,
    exerciseInitialMetrics:{completed_interval:true,success:true,simulated_duration_s:.2,observed_substeps:48,valid_substeps:48}});
  await missing.ready();
  assert.match(missing.ids.get('exercise-outcome').textContent,/unavailable/);
  assert.doesNotMatch(missing.ids.get('exercise-outcome').textContent,/^Passed/);
  assert.equal(missing.ids.get('record').disabled,true);assert.equal(missing.ids.get('joint').children.length,0);
  const ui=fixture('shared-world',{sharedExercise:true,exerciseInitialMetrics:{contact_requirements:[{
    object_id:'block',wrist:'left',kind:'upward_support',start_step:0,end_step:6,qualifying_fraction:null,
    observed_duration_s:0,required_duration_s:.2,max_gap_s:0,satisfied:false}]}});await ui.ready();
  ui.ids.get('exercise-metrics-details').open=true;ui.ids.get('exercise-metrics-details').dispatch('toggle');
  const coverage=ui.ids.get('exercise-contacts').children[0].children[2].textContent;
  assert.match(coverage,/Qualifying fraction unavailable/);assert.doesNotMatch(coverage,/0\.0%/);
  assert.match(ui.ids.get('exercise-outcome').textContent,/In progress/,'The UI does not rescore missing display values');
}

function attachedExercisePolicy(kind = 'searched_gains', overrides = {}) {
  return {profile:'shared-exercise-policy-v1',policy_sha256:'c'.repeat(64),selected_feedback_kind:kind,
    privileged_simulator_features:true,training_score:.7,training_score_improved_over_authored:true,
    evaluation_status:'not_attached',evaluation_successes:null,scope:'Synthetic training metadata, not independent evaluation',...overrides};
}

async function sharedExerciseOptionalPolicyRoutesHumanOnlyAndPreservesRecording() {
  for(const rootAssistance of [false,true]) {
    const ui=fixture('shared-world',{sharedExercise:true,rootAssistance,exercisePolicyAvailable:true,exercisePolicy:attachedExercisePolicy()});await ui.ready();
    assert.equal(ui.ids.get('policy').disabled,false);assert.equal(ui.ids.get('start-learned').disabled,false);
    assert.equal(ui.ids.get('joint').children.length,rootAssistance?22:19);
    assert.equal(ui.ids.get('actor').disabled,true);assert.match(ui.ids.get('policy').textContent,/Searched human feedback/);
    assert.match(ui.ids.get('learning-description').textContent,/simulator-state and reference features, not video pixels/);
    assert.match(ui.ids.get('exercise-evaluation-summary').textContent,/Independent evaluation unavailable/);
    assert.equal(ui.ids.get('scores').children.length,0);assert.equal(ui.ids.get('learning-curve').hidden,true);
    assert.equal(ui.ids.get('exercise-policy-links').children.length,1);
    assert(ui.ids.get('exercise-policy-links').children[0].href.includes('/files/run/policy.json'));
    ui.ids.get('policy').click();ui.ids.get('step').click();await drain();
    assert.deepEqual(ui.pending[0].data,{steps:1,policy:true});ui.resolveNext('/api/step');await drain();
    assert.match(ui.ids.get('mode-description').textContent,/Equipment has no force controls/);
    assert.match(ui.ids.get('exercise-outcome').textContent,/In progress/);
    ui.ids.get('record').click();await drain();
    assert.equal(ui.pending[0].path,'/api/record/start');assert.deepEqual(ui.pending[0].data,{});
    assert.equal(ui.ids.get('policy').disabled,true,'Start/reset remains serialized');
    ui.resolveNext('/api/record/start');await drain();assert.deepEqual(ui.pending[0].data,{steps:3,policy:true});
    assert.equal(ui.ids.get('record').disabled,true);assert.equal(ui.ids.get('joint').disabled,true);
    ui.ids.get('reset').click();await drain();assert(!ui.pending.some(request=>request.path==='/api/reset'));
    ui.resolveNext('/api/step');await drain();ui.resolveNext('/api/reset');await drain();
    assert.equal(ui.backendTaskState().task_metrics.observed_substeps,0);assert.match(ui.ids.get('record-status').textContent,/Recording/);
    ui.ids.get('stop-record').click();await drain();ui.resolveNext('/api/record/stop');await drain();
    assert(!ui.requests.some(request=>['/api/actor','/api/joint'].includes(request.path)));
    assert(!ui.requests.some(request=>/\/(policy|training|rl-evaluation)\.json/.test(request.path)),'Human-only metadata must not trigger legacy learning artifact reads');
    assert.equal(ui.maxActiveSteps(),1);
  }
}

async function sharedExerciseBaselineSelectionIsNotCalledNewLearning() {
  for(const [kind,label,explanation] of [['coast_unchanged','Selected coast baseline',/unchanged zero-command baseline/],
      ['authored_unchanged','Selected authored baseline',/unchanged programmed feedback/]]) {
    const ui=fixture('shared-world',{sharedExercise:true,exercisePolicyAvailable:true,exercisePolicy:attachedExercisePolicy(kind)});await ui.ready();
    assert.equal(ui.ids.get('policy').textContent,label);assert.match(ui.ids.get('exercise-policy-summary').textContent,explanation);
    assert.match(ui.ids.get('exercise-policy-summary').textContent,/not new learned gains/);
    assert.match(ui.ids.get('training-caption').textContent,/not an independent evaluation result/);
    ui.ids.get('policy').click();ui.ids.get('step').click();await drain();
    assert.deepEqual(ui.pending[0].data,{steps:1,policy:true},'Installed baseline policy intent remains distinct from the coast/reference selector');
    ui.resolveNext('/api/step');await drain();
  }
}

async function sharedExerciseResidualPolicyExplainsItsAuthoredBaseAndKeepsHumanOnlyRequests() {
  for (const rootAssistance of [false, true]) {
    const policy=attachedExercisePolicy('searched_joint_residuals',{profile:'shared-exercise-policy-v2'});
    const ui=fixture('shared-world',{sharedExercise:true,rootAssistance,exercisePolicyAvailable:true,exercisePolicy:policy});await ui.ready();
    assert.equal(ui.ids.get('policy').disabled,false);
    assert.equal(ui.ids.get('policy').textContent,'Searched joint corrections');
    assert.match(ui.ids.get('exercise-policy-summary').textContent,/phase-dependent joint corrections on top of programmed feedback/);
    assert.match(ui.ids.get('exercise-policy-summary').textContent,/permitted root assistance remains programmed, not learned/);
    assert.match(ui.ids.get('exercise-evaluation-summary').textContent,/Independent evaluation unavailable/);
    assert.equal(ui.ids.get('joint').children.length,rootAssistance?22:19);
    assert.equal(ui.ids.get('actor').disabled,true);
    ui.ids.get('policy').click();ui.ids.get('step').click();await drain();
    assert.deepEqual(ui.pending[0].data,{steps:1,policy:true});ui.resolveNext('/api/step');await drain();
    assert(!ui.requests.some(request=>['/api/actor','/api/joint'].includes(request.path)));
  }
  const zero=fixture('shared-world',{sharedExercise:true,exercisePolicyAvailable:true,
    exercisePolicy:attachedExercisePolicy('authored_unchanged',{profile:'shared-exercise-policy-v2',training_score_improved_over_authored:false})});await zero.ready();
  assert.equal(zero.ids.get('policy').textContent,'Selected authored baseline');
  assert.match(zero.ids.get('exercise-policy-summary').textContent,/unchanged programmed feedback, not new learned gains/);
}

async function sharedExerciseMissingPolicyMetadataNeverEnablesLearnedControl() {
  for(const [available,policy] of [[false,attachedExercisePolicy()],[true,null],['true',attachedExercisePolicy()],
      [true,attachedExercisePolicy('unknown')],[true,attachedExercisePolicy('searched_gains',{policy_sha256:'invalid'})],
      [true,attachedExercisePolicy('searched_gains',{privileged_simulator_features:false})],
      [true,attachedExercisePolicy('searched_joint_residuals')],
      [true,attachedExercisePolicy('searched_gains',{profile:'shared-exercise-policy-v2'})],
      [true,attachedExercisePolicy('coast_unchanged',{profile:'shared-exercise-policy-v2'})],
      [true,attachedExercisePolicy('searched_joint_residuals',{profile:'shared-exercise-policy-v3'})],
      [true,attachedExercisePolicy('searched_joint_residuals',{profile:'shared-exercise-policy-v3',training_feasible:'true'})]]) {
    const ui=fixture('shared-world',{sharedExercise:true,exercisePolicyAvailable:available,exercisePolicy:policy});await ui.ready();
    assert.equal(ui.ids.get('policy').disabled,true);assert.equal(ui.ids.get('start-learned').disabled,true);
    assert.equal(ui.ids.get('mode-coast').disabled,false);assert.equal(ui.ids.get('reference-servo').disabled,false);
    const before=ui.requests.length;ui.ids.get('policy').click();ui.ids.get('start-learned').click();await drain();assert.equal(ui.requests.length,before);
    assert.match(ui.ids.get('exercise-policy-summary').textContent,/No validated human-only policy/);
  }
  const missing=fixture('shared-world',{sharedExercise:true,missingExerciseState:true,exercisePolicyAvailable:true,exercisePolicy:attachedExercisePolicy()});await missing.ready();
  assert.equal(missing.ids.get('policy').disabled,true);assert.equal(missing.ids.get('start-learned').disabled,true);
}

async function sharedExerciseFeasibilityPolicySeparatesLimitsFromRewardAndEvaluation() {
  for (const feasible of [false,true]) {
    const policy=attachedExercisePolicy('searched_joint_residuals',{profile:'shared-exercise-policy-v3',
      training_feasible:feasible,training_score_improved_over_authored:false});
    const ui=fixture('shared-world',{sharedExercise:true,exercisePolicyAvailable:true,exercisePolicy:policy});await ui.ready();
    assert.equal(ui.ids.get('policy').disabled,false);
    assert.match(ui.ids.get('training-caption').textContent,/prioritized task limits before reward/);
    assert.match(ui.ids.get('training-caption').textContent,feasible?/Both training starts passed every task limit/:/At least one training start still failed/);
    assert.match(ui.ids.get('training-caption').textContent,/Independent results are shown separately/);
    assert.match(ui.ids.get('exercise-evaluation-summary').textContent,/Independent evaluation unavailable/);
    assert.match(ui.ids.get('exercise-outcome').textContent,/In progress/);
    ui.ids.get('policy').click();ui.ids.get('step').click();await drain();
    assert.deepEqual(ui.pending[0].data,{steps:1,policy:true});ui.resolveNext('/api/step');await drain();
  }
}

async function sharedExerciseEvaluationStaysSeparateFromTrainingAndLiveOutcome() {
  const scores=Object.fromEntries(['coast','random','authored','learned'].map((name,index)=>[name,{successes:[0,0,1,2][index],episodes:3}]));
  const policy=attachedExercisePolicy('searched_gains',{evaluation_status:'evaluated',evaluation_successes:scores});
  const ui=fixture('shared-world',{sharedExercise:true,exercisePolicyAvailable:true,exercisePolicy:policy,
    exerciseInitialMetrics:{completed_interval:true,success:false,simulated_duration_s:.2,observed_substeps:48,valid_substeps:48,failed_gates:['equipment_stability:block']}});await ui.ready();
  assert.match(ui.ids.get('exercise-evaluation-summary').textContent,/Attached independent changed-start evaluation/);
  assert.deepEqual(ui.ids.get('scores').children.map(box=>box.children[0].textContent),['0/3','0/3','1/3','2/3']);
  assert.match(ui.ids.get('training-caption').textContent,/Training selection score: 0\.700/);
  assert.match(ui.ids.get('exercise-outcome').textContent,/^Not passed/);assert.equal(ui.ids.get('play-state').textContent,'Trial finished');
  assert(!ui.ids.get('live').drawCalls.some(call=>call.name==='setLineDash'&&call.args[0]?.length));
  assert(!ui.requests.some(request=>/\/(training|rl-evaluation)\.json/.test(request.path)));
  for(const change of [{evaluation_status:'failed'},{evaluation_successes:null},
      {evaluation_successes:{...scores,learned:{successes:null,episodes:3}}},
      {evaluation_successes:{...scores,learned:{successes:1,episodes:2}}}]) {
    const invalid=fixture('shared-world',{sharedExercise:true,exercisePolicyAvailable:true,exercisePolicy:{...policy,...change}});await invalid.ready();
    assert.equal(invalid.ids.get('scores').children.length,0);assert.match(invalid.ids.get('exercise-evaluation-summary').textContent,/unavailable or incomplete/);
    assert.equal(invalid.ids.get('policy').disabled,false,'Missing evaluation does not erase a validated installed controller');
  }
}

async function sharedExerciseComparisonDisclosesResetScopeAndLegacyUnknowns() {
  const scores=Object.fromEntries(['coast','random','authored','learned'].map(name=>[name,{successes:0,episodes:3}]));
  const humanOnly={profile:'shared-exercise-human-jitter-v1',human_jitter_scale:.1,object_position_jitter_world:0,
    object_velocity_jitter_world_s:0,equipment_stability_reference:'actual_reset_state',equipment_reset_state:'unchanged_scene_pose_and_velocity'};
  const legacy={profile:'shared-exercise-reset-v1',human_jitter_scale:.1,object_position_jitter_world:.025,
    object_velocity_jitter_world_s:.12,equipment_stability_reference:'actual_reset_state'};
  for(const [protocol,expected] of [[humanOnly,/human-only perturbations.*exact scene pose and velocity.*not a changed-equipment test/],
      [legacy,/Legacy reset.*position jitter 0\.025.*velocity jitter 0\.120.*Support-preserving starts are not established/],
      [undefined,/Reset protocol unspecified/],[null,/Reset protocol unspecified/],
      [{...humanOnly,object_velocity_jitter_world_s:.12},/Reset protocol unspecified/]]) {
    const policy=attachedExercisePolicy('authored_unchanged',{evaluation_status:'evaluated',evaluation_successes:scores,evaluation_reset_protocol:protocol});
    const ui=fixture('shared-world',{sharedExercise:true,exercisePolicyAvailable:true,exercisePolicy:policy});await ui.ready();
    assert.match(ui.ids.get('exercise-evaluation-summary').textContent,expected);
    assert.deepEqual(ui.ids.get('scores').children.map(box=>box.children[0].textContent),Array(4).fill('0/3'));
    assert.equal(ui.ids.get('policy').textContent,'Selected authored baseline');
    assert(!ui.requests.some(request=>/\/(training|rl-evaluation)\.json/.test(request.path)));
  }
}

function savedFailureInspection(session = 'session-123456789abc') {
  const crossing = {object_id:'block',metric:'speed_model_units_s',threshold:.15,value:.251,previous_safe_elapsed_s:1.866667,
    first_failed_elapsed_s:1.9,event_index:66,control_step:57,reference_clock_s:1.966667,last_valid_geometry_elapsed_s:1.9,
    engine_state_status:'current',source_frame_bracket:{lower:{source_frame_index:58},upper:{source_frame_index:62},frames_without_reference_between:[59,60,61]},
    boundary_contacts:[{objects:['right_elbow_geom','block'],human_geom:'right_elbow_geom',human_segment:'right_forearm'}]};
  const spin = {...clone(crossing),metric:'angular_speed_rad_s',value:2.236,threshold:2};
  const first = {...clone(crossing),crossings:[clone(crossing),clone(spin)]};
  const episode = {reset_event_index:9,seed:0,controller_counts:{learned:254},first_equipment_failure:first,
    first_crossings:[crossing,spin],terminal:{completed_interval:true,success:false,failed_gates:['equipment_stability:block'],elapsed_s:8.466667}};
  const prior = clone(episode);prior.reset_event_index=0;prior.controller_counts={manual:2,coast:2,authored:2,learned:2};
  prior.first_equipment_failure.first_failed_elapsed_s=.233333;prior.first_equipment_failure.event_index=7;prior.terminal.completed_interval=false;
  return {status:'available',session,timeline:{profile:'shared-exercise-failure-timeline-v1',scene_sha256:'a'.repeat(64),task_sha256:'b'.repeat(64),
    gates_sha256:'c'.repeat(64),policy_sha256:null,integrity:{manifest_sha256:'d'.repeat(64),verified_scope:'scene_task_policy_journal_metadata_only',media_verified:false,action_replay_performed:false},
    source_clock:{kind:'declared_human_reference_clock',native_video_time_available:false},episodes:[prior,episode]}};
}

function explainButton(ui, index = 0) { return ui.ids.get('recordings').children[index].children.find(child => child.tagName === 'BUTTON'); }

async function savedTrialExplainsEquipmentFailureWithoutChangingLiveState() {
  const result=savedFailureInspection();const ui=fixture('shared-world',{sharedExercise:true,savedRecordings:[{name:result.session,complete:true}],recordingInspection:result});await ui.ready();
  assert(!ui.requests.some(row=>row.path.startsWith('/api/recording-inspection')),'Read only after an explicit selection');
  const before=ui.backendTaskState(), requests=ui.requests.length, media=ui.ids.get('source').currentTime;
  explainButton(ui).click();await drain();
  assert.equal(ui.requests.length,requests+1);assert.deepEqual(ui.backendTaskState(),before);assert.equal(ui.ids.get('source').currentTime,media);
  assert.equal(ui.ids.get('recording-diagnosis-content').hidden,false);
  assert.equal(ui.ids.get('recording-diagnosis-episode').value,'1');
  assert.match(ui.ids.get('recording-diagnosis-outcome').textContent,/interval complete; task did not pass/);
  assert.match(ui.ids.get('recording-diagnosis-first').textContent,/1\.900s after reset.*speed.*0\.251.*allowed 0\.150.*spin/);
  assert.match(ui.ids.get('recording-diagnosis-clock').textContent,/1\.867–1\.900s.*event 66.*reference clock: 1\.967s.*58–62.*3 intervening/);
  assert.match(ui.ids.get('recording-diagnosis-contacts').textContent,/right forearm collision shape.*Co-occurrence, not proof of cause/);
  assert.equal(ui.ids.get('recording-diagnosis-crossings').children.length,2);
  ui.ids.get('recording-diagnosis-episode').value='0';ui.ids.get('recording-diagnosis-episode').dispatch('change');
  assert.match(ui.ids.get('recording-diagnosis-first').textContent,/0\.233s/);
  assert.match(ui.ids.get('recording-diagnosis-outcome').textContent,/before a complete passing task/);
  assert.equal(ui.requests.length,requests+1,'Episode selection is local metadata, not a new request');
  explainButton(ui).click();await drain();assert.equal(ui.requests.length,requests+1,'Repeated selection reuses this summary');
  assert(ui.ids.get('recording-diagnosis-links').children[0].href.includes('/replay.html?'));
}

async function savedTrialInspectionBoundsPollingAndOffersRetryAfterHiding() {
  const result=savedFailureInspection();const ui=fixture('shared-world',{sharedExercise:true,savedRecordings:[{name:result.session,complete:true}],
    recordingInspection:result,recordingInspectionLoading:40});await ui.ready();explainButton(ui).click();await drain();
  for(let index=0;index<11;index++)await ui.nextTimer();
  assert.equal(ui.requests.filter(row=>row.path.startsWith('/api/recording-inspection')).length,12);
  assert.equal(ui.timers.size,0);assert.equal(ui.ids.get('recording-diagnosis-retry').hidden,false);
  ui.ids.get('recording-diagnosis-retry').click();await drain();assert(ui.timers.size>0);
  ui.visibility(true);assert.equal(ui.timers.size,0);assert.equal(ui.ids.get('recording-diagnosis-retry').hidden,false);
  assert.equal(ui.ids.get('recording-diagnosis-content').hidden,true);
}

async function savedTrialSelectionIgnoresSupersededRepliesAndKeepsOneRequest() {
  const first=savedFailureInspection(),second=savedFailureInspection('session-abcdef012345');
  const ui=fixture('shared-world',{sharedExercise:true,deferRecordingInspection:true,
    savedRecordings:[first,second].map(row=>({name:row.session,complete:true}))});await ui.ready();
  explainButton(ui,0).click();await drain();explainButton(ui,1).click();await drain();
  assert.equal(ui.pending.length,1);
  ui.resolveRecording(first);await drain();
  assert.equal(ui.ids.get('recording-diagnosis-content').hidden,true);assert.equal(ui.pending.length,1);
  assert(ui.pending[0].path.endsWith(second.session));
  ui.resolveRecording(second);await drain();
  assert.match(ui.ids.get('recording-diagnosis-provenance').textContent,new RegExp(second.session));
  assert.equal(ui.pending.length,0);
}

async function savedTrialMissingEquipmentFailureIsNotAPassOrANativeClock() {
  const result=savedFailureInspection();for(const episode of result.timeline.episodes){episode.first_equipment_failure=null;episode.first_crossings=[];episode.terminal.failed_gates=['angular_rmse_rad'];}
  const ui=fixture('shared-world',{sharedExercise:true,savedRecordings:[{name:result.session,complete:true}],recordingInspection:result});await ui.ready();
  explainButton(ui).click();await drain();assert.match(ui.ids.get('recording-diagnosis-first').textContent,/No equipment-limit crossing.*does not mean the task passed/);
  assert.match(ui.ids.get('recording-diagnosis-episode-evidence').textContent,/angular_rmse_rad/);
  assert.match(ui.ids.get('recording-diagnosis-clock').textContent,/No failure clock is inferred/);
  const invalid=clone(result);invalid.timeline.source_clock.native_video_time_available=true;
  const bad=fixture('shared-world',{sharedExercise:true,savedRecordings:[{name:result.session,complete:true}],recordingInspection:invalid});await bad.ready();
  explainButton(bad).click();await drain();assert.equal(bad.ids.get('recording-diagnosis-content').hidden,true);
  assert.match(bad.ids.get('recording-diagnosis-status').textContent,/unavailable.*No time or success is inferred/);
}

checks.push(sharedExerciseShowsImmutableTaskPassiveEquipmentAndExplicitAssistance,sharedExerciseManualCoastAndAuthoredRequestsStayDistinct,
  sourceAndSimulationWindowsKeepPlaybackIndependent,sourceWindowPreservesUnknownAndFailedGeometryClocks,
  sharedWorldV2FullReferenceWindowDoesNotClipOrClaimFullVideo,
  sharedExerciseReportsWholeIntervalMetricsAndExactOutcomeStates,sharedExerciseMissingAndFailureStatesNeverInventTimeOrSuccess,
  sharedExerciseRecordingAndInspectionPreserveTheTaskWrapper,sharedExerciseMissingContractAndContactFractionStayUnavailable,
  sharedExerciseOptionalPolicyRoutesHumanOnlyAndPreservesRecording,sharedExerciseBaselineSelectionIsNotCalledNewLearning,
  sharedExerciseResidualPolicyExplainsItsAuthoredBaseAndKeepsHumanOnlyRequests,
  sharedExerciseFeasibilityPolicySeparatesLimitsFromRewardAndEvaluation,
  sharedExerciseMissingPolicyMetadataNeverEnablesLearnedControl,sharedExerciseEvaluationStaysSeparateFromTrainingAndLiveOutcome,
  sharedExerciseComparisonDisclosesResetScopeAndLegacyUnknowns,
  savedTrialExplainsEquipmentFailureWithoutChangingLiveState,savedTrialInspectionBoundsPollingAndOffersRetryAfterHiding,
  savedTrialSelectionIgnoresSupersededRepliesAndKeepsOneRequest,savedTrialMissingEquipmentFailureIsNotAPassOrANativeClock,
  fiveStagesExposeCountsAndLazyGeometryWithoutChangingPhysics,inspectionNamesKeepControlIdsAndUnavailablePolicyDistinct,
  inspectionPollingIsBoundedCachedAndIndependentOfRecording,inspectionTimeoutOffersRetryWithoutAnUnboundedPollLoop,
  poseCoverageSeeksSourceWithoutChangingSimulationOrFetchingRawPoses,
  failedPoseRunRemainsInspectableAndSeeksAfterMetadataLoads,coverageBackgroundLoadDoesNotRestartOrPollAfterCompletion,
  spatialRigidExposesEightControlsThreeViewsAndHonestUntrainedState,spatialRigidLearnedModeUnlocksOnlyWhenPolicyIsSealed,
  manualObjectVideosDoNotClaimTrainingOrFetchMissingPolicies,
  multiObjectActorSelectionLocksDuringRecordingAndRestoresTrainedActor,spatialRigidActorSelectionSwitchesDynamicObjectsOnly,
  evidenceProvenanceSeparatesSourceKinematicsControllersAndSimulation,
  sharedWorldShowsBodyObjectsAndContactsInAllThreeViews,sharedWorldRoutesHumanMotorVectorsAndSelectedObjectActions,
  sharedWorldPolicyAvailabilityAndControllerPayloadsStayExplicit,sharedWorldRecordingLocksActorAndSerializesReset,
  sharedWorldKinematicScrubbingCannotAdvanceLivePhysics);
export {Element, html, script, fixture, drain};
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const requestedChecks = new Set(process.argv.slice(2));
  const selectedChecks = requestedChecks.size ? checks.filter(check => requestedChecks.has(check.name)) : checks;
  if (requestedChecks.size && selectedChecks.length !== requestedChecks.size) throw new Error('Unknown UI check name');
  let failed=0;
  for(const check of selectedChecks){try{await check();console.log(`PASS ${check.name}`);}catch(error){failed++;console.error(`FAIL ${check.name}: ${error.stack}`);}}
  if(failed)process.exitCode=1;
}
