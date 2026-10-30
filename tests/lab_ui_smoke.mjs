/** Deterministic UI lifecycle checks without a browser or a development server. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

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
      this.drawCalls.push({name, args});
    }), set: (object, name, value) => (object[name] = value, true)});
  }
  setPointerCapture() {}
  releasePointerCapture() {}
  scrollIntoView() {}
  focus() {}
}

function fixture(profile = 'planar', {deferArtifacts = false, spatial = false, mobile = false, wholeBody = false, cameraBody = false, spatialTrained = false, suppliedGround = false, geometryMode = null, partialSegments = false, floorNormal = null, initialFailure = false, legacyPackageEvent = false, feasibilityStatus = "infeasible", temporalIdentity = false, sourceInterval = false, coverage = null, poseCoverage = null, coverageLoading = 0, generationFailed = false} = {}) {
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
  let state = {objects: [{id: 'actor', position: [.2,.3], velocity: [0,0]}], target: scene.target,
    steps: 0, time_s: 0, terminated: false, truncated: false, success: false};
  if (profile === 'spatial-rigid') {
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
      contact_details: [{objects: ['block', 'floor'], position: [.24, .36, .04], normal: [0, 0, 1], distance: -.0002, normal_force: 9.81}],
      objects: [
        {id: 'block', role: 'active', shape: 'box', size: [.06, .05, .05], color: '#965644', position: [.24, .36, .065], quaternion: [1, 0, 0, 0], velocity: [0, 0, 0], angular_velocity: [0, 0, 0]},
        {id: 'companion', role: 'passive', shape: 'sphere', radius: .035, color: '#C7795D', position: [.5, .42, .075], quaternion: [1, 0, 0, 0], velocity: [0, 0, 0], angular_velocity: [0, 0, 0]},
        {id: 'floor', role: 'static', shape: 'box', size: [.96, .76, .04], color: '#D8C9BA', position: [.5, .4, .02], quaternion: [1, 0, 0, 0], velocity: [0, 0, 0], angular_velocity: [0, 0, 0]},
      ], terminated: false, truncated: false, success: false};
  }
  if (profile === 'exercise') {
    scene.profile=mobile?'exercise-v3':spatial?'exercise-v2':'exercise-v1';scene.joint_names=['left_knee','left_elbow'];
    state.human={height:1.8,segments:[{id:'torso',a:[0,0,.8],b:[0,0,1.5],radius:.1}],
      muscles:[{name:'left_knee_flexor',a:[0,0,.7],b:[0,.1,.3],activation:.4,status:'illustrative'}],
      angles:{left_knee:.2,left_elbow:.3},selected_joint:'left_knee',target_angle:.8};
    if(spatial||mobile)Object.assign(state.human,{view_height:2.5,view_origin:[0,0,.8]});
    if(mobile)state.human.mobility={support_enabled:true,translation:[0,0,0],contact_count:0,contacts:[]};
  }
  if (wholeBody) {
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
  if (partialSegments) {
    scene.human.root_method='two-view-body-v2';
    scene.human.geometry_coverage.left_shoulder={two_view_estimate:59,two_view_proximal_monocular_distal_and_twist:42,monocular_prior_missing_points:11};
    scene.human.geometry_coverage.left_hip={two_view_estimate:90,two_view_proximal_monocular_distal_and_twist:22};
    scene.human.geometry_coverage.right_hip={two_view_estimate:80,two_view_proximal_monocular_distal_and_twist:15,monocular_prior_missing_points:17};
  }
  if (initialFailure) {state.terminated=true;state.simulation_failure={reason:'Synthetic nonfinite dynamics',scope:'Last valid state shown'};}
  let backendReady = false, backendRecording = false, exportStatus = null;
  let savedSessions = [];
  const isSpatialUntrained = profile === 'spatial-rigid' && !spatialTrained;
  const status = () => ({status: backendReady ? generationFailed ? 'failed' : 'ready' : 'running', profile, directory: '/synthetic/lab',
    learned_policy_available: !(cameraBody || isSpatialUntrained),
    ...(generationFailed ? {error:{message:'Synthetic missing source poses'}} : {}),
    ...(suppliedGround ? {ground_clearance:{status:'descriptive_no_contact_evidence',summary:{minimum:.076,median:.356,maximum:.630,positive_gap_poses:112,reference_poses:112}}} : {}),
    ...(sourceInterval ? {requested_source_interval:{first_frame:2580,last_frame:2760,initialization_frame:2620}} : {}),
    ...(wholeBody || cameraBody ? {reference_feasibility:{status:feasibilityStatus,summary:{max_penetration_model_units:.42,
      failed_source_poses:3,source_pose_checks:10,failed_control_grid_poses:7,control_grid_checks:30}}} : {}),
    ...(temporalIdentity ? {result:{pose_identity:{mode:'temporal',repaired_frames:2}}} : {}),
    events: legacyPackageEvent && backendReady ? [{index:1,stage:'whole_body',message:'Existing packaged run without elapsed time'}] : [{index: 0, stage: backendReady ? 'ready' : 'source', elapsed_seconds: .1, message: 'Synthetic UI fixture'}],
    scene, snapshot: state, recording: backendRecording, sessions: savedSessions,
    ...(exportStatus ? {export:{status:exportStatus}} : {}),
    artifacts: [...(backendReady ? [...(profile === 'spatial-rigid' ? ['scene.json', 'interventions.json'] : ['source.mp4', ...(!generationFailed ? ['tracks.gif'] : [])]), ...(!isSpatialUntrained ? ['training.json','rl-evaluation.json'] : []),'repair-ledger.json','reconstruction.json'] : []), ...(coverage ? ['coverage-summary.json', 'reconstruction-coverage.json'] : [])].map(name => ({name,bytes:500}))});
  const artifacts = {'training.json': {completed_episodes: 10, steps: 200, seconds: .3,
    history: [{episode: 10,recent_training_success_rate: .4}]},
    'rl-evaluation.json': {policies: {q_learning: {successes: 3,episodes: 5}, random: {successes: 1,episodes: 5}}},
    'repair-ledger.json': [{proposal: {damping:.8},accepted:true,reason:'Synthetic test'}],
    'reconstruction.json': {before:{rmse_px:10},after:{rmse_px:4}},
    'policy.json': {selected_joint:'left_knee'}};
  if (coverage) artifacts['coverage-summary.json'] = coverage;
  if (cameraBody || isSpatialUntrained) {
    delete artifacts['policy.json']; delete artifacts['training.json']; delete artifacts['rl-evaluation.json'];
    artifacts['reconstruction.json'] = {scope: isSpatialUntrained ? 'Unvalidated 3D rigid scene' : 'Unvalidated camera-aware candidate'};
    artifacts['repair-ledger.json'] = {proposals: []};
  }
  if (profile === 'spatial-rigid' && spatialTrained) {
    artifacts['policy.json'] = {profile: 'spatial-policy-v1', controlled_id: 'block'};
    artifacts['rl-evaluation.json'] = {policies: {learned: {successes: 8, episodes: 10}, random: {successes: 1, episodes: 10}, coast: {successes: 0, episodes: 10}, brake: {successes: 0, episodes: 10}}};
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
  if(profile==='exercise')artifacts['human-reference.json']={frames:[0,1.5,3,4.5].map(source_time_s=>({...clone(state),source_time_s}))};
  const requests = [], pending = [];
  let activeSteps = 0, maxActiveSteps = 0;
  function response(value,ok=true) { return {ok, status: ok?200:400, headers: {get: () => '500'}, json: async () => clone(value)}; }
  function fetch(path, options = {}) {
    const request = {path, data: options.body ? JSON.parse(options.body) : null, options};
    requests.push(request);
    if (path === '/api/status') return Promise.resolve(response(status()));
    if (path === '/api/exercise-coverage') return Promise.resolve(response(coverageLoading-- > 0 ? {status:'loading'} : poseCoverage || {status:'unavailable'}));
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
    } else if (path === '/api/reset' || path === '/api/record/start') {
      state = clone(state); state.steps = 0; state.time_s = 0;
      state.objects[0].position = profile === 'spatial-rigid' ? [.24, .36, .065] : [.2,.3];
      state.terminated = state.truncated = state.success = false;
      delete state.simulation_failure;
    }
    if (path === '/api/record/start') backendRecording = true;
    if (path === '/api/record/stop') {backendRecording = false;exportStatus = 'running';savedSessions = [{name:'trial-001',complete:false}];}
    request.resolve(response({snapshot: state, ...(path==='/api/joint'?{scene}:{}), ...(path.includes('/record/') ? {recording:backendRecording} : {}), ...(path==='/api/record/stop'?{export:'running'}:{})}));
  }
  const storage = new Map();
  vm.runInNewContext(script, {document, window, location: {hash:'#token=synthetic-token',pathname:'/'},
    sessionStorage:{getItem:key=>storage.get(key),setItem:(key,value)=>storage.set(key,value)},
    history:{replaceState(){}}, URLSearchParams, AbortSignal, fetch, setTimeout, clearTimeout,
    performance:{now:()=>now}, console}, {filename:'lab.js'});
  return {ids, pending, requests, timers, resolveNext, document,
    rejectNext(path,message) {const index=pending.findIndex(request=>request.path===path);assert(index>=0);pending.splice(index,1)[0].resolve(response({error:message},false));},
    resolveArtifacts(){for(const request of [...pending])if('artifactValue' in request)resolveNext(request.path);},
    backendState:()=>clone(state),
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
      assert.match(ids.get('status').textContent,/ready/i); assert.equal(ids.get('toggle').disabled,false); assert.equal(ids.get('objects').children.length,profile==='spatial-rigid'?3:1);
      assert.equal(ids.get('scores').children.length,(cameraBody||isSpatialUntrained)?0:(wholeBody||spatialTrained)?4:2); this.stage('play'); }
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
  assert.match(ui.ids.get('spatial-readout').textContent, /contact block↔floor.*Fn 9\.81N/);
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
  ui.ids.get('policy').click(); ui.ids.get('step').click(); await drain();
  assert.deepEqual(ui.pending[0].data, {action: 0, steps: 1, policy: true});
  ui.resolveNext('/api/step'); await drain();
}

checks.push(poseCoverageSeeksSourceWithoutChangingSimulationOrFetchingRawPoses,
  failedPoseRunRemainsInspectableAndSeeksAfterMetadataLoads,coverageBackgroundLoadDoesNotRestartOrPollAfterCompletion,
  spatialRigidExposesEightControlsThreeViewsAndHonestUntrainedState,spatialRigidLearnedModeUnlocksOnlyWhenPolicyIsSealed);
const requestedChecks = new Set(process.argv.slice(2));
const selectedChecks = requestedChecks.size ? checks.filter(check => requestedChecks.has(check.name)) : checks;
if (requestedChecks.size && selectedChecks.length !== requestedChecks.size) throw new Error('Unknown UI check name');
let failed=0;
for(const check of selectedChecks){try{await check();console.log(`PASS ${check.name}`);}catch(error){failed++;console.error(`FAIL ${check.name}: ${error.stack}`);}}
if(failed)process.exitCode=1;
