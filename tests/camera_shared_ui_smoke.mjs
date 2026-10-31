/** Analytical DOM checks only: no browser, native engine, media, network or source admission. */
import assert from 'node:assert/strict';
import {performance} from 'node:perf_hooks';
import {pathToFileURL} from 'node:url';
import {liveFixture, replayFixture} from './shared_exercise_feet_ui_smoke.mjs';
import {drain} from './lab_ui_smoke.mjs';

const copy=value=>JSON.parse(JSON.stringify(value));
const hash=char=>char.repeat(64);
const pins=['scene_sha256','camera_coupling_sha256','source_sha256','body_sha256','source_clock_sha256',
  'metric_context_sha256','kernel_evaluator_sha256','adapter_implementation_sha256','evaluator_sha256'];
function cameraFixture(roots=false){
  const names=['torso_twist','torso_lateral','torso_flex',...['left','right'].flatMap(side=>
    ['shoulder_twist','shoulder_lateral','shoulder','elbow','hip_twist','hip_lateral','hip','knee'].map(joint=>`${side}_${joint}`)),
    'root_x','root_y','root_z'];
  const identity=Object.fromEntries(pins.map((key,i)=>[key,hash('123456789'[i])])),taskHash=hash('a');
  const dt=1001/30000,substeps=9,start=dt,horizon=3,control={root_assistance:roots?'explicit_3_axis':'disabled',object_forces:'forbidden'};
  const scope={success_scope:'authored_simulated_camera_tracking_criteria',physical_validation:'unverified',source_contact_verified:false,
    ground_support_verified:false,equipment_scoring:'unsupported',foot_controls:'unsupported',learning:'unqualified',
    body_scale:'supplied_camera_root_scale_unchanged_monocular_shape',body_mass:'unchanged_authored_model',
    clock_alignment:'original_source_grid_with_unchanged_body_period'};
  const scene={profile:'shared-world-v6',dt,substeps,objects:[],target:null,bounds:[3,2,2.5],controlled_id:'human',learned_policy_available:false,
    task:{object_goal:null,start_offset_s:0,horizon_steps:horizon},human_spec:{dt:1/30},
    action_contract:{human_motor_count:22,names,object_force_channels:0,dynamic_object_ids:[],units:Array(22).fill('model'),physical_limits:Array(22).fill(30)},
    camera_coupling:{...Object.fromEntries(['source_sha256','body_sha256','source_clock_sha256'].map(key=>[key,identity[key]])),
      clock:{control_dt_s:dt,source_fps:30000/1001,source_time_offset_s:start}}};
  const task={schema_version:1,profile:'camera-shared-task-v1',...identity,control,interval:{start_offset_s:0,horizon_steps:horizon},scope,
    gate_profile:'shared-exercise-gates-v1',gates_sha256:hash('b'),metric_context:{schema_version:1,profile:'camera-shared-metric-context-v1',
      scene_sha256:identity.scene_sha256,kernel:'weft.shared_exercise_task._ExerciseMetrics',kernel_branch:'shared-exercise-task-v2',valid_legacy_task:false,
      projection:'identity_22_motor_samples_without_equipment',control_dt_s:dt,source_body_dt_s:1/30,substeps,kernel_arguments_sha256:hash('c'),
      kernel_arguments:{profile:'shared-exercise-task-v2',interval:{start_offset_s:0,horizon_steps:horizon},control,equipment_ids:[],contact_requirements:[],source_scope:scope}}};
  const count=roots?22:19,action_contract={names:names.slice(0,count),joint_channels:Array.from({length:19},(_,i)=>i),
    root_assistance_channels:roots?[19,20,21]:[],root_assistance:control.root_assistance,object_force_channels:[],normalized_bounds:[-1,1]};
  const capabilities={scene_profile:scene.profile,task_contract:{task_type:task.profile,identity:{...identity,task_sha256:taskHash}},
    action_contract:{dimension:count,actuator_names:action_contract.names,object_force_channels:[]},controllers:{manual:true,coast:true,
      authored_reference_controller:true,learned_policy_available:false,trained_target_id:null}};
  function state(observed,valid,failed=false,complete=false){
    const elapsed=valid*dt/substeps,geometryTime=failed?dt:elapsed;
    const engine={profile:'shared-world-v6',camera_coupling_sha256:identity.camera_coupling_sha256,controlled_id:'human',objects:[],target:null,
      time_s:geometryTime,source_time_s:start+geometryTime,steps:failed?1:observed/substeps,substeps,normalized_action:Array(22).fill(0),
      contact_count:0,contact_details:[],contact_summary:{human_object:0},coupled_metrics:{},terminated:false,truncated:false,
      human:{height:1.8,action_names:names,segments:[{id:'torso',a:[1,1,.8],b:[1,1,1.5],radius:.1}],muscles:[],
        control_mode:'direct_motor_actions',angles:Object.fromEntries(names.map(name=>[name,0])),reference:false,
        mobility:{floor_height:0,support_force:[0,0,0],translation:[0,0,0]}}};
    const metrics={profile:'camera-shared-metrics-v1',kernel_branch:'shared-exercise-task-v2',task_sha256:taskHash,
      ...Object.fromEntries(pins.filter(key=>key!=='source_sha256').map(key=>[key,identity[key]])),
      gate_profile:task.gate_profile,gates_sha256:task.gates_sha256,reward_profile:'human_tracking_effort_v1',root_assistance:control.root_assistance,
      virtual_object_forces:'forbidden',gate_applicability:{human_tracking:'applied',equipment_stability:'not_applicable',contact_requirements:'not_applicable'},
      source_scope:scope,equipment:[],contact_requirements:[],success:complete,completed_interval:complete,failure:failed?'analytical sample failure':null,
      failed_gates:failed?['analytical sample failure']:[],observed_substeps:observed,valid_substeps:valid,simulated_duration_s:elapsed,
      valid_observed_duration_s:elapsed,declared_duration_s:horizon*dt,observed_source_time_s:start+elapsed,
      clock_status:failed?'unavailable_after_last_observed_time':'observed',angular_rmse_rad:valid?.1:null,root_rmse_model_units:valid?.01:null,
      peak_angular_rmse_rad:valid?.15:0,peak_root_rmse_model_units:valid?.02:0,max_floor_penetration_model_units:0,
      joint_effort_mean_squared:0,root_assistance_mean_squared:0,reward_integral:0,mean_reward:0,failure_reward_penalty:failed?-1:0};
    return {schema_version:1,profile:'camera-shared-state-v1',...identity,task_sha256:taskHash,engine_state:engine,task_metrics:metrics,
      last_action:Array(count).fill(0),engine_state_status:failed?'last_valid_control_boundary':'current',terminated:failed||complete,truncated:false};
  }
  // This harness flag only skips its feet-geometry augmentation; these are NOT actual saved native states.
  const data={actual_saved:true,evidence_kind:'authored_analytical_fixture',scene,task,action_contract,capabilities,
    states:{initial:state(0,0),active:state(9,9),failed:state(12,11,true),complete:state(27,27,false,true)}};
  data.replay_bundle={scene,exercise_task:task,camera_identity:capabilities.task_contract.identity,
    events:['initial','active','failed','complete'].map((name,i)=>({type:i?'step':'start',controller:i===3?'authored':'manual',
      snapshot:data.states[name].engine_state,exercise_metrics:data.states[name].task_metrics,engine_state_status:data.states[name].engine_state_status}))};
  return copy(data);
}

async function manualCoastAuthoredUseOnlyExplicitHumanChannels(){
  for(const roots of [false,true]){const ui=liveFixture(cameraFixture(roots));await ui.ready();
    assert.equal(ui.ids.get('toggle').disabled,false);assert.equal(ui.ids.get('joint').children.length,roots?22:19);
    assert.equal(ui.ids.get('policy').disabled,true);assert.equal(ui.ids.get('start-learned').disabled,true);
    assert.match(ui.ids.get('world-heading').textContent,/supplied camera and ground/);
    assert.match(ui.ids.get('play-boundary').textContent,/Feet.*unqualified/);
    assert.match(ui.ids.get('window-clocks').textContent,/29\.970030.*0\.033367.*0\.033333/);
    ui.ids.get('exercise-metrics-details').open=true;ui.ids.get('exercise-metrics-details').dispatch('toggle');
    assert.equal(ui.ids.get('exercise-equipment').children.length,0);assert.equal(ui.ids.get('exercise-contacts').children.length,0);
    ui.motor(roots?21:18);ui.action();await drain();let action=ui.pending[0].data.action;
    assert.equal(action.length,roots?22:19);assert.equal(action.at(-1),.25);assert(action.slice(0,-1).every(v=>v===0));
    await ui.reply('/api/step');ui.ids.get('mode-coast').click();ui.ids.get('step').click();await drain();
    assert.deepEqual(ui.pending[0].data,{steps:1,controller:'coast'});await ui.reply('/api/step');
    ui.ids.get('reference-servo').click();ui.ids.get('step').click();await drain();
    assert.deepEqual(ui.pending[0].data,{steps:1,controller:'reference'});await ui.reply('/api/step');
    assert(!ui.requests.some(row=>/api\/(actor|joint|appearance)/.test(row.path)));
  }
}
async function malformedIdentitiesAndMetricsPauseWithoutRelabeling(){
  const mutations=[d=>d.task.profile='shared-exercise-task-v2',d=>d.capabilities.task_contract.identity.source_sha256=hash('0'),
    d=>d.task.metric_context.valid_legacy_task=true,d=>d.task.metric_context.kernel_arguments.control.object_forces='allowed',
    d=>d.states.initial.terminated=true,d=>d.action_contract.root_assistance_channels=[19],
    d=>d.capabilities.controllers.learned_policy_available=true,d=>d.states.initial.source_clock_sha256=hash('0'),
    d=>d.states.initial.engine_state.profile='shared-world-v3',d=>d.states.initial.task_metrics.profile='shared-exercise-task-v2',
    d=>d.states.initial.task_metrics.equipment=[{}],d=>d.states.initial.task_metrics.observed_source_time_s=0,
    d=>d.states.initial.task_metrics.angular_rmse_rad=0,d=>d.states.initial.task_metrics.valid_substeps=1,
    d=>d.states.initial.engine_state.normalized_action[19]=.25,d=>d.states.initial.last_action[0]=true,
    d=>d.states.initial.engine_state.human.segments=Array(33).fill(d.states.initial.engine_state.human.segments[0]),
    d=>d.states.initial.engine_state.contact_details=Array(257).fill({position:[0,0,0]})];
  for(const change of mutations){const ui=liveFixture(cameraFixture(),change);await ui.ready();
    assert.equal(ui.ids.get('toggle').disabled,true);assert.equal(ui.ids.get('record').disabled,true);
    assert.match(ui.ids.get('exercise-outcome').textContent,/unavailable/);ui.action();await drain();assert.equal(ui.pending.length,0);}
}
async function sourceAndFailureClocksRemainDistinct(){
  const ui=liveFixture(cameraFixture());await ui.ready();assert.match(ui.ids.get('exercise-body-errors').textContent,/mean unavailable/);
  ui.update('active');assert.match(ui.ids.get('exercise-equipment-summary').textContent,/Source clock: 0\.067s.*geometry source clock 0\.067s/);
  ui.update('failed');assert.match(ui.ids.get('exercise-equipment-summary').textContent,/0\.074s.*0\.067s/);
  assert.match(ui.ids.get('exercise-failure').textContent,/last valid geometry/);assert.equal(ui.ids.get('step').disabled,true);
  ui.update('complete');assert.match(ui.ids.get('exercise-outcome').textContent,/Passed.*interval complete/);
  assert.equal(ui.ids.get('step').disabled,true);
}
async function recordingWaitsForActionsAndResets(){
  const ui=liveFixture(cameraFixture());await ui.ready();ui.ids.get('record').click();await drain();
  assert.equal(ui.pending[0].path,'/api/record/start');await ui.reply('/api/record/start','initial');
  assert.equal(ui.pending.filter(row=>row.path==='/api/step').length,1);
  ui.ids.get('reset').click();await drain();assert(!ui.pending.some(row=>row.path==='/api/reset'));
  await ui.reply('/api/step');await ui.reply('/api/reset','initial');
  ui.ids.get('stop-record').click();await drain();assert.equal(ui.pending[0].path,'/api/record/stop');await ui.reply('/api/record/stop');
  assert(!ui.requests.some(row=>/train|appearance|policy.json/.test(row.path)));
}

function cameraReceipt(data){
  const session='camera-analytic',identities={...data.capabilities.task_contract.identity};delete identities.scene_sha256;
  Object.assign(identities,{source_run_manifest_sha256:hash('d'),recording_manifest_sha256:hash('e'),receipt_manifest_sha256:hash('f'),
    receipt_sha256:hash('a'),recording_scene_sha256:hash('b'),engine_scene_sha256:data.task.scene_sha256,policy_sha256:null});
  const paths={source_run_manifest_sha256:'run/manifest.json',recording_manifest_sha256:`sessions/${session}/manifest.json`,
    receipt_manifest_sha256:`sessions/${session}/evidence/manifest.json`,receipt_sha256:`sessions/${session}/evidence/receipt.json`,
    recording_scene_sha256:`sessions/${session}/scene.json`,engine_scene_sha256:'run/scene.json',task_sha256:'run/exercise-task.json'};
  return {schema_version:1,profile:'recording-evidence-v1',session,partition:'exposed_development',physical_validation:'unverified',identities,
    verified_metadata:Object.entries(paths).map(([key,path])=>({source_name:path,path,sha256:identities[key],bytes:100})),
    recorded:{events:4,actions:2,resets:1},replay:{status:'passed',protocol:'camera-shared-full-state-replay-v1',events_checked:4,actions_checked:2,resets_checked:1,learned_actions_verified:null}};
}
async function receiptRequiresCameraPinsAndNoLearnedProof(){
  const ui=liveFixture(cameraFixture());await ui.ready();const saved=cameraReceipt(ui.data),api=ui.context.feetTest;
  assert(api.recordingEvidenceValid(saved,saved.session));api.applyRecordingEvidence(saved);
  assert.match(ui.ids.get('recording-evidence-replay').textContent,/Actual camera state.*No learned-policy check/);
  assert.match(ui.ids.get('recording-evidence-status').textContent,/not rerun/);
  for(const mutate of [v=>v.replay.protocol='shared-exercise-exact-action-state-task-v1',v=>v.identities.body_sha256=hash('0'),
    v=>v.replay.learned_actions_verified=0,v=>v.identities.policy_sha256=hash('0')]){
    const changed=copy(saved);mutate(changed);assert.equal(api.recordingEvidenceValid(changed,changed.session),false);}
}
function offlineReplayRequiresActualProfilesAndExplicitBundleIdentity(){
  for(const roots of [false,true]){const replay=replayFixture(cameraFixture(roots));
    assert.match(replay.ids.get('recording-title').textContent,/Camera-conditioned/);assert.equal(replay.ids.get('play').disabled,false);
    assert.match(replay.ids.get('camera-status').textContent,/mean unavailable/);replay.select(1);
    assert.match(replay.ids.get('camera-status').textContent,/9 valid \/ 9 observed/);replay.turn();
    replay.select(2);assert.match(replay.ids.get('camera-status').textContent,/0\.074s.*0\.067s.*last valid boundary/);
    replay.select(3);assert.match(replay.ids.get('status').textContent,/passed.*interval complete/);
  }
  for(const change of [d=>delete d.replay_bundle.camera_identity,d=>d.replay_bundle.camera_identity.body_sha256=hash('0'),
    d=>d.replay_bundle.events[0].snapshot.profile='shared-world-v3',d=>d.replay_bundle.events[0].exercise_metrics.task_sha256=hash('0'),
    d=>d.replay_bundle.events[0].exercise_metrics.angular_rmse_rad=0]){
    const data=cameraFixture();change(data);const replay=replayFixture(data);assert.equal(replay.ids.get('play').disabled,true);
    assert.match(replay.ids.get('status').textContent,/unavailable/);assert(!replay.ids.get('view').drawCalls.some(row=>row.name==='stroke'));}
  const data=cameraFixture();data.replay_bundle.events[1].controller='learned';const replay=replayFixture(data);replay.select(1);
  assert.equal(replay.ids.get('play').disabled,true);assert.match(replay.ids.get('status').textContent,/unavailable/);
}
async function repeatedUpdatesReuseClockTextAndOfflineValidation(){
  for(const stress of [false,true]){
    const data=cameraFixture();
    if(stress){for(const state of Object.values(data.states)){
      state.engine_state.human.segments=Array.from({length:32},(_,i)=>({...state.engine_state.human.segments[0],id:`analytic_${i}`}));
      state.engine_state.human.muscles=Array.from({length:38},(_,i)=>({name:`analytic_${i}`,a:[1,1,.8],b:[1,1,1.5],activation:.1}));
      state.engine_state.contact_details=Array.from({length:256},()=>({kind:'human_floor',position:[1,1,0]}));
    }
    data.replay_bundle.events.forEach((event,i)=>{event.snapshot=data.states[['initial','active','failed','complete'][i]].engine_state;});}
    const ui=liveFixture(data);await ui.ready();ui.update('active');assert.equal(ui.ids.get('step').disabled,false);
    const watched=[ui.ids.get('window-live'),ui.ids.get('exercise-equipment-summary')];let writes=0;
    for(const node of watched){let saved=node.textContent;Object.defineProperty(node,'textContent',{get:()=>saved,set:value=>{saved=value;writes++;}});}
    const start=performance.now();for(let i=0;i<200;i++)ui.update('active');const liveMs=performance.now()-start;
    assert.equal(writes,0);assert.equal(ui.pending.length,0);
    const replay=replayFixture(data);replay.select(1);const output=replay.ids.get('camera-status');let cached=output.textContent,replayWrites=0;
    Object.defineProperty(output,'textContent',{get:()=>cached,set:value=>{cached=value;replayWrites++;}});
    const replayStart=performance.now();for(let i=0;i<200;i++)replay.turn();const replayMs=performance.now()-replayStart;
    assert.equal(replayWrites,0);
    console.log(JSON.stringify({measurement:'camera_200_unchanged_scripted_updates',fixture:stress?'bounded_geometry_stress':'tiny_analytical',
      live_ms:liveMs,replay_ms:replayMs,live_clock_and_summary_text_writes:writes,offline_diagnostic_text_writes:replayWrites,
      scope:'Analytical mocked DOM; not browser/native responsiveness'}));
  }
}

const checks=[manualCoastAuthoredUseOnlyExplicitHumanChannels,malformedIdentitiesAndMetricsPauseWithoutRelabeling,sourceAndFailureClocksRemainDistinct,
  recordingWaitsForActionsAndResets,receiptRequiresCameraPinsAndNoLearnedProof,offlineReplayRequiresActualProfilesAndExplicitBundleIdentity,
  repeatedUpdatesReuseClockTextAndOfflineValidation];
export {cameraFixture,cameraReceipt,liveFixture,replayFixture};
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
  const selected=checks.filter(check=>process.argv.length<3||process.argv.slice(2).includes(check.name));
  assert(selected.length);let failed=0;
  for(const check of selected){try{await check();console.log(`PASS ${check.name}`);}catch(error){failed++;console.error(`FAIL ${check.name}\n${error.stack}`);}}
  console.log(JSON.stringify({passed:selected.length-failed,failed,scope:'Analytical camera UI only'}));process.exitCode=failed?1:0;
}
