/** Saved-boundary presentation checks; no browser, decoder, server or physics. */
import assert from 'node:assert/strict';
import {fixture, drain} from './lab_ui_smoke.mjs';

const copy = value => JSON.parse(JSON.stringify(value));
const thresholds = {angular_rmse_rad:.35,root_rmse_model_units:.15,peak_angular_rmse_rad:.7,
  peak_root_rmse_model_units:.3,max_floor_penetration_model_units:.025};
const aggregation = metric => metric.startsWith('peak_') || metric.startsWith('max_') ? 'cumulative_peak' : 'running_rms';
function crossing(metric, overrides = {}) {
  return {metric,threshold:thresholds[metric],value:.958023,previous_value:.648234,aggregation:aggregation(metric),
    status:'first_reported_boundary',event_index:29,control_step:20,previous_safe_elapsed_s:19/30,first_failed_elapsed_s:20/30,
    last_valid_geometry_elapsed_s:20/30,engine_state_status:'current',clock_status:'observed',reference_clock_s:20/30,
    source_frame_bracket:{lower:{source_frame_index:20},upper:{source_frame_index:20},frames_without_reference_between:[]},...overrides};
}
function summary() {
  const peak = crossing('peak_angular_rmse_rad');
  const mean = crossing('angular_rmse_rad',{value:.358085,previous_value:.314523,event_index:30,control_step:21,
    previous_safe_elapsed_s:20/30,first_failed_elapsed_s:21/30,reference_clock_s:21/30});
  const terminal = Object.entries(thresholds).map(([metric,threshold]) => ({metric,threshold,aggregation:aggregation(metric),
    value:metric==='peak_angular_rmse_rad'?.958023:metric==='angular_rmse_rad'?.153040:0,
    status:metric==='peak_angular_rmse_rad'?'above_limit':'within_limit'}));
  const human = {profile:'shared-exercise-human-tracking-timeline-v1',first_failure:{...copy(peak),episode_index:1,reset_event_index:9,crossings:[copy(peak)]},
    first_crossings:[peak,mean],terminal};
  const episode = {episode_index:1,reset_event_index:9,seed:12,controller_counts:{learned:234},human_tracking:human,
    first_equipment_failure:null,first_crossings:[],terminal:{completed_interval:true,success:false,failed_gates:['peak_angular_rmse_rad'],elapsed_s:7.8}};
  const initial = copy(episode);initial.episode_index=0;initial.reset_event_index=0;initial.terminal.completed_interval=false;
  initial.terminal.failed_gates=[];initial.human_tracking.first_failure=null;initial.human_tracking.first_crossings=[];
  initial.human_tracking.terminal.forEach(row=>{row.value=row.aggregation==='running_rms'?null:0;row.status=row.value===null?'unavailable':'within_limit';});
  return {status:'available',session:'session-123456789abc',timeline:{profile:'shared-exercise-failure-timeline-v1',events:244,
    scene_sha256:'a'.repeat(64),task_sha256:'b'.repeat(64),gates_sha256:'c'.repeat(64),policy_sha256:null,
    integrity:{manifest_sha256:'d'.repeat(64),verified_scope:'scene_task_policy_journal_metadata_only',media_verified:false,action_replay_performed:false},
    source_clock:{kind:'declared_human_reference_clock',native_video_time_available:false},equipment_stability_applicability:'not_applicable',
    episodes:[initial,episode]}};
}
async function inspect(result) {
  const ui = fixture('shared-world',{sharedExercise:true,savedRecordings:[{name:result.session,complete:true}],recordingInspection:result});
  await ui.ready();
  const before = {requests:ui.requests.length,state:ui.backendTaskState(),videoTime:ui.ids.get('source').currentTime};
  assert(!ui.requests.some(row=>row.path.startsWith('/api/recording-inspection')));
  ui.ids.get('recordings').children[0].children.find(child=>child.tagName==='BUTTON').click();await drain();
  assert.equal(ui.requests.length,before.requests+1);assert.deepEqual(ui.backendTaskState(),before.state);
  assert.equal(ui.ids.get('source').currentTime,before.videoTime);
  return ui;
}
const text = (ui,id) => ui.ids.get(id).textContent;
const rows = (ui,id) => ui.ids.get(id).children.map(row=>row.children.map(cell=>cell.textContent));

async function peakFailureAndRecoveredAverageRemainDistinct() {
  const ui=await inspect(summary());
  assert.equal(ui.ids.get('recording-diagnosis-content').hidden,false);
  assert.equal(text(ui,'recording-diagnosis-heading'),'Saved trial: body tracking');
  assert.match(text(ui,'recording-diagnosis-outcome'),/interval complete; task did not pass/);
  assert.match(text(ui,'recording-diagnosis-human'),/Peak body-angle error: 0\.958 rad; allowed 0\.700 rad/);
  assert.match(text(ui,'recording-diagnosis-human'),/0\.153 rad ended below 0\.350 rad after an earlier crossing/);
  assert.doesNotMatch(text(ui,'recording-diagnosis-human'),/stayed below/);
  assert.match(text(ui,'recording-diagnosis-human-first'),/0\.633–0\.667s after reset.*step 20, event 29.*retain earlier deviations/);
  assert.match(text(ui,'recording-diagnosis-first'),/not applicable.*none are declared.*no success credit/);
  assert.equal(ui.ids.get('recording-diagnosis-contacts').hidden,true);
  assert.match(text(ui,'recording-diagnosis-human-clock'),/frames 20–20.*No exact substep or cause.*Native video time is unavailable/);
  assert.match(text(ui,'recording-diagnosis-limits'),/not the current simulation.*does not identify a culprit joint.*uncalibrated/);
  assert.equal(rows(ui,'recording-diagnosis-human-metrics').length,5);
  const crossings=rows(ui,'recording-diagnosis-crossings');assert.equal(crossings.length,2);
  assert.match(crossings[1].join(' '),/Average body-angle error.*0\.358 \/ 0\.350.*30 · step 21 · average may recover/);
  assert.match(ui.ids.get('recording-diagnosis-links').children[0].textContent,/select state 30 of 244 \(event 29\)/);
  assert.match(ui.ids.get('recording-diagnosis-links').children[0].href,/sessions\/session-123456789abc\/replay\.html\?/);
  const requestCount=ui.requests.length;
  ui.ids.get('recording-diagnosis-episode').value='0';ui.ids.get('recording-diagnosis-episode').dispatch('change');
  assert.match(text(ui,'recording-diagnosis-human'),/unavailable.*No missing value is treated as zero or a pass/);
  assert.match(rows(ui,'recording-diagnosis-human-metrics')[0].join(' '),/unavailable \/ 0\.350.*Unavailable — not a pass/);
  assert.equal(rows(ui,'recording-diagnosis-crossings').length,0);assert.equal(ui.requests.length,requestCount);
}

async function missingPrecedingMeasurementDoesNotInventSafeTime() {
  const value=summary(),human=value.timeline.episodes[1].human_tracking;
  const first=crossing('angular_rmse_rad',{previous_value:null,previous_safe_elapsed_s:null,status:'first_observed_boundary'});
  human.first_crossings=[first];human.first_failure={...copy(first),episode_index:1,reset_event_index:9,crossings:[copy(first)]};
  const ui=await inspect(value);
  assert.equal(ui.ids.get('recording-diagnosis-content').hidden,false);
  assert.match(text(ui,'recording-diagnosis-human-first'),/first reported 0\.667s; previous safe boundary unavailable.*Running averages can recover/);
  assert.doesNotMatch(text(ui,'recording-diagnosis-human-first'),/0\.000–/);
}

async function resetFailureRetainsLastValidGeometryClock() {
  const value=summary(),human=value.timeline.episodes[1].human_tracking;
  const first=crossing('max_floor_penetration_model_units',{value:.04,previous_value:null,previous_safe_elapsed_s:null,
    status:'failed_at_reset',event_index:9,control_step:0,first_failed_elapsed_s:0,reference_clock_s:2,
    engine_state_status:'last_valid_control_boundary',last_valid_geometry_elapsed_s:0});
  human.first_crossings=[first];human.first_failure={...copy(first),episode_index:1,reset_event_index:9,crossings:[copy(first)]};
  const ui=await inspect(value);
  assert.equal(ui.ids.get('recording-diagnosis-content').hidden,false);
  assert.match(text(ui,'recording-diagnosis-human-first'),/first reported 0\.000s; previous safe boundary unavailable.*step 0, event 9/);
  assert.match(text(ui,'recording-diagnosis-human-clock'),/reference clock: 2\.000s.*last valid boundary at 0\.000s, not a terminal pose/);
}

async function recoveredRunningRmsDoesNotChangeRecordedTaskOutcome() {
  const value=summary(),episode=value.timeline.episodes[1],human=episode.human_tracking;
  const first=crossing('angular_rmse_rad',{value:.4,previous_value:.3});
  human.first_crossings=[first];human.first_failure={...copy(first),episode_index:1,reset_event_index:9,crossings:[copy(first)]};
  human.terminal.forEach(row=>{row.value=0;row.status='within_limit';});
  episode.terminal.completed_interval=false;episode.terminal.failed_gates=[];
  const ui=await inspect(value);
  assert.match(text(ui,'recording-diagnosis-human'),/within their limits at the end.*task result remains separate/);
  assert.match(text(ui,'recording-diagnosis-outcome'),/before a complete passing task interval/);
  assert.match(text(ui,'recording-diagnosis-human-first'),/Running averages can recover/);
}

async function legacySummariesLeaveBodyDiagnosisUnavailable() {
  const value=summary();delete value.timeline.equipment_stability_applicability;
  for(const episode of value.timeline.episodes)delete episode.human_tracking;
  const ui=await inspect(value);
  assert.equal(ui.ids.get('recording-diagnosis-content').hidden,false);
  assert.match(text(ui,'recording-diagnosis-human'),/unavailable in this older summary.*no body pass or failure timing is inferred/);
  assert.equal(rows(ui,'recording-diagnosis-human-metrics').length,0);
  assert.match(text(ui,'recording-diagnosis-first'),/No equipment-limit crossing.*does not mean the task passed/);
}

async function malformedHumanSummariesFailClosed() {
  const mutations=[
    h=>{h.terminal[0].metric='unsupported';}, h=>{h.terminal[1]=copy(h.terminal[0]);},
    h=>{h.terminal[0].aggregation='cumulative_peak';},h=>{h.terminal[0].status='unavailable';},
    h=>{h.first_crossings[0].previous_value=null;},h=>{h.first_crossings[0].first_failed_elapsed_s=-1;},
    h=>{h.first_failure=null;},h=>{h.first_failure.crossings[0].value=5;},
  ];
  for(const mutate of mutations){
    const value=summary();mutate(value.timeline.episodes[1].human_tracking);const ui=await inspect(value);
    assert.equal(ui.ids.get('recording-diagnosis-content').hidden,true);
    assert.match(text(ui,'recording-diagnosis-status'),/unavailable.*No time or success is inferred/);
  }
}

async function mixedEquipmentAndBodyFailuresShareOneEpisode() {
  const value=summary();value.timeline.equipment_stability_applicability='applied';
  const episode=value.timeline.episodes[1],equipment={object_id:'block',metric:'speed_model_units_s',threshold:.15,value:.2,
    previous_safe_elapsed_s:.1,first_failed_elapsed_s:.2,event_index:15,control_step:6,reference_clock_s:.2,
    boundary_contacts:[{objects:['right_elbow_geom','block'],human_geom:'right_elbow_geom',human_segment:'right_forearm'}]};
  episode.first_crossings=[equipment];episode.first_equipment_failure={...copy(equipment),crossings:[copy(equipment)]};
  const ui=await inspect(value);
  assert.equal(rows(ui,'recording-diagnosis-crossings').length,3);
  assert.match(text(ui,'recording-diagnosis-first'),/speed.*0\.200.*allowed 0\.150/);
  assert.match(text(ui,'recording-diagnosis-contacts'),/right forearm.*not proof of cause/);
  assert.equal(ui.ids.get('recording-diagnosis-contacts').hidden,false);
  assert.match(ui.ids.get('recording-diagnosis-links').children[0].textContent,/state 16 of 244 \(event 15\)/);
}

for(const check of [peakFailureAndRecoveredAverageRemainDistinct,missingPrecedingMeasurementDoesNotInventSafeTime,
  resetFailureRetainsLastValidGeometryClock,recoveredRunningRmsDoesNotChangeRecordedTaskOutcome,
  legacySummariesLeaveBodyDiagnosisUnavailable,malformedHumanSummariesFailClosed,mixedEquipmentAndBodyFailuresShareOneEpisode]) {
  await check();console.log(`PASS ${check.name}`);
}
