/** Analytical DOM/worker payloads only; no native, browser, network, media or source admission. */
import assert from 'node:assert/strict';
import {performance} from 'node:perf_hooks';
import {pathToFileURL} from 'node:url';
import {cameraFixture,cameraReceipt,liveFixture,replayFixture} from './camera_shared_ui_smoke.mjs';
import {drain} from './lab_ui_smoke.mjs';

const copy=value=>JSON.parse(JSON.stringify(value)),hash=char=>char.repeat(64);
function policyFixture(roots=false,zero=false){
  const data=cameraFixture(roots),identity=data.capabilities.task_contract.identity;
  const summary={selected_feedback_kind:zero?'authored_unchanged':'searched_phase_feedback_residuals',
    nonzero_coefficients:zero?0:2,root_assistance:data.task.control.root_assistance,learning_qualification:'unqualified_research'};
  const admission={schema_version:1,profile:'camera-shared-policy-admission-v1',status:'admitted',...identity,
    policy_sha256:hash('d'),training_manifest_sha256:hash('e'),evaluation_manifest_sha256:hash('f'),
    original_runtime_sha256:{'camera_shared_policy.py':hash('a'),'recording.py':hash('b')},
    execution_runtime_sha256:{'camera_shared_policy.py':hash('a'),'recording.py':hash('c')},
    runtime_versions:Object.fromEntries(['python','numpy','mujoco','gymnasium'].map(name=>[name,'analytical'])),
    compatibility:{profile:'camera-shared-policy-integration-compatibility-v1',
      allowed_changed_files:['camera_shared_learning.py','recording.py','shared_exercise_lab.py'],
      changed_files:{'recording.py':{original_sha256:hash('b'),current_sha256:hash('c')}},verifier_delta:null},
    compatibility_sha256:hash('a'),summary,privileged_simulator_features:true,
    scope:'External bounded-search controller with unchanged source/task qualification; exact actor/physics/scorer closure, explicitly versioned integration adapters; not physical validation.'};
  const view={profile:admission.profile,status:'admitted',...identity,admission_sha256:hash('b'),policy_sha256:admission.policy_sha256,
    training_manifest_sha256:admission.training_manifest_sha256,evaluation_manifest_sha256:admission.evaluation_manifest_sha256,
    compatibility_sha256:admission.compatibility_sha256,original_runtime_identity_sha256:hash('c'),execution_runtime_identity_sha256:hash('d'),
    privileged_simulator_features:true,summary};
  data.capabilities.controllers={...data.capabilities.controllers,learned_policy_available:true,trained_target_id:'human',camera_policy_admission:view};
  data.status_overlay={learned_policy_available:true,exercise_policy:{profile:'camera-shared-policy-v1',policy_sha256:admission.policy_sha256,
    admission_profile:admission.profile,admission_sha256:view.admission_sha256,selected_feedback_kind:summary.selected_feedback_kind,
    privileged_simulator_features:true,training_score:.4,training_score_improved_over_authored:!zero,evaluation_status:'evaluated',
    evaluation_successes:{coast:{successes:0,episodes:3},random:{successes:0,episodes:3},authored:{successes:2,episodes:3},learned:{successes:1,episodes:3}},
    evaluation_reset_protocol:{profile:'shared-exercise-human-jitter-v1',human_jitter_scale:.1,equipment_reset_state:'unchanged_scene_pose_and_velocity',
      object_position_jitter_world:0,object_velocity_jitter_world_s:0},root_assistance:summary.root_assistance,learning_qualification:'unqualified_research'}};
  Object.assign(data.replay_bundle,{camera_policy_admission:admission,policy_admission_sha256:view.admission_sha256,
    policy_sha256:admission.policy_sha256,camera_policy_summary:summary});
  const event=data.replay_bundle.events[1],count=roots?22:19,delta=zero?0:.05;
  Object.assign(event,{controller:'learned',action:[delta,...Array(count-1).fill(0)],policy_sha256:admission.policy_sha256,
    policy_admission_sha256:view.admission_sha256,policy_sample:{phase:0,authored_action:Array(count).fill(0),applied_joint_delta:[delta,...Array(18).fill(0)]}});
  event.snapshot.normalized_action=[...event.action,...(roots?[]:[0,0,0])];
  return copy(data);
}

async function admittedControlsAndRecordingStaySerialized(){
  for(const roots of [false,true]){const ui=liveFixture(policyFixture(roots));await ui.ready();
    assert.equal(ui.ids.get('policy').disabled,false);assert.equal(ui.ids.get('joint').children.length,roots?22:19);
    assert.match(ui.ids.get('exercise-policy-summary').textContent,/experimental trained controller.*19 joint.*simulated body/);
    assert.match(ui.ids.get('exercise-policy-summary').textContent,roots?/Body-position assistance remains programmed/:/Body-position assistance is disabled/);
    assert.match(ui.ids.get('exercise-evaluation-summary').textContent,/camera and ground priors remain fixed/);
    assert(ui.ids.get('scores').children.some(node=>node.children.some(child=>child.textContent==='1/3')));
    ui.ids.get('policy').click();ui.ids.get('step').click();await drain();
    assert.deepEqual(ui.pending[0].data,{steps:1,policy:true});
    ui.ids.get('record').click();await drain();assert.equal(ui.pending.length,1);
    await ui.reply('/api/step');assert.equal(ui.pending[0].path,'/api/record/start');await ui.reply('/api/record/start','initial');
    assert.equal(ui.pending.length,1);assert.deepEqual(ui.pending[0].data,{steps:3,policy:true});await ui.reply('/api/step');
    ui.ids.get('stop-record').click();await drain();await ui.reply('/api/record/stop');
    assert(!ui.requests.some(row=>/train|appearance|policy.json|api\/actor/.test(row.path)));
  }
  const zero=liveFixture(policyFixture(true,true));await zero.ready();
  assert.match(zero.ids.get('exercise-policy-summary').textContent,/selected unchanged programmed feedback; it did not learn new joint corrections/);
}

async function mismatchedPolicyDisablesLearnedAndMalformedAdmissionPauses(){
  for(const mutate of [d=>d.status_overlay.exercise_policy.policy_sha256=hash('0'),
    d=>d.status_overlay.exercise_policy.admission_sha256=hash('0'),d=>d.status_overlay.exercise_policy.profile='shared-exercise-policy-v4']){
    const data=policyFixture();mutate(data);const ui=liveFixture(data);await ui.ready();
    assert.equal(ui.ids.get('policy').disabled,true);assert.equal(ui.ids.get('step').disabled,false);
  }
  for(const mutate of [d=>d.capabilities.controllers.camera_policy_admission=null,
    d=>d.capabilities.controllers.camera_policy_admission.source_clock_sha256=hash('0'),
    d=>d.capabilities.controllers.camera_policy_admission.summary.root_assistance='explicit_3_axis',
    d=>d.scene.learned_policy_available=true]){
    const data=policyFixture();mutate(data);const ui=liveFixture(data);await ui.ready();assert.equal(ui.ids.get('step').disabled,true);
  }
}

async function policyReceiptRequiresDistinctProtocolAndBothIdentities(){
  const ui=liveFixture(policyFixture(true));await ui.ready();const value=cameraReceipt(ui.data),view=ui.data.capabilities.controllers.camera_policy_admission;
  Object.assign(value.identities,{policy_sha256:view.policy_sha256,policy_admission_sha256:view.admission_sha256});
  value.verified_metadata.push(...[['run/policy.json',view.policy_sha256],['run/camera-policy-admission.json',hash('9')]].map(([path,sha256])=>({source_name:path,path,sha256,bytes:100})));
  Object.assign(value.replay,{protocol:'camera-shared-policy-full-state-replay-v1',learned_actions_verified:1});
  const api=ui.context.feetTest;assert(api.recordingEvidenceValid(value,value.session));api.applyRecordingEvidence(value);
  assert.match(ui.ids.get('recording-evidence-replay').textContent,/Exact external policy decisions matched: 1.*root assistance stays authored/);
  for(const mutate of [v=>v.replay.protocol='camera-shared-full-state-replay-v1',v=>v.replay.learned_actions_verified=null,
    v=>v.replay.learned_actions_verified=3,v=>v.identities.policy_admission_sha256=hash('0'),v=>v.identities.policy_sha256=hash('0')]){
    const invalid=copy(value);mutate(invalid);assert.equal(api.recordingEvidenceValid(invalid,invalid.session),false);
  }
  value.replay.learned_actions_verified=0;assert(api.recordingEvidenceValid(value,value.session));
}

function offlinePolicyKeepsLearnedMetadataAndAuthoredRoots(){
  for(const roots of [false,true]){const replay=replayFixture(policyFixture(roots));assert.equal(replay.ids.get('play').disabled,false);replay.select(1);
    assert.equal(replay.ids.get('play').disabled,false);assert.match(replay.ids.get('controller').textContent,/experimental trained controller.*simulated body/);
    assert.match(replay.ids.get('controller').textContent,roots?/programmed, not learned/:/disabled/);
    assert.match(replay.ids.get('camera-status').textContent,/not action replay or per-frame cryptographic identity/);
    replay.select(3);assert.match(replay.ids.get('controller').textContent,/authored feedback/);
  }
  for(const mutate of [d=>d.replay_bundle.policy_admission_sha256=null,
    d=>d.replay_bundle.camera_policy_admission.source_clock_sha256=hash('0'),
    d=>d.replay_bundle.camera_policy_admission.scope='physically qualified',
    d=>d.replay_bundle.camera_policy_admission.runtime_versions.numpy=true,
    d=>d.replay_bundle.events[1].policy_sha256=hash('0'),d=>d.replay_bundle.events[1].policy_sample.applied_joint_delta[0]=.3,
    d=>d.replay_bundle.events[1].action[21]=.1]){
    const data=policyFixture(true);mutate(data);const replay=replayFixture(data);replay.select(1);
    assert.equal(replay.ids.get('play').disabled,true);assert.match(replay.ids.get('status').textContent,/unavailable/);
  }
}

function offlineResidualRoundoffMatchesRecorderTolerance(){
  for(const [delta,accepted] of [[.2500000000000001,true],[.25+2e-12,false]]){
    const data=policyFixture(true),row=data.replay_bundle.events[1];
    row.action[0]=.25;row.snapshot.normalized_action[0]=.25;
    row.policy_sample.authored_action[0]=.25-delta;row.policy_sample.applied_joint_delta[0]=delta;
    const replay=replayFixture(data);replay.select(1);assert.equal(replay.ids.get('play').disabled,!accepted);
  }
}

async function repeatedAdmittedUpdatesDoNotRebuildUnchangedLabels(){
  const ui=liveFixture(policyFixture(true));await ui.ready();ui.update('active');
  const output=ui.ids.get('exercise-policy-summary');let liveWrites=0,value=output.textContent;
  Object.defineProperty(output,'textContent',{get:()=>value,set:next=>{liveWrites++;value=next;}});
  const start=performance.now();for(let i=0;i<200;i++)ui.update('active');const liveMs=performance.now()-start;
  assert.equal(liveWrites,0);assert.equal(ui.pending.length,0);
  const replay=replayFixture(ui.data);replay.select(1);const status=replay.ids.get('camera-status');let replayWrites=0,cached=status.textContent;
  Object.defineProperty(status,'textContent',{get:()=>cached,set:next=>{replayWrites++;cached=next;}});
  const replayStart=performance.now();for(let i=0;i<200;i++)replay.turn();
  assert.equal(replayWrites,0);
  console.log(JSON.stringify({measurement:'200_unchanged_admitted_camera_scripted_updates',fixture:'tiny_analytical',
    live_ms:liveMs,replay_ms:performance.now()-replayStart,live_label_writes:liveWrites,replay_label_writes:replayWrites,scope:'Mock DOM/worker only, not browser performance'}));
}

const tests={admittedControlsAndRecordingStaySerialized,mismatchedPolicyDisablesLearnedAndMalformedAdmissionPauses,
  policyReceiptRequiresDistinctProtocolAndBothIdentities,offlinePolicyKeepsLearnedMetadataAndAuthoredRoots,
  offlineResidualRoundoffMatchesRecorderTolerance,repeatedAdmittedUpdatesDoNotRebuildUnchangedLabels};
export {policyFixture};
if(import.meta.url===pathToFileURL(process.argv[1]||'').href){
  const names=process.argv.slice(2),selected=names.length?names:Object.keys(tests);let failed=0;
  for(const name of selected){try{assert(tests[name],name);await tests[name]();console.log(`PASS ${name}`);}catch(error){failed++;console.error(`FAIL ${name}`,error);}}
  console.log(JSON.stringify({passed:selected.length-failed,failed,scope:'Analytical external camera-policy UI only'}));process.exitCode=failed?1:0;
}
