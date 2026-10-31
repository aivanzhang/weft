/** Current presentation against untouched saved native payloads; no native/runtime imports. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {performance} from 'node:perf_hooks';
import {liveFixture,replayFixture} from './shared_exercise_feet_ui_smoke.mjs';
import {drain} from './lab_ui_smoke.mjs';

const started=performance.now(),root=path.resolve(process.argv[2]),cache=new Map(),evidence=[];
function read(name,limit,expected){
  const file=path.join(root,name);
  if(!cache.has(file)){
    const stat=fs.lstatSync(file);assert(stat.isFile()&&!stat.isSymbolicLink());assert(stat.size<=limit);
    const raw=fs.readFileSync(file);assert(raw.length<=limit);const sha256=createHash('sha256').update(raw).digest('hex');
    cache.set(file,{raw,sha256});evidence.push({path:name,bytes:raw.length,sha256});
  }
  const item=cache.get(file);assert(item.raw.length<=limit);if(expected)assert.equal(item.sha256,expected);
  return item.raw;
}
const json=(name,limit,expected)=>JSON.parse(read(name,limit,expected));
function savedCase(name){
  const acceptance=json(`${name}/acceptance.json`,64<<10),run=`${name}/lab/run`,session=`${name}/lab/sessions/${acceptance.receipt.session}`;
  assert.match(acceptance.receipt.session,/^session-[0-9a-f]{12}$/);
  const manifest=json(`${run}/manifest.json`,128<<10,acceptance.lab.manifest_sha256);
  const scene=json(`${run}/scene.json`,1<<20,manifest.files['scene.json']);
  const task=json(`${run}/exercise-task.json`,128<<10,manifest.files['exercise-task.json']);
  const seal=json(`${session}/manifest.json`,64<<10,acceptance.receipt.identities.recording_manifest_sha256);
  const events=read(`${session}/events.jsonl`,2<<20,seal.files['events.jsonl']).toString().trim().split('\n').map(line=>JSON.parse(line));
  const exported=read(`${session}/replay.html`,2<<20,seal.files['replay.html']).toString();
  const bundle=JSON.parse(exported.match(/<script id="data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
  assert.deepEqual(bundle.scene,scene);assert.deepEqual(bundle.exercise_task,task);assert.equal(bundle.events.length,events.length);
  for(const [index,event] of events.entries()){
    assert.equal(event.snapshot.scene_sha256,task.scene_sha256);assert.equal(event.snapshot.task_sha256,acceptance.lab.task_sha256);
    assert.equal(event.snapshot.engine_state.profile,'shared-world-v5');
    assert(!Object.hasOwn(event.snapshot.engine_state,'scene_profile'));assert(!Object.hasOwn(event.snapshot.engine_state,'scene_sha256'));
    assert.deepEqual(bundle.events[index].snapshot,event.snapshot.engine_state);
    assert.deepEqual(bundle.events[index].exercise_metrics,event.snapshot.task_metrics);
    assert.deepEqual(bundle.events[index].ankle_diagnostics,event.snapshot.ankle_diagnostics);
  }
  // Pure projection of the saved control contract, not a new environment or capability qualification.
  const assisted=task.control.root_assistance==='explicit_3_axis';
  const indices=[...Array.from({length:19},(_,i)=>i),22,23,24,25,26,27,...(assisted?[19,20,21]:[])];
  assert.equal(indices.length,events[0].snapshot.last_action.length);
  const action_contract={names:indices.map(i=>scene.action_contract.names[i]),units:indices.map(i=>scene.action_contract.units[i]),
    physical_limits:indices.map(i=>scene.action_contract.physical_limits[i]),normalized_bounds:[-1,1],engine_motor_indices:indices,
    joint_channels:Array.from({length:25},(_,i)=>i),body_joint_channels:Array.from({length:19},(_,i)=>i),ankle_channels:[19,20,21,22,23,24],
    root_assistance_channels:assisted?[25,26,27]:[],root_assistance:task.control.root_assistance,ankle_target_mode:task.control.ankle_target_mode,object_force_channels:[]};
  const identity={...Object.fromEntries(['scene_sha256','base_scene_sha256','base_task_sha256','projection_profile','projection_contract_sha256','adapter_implementation_sha256','foot_reference_sha256'].map(key=>[key,task[key]])),task_sha256:acceptance.lab.task_sha256};
  return {actual_saved:true,scene,task,action_contract,capabilities:{profile:'environment-capabilities-v1',task_contract:{identity}},
    states:{initial:events[0].snapshot,active:events.find(event=>event.type==='step').snapshot},events,replay_bundle:bundle};
}
const cases=['test_native_feet_candidate_lab0','test_native_feet_candidate_lab1'].map(savedCase),results=[];
for(const data of cases){
  for(const kind of ['live','replay']){
    const name=`saved_native_${kind}_${data.action_contract.names.length}_controls`;
    try{
      if(kind==='live'){
        const ui=liveFixture(data);await ui.ready();assert.equal(ui.ids.get('toggle').disabled,false,'Real initial state must enable controls');
        assert.equal(ui.ids.get('joint').children.length,data.action_contract.names.length);
        ui.motor(19);ui.action();await drain();assert.equal(ui.pending[0].data.action[19],.25);await ui.reply('/api/step');
        for(const event of data.events){ui.update(event.snapshot);assert.doesNotMatch(ui.ids.get('exercise-outcome').textContent,/unavailable/);}
        for(const corrupt of [value=>value.scene_sha256='0'.repeat(64),value=>value.base_task_sha256='0'.repeat(64),
          value=>value.engine_state.profile='shared-world-v4',value=>value.engine_state.human.feet.reference_sha256='0'.repeat(64)]){
          const changed=structuredClone(data.states.initial);corrupt(changed);ui.update(changed);
          assert.equal(ui.ids.get('toggle').disabled,true);assert.match(ui.ids.get('exercise-outcome').textContent,/unavailable/);
        }
        assert(!ui.requests.some(row=>/api\/(actor|appearance)/.test(row.path)));
      }else{
        const replay=replayFixture(data);
        for(let i=0;i<data.events.length;i++){replay.select(i);assert.doesNotMatch(replay.ids.get('status').textContent,/unavailable/);
          assert.match(replay.ids.get('feet-status').textContent,/Unscored simulated diagnostics/);}
        for(const corrupt of [bundle=>bundle.events[0].snapshot.profile='shared-world-v4',
          bundle=>bundle.events[0].snapshot.human.feet.reference_sha256='0'.repeat(64),
          bundle=>bundle.events[0].exercise_metrics.task_sha256='0'.repeat(64)]){
          const changed=structuredClone(data);corrupt(changed.replay_bundle);
          assert.match(replayFixture(changed).ids.get('status').textContent,/unavailable/);
        }
      }
      results.push({name,status:'passed'});console.log(`PASS ${name}`);
    }catch(error){results.push({name,status:'failed',error:String(error.stack)});console.error(`FAIL ${name}: ${error.stack}`);}
  }
}
const receipt={profile:'saved-native-feet-presentation-checks-v1',results,files:evidence,read_bytes:evidence.reduce((sum,row)=>sum+row.bytes,0),elapsed_ms:performance.now()-started,
  scope:'Saved native scene/task/journal and flattened replay payloads exercised with current scripted UI. Only capability identity/control mapping projected from saved metadata. No native execution, source/media decoding or browser. Saved exports are unchanged and contain the prior frozen UI.',native_runtime_label:'e5e9d55',read_cache:'Each selected file read and hashed once within this check'};
if(process.argv[3])fs.writeFileSync(process.argv[3],JSON.stringify(receipt,null,2)+'\n');
console.log(JSON.stringify({results:results.map(({name,status})=>({name,status})),read_bytes:receipt.read_bytes,elapsed_ms:receipt.elapsed_ms}));
if(results.some(row=>row.status==='failed'))process.exitCode=1;
