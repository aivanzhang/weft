import test from 'node:test';
import assert from 'node:assert/strict';
import { selectedResponse, observedBackendPath } from './studio-live-response.mjs';
const id='a'.repeat(24), other='b'.repeat(24), sceneID='c'.repeat(24);
const binding={id:sceneID,title:'Pair',asset_id:'asset-1',asset_name:'Owl'};
const job={id,status:'queued',scene:binding};
const scene={id:sceneID,assets:[{id:'asset-1',name:'Owl',job:{...job,status:'running'}},{id:'asset-2',name:'Rabbit',job:{id:other}}]};
test('direct child establishes membership; scene polling advances queued snapshot to running',()=>{
  const initial=selectedResponse(`/api/studio/jobs/${id}/start`,'POST',job,id);
  const current=selectedResponse(`/api/studio/scenes/${sceneID}`,'GET',scene,id,initial.binding);
  assert.equal(current.value.status,'running');assert.equal(current.value.id,id);assert.deepEqual(current.binding,binding);
});
test('unknown scene, list, and unsupported mutation routes never supply a selected snapshot',()=>{
  for(const [pathname,method,bound] of [[`/api/studio/scenes/${sceneID}`,'GET',null],['/api/studio/scenes','GET',binding],
    [`/api/studio/scenes/${other}`,'GET',binding],[`/api/studio/scenes/${sceneID}`,'POST',binding],
    [`/api/studio/jobs/${id}/plan`,'POST',binding]])assert.equal(selectedResponse(pathname,method,scene,id,bound),null);
});
test('wrong scene identity, duplicate child and membership tampering reject',()=>{
  for(const mutate of [s=>s.id=other,s=>s.assets.push(s.assets[0]),s=>s.assets[1]=structuredClone(s.assets[0]),
    s=>s.assets[0].id='foreign',s=>s.assets[0].name='Rabbit',s=>s.assets[0].job.scene.id=other,
    s=>s.assets[0].job.id=other]) {
    const altered=structuredClone(scene);mutate(altered);
    assert.throws(()=>selectedResponse(`/api/studio/scenes/${sceneID}`,'GET',altered,id,binding));
  }
  assert.throws(()=>selectedResponse(`/api/studio/jobs/${id}`,'GET',{...job,scene:null},id,binding));
});
test('single jobs remain supported and polling counters include scene reads but no file requests',()=>{
  assert.deepEqual(selectedResponse(`/api/studio/jobs/${id}`,'GET',{id,status:'running'},id),{value:{id,status:'running'},binding:null});
  for(const p of ['/api/studio/session','/api/studio/jobs',`/api/studio/jobs/${id}`,'/api/studio/scenes',`/api/studio/scenes/${sceneID}`])assert.equal(observedBackendPath(p),true);
  for(const p of [`/api/studio/jobs/${id}/files/anything`,`/api/studio/jobs/${id}/start`,'/other'])assert.equal(observedBackendPath(p),false);
});
