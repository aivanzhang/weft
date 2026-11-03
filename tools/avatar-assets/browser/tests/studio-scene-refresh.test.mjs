import test from 'node:test';
import assert from 'node:assert/strict';
import { selectedJobSnapshot } from '../studio.js';
const id = 'a'.repeat(24), other = 'b'.repeat(24), sceneId = 'c'.repeat(24);
const job = { id, status: 'running', scene: { id: sceneId } };
const scene = { id: sceneId, assets: [{ id: 'first', job }, { id: 'second', job: { id: other, status: 'created', scene: { id: sceneId } } }] };

test('twenty known-scene refreshes use twenty scene reads and zero duplicate selected-job reads', async () => {
  const reads = [], read = async path => { reads.push(path); return structuredClone(scene); };
  for (let i=0;i<20;i++) {
    const snapshot = await selectedJobSnapshot(id,job,read);
    assert.equal(snapshot.job.id,id);assert.equal(snapshot.scene.assets[1].job.id,other);
  }
  assert.equal(reads.length,20);assert.ok(reads.every(path => path === `/api/studio/scenes/${sceneId}`));
});
test('initial discovery uses job then scene; ordinary single job uses exactly one read', async () => {
  const reads = [];
  await selectedJobSnapshot(id,null,async path => { reads.push(path); return path.includes('/scenes/') ? scene : job; });
  assert.deepEqual(reads,[`/api/studio/jobs/${id}`,`/api/studio/scenes/${sceneId}`]);
  reads.length=0;
  const plain = { id, status: 'created' };
  assert.deepEqual(await selectedJobSnapshot(id,plain,async path => { reads.push(path); return plain; }),{job:plain,scene:null});
  assert.deepEqual(reads,[`/api/studio/jobs/${id}`]);
});
test('missing or changed scene binding fails without extra fetch or silent fallback', async () => {
  for (const response of [{...scene,id:other},{...scene,assets:[]},{...scene,assets:[{job:{...job,scene:{id:other}}}]}]) {
    const reads=[];
    await assert.rejects(selectedJobSnapshot(id,job,async path => { reads.push(path);return response; }),/bound scene/);
    assert.equal(reads.length,1);
  }
  await assert.rejects(selectedJobSnapshot(id,null,async () => ({id:other,status:'created'})),/snapshot changed/);
});
