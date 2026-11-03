import test from 'node:test';
import assert from 'node:assert/strict';
import { sceneListRefreshDue } from '../studio.js';
test('50 seconds of active scene polling refreshes immutable library at most every15seconds', () => {
  let last=null, lists=0;
  for (let time=0;time<50000;time+=2500) if(sceneListRefreshDue(last,time)){last=time;lists++;}
  assert.equal(lists,4);
  assert.equal((20*2+lists)/50,.88); // Existing jobs list + selected scene remain live on every tick.
});
test('manual invalidation and clock rollback force refresh; nothing caches job authorization', () => {
  assert.equal(sceneListRefreshDue(null,1000),true);
  assert.equal(sceneListRefreshDue(1000,15999),false);
  assert.equal(sceneListRefreshDue(1000,16000),true);
  assert.equal(sceneListRefreshDue(1000,999),true);
});
