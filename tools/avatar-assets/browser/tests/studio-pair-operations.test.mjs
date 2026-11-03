import test from 'node:test';
import assert from 'node:assert/strict';
import { admitPairMutation } from './studio-pair-operations.mjs';
const ids=['a'.repeat(24),'b'.repeat(24)];
test('explicit bootstrap permits one save plus one Plan per exact newly saved child',()=>{
  const seen=new Set();assert.equal(admitPairMutation('POST','/api/studio/scenes',[],true,seen),'create');
  for(const id of ids)assert.equal(admitPairMutation('POST',`/api/studio/jobs/${id}/plan`,ids,true,seen),'plan');
  assert.equal(seen.size,3);
  for(const pathname of seen)assert.throws(()=>admitPairMutation('POST',pathname,ids,true,seen),/duplicate/);
});
test('save-only forbids Plan; both modes forbid Start, retry, cancel and unrelated IDs',()=>{
  for(const mode of [false,true])for(const operation of ['start','retry','cancel'])
    assert.throws(()=>admitPairMutation('POST',`/api/studio/jobs/${ids[0]}/${operation}`,ids,mode,new Set()));
  assert.throws(()=>admitPairMutation('POST',`/api/studio/jobs/${ids[0]}/plan`,ids,false,new Set()));
  assert.throws(()=>admitPairMutation('POST',`/api/studio/jobs/${'c'.repeat(24)}/plan`,ids,true,new Set()));
  assert.throws(()=>admitPairMutation('DELETE','/api/studio/scenes',ids,true,new Set()));
});
