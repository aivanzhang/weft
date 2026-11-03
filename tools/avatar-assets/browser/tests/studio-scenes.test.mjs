import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { explicitSceneRequest } from '../studio.js';
const one = { name: 'Otter', prompt: 'One river otter', style: 'Soft toy', custom_actions: ['Roll over'], preset: 'preview', reference_id: 'a'.repeat(32) };
const two = { name: 'Heron', prompt: 'One blue heron', style: 'Painted wood', custom_actions: ['Spread wings'], preset: 'draft' };
test('two explicit subjects keep distinct actions, styles and optional references', () => {
  const scene = explicitSceneRequest('Animal pair', one, two);
  assert.deepEqual(scene.assets.map(a => [a.id,a.name,a.custom_actions]), [['asset-1','Otter',['Roll over']],['asset-2','Heron',['Spread wings']]]);
  assert.equal(scene.assets[0].reference_id, one.reference_id);assert.equal(scene.assets[1].reference_id,undefined);
  assert.deepEqual(explicitSceneRequest('Pair',{...one,budget:99,source:'/tmp/fused.glb'},two).assets[0],scene.assets[0]);
  assert.equal(one.id,undefined);
});
test('missing, duplicate and oversized explicit subjects are rejected', () => {
  for (const bad of [{...two,name:'otter'},{...two,prompt:''},{...two,name:'x'.repeat(121)},{...two,custom_actions:['x'.repeat(601)]}]) assert.throws(() => explicitSceneRequest('Pair',one,bad));
  assert.throws(() => explicitSceneRequest('',one,two));
});
test('UI discloses separate jobs and nonautomatic decomposition, not split-mesh success', async () => {
  const html = await readFile(new URL('../studio.html',import.meta.url),'utf8');
  for (const id of ['separate-assets','second-prompt','second-actions','second-reference','scene-panel','scene-assets']) assert.match(html,new RegExp(`id="${id}"`));
  assert.match(html,/Automatic free-text splitting and synchronized interactions are not available/);
  assert.match(html,/saving starts no paid work/);
  assert.match(html,/no fused model is split/);
});
test('row cache excludes ticking elapsed time and missing children have explicit local recovery', async () => {
  const source = await readFile(new URL('../studio.js',import.meta.url),'utf8');
  const render = source.slice(source.indexOf('  function renderScene('), source.indexOf('  async function loadScene('));
  assert.match(render,/signature === state.sceneSignature/);
  assert.doesNotMatch(render,/elapsed_seconds/);
  assert.match(render,/Finish saving pair/);assert.match(render,/\/finish-save/);
  assert.doesNotMatch(render,/\/start|\/plan/);
});
