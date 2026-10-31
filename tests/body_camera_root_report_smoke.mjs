// Scripted DOM only: no browser, source media or runtime controls.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {performance} from 'node:perf_hooks';
import {createHash} from 'node:crypto';

const file = process.argv[2];
assert(file && fs.statSync(file).size <= 8 * 1024 * 1024);
const htmlBytes = fs.readFileSync(file);
const html = htmlBytes.toString('utf8');
const scripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];
assert.equal(scripts.length, 2);
const data = JSON.parse(scripts[0][1]);
let fetches = 0, timers = 0, created = 0;
class Element {
  constructor(tag='div') { this.tagName=tag;this.children=[];this.attributes={};this.listeners=new Map();this.value='';this.ownText='';this.disabled=false;created++; }
  set textContent(value) { this.ownText=String(value);this.children=[]; }
  get textContent() { return this.ownText+this.children.map(child=>child.textContent).join(''); }
  set innerHTML(_) { assert.fail('Untrusted metadata entered innerHTML'); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.ownText='';this.children=[...nodes]; }
  setAttribute(key,value) { this.attributes[key]=String(value); }
  addEventListener(kind,handler) { this.listeners.set(kind,handler); }
  fire(kind) { this.listeners.get(kind)?.({target:this}); }
  click() { if(!this.disabled)this.fire('click'); }
}
const elements=new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(match=>[match[1],new Element()]));
const el=id=>{assert(elements.has(id),id);return elements.get(id);};
el('data').textContent=scripts[0][1];
const document={getElementById:el,createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag)};
const sandbox={document,fetch:()=>{fetches++;assert.fail('Network request');},setTimeout:()=>{timers++;assert.fail('Timer');},setInterval:()=>{timers++;assert.fail('Poll');}};
const start=performance.now();
vm.runInNewContext(scripts[1][1],sandbox,{timeout:1000});
const count=node=>1+node.children.reduce((sum,child)=>sum+count(child),0);
const live=()=>[...elements.values()].reduce((sum,node)=>sum+count(node),0);
assert.equal(el('sample').max,String(data.samples.length-1));
assert.equal(el('points').children.length,data.samples[0].points.length);
assert(el('projection').children.length<=36);
assert.equal(el('previous').disabled,true);
assert.match(el('identity').textContent,/Source knot 1\//);
el('next').click();
assert.equal(el('sample').value,'1');
const fmt=value=>value===null?'Unavailable':Number(value).toFixed(5);
const vector=value=>value.map(fmt).join(', ');
const select=index=>{el('sample-number').value=String(index+1);el('sample-number').fire('change');};
const checkRow=index=>{
  select(index);const row=data.samples[index];
  assert.equal(el('points').children.length,row.points.length);
  assert.equal(el('origin').textContent,row.origin_label+' · '+row.landmark_source);
  assert(el('roots').textContent.includes(`Original [${vector(row.root_before)}] → proposed [${vector(row.root_after)}]`));
  for(let j=0;j<row.points.length;j++){
    const point=row.points[j],cells=el('points').children[j].children.map(cell=>cell.textContent);
    assert.deepEqual(cells,[point.name,point.status,fmt(point.weight),point.observed===null?'Unavailable':vector(point.observed),fmt(point.error_before),fmt(point.error_after)]);
  }
  const visible=row.points.flatMap(point=>[point.observed,point.before,point.after]).filter(point=>point!==null&&point[0]>=0&&point[0]<=data.image.width&&point[1]>=0&&point[1]<=data.image.height).length;
  assert.equal(el('projection').children.length,visible);
};
const missingDecoder=data.samples.findIndex(row=>row.decoder_timestamp_s===null);
if(missingDecoder>=0){checkRow(missingDecoder);assert.match(el('identity').textContent,/decoder Unavailable/);}
const unknown=data.samples.findIndex(row=>row.points.some(point=>point.weight===0));
if(unknown>=0)checkRow(unknown);
const known=data.samples.findIndex(row=>row.points.some(point=>point.weight>0));
if(known>=0)checkRow(known);
checkRow(data.samples.length-1);
const baseline=live();
for(let i=0;i<50;i++)el('sample').fire('input');
assert.equal(live(),baseline);
el('sample-number').value=String(data.samples.length);el('sample-number').fire('change');
assert.equal(el('next').disabled,true);assert.equal(el('sample').value,String(data.samples.length-1));
el('sample-number').value='-100';el('sample-number').fire('change');assert.equal(el('sample').value,'0');
el('sample-number').value='NaN';el('sample-number').fire('change');assert.equal(el('sample').value,'0');
assert.equal(fetches,0);assert.equal(timers,0);
assert(!/\b(?:fetch|XMLHttpRequest|WebSocket|setInterval|requestAnimationFrame)\s*\(/.test(scripts[1][1]));
assert(live()<240,'Only one source knot may have rendered detail');
process.stdout.write(JSON.stringify({checks:6,fetches,timers,live_nodes:live(),created_nodes:created,
  html_sha256:createHash('sha256').update(htmlBytes).digest('hex'),html_bytes:htmlBytes.length,
  source_knots:data.samples.length,known_endpoints:data.samples.reduce((sum,row)=>sum+row.points.filter(point=>point.weight>0).length,0),
  unknown_endpoints:data.samples.reduce((sum,row)=>sum+row.points.filter(point=>point.weight===0).length,0),
  missing_decoder_rows:data.samples.filter(row=>row.decoder_timestamp_s===null).length,
  exercised_missing_decoder:missingDecoder>=0,exercised_unknown_endpoint:unknown>=0,exercised_known_endpoint:known>=0,
  seconds:(performance.now()-start)/1000,measurement:'Scripted DOM only; not browser responsiveness'})+'\n');
