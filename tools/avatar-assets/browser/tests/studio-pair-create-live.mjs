// Save two real UI cards; optional unfunded Prepare stops before paid planning. Never grant, Start or retry.
import assert from 'node:assert/strict';
import { spawn, execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { createHash } from 'node:crypto';
import { lstat, mkdir, readFile, realpath, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { chromium } from 'playwright-core';
import { explicitSceneRequest } from '../studio.js';
import { observePreparation } from './studio-prepare-observation.mjs';
import { admitPairMutation } from './studio-pair-operations.mjs';

const root=fileURLToPath(new URL('../../',import.meta.url));
const [folder,inputArg,anchor,...extra]=process.argv.slice(2);
const prepareUnfunded=extra.length===1 && extra[0]==='--prepare-unfunded';
assert(folder && inputArg && /^[a-f0-9]{24}$/.test(anchor) && (!extra.length || prepareUnfunded),
  'Usage: node browser/tests/studio-pair-create-live.mjs outputs/NEW examples/requests/PAIR.json COMPLETED_ANCHOR_JOB [--prepare-unfunded]');
const input=path.resolve(root,inputArg), info=await lstat(input);
assert.equal(await realpath(input),input);assert(info.isFile() && !info.isSymbolicLink() && info.size<=65536);
assert(['examples','outputs'].some(base=>input.startsWith(path.join(root,base)+path.sep)));
const inputBytes=await readFile(input), config=JSON.parse(inputBytes);
assert.equal(config.assets?.length,2);assert(!config.relationships?.length,'This harness creates independent assets, not interactions');
assert(config.assets.every(a=>!a.reference_id && !a.source && !a.placement),'This initial live harness is text-only and fresh-source');
assert.equal(config.assets[0].preset,config.assets[1].preset,'The UI currently uses one shared output preset');
const body=explicitSceneRequest(config.title,...config.assets);
const out=path.resolve(root,folder);assert.equal(await realpath(path.dirname(out)),await realpath(path.join(root,'outputs')));
await mkdir(out,{mode:0o700});
const save=async(name,value)=>writeFile(path.join(out,name),JSON.stringify(value,null,2)+'\n',{flag:'wx',mode:0o600});
const digest=bytes=>createHash('sha256').update(bytes).digest('hex');
const environment=Object.fromEntries(['HOME','USER','LOGNAME','TMPDIR','PATH','LANG'].filter(k=>process.env[k]).map(k=>[k,process.env[k]]));
Object.assign(environment,{PYTHONUNBUFFERED:'1',PYTHONDONTWRITEBYTECODE:'1'});
const report={schema_version:1,status:'running',classification:prepareUnfunded?'actual_website_pair_save_plus_unfunded_preparation':'actual_website_explicit_pair_save_only',
  input:{path:path.relative(root,input),sha256:digest(inputBytes)},anchor_job:anchor,mutations:[],errors:[],cleanup:{},
  paid_work_started:false,paid_planning_started:false,unfunded_preparation_requested:prepareUnfunded,source_generation_started:false};
const started=performance.now(), pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
let server,browser,origin,timer,requests=0,logBytes=0;
const seenMutations=new Set();
const exited=child=>child.exitCode!==null || child.signalCode!==null;
try {
  await save('explicit-cards.json',body);
  report.sources=[];
  for(const name of ['browser/tests/studio-pair-create-live.mjs','browser/tests/studio-pair-operations.mjs','browser/tests/studio-prepare-observation.mjs','browser/studio.js','browser/studio.html','scripts/verify_studio_live.py','avatar_assets/studio.py','avatar_assets/studio_jobs.py','avatar_assets/scene_requests.py'])
    report.sources.push({path:name,sha256:digest(await readFile(path.join(root,name)))});
  server=spawn(path.join(root,'.venv/bin/python'),['-B','scripts/verify_studio_live.py','--job',anchor,'--serve','--max-seconds','150'],
    {cwd:root,env:environment,stdio:['ignore','pipe','pipe']});report.server_pid=server.pid;
  origin=await new Promise((resolve,reject)=>{
    let pending='';const timeout=setTimeout(()=>reject(new Error('Server readiness timed out')),15000);
    server.stdout.on('data',chunk=>{pending+=chunk.toString();if(pending.length>4096)return reject(new Error('Server readiness bound exceeded'));
      if(!pending.includes('\n'))return;clearTimeout(timeout);
      try{const ready=JSON.parse(pending.split('\n')[0]);assert.equal(ready.job_id,anchor);assert(Number.isInteger(ready.port)&&ready.port>0&&ready.port<=65535);resolve(`http://127.0.0.1:${ready.port}`);}catch(error){reject(error);}});
    server.once('error',reject);server.once('exit',()=>{clearTimeout(timeout);reject(new Error('Server exited before readiness'));});
    server.stderr.on('data',chunk=>{logBytes+=chunk.length;if(logBytes>1048576)server.kill('SIGTERM');});
  });
  browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,
    timeout:20000,env:environment,args:['--disable-background-networking','--disable-sync','--no-first-run']});
  const context=await browser.newContext({viewport:{width:1280,height:1000}});
  await context.route('**/*',async route=>{
    const req=route.request(),url=new URL(req.url());
    if(++requests>200 || url.origin!==origin)return route.abort();
    if(req.method()!=='GET') {
      const operation=admitPairMutation(req.method(),url.pathname,(report.children||[]).map(c=>c.job_id),prepareUnfunded,seenMutations);
      assert.deepEqual(req.postDataJSON(),operation==='create'?body:{});assert(req.headers()['x-avatar-studio-token']);
      if(operation==='create')assert(req.headers()['x-avatar-request-id']);
      report.mutations.push({method:'POST',path:url.pathname,csrf_present:true,operation});
    }
    return route.continue();
  });
  const page=await context.newPage();page.setDefaultTimeout(15000);timer=setTimeout(()=>browser.close(),120000);
  await page.goto(origin,{waitUntil:'domcontentloaded'});
  await page.locator('#new-request').click();await page.locator('#separate-assets').check();
  await page.locator('#scene-title').fill(body.title);await page.locator('#asset-one-name').fill(body.assets[0].name);
  await page.locator('#prompt').fill(body.assets[0].prompt);await page.locator('#style').fill(body.assets[0].style);
  await page.locator('#actions').fill(body.assets[0].custom_actions.join('\n'));
  await page.locator('#preset').selectOption(body.assets[0].preset);
  await page.locator('#asset-two-name').fill(body.assets[1].name);await page.locator('#second-prompt').fill(body.assets[1].prompt);
  await page.locator('#second-style').fill(body.assets[1].style);await page.locator('#second-actions').fill(body.assets[1].custom_actions.join('\n'));
  const response=page.waitForResponse(value=>value.url()===origin+'/api/studio/scenes' && value.request().method()==='POST');
  await page.locator('#save-request').click();const savedResponse=await response;const saved=await savedResponse.json();
  assert(savedResponse.ok(),JSON.stringify(saved));assert.equal(saved.mode,'explicit_independent_assets');
  assert.equal(saved.assets.length,2);assert.equal(new Set(saved.assets.map(a=>a.job.id)).size,2);
  for(const [index,asset]of saved.assets.entries()) {
    assert.equal(asset.job.status,'created');assert.equal(asset.name,body.assets[index].name);
    for(const key of ['prompt','style','custom_actions','preset'])assert.deepEqual(asset.job.request[key],body.assets[index][key]);
    assert.equal(asset.job.request.reuse_model_selected,false);assert.equal(asset.job.request.reference_selected,false);
  }
  await save('created-scene.json',saved);report.scene_id=saved.id;
  report.children=saved.assets.map((a,i)=>({subject_id:config.assets[i].id,asset_id:a.id,name:a.name,job_id:a.job.id}));
  await page.waitForFunction(()=>!document.getElementById('scene-panel').hidden && document.querySelectorAll('#scene-assets button').length===2);
  assert.match(await page.locator('#scene-panel').innerText(),/Prepare and Start each asset separately/);
  for(let i=0;i<2;i++) {
    await page.locator('#scene-assets button').nth(i).click();
    await page.waitForFunction(prompt=>document.getElementById('request-prompt').textContent===prompt,body.assets[i].prompt);
    assert.equal(await page.locator('#job-status').innerText(),'Request saved');
  }
  await page.reload({waitUntil:'domcontentloaded'});
  await page.getByRole('button',{name:body.title+' Separate asset pair',exact:true}).click();
  await page.waitForFunction(()=>!document.getElementById('scene-panel').hidden);
  if(prepareUnfunded) {
    report.prepared=[];
    for(const [index,child]of report.children.entries()) {
      await page.locator('#scene-assets button').nth(index).click();
      await page.waitForFunction(prompt=>document.getElementById('request-prompt').textContent===prompt,body.assets[index].prompt);
      const prepared=await observePreparation({
        receivePost:async()=>{const response=await page.waitForResponse(value=>value.url()===`${origin}/api/studio/jobs/${child.job_id}/plan`&&value.request().method()==='POST');return{ok:response.ok(),status:response.status(),body:await response.json()};},
        click:()=>page.locator('#prepare-plan').click(),savePost:value=>save(`prepare-${index}-response.json`,value),pause,maxMs:45000,
        readJob:async()=>{const response=await context.request.get(`${origin}/api/studio/jobs/${child.job_id}`);assert(response.ok());const result=await response.json();return result.job??result;},
      });
      await save(`prepared-${index}.json`,prepared);assert.equal(prepared.status,'awaiting_authorization');assert.equal(prepared.plan_id,null);
      const {stdout}=await promisify(execFile)(path.join(root,'.venv/bin/python'),['-B','scripts/verify_studio_live.py','--job',child.job_id,'--allow-unplanned'],
        {cwd:root,env:environment,timeout:10000,maxBuffer:1048576});
      const proof=JSON.parse(stdout);assert.equal(proof.plan_available,false);assert.equal(proof.owner.active_lease_job,null);
      await save(`unfunded-${index}-proof.json`,proof);
      const input=proof.files.find(file=>file.path===`outputs/studio-${child.job_id}/request.json`);assert(input);
      report.prepared.push({job_id:child.job_id,request_sha256:input.sha256,plan_available:false,api_calls_verified_zero:true});
    }
  }
  const screenshot=await page.screenshot({fullPage:true});assert(screenshot.length<=8*1024**2);
  await writeFile(path.join(out,'saved-pair.png'),screenshot,{flag:'wx',mode:0o600});
  report.screenshot={path:'saved-pair.png',bytes:screenshot.length,sha256:digest(screenshot)};
  assert.equal(report.mutations.length,prepareUnfunded?3:1);report.status='passed';
} catch(error) {report.status='failed';report.errors.push(error.message);}
finally {
  clearTimeout(timer);if(browser)await browser.close();report.cleanup.browser_exit_verified=!browser||!browser.isConnected();
  if(server&&!exited(server))server.kill('SIGTERM');for(let i=0;server&&!exited(server)&&i<50;i++)await pause(100);
  report.cleanup.server_exit_verified=!server||exited(server);report.request_count=requests;
  report.elapsed_seconds=(performance.now()-started)/1000;await save('result.json',report);
  console.log(JSON.stringify({status:report.status,scene_id:report.scene_id,children:report.children,errors:report.errors,cleanup:report.cleanup}));
  process.exitCode=report.status==='passed'&&report.cleanup.server_exit_verified?0:1;
}
