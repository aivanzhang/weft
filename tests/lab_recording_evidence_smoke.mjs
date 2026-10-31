/** Selected saved receipts only: scripted DOM, no browser, server, media or physics. */
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {performance} from 'node:perf_hooks';
import {Element, fixture, html, drain} from './lab_ui_smoke.mjs';

const copy = value => JSON.parse(JSON.stringify(value));
const sha = character => character.repeat(64);
const firstSession = 'session-123456789abc', secondSession = 'session-abcdef012345';
const endpoint = name => `/api/recording-evidence?session=${name}`;
const evidenceRequests = ui => ui.requests.filter(row => row.path.startsWith('/api/recording-evidence?'));
const descendants = element => [element, ...element.children.flatMap(descendants)];
const content = element => descendants(element).map(child => child.textContent).join('\n');
const evidenceButton = (ui, index = 0) => ui.ids.get('recordings').children[index].children.find(child => child.tagName === 'BUTTON' && child.textContent === 'Recording evidence');

function evidenceFixture(session = firstSession) {
  const identities = {source_run_manifest_sha256:sha('a'), recording_manifest_sha256:sha('b'),
    receipt_manifest_sha256:sha('c'), receipt_sha256:sha('d'), recording_scene_sha256:sha('e'),
    engine_scene_sha256:sha('f'), task_sha256:sha('0'), policy_sha256:sha('1')};
  const entries = [['run/manifest.json','a'], ['run/scene.json','2'], ['run/exercise-task.json','3'], ['run/policy.json','1'],
    [`sessions/${session}/manifest.json`,'b'], [`sessions/${session}/recording.json`,'4'],
    [`sessions/${session}/scene.json`,'5'], [`sessions/${session}/events.jsonl`,'6'],
    [`sessions/${session}/evidence/manifest.json`,'c'], [`sessions/${session}/evidence/receipt.json`,'d']];
  return {schema_version:1, profile:'recording-evidence-v1', status:'available', session,
    recorded:{events:7, actions:4, resets:2}, replay:{status:'passed', protocol:'shared-exercise-exact-action-state-task-v1',
      events_checked:7, actions_checked:4, resets_checked:2, learned_actions_verified:1}, identities,
    verified_metadata:entries.map(([path,digest]) => ({source_name:path,path,sha256:sha(digest),bytes:120})),
    partition:'exposed_development', physical_validation:'unverified', scope:'Saved verification; no new action replay or physical validation.'};
}

function recordingFixture(profile = 'shared-world', options = {}, {honorAbort = true, injectController = true} = {}) {
  const run = vm.runInNewContext, aborts = [];
  let ui;
  // Inject cancellation into this test only; the shared fixture stays unchanged.
  vm.runInNewContext = (source, sandbox, settings) => {
    if (injectController) sandbox.AbortController = class extends AbortController {
      abort() { if (!this.signal.aborted) aborts.push(this.signal); super.abort(); }
    };
    const fetch = sandbox.fetch;
    sandbox.fetch = (path, options) => {
      const pending = fetch(path, options);
      if (!path.startsWith('/api/recording-evidence?') || !options.signal) return pending;
      return new Promise((resolve, reject) => {
        const onAbort = () => {
          if (!honorAbort) return;
          const index = ui.pending.findIndex(row => row.options.signal === options.signal);
          if (index >= 0) ui.pending.splice(index, 1);
          reject(new Error('Aborted mock metadata request'));
        };
        options.signal.addEventListener('abort', onAbort, {once:true});
        pending.then(value => { options.signal.removeEventListener('abort', onAbort); resolve(value); }, reject);
      });
    };
    return run(source, sandbox, settings);
  };
  try {
    ui = fixture(profile, {sharedExercise:profile === 'shared-world', savedRecordings:[{name:firstSession,complete:true}], ...options});
  } finally { vm.runInNewContext = run; }
  return Object.assign(ui, {aborts});
}

async function resolveEvidence(ui, value, name = null, ok = true) {
  const index = ui.pending.findIndex(row => row.path.startsWith('/api/recording-evidence?') && (!name || row.path === endpoint(name)));
  assert(index >= 0, 'Expected a pending selected-recording metadata read');
  ui.pending.splice(index, 1)[0].resolve({ok,status:ok ? 200 : 500,json:async () => copy(value)});
  await drain();
}

async function explicitSelectionShowsAccessibleReceiptWithoutMovingControls() {
  const ui = recordingFixture(); await ui.ready();
  assert.equal(evidenceRequests(ui).length, 0);
  assert.equal(ui.ids.get('recording-evidence').hidden, true);
  const before = ui.backendTaskState(), sourceTime = ui.ids.get('source').currentTime, count = ui.requests.length;
  const button = evidenceButton(ui);
  assert.equal(button.getAttribute('aria-controls'), 'recording-evidence');
  assert.equal(button.getAttribute('aria-label'), 'Recording evidence for trial 1');
  button.click(); await drain(); button.click(); await drain();
  assert.equal(evidenceRequests(ui).length, 1, 'Repeated click during one read is deduplicated');
  assert.equal(ui.pending.length, 1);
  await resolveEvidence(ui, evidenceFixture());
  assert.equal(ui.requests.length, count + 1);
  assert.deepEqual(ui.backendTaskState(), before);
  assert.equal(ui.ids.get('source').currentTime, sourceTime);
  assert.equal(ui.ids.get('recording-evidence-content').hidden, false);
  assert.match(ui.ids.get('recording-evidence-counts').textContent, /4 actions · 2 resets · 7 total events, including the start event/);
  assert.match(ui.ids.get('recording-evidence-replay').textContent, /verified in the saved receipt.*4 actions, 2 resets and 7 events.*Learned actions verified: 1/);
  assert.match(ui.ids.get('recording-evidence-status').textContent, /Replay was not rerun/);
  assert.equal(ui.ids.get('recording-evidence-identities').children.length, 8);
  assert(ui.ids.get('recording-evidence-identities').children.every(row => row.children[0].getAttribute('scope') === 'row'));
  assert.match(html, /id="recording-evidence" class="task-results" aria-labelledby="recording-evidence-heading"/);
  assert.match(html, /id="recording-evidence-status" role="status"/);
  assert.match(html, /not task success or physical accuracy/);
  assert.match(html, /does not rerun actions or verify exported media/);
  assert.equal(ui.timers.size, 0);
}

async function spatialSavedStateExportKeepsUnavailableReplayCountsDistinctFromZero() {
  const ui = recordingFixture('spatial-rigid'); await ui.ready();
  const result = evidenceFixture();
  result.recorded = {events:1,actions:0,resets:0};
  result.replay = {status:'not_performed',protocol:null,events_checked:null,actions_checked:null,resets_checked:null,learned_actions_verified:null};
  result.identities.task_sha256 = result.identities.policy_sha256 = null;
  result.verified_metadata = result.verified_metadata.filter(row => !['run/exercise-task.json','run/policy.json'].includes(row.path));
  evidenceButton(ui).click(); await resolveEvidence(ui, result);
  assert.match(ui.ids.get('recording-evidence-counts').textContent, /0 actions · 0 resets · 1 total events/);
  assert.match(ui.ids.get('recording-evidence-replay').textContent, /not performed.*unavailable, not zero.*saved-state export is not an exact/);
  assert.equal(ui.ids.get('recording-evidence-identities').children.filter(row => row.children[1].textContent === 'Unavailable — not recorded').length, 2);
  assert.equal(evidenceRequests(ui).length, 1);
  evidenceButton(ui).click(); await resolveEvidence(ui, evidenceFixture());
  assert.equal(ui.ids.get('recording-evidence-content').hidden, true, 'A spatial lab cannot present the shared-exercise replay protocol as its verification');
  assert.match(ui.ids.get('recording-evidence-status').textContent, /unavailable.*not verified/);
}

async function newSpatialReceiptShowsExactActionsWithoutClaimingPolicyDecisionChecks() {
  const ui = recordingFixture('spatial-rigid'); await ui.ready();
  const result = evidenceFixture();
  result.replay.protocol = 'spatial-rigid-exact-action-state-v1';
  result.replay.learned_actions_verified = null;
  result.identities.task_sha256 = result.identities.policy_sha256 = null;
  result.verified_metadata = result.verified_metadata.filter(row => !['run/exercise-task.json','run/policy.json'].includes(row.path));
  const before = ui.backendState();
  evidenceButton(ui).click(); await resolveEvidence(ui, result);
  assert.equal(ui.ids.get('recording-evidence-content').hidden, false);
  assert.match(ui.ids.get('recording-evidence-replay').textContent, /verified in the saved receipt.*4 actions, 2 resets and 7 events/);
  assert.match(ui.ids.get('recording-evidence-replay').textContent, /learned-policy decisions were not re-evaluated/);
  assert.deepEqual(ui.backendState(), before);
  for (const change of [value => value.replay.learned_actions_verified = 1,
    value => value.identities.task_sha256 = sha('0'), value => value.replay.actions_checked--]) {
    const bad = copy(result); change(bad);
    evidenceButton(ui).click(); await resolveEvidence(ui, bad);
    assert.equal(ui.ids.get('recording-evidence-content').hidden, true);
  }
  const exercise = recordingFixture(); await exercise.ready();
  evidenceButton(exercise).click(); await resolveEvidence(exercise, result);
  assert.equal(exercise.ids.get('recording-evidence-content').hidden, true);
}

async function missingLegacyReceiptAndUnknownFieldsNeverBecomeVerifiedReplay() {
  const ui = recordingFixture(); await ui.ready(); evidenceButton(ui).click();
  await resolveEvidence(ui, {status:'unavailable',session:firstSession,reason:'Legacy export has no receipt'});
  assert.match(ui.ids.get('recording-evidence-status').textContent, /unavailable: Legacy export has no receipt.*not verified/);
  assert.equal(ui.ids.get('recording-evidence-content').hidden, true);
  assert.equal(ui.ids.get('recording-evidence-retry').hidden, false);
  const mutations = [value => delete value.recorded.actions, value => value.recorded.actions = -1,
    value => value.recorded.events++, value => value.replay.actions_checked--,
    value => value.replay.protocol = 'saved-state-export-v1', value => value.profile = 'unknown-v2',
    value => value.physical_validation = 'verified', value => value.identities.task_sha256 = null,
    value => value.identities.policy_sha256 = null,
    value => {value.replay = {status:'not_performed',protocol:null,events_checked:0,actions_checked:0,resets_checked:0,learned_actions_verified:0};}];
  for (const change of mutations) {
    const result = evidenceFixture(); change(result);
    ui.ids.get('recording-evidence-retry').click(); await resolveEvidence(ui, result);
    assert.equal(ui.ids.get('recording-evidence-content').hidden, true);
    assert.match(ui.ids.get('recording-evidence-status').textContent, /unavailable.*not verified/);
    assert.equal(ui.ids.get('recording-evidence-links').children.length, 0);
  }
}

function spatialPolicyEvidenceFixture(learned = 1) {
  const result = evidenceFixture();
  result.replay.protocol = 'spatial-rigid-exact-action-state-policy-v2';
  result.replay.learned_actions_verified = learned;
  result.identities.task_sha256 = null;
  if (learned === null) result.identities.policy_sha256 = null;
  result.verified_metadata = result.verified_metadata.filter(row => row.path !== 'run/exercise-task.json'
    && (learned !== null || row.path !== 'run/policy.json'));
  return result;
}

async function spatialPolicyReceiptsSeparateDecisionsFromManualAndLegacyActions() {
  for (const learned of [1, 0, null]) {
    const ui = recordingFixture('spatial-rigid'); await ui.ready();
    evidenceButton(ui).click(); await resolveEvidence(ui, spatialPolicyEvidenceFixture(learned));
    assert.equal(ui.ids.get('recording-evidence-content').hidden, false);
    const text = ui.ids.get('recording-evidence-replay').textContent;
    assert.match(text, /Exact action replay: verified/);
    if (learned === null) assert.match(text, /No learned policy was bound/);
    else assert.match(text, new RegExp(`Learned decisions matched the recorded policy: ${learned}\\. Manual actions are not policy-checked`));
  }
  const exercise = recordingFixture(); await exercise.ready();
  evidenceButton(exercise).click(); await resolveEvidence(exercise, spatialPolicyEvidenceFixture());
  assert.equal(exercise.ids.get('recording-evidence-content').hidden, true);
}

async function spatialPolicyReceiptsRejectMissingOrContradictoryDecisionIdentities() {
  const mutations = [value => value.identities.policy_sha256 = null,
    value => value.replay.learned_actions_verified = null,
    value => value.replay.learned_actions_verified = 5,
    value => value.replay.learned_actions_verified = -1,
    value => value.replay.learned_actions_verified = 1.5,
    value => value.identities.task_sha256 = sha('0'),
    value => value.verified_metadata.find(row => row.path === 'run/policy.json').sha256 = sha('2')];
  for (const mutate of mutations) {
    const ui = recordingFixture('spatial-rigid'); await ui.ready();
    const value = spatialPolicyEvidenceFixture(); mutate(value);
    evidenceButton(ui).click(); await resolveEvidence(ui, value);
    assert.equal(ui.ids.get('recording-evidence-content').hidden, true);
    assert.match(ui.ids.get('recording-evidence-status').textContent, /unavailable.*not verified/);
  }
}

async function exactLocalLinksBindSessionAndByteHashesWithoutConfusingCanonicalHashes() {
  const ui = recordingFixture(); await ui.ready(); const result = evidenceFixture();
  result.verified_metadata.push({source_name:'https://example.test/receipt.json',path:'https://example.test/receipt.json',sha256:sha('d'),bytes:1});
  evidenceButton(ui).click(); await resolveEvidence(ui, result);
  assert.equal(ui.ids.get('recording-evidence-content').hidden, false, 'Canonical JSON hashes deliberately differ from file-byte hashes');
  assert.match(content(ui.ids.get('recording-evidence-identities')), /canonical JSON/);
  const links = ui.ids.get('recording-evidence-links').children;
  assert.deepEqual(links.map(link => link.href), ['receipt.json','manifest.json'].map(file => `/files/sessions/${firstSession}/evidence/${file}?token=synthetic-token`));
  assert(links.every(link => link.download.endsWith('.json')));
  const changes = [value => value.session = secondSession,
    value => value.identities.receipt_sha256 = sha('0'), value => value.identities.source_run_manifest_sha256 = sha('0'),
    value => value.identities.policy_sha256 = sha('0'), value => value.identities.engine_scene_sha256 = 'not-a-hash',
    value => value.verified_metadata.at(-1).path = 'https://example.test/receipt.json',
    value => value.verified_metadata.at(-1).path = `sessions/${secondSession}/evidence/receipt.json`,
    value => value.verified_metadata.at(-1).source_name = `sessions/${firstSession}/evidence/../receipt.json`,
    value => value.verified_metadata.push(copy(value.verified_metadata.at(-1))),
    value => value.verified_metadata = value.verified_metadata.filter(row => row.path !== 'run/scene.json')];
  for (const change of changes) {
    const bad = evidenceFixture(); change(bad); evidenceButton(ui).click(); await resolveEvidence(ui, bad);
    assert.equal(ui.ids.get('recording-evidence-content').hidden, true);
    assert.equal(ui.ids.get('recording-evidence-links').children.length, 0);
  }
}

async function reasonTextIsEscapedAndMalformedSessionsCannotOfferReads() {
  const descriptor = Object.getOwnPropertyDescriptor(Element.prototype, 'innerHTML');
  Object.defineProperty(Element.prototype, 'innerHTML', {configurable:true,set() {throw new Error('Unsafe innerHTML');}});
  try {
    const ui = recordingFixture('planar', {savedRecordings:[{name:firstSession,complete:true},
      {name:'../outside',complete:true},{name:secondSession,complete:false}]}); await ui.ready();
    assert.equal(evidenceButton(ui, 1), undefined); assert.equal(evidenceButton(ui, 2), undefined);
    evidenceButton(ui).click();
    const reason = '<img src=x onerror="alert(1)"> & <script>unsafe</script>';
    await resolveEvidence(ui, {status:'unavailable',session:firstSession,reason});
    assert(ui.ids.get('recording-evidence-status').textContent.includes(reason));
    assert.equal(ui.ids.get('recording-evidence-status').children.length, 0);
  } finally {
    if (descriptor) Object.defineProperty(Element.prototype, 'innerHTML', descriptor); else delete Element.prototype.innerHTML;
  }
}

async function changingSelectionAbortsTheOldFetchBeforeStartingTheNextOne() {
  const ui = recordingFixture('shared-world', {savedRecordings:[firstSession,secondSession].map(name => ({name,complete:true}))}); await ui.ready();
  evidenceButton(ui).click(); await drain(); const previous = ui.pending[0];
  evidenceButton(ui, 1).click(); await drain();
  assert.equal(ui.aborts.length, 1); assert.equal(previous.options.signal.aborted, true);
  assert.equal(ui.pending.length, 1); assert.equal(ui.pending[0].path, endpoint(secondSession));
  assert.equal(ui.ids.get('recording-evidence-content').hidden, true);
  await resolveEvidence(ui, evidenceFixture(secondSession));
  assert.match(ui.ids.get('recording-evidence-session').textContent, new RegExp(secondSession));
  assert.equal(ui.timers.size, 0);
}

async function ignoredCancellationAndOlderRuntimeStillRejectStaleReplies() {
  for (const settings of [{honorAbort:false},{injectController:false,honorAbort:false}]) {
    const ui = recordingFixture('shared-world', {savedRecordings:[firstSession,secondSession].map(name => ({name,complete:true}))}, settings); await ui.ready();
    evidenceButton(ui).click(); await drain(); evidenceButton(ui, 1).click(); await drain(); evidenceButton(ui).click(); await drain();
    assert.equal(ui.pending.length, 1, 'No overlapping request while an old response ignores cancellation');
    const stale = evidenceFixture(); stale.recorded = {events:101,actions:99,resets:1};
    Object.assign(stale.replay, {events_checked:101,actions_checked:99,resets_checked:1});
    await resolveEvidence(ui, stale);
    assert.equal(ui.ids.get('recording-evidence-content').hidden, true, 'A→B→A still rejects the old A generation');
    assert.equal(evidenceRequests(ui).length, 2); assert.equal(ui.pending.length, 1);
    await resolveEvidence(ui, evidenceFixture());
    assert.match(ui.ids.get('recording-evidence-counts').textContent, /4 actions · 2 resets/);
    assert.equal(ui.timers.size, 0);
  }
}

async function hidingOrLeavingCancelsReadsAndRequiresExplicitRetry() {
  for (const hide of [ui => ui.visibility(true), ui => ui.dispatchWindow('pagehide')]) {
    const ui = recordingFixture(); await ui.ready(); evidenceButton(ui).click(); await drain(); hide(ui); await drain();
    assert.equal(ui.aborts.length, 1); assert.equal(ui.pending.length, 0); assert.equal(ui.timers.size, 0);
    assert.equal(ui.ids.get('recording-evidence-retry').hidden, false);
    assert.match(ui.ids.get('recording-evidence-status').textContent, /read paused/);
    ui.visibility(false); await ui.nextTimer();
    assert.equal(evidenceRequests(ui).length, 1, 'Visibility resumes ordinary status, not receipt polling');
    ui.ids.get('recording-evidence-retry').click(); await resolveEvidence(ui, evidenceFixture());
    assert.equal(evidenceRequests(ui).length, 2); assert.equal(ui.timers.size, 0);
    evidenceButton(ui).click(); await drain(); hide(ui); await drain();
    assert.equal(ui.aborts.length, 2); assert.equal(ui.pending.length, 0); assert.equal(ui.timers.size, 0);
    assert.equal(ui.ids.get('recording-evidence-content').hidden, false, 'Previously checked receipt remains visible');
    assert.match(ui.ids.get('recording-evidence-status').textContent, /read paused/);
    assert.equal(ui.ids.get('recording-evidence-retry').hidden, false);
  }
}

async function loadingPollsStopAfterTwelveReadsAndNeverOverlap() {
  const ui = recordingFixture(); await ui.ready(); evidenceButton(ui).click(); await drain();
  for (let index = 0; index < 12; index++) {
    assert.equal(ui.pending.length, 1);
    await resolveEvidence(ui, {status:'loading',session:firstSession});
    assert.equal(ui.pending.length, 0);
    if (index < 11) { assert.equal(ui.timers.size, 1); await ui.nextTimer(); }
  }
  assert.equal(evidenceRequests(ui).length, 12); assert.equal(ui.timers.size, 0);
  assert.equal(ui.ids.get('recording-evidence-retry').hidden, false);
  ui.ids.get('recording-evidence-retry').click(); await resolveEvidence(ui, {status:'loading',session:firstSession});
  ui.visibility(true); assert.equal(ui.timers.size, 0);
  assert.equal(evidenceRequests(ui).length, 13);
}

async function aTimedOutReadOffersRetryWithoutInventingEvidence() {
  const ui = recordingFixture(); await ui.ready(); evidenceButton(ui).click(); await drain();
  await ui.nextTimer();
  assert.equal(ui.aborts.length, 1); assert.equal(ui.pending.length, 0);
  assert.equal(ui.ids.get('recording-evidence-content').hidden, true);
  assert.equal(ui.ids.get('recording-evidence-retry').hidden, false);
  assert.match(ui.ids.get('recording-evidence-status').textContent, /request limit.*unavailable/);
  assert.equal(ui.timers.size, 0);
  ui.ids.get('recording-evidence-retry').click(); await resolveEvidence(ui, {error:'metadata read failed'}, null, false);
  assert.equal(ui.ids.get('recording-evidence-content').hidden, true);
  assert.equal(ui.timers.size, 0);
}

async function unchangedExplicitRefreshKeepsNodesAndDisclosureStateStable() {
  const ui = recordingFixture(); await ui.ready(); const data = evidenceFixture(); data.replay.learned_actions_verified = null;
  evidenceButton(ui).click(); await resolveEvidence(ui, data);
  assert.match(ui.ids.get('recording-evidence-replay').textContent, /Learned actions verified: unavailable/);
  const rows = [...ui.ids.get('recording-evidence-identities').children], links = [...ui.ids.get('recording-evidence-links').children];
  const details = ui.ids.get('recording-evidence-details'); details.open = true; details.dispatch('toggle');
  const before = ui.backendTaskState(), count = ui.requests.length, start = performance.now();
  for (let index = 0; index < 20; index++) {
    data.inspection = {elapsed_s:index / 1000};
    evidenceButton(ui).click(); await resolveEvidence(ui, data);
  }
  const elapsed = performance.now() - start;
  assert.equal(ui.requests.length, count + 20, 'Only explicit refreshes read the selected bounded endpoint');
  assert.equal(details.open, true);
  ui.ids.get('recording-evidence-identities').children.forEach((row, index) => assert.equal(row, rows[index]));
  ui.ids.get('recording-evidence-links').children.forEach((link, index) => assert.equal(link, links[index]));
  assert.deepEqual(ui.backendTaskState(), before); assert.equal(ui.timers.size, 0);
  const nodes = descendants(ui.ids.get('recording-evidence-identities')).length + descendants(ui.ids.get('recording-evidence-links')).length;
  assert.equal(nodes, 28);
  console.log(`Scripted DOM only: 20 unchanged explicit refreshes ${elapsed.toFixed(3)}ms; ${nodes} identity/link nodes. Not browser responsiveness.`);
}

const checks = [explicitSelectionShowsAccessibleReceiptWithoutMovingControls,
  spatialSavedStateExportKeepsUnavailableReplayCountsDistinctFromZero,
  newSpatialReceiptShowsExactActionsWithoutClaimingPolicyDecisionChecks,
  spatialPolicyReceiptsSeparateDecisionsFromManualAndLegacyActions,
  spatialPolicyReceiptsRejectMissingOrContradictoryDecisionIdentities,
  missingLegacyReceiptAndUnknownFieldsNeverBecomeVerifiedReplay,
  exactLocalLinksBindSessionAndByteHashesWithoutConfusingCanonicalHashes,
  reasonTextIsEscapedAndMalformedSessionsCannotOfferReads,
  changingSelectionAbortsTheOldFetchBeforeStartingTheNextOne,
  ignoredCancellationAndOlderRuntimeStillRejectStaleReplies,
  hidingOrLeavingCancelsReadsAndRequiresExplicitRetry,
  loadingPollsStopAfterTwelveReadsAndNeverOverlap,
  aTimedOutReadOffersRetryWithoutInventingEvidence,
  unchangedExplicitRefreshKeepsNodesAndDisclosureStateStable];
for (const check of checks) { await check(); console.log(`PASS ${check.name}`); }
console.log(`Passed ${checks.length} selected-recording evidence UI checks.`);
