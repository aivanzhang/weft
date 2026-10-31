/** Saved scorecard presentation only: no server, browser, models or physics. */
import assert from 'node:assert/strict';
import {Element, fixture, html, drain} from './lab_ui_smoke.mjs';

const dimensions = [
  'execution_actions', 'perception_identity', 'geometry_camera', 'temporal_dynamics',
  'physical_relationships', 'hidden_state_continuity', 'parameter_validity', 'causal_response',
  'long_run_stability', 'robustness_coverage', 'uncertainty_abstention', 'task_learning_utility',
  'reproducibility_integrity', 'practical_efficiency',
];
const sha = letter => letter.repeat(64);
const pointer = (name, digest = sha('b')) => ({source_name:name, path:`run/${name}`, sha256:digest});
const metric = (label, value, extra = {}) => ({id:'saved_measurement', label, value, unit:'trials',
  numerator:null, denominator:null, recorded_gate:null, evidence_kind:'simulated_development',
  partition:'exposed_development', evidence:[pointer('result.json')], ...extra});
const copy = value => JSON.parse(JSON.stringify(value));
const descendants = element => [element, ...element.children.flatMap(descendants)];
const content = element => descendants(element).map(child => child.textContent).join('\n');
const category = (ui, id) => ui.ids.get('evidence-scorecard-rows').children.find(row => row.dataset.category === id);
function expand(ui, id) { const row = category(ui, id); row.open = true; row.dispatch('toggle'); return row; }
const byTag = (root, tag) => descendants(root).filter(element => element.tagName === tag);
const measurementRows = root => byTag(root, 'TBODY').flatMap(body => body.children);

function inspectionFixture() {
  const rows = dimensions.map(id => ({id, label:id, status:'unavailable', evidence_kind:'unknown', partition:'unknown',
    metrics:[metric('Independent measurement', null, {evidence_kind:'unknown', partition:'unknown', evidence:[]})],
    failure_examples:[], failure_examples_omitted:0, evidence:[],
    uncertainty_and_missing_measurements:['Independent physical measurements are unavailable.']}));
  const learning = rows.find(row => row.id === 'task_learning_utility');
  Object.assign(learning, {status:'failure_reported', evidence_kind:'simulated_development', partition:'exposed_development',
    metrics:[metric('Physical transfer trials', null, {evidence_kind:'unknown', partition:'unknown', evidence:[]}),
      metric('learned saved simulator successes', 0, {numerator:0, denominator:3}),
      metric('authored saved simulator successes', 3, {numerator:3, denominator:3}),
      metric('learned saved mean reward', null, {unit:'reward'})], evidence:[pointer('result.json')],
    uncertainty_and_missing_measurements:['Saved simulation only; no independent policy/trial replay.',
      'Unchanged selection is not evidence of improvement over the authored controller.']});
  const geometry = rows.find(row => row.id === 'physical_relationships');
  Object.assign(geometry, {status:'failure_reported', evidence_kind:'software_check', partition:'construction',
    metrics:[metric('Source pose floor-limit failures', 8, {unit:'poses', numerator:8, denominator:117,
      evidence_kind:'software_check', partition:'construction'}),
      metric('Saved peak floor penetration', .04, {unit:'model_units', evidence_kind:'software_check', partition:'construction',
        recorded_gate:{name:'max_floor_penetration_model_units', operator:'<=', value:.025, unit:'model_units'}})],
    evidence:[pointer('result.json')], failure_examples:[{kind:'source_pose', source_frame_index:0, time_s:0,
      clock:'declared_reference_clock', max_penetration_model_units:.04}], failure_examples_omitted:7});
  const integrity = rows.find(row => row.id === 'reproducibility_integrity');
  const manifest = {source_name:'manifest.json', path:null, sha256:sha('a')};
  Object.assign(integrity, {status:'evidence_available', evidence_kind:'software_check', partition:'construction',
    metrics:[metric('Selected metadata identities', 2, {unit:'artifacts', evidence_kind:'software_check', partition:'construction', evidence:[manifest]})],
    evidence:[manifest]});
  return {status:'available', profile:'lab-inspection-v1', source_manifest_sha256:sha('a'),
    source:{sha256:sha('c'), decoded_frames:10, scope:'Saved source count, not independent accuracy.'},
    tracking:{reference_poses:null, scope:'No compact tracking measurements.'}, actors:[], stages:[],
    geometry:{status:'no_saved_comparison', panels:[], assumptions:[]}, artifact_links:[],
    verified_metadata:[{...pointer('manifest.json', sha('a')), bytes:300}, {...pointer('result.json'), bytes:500}],
    evidence_scorecard:{schema_version:1, profile:'weft-evidence-scorecard-v1', source_manifest_sha256:sha('a'),
      scope:'Evidence availability and saved failures, not an overall physical grade.',
      identity_scope:'Matching sealed summary identities only; no policy/trial replay or full-package admission.', rows}};
}

async function accessibleCollapsedCategoriesKeepAvailabilitySeparateFromSuccess() {
  const ui = fixture('planar', {inspection:inspectionFixture()}); await ui.ready();
  const rows = ui.ids.get('evidence-scorecard-rows').children;
  assert.deepEqual(rows.map(row => row.dataset.category), dimensions);
  assert(rows.every(row => row.tagName === 'DETAILS' && !row.open && row.children[0].tagName === 'SUMMARY'));
  assert.equal(byTag(ui.ids.get('evidence-scorecard-rows'), 'TABLE').length, 0, 'Closed categories do not build hidden tables');
  assert.match(content(category(ui, 'reproducibility_integrity')), /Evidence available — not a pass/);
  assert.match(content(category(ui, 'causal_response')), /Unavailable — not measured here/);
  assert.match(content(category(ui, 'task_learning_utility')), /Failures recorded/);
  assert.match(ui.ids.get('evidence-scorecard-scope').textContent, /no policy\/trial replay/);
  assert.match(html, /id="evidence-scorecard" aria-labelledby="evidence-scorecard-heading"/);
  assert.match(html, /id="evidence-scorecard-status" role="status"/);
  assert.match(html, /Unavailable means not measured here, not zero error/);
  assert.match(html, /Simulator outcomes do not establish physical accuracy/);
  ui.stage('repair'); assert.equal(ui.ids.get('repair').hidden, false);
}

async function savedCountsNullsLimitsAndFailureClocksRemainDistinct() {
  const inspection = inspectionFixture();
  inspection.evidence_scorecard.rows.find(row => row.id === 'task_learning_utility').metrics.push(
    metric('No sampled trials', 0, {numerator:0, denominator:0}));
  const ui = fixture('planar', {inspection}); await ui.ready();
  const learning = expand(ui, 'task_learning_utility'), rows = measurementRows(learning);
  assert.equal(rows[0].children[1].textContent, 'Unavailable');
  assert.equal(rows[1].children[1].textContent, '0 / 3 trials');
  assert.equal(rows[2].children[1].textContent, '3 / 3 trials');
  assert.equal(rows[3].children[1].textContent, 'Unavailable');
  assert.equal(rows[4].children[1].textContent, '0 / 0 trials', 'An empty denominator stays explicit; no percentage is inferred');
  assert.match(rows[1].children[2].textContent, /simulated development.*exposed development/);
  assert.match(rows[0].children[2].textContent, /unknown.*unknown/);
  assert.match(content(learning), /no independent policy\/trial replay/);
  const geometry = expand(ui, 'physical_relationships');
  assert.match(content(geometry), /8 \/ 117 poses/);
  assert.match(content(geometry), /Recorded limit: max floor penetration model units <= 0.025 model units/);
  assert.match(content(geometry), /"source_frame_index": 0/);
  assert.match(content(geometry), /"time_s": 0/);
  assert.match(content(geometry), /declared_reference_clock/);
  assert.match(content(geometry), /7 additional examples omitted/);
  assert(byTag(geometry, 'TH').every(th => ['row','col'].includes(th.getAttribute('scope'))));
}

async function linksRequireExactVerifiedLocalIdentityAndManifestBinding() {
  const inspection = inspectionFixture(), row = inspection.evidence_scorecard.rows[0];
  const good = pointer('result.json'), manifest = {source_name:'manifest.json', path:null, sha256:sha('a')};
  const invalid = [
    {...good, sha256:sha('c')}, {...good, path:'run/other.json'},
    {source_name:'manifest.json', path:null, sha256:sha('c')},
    pointer('unverified.json'), {...pointer('../escape.json'), path:'run/../escape.json'},
    {...pointer('external.json'), path:'https://example.test/result.json'},
  ];
  inspection.artifact_links.push({name:'unverified.json', sha256:sha('b'), bytes:20, label:'Artifact listing is not verification'});
  inspection.verified_metadata.push({...pointer('../escape.json'), bytes:20}, {...invalid[5], bytes:20});
  Object.assign(row, {status:'evidence_available', metrics:[metric('Exact pointers', 1, {evidence:[good, manifest, ...invalid]})], evidence:[]});
  const ui = fixture('planar', {inspection}); await ui.ready();
  const requestCount = ui.requests.length, opened = expand(ui, 'execution_actions'), anchors = byTag(opened, 'A');
  assert.deepEqual(anchors.map(anchor => anchor.href), [
    '/files/run/result.json?token=synthetic-token', '/files/run/manifest.json?token=synthetic-token',
  ]);
  assert.equal((content(opened).match(/verified local download unavailable/g) || []).length, invalid.length);
  assert.match(content(opened), new RegExp(`SHA-256: ${sha('b')}`));
  assert.equal(ui.requests.length, requestCount, 'Rendering evidence links does not fetch artifacts');
  const unbound = inspectionFixture(); unbound.evidence_scorecard.source_manifest_sha256 = sha('c');
  const second = fixture('planar', {inspection:unbound}); await second.ready();
  assert.equal(second.ids.get('evidence-scorecard-rows').children.length, 0, 'A different manifest rejects the entire card');
  assert.match(second.ids.get('evidence-scorecard-status').textContent, /Evidence summary unavailable/);
}

async function crossRunAndInvalidManifestIdentitiesRejectTheCard() {
  const mutations = [
    inspection => { inspection.source_manifest_sha256 = sha('c'); },
    inspection => { inspection.evidence_scorecard.source_manifest_sha256 = sha('c'); inspection.verified_metadata[0].sha256 = sha('c'); },
    inspection => { delete inspection.source_manifest_sha256; },
    inspection => { delete inspection.evidence_scorecard.source_manifest_sha256; },
    ...[null, '', 'A'.repeat(64), 'a'.repeat(63), `${sha('a')}\n`, 123].map(value => inspection => {
      inspection.source_manifest_sha256 = value; inspection.evidence_scorecard.source_manifest_sha256 = value;
    }),
  ];
  for (const mutate of mutations) {
    const inspection = inspectionFixture(); mutate(inspection);
    const ui = fixture('planar', {inspection}); await ui.ready();
    assert.match(ui.ids.get('evidence-scorecard-status').textContent, /Evidence summary unavailable/);
    assert.equal(ui.ids.get('evidence-scorecard-rows').children.length, 0);
    assert.equal(ui.ids.get('toggle').disabled, false);
    assert.equal(ui.timers.size, 0);
  }
}

async function savedTextCannotBecomeMarkupOrExecutableLinks() {
  const payload = '<img src=x onerror="alert(1)"><script>alert(2)</script>', inspection = inspectionFixture();
  const row = inspection.evidence_scorecard.rows[0];
  Object.assign(row, {status:'failure_reported', uncertainty_and_missing_measurements:[payload],
    metrics:[metric(payload, 0, {evidence_kind:payload, partition:payload, evidence:[{source_name:'javascript:alert(1)', path:'javascript:alert(1)', sha256:sha('a')}]})],
    failure_examples:[{message:`first line\n${payload}`}], evidence:[]});
  inspection.evidence_scorecard.scope = payload;
  const ui = fixture('planar', {inspection}); await ui.ready();
  const opened = expand(ui, 'execution_actions');
  assert.match(content(opened), /<img src=x onerror=/);
  assert.match(content(opened), /first line\\n/);
  assert.equal(byTag(opened, 'IMG').length + byTag(opened, 'SCRIPT').length, 0);
  assert.equal(byTag(opened, 'A').length, 0);
  assert.equal(ui.ids.get('evidence-scorecard-scope').textContent.startsWith(payload), true);
}

async function missingUnknownAndPartialScorecardsStayUnavailable() {
  for (const value of [undefined, null, {}, {schema_version:1, profile:'future-scorecard', rows:[]},
    {schema_version:2, profile:'weft-evidence-scorecard-v1', rows:[]},
    {schema_version:1, profile:'weft-evidence-scorecard-v1', rows:[{id:'unknown_category'}]}]) {
    const inspection = inspectionFixture(); inspection.evidence_scorecard = value;
    const ui = fixture('planar', {inspection}); await ui.ready();
    assert.match(ui.ids.get('evidence-scorecard-status').textContent, /Evidence summary unavailable/);
    assert.equal(ui.ids.get('evidence-scorecard-rows').children.length, 0);
    assert.equal(ui.ids.get('toggle').disabled, false);
  }
  const absentInspection = fixture(); await absentInspection.ready();
  assert.match(absentInspection.ids.get('evidence-scorecard-status').textContent, /Evidence summary unavailable/);
  const partial = inspectionFixture();
  partial.evidence_scorecard.rows = [{id:'execution_actions', status:'unrecognized'},
    {id:'task_learning_utility', status:'evidence_available', metrics:[metric('Missing count', null, {numerator:0, denominator:3}),
      {label:'Missing numeric field'}, metric('Numeric strings are not measurements', '0')]}];
  delete partial.verified_metadata;
  const ui = fixture('planar', {inspection:partial}); await ui.ready();
  assert.equal(ui.ids.get('evidence-scorecard-rows').children.length, 14);
  assert.match(content(expand(ui, 'execution_actions')), /Measurements unavailable/);
  assert.match(content(expand(ui, 'causal_response')), /Unavailable — not measured here/);
  assert(measurementRows(expand(ui, 'task_learning_utility')).every(row => row.children[1].textContent === 'Unavailable'));
  const duplicate = inspectionFixture(); duplicate.evidence_scorecard.rows[1] = copy(duplicate.evidence_scorecard.rows[0]);
  const repeated = fixture('planar', {inspection:duplicate}); await repeated.ready();
  assert.match(repeated.ids.get('evidence-scorecard-status').textContent, /Evidence summary unavailable/);
}

async function refreshesPreserveDisclosuresWithoutExtraPollingOrPhysics() {
  const inspection = inspectionFixture(), ui = fixture('shared-world', {sharedExercise:true, inspection, inspectionLoading:1});
  await ui.ready(); assert.equal(ui.ids.get('evidence-scorecard-rows').children.length, 0);
  await ui.nextTimer();
  const before = {state:ui.backendTaskState(), actor:ui.ids.get('actor').value, mode:ui.ids.get('mode-description').textContent,
    requests:ui.requests.length, inspections:ui.requests.filter(request => request.path === '/api/inspection').length};
  const opened = expand(ui, 'task_learning_utility'), table = byTag(opened, 'TABLE')[0], count = descendants(opened).length;
  for (let index = 0; index < 5; index++) { opened.open = false; opened.dispatch('toggle'); opened.open = true; opened.dispatch('toggle'); }
  assert.equal(descendants(opened).length, count); assert.equal(byTag(opened, 'TABLE')[0], table);
  assert.equal(ui.requests.length, before.requests); assert.equal(ui.timers.size, 0);
  ui.visibility(true); ui.visibility(false); await ui.nextTimer();
  assert.equal(ui.requests.length, before.requests + 1);
  assert.equal(ui.requests.filter(request => request.path === '/api/inspection').length, before.inspections);
  ui.ids.get('inspection-retry').click(); await ui.nextTimer();
  assert.equal(category(ui, 'task_learning_utility'), opened); assert.equal(opened.open, true);
  assert.equal(byTag(opened, 'TABLE')[0], table); assert.equal(ui.timers.size, 0);
  assert.deepEqual(ui.backendTaskState(), before.state); assert.equal(ui.ids.get('actor').value, before.actor);
  assert.equal(ui.ids.get('mode-description').textContent, before.mode);
  inspection.evidence_scorecard.rows.find(row => row.id === 'task_learning_utility').metrics[1].value = 1;
  inspection.evidence_scorecard.rows.find(row => row.id === 'task_learning_utility').metrics[1].numerator = 1;
  ui.ids.get('inspection-retry').click(); await ui.nextTimer();
  assert.notEqual(category(ui, 'task_learning_utility'), opened);
  assert.match(content(expand(ui, 'task_learning_utility')), /1 \/ 3 trials/);
  assert.deepEqual(ui.backendTaskState(), before.state);
  assert(!ui.requests.some(request => request.path === '/api/step'));
  await drain();
}

async function maximumMetricEnvelopeStaysLazyAndBounded() {
  const inspection = inspectionFixture();
  for (const row of inspection.evidence_scorecard.rows) {
    row.status = 'evidence_available'; row.evidence = []; row.failure_examples = []; row.failure_examples_omitted = 0;
    row.metrics = Array.from({length:16}, (_, index) => metric(`Saved development count ${index}`, index, {evidence:[]}));
  }
  assert(Buffer.byteLength(JSON.stringify(inspection.evidence_scorecard)) <= 64 * 1024);
  const started = performance.now(), ui = fixture('planar', {inspection}); await ui.ready();
  const root = ui.ids.get('evidence-scorecard-rows'), initialNodes = descendants(root).length;
  assert.equal(byTag(root, 'TABLE').length, 0);
  expand(ui, dimensions[0]); assert.equal(measurementRows(root).length, 16);
  for (const id of dimensions.slice(1)) expand(ui, id);
  assert.equal(measurementRows(root).length, 224); assert.equal(byTag(root, 'TABLE').length, 14);
  const nodeCount = descendants(root).length, first = root.children[0];
  ui.ids.get('inspection-retry').click(); await ui.nextTimer();
  assert.equal(root.children[0], first); assert.equal(descendants(root).length, nodeCount);
  assert.equal(ui.timers.size, 0);
  console.log(`MEASURE scripted DOM: ${initialNodes} initial nodes, ${nodeCount} expanded nodes, 224 metrics, one unchanged refresh, ${(performance.now() - started).toFixed(2)} ms; not browser timing`);
}

async function unchangedThenMismatchedRefreshClearsThePriorCard() {
  const inspection = inspectionFixture(), ui = fixture('planar', {inspection}); await ui.ready();
  const opened = expand(ui, 'reproducibility_integrity'), originalState = ui.backendState();
  assert.equal(byTag(opened, 'A').length, 2);
  ui.ids.get('inspection-retry').click(); await ui.nextTimer();
  assert.equal(category(ui, 'reproducibility_integrity'), opened);
  assert.equal(opened.open, true);
  inspection.source_manifest_sha256 = sha('c');
  ui.ids.get('inspection-retry').click(); await ui.nextTimer();
  assert.match(ui.ids.get('evidence-scorecard-status').textContent, /Evidence summary unavailable/);
  assert.equal(ui.ids.get('evidence-scorecard-rows').children.length, 0, 'Changing only the inspection identity invalidates the cached card');
  assert.equal(byTag(ui.ids.get('evidence-scorecard-rows'), 'A').length, 0);
  assert.equal(ui.timers.size, 0); assert.deepEqual(ui.backendState(), originalState);
  inspection.source_manifest_sha256 = sha('a');
  ui.ids.get('inspection-retry').click(); await ui.nextTimer();
  assert.equal(ui.ids.get('evidence-scorecard-rows').children.length, 14);
  assert.notEqual(category(ui, 'reproducibility_integrity'), opened);
  assert.equal(byTag(expand(ui, 'reproducibility_integrity'), 'A').length, 2);
  assert.deepEqual(ui.backendState(), originalState);
  assert(!ui.requests.some(request => request.path === '/api/step'));
}

for (const name of ['innerHTML','outerHTML']) Object.defineProperty(Element.prototype, name, {
  configurable:true, set() { throw new Error(`Evidence rendering must not use ${name}`); },
});
for (const check of [accessibleCollapsedCategoriesKeepAvailabilitySeparateFromSuccess,
  savedCountsNullsLimitsAndFailureClocksRemainDistinct, linksRequireExactVerifiedLocalIdentityAndManifestBinding,
  crossRunAndInvalidManifestIdentitiesRejectTheCard, savedTextCannotBecomeMarkupOrExecutableLinks,
  missingUnknownAndPartialScorecardsStayUnavailable, refreshesPreserveDisclosuresWithoutExtraPollingOrPhysics,
  maximumMetricEnvelopeStaysLazyAndBounded, unchangedThenMismatchedRefreshClearsThePriorCard]) {
  await check(); console.log(`PASS ${check.name}`);
}
