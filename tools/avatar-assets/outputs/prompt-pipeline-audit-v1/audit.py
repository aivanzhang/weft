"""Read-only release audit; media decoding/native checks are linked, not repeated."""
from datetime import datetime
import json
from pathlib import Path
import resource
import sqlite3
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from avatar_assets.safety import artifacts, read_json, sha256, write_json

started = time.monotonic()
out = Path(__file__).resolve().parent
names = ['agent-dragon-v1', 'agent-dragon-v2', 'agent-dragon-v3',
         'agent-robot-v1', 'agent-robot-v2', 'agent-robot-v3']
usage = {k: 0 for k in ('input_tokens', 'output_tokens', 'cached_input_tokens', 'reasoning_output_tokens')}
calls, jobs, deliveries = 0, [], []
for name in names:
    folder = ROOT/'outputs'/name
    request = read_json(folder/'request.json')
    assert request['max_credits'] == 0 and request['reuse_model'] is not None
    assert sha256(folder/request['reuse_model']['path']) == request['reuse_model']['sha256']
    receipts = list(folder.glob('agents/*/receipt.json'))
    for event_file in folder.glob('agents/*/agent.jsonl'):
        calls += 1
        for line in event_file.read_text().splitlines():
            event = json.loads(line)
            if event.get('type') == 'turn.completed':
                for key in usage:
                    usage[key] += event['usage'].get(key, 0)
        process = read_json(event_file.parent/'agent-process.json')
        assert process['status'] == 'finished' and process['exit_code'] == 0 and not process['manual_stop_required']
    files = [p for p in folder.rglob('*') if p.is_file()]
    jobs.append({'job': name, 'status': read_json(folder/'status.json')['status'], 'agent_calls': len(receipts),
                 'files': len(files), 'bytes': sum(p.stat().st_size for p in files),
                 'request_sha256': sha256(folder/'request.json'), 'status_sha256': sha256(folder/'status.json')})
    if not (folder/'delivery.json').exists():
        continue
    delivery = read_json(folder/'delivery.json')
    assert delivery['status'] == 'complete' and delivery['generation_mode'] == 'reused_generated_source'
    assert not any(delivery[k] for k in ('human_approval', 'quality_equivalence_verified', 'weft_runtime_verified', 'loop_verified'))
    for path, checksum in read_json(folder/'delivery-receipt.json').items():
        assert sha256(folder/path) == checksum
    for pack in delivery['packs']:
        assert sha256(folder/pack['path']/'manifest.json') == pack['manifest_sha256']
    model = folder/Path(delivery['model']).parent
    manifest = read_json(model/'manifest.json')
    media = read_json(folder/'media/manifest.json')
    spec = request['media_spec']
    assert media['parent_3d_manifest_sha256'] == sha256(model/'manifest.json')
    assert [c['id'] for c in media['clips']] == [s['id'] for s in read_json(folder/'plan.json')['states']]
    assert media['portrait']['width'] == media['portrait']['height'] == spec['portrait']
    clip_summary = []
    for clip in media['clips']:
        info = clip['inspection']
        assert info['width'] == info['height'] == spec['size'] and info['fps'] == 24
        assert info['duration_seconds'] == spec['seconds'] and info['frames'] == spec['seconds']*24
        assert info['codec'] == 'h264' and not info['audio'] and not info['has_transparency'] and info['motion_detected']
        assert sha256(folder/'media'/clip['path']) == info['sha256']
        frames = sorted((folder/'media/frames'/clip['id']).glob('*.png'))
        assert len(frames) == info['frames']+1
        assert sha256(folder/'media'/clip['fallback']) == sha256(frames[0])
        clip_summary.append({'id':clip['id'], 'frames':info['frames'], 'unique_decoded_thumbnails':info['unique_thumbnails'],
                             'source_pngs':len(frames), 'exact_first_frame_fallback':True})
    browser = ROOT/'outputs'/('browser-agent-dragon-v1' if 'dragon' in name else 'browser-agent-robot-v1')/'browser-test.json'
    proof = read_json(browser)
    assert proof['status'] == 'passed' and not proof['errors'] and not proof['external_requests']
    assert proof['cleanup']['server_exit_verified'] and proof['cleanup']['browser_exit_verified']
    events = sorted((folder/'events').glob('*.json'))
    elapsed = (datetime.fromisoformat(read_json(events[-1])['recorded'])-
               datetime.fromisoformat(read_json(events[0])['recorded'])).total_seconds()
    deliveries.append({'job': name, 'delivery_sha256': sha256(folder/'delivery.json'),
        'model_sha256': sha256(model/'model.glb'), 'vertices':manifest['rig_verification']['mesh_vertices'],
        'bones':len(manifest['rig_verification']['bones']), 'execution_event_span_seconds':elapsed,
        'media_spec':spec, 'clips':clip_summary, 'browser_evidence':str(browser.relative_to(ROOT)),
        'browser_sha256':sha256(browser), 'media_render':{k:media['job'][k] for k in
            ('elapsed_seconds','sampled_peak_rss_bytes','process_exit_verified')}, 'packs':delivery['packs']})

assert calls == 28 and len(deliveries) == 2
original = Path('/Users/ivan/Downloads/Poo avatar assets.zip')
archive_hash = sha256(original)
assert archive_hash == 'cb7eac375b5fcef0cd6c3f9457da7911f08f73337178c31fe33bfa090e6a54cb'
ledger_path = ROOT.parents[1]/'.local/meshy/ledger.sqlite3'
with sqlite3.connect(ledger_path.as_uri()+'?mode=ro', uri=True) as db:
    count, reserved = db.execute('SELECT count(*),sum(credits) FROM calls').fetchone()
    assert (count, reserved) == (15, 167)
report = {'schema_version':1, 'kind':'prompt_pipeline_delivery_audit', 'status':'passed',
    'recorded':datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(), 'jobs':jobs, 'deliveries':deliveries,
    'original_archive_sha256':archive_hash, 'real_agent_calls':calls, 'agent_usage':usage,
    'agent_model_version':'Unpinned CLI default; exact invocations and CLI version retained', 'usd_cost':None,
    'new_meshy_calls':0, 'new_meta_calls':0, 'canonical_meshy_submissions':count, 'canonical_meshy_credits':reserved,
    'no_active_owned_jobs':True, 'persistent_server':False,
    'scope':'Hashes, recorded independent native/media/browser evidence and actual fallback/source continuity; no new synthesis, human approval, Muse parity or physics proof',
    'elapsed_seconds':round(time.monotonic()-started,6), 'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
write_json(out/'report.json',report)
write_json(out/'manifest.json',{'schema_version':1, 'kind':report['kind'], 'status':'passed', 'artifacts':artifacts(out)})
print(json.dumps({'status':'passed','out':str(out),'agent_calls':calls,'deliveries':len(deliveries),
                  'seconds':report['elapsed_seconds'],'peak_rss_bytes':report['process_peak_rss_bytes']}))
