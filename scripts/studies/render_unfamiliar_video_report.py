#!/usr/bin/env python3
"""Render a portable three-case preflight-refusal report with embedded source clips."""
import argparse
import base64
from hashlib import sha256
from html import escape
import json
from pathlib import Path

MAX_HTML = 16 << 20
MAX_VIDEOS = 11 << 20


def render(study, output):
    root = Path(study).resolve(); output = Path(output)
    if output.exists(): raise ValueError('Preserve prior report; choose a new output')
    assessment = json.loads((root/'assessment.json').read_bytes())
    plan = json.loads((root/'evaluation-plan.json').read_bytes())
    if len(assessment['cases']) != 3 or {c['id'] for c in assessment['cases']} != {c['id'] for c in plan['cases']}:
        raise ValueError('Report requires all three original cases')
    if any(c.get('status') != 'failed' or c.get('stage') != 'preflight' for c in assessment['cases']):
        raise ValueError('This report profile requires three actual preflight refusals')
    sources = {c['id']: c for c in plan['cases']}; panels=[]; total=0
    for case in assessment['cases']:
        source=sources[case['id']]; video=Path(source['upload']); size=video.stat().st_size
        total+=size
        if total>MAX_VIDEOS: raise ValueError('Portable video byte bound exceeded; no cases omitted')
        raw=video.read_bytes()
        if sha256(raw).hexdigest()!=source['upload_sha256']:raise ValueError('Report video source changed')
        provenance=json.loads(Path(source['provenance']).read_bytes())
        receipt=json.loads((root/'normalized'/f"{case['id']}-receipt.json").read_bytes())
        h=lambda value:escape(str(value),quote=True)
        link=lambda path,label:f'<a href="{h(path)}">{h(label)}</a>'
        times=', '.join(f"{f['frame']} ({f['decoder_timestamp_s']:.3f}s)" for f in case['competing_frames'])
        counts='; '.join(f'{k}: {v}' for k,v in case['status_counts'].items())
        evidence=f"evaluation/{case['id']}/preflight/body_source"
        panels.append(f'''<section><h2>{h(provenance['title'])}</h2>
<p><strong>Refused at {h(case['stage'])}.</strong> {h(case['error'])}</p>
<video controls preload="none" aria-label="{h(provenance['title'])}" src="data:video/mp4;base64,{base64.b64encode(raw).decode()}"></video>
<p>{h(case['decoded_frames'])} decoded frames; {h(case['usable_frames'])}/{h(case['sampled_frames'])} usable sampled poses ({case['usable_fraction']:.1%}). Runtime {case['seconds']:.2f}s; peak RSS {case['peak_rss_bytes']/1e6:.1f} MB.</p>
<p>Sample statuses: {h(counts)}. “multiple_people” means unresolved competing model hypotheses, not verified person count.</p>
<details><summary>Competing sample frame IDs and decoder times</summary><p>{h(times)}</p></details>
<p>Body/ground construction, lab admission and authored task rollout: <strong>not reached</strong>. The original global ambiguity and coverage gates were unchanged. No native-clock body binding was produced.</p>
<p>{link(provenance['source_page'],'Original source and attribution')} · {h(provenance['author'])} · {link(provenance['license_url'],provenance['license'])}. Derivative retains the stated license.</p>
<p>Full-span H264 CRF18/yuv420p MP4, audio omitted. No crop, resizing or frame-rate resampling; original and derivative decoded counts/dimensions match. Maximum decoder-time difference: {receipt['max_decoder_timestamp_difference_s']:.3g}s. This is not independently calibrated capture timing.</p>
<details><summary>Exact identities and retained evidence</summary><dl>
<dt>Original SHA256</dt><dd>{h(source['original_sha256'])}</dd><dt>Normalized upload SHA256</dt><dd>{h(source['upload_sha256'])}</dd>
<dt>Raw pose SHA256</dt><dd>{h(case['raw_pose_sha256'])}</dd><dt>Evidence SHA256</dt><dd>{h(case['raw_evidence_sha256'])}</dd>
<dt>Original refusal path</dt><dd>{h(evidence+'/failure.json')}</dd></dl>
<p>{link(evidence+'/failure.json','Failure receipt')} · {link(evidence+'/evidence.json','All sampled evidence')} · {link(evidence+'/pose-landmarks.json','All raw pose alternatives')} · {link('evaluation/'+case['id']+'/supervisor-result.json','Worker/resource receipt')}</p></details></section>''')
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Weft — three unfamiliar video results</title>
<style>body{font:16px/1.5 system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#222;background:#fafafa}section{background:white;padding:1.4rem;margin:1.5rem 0;border:1px solid #ddd;border-radius:8px}video{display:block;max-width:100%;max-height:440px;margin:1rem auto}dd{font:12px/1.5 monospace;overflow-wrap:anywhere;margin:.25rem 0 1rem}summary{cursor:pointer}h1,h2{line-height:1.2}a{color:#175c73}code{overflow-wrap:anywhere}</style>
<h1>Three unfamiliar video results</h1><p><strong>0/3 reached a playable simulation.</strong> All three real source videos stopped at the existing ambiguous-pose gate. No failed cases were replaced or retried with different model settings. Task success was not evaluated because no lab was admitted.</p>
<p>Fixed development study · November 3, 2026 (PST) · $0 paid APIs · one sequential local worker per clip · all workers exited and were reaped.</p>
<p>One frozen setup: Lite CPU, VIDEO/raw, maximum two pose hypotheses, bounded sampling, requested native timing, visible-chain-v2 anchor. Explicit conditional model: human plus authored ground; equipment, other people and contacts remain unmodeled. No claimed ground-only inventory, training, avatar, rendering or physical validation.</p>
<p>No prior exact repository exposure was found for these titles before selection; model-training exposure is unknown. Two clips share one performer. These are normalized-upload development checks, not a universal-generalization or independent physical benchmark. The videos below are embedded and work offline; detailed JSON links require the adjacent study folder.</p>'''+''.join(panels)+'''<footer><p>Source videos and failures are evidence, not proof that the scenes are impossible to reconstruct. Changing settings after these outcomes would be a separate development experiment, not a repaired score for this frozen study.</p><p><a href="evaluation-plan.json">Frozen plan</a> · <a href="assessment.json">Compact result</a> · <a href="evaluation/result.json">Complete stage/accounting result</a> · <a href="normalized/attribution.json">Source/derivative attribution</a></p></footer></html>'''
    encoded=html.encode()
    if len(encoded)>MAX_HTML:raise ValueError('Report exceeds 16 MiB')
    output.write_bytes(encoded)
    return {'path':str(output),'bytes':len(encoded),'sha256':sha256(encoded).hexdigest(),'embedded_videos':3,'network_dependencies':0,'visual_browser_qa':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    print(json.dumps(render(a.study,a.out)))
