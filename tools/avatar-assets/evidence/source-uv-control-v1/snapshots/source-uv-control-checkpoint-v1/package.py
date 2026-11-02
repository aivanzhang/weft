# Exact byte custody; large native/model arrays remain explicit local references.
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,shutil,time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
inputs=json.loads((HERE/'inputs.json').read_text());out=ROOT/inputs['output']
assert out.is_dir() and not list(out.iterdir())
started=time.monotonic()
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,d):
    with p.open('x') as f:json.dump(d,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
rows=[]; references=[];seen={};selected_bytes=0;copied_bytes=0
for folder in inputs['folders']:
    origin=ROOT/'outputs'/folder
    assert origin.is_dir()
    for path in sorted(origin.rglob('*')):
        assert not path.is_symlink()
        if not path.is_file() or '__pycache__' in path.parts or '.cache' in path.parts:continue
        size=path.stat().st_size
        row={'origin':str(path.relative_to(ROOT)),'bytes':size,'sha256':sha(path)}
        if path.suffix in ('.glb','.blend','.npz'):
            references.append(row);continue
        assert path.suffix in ('.py','.json','.log','.jsonl','.png'),str(path)
        assert size<=inputs['limits']['single_file_bytes']
        selected_bytes+=size
        assert len(rows)<inputs['limits']['selected_files'] and selected_bytes<=inputs['limits']['selected_bytes']
        key=(row['sha256'],size)
        if key not in seen:
            target=Path('snapshots')/folder/path.relative_to(origin)
            (out/target).parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,out/target)
            assert sha(out/target)==row['sha256']
            seen[key]=str(target);copied_bytes+=size
        row['snapshot']=seen[key]; rows.append(row)
        assert time.monotonic()-started<30
for ref in inputs['external_references']:
    path=Path(ref['path']);path=path if path.is_absolute() else ROOT/path
    assert sha(path)==ref['sha256'],str(path)
    references.append({'origin':ref['path'],'bytes':path.stat().st_size,'sha256':ref['sha256'],'unchanged':True})
for path in (HERE/'package.py', HERE/'inputs.json', HERE/'preparation-recovery.json'):
    target=Path('snapshots/source-uv-control-checkpoint-v1')/path.name
    (out/target).parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(path,out/target)
    size=path.stat().st_size; digest=sha(path)
    assert sha(out/target)==digest
    rows.append({'origin':str(path.relative_to(ROOT)),'bytes':size,'sha256':digest,'snapshot':str(target)})
    seen[(digest,size)]=str(target);selected_bytes+=size;copied_bytes+=size
write(out/'manifest.json',{'schema_version':1,'recorded':datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M:%S %Z'),
 'status':'component_checks_pass_appearance_rejected','artifacts':rows,'local_binary_references_not_copied':references,
 'source_recipe':{'path':str((HERE/'package.py').relative_to(ROOT)),'sha256':sha(HERE/'package.py')},
 'source_inputs_sha256':sha(HERE/'inputs.json'),'source_receipts':len(rows),'unique_snapshots':len(seen),
 'source_bytes':selected_bytes,'copied_bytes':copied_bytes,'deduplication':'Identical bytes reuse one exact snapshot path; no normalization',
 'verified_focused_tests':55,'native_guard_real_source_check':True,'actual_saved_report_cpu_check':True,
 'new_provider_calls':0,'new_pipeline_role_calls':0,'quality_approved':False,'fresh_studio_acceptance':False,
 'public_fringe_enabled':False,'approved_assets_unchanged':True,'weft_consumer_changed':False,
 'limits':inputs['limits'],'elapsed_seconds':round(time.monotonic()-started,3)})
with (out/'README.md').open('x') as f:
 f.write('''# Source-UV control and export guard evidence\n\nExact snapshots of the local actual-source experiment, rejected appearance, export\nguard and custody verification. `manifest.json` maps every origin to a byte-exact\nsnapshot; duplicate bytes reuse the first snapshot. Logs and failed attempts are\nretained without normalization. Large GLB/Blend/NumPy sources stay at the listed\nimmutable local paths with hashes; this compact bundle alone is not a runnable\navatar or complete source pack. Runtime caches are excluded.\n\nThe actual control has a rig and sampled motion/skin proofs, but the eight real\nrendered views/poses still show folded shoulder/cheek surfaces and sparse fur.\nIt is rejected and does not replace approved models. The generic guard passes\n20 analytic checks, 22 diagnostic checks and 13 distinct integration contracts;\nthe initial path-alias test failure and two changed-case reruns are retained.\nReal native admission and exact saved-report diagnostic checks also pass. None\nof these component checks proves a fresh Studio delivery, arbitrary actions,\ncontinuous flight, seamless loops, physical correctness or Muse-level quality.\n\nSee `../../docs/source-uv-control.md` for commands, resource measurements, source\nassumptions, immutable model identities, limitations and the next safe action.\nNo provider or pipeline AI-role calls were made for this increment.\n''')
print(json.dumps({'origins':len(rows),'unique_snapshots':len(seen),'bytes':copied_bytes,'manifest_sha256':sha(out/'manifest.json'),'seconds':round(time.monotonic()-started,3)}))
