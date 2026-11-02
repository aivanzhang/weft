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
        assert path.suffix in ('.py','.json','.log','.jsonl','.png','.md'),str(path)
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
for path in (HERE/'package.py', HERE/'inputs.json'):
    target=Path('snapshots/quality-material-clumps-checkpoint-v1')/path.name
    (out/target).parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(path,out/target)
    size=path.stat().st_size; digest=sha(path)
    assert sha(out/target)==digest
    rows.append({'origin':str(path.relative_to(ROOT)),'bytes':size,'sha256':digest,'snapshot':str(target)})
    seen[(digest,size)]=str(target);selected_bytes+=size;copied_bytes+=size
write(out/'manifest.json',{'schema_version':1,'recorded':datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M:%S %Z'),
 'status':inputs['status'],'artifacts':rows,'local_binary_references_not_copied':references,
 'source_recipe':{'path':str((HERE/'package.py').relative_to(ROOT)),'sha256':sha(HERE/'package.py')},
 'source_inputs_sha256':sha(HERE/'inputs.json'),'source_receipts':len(rows),'unique_snapshots':len(seen),
 'source_bytes':selected_bytes,'copied_bytes':copied_bytes,'deduplication':'Identical bytes reuse one exact snapshot path; no normalization',
 'verified_focused_tests':inputs['verified_focused_tests'],'actual_lamp_public_finisher':True,'actual_lamp_independent_audit':True,
 'new_provider_calls':0,'new_pipeline_role_calls':0,'quality_approved':False,'fresh_studio_acceptance':False,
 'public_fringe_enabled':False,'approved_assets_unchanged':True,'weft_consumer_changed':False,
 'limits':inputs['limits'],'elapsed_seconds':round(time.monotonic()-started,3)})
with (out/'README.md').open('x') as f:
 f.write('# Material finishing and experimental coat evidence\n\nExact snapshots bind the non-furry material-sampling fix, existing authored lamp\npublic-finisher run and independent audit, source-neutral rig metadata, stopped\ncheek reflection/coat strategies, and their bounded checks. The manifest maps\nevery origin to byte-identical snapshots; duplicate content is stored once.\nLarge GLB, Blend and NumPy files remain at immutable local paths with hashes.\nThis compact bundle alone is not a runnable avatar or a fresh generated example.\n\nThe lamp preserves image bytes, material definitions, geometry/weights and sampled\nmotion, but its normal/tangent drift remains explicit. Its 256px renders are\nintegration drafts. Fur experiments do not replace approved avatars or enable the\npublic fringe path. Preserve failures and manifest errata; analytic test passes\nare not visual acceptance. No provider or generation-pipeline role calls occurred.\n\nSee ../../docs/quality-next-step.md for commands, source/rig identities, results,\nresource limits, job ownership, pending allowance and the next safe action.\nMuse parity, arbitrary actions and accepted hands-off Studio delivery remain open.\n')
print(json.dumps({'origins':len(rows),'unique_snapshots':len(seen),'bytes':copied_bytes,'manifest_sha256':sha(out/'manifest.json'),'seconds':round(time.monotonic()-started,3)}))
