import hashlib,json,sys,traceback
from pathlib import Path
out=Path(__file__).resolve().parent
sys.path.insert(0,str(out/'source'))
from avatar_assets.source_skin_fidelity import restore_source_skin
p=json.loads((out/'prepared.json').read_text())
for row in p['source_pins']:
    assert hashlib.sha256((out/'source'/row['path']).read_bytes()).hexdigest()==row['sha256']
try:
    result=restore_source_skin(p['inputs']['source']['path'],out/'model.glb',p['inputs']['snapshot']['path'],p['inputs']['snapshot_report']['path'],experimental=True,expected_hashes={key:value['sha256'] for key,value in p['inputs'].items()})
except Exception as error:
    (out/'restore-failure.json').write_text(json.dumps({'status':'rejected','type':type(error).__name__,'message':str(error)},indent=2)+'\n')
    raise
(out/'restore-report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':'restored','result_sha256':result['result_sha256'],'vertices':result['source_vertices'],'exported_entries':result['exported_vertex_entries'],'triangles':result['triangles'],'elapsed_seconds':result['elapsed_seconds'],'sampled_peak_rss_bytes':result['sampled_peak_rss_bytes']}))
