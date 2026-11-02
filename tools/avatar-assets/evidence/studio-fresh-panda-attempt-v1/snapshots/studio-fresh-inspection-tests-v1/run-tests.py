import hashlib,json,subprocess,sys,time,unittest
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
sys.path.insert(0,str(root));sys.path.insert(0,str(root/'tests'))
pins=json.loads((out/'source-pins.json').read_text())
def check():
    for item in pins:
        assert hashlib.sha256((root/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
        assert hashlib.sha256((out/item['saved']).read_bytes()).hexdigest()==item['sha256'],item['saved']
check();started=time.monotonic()
suite=unittest.defaultTestLoader.loadTestsFromName('test_studio_live_verifier')
result=unittest.TextTestRunner(verbosity=2).run(suite)
summary={'provenance':'analytic_fixtures_not_generated_assets','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'elapsed_seconds':time.monotonic()-started,'python':sys.version,'server_started':False,'browser_started':False,'provider_calls':0,'agent_calls':0}
check()
command=['/opt/homebrew/bin/node', '--check', 'browser/tests/studio-live.mjs']
syntax=subprocess.run(command,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=10,check=False)
summary['browser_syntax']={'command':command,'exit_code':syntax.returncode,'output':syntax.stdout,'executed_browser':False}
(out/'test-result.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
raise SystemExit(0 if result.wasSuccessful() and syntax.returncode==0 else 1)
