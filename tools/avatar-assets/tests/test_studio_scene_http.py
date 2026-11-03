"""HTTP handler fixtures without sockets, workers or provider calls."""
from copy import deepcopy
from email.message import Message
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from avatar_assets.studio import StudioHandler
from avatar_assets.studio_jobs import StudioJobs
from test_studio_jobs import FixtureExecutor
from test_studio_scene_jobs import cards


class SceneHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name).resolve()/'tools/avatar-assets';root.mkdir(parents=True)
        self.executor=FixtureExecutor();self.jobs=StudioJobs(root,executor=self.executor)
        self.resolved=[]
        def resolve(identifier):
            if identifier!='a'*32:raise ValueError('Unknown selected reference')
            self.resolved.append(identifier)
            folder=root/'references';folder.mkdir(exist_ok=True)
            image=folder/'first.png'
            if not image.exists():image.write_bytes(b'analytic reference')
            return image
        self.server=SimpleNamespace(token='analytic-token',manager=self.jobs,references=SimpleNamespace(resolve=resolve))

    def dispatch(self,method,path,value=None,token='analytic-token',request_id='pair'):
        handler=object.__new__(StudioHandler);handler.server=self.server;handler.path=path
        raw=json.dumps(value).encode();handler.rfile=io.BytesIO(raw)
        headers=Message();headers['Content-Type']='application/json';headers['Content-Length']=str(len(raw))
        if token is not None:headers['X-Avatar-Studio-Token']=token
        if request_id is not None:headers['X-Avatar-Request-Id']=request_id
        handler.headers=headers;handler._allowed=lambda:True
        results=[];handler._reply=lambda status,body:results.append((status,body))
        handler._error=lambda status,message,*args:results.append((status,{'error':message}))
        handler.do_POST() if method=='POST' else handler._get()
        return results[0]

    def test_post_and_get_return_two_separate_safe_jobs_without_start(self):
        value={'title':'Animal pair','assets':cards()};value['assets'][0]['reference_id']='a'*32
        status,saved=self.dispatch('POST','/api/studio/scenes',value)
        self.assertEqual(status,201);self.assertEqual(len(saved['assets']),2)
        self.assertEqual(self.resolved,['a'*32]);self.assertEqual(self.executor.calls,[])
        status,loaded=self.dispatch('GET','/api/studio/scenes/'+saved['id'])
        self.assertEqual(status,200);self.assertEqual(loaded,saved)
        self.assertNotIn('original_path',json.dumps(saved));self.assertNotIn(str(self.jobs.root),json.dumps(saved))
        duplicate=self.dispatch('POST','/api/studio/scenes',value)[1]
        self.assertEqual(duplicate['id'],saved['id']);self.assertEqual(self.jobs.list()['total'],2)

    def test_no_token_unknown_reference_or_path_budget_source_injection(self):
        base={'title':'Pair','assets':cards()}
        self.assertEqual(self.dispatch('POST','/api/studio/scenes',base,token=None)[0],403)
        for field in ['max_calls','budget','source','reuse_model','api_key','reference']:
            value=deepcopy(base);value['assets'][1][field]='/tmp/unrelated'
            self.assertEqual(self.dispatch('POST','/api/studio/scenes',value)[0],400)
        value=deepcopy(base);value['assets'][1]['reference_id']='missing'
        self.assertEqual(self.dispatch('POST','/api/studio/scenes',value)[0],400)
        self.assertEqual(self.dispatch('POST','/api/studio/scenes',{**base,'start':True})[0],400)
        self.assertEqual(self.dispatch('POST','/api/studio/scenes',base,request_id=None)[0],400)
        self.assertEqual(self.jobs.list()['total'],0);self.assertEqual(self.executor.calls,[])

    def test_partial_invalid_pair_and_stale_request_id_do_not_duplicate(self):
        value={'title':'Pair','assets':cards()};value['assets'][1]['custom_actions']=['x'*601]
        self.assertEqual(self.dispatch('POST','/api/studio/scenes',value)[0],400)
        self.assertEqual(self.jobs.list()['total'],0)
        value['assets']=cards();self.assertEqual(self.dispatch('POST','/api/studio/scenes',value)[0],201)
        value['assets'][1]['prompt']='Different bird'
        self.assertEqual(self.dispatch('POST','/api/studio/scenes',value)[0],400)
        self.assertEqual(self.jobs.list()['total'],2)

    def test_finish_save_routes_saved_inputs_only_and_scene_list_is_read_only(self):
        from unittest.mock import patch
        value={'title':'Interrupted','assets':cards()}
        with patch.object(self.jobs,'create',side_effect=OSError('Analytic interrupted save')):
            self.assertEqual(self.dispatch('POST','/api/studio/scenes',value)[0],400)
        status,listing=self.dispatch('GET','/api/studio/scenes')
        self.assertEqual(status,200);scene=listing['scenes'][0]['id'];route='/api/studio/scenes/'+scene+'/finish-save'
        self.assertEqual(self.jobs.list()['total'],0)
        self.assertEqual(self.dispatch('POST',route,{'prompt':'Changed'})[0],400)
        self.assertEqual(self.dispatch('POST',route,{},token=None)[0],403)
        self.assertEqual(self.dispatch('POST',route,{})[0],200)
        self.assertEqual(self.dispatch('POST',route,{})[0],200)
        self.assertEqual(self.jobs.list()['total'],2);self.assertEqual(self.executor.calls,[])
