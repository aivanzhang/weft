"""Mock transport performance; no network or API credentials."""
from pathlib import Path
import hashlib
import json
import resource
import sys
import tempfile
import time
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import test_openai_stream as fixture
from avatar_assets import openai_stream as transport
resource.setrlimit(resource.RLIMIT_CPU,(30,30))
results=[]
with tempfile.TemporaryDirectory() as temporary:
 folder=Path(temporary).resolve()
 large=fixture.events();large[1]['delta']='';large[-1]['response']['output'][0]['content'][0]['text']='x'*(4*1024**2-2048)
 workloads=[('near_4m_event_64k_chunks',fixture.encode(large),65536,large[-1]['response']),
            ('near_4m_event_1k_chunks',fixture.encode(large),1024,large[-1]['response'])]
 small=[fixture.events()[0]]+[{'type':'response.output_text.delta','sequence_number':n,'delta':'x'} for n in range(1,transport.MAX_EVENTS-1)]+[{**fixture.events()[-1],'sequence_number':transport.MAX_EVENTS-1}]
 workloads.append(('65536_events_64k_chunks',fixture.encode(small),65536,fixture.completed()))
 for index,(name,raw,size,expected) in enumerate(workloads):
  response=fixture.FakeResponse(raw,chunks=[raw[n:n+size] for n in range(0,len(raw),size)])
  conn=fixture.FakeConnection(response);started=time.perf_counter()
  with patch.object(transport.http.client,'HTTPSConnection',return_value=conn):
   result=transport.post_stream_response({'stream':True,'store':False,'tools':[],'model':'synthetic'},60,fixture.KEY,trace_path=folder/f'{index}.json',client_request_id=fixture.CLIENT)
  elapsed=time.perf_counter()-started
  expected_raw=transport._canonical(expected);assert result==expected_raw
  trace=json.loads((folder/f'{index}.json').read_text());assert trace['status']=='completed'
  results.append({'workload':name,'wire_bytes':len(raw),'chunk_bytes':size,'events':trace['events'],'seconds':elapsed,'output_bytes':len(result),'output_sha256':hashlib.sha256(result).hexdigest(),'expected_output_sha256':hashlib.sha256(expected_raw).hexdigest(),'http_requests':len(conn.requests),'peak_rss_bytes_cumulative':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<512*1024**2
out=Path(__file__).with_name('performance.json');out.write_text(json.dumps({'classification':'Mocked transport and analytic payload performance, not provider latency','cpu_limit_seconds':30,'measured_peak_rss_acceptance_bytes':512*1024**2,'workloads':results},indent=2)+'\n');print(out.read_text())
