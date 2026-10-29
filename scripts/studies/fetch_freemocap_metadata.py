"""Bounded public metadata discovery; no package installation or archive download."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json


SOURCES={
    'test-release.json':'https://api.github.com/repos/freemocap/skellysamples/releases/tags/test_data_v06_09_25',
    'sample-release.json':'https://api.github.com/repos/freemocap/skellysamples/releases/tags/sample_data_v06_12_25',
    'README.md':'https://raw.githubusercontent.com/freemocap/skellysamples/main/README.md',
    'LICENSE.txt':'https://raw.githubusercontent.com/freemocap/skellysamples/main/LICENSE',
    'testing.html':'https://docs.freemocap.org/build/testing/',
    'conftest.py.txt':'https://raw.githubusercontent.com/freemocap/freemocap/main/freemocap/tests/pipelines/conftest.py',
}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    out=parser.parse_args().out.resolve();root=Path(__file__).resolve().parents[2]
    if out.parent!=root/'artifacts':raise ValueError('Use a new top-level artifacts directory')
    out.mkdir(exist_ok=False);shutil.copyfile(__file__,out/Path(__file__).name)
    write_json(out/'plan.json',{'created_at':local_time(),'sources':SOURCES,'maximum_requests':6,
        'maximum_bytes_per_response':500000,'maximum_workers':2,'timeout_seconds_per_request':15,
        'paid_calls':0,'no_installation':True,'no_archive_download':True,'no_upload':True,
        'purpose':'Inspect public calibrated movement sample availability and source/license statements'})
    started=time.perf_counter()
    def fetch(item):
        filename,url=item;result={'file':filename,'url':url,'retrieved_at':local_time()}
        try:
            request=Request(url,headers={'User-Agent':'Weft-local-source-research','Accept':'application/vnd.github+json' if filename.endswith('.json') else '*/*'})
            with urlopen(request,timeout=15) as response:
                raw=response.read(500001)
                if len(raw)>500000:raise ValueError('Public metadata exceeds its response bound')
                result.update(status=response.status,final_url=response.url,content_type=response.headers.get('Content-Type'),
                    etag=response.headers.get('ETag'))
            (out/filename).write_bytes(raw)
            result.update(bytes=len(raw),sha256=digest(out/filename))
        except (HTTPError,URLError,ValueError,TimeoutError) as error:
            result.update(error_type=type(error).__name__,error=str(error),bytes=0)
        return result
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(fetch,SOURCES.items()))
    assets={}
    for filename in ('test-release.json','sample-release.json'):
        if (out/filename).exists():
            data=json.loads((out/filename).read_text())
            assets[filename]=[{k:a.get(k) for k in ('id','name','size','digest','browser_download_url','content_type')} for a in data.get('assets',[])]
    report={'created_at':local_time(),'status':'metadata_complete' if all('error' not in r for r in results) else 'metadata_partial',
        'requests':results,'assets':assets,'bytes_received':sum(r['bytes'] for r in results),
        'runtime_seconds':time.perf_counter()-started,'paid_calls':0,'archives_downloaded':0,
        'license_scope':'Record repository and archive notices separately; do not infer data rights from software marketing.',
        'physical_validation':'Availability of calibration data does not establish its quality or measured anatomical truth.'}
    write_json(out/'report.json',report)
    print(json.dumps({'status':report['status'],'assets':assets,'requests':[{'file':r['file'],'bytes':r['bytes'],'error':r.get('error')} for r in results],
        'verification':seal_suite(out)}))


if __name__=='__main__':main()
