"""Acquire one byte-pinned public human-motion sample without decoding reserved video."""

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import zipfile

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json


ROOT=Path(__file__).resolve().parents[2]
METADATA=ROOT/'artifacts/freemocap-source-discovery-02'
ASSETS={
    'test':{'size':37864781,'sha256':'543a38c86e591476975a8c3d8dd05c5243777a769c696e116c6aaea43f5170b1',
        'metadata':'test-release.json','tag':'test_data_v06_09_25','file':'freemocap_test_data.zip','uncompressed_limit':90_000_000},
    'full':{'size':166958457,'sha256':'c23d6a24f9ed8cca74bccf171128543d661b47e5780df5b446d60063ac4acf4f',
        'metadata':'sample-release.json','tag':'sample_data_v06_12_25','file':'freemocap_sample_data.zip','uncompressed_limit':350_000_000},
}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--variant',choices=tuple(ASSETS),default='test')
    args=parser.parse_args();out=args.out.resolve();pin=ASSETS[args.variant]
    expected_size=pin['size'];expected_sha=pin['sha256']
    url=f'https://github.com/freemocap/skellysamples/releases/download/{pin["tag"]}/{pin["file"]}'
    if out.parent!=ROOT/'artifacts':raise ValueError('Use a new top-level artifacts directory')
    release_path=METADATA/pin['metadata']
    if digest(release_path)!=json.loads((METADATA/'manifest.json').read_text())['files'][pin['metadata']]:
        raise ValueError('Pinned release metadata changed')
    assets=json.loads(release_path.read_text())['assets']
    if len(assets)!=1:raise ValueError('Test release inventory changed')
    asset=assets[0]
    if (asset['size'],asset['digest'],asset['browser_download_url'])!=(expected_size,'sha256:'+expected_sha,url):
        raise ValueError('Public asset differs from the prepared byte-pinned request')
    out.mkdir(exist_ok=False);shutil.copyfile(__file__,out/Path(__file__).name)
    write_json(out/'pre-run.json',{'created_at':local_time(),'variant':args.variant,'url':url,'archive_bytes':expected_size,'archive_sha256':expected_sha,
        'release_metadata_sha256':digest(release_path),'maximum_download_seconds':120,'request_timeout_seconds':15,
        'maximum_archive_members':100,'maximum_uncompressed_bytes':pin['uncompressed_limit'],'maximum_extracted_files':15,
        'planned_local_storage_bytes_below':expected_size+pin['uncompressed_limit']+2_000_000,'automatic_retries':0,'paid_calls':0,
        'reserved_view_rule':'Keep the lexicographically last synchronized camera video pixel-unread until a separate comparison protocol is frozen.',
        'calibration_use':'Supplied camera parameters may be inspected for all views; no human-pose target or video pixel is decoded in acquisition.',
        'license_scope':'Repository AGPL-3.0 notice retained; no separate release-data license asserted. Local/private inspection, no redistribution.',
        'no_installation':True,'no_upload':True,'no_account':True,'no_agreement_acceptance':True})
    report={'created_at':local_time(),'status':'acquisition_started','paid_calls':0,'video_frames_decoded':0}
    started=time.perf_counter();received=0
    try:
        request=Request(url,headers={'User-Agent':'Weft-local-source-research','Accept-Encoding':'identity'})
        with urlopen(request,timeout=15) as response:
            parsed=urlsplit(response.url)
            if response.status!=200 or parsed.scheme!='https' or parsed.hostname not in {'github.com','release-assets.githubusercontent.com','objects.githubusercontent.com'}:
                raise ValueError('Unexpected public release response origin or status')
            reported=response.headers.get('Content-Length')
            if reported is not None and int(reported)!=expected_size:raise ValueError('Release content length changed')
            report['download_host']=parsed.hostname
            with (out/'source.zip').open('xb') as target:
                while True:
                    chunk=response.read(1024*1024)
                    if not chunk:break
                    received+=len(chunk)
                    if received>expected_size:raise ValueError('Archive exceeds the pinned byte bound')
                    target.write(chunk)
                    if time.perf_counter()-started>120:raise TimeoutError('Archive download exceeded its work bound')
        if received!=expected_size or digest(out/'source.zip')!=expected_sha:
            raise ValueError('Public archive bytes differ from the provider digest')
        report['download_seconds']=time.perf_counter()-started
        with zipfile.ZipFile(out/'source.zip') as archive:
            members=archive.infolist()
            if len(members)>100 or sum(m.file_size for m in members)>pin['uncompressed_limit']:
                raise ValueError('Archive inventory exceeds the extraction envelope')
            inventory=[]
            for member in members:
                path=PurePosixPath(member.filename)
                if path.is_absolute() or '..' in path.parts or '\\' in member.filename or member.flag_bits&1 or stat.S_ISLNK(member.external_attr>>16):
                    raise ValueError('Archive contains an unsafe or encrypted member')
                inventory.append({'name':member.filename,'bytes':member.file_size,'compressed_bytes':member.compress_size,'crc32':member.CRC})
            write_json(out/'inventory.json',inventory)
            videos=sorted([m for m in members if not m.is_dir() and '/synchronized_videos/' in '/'+m.filename and m.filename.endswith('.mp4')],key=lambda m:m.filename)
            if len(videos)!=3:raise ValueError('Expected exactly three synchronized original camera streams')
            small=[m for m in members if not m.is_dir() and PurePosixPath(m.filename).suffix.lower() in {'.json','.toml','.txt','.md','.csv'} and m.file_size<=2_000_000]
            selected=videos+small
            if len(selected)>15:raise ValueError('Metadata/member count exceeds the frozen envelope')
            roles={m.filename:('reserved_comparison' if i==2 else 'construction') for i,m in enumerate(videos)}
            write_json(out/'view-partition-before-pixels.json',{'created_at':local_time(),'video_frames_decoded':0,
                'roles':roles,'rule':'First two lexicographic synchronized views for construction; third reserved for comparison.',
                'motion_timing_verified':False,'camera_quality_verified':False,'physical_ground_verified':False})
            extracted=[]
            for index,member in enumerate(selected):
                if member.compress_type not in (0,8) or member.file_size>expected_size:raise ValueError('Member exceeds codec/size bounds')
                name=f'member-{index:02d}-'+re.sub(r'[^A-Za-z0-9_.-]','_',PurePosixPath(member.filename).name)
                with archive.open(member) as source,(out/name).open('xb') as target:
                    count=0
                    while True:
                        chunk=source.read(1024*1024)
                        if not chunk:break
                        count+=len(chunk)
                        if count>member.file_size:raise ValueError('Extracted member exceeds its declared size')
                        target.write(chunk)
                if count!=member.file_size:raise ValueError('Incomplete archive member')
                extracted.append({'member':member.filename,'file':name,'bytes':count,'sha256':digest(out/name),
                    'role':roles.get(member.filename,'metadata'),'video_frames_decoded':0 if member in videos else None})
        shutil.copyfile(METADATA/'LICENSE.txt',out/'repository-LICENSE.txt')
        report.update(status='public_sample_acquired_uninspected',variant=args.variant,archive_sha256=expected_sha,extracted=extracted,
            source_release_url=json.loads(release_path.read_text())['html_url'],
            docs_timing_claim='Test footage is described as downsampled. Correspondence of the advertised full sample and physical capture timing require separate verification.',
            physical_validation='Unverified supplied calibration; no measured anatomy, contact or muscle-force claim.')
    except Exception as error:
        report.update(status='acquisition_failed_preserved',error_type=type(error).__name__,error=str(error))
    report.update(bytes_received=received,runtime_seconds=time.perf_counter()-started)
    write_json(out/'report.json',report)
    print(json.dumps({'status':report['status'],'bytes_received':received,'extracted':report.get('extracted'),
        'error':report.get('error'),'verification':seal_suite(out)}))


if __name__=='__main__':main()
