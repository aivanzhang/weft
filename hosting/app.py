"""Private container endpoint; the Worker owns authentication, quotas, and durable files."""
import asyncio
import base64
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import zipfile

import anyio
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

ROOT = Path(__file__).resolve().parents[1]
lock = asyncio.Lock()


async def generate(request):
    if lock.locked(): return JSONResponse({'error':'busy'},status_code=409)
    async with lock:
        temp = tempfile.TemporaryDirectory(prefix='weft-job-')
        directory = Path(temp.name)
        try:
            size = 0
            async with asyncio.timeout(45):
                async with await anyio.open_file(directory/'upload.mp4','xb') as target:
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size>10*1024**2: raise ValueError('upload_too_large')
                        await target.write(chunk)
            env = {'PATH':'/usr/local/bin:/usr/bin:/bin','PYTHONPATH':str(ROOT/'src'),
                   'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1',
                   'OPENCV_FFMPEG_CAPTURE_OPTIONS':'protocol_whitelist;file'}
            with (directory/'process.log').open('wb') as log:
                process = await asyncio.create_subprocess_exec(sys.executable,str(ROOT/'hosting/job.py'),str(directory),
                    cwd=directory,env=env,stdout=log,stderr=log,start_new_session=True)
                try:
                    await asyncio.wait_for(process.wait(),timeout=180)
                except (TimeoutError,asyncio.CancelledError):
                    if process.returncode is None: os.killpg(process.pid,signal.SIGKILL)
                    await process.wait()
                    raise
            success = process.returncode==0 and (directory/'result.json').is_file()
            if not success:
                def archive_failure():
                    with zipfile.ZipFile(directory/'environment.zip','x',zipfile.ZIP_DEFLATED) as archive:
                        for path in directory.rglob('*'):
                            if path.is_file() and path.name!='environment.zip': archive.write(path,str(path.relative_to(directory)))
                await run_in_threadpool(archive_failure)
            archive = directory/'environment.zip'
            if archive.stat().st_size>32*1024**2: raise ValueError('result_too_large')
            headers = {'X-Weft-Status':'complete' if success else 'failed'}
            if success:
                headers['X-Weft-Result'] = base64.b64encode((directory/'result.json').read_bytes()).decode()
            return FileResponse(archive,media_type='application/zip',headers=headers,background=BackgroundTask(temp.cleanup))
        except (ValueError,TimeoutError,OSError):
            temp.cleanup()
            return JSONResponse({'error':'generation_failed'},status_code=422)
        except BaseException:
            temp.cleanup()
            raise


async def health(request):
    return JSONResponse({'status':'ok'})


app = Starlette(routes=[Route('/generate',generate,methods=['POST']),Route('/health',health)])
