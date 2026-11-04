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
# One image serves both fixed pipelines; the request only selects which, never a command.
# The exercise lane decodes and poses a human clip, so it needs a larger upload and a
# longer deadline than the ball lane, whose limits would kill it mid-pose.
PIPELINES = {
    "ball": {"job": "hosting/job.py", "upload": 10 * 1024**2, "read": 45, "deadline": 180},
    "exercise": {"job": "hosting/exercise_job.py", "upload": 40 * 1024**2, "read": 120,
                 "deadline": 400},
}
lock = asyncio.Lock()


async def generate(request):
    pipeline = PIPELINES.get(request.headers.get("x-weft-pipeline", "ball"))
    if pipeline is None: return JSONResponse({'error':'unknown_pipeline'},status_code=400)
    if lock.locked(): return JSONResponse({'error':'busy'},status_code=409)
    async with lock:
        temp = tempfile.TemporaryDirectory(prefix='weft-job-')
        directory = Path(temp.name)
        try:
            size = 0
            async with asyncio.timeout(pipeline['read']):
                async with await anyio.open_file(directory/'upload.mp4','xb') as target:
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size>pipeline['upload']: raise ValueError('upload_too_large')
                        await target.write(chunk)
            env = {'PATH':'/usr/local/bin:/usr/bin:/bin','PYTHONPATH':str(ROOT/'src'),
                   'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1',
                   'OPENCV_FFMPEG_CAPTURE_OPTIONS':'protocol_whitelist;file',
                   'WEFT_POSE_MODEL':os.environ.get('WEFT_POSE_MODEL','/app/models/pose_landmarker_lite.task')}
            with (directory/'process.log').open('wb') as log:
                process = await asyncio.create_subprocess_exec(sys.executable,str(ROOT/pipeline['job']),str(directory),
                    cwd=directory,env=env,stdout=log,stderr=log,start_new_session=True)
                try:
                    await asyncio.wait_for(process.wait(),timeout=pipeline['deadline'])
                except (TimeoutError,asyncio.CancelledError):
                    if process.returncode is None: os.killpg(process.pid,signal.SIGKILL)
                    await process.wait()
                    raise
            success = process.returncode==0 and (directory/'result.json').is_file()
            if not success:
                def archive_failure():
                    # Diagnostics only, and bounded: a failed exercise lab writes far more than
                    # the response cap, so only the logs and small stage records come back.
                    with zipfile.ZipFile(directory/'environment.zip','x',zipfile.ZIP_DEFLATED) as archive:
                        for path in sorted(directory.rglob('*')):
                            if not path.is_file() or path.name=='environment.zip': continue
                            if path.suffix not in ('.log','.json','.txt','.xml'): continue
                            if path.stat().st_size>2*1024**2: continue
                            archive.write(path,str(path.relative_to(directory)))
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
    return JSONResponse({'status':'ok','pipelines':sorted(PIPELINES)})


app = Starlette(routes=[Route('/generate',generate,methods=['POST']),Route('/health',health)])
