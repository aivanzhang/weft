"""Private, crash-conservative queue; request handlers never run the avatar pipeline."""

from contextlib import contextmanager
from datetime import datetime
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import time
import uuid
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
MAX_JOBS = 256
MAX_QUEUED = 4
MAX_INVOCATIONS = 8
MAX_JSON = 1024**2
HEARTBEAT_STALE = 15
STORED_BYTES_LIMIT = 8*1024**3
ACTIVE_RESERVATION_BYTES = 2*1024**3+64*1024**2
TERMINAL = {'complete', 'unsupported', 'failed', 'needs_inspection'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _private_directory(path):
    path = Path(path)
    _require(not any(p.is_symlink() for p in (path, *path.parents)), 'Studio state cannot contain symlinks')
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.stat()
    _require(info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700,
             'Studio state requires current-user ownership and mode 0700')
    return path.resolve()


def _write_private(path, value, *, replace=False):
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp') if replace else path
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(temporary, path)


def _json(path, default=None):
    if not path.exists():
        return default
    _require(not any(p.is_symlink() for p in (path,*path.parents)) and path.is_file() and path.stat().st_size <= MAX_JSON,
             'Studio JSON file is unsafe or oversized')
    return json.loads(path.read_text(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def _safe_text(value, limit=400):
    if not isinstance(value, str):
        return ''
    value = re.sub(r'msy_[A-Za-z0-9_-]+|\bsk-[A-Za-z0-9_-]+', '[redacted]', value)
    value = re.sub(r'(?i)bearer\s+\S+', '[redacted]', value)
    value = re.sub(r'https?://\S+', '[external URL omitted]', value)
    value = re.sub(r'(?:/Users/|/home/|/private/|/tmp/)[^\s\"\x27]+', '[local path]', value)
    return ''.join(c for c in value if c in '\n\t' or ord(c) >= 32)[:limit]


def _local_time(value):
    return datetime.fromtimestamp(value, ZoneInfo('America/Los_Angeles')).isoformat() if value else None


def _stage_progress(folder, job_status):
    events = folder/'events'
    if not events.is_dir():
        return []
    _require(not events.is_symlink(), 'Pipeline event directory cannot be a symlink')
    paths=[]
    for index,path in enumerate(events.iterdir()):
        _require(index<128, 'Pipeline event inventory exceeds its bound')
        if path.suffix=='.json':paths.append(path)
    _require(sum(path.stat().st_size for path in paths)<=MAX_JSON, 'Pipeline event bytes exceed snapshot bound')
    stages={};last_key=None
    for path in sorted(paths):
        event=_json(path,{})
        stage=_safe_text(event.get('stage'),80)
        if not stage:continue
        attempt=event.get('attempt')
        _require(attempt is None or type(attempt) is int and 0<=attempt<=2, 'Invalid stage attempt')
        key=(stage,attempt);last_key=key
        label={'deformation_gate':'Check skin stretching',
               'portable_attachment':'Check portable fur attachment'}.get(stage,stage.replace('_',' ').title())
        item={'id':stage if attempt is None else f'{stage}:{attempt}', 'stage':stage,
              'label':label if attempt is None else f'{label} · attempt {attempt+1}',
              'status':_safe_text(event.get('status'),80),
              'detail':_safe_text(event.get('reason',event.get('error')),500), 'attempt':attempt}
        elapsed=event.get('elapsed_seconds')
        if type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>=0:
            item['elapsed_seconds']=elapsed
        stages[key]=item
    _require(len(stages)<=40, 'Too many distinct progress stages')
    for key,item in stages.items():
        if item['status']=='running' and (key!=last_key or job_status not in
                ('queued','launching','planning','running','cancel_requested')):
            # Legacy or interrupted events cannot establish completion or live ownership.
            item.update(status='unresolved',recorded_status='running',
                        detail='No terminal stage receipt; last recorded as running. '+item['detail'])
    return list(stages.values())


def process_identity(pid):
    """Read a bounded OS start identity; never equate an existing PID with ownership."""
    _require(type(pid) is int and pid > 0, 'Invalid worker PID')
    if sys.platform.startswith('linux'):
        value = Path(f'/proc/{pid}/stat').read_text()
        return 'linux-start-ticks:'+value[value.rfind(')')+2:].split()[19]
    if sys.platform=='darwin':
        # SDK proc_bsdinfo has48 bytes of IDs,48 of names,24 of flags, then two uint64 times.
        library=ctypes.CDLL('/usr/lib/libproc.dylib',use_errno=True)
        library.proc_pidinfo.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_int]
        library.proc_pidinfo.restype=ctypes.c_int
        buffer=ctypes.create_string_buffer(136)
        size=library.proc_pidinfo(pid,3,0,buffer,len(buffer))
        if size!=len(buffer):raise OSError(ctypes.get_errno(),'Worker start identity unavailable')
        values=buffer.raw
        _require(int.from_bytes(values[12:16],sys.byteorder)==pid and
                 int.from_bytes(values[20:24],sys.byteorder)==os.getuid(),'Worker identity is not owned by this user')
        second=int.from_bytes(values[120:128],sys.byteorder)
        microsecond=int.from_bytes(values[128:136],sys.byteorder)
        _require(second>0 and 0<=microsecond<1000000,'Invalid worker start identity')
        return f'darwin-start:{second}:{microsecond}'
    result = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'lstart='], capture_output=True,
                            timeout=1, check=False, env={'PATH':'/usr/bin:/bin', 'LC_ALL':'C'})
    _require(result.returncode == 0 and 0 < len(result.stdout) <= 128,
             'Worker start identity is unavailable; inspect preserved owner before resuming')
    return 'os-start:'+result.stdout.decode('ascii').strip()


class _DetachedExecutor:
    provenance = 'actual_pipeline'

    def __init__(self):
        self.children = []

    def launch(self, invocation):
        self.children = [child for child in self.children if child.poll() is None]
        env = {key:os.environ[key] for key in ('HOME','USER','LOGNAME','TMPDIR','PATH','LANG') if key in os.environ}
        env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', PYTHONUNBUFFERED='1')
        command = [sys.executable, '-m', 'avatar_assets.studio_worker', '--root', invocation['root'],
                   '--state-dir', invocation['state_dir'], '--job', invocation['job_id'], '--token', invocation['token']]
        log = Path(invocation['invocation_dir'])/'supervisor.log'
        fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            process = subprocess.Popen(command, cwd=invocation['root'], env=env, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        self.children.append(process)
        return {'pid':process.pid, 'start_identity':process_identity(process.pid), 'provenance':self.provenance}


class StudioJobs:
    def __init__(self, root=ROOT, state_dir=None, executor=None):
        self.root = Path(root).resolve()
        proposed = Path(state_dir) if state_dir is not None else self.root/'.local/studio'
        _require(proposed.resolve().is_relative_to(self.root/'.local'), 'Studio state must stay beneath tool .local/')
        self.state_dir = _private_directory(proposed)
        binding = self.root/'.local/studio-registry.json'
        if not binding.exists():
            try:
                _write_private(binding,{'schema_version':1,'state_dir':str(self.state_dir)})
            except FileExistsError:
                pass
        _require(_json(binding)=={'schema_version':1,'state_dir':str(self.state_dir)},
                 'This tool root already owns a different canonical studio registry')
        _require(binding.stat().st_uid==os.getuid() and stat.S_IMODE(binding.stat().st_mode)==0o600,
                 'Studio registry binding must be owner-only')
        self.jobs_dir = _private_directory(self.state_dir/'jobs')
        self.executor = executor or _DetachedExecutor()
        _require(callable(getattr(self.executor, 'launch', None)), 'Studio executor requires launch(invocation)')
        self.db_path = self.state_dir/'jobs.sqlite3'
        if not self.db_path.exists():
            try:
                os.close(os.open(self.db_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600))
            except FileExistsError:
                pass
        info = self.db_path.lstat()
        _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o600,
                 'Studio database must be an owner-only regular file')
        with self._db() as db:
            db.execute('PRAGMA max_page_count=8192')
            db.executescript('''CREATE TABLE IF NOT EXISTS jobs(
                id TEXT PRIMARY KEY, idempotency TEXT UNIQUE, request_digest TEXT NOT NULL,
                request TEXT NOT NULL, output TEXT UNIQUE NOT NULL, created REAL NOT NULL,
                status TEXT NOT NULL, queued_at REAL, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS active(id INTEGER PRIMARY KEY CHECK(id=1), job TEXT, token TEXT);
                CREATE TABLE IF NOT EXISTS journal(seq INTEGER PRIMARY KEY, job TEXT, event TEXT NOT NULL);''')
        self._dispatch()  # Only explicitly queued, never previously running/uncertain, work can resume.

    @contextmanager
    def _db(self, transaction=False):
        db = sqlite3.connect(self.db_path, timeout=.5)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA synchronous=FULL')
        try:
            if transaction:
                db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _row(db, job_id):
        _require(isinstance(job_id, str) and re.fullmatch(r'[a-f0-9]{24}', job_id), 'Invalid studio job ID')
        row = db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
        _require(row is not None, 'Unknown studio job')
        return dict(row)

    @staticmethod
    def _update(db, row, data, status=None):
        data['updated'] = time.time()
        db.execute('UPDATE jobs SET status=?,queued_at=?,data=? WHERE id=?',
                   (status or row['status'], data.get('queued_at'), _canonical(data), row['id']))

    @staticmethod
    def _journal(db, job, event):
        _require(db.execute('SELECT COUNT(*) FROM journal WHERE job=?',(job,)).fetchone()[0] < 128,
                 'Studio lifecycle event limit reached')
        db.execute('INSERT INTO journal(job,event) VALUES(?,?)', (job, _canonical({'time':time.time(), **event})))

    def _selected(self, value, reference):
        if value is None:
            return None
        path = Path(value)
        _require(not any(p.is_symlink() for p in (path, *path.parents)), 'Selected input cannot contain symlinks')
        path = path.resolve()
        _require(any(path.is_relative_to(base) for base in
                     (self.root/'outputs', self.root/'references', self.state_dir/'uploads')),
                 'Selected input must come from approved tool outputs, references or studio uploads')
        _require(path.is_file() and path.suffix.lower() in
                 (('.png','.jpg','.jpeg','.webp') if reference else ('.glb',)), 'Unsupported selected input')
        info = path.stat()
        _require(0 < info.st_size <= (8 if reference else 64)*1024**2, 'Selected input exceeds size bound')
        return {'path':str(path), 'bytes':info.st_size, 'mtime_ns':info.st_mtime_ns,
                'device':info.st_dev, 'inode':info.st_ino}

    def create(self, prompt, style='', custom_actions=(), preset='preview', reference=None, reuse_model=None,
               *, idempotency_key=None):
        _require(isinstance(prompt, str) and 1 <= len(prompt.strip()) and len(prompt) <= 4000, 'Prompt must be1–4000 characters')
        _require(isinstance(style, str) and len(style) <= 800, 'Style must be at most800 characters')
        _require(isinstance(custom_actions, (list,tuple)) and len(custom_actions) <= 8
                 and all(isinstance(a,str) and a.strip() and len(a) <= 600 for a in custom_actions),
                 'Custom actions require at most eight nonempty descriptions of at most600 characters')
        _require(len(set(custom_actions)) == len(custom_actions), 'Duplicate custom action')
        _require(preset in ('draft','preview','muse'), 'Unknown studio preset')
        if idempotency_key is not None:
            _require(isinstance(idempotency_key,str) and re.fullmatch(r'[A-Za-z0-9_-]{1,100}',idempotency_key),
                     'Invalid idempotency key')
        request = {'prompt':prompt, 'style':style, 'custom_actions':list(custom_actions), 'preset':preset,
                   'reference':self._selected(reference,True), 'reuse_model':self._selected(reuse_model,False)}
        checksum = _digest(request)
        with self._db(True) as db:
            existing = db.execute('SELECT * FROM jobs WHERE idempotency=?',(idempotency_key,)).fetchone() if idempotency_key else None
            if existing:
                _require(existing['request_digest'] == checksum, 'Idempotency key already belongs to a different immutable request')
                job_id = existing['id']
            else:
                _require(db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] < MAX_JOBS, 'Studio retains at most256 jobs; preserve/export history before creating more')
                job_id = uuid.uuid4().hex[:24]
                folder = _private_directory(self.jobs_dir/job_id)
                _write_private(folder/'request.json', request)
                created = time.time()
                data = {'updated':created,'operation':None,'invocations':0,'cancel_requested':False,
                        'provenance':getattr(self.executor,'provenance','analytic_fixture')}
                db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?)',
                    (job_id,idempotency_key,checksum,_canonical(request),'studio-'+job_id,created,'created',None,_canonical(data)))
                self._journal(db,job_id,{'status':'created','request_digest':checksum})
        return self.get(job_id)

    def resolve_job(self, job_id):
        with self._db() as db:
            row = self._row(db,job_id)
        path = self.root/'outputs'/row['output']
        _require(not any(p.is_symlink() for p in (path, *path.parents)), 'Studio output cannot contain symlinks')
        return path

    def _plan_id(self, row):
        folder = self.root/'outputs'/row['output']
        if not (folder/'plan.json').is_file():
            return None
        return _digest({'request':row['request_digest'], 'plan':_json(folder/'plan.json'),
                        'generation':_json(folder/'generation/plan.json')})

    def _authorization(self, row):
        request = json.loads(row['request'])
        result = {'can_start':False,'estimated_credits':0 if request['reuse_model'] else 35,
                  'reason':'Plan the immutable request first.', 'execution_gate':'canonical_ledger_and_explicit_start'}
        if not self._plan_id(row):
            return result
        if request['reuse_model']:
            return {**result,'can_start':True,'reason':'Explicit source reuse requires no new Meshy credits; existing bounded agent/local execution still applies.'}
        generation = self.root/'outputs'/row['output']/'generation'
        plan = _json(generation/'plan.json')
        if not isinstance(plan,dict) or plan.get('max_credits') != 35 or plan.get('estimated_credits') != 35:
            return {**result,'reason':'The prepared generation plan must retain the expected35-credit ceiling.'}
        ledger = self.root.parents[1]/'.local/meshy/ledger.sqlite3'
        if not ledger.is_file() or any(p.is_symlink() for p in (ledger,*ledger.parents)):
            return {**result,'reason':'No existing canonical Meshy allowance is available. This UI never creates or increases an allowance.'}
        info = ledger.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            return {**result,'reason':'Canonical ledger permissions need local inspection.'}
        try:
            db = sqlite3.connect(ledger.as_uri()+'?mode=ro',uri=True,timeout=.2)
            try:
                _require(db.execute('SELECT COUNT(*) FROM calls').fetchone()[0] <= 10000, 'Ledger read bound exceeded')
                budget = db.execute('SELECT credit_limit FROM budget WHERE id=1').fetchone()
                used = dict(db.execute('SELECT job,SUM(credits) FROM calls GROUP BY job'))
                grants = {r[0]:(r[1],r[2]) for r in db.execute('SELECT job,credits,plan_hash FROM budget_grants')}
                bound = db.execute('SELECT plan_hash,credit_limit,output FROM jobs WHERE id=?',(plan['id'],)).fetchone()
                states = list(db.execute('SELECT state,task_id FROM calls WHERE job=?',(plan['id'],)))
            finally:
                db.close()
        except (sqlite3.Error, ValueError, KeyError):
            return {**result,'reason':'The canonical allowance could not be verified read-only; inspect it locally before execution.'}
        if budget is None:
            return {**result,'reason':'Canonical base allowance has not been explicitly configured.'}
        if (type(budget[0]) is not int or not 1<=budget[0]<=10000
                or any(type(value) is not int or value<0 for value in used.values())
                or any(type(value[0]) is not int or not 1<=value[0]<=500 for value in grants.values())):
            return {**result,'reason':'Canonical allowance contains invalid numeric records; inspect it locally.'}
        digest = _digest(plan)
        if ((bound and (bound[0] != digest or bound[1] != 35 or Path(bound[2]) != generation))
                or (plan['id'] in grants and grants[plan['id']][1] != digest)):
            return {**result,'reason':'Canonical authorization belongs to a different immutable plan; no execution allowed.'}
        if any(state in ('submitting','uncertain','FAILED','CANCELED') or not task for state,task in states):
            return {**result,'reason':'A preserved provider attempt failed or has an uncertain charge; inspect it rather than submitting again.'}
        base_used = sum(max(0,credits-grants.get(job,(0,None))[0]) for job,credits in used.items())
        consumed, grant = used.get(plan['id'],0), grants.get(plan['id'],(0,None))[0]
        required = max(0,35-consumed)
        additional_base = max(0,consumed+required-grant)-max(0,consumed-grant)
        remaining = max(0,budget[0]-base_used)
        allowed = consumed <= 35 and additional_base <= remaining
        return {**result,'can_start':allowed,'remaining_base_credits':remaining,'remaining_job_credits':required,
                'reason':('Existing canonical allowance covers this exact job; Start requests execution without a grant or purchase.' if allowed else
                          f'Existing canonical allowance is insufficient: this job needs {required} remaining credits and {additional_base} base credits; {remaining} base credits remain. No allowance was added.')}

    def _enqueue(self, job_id, operation, plan_id=None):
        with self._db(True) as db:
            row = self._row(db,job_id); data = json.loads(row['data'])
            current_plan = self._plan_id(row)
            if plan_id is not None:
                _require(isinstance(plan_id,str) and plan_id == current_plan, 'Stale or invalid immutable plan ID')
            if row['status'] in ('queued','launching','planning','running','cancel_requested'):
                _require(data['operation'] == operation, 'Another operation already owns this job')
            elif row['status'] == 'complete' or operation == 'plan' and current_plan and row['status'] in ('planned','awaiting_authorization'):
                pass
            else:
                _require(row['status'] not in TERMINAL, 'This terminal or uncertain job cannot be automatically replayed; inspect preserved evidence')
                if operation == 'execute':
                    authorization = self._authorization(row)
                    _require(authorization['can_start'], authorization['reason'])
                _require(db.execute("SELECT COUNT(*) FROM jobs WHERE status='queued'").fetchone()[0] < MAX_QUEUED,
                         'Studio queue is full (four queued jobs)')
                _require(data['invocations'] < MAX_INVOCATIONS,'Studio invocation bound reached')
                data.update(operation=operation,queued_at=time.time(),cancel_requested=False,error=None,
                            expected_plan_id=current_plan if operation=='execute' else None)
                self._update(db,row,data,'queued');self._journal(db,job_id,{'status':'queued','operation':operation})
        self._dispatch()
        return self.get(job_id)

    def plan(self, job_id):
        return self._enqueue(job_id,'plan')

    def start(self, job_id, plan_id=None):
        return self._enqueue(job_id,'execute',plan_id)

    def release_inspected(self,job_id,*,expected_pid,expected_identity,reason):
        """Seal exact failed-owner custody and release its lease; never restart or dispatch work."""
        from avatar_assets.studio_inspection import (collect,custody_locks,recheck,collect_generation,
            recheck_generation,include_previous_plans,collect_api_planning,recheck_api_planning)
        _require(type(expected_pid) is int and expected_pid>0 and isinstance(expected_identity,str)
                 and 0<len(expected_identity)<=256,'Explicit prior process identity is required')
        reason=_safe_text(reason,800)
        _require(bool(reason.strip()),'Inspection release requires a recorded reason')

        def owned(db):
            row=self._row(db,job_id);data=json.loads(row['data']);request=json.loads(row['request'])
            active=db.execute('SELECT job,token FROM active WHERE id=1').fetchone()
            _require(row['status']=='needs_inspection','Only a retained needs_inspection job can be released')
            _require(data.get('owner_pid')==expected_pid and data.get('owner_identity')==expected_identity,
                     'Retained worker identity changed')
            _require(active and tuple(active)==(job_id,data.get('token')),'Exact old lease is no longer held')
            if request.get('reuse_model') is None:
                _require((self.root/'outputs'/row['output']/'generation').is_dir(),
                         'Remote-generation custody requires its preserved generation folder')
            _require(_digest(request)==row['request_digest'] and _json(self.jobs_dir/job_id/'request.json')==request,
                     'Immutable Studio request receipt changed')
            _require(not db.execute("SELECT 1 FROM jobs WHERE status='queued' LIMIT 1").fetchone(),
                     'Queued work must be resolved separately before inspection release')
            return row,data,request

        with self._db() as db:row,data,request=owned(db)
        folder=self.root/'outputs'/row['output']
        fresh=request.get('reuse_model') is None
        prepared=_json(folder/'request.json',{})
        runner=_json(folder/'agents/runner.json',{})
        api_planning=not fresh and (isinstance(prepared,dict) and 'agent_backend' in prepared
                      or isinstance(runner,dict) and runner.get('execution_kind')=='openai_responses_api')
        if not fresh:
            _require(not (folder/'generation').exists(),'Remote-generation artifacts need separate inspection')
        invocation=self.state_dir/data['invocation_dir']
        _require(invocation.resolve()==invocation and invocation.is_relative_to(self.jobs_dir/job_id),
                 'Retained invocation path is unsafe')
        owner=_json(invocation/'owner.json')
        _require(isinstance(owner,dict) and owner.get('token')==data.get('token'),'Supervisor receipt lease differs')
        meta=_json(invocation/'invocation.json')
        _require(isinstance(meta,dict) and meta.get('job_id')==job_id and meta.get('request_digest')==row['request_digest'],
                 'Invocation request binding changed')
        locks=[self.state_dir/'worker.lock',self.jobs_dir/job_id/'worker.lock',folder/'pipeline.lock']
        if fresh:
            locks.append(folder/'generation/job.lock')
        if api_planning:
            locks.append(folder/'agents/runner.lock')
        with custody_locks(locks):
            proof=collect(self.root,folder,invocation,expected_pid,expected_identity,
                          **({'operation':'plan'} if api_planning else {}))
            proof['locks']=[str(path.relative_to(self.root)) for path in locks]
            if fresh:
                proof=include_previous_plans(self.root,invocation,proof,row['request_digest'])
                proof['generation']=collect_generation(self.root,folder,request)
                proof['scope']='Exited local owners plus separately bound successful remote downloads; no replay or acceptance.'
            if api_planning:
                _require(not fresh,'API fresh-source release requires separate remote-task inspection')
                proof=include_previous_plans(self.root,invocation,proof,row['request_digest'])
                proof['api_planning']=collect_api_planning(self.root,folder,request,self.jobs_dir/job_id/'pipeline-request.json')
                proof['scope']='Exited local owners plus one settled API planner; rejected semantics preserved, no replay or acceptance.'
            with self._db(True) as db:
                current,current_data,current_request=owned(db)
                _require(current['request_digest']==row['request_digest'] and current_request==request
                         and current_data.get('invocation_dir')==data['invocation_dir'],'Inspected request/invocation changed')
                proof['final_process_check']=recheck(self.root,proof)
                if fresh:
                    recheck_generation(proof['generation'])
                if api_planning:
                    recheck_api_planning(proof['api_planning'])
                record={'schema_version':1,'job_id':job_id,'previous_status':'needs_inspection',
                        'requested_transition':'failed_and_release_exact_lease',
                        'expected_pid':expected_pid,'expected_identity':expected_identity,'reason':reason,
                        'request_digest':row['request_digest'],'recorded_local':_local_time(time.time()),
                        'no_dispatch':True,'no_cost_reset':True,'proof':proof}
                folder=self.jobs_dir/job_id
                _require(sum(1 for _ in folder.glob('inspection-release-*.json'))<4,'Inspection receipt bound reached')
                path=folder/('inspection-release-'+uuid.uuid4().hex+'.json')
                _write_private(path,record)
                checksum=hashlib.sha256(path.read_bytes()).hexdigest()
                relative=str(path.relative_to(self.state_dir))
                current_data.update(inspection_release={'path':relative,'sha256':checksum},
                                    error=current_data.get('error') or 'Failed job released after explicit custody inspection.')
                self._update(db,current,current_data,'failed')
                self._journal(db,job_id,{'status':'failed','action':'inspection_release',
                                        'receipt':relative,'sha256':checksum,'no_dispatch':True,'no_cost_reset':True})
                db.execute('DELETE FROM active WHERE id=1 AND job=? AND token=?',(job_id,data['token']))
        return {'status':'failed','job_id':job_id,'lease_released':True,'receipt':relative,
                'sha256':checksum,'no_dispatch':True,'no_cost_reset':True}

    def adopt_cached_planner_revision(self,job_id,revision,*,expected_revision_sha256):
        """Explicit local adoption after custody release; Start remains a separate action."""
        from avatar_assets.pipeline import _pipeline_sources
        from avatar_assets.pipeline_recovery import verify_runtime_revision
        from avatar_assets.studio_inspection import custody_locks
        from avatar_assets.safety import sha256,write_json
        revision=Path(revision)
        _require(revision.resolve()==revision and revision.parent==self.root/'outputs'
                 and sha256(revision/'manifest.json')==expected_revision_sha256,'Reviewed runtime revision changed')

        def released(db):
            row=self._row(db,job_id);data=json.loads(row['data'])
            _require(row['status']=='failed' and not db.execute('SELECT 1 FROM active').fetchone()
                     and not db.execute("SELECT 1 FROM jobs WHERE status='queued'").fetchone(),
                     'Cached recovery requires an explicitly released failed job and a free worker lane')
            receipt=data.get('inspection_release')
            _require(isinstance(receipt,dict),'Failed job lacks a custody release')
            path=self.state_dir/receipt['path']
            _require(path.resolve()==path and path.is_relative_to(self.jobs_dir/job_id)
                     and sha256(path)==receipt['sha256'],'Failed-job release receipt changed')
            proof=_json(path)
            _require(proof.get('job_id')==job_id and proof.get('request_digest')==row['request_digest']
                     and proof.get('requested_transition')=='failed_and_release_exact_lease'
                     and proof.get('proof',{}).get('api_planning',{}).get('policy')=='completed-api-planning-custody-v1',
                     'This job has no settled API planner custody release')
            return row,data

        with self._db() as db:row,data=released(db)
        output=self.root/'outputs'/row['output']
        locks=[self.state_dir/'worker.lock',self.jobs_dir/job_id/'worker.lock',output/'pipeline.lock',output/'agents/runner.lock']
        with custody_locks(locks),self._db(True) as db:
            current,current_data=released(db)
            _require(current['request_digest']==row['request_digest'] and current_data==data,'Failed job changed during local review')
            verified=verify_runtime_revision(output,_pipeline_sources(),revision=revision)
            _require(not (output/'runtime-revision.json').exists() and not (output/'plan.json').exists(),
                     'Runtime adoption is append-only; no overwrite or automatic repeat')
            pointer={'schema_version':1,'path':str(revision.relative_to(self.root)),'sha256':expected_revision_sha256}
            write_json(output/'runtime-revision.json',pointer)
            write_json(output/'plan.json',verified['plan'])
            current_data.update(runtime_revision=pointer,error=None,plan_id=self._plan_id(current))
            self._update(db,current,current_data,'planned')
            self._journal(db,job_id,{'status':'planned','action':'cached_planner_runtime_revision',
                                    'revision':pointer,'new_planner_calls':0,'no_dispatch':True,'no_cost_reset':True})
        return {'status':'planned','job_id':job_id,'plan_id':self._plan_id(current),'revision':pointer,
                'new_planner_calls':0,'no_dispatch':True,'no_cost_reset':True}

    def _dispatch(self):
        with self._db(True) as db:
            if db.execute('SELECT job FROM active WHERE id=1').fetchone():
                return
            found = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY queued_at,id LIMIT 1").fetchone()
            if not found:
                return
            row = dict(found);data = json.loads(row['data']);token = uuid.uuid4().hex
            measurements = [json.loads(item[0]).get('retained_bytes',0) for item in db.execute('SELECT data FROM jobs')]
            _require(all(type(value) is int and value>=0 for value in measurements),'Invalid cached studio storage measurement')
            measured = sum(measurements)
            # Accounting is updated by the detached supervisor, never by request-thread tree scans.
            if measured+ACTIVE_RESERVATION_BYTES>STORED_BYTES_LIMIT:
                data['error']='Studio retained-output capacity is full;8GiB total with a2GiB pipeline plus64MiB controller reservation is required. Nothing was deleted; inspect/archive history before continuing.'
                self._update(db,row,data)
                return
            data.update(invocations=data['invocations']+1,token=token,started=time.time(),heartbeat=time.time(),
                        ended=None,owner_pid=None,owner_identity=None,queued_at=None)
            invocation = _private_directory(self.jobs_dir/row['id']/f"run-{data['invocations']:04d}")
            payload = {'schema_version':1,'root':str(self.root),'state_dir':str(self.state_dir),
                       'job_id':row['id'],'token':token,'operation':data['operation'],
                       'invocation_dir':str(invocation),'request_digest':row['request_digest']}
            snapshots=_private_directory(invocation/'sources')
            payload['sources']=[]
            for name in ('studio_jobs.py','studio_worker.py','agent_backend.py','openai_runner.py'):
                source=Path(__file__).parent/name;target=snapshots/name
                shutil.copyfile(source,target);target.chmod(0o600)
                payload['sources'].append({'path':'sources/'+name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
            _write_private(invocation/'invocation.json',payload)
            data['invocation_dir'] = str(invocation.relative_to(self.state_dir))
            self._update(db,row,data,'launching')
            db.execute('INSERT INTO active VALUES(1,?,?)',(row['id'],token))
        try:
            owner = self.executor.launch(payload)
            _require(type(owner.get('pid')) is int and owner['pid'] > 0 and isinstance(owner.get('start_identity'),str)
                     and 0 < len(owner['start_identity']) <= 256, 'Worker launch returned no verifiable owner identity')
            with self._db(True) as db:
                row = self._row(db,row['id']);data = json.loads(row['data'])
                if data.get('token') == token and row['status'] == 'launching':
                    data.update(owner_pid=owner['pid'],owner_identity=owner['start_identity'],
                                provenance=owner.get('provenance','analytic_fixture'))
                    self._update(db,row,data)
        except Exception as error:
            with self._db(True) as db:
                row = self._row(db,row['id']);data = json.loads(row['data'])
                data.update(error='Worker launch is uncertain; inspect the preserved invocation before retrying.',
                            launch_error_type=type(error).__name__)
                self._update(db,row,data,'needs_inspection')

    def cancel_cooperative(self, job_id):
        with self._db(True) as db:
            row = self._row(db,job_id);data = json.loads(row['data'])
            if row['status'] in TERMINAL or row['status'] in ('cancelled','cancel_requested'):
                pass
            elif row['status'] in ('created','queued','planned','awaiting_authorization'):
                data.update(cancel_requested=True,queued_at=None,ended=time.time())
                self._update(db,row,data,'cancelled');self._journal(db,job_id,{'status':'cancelled_before_work'})
            else:
                data['cancel_requested'] = True
                path = self.state_dir/data['invocation_dir']/'cancel.json'
                if not path.exists():
                    _write_private(path,{'requested_at':time.time(),'token':data['token'],
                        'scope':'Stop only at safe pipeline boundaries; do not erase handles, reservations or outputs.'})
                self._update(db,row,data,'cancel_requested');self._journal(db,job_id,{'status':'cancel_requested'})
        return self.get(job_id)

    def _worker_claim(self, job_id, token, pid, identity):
        with self._db(True) as db:
            row = self._row(db,job_id);data = json.loads(row['data'])
            active = db.execute('SELECT job,token FROM active WHERE id=1').fetchone()
            _require(active and tuple(active) == (job_id,token) and data.get('token') == token,
                     'Worker does not own the active lease')
            _require(row['status'] in ('launching','cancel_requested'), 'Worker invocation already claimed or uncertain')
            if data.get('owner_pid') is not None:
                _require(data['owner_pid'] == pid and data['owner_identity'] == identity,'Worker PID/start identity changed')
            data.update(owner_pid=pid,owner_identity=identity,heartbeat=time.time())
            self._update(db,row,data,'cancel_requested' if data.get('cancel_requested') else
                         'planning' if data['operation']=='plan' else 'running')
            self._journal(db,job_id,{'status':'worker_claimed','operation':data['operation'],'pid':pid,'start_identity':identity})
            return row,json.loads(row['request']),data

    def _heartbeat(self, job_id, token, *, retained_bytes=None, output_bytes=None):
        with self._db(True) as db:
            row = self._row(db,job_id);data = json.loads(row['data'])
            _require(data.get('token') == token,'Worker lease changed')
            if retained_bytes is not None:
                _require(type(retained_bytes) is int and retained_bytes>=0 and type(output_bytes) is int and 0<=output_bytes<=retained_bytes,
                         'Invalid studio retained-byte measurement')
                data.update(retained_bytes=retained_bytes,output_bytes=output_bytes,storage_measured_at=time.time())
            data['heartbeat'] = time.time();self._update(db,row,data)

    def _worker_complete(self, job_id, token, status, *, safe_to_release, error=None):
        _require(status in ('planned','awaiting_authorization','complete','unsupported','cancelled','failed','needs_inspection','checkpointed'),
                 'Unknown worker terminal status')
        with self._db(True) as db:
            row = self._row(db,job_id);data = json.loads(row['data'])
            active = db.execute('SELECT job,token FROM active WHERE id=1').fetchone()
            _require(active and tuple(active)==(job_id,token),'Worker completion does not own the lease')
            if status == 'checkpointed':
                status = 'planned'
            data.update(ended=time.time(),heartbeat=time.time(),error=_safe_text(error) if error else None,
                        plan_id=self._plan_id(row))
            self._update(db,row,data,status);self._journal(db,job_id,{'status':status,'safe_to_release':bool(safe_to_release)})
            if safe_to_release:
                db.execute('DELETE FROM active WHERE id=1')

    def _artifact(self, folder, relative, kind, label, identifier=None):
        if not isinstance(relative,str) or Path(relative).is_absolute() or '..' in Path(relative).parts:
            return None
        path = folder/relative
        maximum={'.png':16,'.jpg':16,'.jpeg':16,'.webp':16,'.mp4':128,'.glb':64,'.blend':512,'.zip':512}
        if (not path.is_file() or any(p.is_symlink() for p in (path,*path.parents))
                or not path.resolve().is_relative_to(folder.resolve()) or path.stat().st_size > 512*1024**2
                or path.suffix.lower() not in maximum or path.stat().st_size>maximum[path.suffix.lower()]*1024**2
                or any(part in ('inputs','sources','agents','generation') for part in Path(relative).parts)):
            return None
        return {'path':str(path.relative_to(self.root)), 'kind':kind, 'id':identifier or Path(relative).stem,
                'label':_safe_text(label,100),'bytes':path.stat().st_size}

    def _snapshot(self,row):
        data,request = json.loads(row['data']),json.loads(row['request'])
        folder = self.root/'outputs'/row['output'];status = row['status']
        if status in ('launching','planning','running','cancel_requested') and time.time()-data.get('heartbeat',0)>HEARTBEAT_STALE:
            status = 'needs_inspection'
        stages = _stage_progress(folder,status)
        authorization = self._authorization(row)
        plan = _json(folder/'plan.json')
        plan_id = self._plan_id(row)
        previews = []
        for relative in ['inspection/front.png','inspection/three-quarter.png']+[
                f'{prefix}-{i}/portrait.png' for i in range(3) for prefix in ('model','media','media-portable','media-continuation')]:
            item = self._artifact(folder,relative,'image','Work-in-progress preview')
            if item:previews.append(item)
        reviews=[]
        for attempt in range(3):
            review = _json(folder/f'review-{attempt}.json')
            if isinstance(review,dict) and isinstance(review.get('review'),dict):
                value=review['review'];reviews.append({'attempt':attempt,'status':_safe_text(value.get('verdict'),80),
                                                      'summary':_safe_text(value.get('summary'),800)})
        delivery = _json(folder/'delivery.json') if status=='complete' else None
        artifact_error=None
        if status=='complete' and (not isinstance(delivery,dict) or delivery.get('status')!='complete'):
            status='needs_inspection';artifact_error='Completed job is missing its sealed delivery record; nothing will be regenerated automatically.'
        result = None
        if delivery:
            result={'videos':[],'downloads':[],'limitations':[_safe_text(s,600) for s in delivery.get('limitations',[])[:20]]}
            for field,label in [('model','3D model'),('editable_source','Editable source')]:
                item=self._artifact(folder,delivery.get(field),'model' if field=='model' else 'download',label)
                if item:
                    result['downloads'].append({**item,'filename':Path(item['path']).name})
                    if field=='model':result['model_path']=item['path']
            media_folder=self._delivery_media(folder,delivery)
            media=_json(folder/media_folder/'manifest.json',{}) if media_folder else {}
            portrait = ('portrait.webp' if (folder/media_folder/'portrait.webp').exists() else 'portrait.png') if media_folder else None
            item=self._artifact(folder,media_folder+'/'+portrait,'image','Portrait') if portrait else None
            if item:result['portrait_path']=item['path'];previews.append(item)
            for clip in media.get('clips',[])[:8]:
                if not isinstance(clip,dict):continue
                identifier=clip.get('id',clip.get('state'))
                if not isinstance(identifier,str) or not re.fullmatch(r'[a-z][a-z0-9_-]{0,63}',identifier):continue
                video=self._artifact(folder,media_folder+'/animations/'+identifier+'.mp4','video',identifier,identifier)
                poster=self._artifact(folder,media_folder+'/fallbacks/'+identifier+'.png','image',identifier,identifier)
                if video:
                    result['videos'].append({**video,'loop_verified':False,'poster_path':poster['path'] if poster else None})
                    result['downloads'].append({**video,'filename':identifier+'.mp4'})
                    if poster:previews.append(poster)
            if 'model_path' not in result:
                status='needs_inspection';artifact_error='Completed delivery model is missing or outside the artifact allowlist.'
        error=data.get('error')
        error=artifact_error or error
        if status=='needs_inspection' and not error:error='Worker heartbeat or ownership is uncertain. Inspect preserved handles; no automatic duplicate will start.'
        can_resume=status in ('planned','awaiting_authorization','cancelled') and plan_id is not None
        return {'id':row['id'],'title':_safe_text(request['prompt'],80),'status':status,
            'created_at':_local_time(row['created']),'updated_at':_local_time(data['updated']),
            'elapsed_seconds':round(max(0,(data.get('ended') or time.time())-data['started']),2) if data.get('started') else 0,
            'request':{'prompt':_safe_text(request['prompt'],4000),'style':_safe_text(request['style'],800),
                       'custom_actions':[_safe_text(a,600) for a in request['custom_actions']],'preset':request['preset'],
                       'reference_selected':request['reference'] is not None,'reuse_model_selected':request['reuse_model'] is not None},
            'operation':data.get('operation'),'queued':status=='queued','cancel_requested':data.get('cancel_requested',False),
            'stages':stages,'plan_id':plan_id,
            'plan':{'id':plan_id,'summary':_safe_text(plan.get('subject',plan.get('reason','')),800),
                    'estimate':{'credits':authorization['estimated_credits'],'amount':None,'currency':'USD',
                                'note':'Account dollar price and agent usage price are unknown. No paid retries or allowance changes.'},
                    'authorization':authorization,'warnings':[_safe_text(s,600) for s in plan.get('limitations',[])[:20]]} if isinstance(plan,dict) else None,
            'capabilities':{'plan':status in ('created','cancelled') or status=='awaiting_authorization' and plan_id is None,
                            'start':can_resume and authorization['can_start'],
                            'cancel':status in ('created','queued','planning','running','launching','cancel_requested'),
                            'resume':can_resume,'session_required':True,'execution_gate':'canonical_ledger_and_explicit_start'},
            'previews':list({p['path']:p for p in previews}.values()),'reviews':reviews,'result':result,'error':{'message':_safe_text(error,800)} if error else None,
            'storage':{'retained_bytes_measured':data.get('retained_bytes',0),'pipeline_bytes_measured':data.get('output_bytes',0),
                       'measured_at':_local_time(data.get('storage_measured_at')),'total_limit_bytes':STORED_BYTES_LIMIT,
                       'active_reservation_bytes':ACTIVE_RESERVATION_BYTES,'automatic_deletion':False},
            'provenance':data.get('provenance','actual_pipeline')}

    def _delivery_media(self,folder,delivery):
        candidates=[delivery.get('media_pack')]
        candidates += [item.get('path') for item in delivery.get('packs',[])[:8] if isinstance(item,dict)]
        gallery=_json(folder/'gallery.json',{})
        candidates += [item.get('media_pack') for item in gallery.get('assets',[])[:1] if isinstance(item,dict)]
        attempt=delivery.get('selected_attempt')
        if type(attempt) is int and 0<=attempt<=2:candidates.append(f'media-portable-{attempt}')
        candidates.append('media-portable')  # Read older completed jobs without altering their packs.
        for relative in candidates:
            if not isinstance(relative,str) or Path(relative).is_absolute() or '..' in Path(relative).parts:continue
            path=self.root/relative if relative.startswith('outputs/') else folder/relative
            if (path.is_relative_to(folder) and re.fullmatch(r'media-portable(?:-[0-2])?',path.name)
                    and not any(p.is_symlink() for p in (path,*path.parents)) and (path/'manifest.json').is_file()):
                return str(path.relative_to(folder))
        return None

    def get(self,job_id):
        with self._db() as db:row=self._row(db,job_id)
        return self._snapshot(row)

    def list(self,limit=20,offset=0):
        _require(type(limit) is int and 1<=limit<=50 and type(offset) is int and 0<=offset<=MAX_JOBS,'Invalid studio list bounds')
        with self._db() as db:
            rows=[dict(r) for r in db.execute('SELECT * FROM jobs ORDER BY created DESC,id LIMIT ? OFFSET ?',(limit,offset))]
            total=db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]
        return {'jobs':[self._snapshot(row) for row in rows],'total':total,'offset':offset,'limit':limit}
