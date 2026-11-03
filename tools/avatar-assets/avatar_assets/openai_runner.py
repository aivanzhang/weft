"""Server-only structured agents; explicit dollar reservations survive uncertain requests."""
from contextlib import contextmanager
import base64
import fcntl
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import sqlite3
import stat
import time

from avatar_assets.agent_runner import (INSTRUCTIONS, MAX_PROMPT_BYTES, MAX_SCHEMA_BYTES, MAX_ANSWER_BYTES,
    _canonical, _schema_check, _validate, _image_inputs, _call_artifacts)
from avatar_assets.safety import read_json, require, sha256, slug, write_json
from avatar_assets.openai_models import MODEL, PRICING, price_profile, usage_cost_microdollars, reserve_microdollars

ROOT = Path(__file__).resolve().parents[1]
MAX_RESPONSE = 4 * 1024**2
MAX_REQUEST = 48 * 1024**2
KEY_FILE = ROOT.parents[1] / '.local/credentials/openai-api-key'
EXPOSURE_POLICY = 'validated-settled-exposure-v1'
BUDGET_POLICY = 'append-only-lower-ceiling-v1'
REASONING_EFFORTS = ('none', 'low', 'medium', 'high')


def validate_reasoning_effort(value):
    require(isinstance(value,str) and value in REASONING_EFFORTS,'Unsupported bounded reasoning effort')
    return value


def validate_reasoning_usage(usage, output_tokens):
    details=usage.get('output_tokens_details')
    if details is not None:
        require(isinstance(details,dict),'Invalid reported output token details')
        if 'reasoning_tokens' in details:
            _integer(details['reasoning_tokens'],0,output_tokens,'reported reasoning tokens')


def response_answer_text(response, *, reasoning_effort='none'):
    """Reasoning metadata is not an answer; tools and multiple final messages stay forbidden."""
    effort=validate_reasoning_effort(reasoning_effort)
    require(isinstance(response,dict) and response.get('status')=='completed','Incomplete API response')
    items=response.get('output')
    require(isinstance(items,list) and 1<=len(items)<=16,'Response contains unexpected output structure')
    messages=[];reasoning_bytes=0
    for item in items:
        require(isinstance(item,dict),'Invalid API output item')
        if item.get('type')=='reasoning':
            require(effort!='none' and {'id','type','summary'}<=set(item)
                    and set(item)<={'id','type','summary','content','encrypted_content','status'}
                    and isinstance(item['id'],str) and item['id'].startswith('rs_') and len(item['id'])<=256
                    and item.get('status') in (None,'completed'),'Unexpected API reasoning item')
            for key,kind in (('summary','summary_text'),('content','reasoning_text')):
                parts=item.get(key,[])
                require(isinstance(parts,list) and len(parts)<=32,'Reasoning metadata exceeds item bound')
                for part in parts:
                    require(isinstance(part,dict) and set(part)=={'type','text'} and part['type']==kind
                            and isinstance(part['text'],str),'Unexpected reasoning metadata content')
                    reasoning_bytes+=len(part['text'].encode())
            encrypted=item.get('encrypted_content')
            if encrypted is not None:
                require(isinstance(encrypted,str),'Unexpected encrypted reasoning metadata')
                reasoning_bytes+=len(encrypted.encode())
            require(reasoning_bytes<=256*1024,'Reasoning metadata exceeds256KiB')
        else:
            require(item.get('type')=='message' and item.get('role')=='assistant'
                    and item.get('status')=='completed' and item.get('phase') in (None,'final_answer'),
                    'Response contains unexpected tool activity or message structure')
            messages.append(item)
    require(len(messages)==1 and items[-1] is messages[0],'Response must contain exactly one final assistant message')
    parts=messages[0].get('content')
    require(isinstance(parts,list) and len(parts)==1 and isinstance(parts[0],dict)
            and parts[0].get('type')=='output_text','Response refused or omitted structured text')
    answer=parts[0].get('text')
    require(isinstance(answer,str) and len(answer.encode())<=MAX_ANSWER_BYTES,'Structured answer exceeds1MiB')
    return answer


def digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def stream_client_request_id(job_id, role, request):
    return 'avatar-'+hashlib.sha256((job_id+'/'+role+'/'+digest(request)).encode()).hexdigest()[:48]


def verify_stream_trace(folder, request, settings, response_id):
    """A transport trace binds identity, never permission to settle an unknown charge."""
    from avatar_assets.safety import local_file
    base = {key:value for key,value in request.items() if key != 'client_request_id'}
    expected = stream_client_request_id(settings['job_id'],request['role'],base)
    trace = read_json(local_file(Path(folder)/'transport.json',8192))
    require(request.get('client_request_id') == expected and trace.get('client_request_id') == expected
            and trace.get('schema_version') == 1 and trace.get('transport') == 'sse-v1'
            and trace.get('response_id') == response_id and trace.get('status') == 'completed',
            'Streaming response identity differs from its saved transport trace')
    return trace


def _private(path):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'Private path contains symlink')
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.stat()
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700,
            'Private directory must be owned by this user with mode0700')
    return path.resolve()


def _integer(value, minimum, maximum, name):
    require(type(value) is int and minimum <= value <= maximum, 'Invalid '+name)
    return value


def _approval(value):
    require(isinstance(value, dict) and value.get('decision') == 'approved'
            and isinstance(value.get('user_reply'), str) and 0 < len(value['user_reply']) <= 4000
            and len(_canonical(value)) <= 8192, 'Explicit administrator spending approval is required')
    return _canonical(value).decode()


def _tightening_approval(value, ceiling):
    record = _approval(value)
    require(value.get('operation') == 'tighten_project_ceiling'
            and type(value.get('max_microdollars')) is int and value['max_microdollars'] == ceiling,
            'Ceiling reduction approval must bind the exact lower amount')
    return record


def budget_snapshot(db):
    """Read the original approval and its strictly decreasing, append-only ceiling chain."""
    original = db.execute('SELECT ceiling FROM budget WHERE id=1').fetchone()
    has_reductions = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='budget_reductions'").fetchone() is not None
    count = db.execute('SELECT COUNT(*) FROM budget_reductions').fetchone()[0] if has_reductions else 0
    require(count <= 1024, 'API budget reduction inventory exceeds1024 records')
    if original is None:
        require(count == 0, 'Budget reductions have no original approval')
        return None
    initial = current = _integer(original[0], 1, 1000000000, 'original project dollar ceiling')
    rows = db.execute('SELECT id,previous_ceiling,ceiling,approval FROM budget_reductions ORDER BY id') if count else ()
    for number, (index, previous, ceiling, approval) in enumerate(rows, 1):
        require(type(index) is int and index == number and type(previous) is int and previous == current,
                'Broken API budget reduction chain')
        _integer(ceiling, 1, current-1, 'lower project dollar ceiling')
        require(isinstance(approval, str) and len(approval.encode()) <= 8192,
                'Invalid budget reduction approval')
        require(_tightening_approval(_decode(approval.encode()), ceiling) == approval,
                'Budget reduction approval differs from its record')
        current = ceiling
    return {'policy': BUDGET_POLICY, 'original_ceiling_microdollars': initial,
            'effective_ceiling_microdollars': current, 'reductions': count}


def _profile_bindings(db, keys=None):
    exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='call_profiles'").fetchone()
    if not exists:
        return {}
    require(db.execute('SELECT COUNT(*) FROM call_profiles').fetchone()[0] <= 10000,
            'API price binding inventory exceeds10000 calls')
    require(db.execute('SELECT COUNT(*) FROM call_profiles p LEFT JOIN calls c ON p.job=c.job AND p.role=c.role '
                       'WHERE c.job IS NULL').fetchone()[0] == 0, 'API price binding has no original call')
    groups = [None] if keys is None else [keys[i:i+128] for i in range(0, len(keys), 128)]
    result = {}
    for group in groups:
        where = '' if group is None else ' WHERE (job,role) IN ('+','.join('(?,?)' for _ in group)+')'
        args = () if group is None else tuple(v for pair in group for v in pair)
        sizes = db.execute('SELECT length(CAST(binding AS BLOB)),typeof(binding) FROM call_profiles'+where, args)
        require(all(kind == 'text' and type(size) is int and 1 <= size <= 4096 for size, kind in sizes),
                'API price binding is missing or oversized')
        for job, role, raw in db.execute('SELECT job,role,binding FROM call_profiles'+where, args):
            require(isinstance(job, str) and isinstance(role, str) and (job, role) not in result,
                    'Invalid API call price binding')
            binding = _decode(raw.encode())
            require(isinstance(binding, dict) and _canonical(binding).decode() == raw,
                    'Noncanonical API call price binding')
            result[job, role] = binding
    return result


def attach_call_profiles(db, rows):
    """Bind selected SQL rows with indexed queries before closing their connection."""
    result = []
    for row in rows:
        require(len(result) < 10000, 'API call inventory exceeds10000 calls')
        result.append(dict(row))
    bindings = _profile_bindings(db, list(dict.fromkeys((c['job'], c['role']) for c in result)))
    for call in result:
        binding = bindings.get((call['job'], call['role']))
        if binding is not None:
            call['model_profile'] = binding
        call_model(call)
    return result


def bound_call(db, row):
    return attach_call_profiles(db, [row])[0]


def call_model(call):
    call = dict(call)
    if 'model_profile' not in call:
        return MODEL
    binding = call['model_profile']
    require(isinstance(binding, dict) and set(binding) == {'model', 'pricing', 'max_output_tokens',
            'request_digest', 'reserved_microdollars'}, 'Invalid API call model profile')
    profile = price_profile(binding['model'])
    require(_canonical(binding['pricing']) == _canonical(profile) and binding['request_digest'] == call.get('request_digest')
            and type(binding['reserved_microdollars']) is int
            and binding['reserved_microdollars'] == call['reserved']
            and call['reserved'] == reserve_microdollars(binding['max_output_tokens'], binding['model']),
            'API call price profile differs from its reservation')
    return binding['model']


def call_exposure(call):
    """Unknown charges keep their full hold; only validated settled usage reduces exposure."""
    call = dict(call)
    model = call_model(call)
    reserved = _integer(call['reserved'], 1, 1000000000, 'stored reservation')
    if call['state'] == 'reserved_uncertain':
        return reserved
    require(call['state'] == 'settled', 'Unknown API call accounting state')
    require(isinstance(call['usage'], str) and len(call['usage'].encode()) <= 16384,
            'Settled API usage is missing or oversized')
    usage = _decode(call['usage'].encode())
    require(isinstance(usage, dict), 'Invalid settled API usage')
    inp = _integer(usage.get('input_tokens'), 0, price_profile(model)['context_tokens'], 'settled input tokens')
    output = _integer(usage.get('output_tokens'), 0, call.get('model_profile', {}).get('max_output_tokens', 32768), 'settled output tokens')
    _integer(usage.get('total_tokens'), inp+output, inp+output, 'settled total tokens')
    validate_reasoning_usage(usage,output)
    expected = usage_cost_microdollars(model, inp, output)
    require(type(call['actual']) is int and call['actual'] == expected <= reserved
            and isinstance(call['response_id'], str) and call['response_id'].startswith('resp_')
            and len(call['response_id']) <= 256, 'Settled API cost lacks matching validated usage')
    return expected


def accounting_snapshot(db):
    """Bounded read-only accounting shared by readiness and transactional reservation."""
    require(db.execute('SELECT COUNT(*) FROM calls').fetchone()[0] <= 10000,
            'API accounting inventory exceeds10000 calls')
    has_retirements = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='retirements'").fetchone() is not None
    retired = {}
    if has_retirements:
        require(db.execute('SELECT COUNT(*) FROM retirements').fetchone()[0] <= 10000,
                'API retirement inventory exceeds10000 jobs')
        for parent, amount in db.execute('SELECT parent,actual FROM retirements'):
            require(isinstance(parent, str) and parent not in retired, 'Invalid retired job identity')
            retired[parent] = _integer(amount, 0, 1000000000, 'retired job cost')
    jobs = {}
    bindings = _profile_bindings(db)
    columns = ('job', 'role', 'request_digest', 'reserved', 'state', 'response_id', 'actual', 'usage')
    for row in db.execute('SELECT job,role,request_digest,reserved,state,response_id,actual,usage FROM calls'):
        call = dict(zip(columns, row))
        binding = bindings.pop((call['job'], call['role']), None)
        if binding is not None:
            call['model_profile'] = binding
        require(isinstance(call['job'], str) and 1 <= len(call['job']) <= 64, 'Invalid accounting job identity')
        amount = call_exposure(call)
        total = jobs.setdefault(call['job'], {'calls': 0, 'reserved_microdollars': 0,
            'effective_exposure_microdollars': 0, 'settled_actual_microdollars': 0,
            'uncertain_reserved_microdollars': 0})
        total['calls'] += 1
        total['reserved_microdollars'] += call['reserved']
        total['effective_exposure_microdollars'] += amount
        total['settled_actual_microdollars' if call['state'] == 'settled' else 'uncertain_reserved_microdollars'] += amount
    require(not bindings, 'API price binding has no original call')
    for parent, amount in retired.items():
        total = jobs.get(parent)
        require(total is not None and total['uncertain_reserved_microdollars'] == 0
                and total['settled_actual_microdollars'] == amount, 'Retired job cost differs from settled call history')
    active = sum(total['effective_exposure_microdollars'] for job, total in jobs.items() if job not in retired)
    return {'policy': EXPOSURE_POLICY, 'jobs': jobs, 'retired_jobs': retired,
            'active_exposure_microdollars': active, 'retired_actual_microdollars': sum(retired.values()),
            'project_exposure_microdollars': active + sum(retired.values())}


class OpenAILedger:
    def __init__(self, folder=None):
        self.folder = _private(folder or ROOT / '.local/openai')
        path = self.folder / 'ledger.sqlite3'
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                    and stat.S_IMODE(info.st_mode) == 0o600, 'Ledger must be owner-only')
        finally:
            os.close(fd)
        self.db = sqlite3.connect(path, timeout=2)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA max_page_count=8192')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS budget(id INTEGER PRIMARY KEY CHECK(id=1), ceiling INTEGER, approval TEXT);
            CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, digest TEXT, calls INTEGER, ceiling INTEGER, approval TEXT);
            CREATE TABLE IF NOT EXISTS calls(job TEXT, role TEXT, request_digest TEXT, reserved INTEGER,
                state TEXT, response_id TEXT, actual INTEGER, usage TEXT, PRIMARY KEY(job,role));
            CREATE TABLE IF NOT EXISTS retirements(parent TEXT PRIMARY KEY, successor TEXT UNIQUE,
                actual INTEGER, approval TEXT);
            CREATE TABLE IF NOT EXISTS call_profiles(job TEXT, role TEXT, binding TEXT, PRIMARY KEY(job,role));
            CREATE TABLE IF NOT EXISTS budget_reductions(id INTEGER PRIMARY KEY,
                previous_ceiling INTEGER, ceiling INTEGER, approval TEXT);
        ''')
        from avatar_assets.openai_grant_closure import ensure_schema
        ensure_schema(self.db)

    def close(self):
        self.db.close()

    @contextmanager
    def transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def configure(self, max_microdollars, approval):
        _integer(max_microdollars, 1, 1000000000, 'project dollar ceiling')
        record = _approval(approval)
        with self.transaction():
            row = self.db.execute('SELECT ceiling,approval FROM budget WHERE id=1').fetchone()
            require(row is None or tuple(row) == (max_microdollars, record), 'Project allowance is immutable')
            if row is None:
                self.db.execute('INSERT INTO budget VALUES(1,?,?)', (max_microdollars, record))

    def tighten_budget(self, max_microdollars, approval):
        """Lower future spending without rewriting grants or releasing uncertain charges."""
        _integer(max_microdollars, 1, 1000000000, 'lower project dollar ceiling')
        record = _tightening_approval(approval, max_microdollars)
        with self.transaction():
            budget = budget_snapshot(self.db)
            require(budget is not None, 'Configure the approved project allowance first')
            previous = budget['effective_ceiling_microdollars']
            require(max_microdollars < previous, 'Project ceiling reductions must strictly decrease the current limit')
            require(budget['reductions'] < 1024, 'API budget reduction inventory exceeds1024 records')
            exposure = accounting_snapshot(self.db)['project_exposure_microdollars']
            require(exposure <= max_microdollars, 'Known costs and uncertain holds exceed the proposed lower ceiling')
            self.db.execute('INSERT INTO budget_reductions VALUES(?,?,?,?)',
                            (budget['reductions']+1, previous, max_microdollars, record))
            return {**budget_snapshot(self.db), 'project_exposure_microdollars': exposure}

    def authorize_job(self, job_id, job_digest, max_calls, max_microdollars, approval):
        slug(job_id)
        require(isinstance(job_digest, str) and len(job_digest) == 64
                and all(c in '0123456789abcdef' for c in job_digest), 'Invalid immutable job digest')
        _integer(max_calls, 1, 12, 'job call ceiling')
        _integer(max_microdollars, 1, 1000000000, 'job dollar ceiling')
        record = _approval(approval)
        expected = (job_digest, max_calls, max_microdollars, record)
        with self.transaction():
            budget = budget_snapshot(self.db)
            require(budget is not None, 'Configure the approved project allowance first')
            row = self.db.execute('SELECT digest,calls,ceiling,approval FROM jobs WHERE id=?', (job_id,)).fetchone()
            require(row is None or tuple(row) == expected, 'Job approval cannot be changed or reset')
            if row is None:
                allocated = self._allocated()
                require(allocated + max_microdollars <= budget['effective_ceiling_microdollars'], 'Project allowance is already allocated')
                self.db.execute('INSERT INTO jobs VALUES(?,?,?,?,?)', (job_id, *expected))

    def _allocated(self):
        from avatar_assets.openai_grant_closure import allocated_grants
        return allocated_grants(self.db)

    def continue_job(self, parent_id, parent_digest, job_id, job_digest, max_calls, max_microdollars, approval):
        """Explicitly retire settled work and grant one successor without resetting history."""
        slug(parent_id); slug(job_id)
        require(parent_id != job_id, 'Continuation needs a new immutable job')
        for value in (parent_digest, job_digest):
            require(isinstance(value, str) and len(value) == 64
                    and all(c in '0123456789abcdef' for c in value), 'Invalid continuation job digest')
        _integer(max_calls, 1, 12, 'successor call ceiling')
        _integer(max_microdollars, 1, 1000000000, 'successor dollar ceiling')
        record = _approval(approval)
        require(approval.get('operation') == 'retire_and_continue'
                and approval.get('parent_job_id') == parent_id
                and approval.get('successor_job_id') == job_id
                and type(approval.get('additional_calls')) is int
                and approval['additional_calls'] == max_calls,
                'Continuation requires explicit approval bound to both jobs and the new call count')
        with self.transaction():
            from avatar_assets.openai_grant_closure import require_open_grant
            require_open_grant(self.db, parent_id)
            parent = self.db.execute('SELECT * FROM jobs WHERE id=?', (parent_id,)).fetchone()
            require(parent is not None and parent['digest'] == parent_digest, 'Continuation parent binding differs')
            require(not self.db.execute('SELECT 1 FROM retirements WHERE parent=?', (parent_id,)).fetchone(),
                    'Parent is already retired; continuation cannot be repeated')
            require(not self.db.execute('SELECT 1 FROM jobs WHERE id=?', (job_id,)).fetchone(),
                    'Successor already has an immutable grant')
            calls = self.db.execute('SELECT * FROM calls WHERE job=?', (parent_id,)).fetchall()
            require(0 < len(calls) <= parent['calls'], 'Continuation requires recorded parent calls')
            actual = 0
            for call in calls:
                require(call['state'] == 'settled', 'Unknown parent charges cannot be released')
                actual += call_exposure(bound_call(self.db, call))
            budget = budget_snapshot(self.db)
            require(budget is not None and self._allocated()-parent['ceiling']+actual+max_microdollars <= budget['effective_ceiling_microdollars'],
                    'Continuation exceeds the unchanged project allowance')
            # BEGIN IMMEDIATE makes retirement, accounting and the successor grant indivisible.
            self.db.execute('INSERT INTO retirements VALUES(?,?,?,?)', (parent_id, job_id, actual, record))
            self.db.execute('INSERT INTO jobs VALUES(?,?,?,?,?)',
                            (job_id, job_digest, max_calls, max_microdollars, record))

    def binding(self, job_id, job_digest, max_calls):
        row = self.db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
        require(row is not None and row['digest'] == job_digest and row['calls'] == max_calls,
                'Missing or changed job-specific API allowance')
        return dict(row)

    def reserve(self, job_id, job_digest, max_calls, role, request_digest, amount, *, model=None, max_output_tokens=16384):
        slug(role)
        require(isinstance(request_digest, str) and len(request_digest) == 64
                and all(c in '0123456789abcdef' for c in request_digest), 'Invalid call digest')
        _integer(amount, 1, 1000000000, 'reservation amount')
        binding = None
        if model is not None:
            binding = {'model': model, 'pricing': price_profile(model), 'max_output_tokens': max_output_tokens,
                       'request_digest': request_digest, 'reserved_microdollars': amount}
            call_model({'model_profile': binding, 'request_digest': request_digest, 'reserved': amount})
        with self.transaction():
            from avatar_assets.openai_grant_closure import require_open_grant
            require_open_grant(self.db, job_id)
            job = self.binding(job_id, job_digest, max_calls)
            require(not self.db.execute('SELECT 1 FROM retirements WHERE parent=?', (job_id,)).fetchone(),
                    'Retired API job cannot reserve more calls')
            require(not self.db.execute('SELECT 1 FROM calls WHERE job=? AND role=?', (job_id, role)).fetchone(),
                    'API call is already reserved; no automatic resubmission')
            exposure = accounting_snapshot(self.db)
            current = exposure['jobs'].get(job_id, {'calls': 0, 'effective_exposure_microdollars': 0})
            require(current['calls'] < max_calls and current['effective_exposure_microdollars'] + amount <= job['ceiling'],
                    'API job spending allowance exhausted')
            budget = budget_snapshot(self.db)
            require(budget is not None and exposure['project_exposure_microdollars'] + amount <=
                    budget['effective_ceiling_microdollars'], 'API project spending allowance exhausted')
            self.db.execute('INSERT INTO calls VALUES(?,?,?,?,?,NULL,NULL,NULL)',
                            (job_id, role, request_digest, amount, 'reserved_uncertain'))
            if binding is not None:
                self.db.execute('INSERT INTO call_profiles VALUES(?,?,?)', (job_id, role, _canonical(binding).decode()))

    def settle(self, job_id, role, response_id, usage, actual):
        _integer(actual, 0, 1000000000, 'settled cost')
        with self.transaction():
            row = self.db.execute('SELECT * FROM calls WHERE job=? AND role=?', (job_id, role)).fetchone()
            require(row is not None and row['state'] == 'reserved_uncertain', 'Call settlement differs from its reservation')
            require(actual <= row['reserved'], 'Provider usage exceeds conservative reservation; inspect before more work')
            usage_json = _canonical(usage).decode()
            call_exposure({**bound_call(self.db, row), 'state': 'settled', 'response_id': response_id, 'actual': actual, 'usage': usage_json})
            self.db.execute('UPDATE calls SET state=?,response_id=?,actual=?,usage=? WHERE job=? AND role=?',
                            ('settled', response_id, actual, usage_json, job_id, role))

    def call(self, job_id, role):
        row = self.db.execute('SELECT * FROM calls WHERE job=? AND role=?', (job_id, role)).fetchone()
        return bound_call(self.db, row) if row is not None else None


def read_api_key(path=KEY_FILE):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'API key path contains symlink')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as handle:
        info = os.fstat(handle.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o600
                and 10 <= info.st_size <= 4096, 'API key must be owner-only bounded regular file')
        key = handle.read(4097).strip()
    require(10 <= len(key) <= 4096 and all(33 <= b <= 126 for b in key), 'Invalid API key format')
    return key.decode('ascii')


def post_response(payload, timeout, key):
    """Fixed TLS endpoint; the owning worker supplies the hard wall-time limit.

    Socket deadlines bound reads/writes; stdlib DNS resolution needs the outer worker limit.
    """
    body = _canonical(payload)
    require(len(body) <= MAX_REQUEST, 'API request exceeds48MiB')
    started = time.monotonic()
    connection = http.client.HTTPSConnection('api.openai.com', timeout=timeout)
    try:
        connection.request('POST', '/v1/responses', body=body,
                           headers={'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'})
        remaining = timeout - (time.monotonic()-started)
        require(remaining > 0, 'API request deadline exceeded')
        transport_socket = connection.sock
        transport_socket.settimeout(remaining)
        response = connection.getresponse()
        require(response.status == 200, 'OpenAI HTTP request failed with status '+str(response.status))
        require(response.getheader('Content-Encoding') in (None, 'identity'), 'Compressed API responses are unsupported')
        declared = response.getheader('Content-Length')
        require(declared is None or declared.isdigit() and int(declared) <= MAX_RESPONSE, 'API response is oversized')
        raw = bytearray()
        while True:
            remaining = timeout - (time.monotonic()-started)
            require(remaining > 0, 'API response deadline exceeded')
            # HTTPConnection clears its socket on Connection: close; the response still owns it.
            transport_socket.settimeout(remaining)
            chunk = response.read1(min(65536, MAX_RESPONSE+1-len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
            require(len(raw) <= MAX_RESPONSE, 'API response exceeds4MiB')
            if response.isclosed():
                break
        require(declared is None or len(raw) == int(declared), 'Incomplete API response body')
        return bytes(raw)
    except (OSError, http.client.HTTPException):
        raise ValueError('OpenAI transport failed; reservation retained, no automatic retry') from None
    finally:
        connection.close()


def _decode(raw):
    require(isinstance(raw, bytes) and len(raw) <= MAX_RESPONSE, 'Invalid bounded API response')
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, 'Duplicate API JSON key')
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite API response')))


def _strict(schema):
    _schema_check(schema)
    require(isinstance(schema, dict), 'Strict API schema must be an object schema')
    require('oneOf' not in schema and 'default' not in schema, 'Unsupported strict API schema keyword')
    if schema.get('type') == 'object':
        require(schema.get('additionalProperties') is False
                and set(schema.get('required', [])) == set(schema.get('properties', {})),
                'Strict API objects must forbid extras and require every property')
    for key in ('properties', '$defs'):
        for child in schema.get(key, {}).values():
            _strict(child)
    if 'items' in schema:
        _strict(schema['items'])
    for child in schema.get('anyOf', []):
        _strict(child)


class OpenAIRunner:
    def __init__(self, root, *, job_id, job_digest, ledger=None, max_calls=5, timeout=240,
                 model=MODEL, max_output_tokens=16384, reasoning_effort='none', transport=None):
        slug(job_id)
        profile = price_profile(model)
        validate_reasoning_effort(reasoning_effort)
        require(transport is None or transport=='sse-v1', 'Unsupported API transport')
        _integer(max_calls, 1, 12, 'call ceiling')
        require(type(timeout) in (int,float) and math.isfinite(timeout) and 1 <= timeout <= 600, 'Invalid API timeout')
        self.reservation = reserve_microdollars(max_output_tokens, model)
        self.ledger = ledger or OpenAILedger()
        self.ledger.binding(job_id, job_digest, max_calls)
        self.root = _private(root)
        self.job_id, self.job_digest, self.max_calls = job_id, job_digest, max_calls
        self.timeout = timeout
        self.settings = {'schema_version':1, 'execution_kind':'openai_responses_api', 'model':model,
            'max_calls':max_calls, 'timeout':timeout, 'max_output_tokens':max_output_tokens,
            'job_id':job_id, 'job_digest':job_digest, 'automatic_retries':0, 'tools':[], 'store':False,
            'service_tier':'default', 'reasoning':{'effort':reasoning_effort}, 'truncation':'disabled', 'pricing':profile, 'implementation_sha256':sha256(Path(__file__)), 'reservation_microdollars_per_call':self.reservation,
            'reservation_policy':'validated-settled-exposure-v1 with append-only-lower-ceiling-v1: reserve the full1.05M context at long-context rates plus maximum output before every call. Validated settled upper costs replace holds in exposure; the latest validated lower project ceiling applies. Uncertain holds, original rows, call caps and explicit retry approval remain.'}
        if model != MODEL:
            self.settings['reservation_policy'] = 'full-pinned-model-context-v1: reserve the entire priced model context plus maximum output; preserve every prior uncertain hold and grant.'
            self.settings['model_profile_implementation_sha256'] = sha256(Path(__file__).with_name('openai_models.py'))
        if transport is not None:
            self.settings.update(transport=transport,
                transport_implementation_sha256=sha256(Path(__file__).with_name('openai_stream.py')))
        with self._lock():
            path = self.root/'runner.json'
            if path.exists():
                require(read_json(path) == self.settings, 'API runner settings cannot change')
            else:
                require(not any(p.is_dir() for p in self.root.iterdir()), 'Existing calls lack runner settings')
                write_json(path, self.settings)

    @contextmanager
    def _lock(self):
        fd = os.open(self.root/'runner.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'a') as stream:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('Another worker owns this API runner') from None
            yield

    def call(self, role, prompt, schema, images=()):
        slug(role)
        require(isinstance(prompt,str) and 0 < len(prompt.encode()) <= MAX_PROMPT_BYTES, 'API prompt exceeds bound')
        require(len(_canonical(schema)) <= MAX_SCHEMA_BYTES, 'API schema exceeds bound')
        _strict(schema)
        require(schema.get('type') == 'object', 'API response root must be an object')
        selected = _image_inputs(images)
        request = {'role':role, 'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                   'schema_sha256':digest(schema), 'settings_sha256':digest(self.settings),
                   'images':[{k:v for k,v in row.items() if k != 'path'} for row in selected]}
        streaming = self.settings.get('transport') == 'sse-v1'
        if streaming:
            # A role can submit only once; this opaque ID survives a lost HTTP response.
            request['client_request_id'] = stream_client_request_id(self.job_id,role,request)
        with self._lock():
            out = self.root/role
            prior = self.ledger.call(self.job_id,role)
            if out.exists():
                require(not out.is_symlink() and read_json(out/'request.json') == request, 'Cached API inputs changed')
                receipt = read_json(out/'receipt.json')
                require(receipt.get('status') == 'succeeded' and receipt['artifacts'] == _call_artifacts(out),
                        'API call failed, uncertain or changed; no automatic resubmission')
                require(prior and prior['state']=='settled' and prior['request_digest']==digest(request)
                        and prior['response_id']==receipt['response_id']
                        and call_model(prior)==self.settings['model'], 'API cached ledger custody differs')
                answer = read_json(out/'answer.json'); _validate(answer,schema)
                return answer
            require(prior is None, 'API reservation exists without its output; inspect instead of resubmitting')
            key = read_api_key()
            out.mkdir(mode=0o700)
            write_json(out/'request.json',request)
            write_json(out/'schema.json',schema)
            (out/'prompt.txt').write_text(prompt)
            content = [{'type':'input_text','text':prompt}]
            for index,row in enumerate(selected):
                raw = row['path'].read_bytes()
                require(hashlib.sha256(raw).hexdigest()==row['sha256'], 'Selected image changed')
                (out/f'image-{index:02d}{row["suffix"]}').write_bytes(raw)
                mime = {'.png':'image/png','.jpg':'image/jpeg','.jpeg':'image/jpeg','.webp':'image/webp'}[row['suffix']]
                content.append({'type':'input_image','image_url':'data:'+mime+';base64,'+base64.b64encode(raw).decode(), 'detail':'high'})
            payload = {'model':self.settings['model'], 'instructions':INSTRUCTIONS, 'input':[{'role':'user','content':content}],
                       'text':{'format':{'type':'json_schema','name':'avatar_'+role.replace('-','_'),'schema':schema,'strict':True}},
                       'max_output_tokens':self.settings['max_output_tokens'], 'reasoning':self.settings['reasoning'],
                       'store':False,'tools':[],'tool_choice':'none','service_tier':'default','truncation':'disabled'}
            if streaming:
                payload['stream'] = True
            require(len(_canonical(payload)) <= MAX_REQUEST, 'API request exceeds bound')
            write_json(out/'payload-metadata.json',{'sha256':digest(payload),'bytes':len(_canonical(payload)),
                       'instructions':INSTRUCTIONS,'endpoint':'https://api.openai.com/v1/responses'})
            self.ledger.reserve(self.job_id,self.job_digest,self.max_calls,role,digest(request),self.reservation,
                                model=self.settings['model'],max_output_tokens=self.settings['max_output_tokens'])
            started = time.monotonic()
            try:
                if streaming:
                    from avatar_assets.openai_stream import post_stream_response
                    require(sha256(Path(__file__).with_name('openai_stream.py')) ==
                            self.settings['transport_implementation_sha256'], 'Streaming transport source changed')
                    raw = post_stream_response(payload,self.timeout,key,trace_path=out/'transport.json',
                                               client_request_id=request['client_request_id'])
                else:
                    raw = post_response(payload,self.timeout,key)
                require(len(raw)<=MAX_RESPONSE,'API response exceeds bound')
                (out/'response.json').write_bytes(raw)
                response = _decode(raw)
                require(isinstance(response, dict), 'Invalid API response object')
                require(response.get('status')=='completed' and response.get('model')==self.settings['model']
                        and response.get('service_tier')=='default', 'Incomplete or unexpected API response')
                rid = response.get('id')
                require(isinstance(rid,str) and rid.startswith('resp_') and len(rid)<=256,'Invalid response identity')
                if streaming:
                    verify_stream_trace(out,request,self.settings,rid)
                usage = response.get('usage',{})
                inp = _integer(usage.get('input_tokens'),0,self.settings['pricing']['context_tokens'],'reported input tokens')
                output = _integer(usage.get('output_tokens'),0,self.settings['max_output_tokens'],'reported output tokens')
                _integer(usage.get('total_tokens'),inp+output,inp+output,'reported total tokens')
                validate_reasoning_usage(usage,output)
                cost = usage_cost_microdollars(self.settings['model'],inp,output)
                self.ledger.settle(self.job_id,role,rid,usage,cost)
                answer_text = response_answer_text(response,reasoning_effort=self.settings['reasoning']['effort'])
                answer = _decode(answer_text.encode()); _validate(answer,schema)
                write_json(out/'answer.json',answer)
                write_json(out/'receipt.json',{'schema_version':1,'status':'succeeded','execution_kind':'openai_responses_api',
                    'response_id':rid,'model':self.settings['model'],'usage':usage,'actual_microdollars_upper_estimate':cost,
                    'reserved_microdollars':self.reservation,'elapsed_seconds':time.monotonic()-started,
                    'tool_activity':False,'answer_sha256':sha256(out/'answer.json'),'artifacts':_call_artifacts(out)})
                return answer
            except BaseException:
                if not (out/'receipt.json').exists():
                    write_json(out/'receipt.json',{'schema_version':1,'status':'failed_or_uncertain',
                        'execution_kind':'openai_responses_api','automatic_resubmission_allowed':False,
                        'error':'API call failed validation or transport; inspect bounded response and ledger. Secrets omitted.',
                        'reserved_microdollars':self.reservation,'elapsed_seconds':time.monotonic()-started,
                        'artifacts':_call_artifacts(out)})
                raise
