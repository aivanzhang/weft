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

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'gpt-5.4-2026-03-05'
PRICING = {'model': MODEL, 'checked': '2026-11-02', 'source': 'https://developers.openai.com/api/docs/models/gpt-5.4',
           'context_tokens': 1050000, 'long_context_threshold': 272000,
           'input_microdollars_per_token': 2.5, 'output_microdollars_per_token': 15,
           'long_input_multiplier': 2, 'long_output_multiplier': 1.5}
MAX_RESPONSE = 4 * 1024**2
MAX_REQUEST = 48 * 1024**2
KEY_FILE = ROOT.parents[1] / '.local/credentials/openai-api-key'


def digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


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


def reserve_microdollars(max_output_tokens=16384):
    _integer(max_output_tokens, 256, 32768, 'output token ceiling')
    # No tokenizer estimate substitutes for a bound, including images and schema overhead.
    return 1050000 * 5 + math.ceil(max_output_tokens * 22.5)


def _approval(value):
    require(isinstance(value, dict) and value.get('decision') == 'approved'
            and isinstance(value.get('user_reply'), str) and 0 < len(value['user_reply']) <= 4000
            and len(_canonical(value)) <= 8192, 'Explicit administrator spending approval is required')
    return _canonical(value).decode()


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
        ''')

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

    def authorize_job(self, job_id, job_digest, max_calls, max_microdollars, approval):
        slug(job_id)
        require(isinstance(job_digest, str) and len(job_digest) == 64
                and all(c in '0123456789abcdef' for c in job_digest), 'Invalid immutable job digest')
        _integer(max_calls, 1, 12, 'job call ceiling')
        _integer(max_microdollars, 1, 1000000000, 'job dollar ceiling')
        record = _approval(approval)
        expected = (job_digest, max_calls, max_microdollars, record)
        with self.transaction():
            budget = self.db.execute('SELECT ceiling FROM budget WHERE id=1').fetchone()
            require(budget is not None, 'Configure the approved project allowance first')
            row = self.db.execute('SELECT digest,calls,ceiling,approval FROM jobs WHERE id=?', (job_id,)).fetchone()
            require(row is None or tuple(row) == expected, 'Job approval cannot be changed or reset')
            if row is None:
                allocated = self._allocated()
                require(allocated + max_microdollars <= budget[0], 'Project allowance is already allocated')
                self.db.execute('INSERT INTO jobs VALUES(?,?,?,?,?)', (job_id, *expected))

    def _allocated(self):
        active = self.db.execute('SELECT COALESCE(SUM(ceiling),0) FROM jobs WHERE id NOT IN '
                                 '(SELECT parent FROM retirements)').fetchone()[0]
        retired = self.db.execute('SELECT COALESCE(SUM(actual),0) FROM retirements').fetchone()[0]
        return active + retired

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
                require(isinstance(call['usage'], str), 'Settled parent usage is missing')
                usage = _decode(call['usage'].encode())
                require(isinstance(usage, dict), 'Invalid settled parent usage')
                inp = _integer(usage.get('input_tokens'), 0, 1050000, 'settled input tokens')
                output = _integer(usage.get('output_tokens'), 0, 32768, 'settled output tokens')
                _integer(usage.get('total_tokens'), inp+output, inp+output, 'settled total tokens')
                expected = math.ceil(inp*(5 if inp > 272000 else 2.5) + output*(22.5 if inp > 272000 else 15))
                require(type(call['actual']) is int and call['actual'] == expected <= call['reserved']
                        and isinstance(call['response_id'], str) and call['response_id'].startswith('resp_'),
                        'Settled parent cost lacks matching validated usage')
                actual += expected
            budget = self.db.execute('SELECT ceiling FROM budget WHERE id=1').fetchone()
            require(budget is not None and self._allocated()-parent['ceiling']+actual+max_microdollars <= budget[0],
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

    def reserve(self, job_id, job_digest, max_calls, role, request_digest, amount):
        slug(role)
        require(isinstance(request_digest, str) and len(request_digest) == 64
                and all(c in '0123456789abcdef' for c in request_digest), 'Invalid call digest')
        _integer(amount, 1, 1000000000, 'reservation amount')
        with self.transaction():
            job = self.binding(job_id, job_digest, max_calls)
            require(not self.db.execute('SELECT 1 FROM retirements WHERE parent=?', (job_id,)).fetchone(),
                    'Retired API job cannot reserve more calls')
            require(not self.db.execute('SELECT 1 FROM calls WHERE job=? AND role=?', (job_id, role)).fetchone(),
                    'API call is already reserved; no automatic resubmission')
            count, spent = self.db.execute('SELECT COUNT(*),COALESCE(SUM(reserved),0) FROM calls WHERE job=?', (job_id,)).fetchone()
            require(count < max_calls and spent + amount <= job['ceiling'], 'API job spending allowance exhausted')
            self.db.execute('INSERT INTO calls VALUES(?,?,?,?,?,NULL,NULL,NULL)',
                            (job_id, role, request_digest, amount, 'reserved_uncertain'))

    def settle(self, job_id, role, response_id, usage, actual):
        _integer(actual, 0, 1000000000, 'settled cost')
        with self.transaction():
            row = self.db.execute('SELECT * FROM calls WHERE job=? AND role=?', (job_id, role)).fetchone()
            require(row is not None and row['state'] == 'reserved_uncertain', 'Call settlement differs from its reservation')
            require(actual <= row['reserved'], 'Provider usage exceeds conservative reservation; inspect before more work')
            self.db.execute('UPDATE calls SET state=?,response_id=?,actual=?,usage=? WHERE job=? AND role=?',
                            ('settled', response_id, actual, _canonical(usage).decode(), job_id, role))

    def call(self, job_id, role):
        row = self.db.execute('SELECT * FROM calls WHERE job=? AND role=?', (job_id, role)).fetchone()
        return dict(row) if row is not None else None


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
                 model=MODEL, max_output_tokens=16384):
        slug(job_id)
        require(model == MODEL, 'Model lacks an approved pinned price profile')
        _integer(max_calls, 1, 12, 'call ceiling')
        require(type(timeout) in (int,float) and math.isfinite(timeout) and 1 <= timeout <= 600, 'Invalid API timeout')
        self.reservation = reserve_microdollars(max_output_tokens)
        self.ledger = ledger or OpenAILedger()
        self.ledger.binding(job_id, job_digest, max_calls)
        self.root = _private(root)
        self.job_id, self.job_digest, self.max_calls = job_id, job_digest, max_calls
        self.timeout = timeout
        self.settings = {'schema_version':1, 'execution_kind':'openai_responses_api', 'model':model,
            'max_calls':max_calls, 'timeout':timeout, 'max_output_tokens':max_output_tokens,
            'job_id':job_id, 'job_digest':job_digest, 'automatic_retries':0, 'tools':[], 'store':False,
            'service_tier':'default', 'reasoning':{'effort':'none'}, 'truncation':'disabled', 'pricing':PRICING, 'implementation_sha256':sha256(Path(__file__)), 'reservation_microdollars_per_call':self.reservation,
            'reservation_policy':'Full1.05Mcontext at long-context rates plus maximum output; only explicit settled-parent retirement releases unused holds.'}
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
        with self._lock():
            out = self.root/role
            prior = self.ledger.call(self.job_id,role)
            if out.exists():
                require(not out.is_symlink() and read_json(out/'request.json') == request, 'Cached API inputs changed')
                receipt = read_json(out/'receipt.json')
                require(receipt.get('status') == 'succeeded' and receipt['artifacts'] == _call_artifacts(out),
                        'API call failed, uncertain or changed; no automatic resubmission')
                require(prior and prior['state']=='settled' and prior['request_digest']==digest(request)
                        and prior['response_id']==receipt['response_id'], 'API cached ledger custody differs')
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
            payload = {'model':MODEL, 'instructions':INSTRUCTIONS, 'input':[{'role':'user','content':content}],
                       'text':{'format':{'type':'json_schema','name':'avatar_'+role.replace('-','_'),'schema':schema,'strict':True}},
                       'max_output_tokens':self.settings['max_output_tokens'], 'reasoning':{'effort':'none'},
                       'store':False,'tools':[],'tool_choice':'none','service_tier':'default','truncation':'disabled'}
            require(len(_canonical(payload)) <= MAX_REQUEST, 'API request exceeds bound')
            write_json(out/'payload-metadata.json',{'sha256':digest(payload),'bytes':len(_canonical(payload)),
                       'instructions':INSTRUCTIONS,'endpoint':'https://api.openai.com/v1/responses'})
            self.ledger.reserve(self.job_id,self.job_digest,self.max_calls,role,digest(request),self.reservation)
            started = time.monotonic()
            try:
                raw = post_response(payload,self.timeout,key)
                require(len(raw)<=MAX_RESPONSE,'API response exceeds bound')
                (out/'response.json').write_bytes(raw)
                response = _decode(raw)
                require(isinstance(response, dict), 'Invalid API response object')
                require(response.get('status')=='completed' and response.get('model')==MODEL
                        and response.get('service_tier')=='default', 'Incomplete or unexpected API response')
                rid = response.get('id')
                require(isinstance(rid,str) and rid.startswith('resp_') and len(rid)<=256,'Invalid response identity')
                usage = response.get('usage',{})
                inp = _integer(usage.get('input_tokens'),0,1050000,'reported input tokens')
                output = _integer(usage.get('output_tokens'),0,self.settings['max_output_tokens'],'reported output tokens')
                _integer(usage.get('total_tokens'),inp+output,inp+output,'reported total tokens')
                cost = math.ceil(inp*(5 if inp>272000 else 2.5)+output*(22.5 if inp>272000 else 15))
                self.ledger.settle(self.job_id,role,rid,usage,cost)
                messages = response.get('output')
                require(isinstance(messages,list) and len(messages)==1 and messages[0].get('type')=='message'
                        and messages[0].get('role')=='assistant' and messages[0].get('status')=='completed',
                        'Response contains unexpected tool activity or message structure')
                parts = messages[0].get('content')
                require(isinstance(parts,list) and len(parts)==1 and parts[0].get('type')=='output_text',
                        'Response refused or omitted structured text')
                answer_text = parts[0].get('text')
                require(isinstance(answer_text, str) and len(answer_text.encode()) <= MAX_ANSWER_BYTES,
                        'Structured answer exceeds1MiB')
                answer = _decode(answer_text.encode()); _validate(answer,schema)
                write_json(out/'answer.json',answer)
                write_json(out/'receipt.json',{'schema_version':1,'status':'succeeded','execution_kind':'openai_responses_api',
                    'response_id':rid,'model':MODEL,'usage':usage,'actual_microdollars_upper_estimate':cost,
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
