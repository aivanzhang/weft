"""Owner-configured API selection; readiness never creates grants or reads secrets."""

from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat

from avatar_assets.openai_runner import MODEL, reserve_microdollars
from avatar_assets.safety import require

DEFAULT = {'provider': 'openai', 'model': MODEL, 'max_calls': 5, 'max_repairs': 1,
           'timeout': 240, 'max_output_tokens': 16384}


def validate_settings(value):
    require(isinstance(value, dict) and set(value) == set(DEFAULT), 'Invalid server AI configuration fields')
    require(value['provider'] == 'openai' and value['model'] == MODEL, 'Unsupported server AI provider or model')
    for field, low, high in (('max_calls', 3, 12), ('max_repairs', 0, 2),
                             ('timeout', 30, 600), ('max_output_tokens', 256, 32768)):
        require(type(value[field]) is int and low <= value[field] <= high, 'Invalid server AI '+field)
    require(value['max_calls'] >= 3 + 2 * value['max_repairs'], 'AI call cap cannot cover the selected repair limit')
    return dict(value)


def _private_file(path, maximum, minimum=1):
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'Private configuration path contains a symlink')
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
            and stat.S_IMODE(info.st_mode) == 0o600 and minimum <= info.st_size <= maximum,
            'Private configuration must be a bounded owner-only regular file')
    return info


def server_settings(root):
    path = Path(root) / '.local/openai-backend.json'
    if not path.exists() and not path.is_symlink():
        return dict(DEFAULT)
    _private_file(path, 4096)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        value = json.loads(stream.read(4097))
    return validate_settings(value)


def pipeline_limits(settings):
    settings = validate_settings(settings)
    return {'max_repairs': settings['max_repairs'], 'max_agent_calls': settings['max_calls'],
            'agent_timeout': settings['timeout'], 'agent_model': settings['model']}


def readiness(root, *, job_id=None, job_digest=None, settings=None):
    root = Path(root)
    result = {'provider': 'openai', 'configured': False, 'key_ready': False,
              'project_budget_ready': False, 'job_grant_ready': False, 'can_run': False,
              'currency': 'USD', 'mesh_generation_currency': 'Meshy credits',
              'reason': 'Administrator OpenAI backend configuration is required.'}
    try:
        current = server_settings(root)
        if settings is not None:
            require(validate_settings(settings) == current, 'Saved AI backend differs from current owner configuration')
        result.update(model=current['model'], max_calls=current['max_calls'],
                      max_repairs=current['max_repairs'],
                      reservation_microdollars_per_call=reserve_microdollars(current['max_output_tokens']))
        result['maximum_reservation_microdollars'] = result['reservation_microdollars_per_call'] * current['max_calls']
        config = root / '.local/openai-backend.json'
        result['configured'] = config.is_file()
        key = root.parents[1] / '.local/credentials/openai-api-key'
        try:
            _private_file(key, 4096, 10)
            result['key_ready'] = True
        except (OSError, ValueError):
            pass
        ledger = root / '.local/openai/ledger.sqlite3'
        if ledger.exists() or ledger.is_symlink():
            _private_file(ledger, 64 * 1024**2)
            with closing(sqlite3.connect(ledger.as_uri() + '?mode=ro', uri=True, timeout=.2)) as db:
                db.execute('PRAGMA query_only=ON')
                budget = db.execute('SELECT ceiling FROM budget WHERE id=1').fetchone()
                result['project_budget_ready'] = budget is not None and budget[0] > 0
                global_reserved = db.execute('SELECT COALESCE(SUM(reserved),0) FROM calls').fetchone()[0]
                result['project_remaining_microdollars'] = max(0, budget[0] - global_reserved) if budget else 0
                if job_id is not None:
                    grant = db.execute('SELECT digest,calls,ceiling FROM jobs WHERE id=?', (job_id,)).fetchone()
                    result['job_grant_ready'] = bool(grant and grant[0] == job_digest and grant[1] == current['max_calls'])
                    if result['job_grant_ready']:
                        count, reserved = db.execute('SELECT COUNT(*),COALESCE(SUM(reserved),0) FROM calls WHERE job=?', (job_id,)).fetchone()
                        result.update(job_ceiling_microdollars=grant[2], reserved_microdollars=reserved,
                                      calls_reserved=count, remaining_calls=max(0, grant[1] - count),
                                      job_remaining_microdollars=max(0, grant[2] - reserved))
        if not result['configured']:
            return result
        if not result['key_ready']:
            result['reason'] = 'Administrator OpenAI API key configuration is required.'
        elif not result['project_budget_ready']:
            result['reason'] = 'An explicit administrator OpenAI dollar allowance is required; Meshy credits do not cover AI calls.'
        elif job_id is None:
            result['reason'] = 'OpenAI backend configured. Each saved request still needs its own administrator dollar grant.'
        elif not result['job_grant_ready']:
            result['reason'] = 'An explicit OpenAI dollar grant bound to this saved request is required before planning.'
        elif (result['remaining_calls'] < 1
              or min(result['job_remaining_microdollars'], result['project_remaining_microdollars'])
                 < result['reservation_microdollars_per_call']):
            result['reason'] = 'The remaining OpenAI call or dollar allowance cannot fund another bounded call; no automatic increase.'
        else:
            result.update(can_run=True, reason='OpenAI request grant can fund another bounded call; each call rechecks its remaining allowance.')
    except (OSError, ValueError, TypeError, sqlite3.Error):
        result['reason'] = 'OpenAI backend configuration or allowance requires administrator inspection.'
    return result


def request_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
