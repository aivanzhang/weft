"""Read-only custody for reusing a completed text-generated source after proposal rejection."""
from pathlib import Path

from avatar_assets.safety import local_file, read_json, require, sha256

POLICY = 'completed-generated-source-repair-v1'
GEOMETRY_ERROR = 'Authored geometry preflight failed; inspect all measured layer and bone findings'


def _file(parent, relative, limit=4*1024**2):
    path = parent/relative
    require(local_file(path, limit) == path and path.is_relative_to(parent),
            'Generated repair evidence contains an unsafe path')
    return path


def _json(parent, relative):
    return read_json(_file(parent, relative))


def _pin(parent, relative, limit=4*1024**2):
    path = _file(parent, relative, limit)
    return {'path': relative, 'bytes': path.stat().st_size, 'sha256': sha256(path)}


def _receipts(parent, stage, kind):
    folder = parent/'generation/stages'/stage
    require(folder.resolve() == folder and folder.is_dir(), 'Unsafe generation stage')
    rows = sorted(folder.glob(kind+'-*.json'))
    require(1 <= len(rows) <= 128 and
            [row.name for row in rows] == [f'{kind}-{i:03d}.json' for i in range(len(rows))],
            'Generation receipt sequence is missing or outside bounds')
    return rows


def generated_source(parent, request):
    """Authenticate local provider receipts; the caller also seals inventory and paid-agent custody."""
    from avatar_assets.meshy import stage_request
    from avatar_assets.meshy_access import digest
    parent = Path(parent)
    require(parent.is_absolute() and parent.resolve() == parent and request.get('reuse_model') is None,
            'Generated repair requires an unchanged fresh-source parent')
    # Image generation has a different input-redaction contract; admit it only with dedicated proof.
    require(request.get('reference') is None, 'Generated-source repair currently supports text-generated sources only')
    plan = _json(parent, 'plan.json')
    generation = _json(parent, 'generation/plan.json')
    config = _json(parent, 'generation-config.json')
    require(generation.get('config') == config == _json(parent, 'generation/inputs/config.json')
            and config.get('id') == request['id']+'-mesh'
            and config.get('prompt') == plan['generation_prompt']
            and config.get('subject') == plan['subject'] and config.get('style') == plan['style']
            and config.get('height_meters') == plan['height_meters']
            and config.get('rig') == 'none' and config.get('states') == []
            and config.get('generation', {}).get('mode') == 'text',
            'Generated source differs from the paid plan or has provider rigging')
    plan_hash = sha256(_file(parent, 'generation/plan.json'))
    require(_json(parent, 'generation-receipt.json') == {'plan_sha256': plan_hash}
            and generation.get('automatic_paid_retries') == 0
            and generation.get('model') == 'meshy-7.1', 'Generated source plan receipt changed')
    sources = generation.get('sources')
    require(isinstance(sources, list) and len(sources) == 2
            and {row.get('path') for row in sources} == {'meshy.py', 'meshy_access.py'},
            'Generation implementation receipt is missing')
    expected_sources = {row['path']: row['sha256'] for row in sources}
    for row in sources:
        require(_pin(parent, 'generation/sources/'+row['path']) ==
                {**row, 'path': 'generation/sources/'+row['path']}, 'Generation source snapshot changed')
    completed = {}
    class Dependencies:
        def call(self, job, stage):
            require(job == config['id'] and stage in completed, 'Generation dependency is not completed')
            return {'state': 'SUCCEEDED', 'task_id': completed[stage]['task_id']}
    total = 0
    receipt_paths = ['generation-config.json', 'generation/plan.json', 'generation-receipt.json']
    for stage in ('mesh', 'texture'):
        request_paths = _receipts(parent, stage, 'request')
        submission_paths = _receipts(parent, stage, 'submission')
        require(len(request_paths) == len(submission_paths) == 1, 'Generated repair refuses paid retries')
        prefix = 'generation/stages/'+stage+'/'
        recorded_request = _json(parent, prefix+'request-000.json')
        submission = _json(parent, prefix+'submission-000.json')
        endpoint, body, credits = stage_request(parent/'generation', generation, stage, Dependencies())
        require(recorded_request.get('endpoint') == endpoint and recorded_request.get('body') == body
                and recorded_request.get('request_sha256') == digest(body)
                and recorded_request.get('estimated_credits') == submission.get('estimated_credits') == credits,
                'Generated stage request differs from the bound plan')
        statuses = _receipts(parent, stage, 'status')
        last = _json(parent, str(statuses[-1].relative_to(parent)))
        task = last.get('task', {})
        task_id = submission.get('task_id')
        require(isinstance(task_id, str) and 1 <= len(task_id) <= 128
                and task.get('id') == task_id and task.get('status') == 'SUCCEEDED'
                and task.get('task_error') is None and type(task.get('consumed_credits')) is int
                and task['consumed_credits'] == credits, 'Generated stage is not a completed known-charge source')
        for receipt in (recorded_request, submission, last):
            require(receipt.get('execution_sources') == expected_sources, 'Generated stage implementation changed')
        for name, checksum in expected_sources.items():
            require(sha256(_file(parent, prefix+'tools/'+checksum[:16]+'-'+name)) == checksum,
                    'Generated stage execution snapshot changed')
        downloaded = _json(parent, prefix+'downloads/manifest.json')
        require(downloaded.get('kind') == 'meshy_original_downloads'
                and downloaded.get('status') == 'downloaded_not_visually_accepted'
                and downloaded.get('stage') == stage and downloaded.get('task_id') == task_id
                and downloaded.get('plan_sha256') == plan_hash,
                'Generated download receipt belongs to another source')
        rows = downloaded.get('artifacts')
        require(isinstance(rows, list) and 1 <= len(rows) <= 8 and
                len({row.get('path') for row in rows}) == len(rows), 'Invalid generated artifact inventory')
        for row in rows:
            name = row.get('path')
            require(isinstance(name, str) and name in ('model.glb', 'thumbnail.png'),
                    'Unexpected generated source artifact')
            actual = _pin(parent, prefix+'downloads/'+name, 64*1024**2)
            require(actual == {**row, 'path': prefix+'downloads/'+name}, 'Generated artifact changed')
            saved = downloaded.get('downloads', {}).get(name, {})
            require(saved.get('sha256') == row['sha256'] and saved.get('bytes') == row['bytes'],
                    'Generated download hash differs')
        require(any(row['path'] == 'model.glb' for row in rows), 'Generated model download is missing')
        completed[stage] = {'task_id': task_id, 'manifest_sha256': sha256(_file(parent, prefix+'downloads/manifest.json'))}
        total += credits
        receipt_paths += [prefix+'request-000.json', prefix+'submission-000.json',
                          str(statuses[-1].relative_to(parent)), prefix+'downloads/manifest.json']
    costs = _json(parent, 'generation-cost.json')
    require(costs == {'reserved_credits': total, 'provider_reported_credits': total,
                     'paid_submissions': 2, 'estimated_credits': total, 'usd_cost': None}
            and total <= request['max_credits'], 'Generated source cost custody differs')
    source = _pin(parent, 'generation/stages/texture/downloads/model.glb', 64*1024**2)
    require(_json(parent, 'inspection/manifest.json').get('source_glb_sha256') == source['sha256'],
            'Generated source inspection belongs to another model')
    return {**source, 'policy': POLICY, 'generation_plan_sha256': plan_hash,
            'stages': completed, 'receipts': [_pin(parent, name) for name in receipt_paths+['generation-cost.json']]}


def geometry_failure(parent, request, plan, answer, source, attempt=0):
    """Recompute only diagnostics; never modify authored counts or manufacture a native pack."""
    from avatar_assets.author_geometry import author_diagnostics
    from avatar_assets.pipeline_prompts import _author_design, finish_config
    parent = Path(parent)
    require(type(attempt) is int and 0 <= attempt <= 2 and attempt == request['max_repairs'],
            'Geometry continuation must use the final bounded attempt')
    require(not any((parent/name).exists() for name in
                    (f'model-{attempt}', f'finish-{attempt}.json', f'agents/reviewer-{attempt}')),
            'Geometry-only continuation has unexpected native or review artifacts')
    inspection = _json(parent, 'inspection/manifest.json')
    path = _file(parent, source['path'], 64*1024**2)
    require(sha256(path) == source['sha256'] == inspection.get('source_glb_sha256'),
            'Geometry source custody changed')
    native = inspection['native']
    design = _author_design(answer)
    finish_config(answer, request, plan, {'path': str(path), 'sha256': source['sha256']}, attempt,
                  quality_policy=_json(parent, 'quality-policy.json'),
                  bone_parents={bone['name']: bone['parent'] for rig in native.get('rigs', []) for bone in rig['bones']})
    geometry = author_diagnostics(design, path, source['sha256'], native)
    require(geometry == _json(parent, f'author-geometry-{attempt}.json') and
            geometry.get('status') == 'available' and bool(geometry.get('failures')),
            'Geometry failure is missing or differs from exact measured diagnostics')
    findings = {'validation_failure': GEOMETRY_ERROR, 'measurements': {'author_geometry': geometry}}
    terminal = _json(parent, 'unsupported.json')
    require(terminal.get('stage') == 'repair_budget' and terminal.get('status') == 'unsupported'
            and terminal.get('findings') == findings,
            'Geometry terminal findings differ from measured failure')
    return findings
