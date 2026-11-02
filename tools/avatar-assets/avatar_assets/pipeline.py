"""Prompt-to-delivery orchestration; agents propose data, trusted tools execute it."""

from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid
from zoneinfo import ZoneInfo

from PIL import Image, ImageDraw

from avatar_assets.safety import artifacts, local_file, new_directory, read_json, require, sha256, slug, write_json

ROOT = Path(__file__).resolve().parents[1]
PRESETS = {
    "draft": {"size": 256, "portrait": 512, "seconds": 2, "samples": 4},
    "preview": {"size": 720, "portrait": 1600, "seconds": 4, "samples": 8},
    "muse": {"size": 720, "portrait": 1600, "seconds": 8, "samples": 8},
}
MAX_OUTPUT = 2 * 1024**3
MAX_STAGE_START_SECONDS = 3600
PIPELINE_SOURCES = ('pipeline.py', 'pipeline_prompts.py', 'agent_runner.py',
                    'action_intent.py', 'quality_policy.py', 'pipeline_evidence.py', 'action_audit.py',
                    'deformation_gate.py')
PIPELINE_SCRIPT_SOURCES = ('motion_samples.py',)


class InterruptedStage(ValueError):
    """An unsealed stage is not evidence that its native process exited."""


class EvidenceCustodyError(ValueError):
    """Missing or changed bound evidence is not an authoring defect to repair."""


class PipelineCancelled(ValueError):
    """Cooperative cancellation preserves the current bounded stage and receipts."""


def now():
    return datetime.now(ZoneInfo("America/Los_Angeles")).isoformat()


def _json_digest(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _pipeline_sources():
    return {**{name: ROOT/'avatar_assets'/name for name in PIPELINE_SOURCES},
            **{'scripts/'+name: ROOT/'scripts'/name for name in PIPELINE_SCRIPT_SOURCES}}


def _text(value, limit, label, empty=False):
    require(isinstance(value, str) and (empty or bool(value.strip())) and len(value) <= limit,
            "Invalid " + label)
    return value


def _atomic(path, value):
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _image(path):
    path = local_file(path, 8*1024**2)
    require(path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'), 'Reference must be PNG/JPEG/WebP')
    with Image.open(path) as image:
        require(image.width*image.height <= 16_777_216 and getattr(image, 'n_frames', 1) == 1,
                'Reference must be a single image no larger than16MP')
        image.verify()
    return path


def prepare(prompt, output, *, style='', states=(), custom_actions=(), preset='preview', reference=None, reuse_model=None,
            max_credits=0, max_repairs=1, max_agent_calls=10, agent_timeout=240, agent_model=None):
    from avatar_assets.action_intent import validate_custom_actions
    _text(prompt, 4000, 'prompt')
    _text(style, 800, 'style', empty=True)
    custom_actions = validate_custom_actions(list(custom_actions) if isinstance(custom_actions, tuple) else custom_actions)
    require(preset in PRESETS, 'Unknown output preset')
    require(isinstance(states, (tuple, list)) and len(states) <= 8, 'At most eight states')
    states = [slug(s) for s in states]
    require(len(set(states)) == len(states), 'Duplicate requested state')
    require(not states or not custom_actions or len(states) == len(custom_actions),
            'State IDs and custom actions must have matching lengths')
    require(type(max_credits) is int and 0 <= max_credits <= 500, 'Credit cap must be0–500')
    require(type(max_repairs) is int and 0 <= max_repairs <= 2, 'At most two local repairs')
    require(type(max_agent_calls) is int and 3 <= max_agent_calls <= 12, 'Agent calls must be3–12')
    require(type(agent_timeout) is int and 30 <= agent_timeout <= 600, 'Agent deadline must be30–600s')
    if agent_model is not None:
        _text(agent_model, 100, 'agent model')
    output = Path(output)
    require(output.resolve().is_relative_to(ROOT/'outputs'), 'Pipeline jobs must stay under this tool outputs/')
    identifier = slug(output.name)
    require(len(identifier) <= 48, 'Job ID must leave room for stage suffixes')
    ref = _image(reference) if reference else None
    source = local_file(reuse_model, 64*1024**2) if reuse_model else None
    if source:
        from avatar_assets.provider_model import preflight
        preflight(source)
    out = new_directory(output)
    (out/'inputs').mkdir()
    (out/'events').mkdir()
    (out/'sources').mkdir()
    request = {'schema_version': 1, 'id': identifier, 'prompt': prompt, 'style': style,
               'states': states, 'custom_actions': custom_actions, 'preset': preset,
               'media_spec': PRESETS[preset], 'max_credits': max_credits,
               'max_repairs': max_repairs, 'max_agent_calls': max_agent_calls, 'agent_timeout': agent_timeout,
               'agent_model': agent_model, 'reference': None, 'reuse_model': None, 'created': now()}
    for original, name, field in ((ref, 'reference', 'reference'), (source, 'source-model', 'reuse_model')):
        if original:
            target = out/'inputs'/(name+original.suffix.lower())
            shutil.copyfile(original, target)
            request[field] = {'path': str(target.relative_to(out)), 'sha256': sha256(target),
                              'original_path': str(original), 'bytes': target.stat().st_size}
    write_json(out/'request.json', request)
    for name, source_path in _pipeline_sources().items():
        target = out/'sources'/name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source_path, target)
    write_json(out/'input-receipt.json', {'request_sha256': sha256(out/'request.json'), 'sources': artifacts(out/'sources')})
    write_json(out/'estimate.json', {
        'schema_version': 1, 'new_meshy_credits': 0 if source else 35,
        'new_meshy_paid_calls': 0 if source else (1 if ref else 2), 'automatic_paid_retries': 0,
        'maximum_agent_calls': max_agent_calls, 'expected_agent_calls': 3,
        'planning_token_estimate': {'input_per_call': 'Budget assumption50000 input tokens/call, informed by observed15814–42341 tokens including selected evidence; not an enforced token cap. Up to8 selected images.',
                                    'output_per_call': 'typically1000–8000 tokens; transport bytes/deadline are bounded'},
        'agent_billing': 'Existing authenticated local Codex CLI; account-specific usage/price unknown. No account/key creation or purchase.',
        'usd_cost': None, 'local_compute': 'No hosted compute. Preview/native media use4CPU threads/2GiB sampled child RSS.',
        'maximum_local_repairs': max_repairs, 'total_output_bytes': MAX_OUTPUT,
        'per_invocation_stage_start_deadline_seconds': MAX_STAGE_START_SECONDS,
        'deadline_scope': 'No new stage after this invocation deadline; resume opens another watch window, not another call/repair allowance. Each in-flight stage retains its bounded deadline.',
        'recurring_services': 0, 'new_meta_calls': 0,
    })
    return {'status': 'prepared', 'out': str(out), 'estimate': read_json(out/'estimate.json')}


@contextmanager
def _lock(root):
    with (root/'pipeline.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Pipeline job is already running') from None
        yield


class Pipeline:
    def __init__(self, folder, *, agents=None):
        self.root = Path(folder).resolve()
        require(self.root.is_relative_to(ROOT/'outputs') and not Path(folder).is_symlink(), 'Unsafe pipeline path')
        self.request = read_json(self.root/'request.json')
        receipt = read_json(self.root/'input-receipt.json')
        require(sha256(self.root/'request.json') == receipt['request_sha256'], 'Pipeline request changed')
        sealed_delivery = (self.root/'delivery-receipt.json').is_file() and (self.root/'delivery.json').is_file()
        sources = _pipeline_sources()
        names = [source['path'] for source in receipt['sources']]
        require(names and len(names) == len(set(names)) and set(names) <= set(sources),
                'Invalid pipeline source inventory')
        require(sealed_delivery or set(names) == set(sources),
                'Pipeline backend changed; this unfinished job lacks current source pins')
        for source in receipt['sources']:
            require(sha256(self.root/'sources'/source['path']) == source['sha256']
                    and (sealed_delivery or sha256(sources[source['path']]) == source['sha256']),
                    'Pipeline backend changed; preserve this job and prepare a new one')
        for field in ('reference', 'reuse_model'):
            item = self.request[field]
            if item:
                path = local_file(self.root/item['path'])
                require(path.is_relative_to(self.root/'inputs') and sha256(path) == item['sha256'], 'Prepared input changed')
        if agents is None:
            from avatar_assets.agent_runner import AgentRunner
            agents = AgentRunner(self.root/'agents', max_calls=self.request['max_agent_calls'],
                                 timeout=self.request['agent_timeout'], model=self.request['agent_model'])
        self.agents = agents
        self.started = time.monotonic()
        self.cancel_requested = None
        self._active_progress = None

    def event(self, stage, status, **details):
        entries = sorted((self.root/'events').glob('*.json'))
        require(len(entries) < 128, 'Pipeline event limit reached')
        key = (stage, details.get('attempt'))
        active = self._active_progress
        if status == 'running':
            require(active is None, 'The previous stage has no terminal progress receipt')
        elif active and active['key'] == key:
            details['elapsed_seconds'] = round(max(0, time.monotonic()-active['started']), 3)
        value = {'schema_version': 1, 'recorded': now(), 'stage': stage, 'status': status, **details}
        write_json(self.root/'events'/f'{len(entries):04d}.json', value)
        _atomic(self.root/'status.json', value)
        if status == 'running':
            self._active_progress = {'key': key, 'started': time.monotonic()}
        elif active and active['key'] == key:
            self._active_progress = None
        print(json.dumps({'stage': stage, 'status': status}), flush=True)

    def _fail_active_stage(self, error, status='failed'):
        # A failed outer operation must not leave its last stage displayed as live work.
        if self._active_progress:
            stage, attempt = self._active_progress['key']
            details = {'attempt': attempt} if attempt is not None else {}
            self.event(stage, status, reason=str(error)[:4000], **details)

    def guard(self, reserve_mb=0):
        if self.cancel_requested is not None and self.cancel_requested():
            raise PipelineCancelled('Stopped at a safe stage boundary; completed work and provider receipts are retained')
        require(time.monotonic()-self.started <= MAX_STAGE_START_SECONDS, 'Pipeline stage-start deadline reached')
        from avatar_assets.runner import _output_bytes
        require(_output_bytes(self.root) + reserve_mb*1024**2 <= MAX_OUTPUT, 'Pipeline aggregate output bound reached')

    def unsupported(self, stage, reason, **extra):
        self.event(stage, 'unsupported', reason=reason,
                   **({'attempt': extra['attempt']} if 'attempt' in extra else {}))
        value = {'schema_version': 1, 'status': 'unsupported', 'stage': stage, 'reason': reason,
                 'request_sha256': sha256(self.root/'request.json'), 'out': str(self.root), **extra}
        path = self.root/'unsupported.json'
        if not path.exists():
            write_json(path, value)
        return value

    def _pack(self, path, create):
        from avatar_assets.cli import verify_pack
        failure_path = self.root/('stage-failure-'+path.name+'.json')
        if not path.exists():
            self.guard(512)
            try:
                create()
            except Exception as error:
                if (path/'manifest.json').exists() and read_json(path/'manifest.json').get('status') == 'failed':
                    write_json(failure_path, {'manifest_sha256': sha256(path/'manifest.json'), 'error': str(error)})
                raise
        if not (path/'manifest.json').exists():
            raise InterruptedStage('Interrupted stage needs inspection; no automatic replay: '+path.name)
        try:
            manifest = read_json(path/'manifest.json')
            require(isinstance(manifest, dict), 'Stage manifest is not an object')
        except (ValueError, OSError, KeyError, TypeError) as error:
            raise EvidenceCustodyError('Stage manifest requires inspection: '+path.name) from error
        if manifest.get('status') == 'failed':
            try:
                require(manifest.get('artifacts') == artifacts(path), 'Failed stage artifacts changed')
                # The repair prompt must retain the same failure across process restarts.
                failure = read_json(failure_path) if failure_path.exists() else None
                if failure is not None:
                    require(failure['manifest_sha256'] == sha256(path/'manifest.json'), 'Failed stage receipt changed')
                reason = failure['error'] if failure is not None else manifest.get('error') or 'Retained failed stage: '+path.name
            except (ValueError, OSError, KeyError, TypeError) as error:
                raise EvidenceCustodyError('Failed stage evidence requires inspection: '+str(error)) from error
            raise ValueError(reason)
        try:
            verify_pack(path)
        except (ValueError, OSError, KeyError, TypeError) as error:
            raise EvidenceCustodyError('Stage evidence requires inspection: '+str(error)) from error
        return manifest

    def _plan(self):
        from avatar_assets.pipeline_prompts import PLAN_SCHEMA, plan_prompt, validate_plan
        self.guard()
        self.event('planner', 'running')
        images = [self.root/self.request['reference']['path']] if self.request['reference'] else []
        answer = self.agents.call('planner', plan_prompt(self.request), PLAN_SCHEMA, images=images)
        plan = validate_plan(answer, self.request)
        path = self.root/'plan.json'
        if path.exists():
            require(read_json(path) == plan, 'Cached planner result changed')
        else:
            write_json(path, plan)
        self.event('planner', plan['status'])
        return plan

    def _source(self, plan, execute):
        if self.request['reuse_model']:
            item = self.request['reuse_model']
            self.event('geometry', 'reused_generated_source', new_paid_calls=0)
            return {'path': str(self.root/item['path']), 'sha256': item['sha256']}
        from avatar_assets import meshy
        generation = self.root/'generation'
        config = {'schema_version': 1, 'id': self.request['id']+'-mesh',
                  'subject': plan['subject'], 'style': plan['style'], 'prompt': plan['generation_prompt'],
                  'height_meters': plan['height_meters'], 'rig': 'none', 'states': [], 'max_credits': 35,
                  'rights': 'User-requested generation. Preserve source prompt and reference provenance; hidden surfaces are generated assumptions; commercial rights not independently verified.',
                  'generation': {'mode': 'image' if self.request['reference'] else 'text',
                                 'geometry_resolution': '4k', 'texture_resolution': '4k',
                                 'target_polycount': 35000, 'pose_mode': ''}}
        if self.request['reference']:
            config.update(reference=self.request['reference']['path'], reference_sha256=self.request['reference']['sha256'])
        if not generation.exists():
            if not (self.root/'generation-config.json').exists():
                write_json(self.root/'generation-config.json', config)
            require(read_json(self.root/'generation-config.json') == config, 'Prepared generation configuration changed')
            meshy.prepare(self.root/'generation-config.json', generation)
            write_json(self.root/'generation-receipt.json', {'plan_sha256': sha256(generation/'plan.json')})
        require(read_json(self.root/'generation-config.json') == config
                and read_json(generation/'inputs/config.json') == config
                and sha256(generation/'plan.json') == read_json(self.root/'generation-receipt.json')['plan_sha256'],
                'Generation plan changed; no paid request is allowed')
        if not execute or self.request['max_credits'] < 35:
            self.event('geometry', 'awaiting_generation_authorization', estimated_credits=35,
                       prepared_job=str(generation), reason='Execution and a35-credit job cap are required; the canonical ledger also applies.')
            return None
        self.event('geometry', 'running', estimated_credits=35)
        self.guard(512)
        result = meshy.run(generation, max_seconds=900)
        if result['status'] == 'checkpointed':
            self.event('geometry', 'checkpointed', provider=result)
            return None
        from avatar_assets.meshy_access import Ledger
        ledger = Ledger()
        try:
            calls = [ledger.call(config['id'], stage) for stage in result['completed']]
            reported = [json.loads(row['response']).get('consumed_credits') if row['response'] else None for row in calls]
            costs = {'reserved_credits': sum(row['credits'] for row in calls),
                     'provider_reported_credits': sum(reported) if all(type(v) is int for v in reported) else None,
                     'paid_submissions': len(calls), 'estimated_credits': 35, 'usd_cost': None}
        finally:
            ledger.close()
        if not (self.root/'generation-cost.json').exists():
            write_json(self.root/'generation-cost.json', costs)
        require(costs['reserved_credits'] <= self.request['max_credits'],
                'Actual provider credit floor exceeded the job cap; reconcile preserved cost before further work')
        stage = 'mesh' if self.request['reference'] else 'texture'
        path = generation/'stages'/stage/'downloads/model.glb'
        source = {'path': str(local_file(path)), 'sha256': sha256(path)}
        self.event('geometry', 'complete', reason='Generated source saved; rig, motion and quality review are pending.')
        return source

    def _contacts(self, attempt, pack):
        folder = self.root/f'evidence-{attempt}'
        if not folder.exists():
            folder.mkdir()
            (folder/'states').mkdir()
            report = read_json(pack/'finish-report.json')
            for state in report['states']:
                paths = sorted((pack/'renders'/state['id']).glob('*.png'))
                require(2 <= len(paths) <= 8, 'Missing bounded sampled poses')
                columns, tile, row_height = min(4, len(paths)), 720, 748
                sheet = Image.new('RGB', (columns*tile, ((len(paths)+columns-1)//columns)*row_height+24), 'white')
                ImageDraw.Draw(sheet).text((5,4), state['id'], fill='black')
                for i, path in enumerate(paths):
                    x, y = (i%columns)*tile, 24+(i//columns)*row_height
                    with Image.open(path) as image:
                        require(image.width*image.height <= 16_777_216, 'Oversized pose image')
                        image = image.convert('RGB'); image.thumbnail((tile,tile))
                        sheet.paste(image, (x,y))
                    second = state.get('seconds', [f/24 for f in state['frames']])[i]
                    ImageDraw.Draw(sheet).text((x+5,y+tile+4), f'{second:.3f}s', fill='black')
                target = folder/'states'/(state['id']+'.png')
                sheet.save(target)
                require(target.stat().st_size <= 8*1024**2, 'Repair image exceeds bounded agent attachment size')
            write_json(folder/'evidence.json', {'images': artifacts(folder),
                                               'finish_report_sha256': sha256(pack/'finish-report.json')})
        record = read_json(folder/'evidence.json')
        require(record['finish_report_sha256'] == sha256(pack/'finish-report.json'), 'Contact source changed')
        for item in record['images']:
            require(sha256(folder/item['path']) == item['sha256'], 'Review evidence changed')
        return folder, [folder/item['path'] for item in record['images']]

    def _diagnostics(self, pack):
        result = {}
        if (pack/'surface-integrity.json').exists():
            result['surface_integrity'] = read_json(pack/'surface-integrity.json')
        if (pack/'rig-verification.json').exists():
            result['skin_motion'] = read_json(pack/'rig-verification.json')['motion']
        if (pack/'finish-report.json').exists():
            report = read_json(pack/'finish-report.json')
            rig = report.get('authored_rig') or {}
            result['weight_coverage'] = {key: rig[key] for key in
                ('input_world_bounds', 'layers', 'positive_vertices_per_bone') if key in rig}
            result['sampled_states'] = report.get('states', [])
        return result

    def _evidence(self, attempt, pack, preview):
        folder, images = self._contacts(attempt, pack)
        images += [preview/'front.png', preview/'three-quarter.png', pack/'portrait.png']
        extra_count = 3
        if self.request['reference']:
            images.append(self.root/self.request['reference']['path'])
            extra_count += 1
        # Up to eight requested states use one contact montage plus geometry views.
        if len(images) > 8:
            contacts = images[:-extra_count]
            montage = folder/'motion-montage.png'
            if not montage.exists():
                sheet = Image.new('RGB', (1280, ((len(contacts)+1)//2)*372), 'white')
                for i, path in enumerate(contacts):
                    with Image.open(path) as image:
                        image.thumbnail((640,372)); sheet.paste(image,((i%2)*640,(i//2)*372))
                sheet.save(montage)
            images = [montage]+images[-extra_count:]
        return [{'id': f'image-{i:02d}-{p.name}', 'path': str(p), 'sha256': sha256(p),
                 'state_ids': [s['id'] for s in read_json(pack/'manifest.json')['states']] if p == folder/'motion-montage.png' else
                              [p.stem] if p.parent == folder/'states' else [],
                 'kind': 'exported_glb_rest_view' if p.parent == preview else
                         'native_hair_portrait' if p == pack/'portrait.png' else
                         'user_reference' if p.parent == self.root/'inputs' else 'native_motion_contact_sheet'}
                for i, p in enumerate(images)]

    def run(self, *, execute=False, plan_only=False, cancel_requested=None):
        require(type(plan_only) is bool, 'plan_only must be boolean')
        require(cancel_requested is None or callable(cancel_requested), 'Invalid cancellation callback')
        self.cancel_requested = cancel_requested
        with _lock(self.root):
            try:
                return self._run(execute, plan_only=plan_only)
            except PipelineCancelled as error:
                self._fail_active_stage(error, 'cancelled')
                self.event('pipeline', 'cancelled', reason=str(error))
                return {'status': 'cancelled', 'reason': str(error), 'out': str(self.root)}
            except Exception as error:
                self._fail_active_stage(error)
                self.event('pipeline', 'checkpointed_error', error=str(error), automatic_retry=False)
                raise

    def _action_audit(self, attempt, pack, plan, answer):
        from avatar_assets.pipeline_prompts import extract_action_bindings
        from avatar_assets.runner import run_bounded
        from avatar_assets.safety import preserve_failure
        nodes = read_json(pack/'exported-nodes.json')
        by_id = {node['id']: node for node in nodes['nodes']}
        joints = set(nodes['skins'][0]['joints'])
        parents = {}
        for identifier in joints:
            node, seen = by_id[identifier], set()
            parent = node['parent_id']
            while parent is not None and parent not in joints:
                require(parent not in seen, 'Exported hierarchy contains a cycle')
                seen.add(parent)
                parent = by_id[parent]['parent_id']
            require(node['name'] not in parents, 'Exported joint names are not unique')
            parents[node['name']] = by_id[parent]['name'] if parent is not None else None
        bindings = extract_action_bindings(answer, plan, parents)
        folder = self.root/f'action-audit-{attempt}'
        def create():
            new_directory(folder)
            try:
                write_json(folder/'intent.json', {'schema_version': 1, 'states': plan['states'],
                    'action_bindings': bindings, 'height_meters': plan['height_meters'],
                    'source_sha256': sha256(pack/'model.glb')})
                job = run_bounded([sys.executable, '-m', 'avatar_assets.action_audit', str(pack/'model.glb'),
                                   str(folder/'intent.json'), str(folder/'report.json')],
                                  folder/'audit.log', folder, timeout=60, memory_mb=768, output_mb=32)
                write_json(folder/'job.json', job)
                require(job['exit_code'] == 0 and not job['failure'], 'Weighted action audit failed; inspect retained evidence')
                write_json(folder/'manifest.json', {'schema_version': 1, 'kind': 'weighted_action_audit',
                    'status': 'measured', 'model_sha256': sha256(pack/'model.glb'),
                    'implementation_sha256': sha256(ROOT/'avatar_assets/action_audit.py'), 'artifacts': artifacts(folder)})
            except Exception as error:
                preserve_failure(folder, error)
                raise
        proof = self._pack(folder, create)
        require(proof['model_sha256'] == sha256(pack/'model.glb'), 'Action audit source changed')
        return read_json(folder/'report.json')

    def _full_evidence(self, attempt, pack, exported, media):
        from avatar_assets.runner import run_bounded
        from avatar_assets.safety import preserve_failure
        folder = self.root/f'final-evidence-{attempt}'
        def create():
            command = [sys.executable, '-m', 'avatar_assets.pipeline_evidence', str(pack), str(exported), str(media), str(folder)]
            if self.request['reference']:
                command.append(str(self.root/self.request['reference']['path']))
            # The child creates the pack; its control log remains outside that sealed inventory.
            job = run_bounded(command, self.root/f'evidence-worker-{attempt}.log', self.root,
                              timeout=120, memory_mb=768, output_mb=2048)
            write_json(self.root/f'evidence-worker-{attempt}.json', job)
            if job['exit_code'] or job['failure']:
                if folder.exists() and not (folder/'manifest.json').exists():
                    preserve_failure(folder, ValueError('Full-resolution evidence worker failed'))
                raise ValueError('Full-resolution evidence worker failed; inspect retained job')
        proof = self._pack(folder, create)
        require(proof['model_manifest_sha256'] == sha256(pack/'manifest.json')
                and proof['exported_manifest_sha256'] == sha256(exported/'manifest.json')
                and proof['media_manifest_sha256'] == sha256(media/'manifest.json'), 'Final review inputs changed')
        return [{**item, 'path': str(folder/item['path'])} for item in proof['images']], proof['coverage']

    def _deformation(self, attempt, pack, plan, source, config_path):
        from avatar_assets.deformation_gate import audit_finish_pack
        path = self.root/f'deformation-gate-{attempt}.json'
        try:
            report = audit_finish_pack(pack, expected_source_sha256=source['sha256'],
                expected_design_sha256=sha256(config_path), expected_state_ids=[s['id'] for s in plan['states']],
                authored_height=plan['height_meters'])
            if not path.exists():
                events = sorted((self.root/'events').glob('*.json'))
                require(len(events) <= 128, 'Pipeline event inventory exceeds its bound')
                history = [read_json(event) for event in events]
                # Only the current running event can precede a first write; prior work needs its original proof.
                if history and history[-1].get('stage') == 'deformation_gate' and history[-1].get('attempt') == attempt \
                        and history[-1].get('status') == 'running':
                    history = history[:-1]
                require(not any(event.get('stage') == 'deformation_gate' and event.get('attempt') == attempt
                                for event in history), 'Previously started deformation gate has no saved evidence')
                write_json(path, report)
            require(read_json(path) == report, 'Cached deformation gate evidence changed')
        except (ValueError, OSError, KeyError, TypeError) as error:
            raise EvidenceCustodyError('Deformation evidence requires inspection; no automatic repair: '+str(error)) from error
        require(report['passed'], 'Severe sampled skin deformation: '+json.dumps(report['states']))
        return report

    @staticmethod
    def _bind_review(review, evidence, action_audit=None):
        if review['verdict'] != 'accept':
            return
        from avatar_assets.pipeline_prompts import review_citations
        citations = review_citations(evidence, action_audit)
        visual = {e['id'] for e in evidence if e.get('kind') != 'user_reference'}
        for check in review.get('checks', []):
            require(set(check['evidence_ids']) & visual, 'Visual acceptance needs an actual output image')
        for state in review['state_reviews']:
            motion = {e['id'] for e in evidence if state['id'] in e.get('state_ids', [])}
            allowed = {e['id'] for e in citations if state['id'] in e['state_ids']}
            require(set(state['evidence_ids']) & motion and set(state['evidence_ids']) <= allowed,
                    'State acceptance needs its own actual motion evidence')
            for requirement in state.get('requirement_reviews', []):
                require(set(requirement['evidence_ids']) & motion and set(requirement['evidence_ids']) <= allowed,
                        'Each action requirement needs its own state motion image; numeric proof alone is insufficient')

    def _run(self, execute, *, plan_only=False):
        from avatar_assets import pipeline_prompts as prompts
        from avatar_assets.cli import verify_pack
        from avatar_assets.finish_generated import finish
        from avatar_assets.provider_model import preview
        from avatar_assets.review import reload_3d
        from avatar_assets.quality_policy import build_quality_policy, evaluate_quality_evidence
        if (self.root/'delivery.json').exists():
            receipt = read_json(self.root/'delivery-receipt.json')
            for name, checksum in receipt.items():
                require(sha256(self.root/name) == checksum, 'Delivery evidence changed')
            delivery = read_json(self.root/'delivery.json')
            require(delivery['status'] == 'complete' and 5 <= len(delivery['packs']) <= 8
                    and delivery['request_sha256'] == sha256(self.root/'request.json'), 'Invalid completed delivery')
            for item in delivery['packs']:
                require(sha256(self.root/item['path']/'manifest.json') == item['manifest_sha256'], 'Delivery manifest changed')
                verify_pack(self.root/item['path'])
            return {**delivery, 'resumed_without_generation': True}
        if (self.root/'unsupported.json').exists():
            result = read_json(self.root/'unsupported.json')
            require(result['request_sha256'] == sha256(self.root/'request.json'), 'Unsupported result request mismatch')
            _text(result['reason'], 4000, 'unsupported reason')
            return result
        self.guard()
        plan = self._plan()
        if plan['status'] == 'unsupported':
            return self.unsupported('planner', plan['reason'])
        self.guard()
        source = self._source(plan, execute and not plan_only)
        if plan_only and source is not None:
            self.event('pipeline', 'planned', new_paid_calls=0)
            return {'status': 'planned', 'out': str(self.root)}
        if source is None:
            return {'status': read_json(self.root/'status.json')['status'], 'out': str(self.root)}
        inspection = self.root/'inspection'
        self.event('inspection', 'running')
        inspected = self._pack(inspection, lambda: preview(source['path'], inspection, resolution=512, samples=8, rig_guides=True))
        require(inspected['source_glb_sha256'] == source['sha256'], 'Preview belongs to a different source')
        context = {'preflight': inspected['preflight'], 'native': inspected['native']}
        quality = build_quality_policy(self.request, plan, context)
        if not (self.root/'quality-policy.json').exists():
            write_json(self.root/'quality-policy.json', quality)
        require(read_json(self.root/'quality-policy.json') == quality, 'Prepared quality policy changed')
        source_parents = {b['name']: b['parent'] for rig in context['native'].get('rigs', []) for b in rig['bones']}
        self.event('inspection', 'complete', reason='Source geometry and review views inspected; no quality acceptance.')
        previous, findings, selected, repair_images = None, None, None, []
        for attempt in range(self.request['max_repairs']+1):
            self.guard()
            self.event('author', 'running', attempt=attempt)
            answer = self.agents.call('author-'+str(attempt), prompts.author_prompt(self.request, plan, context, previous, findings),
                                      prompts.AUTHOR_SCHEMA, images=[inspection/n for n in
                                          ('guide-front.png','guide-side.png','three-quarter.png')]+repair_images[:5])
            if answer.get('status') == 'unsupported':
                _text(answer['reason'], 4000, 'unsupported author reason')
                return self.unsupported('author', answer['reason'], attempt=attempt)
            config_path = self.root/f'finish-{attempt}.json'
            pack, reload = self.root/f'model-{attempt}', self.root/f'reload-{attempt}'
            exported = self.root/f'exported-preview-{attempt}'
            try:
                config = prompts.finish_config(answer, self.request, plan, source, attempt,
                                               quality_policy=quality, bone_parents=source_parents)
                if config_path.exists():
                    require(read_json(config_path) == config, 'Authored configuration changed')
                else:
                    write_json(config_path, config)
                self.event('author', 'complete', attempt=attempt,
                           reason='Rig/motion configuration validated; generated appearance is not yet accepted.')
                self.event('build', 'running', attempt=attempt)
                built = self._pack(pack, lambda: finish(config_path, pack))
                require(built['source_sha256'] == source['sha256']
                        and sha256(pack/'sources/finish-design.json') == sha256(config_path),
                        'Finished model belongs to a different design/source')
                self.event('build', 'complete', attempt=attempt,
                           reason='Model built and source-bound; independent motion and visual checks are pending.')
                self.event('deformation_gate', 'running', attempt=attempt)
                deformation = self._deformation(attempt, pack, plan, source, config_path)
                self.event('deformation_gate', 'complete', attempt=attempt,
                           reason='No reported gross skin spike; this is not visual acceptance.')
                self.event('native_reload', 'running', attempt=attempt)
                loaded = self._pack(reload, lambda: reload_3d(pack, reload))
                require(loaded['status'] == 'passed'
                        and loaded['source_manifest_sha256'] == sha256(pack/'manifest.json'),
                        'Independent native reload did not pass for this exact model')
                self.event('native_reload', 'complete', attempt=attempt,
                           reason='Exact exported model loaded and moved; this is not visual acceptance.')
                self.event('action_audit', 'running', attempt=attempt)
                action_audit = self._action_audit(attempt, pack, plan, answer)
                require(action_audit['passed'], 'Requested action capability failed: '+json.dumps(action_audit['states']))
                self.event('action_audit', 'complete', attempt=attempt,
                           reason='Numeric movement/stability checks passed; visual action review is pending.')
                self.event('exported_views', 'running', attempt=attempt)
                poses = [{'id': s['id'], 'seconds': s['seconds']} for s in read_json(pack/'finish-report.json')['states']]
                exported_proof = self._pack(exported, lambda: preview(pack/'model.glb', exported, resolution=720,
                                                                      samples=16, motion_states=poses))
                require(exported_proof['source_glb_sha256'] == sha256(pack/'model.glb'), 'Exported appearance preview mismatch')
                self.event('exported_views', 'complete', attempt=attempt,
                           reason='Actual exported views rendered; their quality still requires review.')
                media, portable = self._media(attempt, pack)
                self.event('review_evidence', 'running', attempt=attempt)
                evidence, coverage = self._full_evidence(attempt, pack, exported, media)
                quality_coverage = evaluate_quality_evidence(quality, coverage)
                require(quality_coverage['passed'], 'Incomplete HD review: '+', '.join(quality_coverage['missing']))
                proofs = {'rig': built['rig_verification'], 'native_reload': loaded['status'],
                          'attributes': read_json(pack/'render-attributes.json'), 'action_audit': action_audit,
                          'deformation_gate': deformation,
                          'quality_policy': quality, 'quality_coverage': quality_coverage,
                          'delivered_media': read_json(portable/'manifest.json')['clips']}
                self.event('review_evidence', 'complete', attempt=attempt,
                           reason='Final-media and numeric evidence prepared; no critic verdict yet.')
            except (ValueError, OSError, KeyError, TypeError) as error:
                if isinstance(error, (InterruptedStage, PipelineCancelled, EvidenceCustodyError)):
                    raise
                self._fail_active_stage(error)
                for candidate in (pack, reload, exported, self.root/f'media-{attempt}',
                                  self.root/f'media-continuation-{attempt}', self.root/f'action-audit-{attempt}'):
                    if (candidate/'job.json').exists():
                        job = read_json(candidate/'job.json')
                        require(not job.get('manual_stop_required') and job.get('process_exit_verified'),
                                'Owned native process exit is unresolved; no automatic repair')
                previous, findings = answer, {'validation_failure': str(error), 'measurements': self._diagnostics(pack)}
                if (pack/'finish-report.json').exists() and read_json(pack/'finish-report.json').get('states'):
                    _, repair_images = self._contacts(attempt, pack)
                    findings['repair_images'] = [str(p) for p in repair_images[:5]]
                self.event('attempt', 'rejected', attempt=attempt, reason=str(error))
                continue
            # A failed/uncertain reviewer request is not a reason to buy another model call.
            self.guard()
            self.event('reviewer', 'running', attempt=attempt)
            review = self.agents.call('reviewer-'+str(attempt),
                prompts.critic_prompt(self.request, plan, config, proofs, evidence),
                prompts.critic_schema(evidence, action_audit),
                images=[Path(e['path']) for e in evidence])
            try:
                prompts.validate_review(review, plan, [e['id'] for e in prompts.review_citations(evidence, action_audit)])
                self._bind_review(review, evidence, action_audit)
            except (ValueError, KeyError, TypeError) as error:
                # The agent has exited; a rejected data contract is not an uncertain native process.
                return self.unsupported('reviewer', 'Reviewer response failed its evidence contract: '+str(error)[:3000],
                                        attempt=attempt, failure_kind='review_contract')
            review_path = self.root/f'review-{attempt}.json'
            review_receipt = {'review': review, 'evidence': evidence,
                              'model_manifest_sha256': sha256(pack/'manifest.json'), 'human_approval': False}
            if not review_path.exists():
                write_json(review_path, review_receipt)
            require(read_json(review_path) == review_receipt, 'Cached review receipt changed')
            self.event('reviewer', review['verdict'], attempt=attempt)
            if review['verdict'] == 'accept':
                selected = (attempt, pack, reload, review, media, portable)
                break
            if review['verdict'] == 'unsupported':
                return self.unsupported('reviewer', review['summary'], attempt=attempt, review=review)
            _, repair_images = self._contacts(attempt, pack)
            previous, findings = config, {'review': review, 'measurements': self._diagnostics(pack),
                                         'repair_images': [str(p) for p in repair_images[:5]]}
        if selected is None:
            return self.unsupported('repair_budget', 'No accepted model within the bounded local repair attempts', findings=findings)
        attempt, pack, reload, review, media, portable = selected
        return self._deliver(plan, attempt, pack, reload, review, media, portable)

    def _media(self, attempt, pack):
        from avatar_assets.export2d import export_2d
        from scripts.package_portable_media import package_portable_media
        self.event('media', 'running', attempt=attempt)
        media = self.root/f'media-{attempt}'
        spec = self.request['media_spec']
        try:
            self._pack(media, lambda: export_2d(pack, media, preset='preview', **spec, denoise_portrait=True))
        except ValueError:
            # Only the existing hash-bound wall-time continuation policy can resume frames.
            if (media/'job.json').exists() and read_json(media/'job.json').get('failure') == 'wall-time budget exceeded':
                prior, media = media, self.root/f'media-continuation-{attempt}'
                self._pack(media, lambda: export_2d(pack, media, preset='preview', **spec,
                                                   denoise_portrait=True, resume_frames=prior))
            else:
                raise
        portable = self.root/f'media-portable-{attempt}'
        self._pack(portable, lambda: package_portable_media(media, portable))
        self.event('media', 'complete', attempt=attempt,
                   reason='Portrait, video and static files encoded; independent quality review is pending.')
        return media, portable

    def _deliver(self, plan, attempt, pack, reload, review, media, portable):
        from avatar_assets.cli import verify_pack
        from avatar_assets.gallery import _Catalog
        self.guard()
        self.event('delivery', 'running', attempt=attempt)
        catalog = {'schema_version': 1, 'assets': [{
            'id': self.request['id'], 'title': plan['subject'] if len(plan['subject']) <= 160 else plan['subject'][:159]+'…',
            'kind': 'generated_3d',
            'description': 'AI-generated geometry with agent-authored rig/motions and sampled agent visual review. Not human approval or physics.',
            'pack': str(pack.relative_to(ROOT)), 'media_pack': str(portable.relative_to(ROOT)),
        }]}
        catalog_path = self.root/'gallery.json'
        if not catalog_path.exists():
            write_json(catalog_path, catalog)
        _Catalog(ROOT, str(catalog_path.relative_to(ROOT)))
        packs = [pack, reload, media, portable, self.root/f'exported-preview-{attempt}',
                 self.root/f'action-audit-{attempt}', self.root/f'final-evidence-{attempt}']
        for p in packs:
            verify_pack(p)
        result = {'schema_version': 1, 'status': 'complete', 'out': str(self.root),
                  'scope': 'Generated/reused geometry, agent-authored design, weighted action audit, independent native motion proof and final-media HD review; not universal quality or human approval.',
                  'request_sha256': sha256(self.root/'request.json'), 'plan_sha256': sha256(self.root/'plan.json'),
                  'selected_attempt': attempt, 'model': str(pack.relative_to(self.root)/'model.glb'),
                  'editable_source': str(pack.relative_to(self.root)/'source.blend'),
                  'gallery_catalog': str(catalog_path.relative_to(ROOT)),
                  'generation_mode': 'reused_generated_source' if self.request['reuse_model'] else 'new_meshy_generation',
                  'new_meshy_credits': 0 if self.request['reuse_model'] else read_json(self.root/'generation-cost.json')['provider_reported_credits'],
                  'generation_cost': {'reserved_credits': 0, 'paid_submissions': 0} if self.request['reuse_model'] else read_json(self.root/'generation-cost.json'),
                  'quality_equivalence_verified': False, 'weft_runtime_verified': False, 'human_approval': False,
                  'loop_verified': False, 'review': review,
                  'packs': [{'path': str(p.relative_to(self.root)), 'manifest_sha256': sha256(p/'manifest.json')} for p in packs],
                  'limitations': plan['limitations']+review['limitations'], 'recorded': now()}
        write_json(self.root/'delivery.json', result)
        write_json(self.root/'delivery-receipt.json', {
            name: sha256(self.root/name) for name in ('delivery.json', 'plan.json', 'gallery.json', 'quality-policy.json',
                                                   f'review-{attempt}.json', f'deformation-gate-{attempt}.json')})
        self.event('delivery', 'complete', attempt=attempt, model=result['model'])
        return result


def run(folder, *, execute=False, agents=None, plan_only=False, cancel_requested=None):
    pipeline = Pipeline(folder, agents=agents)
    return pipeline.run(execute=execute, plan_only=plan_only, cancel_requested=cancel_requested)


def add_commands(parent):
    parser = parent.add_parser('generate', help='Prompt-to-assets AI planner, rig author, reviewer and bounded local execution')
    parser.add_argument('--prompt')
    parser.add_argument('--out')
    parser.add_argument('--resume', help='Resume this exact job without repeating completed stages')
    parser.add_argument('--style', default='')
    parser.add_argument('--states', help='Comma-separated state IDs; planner chooses four if omitted')
    parser.add_argument('--action', action='append', default=[], help='Exact custom action description; repeat up to eight times')
    parser.add_argument('--reference')
    parser.add_argument('--reuse-model', help='Explicit reuse of existing generated GLB; no new Meshy call')
    parser.add_argument('--preset', choices=PRESETS, default='preview')
    parser.add_argument('--max-credits', type=int, default=0)
    parser.add_argument('--max-repairs', type=int, default=1)
    parser.add_argument('--max-agent-calls', type=int, default=10)
    parser.add_argument('--agent-timeout', type=int, default=240)
    parser.add_argument('--agent-model')
    parser.add_argument('--prepare-only', action='store_true', help='Save request/cost bounds without model/provider calls')
    parser.add_argument('--plan-only', action='store_true', help='Run the AI planner and stop before generation or native work')
    parser.add_argument('--execute', action='store_true', help='Permit planned Meshy calls within this cap and the existing canonical ledger')


def main(args):
    if args.resume:
        require(not args.prepare_only, '--prepare-only applies to a new job, not --resume')
        require(not args.prompt and not args.out and not args.reference and not args.reuse_model
                and not getattr(args, 'action', None),
                'Resume uses its immutable request; do not supply new inputs')
        folder = args.resume
    else:
        require(args.prompt and args.out, 'Supply --prompt and --out, or --resume')
        prepared = prepare(args.prompt, args.out, style=args.style, states=args.states.split(',') if args.states else (),
                           custom_actions=getattr(args, 'action', []),
                           preset=args.preset, reference=args.reference, reuse_model=args.reuse_model,
                           max_credits=args.max_credits, max_repairs=args.max_repairs, max_agent_calls=args.max_agent_calls,
                           agent_timeout=args.agent_timeout, agent_model=args.agent_model)
        if args.prepare_only:
            return prepared
        folder = args.out
    result = run(folder, execute=args.execute, plan_only=getattr(args, 'plan_only', False))
    return {key: result[key] for key in ('status', 'out', 'gallery_catalog', 'model', 'reason', 'resumed_without_generation') if key in result}
