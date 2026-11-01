"""Prompt-to-delivery orchestration; agents propose data, trusted tools execute it."""

from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
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


class InterruptedStage(ValueError):
    """An unsealed stage is not evidence that its native process exited."""


def now():
    return datetime.now(ZoneInfo("America/Los_Angeles")).isoformat()


def _json_digest(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


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


def prepare(prompt, output, *, style='', states=(), preset='preview', reference=None, reuse_model=None,
            max_credits=0, max_repairs=1, max_agent_calls=5, agent_timeout=240, agent_model=None):
    _text(prompt, 4000, 'prompt')
    _text(style, 800, 'style', empty=True)
    require(preset in PRESETS, 'Unknown output preset')
    require(isinstance(states, (tuple, list)) and len(states) <= 8, 'At most eight states')
    states = [slug(s) for s in states]
    require(len(set(states)) == len(states), 'Duplicate requested state')
    require(type(max_credits) is int and 0 <= max_credits <= 500, 'Credit cap must be0–500')
    require(type(max_repairs) is int and 0 <= max_repairs <= 2, 'At most two local repairs')
    require(type(max_agent_calls) is int and 3 <= max_agent_calls <= 8, 'Agent calls must be3–8')
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
               'states': states, 'preset': preset, 'media_spec': PRESETS[preset], 'max_credits': max_credits,
               'max_repairs': max_repairs, 'max_agent_calls': max_agent_calls, 'agent_timeout': agent_timeout,
               'agent_model': agent_model, 'reference': None, 'reuse_model': None, 'created': now()}
    for original, name, field in ((ref, 'reference', 'reference'), (source, 'source-model', 'reuse_model')):
        if original:
            target = out/'inputs'/(name+original.suffix.lower())
            shutil.copyfile(original, target)
            request[field] = {'path': str(target.relative_to(out)), 'sha256': sha256(target),
                              'original_path': str(original), 'bytes': target.stat().st_size}
    write_json(out/'request.json', request)
    for name in ('pipeline.py', 'pipeline_prompts.py', 'agent_runner.py'):
        shutil.copyfile(ROOT/'avatar_assets'/name, out/'sources'/name)
    write_json(out/'input-receipt.json', {'request_sha256': sha256(out/'request.json'), 'sources': artifacts(out/'sources')})
    write_json(out/'estimate.json', {
        'schema_version': 1, 'new_meshy_credits': 0 if source else 35,
        'new_meshy_paid_calls': 0 if source else (1 if ref else 2), 'automatic_paid_retries': 0,
        'maximum_agent_calls': max_agent_calls, 'expected_agent_calls': 3,
        'planning_token_estimate': {'input_per_call': 'up to approximately24000 text tokens plus up to8 selected images',
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
        for source in receipt['sources']:
            require(sha256(self.root/'sources'/source['path']) == source['sha256']
                    and sha256(ROOT/'avatar_assets'/source['path']) == source['sha256'],
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

    def event(self, stage, status, **details):
        entries = sorted((self.root/'events').glob('*.json'))
        require(len(entries) < 128, 'Pipeline event limit reached')
        value = {'schema_version': 1, 'recorded': now(), 'stage': stage, 'status': status, **details}
        write_json(self.root/'events'/f'{len(entries):04d}.json', value)
        _atomic(self.root/'status.json', value)
        print(json.dumps({'stage': stage, 'status': status}), flush=True)

    def guard(self, reserve_mb=0):
        require(time.monotonic()-self.started <= MAX_STAGE_START_SECONDS, 'Pipeline stage-start deadline reached')
        from avatar_assets.runner import _output_bytes
        require(_output_bytes(self.root) + reserve_mb*1024**2 <= MAX_OUTPUT, 'Pipeline aggregate output bound reached')

    def unsupported(self, stage, reason, **extra):
        self.event(stage, 'unsupported', reason=reason)
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
        manifest = read_json(path/'manifest.json')
        if manifest.get('status') == 'failed':
            require(manifest.get('artifacts') == artifacts(path), 'Failed stage artifacts changed')
            # The repair prompt must retain the same failure across process restarts.
            if failure_path.exists():
                failure = read_json(failure_path)
                require(failure['manifest_sha256'] == sha256(path/'manifest.json'), 'Failed stage receipt changed')
                raise ValueError(failure['error'])
            raise ValueError(manifest.get('error') or 'Retained failed stage: '+path.name)
        verify_pack(path)
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
        return {'path': str(local_file(path)), 'sha256': sha256(path)}

    def _contacts(self, attempt, pack):
        folder = self.root/f'evidence-{attempt}'
        if not folder.exists():
            folder.mkdir()
            (folder/'states').mkdir()
            report = read_json(pack/'finish-report.json')
            for state in report['states']:
                paths = sorted((pack/'renders'/state['id']).glob('*.png'))
                require(2 <= len(paths) <= 8, 'Missing bounded sampled poses')
                columns, tile, row_height = min(4, len(paths)), 320, 348
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
                sheet.save(folder/'states'/(state['id']+'.png'))
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

    def run(self, *, execute=False):
        with _lock(self.root):
            try:
                return self._run(execute)
            except Exception as error:
                self.event('pipeline', 'checkpointed_error', error=str(error), automatic_retry=False)
                raise

    def _run(self, execute):
        from avatar_assets import pipeline_prompts as prompts
        from avatar_assets.cli import verify_pack
        from avatar_assets.finish_generated import finish
        from avatar_assets.provider_model import preview
        from avatar_assets.review import reload_3d
        if (self.root/'delivery.json').exists():
            receipt = read_json(self.root/'delivery-receipt.json')
            for name, checksum in receipt.items():
                require(sha256(self.root/name) == checksum, 'Delivery evidence changed')
            delivery = read_json(self.root/'delivery.json')
            require(delivery['status'] == 'complete' and len(delivery['packs']) == 5
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
        source = self._source(plan, execute)
        if source is None:
            return {'status': read_json(self.root/'status.json')['status'], 'out': str(self.root)}
        inspection = self.root/'inspection'
        self.event('inspection', 'running')
        inspected = self._pack(inspection, lambda: preview(source['path'], inspection, resolution=512, samples=8, rig_guides=True))
        require(inspected['source_glb_sha256'] == source['sha256'], 'Preview belongs to a different source')
        context = {'preflight': inspected['preflight'], 'native': inspected['native']}
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
                config = prompts.finish_config(answer, self.request, plan, source, attempt)
                if config_path.exists():
                    require(read_json(config_path) == config, 'Authored configuration changed')
                else:
                    write_json(config_path, config)
                self.event('build', 'running', attempt=attempt)
                built = self._pack(pack, lambda: finish(config_path, pack))
                require(built['source_sha256'] == source['sha256']
                        and sha256(pack/'sources/finish-design.json') == sha256(config_path),
                        'Finished model belongs to a different design/source')
                loaded = self._pack(reload, lambda: reload_3d(pack, reload))
                require(loaded['status'] == 'passed'
                        and loaded['source_manifest_sha256'] == sha256(pack/'manifest.json'),
                        'Independent native reload did not pass for this exact model')
                exported_proof = self._pack(exported, lambda: preview(pack/'model.glb', exported, resolution=512, samples=8))
                require(exported_proof['source_glb_sha256'] == sha256(pack/'model.glb'), 'Exported appearance preview mismatch')
                evidence = self._evidence(attempt, pack, exported)
                proofs = {'rig': built['rig_verification'], 'native_reload': loaded['status'],
                          'attributes': read_json(pack/'render-attributes.json')}
            except (ValueError, OSError, KeyError, TypeError) as error:
                if isinstance(error, InterruptedStage):
                    raise
                for candidate in (pack, reload, exported):
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
                prompts.critic_prompt(self.request, plan, config, proofs, evidence), prompts.CRITIC_SCHEMA,
                images=[Path(e['path']) for e in evidence])
            prompts.validate_review(review, plan, [e['id'] for e in evidence])
            if review['verdict'] == 'accept':
                for state in review['state_reviews']:
                    allowed = {e['id'] for e in evidence if state['id'] in e['state_ids']}
                    require(set(state['evidence_ids']) & allowed, 'State acceptance needs its actual pose contact sheet')
            review_path = self.root/f'review-{attempt}.json'
            if not review_path.exists():
                write_json(review_path, {'review': review, 'evidence': evidence,
                                         'model_manifest_sha256': sha256(pack/'manifest.json'), 'human_approval': False})
            self.event('reviewer', review['verdict'], attempt=attempt)
            if review['verdict'] == 'accept':
                selected = (attempt, pack, reload, review)
                break
            if review['verdict'] == 'unsupported':
                return self.unsupported('reviewer', review['summary'], attempt=attempt, review=review)
            _, repair_images = self._contacts(attempt, pack)
            previous, findings = config, {'review': review, 'measurements': self._diagnostics(pack),
                                         'repair_images': [str(p) for p in repair_images[:5]]}
        if selected is None:
            return self.unsupported('repair_budget', 'No accepted model within the bounded local repair attempts', findings=findings)
        attempt, pack, reload, review = selected
        return self._deliver(plan, attempt, pack, reload, review)

    def _deliver(self, plan, attempt, pack, reload, review):
        from avatar_assets.cli import verify_pack
        from avatar_assets.export2d import export_2d
        from avatar_assets.gallery import _Catalog
        from scripts.package_portable_media import package_portable_media
        self.event('media', 'running')
        media = self.root/'media'
        spec = self.request['media_spec']
        try:
            self._pack(media, lambda: export_2d(pack, media, preset='preview', **spec, denoise_portrait=True))
        except ValueError:
            # Only the existing hash-bound wall-time continuation policy can resume frames.
            if (media/'job.json').exists() and read_json(media/'job.json').get('failure') == 'wall-time budget exceeded':
                prior, media = media, self.root/'media-continuation'
                self._pack(media, lambda: export_2d(pack, media, preset='preview', **spec,
                                                   denoise_portrait=True, resume_frames=prior))
            else:
                raise
        portable = self.root/'media-portable'
        self._pack(portable, lambda: package_portable_media(media, portable))
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
        packs = [pack, reload, media, portable, self.root/f'exported-preview-{attempt}']
        for p in packs:
            verify_pack(p)
        result = {'schema_version': 1, 'status': 'complete', 'out': str(self.root),
                  'scope': 'Generated/reused geometry, agent-authored design, independent native motion proof, agent visual review and actual media; not universal quality or human approval.',
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
            name: sha256(self.root/name) for name in ('delivery.json', 'plan.json', 'gallery.json', f'review-{attempt}.json')})
        self.event('delivery', 'complete', model=result['model'])
        return result


def run(folder, *, execute=False, agents=None):
    pipeline = Pipeline(folder, agents=agents)
    return pipeline.run(execute=execute)


def add_commands(parent):
    parser = parent.add_parser('generate', help='Prompt-to-assets AI planner, rig author, reviewer and bounded local execution')
    parser.add_argument('--prompt')
    parser.add_argument('--out')
    parser.add_argument('--resume', help='Resume this exact job without repeating completed stages')
    parser.add_argument('--style', default='')
    parser.add_argument('--states', help='Comma-separated state IDs; planner chooses four if omitted')
    parser.add_argument('--reference')
    parser.add_argument('--reuse-model', help='Explicit reuse of existing generated GLB; no new Meshy call')
    parser.add_argument('--preset', choices=PRESETS, default='preview')
    parser.add_argument('--max-credits', type=int, default=0)
    parser.add_argument('--max-repairs', type=int, default=1)
    parser.add_argument('--max-agent-calls', type=int, default=5)
    parser.add_argument('--agent-timeout', type=int, default=240)
    parser.add_argument('--agent-model')
    parser.add_argument('--prepare-only', action='store_true', help='Save request/cost bounds without model/provider calls')
    parser.add_argument('--execute', action='store_true', help='Permit planned Meshy calls within this cap and the existing canonical ledger')


def main(args):
    if args.resume:
        require(not args.prepare_only, '--prepare-only applies to a new job, not --resume')
        require(not args.prompt and not args.out and not args.reference and not args.reuse_model,
                'Resume uses its immutable request; do not supply new inputs')
        folder = args.resume
    else:
        require(args.prompt and args.out, 'Supply --prompt and --out, or --resume')
        prepared = prepare(args.prompt, args.out, style=args.style, states=args.states.split(',') if args.states else (),
                           preset=args.preset, reference=args.reference, reuse_model=args.reuse_model,
                           max_credits=args.max_credits, max_repairs=args.max_repairs, max_agent_calls=args.max_agent_calls,
                           agent_timeout=args.agent_timeout, agent_model=args.agent_model)
        if args.prepare_only:
            return prepared
        folder = args.out
    result = run(folder, execute=args.execute)
    return {key: result[key] for key in ('status', 'out', 'gallery_catalog', 'model', 'reason', 'resumed_without_generation') if key in result}
