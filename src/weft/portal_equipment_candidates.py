"""Prepared-source equipment candidates: bounded discovery, explicit selection, unchanged evidence."""

from hashlib import sha256
import json
import math
from pathlib import Path
import re
import time

from weft.exercise_body_source import _json
from weft.io import canonical_hash, local_time
from weft.lab_worker import atomic_json
from weft.openai_access import PROJECT_ROOT
from weft.shared_world_packaging import read_package_file, read_package_manifest, write_package_files

CATALOG = "equipment-sources.json"
CATALOG_PROFILE = "portal-prepared-equipment-catalog-v1"
PROFILE = "portal-equipment-candidates-v1"
TERMINAL = frozenset({"complete", "failed", "cancelled", "interrupted"})
MAX_CATALOG_BYTES = 64 << 10
MAX_ENTRIES = 12
MAX_CANDIDATES = 3
SCOPE = "Conditional equipment simulations from verified prepared evidence; identities, camera, ground, geometry and contact remain hypotheses. Not automatic reconstruction of an unfamiliar video."


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Equipment candidate operation exceeded its deadline")


def _relative(value):
    if (not isinstance(value, str) or not 1 <= len(value) <= 512 or '\\' in value or ':' in value
            or Path(value).is_absolute() or any(x in {'', '.', '..'} for x in value.split('/'))):
        raise ValueError("Prepared equipment paths must be bounded project-relative paths")
    return value


def _path(project, name):
    path = Path(project).absolute() / _relative(name)
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_dir():
        raise ValueError("Prepared equipment directories must exist without links")
    return path


def _entry(value):
    expected = {'id', 'label', 'source_sha256', 'base', 'equipment', 'mappings'}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Unsupported prepared equipment catalog entry")
    if (not isinstance(value['id'], str) or not re.fullmatch(r'[a-f0-9]{16}', value['id'])
            or not isinstance(value['label'], str) or not 1 <= len(value['label']) <= 100
            or any(ord(c) < 32 for c in value['label'])
            or not isinstance(value['source_sha256'], str) or not re.fullmatch(r'[a-f0-9]{64}', value['source_sha256'])
            or not isinstance(value['mappings'], list) or len(value['mappings']) > 7):
        raise ValueError("Invalid prepared equipment catalog identity")
    for item in [value['base'], value['equipment'], *value['mappings']]:
        if not isinstance(item, dict) or set(item) != {'path', 'manifest_sha256'}:
            raise ValueError("Prepared equipment packages require exact manifest identities")
        _relative(item['path'])
        if not isinstance(item['manifest_sha256'], str) or not re.fullmatch(r'[a-f0-9]{64}', item['manifest_sha256']):
            raise ValueError("Prepared equipment package pin is invalid")
    identity = {key: value[key] for key in ('source_sha256', 'base', 'equipment', 'mappings')}
    if value['id'] != canonical_hash(identity)[:16]:
        raise ValueError("Prepared equipment catalog identity conflicts with its packages")
    return value


def read_catalog(root):
    try:
        raw = read_package_file(root, CATALOG, max_bytes=MAX_CATALOG_BYTES)
    except ValueError as error:
        if isinstance(error.__cause__, FileNotFoundError):
            return {'schema_version': 1, 'profile': CATALOG_PROFILE, 'entries': []}
        raise
    value = _json(raw)
    if (set(value) != {'schema_version', 'profile', 'entries'} or type(value['schema_version']) is not int
            or value['schema_version'] != 1 or value['profile'] != CATALOG_PROFILE
            or not isinstance(value['entries'], list) or len(value['entries']) > MAX_ENTRIES):
        raise ValueError("Prepared equipment catalog exceeds its closed contract")
    identities = [_entry(entry)['id'] for entry in value['entries']]
    if len(identities) != len(set(identities)):
        raise ValueError("Prepared equipment catalog has duplicate identities")
    return value


def availability(job, status, catalog):
    if (job.get('kind', 'video') != 'video' or status.get('status') != 'needs_input'
            or status.get('route') not in (None, 'exercise')):
        return {'available': False, 'matches': [], 'reason': 'Equipment candidate preparation applies to an unresolved original exercise upload.'}
    matches = [{'id': e['id'], 'label': e['label']} for e in catalog['entries'] if e['source_sha256'] == job.get('source_sha256')]
    return {'available': bool(matches), 'matches': matches, 'scope': SCOPE,
        'reason': ('Prepared evidence matches this exact video. Full source custody is rechecked before construction.' if matches else
                   'No verified prepared equipment evidence is registered for this video. A body-only preview is available; unfamiliar equipment reconstruction is not yet automatic.')}


def validate_source_entry(entry, *, project=PROJECT_ROOT, max_seconds=15):
    """A matching upload hash only shortlists; complete source/equivalence validation admits inputs."""
    from weft.equipment_autoproposal import derive_equipment_config
    from weft.shared_world import validate_shared_world
    from weft.shared_world_packaging import collect_shared_source_files
    from weft.shared_world_pipeline import read_equipment_observations
    from weft.source_equivalence import load_source_equivalence

    entry = _entry(entry)
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 30:
        raise ValueError("Prepared equipment validation must be bounded to 30 seconds")
    deadline = time.monotonic() + max_seconds
    base, equipment = (_path(project, entry[key]['path']) for key in ('base', 'equipment'))
    base_manifest, base_files = read_package_manifest(base)
    equipment_manifest, equipment_files = read_package_manifest(equipment)
    if (sha256(base_manifest).hexdigest() != entry['base']['manifest_sha256']
            or sha256(equipment_manifest).hexdigest() != entry['equipment']['manifest_sha256']):
        raise ValueError("Prepared equipment source seals changed")
    if 'scene.json' not in base_files:
        raise ValueError('Prepared base scene lacks manifest coverage')
    scene_raw = read_package_file(base, 'scene.json', max_bytes=16 << 20, expected_sha256=base_files['scene.json'])
    scene = _json(scene_raw)
    if (validate_shared_world(scene) != scene or scene['source_binding']['mode'] != 'source_qualified'
            or scene['source_binding']['source_sha256'] != entry['source_sha256']):
        raise ValueError("Prepared equipment base is not the exact source-qualified uploaded video")
    proof = collect_shared_source_files(scene, base, expected_manifest_sha256=entry['base']['manifest_sha256'], deadline=deadline)
    human = next(row for row in scene['source_binding']['entity_bindings'] if row['world_entity_id'] == 'human')
    human_sha = human['observation_document_sha256']
    human_raw = proof.files[scene['source_binding']['evidence_documents'][human_sha]]
    equipment_raw = read_equipment_observations(equipment, equipment_files)
    documents = {human_sha: human_raw, sha256(equipment_raw).hexdigest(): equipment_raw}
    remaining = (64 << 20) - sum(map(len, documents.values()))
    equivalents, mappings = [], []
    for item in entry['mappings']:
        _check(deadline)
        path = _path(project, item['path'])
        mapping = load_source_equivalence(path, max_input_bytes=remaining, deadline=deadline)
        if mapping.manifest_sha256 != item['manifest_sha256']:
            raise ValueError("Prepared equipment source mapping seal changed")
        equivalents.append(mapping);mappings.append(path);remaining -= mapping.total_input_bytes
    report, _ = derive_equipment_config(documents, source_sha256=entry['source_sha256'], human_ref=(human_sha, human['observation_entity_id']),
        equipment_document_sha256=sha256(equipment_raw).hexdigest(), body_references=scene['human_spec']['reference_evidence'],
        source_equivalences=equivalents, deadline=deadline)
    _check(deadline)
    return {'base_run': base, 'evidence_run': equipment, 'source_mappings': tuple(mappings)}, {
        'source_sha256': entry['source_sha256'], 'base_manifest_sha256': entry['base']['manifest_sha256'],
        'equipment_manifest_sha256': entry['equipment']['manifest_sha256'], 'role_status': report['status'],
        'unresolved_reasons': report['reasons'], 'body_reference_count': len(scene['human_spec']['reference']),
        'raw_object_ids': [row['object_id'] for row in report['inventory']],
        'proposed_assignment': ({'body_correspondence': next(row['object_id'] for row in report['inventory']
            if row['selected_role'] in {'exact_bound_body','conditional_body_correspondence'}),
            'rigid_proxies': [row['object_id'] for row in report['config']['equipment']]}
            if report.get('config') is not None else None),
        'source_slots': scene['source_binding']['total_source_frames'], 'scope': SCOPE}


def register_equipment_source(root, *, source_sha256, base_run, evidence_run, source_mappings=(), label, project=PROJECT_ROOT):
    """Administrator registration only; ordinary upload users never enter paths or source hashes."""
    import fcntl
    import os
    from weft.portal import private_directory

    project = Path(project).absolute()
    def package(value):
        path = Path(value).absolute()
        if not path.is_relative_to(project):
            raise ValueError("Prepared equipment sources must remain inside the project workspace")
        name = path.relative_to(project).as_posix();_path(project, name)
        manifest, _ = read_package_manifest(path)
        return {'path': name, 'manifest_sha256': sha256(manifest).hexdigest()}
    if not isinstance(source_mappings, (list, tuple)) or len(source_mappings) > 7:
        raise ValueError("Prepared equipment requires at most seven source mappings")
    entry = {'source_sha256': source_sha256, 'base': package(base_run), 'equipment': package(evidence_run),
             'mappings': [package(p) for p in source_mappings]}
    entry.update(id=canonical_hash(entry)[:16], label=label)
    validate_source_entry(entry, project=project)
    root = private_directory(root)
    lease = os.open(root / 'portal.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:raise ValueError('Stop the portal before registering equipment evidence') from None
        catalog = read_catalog(root)
        existing = next((e for e in catalog['entries'] if e['id'] == entry['id']), None)
        if existing is not None:
            if existing != entry:raise ValueError('Existing prepared source registration has a different label')
            return {'status': 'already_registered', 'id': entry['id']}
        if len(catalog['entries']) >= MAX_ENTRIES:raise ValueError('Prepared equipment catalog is full')
        catalog['entries'].append(entry)
        if len(json.dumps(catalog).encode()) > MAX_CATALOG_BYTES:raise ValueError('Prepared equipment catalog exceeds its byte bound')
        atomic_json(root / CATALOG, catalog)
        return {'status': 'registered', 'id': entry['id'], 'source_sha256': source_sha256, 'api_calls': 0, 'scope': SCOPE}
    finally:os.close(lease)


def read_status(root):
    try:return _json(read_package_file(root, 'equipment/status.json', max_bytes=32 << 10))
    except ValueError as error:
        if isinstance(error.__cause__, FileNotFoundError):return {}
        raise


def _candidate_lab(root, candidate, *, project):
    from weft.portal_lab_runtime import open_portal_lab
    lab = Path(candidate['lab_directory']).absolute()
    run = root / 'equipment/run'
    if not lab.is_relative_to(run) or any(p.is_symlink() for p in (lab, *lab.parents)):
        raise ValueError('Equipment candidate lab must stay inside its immutable assembly directory')
    config_raw = read_package_file(lab, 'lab-config.json', max_bytes=64 << 10)
    config = _json(config_raw)
    manifest, files = read_package_manifest(lab / 'run')
    if (config.get('control_profile') != 'shared_exercise' or config.get('learned_policy_available') is not False
            or config.get('manifest_sha256') != sha256(manifest).hexdigest()):
        raise ValueError('Equipment candidate must be an exact sealed untrained exercise lab')
    scene = _json(read_package_file(lab / 'run', 'scene.json', max_bytes=16 << 20, expected_sha256=files.get('scene.json')))
    job = _json(read_package_file(root, 'job.json', max_bytes=64 << 10))
    if scene['source_binding']['source_sha256'] != job['source_sha256']:
        raise ValueError('Equipment candidate differs from the uploaded source')
    admitted = open_portal_lab(lab, project=project)
    try:
        summary = admitted.summary()
        if summary.get('scene') != scene or summary.get('learned_policy_available') is not False:
            raise ValueError('Equipment candidate ordinary admission differs')
    finally:admitted.close()
    return {'lab': lab.relative_to(root).as_posix(), 'lab_config_sha256': sha256(config_raw).hexdigest(),
            'manifest_sha256': sha256(manifest).hexdigest(), 'scene_sha256': config['scene_sha256'], 'task_sha256': config['task_sha256'],
            'source_sha256': job['source_sha256'], 'learned_policy_available': False}


def candidate_rows(root, assembled, *, project=PROJECT_ROOT):
    """Closed inspection projection; rejected studies can be inspected without replaying construction."""
    if not isinstance(assembled, dict) or not isinstance(assembled.get('candidates'), list) or not 0 <= len(assembled['candidates']) <= MAX_CANDIDATES:
        raise ValueError('Equipment assembly did not retain a bounded candidate inventory')
    candidates, identifiers = [], set()
    for item in assembled['candidates']:
        if not isinstance(item,dict):raise ValueError('Equipment candidate requires an object')
        identity = item.get('id')
        if not isinstance(identity,str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}',identity) or identity in identifiers:
            raise ValueError('Equipment candidate identities must be bounded and unique')
        identifiers.add(identity)
        assignment=item.get('assignment')
        if (not isinstance(assignment,dict) or set(assignment)!={'body_correspondence','rigid_proxies'}
                or not isinstance(assignment['body_correspondence'],str)
                or not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}',assignment['body_correspondence'])
                or not isinstance(assignment['rigid_proxies'],list) or len(assignment['rigid_proxies'])>12
                or any(not isinstance(i,str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}',i) for i in assignment['rigid_proxies'])
                or len(set(assignment['rigid_proxies']))!=len(assignment['rigid_proxies'])
                or assignment['body_correspondence'] in assignment['rigid_proxies']
                or not isinstance(item.get('status'),str) or not re.fullmatch(r'[a-z_]{1,80}',item['status'])):
            raise ValueError('Equipment candidate role assignment or state is invalid')
        reasons = item.get('reasons', [])
        if not isinstance(reasons,list) or len(reasons)>32:
            raise ValueError('Equipment candidate reasons exceed their bound')
        rendered=[]
        for reason in reasons:
            if isinstance(reason,dict) and isinstance(reason.get('code'),str) and isinstance(reason.get('message'),str):
                text=reason['code']+': '+reason['message']
            elif isinstance(reason,str):text=reason
            else:raise ValueError('Equipment rejection reason requires a code/message or text')
            if len(text)>1200:raise ValueError('Equipment rejection text exceeds its bound')
            rendered.append(text)
        reasons=rendered
        candidate = {'id':identity, 'status':item.get('status'), 'reasons':reasons, 'scope':SCOPE,
                     'assignment':assignment, 'task_comparable':False, 'physical_validation':'unverified',
                     'task_success':'not established', 'selectable':False}
        if item.get('status') == 'ready':
            candidate.update(_candidate_lab(root,item,project=project), selectable=True)
        candidates.append(candidate)
    return candidates


def execute_equipment_job(root, *, project=PROJECT_ROOT):
    """One worker slot, no provider credentials; retain all rejected alternatives and partial output."""
    root = Path(root).absolute()
    work = root / 'equipment'
    started = time.monotonic()
    state = read_status(root)
    if state.get('status') in TERMINAL:return 0 if state['status'] == 'complete' else 1
    try:
        from weft.equipment_conditional_assembly import prepare_conditional_equipment_scene

        job = _json(read_package_file(root, 'job.json', max_bytes=64 << 10))
        request_raw = read_package_file(work, 'request.json', max_bytes=16 << 10)
        request = _json(request_raw)
        if (request.get('profile') != PROFILE or request.get('source_sha256') != job['source_sha256']
                or request.get('entry_sha256') != canonical_hash(request.get('entry'))):
            raise ValueError('Equipment request identity differs')
        entry = _entry(request['entry'])
        catalog = read_catalog(root.parent)
        if entry not in catalog['entries'] or entry['source_sha256'] != job['source_sha256']:
            raise ValueError('Prepared equipment registration changed after queuing')
        if read_package_file(root, 'status.json', max_bytes=32 << 10) != read_package_file(work, 'parent-status.json', max_bytes=32 << 10):
            raise ValueError('Parent project changed during equipment construction')
        if Path(job['source_file']).name != job['source_file']:
            raise ValueError('Equipment source filename is unsafe')
        from weft.source_clock_capture import _file_hash
        if _file_hash(root / job['source_file'], 100_000_000, time.monotonic() + 10) != job['source_sha256']:
            raise ValueError('Uploaded equipment video differs from its exact source identity')
        if (work / 'cancel-request.json').exists():
            atomic_json(work / 'status.json', {'status':'cancelled','message':'Cancelled before equipment construction; upload unchanged.'})
            return 0
        atomic_json(work / 'status.json', {'status':'running','message':'Verifying prepared source evidence and constructing bounded conditional alternatives.'})
        inputs, proof = validate_source_entry(entry, project=project)
        write_package_files(work, {'source-validation.json':(json.dumps(proof,sort_keys=True,indent=2)+'\n').encode()})
        if proof.get('role_status') == 'config_proposed':
            from weft.equipment_autoproposal import prepare_automatic_equipment_scene
            automatic = prepare_automatic_equipment_scene(**inputs, out=work / 'run', max_seconds=90)
            assembled = {'status': automatic['status'], 'reasons': automatic.get('reasons', []), 'candidates': [{
                'id': 'automatic-roles-1', 'status': automatic['status'], 'lab_directory': automatic.get('lab_directory'),
                'reasons': automatic.get('reasons', []), 'assignment': proof['proposed_assignment']}]}
        else:
            assembled = prepare_conditional_equipment_scene(**inputs, out=work / 'run', max_seconds=90, max_candidates=MAX_CANDIDATES)
        candidates = candidate_rows(root, assembled, project=project)
        report = {'schema_version':1,'profile':PROFILE,'source_sha256':job['source_sha256'],
            'request_sha256':sha256(request_raw).hexdigest(), 'source_validation':proof, 'candidates':candidates,
            'scope':SCOPE,'assembly_status':assembled.get('status'),'assembly_reasons':assembled.get('reasons', []),
            'selection':'Explicit one-time selection required; no scores select a candidate.',
            'api_calls':0,'inference_calls':0,'training_runs':0,'seconds':time.monotonic()-started}
        report_raw=(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        if len(report_raw)>128<<10:raise ValueError('Equipment candidate summary exceeds its bound')
        write_package_files(work, {'candidates.json':report_raw})
        atomic_json(work/'status.json', {'status':'complete','message':('Choose a conditional candidate to open its untrained sandbox. Selection locks this project to that world.'
            if any(c['selectable'] for c in candidates) else 'No candidate passed the unchanged constructor. Rejections and source uncertainty are preserved.'),
            'report_sha256':sha256(report_raw).hexdigest(),'seconds':time.monotonic()-started,'api_calls':0})
        return 0
    except Exception as error:
        atomic_json(work/'status.json',{'status':'failed','message':'Equipment construction failed; original upload and partial candidates are preserved. No automatic retry.',
            'error':str(error)[:500],'error_type':type(error).__name__,'seconds':time.monotonic()-started,'api_calls':0})
        return 1


def read_report(root, status):
    if status.get('status') != 'complete':return None
    pin=status.get('report_sha256')
    if not isinstance(pin,str) or not re.fullmatch(r'[a-f0-9]{64}',pin):raise ValueError('Equipment report requires its exact pin')
    report=_json(read_package_file(root,'equipment/candidates.json',max_bytes=128<<10,expected_sha256=pin))
    if report.get('profile')!=PROFILE or report.get('schema_version')!=1:raise ValueError('Unsupported equipment candidate report')
    return report


def select_candidate(root, identity):
    """Metadata-only, before the first sandbox: never switch a recorded project's runtime."""
    root=Path(root).absolute()
    status_raw=read_package_file(root,'status.json',max_bytes=32<<10)
    status=_json(status_raw)
    selection=root/'equipment/selection.json'
    saved=None
    if selection.exists():
        saved=_json(read_package_file(root,'equipment/selection.json',max_bytes=8192))
        if saved.get('candidate_id')!=identity:
            raise ValueError('This project is locked to its selected candidate; existing recordings cannot switch worlds')
        if status.get('status')=='ready' and status.get('equipment_candidate_id')==identity and status.get('lab')==saved.get('lab'):
            return saved
    if status.get('status')!='needs_input' or status.get('lab'):
        raise ValueError('Choose equipment only before this project has opened a sandbox')
    if status_raw!=read_package_file(root,'equipment/parent-status.json',max_bytes=32<<10):
        raise ValueError('Parent upload status differs from the preserved construction request')
    report=read_report(root,read_status(root))
    if report is None:raise ValueError('Finish equipment candidate preparation before selection')
    candidate=next((c for c in report['candidates'] if c.get('id')==identity),None)
    if candidate is None or candidate.get('selectable') is not True:raise ValueError('This equipment candidate was not admitted')
    lab=_path(root,candidate['lab'])
    if not lab.is_relative_to(root/'equipment/run'):raise ValueError('Candidate lab escapes its assembly')
    read_package_file(lab,'lab-config.json',max_bytes=64<<10,expected_sha256=candidate['lab_config_sha256'])
    read_package_file(lab/'run','manifest.json',max_bytes=1<<20,expected_sha256=candidate['manifest_sha256'])
    if saved is not None:
        if any(saved.get(key)!=expected for key,expected in {'lab':candidate['lab'],'manifest_sha256':candidate['manifest_sha256'],
                'report_sha256':read_status(root)['report_sha256'],'profile':PROFILE}.items()):
            raise ValueError('Interrupted candidate selection differs from the preserved source report')
    else:
        saved={'profile':PROFILE,'candidate_id':identity,'lab':candidate['lab'],'manifest_sha256':candidate['manifest_sha256'],
               'report_sha256':read_status(root)['report_sha256'],'selected_at':local_time(),'selection_basis':'explicit_user_choice_not_score',
               'scope':SCOPE}
        write_package_files(root/'equipment',{'selection.json':(json.dumps(saved,sort_keys=True,indent=2)+'\n').encode()})
    atomic_json(root/'status.json', {**status,'status':'ready','stage':'equipment_candidate','route':'exercise','lab':candidate['lab'],
        'equipment_candidate_id':identity,'scope':SCOPE,'events':[*status.get('events',[]),{'stage':'equipment_candidate','time':local_time(),
        'message':'Explicitly selected conditional equipment candidate; no task success or physical accuracy is implied.'}]})
    return saved


def inspect_rejected_assembly(source, entry, *, expected_manifest_sha256, project=PROJECT_ROOT, max_seconds=20):
    """Reuse sealed rejection receipts, not geometry admission, a fresh build or a playable candidate."""
    from weft.equipment_conditional_assembly import derive_conditional_equipment_configs
    from weft.shared_world_pipeline import read_equipment_observations
    from weft.source_equivalence import load_source_equivalence

    if type(max_seconds) not in (int,float) or not math.isfinite(max_seconds) or not 0<max_seconds<=30:
        raise ValueError('Saved equipment inspection requires a bounded deadline')
    deadline=time.monotonic()+max_seconds
    inputs,proof=validate_source_entry(entry,project=project,max_seconds=min(15,max_seconds))
    source=Path(source).absolute()
    if any(p.is_symlink() for p in (source,*source.parents)):
        raise ValueError('Saved assembly paths cannot contain links')
    manifest,pins=read_package_manifest(source)
    if sha256(manifest).hexdigest()!=expected_manifest_sha256 or len(pins)>512:
        raise ValueError('Saved assembly differs from its explicit bounded seal')
    snapshots,total={},len(manifest)
    for name,pin in pins.items():
        _check(deadline)
        data=read_package_file(source,name,max_bytes=min(64<<20,(64<<20)-total),expected_sha256=pin)
        snapshots[name]=data;total+=len(data)
    snapshots['manifest.json']=manifest
    result=_json(snapshots['result.json']);context=_json(snapshots['conditional-context.json']);settings=_json(snapshots['settings.json'])
    if (result.get('status')!='abstained' or result.get('selected_candidate') is not None
            or result.get('training') is not False or result.get('physics_steps')!=0
            or result.get('api_calls')!=0 or result.get('inference_calls')!=0
            or not isinstance(result.get('candidates'),list) or not 1<=len(result['candidates'])<=MAX_CANDIDATES):
        raise ValueError('Saved-only intake accepts a complete nonplayable rejection study')
    for key,path in [('base',inputs['base_run']),('equipment',inputs['evidence_run'])]:
        raw,_=read_package_manifest(path)
        if raw!=snapshots[f'inputs/{key}-manifest.json']:
            raise ValueError('Saved study source manifests differ from the registered prepared inputs')
    base_raw,base_pins=read_package_manifest(inputs['base_run'])
    scene_raw=read_package_file(inputs['base_run'],'scene.json',max_bytes=16<<20,expected_sha256=base_pins['scene.json'])
    if scene_raw!=snapshots['inputs/base-scene.json']:
        raise ValueError('Saved study source body world differs')
    scene=_json(scene_raw);human=next(row for row in scene['source_binding']['entity_bindings'] if row['world_entity_id']=='human')
    hpin=human['observation_document_sha256']
    human_raw=read_package_file(inputs['base_run'],scene['source_binding']['evidence_documents'][hpin],max_bytes=32<<20,expected_sha256=hpin)
    _,equipment_pins=read_package_manifest(inputs['evidence_run'])
    equipment_raw=read_equipment_observations(inputs['evidence_run'],equipment_pins);epin=sha256(equipment_raw).hexdigest()
    documents={hpin:human_raw,epin:equipment_raw}
    if any(snapshots.get(f'inputs/observations/{pin}.json')!=data for pin,data in documents.items()):
        raise ValueError('Saved study observations differ from exact current source documents')
    equivalent=[];remaining=(64<<20)-sum(map(len,documents.values()))
    for path in inputs['source_mappings']:
        item=load_source_equivalence(path,max_input_bytes=remaining,deadline=deadline);remaining-=item.total_input_bytes;equivalent.append(item)
    expected,construction=derive_conditional_equipment_configs(documents,source_sha256=entry['source_sha256'],
        human_ref=(hpin,human['observation_entity_id']),equipment_document_sha256=epin,body_references=scene['human_spec']['reference_evidence'],
        source_equivalences=equivalent,max_candidates=settings['maximum_candidates'],deadline=deadline)
    if context!=expected or snapshots['construction-evidence.json']!=construction:
        raise ValueError('Saved role alternatives differ from independently recomputed construction evidence')
    expected_rows={c['id']:c for c in expected['candidates']}
    if {c.get('id') for c in result['candidates']}!=set(expected_rows):
        raise ValueError('Saved result dropped or added a conditional alternative')
    for candidate in result['candidates']:
        identity=candidate['id'];configuration=expected_rows[identity]
        if (candidate.get('status') in {'ready','deferred','not_attempted'} or candidate.get('lab_directory') is not None
                or candidate.get('assignment')!=configuration['assignment'] or candidate.get('task_comparable') is not False):
            raise ValueError('Saved-only inspection cannot promote or omit an equipment alternative')
        if _json(snapshots[f'configs/{identity}.json'])!=configuration['config']:
            raise ValueError('Saved conditional configuration differs from original source roles')
        relative=f'candidates/{identity}'
        child_manifest,child_files=read_package_manifest(source/relative)
        if sha256(child_manifest).hexdigest()!=candidate.get('assembly_manifest_sha256'):
            raise ValueError('Saved child constructor seal differs from its result')
        child_result=read_package_file(source/relative,'result.json',max_bytes=128<<10,expected_sha256=child_files.get('result.json'))
        child=_json(child_result)
        if (child.get('status')!=candidate['status'] or child.get('reasons')!=candidate['reasons']
                or child.get('lab_directory') is not None):
            raise ValueError('Saved rejected constructor result differs from the candidate inventory')
        snapshots[relative+'/manifest.json']=child_manifest;snapshots[relative+'/result.json']=child_result
        total+=len(child_manifest)+len(child_result)
        if total>64<<20:raise ValueError('Saved rejection receipts exceed their byte bound')
    _check(deadline)
    if read_package_manifest(source)[0]!=manifest:raise ValueError('Saved study seal changed during inspection')
    report={'schema_version':1,'profile':PROFILE,'source_sha256':entry['source_sha256'],
        'source_validation':proof,'candidates':candidate_rows(source,result,project=project),
        'scope':SCOPE,'selection':'No candidate admitted; inspection of an existing sealed study, not a fresh construction.',
        'assembly_status':'abstained','assembly_reasons':result.get('reasons',[]),'api_calls':0,'inference_calls':0,'training_runs':0,
        'reuse':{'profile':'saved-rejected-equipment-study-v1','manifest_sha256':expected_manifest_sha256,
            'source_seals_revalidated':True,'alternatives_independently_rederived':True,'constructor_replayed':False,
            'copied_scope':'Outer conditional receipts plus child manifest/results only; no playable lab or complete child package claimed'}}
    return report,snapshots
