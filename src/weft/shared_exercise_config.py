"""Prepare explicit source-bound choices for the existing shared exercise builder."""

from collections import defaultdict
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import re
import stat
import sys
import time
import zipfile

from weft.equipment_proposal_config import CONFIG_PROFILE, MAX_CONFIG_BYTES, validate_proposal_config
from weft.io import local_time
from weft.observations import adapt_exercise
from weft.shared_world_evidence import build_shared_world_binding, _verify_document_clock
from weft.shared_world_packaging import read_package_file, validate_package_layout, write_package_files
from weft.source_equivalence import load_source_equivalence
from weft.world_evidence import _check_tree, _nonfinite, _unique_object, load_world_evidence

PROFILE = 'shared-exercise-config-package-v1'
SNAPSHOT_PROFILE = 'shared-exercise-config-snapshot-v1'
REQUEST_PROFILE = 'shared-exercise-config-request-v1'
METADATA_PROFILE = 'shared-exercise-config-metadata-v1'
MAX_INPUT_BYTES, MAX_INPUT_FILES = 64 << 20, 256
MAX_OUTPUT_BYTES, MAX_OUTPUT_FILES = 96 << 20, 320
MAX_SNAPSHOT_BYTES = 8 << 20
MAX_SETUP_MANIFEST_BYTES = 256 << 10
REJECTED = {'failed', 'error', 'unsupported', 'rejected', 'cancelled', 'canceled'}
SCOPE = ('Explicit saved-source configuration only. Body, root, floor, geometry, task and lab admission remain '
         'with the unchanged shared-exercise-build path. No measured camera, depth, mass or contact is inferred.')


def _check(deadline, reserve=0.):
    if time.monotonic() + reserve >= deadline:
        raise TimeoutError('Guided configuration cooperative deadline exceeded')


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _pin(raw):
    return sha256(raw).hexdigest()


def _digest(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _json(raw, maximum):
    _require(type(raw) is bytes and 0 < len(raw) <= maximum, 'Bounded exact JSON bytes required')
    value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    _check_tree(value)
    _require(type(value) is dict, 'JSON document must be an object')
    return value


def _encode(value, maximum, deadline):
    _check(deadline)
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    _require(len(raw) <= maximum, 'Complete output exceeds its byte bound; no slots dropped')
    _check(deadline)
    return raw


def _path(value):
    path = Path(value).absolute()
    _require('..' not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
             'Paths must not traverse symlinks or parent components')
    return path


def _identity(path):
    info = path.lstat()
    _require(stat.S_ISREG(info.st_mode), 'Pinned input must be a regular file')
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


class _Reader:
    def __init__(self, deadline):
        self.deadline, self.raw, self.identities = deadline, {}, {}
        self.optional = {}
        self.bytes, self.operations = 0, 0
        self.outer_root, self.outer_pins = None, None

    def pin(self, path, raw, before=None):
        _check(self.deadline)
        path = _path(path)
        if self.outer_root is not None and path.is_relative_to(self.outer_root):
            relative = path.relative_to(self.outer_root).as_posix()
            _require(self.outer_pins.get(relative) == _pin(raw), 'Resume source differs from its exact outer seal')
        after = _identity(path)
        _require(before is None or before == after, 'Input changed during its read')
        old = self.raw.get(path)
        _require(old is None or old == raw, 'Input changed within this preparation')
        if old is None:
            _require(len(self.raw) < MAX_INPUT_FILES, 'Complete inputs exceed256files; do not trim parents')
            self.raw[path], self.identities[path] = raw, after
        self.operations += 1
        self.bytes += len(raw)
        _require(self.bytes <= MAX_INPUT_BYTES, 'Aggregate input reads exceed64MiB')
        return raw

    def read(self, path, maximum=MAX_INPUT_BYTES, expected=None):
        path = _path(path)
        if path in self.raw:
            raw = self.raw[path]
            _require(expected is None or _pin(raw) == expected, 'Cached input pin differs')
            return raw
        _check(self.deadline)
        before = _identity(path)
        cap = min(maximum, MAX_INPUT_BYTES - self.bytes)
        _require(0 <= before[2] <= cap, 'Input exceeds its remaining byte bound')
        raw = read_package_file(path.parent, path.name, max_bytes=max(1, cap), expected_sha256=expected)
        return self.pin(path, raw, before)

    def manifest(self, root, expected=None):
        raw = self.read(root / 'manifest.json', 1 << 20, expected)
        value = _json(raw, 1 << 20); pins = value.get('files')
        _require(type(pins) is dict and 0 < len(pins) <= MAX_OUTPUT_FILES
                 and 'manifest.json' not in pins and all(_digest(p) for p in pins.values()),
                 'Exact bounded parent file map required')
        validate_package_layout(pins)
        return raw, value, pins

    def package(self, root, expected=None):
        raw, manifest, pins = self.manifest(root, expected)
        _require(len(self.raw) + sum(root / name not in self.raw for name in pins) <= MAX_INPUT_FILES,
                 'Complete source parent exceeds the aggregate256-file bound; no subset copied')
        values = {'manifest.json': raw}
        for name, digest in sorted(pins.items()):
            values[name] = self.read(root / name, expected=digest)
        return values, manifest

    def proof(self, root):
        _require(self.watch_optional(root/'failure.json') is None, 'Failed source-mapping marker prevents preparation')
        raw, manifest, pins = self.manifest(root)
        names = sorted([*pins, 'manifest.json'])
        _require(len(names) <= 129, 'Source mapping exceeds the existing proof file bound')
        before = {name: _identity(_path(root / name)) for name in names}
        proof = load_source_equivalence(root, max_input_bytes=MAX_INPUT_BYTES-self.bytes,
            max_files=min(129, MAX_INPUT_FILES-len(self.raw)+1), deadline=self.deadline)
        _require(set(proof.file_names) == set(names) and proof.raw_bytes('manifest.json') == raw,
                 'Source proof inventory changed')
        values = {}
        for name in names:
            values[name] = self.pin(root / name, proof.raw_bytes(name), before[name])
        return proof, values, manifest

    def recheck(self):
        for path, identity in self.identities.items():
            _check(self.deadline)
            _require(_identity(_path(path)) == identity, 'Source changed before terminal publication')
        for path,identity in self.optional.items():
            _check(self.deadline)
            try:current=_identity(_path(path))
            except FileNotFoundError:current=None
            _require(current==identity,'Optional failure/lifecycle marker changed before terminal publication')

    def watch_optional(self, path):
        path=_path(path)
        try:identity=_identity(path)
        except FileNotFoundError:identity=None
        self.optional[path]=identity
        return identity


def _source_descriptor(directory, values, manifest, profile):
    _require(all(manifest.get(key) is None or type(manifest[key]) is str and len(manifest[key]) <= 160
                 for key in ('profile', 'status')), 'Parent profile/status must be bounded scalar metadata')
    return {'directory': directory, 'manifest_sha256': _pin(values['manifest.json']),
            'profile': manifest.get('profile', profile),
            'status': manifest.get('status', 'not_declared_by_parent_seal'), 'files': manifest['files']}


def _issue(code, message):
    return {'code': code, 'message': message}


def _source_ready(manifest, values, label, issues):
    if manifest.get('status') in REJECTED or 'failure.json' in values:
        issues.append(_issue(label+'_parent_failed', 'The exact parent failure is retained; prepare a usable body/evidence source separately.'))
    for name in ('report.json', 'status.json', 'reconstruction.json'):
        if name in values and _json(values[name], 1 << 20).get('status') in REJECTED:
            issues.append(_issue(label+'_saved_status_failed', 'A saved parent status remains failed or rejected.'))


def _body_and_observations(values, issues):
    _require({'human.json', 'evidence.json'} <= values.keys(), 'Human run needs its exact human.json and evidence.json')
    body = _json(values['human.json'], 8 << 20)
    pose = _json(values['evidence.json'], 8 << 20)
    _require(_digest(body.get('source_sha256')) and pose.get('source_sha256') == body['source_sha256'],
             'Original human/pose source identities differ')
    _require(type(pose.get('fps')) in (int, float) and math.isfinite(pose['fps']) and 5 <= pose['fps'] <= 240
             and type(pose.get('frames')) is list and len(pose['frames']) <= 900,
             'Original pose requires a bounded explicit nominal clock and at most900rows')
    if body.get('profile') != 'exercise-v3' or 'camera_reference' in body:
        issues.append(_issue('unsupported_body_profile', 'This existing build route requires a mobile exercise-v3 body without camera_reference.'))
    if 'body-source.json' in values:
        marker = _json(values['body-source.json'], 256 << 10)
        _require(marker.get('profile') == 'exercise-body-source-v1' and marker.get('source_sha256') == body['source_sha256'],
                 'Body-source marker must bind the exact original human source')
        if marker.get('status') != 'construction_only' or marker.get('simulation_ready') is not False:
            issues.append(_issue('unsupported_body_source_status', 'Original body-source construction-only status is required.'))
        _require('observations.json' in values, 'Body-source preparation lacks its original observation document')
        human_raw = values['observations.json']
    else:
        _require('scene.json' in values, 'Mobile run requires its sealed original scene.json')
        scene = _json(values['scene.json'], 8 << 20)
        if scene.get('profile') != 'exercise-v3' or scene.get('source_sha256', body['source_sha256']) != body['source_sha256']:
            issues.append(_issue('unsupported_original_scene', 'The original scene is not the same mobile body source.'))
        # Match the existing builder's pure derivation without importing its native constructor.
        derived=adapt_exercise(pose,scene=scene)
        derived['provenance']['input_artifacts']={name:_pin(values[name]) for name in ('evidence.json','scene.json')}
        human_raw=values.get('observations.json')
        if human_raw is not None and _json(human_raw,16<<20)!=derived:
            issues.append(_issue('legacy_observations_disagree_with_builder',
                'Saved human observations differ from the existing builder derivation; retain both inputs but do not silently replace partitions or source semantics.'))
        if human_raw is None:
            human_raw=(json.dumps(derived,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    _require(len(human_raw) <= 16 << 20, 'Human observation document exceeds16MiB')
    refs, evidence, roots = body.get('reference'), body.get('reference_evidence'), body.get('root_reference')
    _require(type(refs) is list and type(evidence) is list and type(roots) is list
             and 3 <= len(refs) <= 900 and len(refs) == len(evidence) == len(roots),
             'All3–900 original body/reference/root rows are required')
    references = []; previous = -1
    for i, (ref, origin, root) in enumerate(zip(refs, evidence, roots, strict=True)):
        _require(all(type(row) is dict for row in (ref, origin, root)), 'Body reference rows must be mappings')
        match = re.fullmatch(re.escape(body['source_sha256'])+r':frame:(\d+)', str(origin.get('evidence_id','')))
        _require(match is not None and type(origin.get('time_s')) in (int,float)
                 and math.isfinite(origin['time_s']) and origin['time_s'] >= 0
                 and ref.get('time_s') == origin.get('time_s') == root.get('time_s'),
                 'Body reference IDs and clocks must remain exact')
        frame = int(match[1]); _require(previous < frame < 1800, 'Original body reference frames must be ordered and bounded')
        previous = frame
        references.append({'index':i,'source_frame':frame,'evidence_id':origin['evidence_id'],'time_s':origin['time_s']})
    if body.get('mobility', {}).get('floor_height') != 0 or 'ground_plane' in body.get('mobility', {}):
        issues.append(_issue('unsupported_original_floor', 'Existing box proposal requires the unchanged horizontal zero-height body floor; no floor is moved here.'))
    return body, pose, human_raw, references


def _equipment_observations(values):
    names = [n for n in ('scene-observations.json', 'evidence/scene-observations.json') if n in values]
    _require(names and all(values[n] == values[names[0]] for n in names), 'One exact canonical equipment observation identity is required')
    return values[names[0]]


def _availability(raw, role, source_sha, proof, references, deadline):
    pin = _pin(raw); registry = load_world_evidence({pin:raw})
    source = json.loads(registry.construction_bytes())['sources'][0]
    contract = source['source']; count = contract['decoded_frames']
    _require(count <= 1800 and contract.get('decode_interval', [0,count-1]) == [0,count-1]
             and contract.get('source_complete') is not False and len(source['objects']) <= 12,
             'Complete source coverage must fit1800frames/12entities')
    alignment = 'original' if contract['sha256'] == source_sha else 'unavailable' if proof is None else 'verified_derivative'
    if proof is not None:
        _verify_document_clock(source, proof, derivative=alignment == 'verified_derivative')
    frames = {r['source_frame'] for r in references}
    # Filtered evidence metadata never carries withheld payload coordinates.
    slots = []
    for row in source['observations']:
        _check(deadline)
        original_frame = None if alignment == 'unavailable' else row['frame']
        admitted = row['admission'] == 'included' and row['partition'] == 'construction'
        box = row['measurements'].get('box', {}) if admitted else {}
        box_value = box.get('value', [])
        bbox = ((box.get('coordinate_system'),box.get('unit'),box.get('status')) == ('image','pixels','inferred')
                and type(box_value) is list and len(box_value) == 4 and box_value[2] > 0 and box_value[3] > 0)
        reason = ('source_alignment_unavailable' if original_frame is None else 'partition_withheld' if not admitted else
                  'source_visibility_unavailable' if row['visibility'] not in ('detected','partial') else
                  'not_original_body_reference' if original_frame not in frames else
                  'ambiguous_or_missing_evidence' if len(row['evidence_refs']) != 1 else
                  'inferred_image_box_unavailable' if role == 'equipment' and not bbox else None)
        slots.append({'source_frame':row['frame'],'original_frame':original_frame,'time_s':row['time_s'],
            'object_id':row['object_id'],'observation_id':row['id'],'partition':row['partition'],
            'visibility':row['visibility'],'admission':row['admission'],'reason':reason,
            'evidence_ids':[r['evidence_id'] for r in row['evidence_refs']], 'bbox_available':bbox,
            'body_reference_available':original_frame in frames,'selection_eligible':reason is None})
    entities = [{'object_id':o['id'],'label':None,'kind':o['properties']['kind']['value']} for o in source['objects']]
    return {'role':role,'document_sha256':pin,'source_sha256':contract['sha256'],'image':source['image'],
            'time':source['time'],'decoded_frames':count,'metadata_frames':contract['metadata_frames'],
            'alignment':alignment,'entities':entities,'slots':slots}, source


def _snapshot(human_values, equipment_values, input_descriptors, proofs, deadline):
    issues = []
    _source_ready(_json(human_values['manifest.json'],1<<20),human_values,'human',issues)
    _source_ready(_json(equipment_values['manifest.json'],1<<20),equipment_values,'equipment',issues)
    body, pose, human_raw, references = _body_and_observations(human_values, issues)
    equipment_raw = _equipment_observations(equipment_values)
    documents = {_pin(human_raw):human_raw, _pin(equipment_raw):equipment_raw}
    _require(len(documents)==2,'Distinct exact human and equipment observation documents required')
    equipment_source = _json(equipment_raw, 32 << 20)['source']['sha256']; source_sha=body['source_sha256']
    mapping = {p.derivative_sha256:p for p in proofs}
    _require(len(mapping)==len(proofs) and all(p.original_sha256==source_sha for p in proofs), 'Mappings must uniquely bind this original source')
    required = {equipment_source} if equipment_source != source_sha else set()
    _require(set(mapping) <= required, 'Unused source mapping is not admitted')
    if set(mapping) != required:
        issues.append(_issue('source_mapping_required','Equipment bytes differ; rediscover with the exact complete original-to-equipment mapping.'))
    proof = mapping.get(equipment_source)
    human, original = _availability(human_raw,'human',source_sha,proof,references,deadline)
    equipment, other = _availability(equipment_raw,'equipment',source_sha,proof,references,deadline)
    _require(original['source']['sha256']==source_sha,'Human observations must preserve the original source')
    if original['image'] != other['image'] or human['decoded_frames'] != equipment['decoded_frames']:
        issues.append(_issue('source_dimensions_or_coverage_differ','Original source dimensions/complete decoded counts disagree.'))
    count=max(human['decoded_frames'],human['metadata_frames'])
    _require(count<=1800,'Original source slot bound exceeded')
    grouped=defaultdict(list)
    for r in references: grouped[r['source_frame']].append(r['index'])
    _require(all(r['source_frame']<human['decoded_frames'] for r in references),'Body reference exceeds original decoded slots')
    source_slots=[{'source_frame':f,'source_status':'decoded' if f<human['decoded_frames'] else 'metadata_only_not_decoded',
        'nominal_time_s':f/pose['fps'],'body_reference_indices':grouped[f]} for f in range(count)]
    snapshot={'schema_version':1,'profile':SNAPSHOT_PROFILE,'source_sha256':source_sha,'inputs':input_descriptors,
        'body':{'document_sha256':_pin(human_values['human.json']),'profile':body['profile'],
            'reference_count':len(references),'references':references,'validation':'deferred_to_existing_builder'},
        'source_slots':source_slots,'observation_sources':[human,equipment], 'issues':issues,
        'counts':{'original_source_slots':count,'decoded_source_slots':human['decoded_frames'],'body_references':len(references),
            'human_entity_slots':len(human['slots']),'equipment_entity_slots':len(equipment['slots']),
            'metadata_tail_slots':count-human['decoded_frames']},
        'world_admission':'not_evaluated','contact':'unknown','grasp':'unknown','load':'unknown'}
    return snapshot, documents


def _default_request(snapshot_pin):
    return {'schema_version':1,'profile':REQUEST_PROFILE,'snapshot_sha256':snapshot_pin,'human_entity_ref':None,
        'camera':None,'geometry_prior':None,'equipment':[],'target_object_id':None,
        'build_options':{'body_scale':1.0,'root_assistance':False}}


def _compile(request, snapshot, documents, proofs, deadline):
    fields=set(_default_request(request.get('snapshot_sha256')))
    _require(set(request)-{'frame_translation_xy'}==fields and type(request.get('schema_version')) is int
             and request['schema_version']==1 and request['profile']==REQUEST_PROFILE, 'Exact versioned guided request fields required')
    missing=[name for name in ('human_entity_ref','camera','geometry_prior','equipment','target_object_id') if request[name] is None or request[name]==[]]
    errors=[]; options=request['build_options']; valid_options=None
    if (type(options) is dict and set(options)=={'body_scale','root_assistance'}
            and type(options['body_scale']) in (int,float) and math.isfinite(options['body_scale']) and .5<=options['body_scale']<=2.
            and type(options['root_assistance']) is bool):
        valid_options=dict(options)
    else: errors.append(_issue('invalid_build_options','Body scale must be finite [.5,2] and root assistance an explicit boolean.'))
    config=None; frame_selection=None
    if not missing and not errors and not snapshot['issues']:
        try:
            human,equipment=snapshot['observation_sources']; ref=request['human_entity_ref']
            _require(type(ref) is dict and set(ref)=={'document_sha256','object_id'}
                     and ref['document_sha256']==human['document_sha256']
                     and ref['object_id'] in {e['object_id'] for e in human['entities']}, 'Unknown explicit human document/entity reference')
            _require(ref['object_id']=='person-0', 'The existing builder supports only the explicitly selected person-0 binding')
            human_rows={(r['object_id'],r['source_frame']):r for r in human['slots']}
            for original in snapshot['body']['references']:
                row=human_rows.get((ref['object_id'],original['source_frame']))
                _require(row is not None and row['selection_eligible'] and row['evidence_ids']==[original['evidence_id']]
                         and row['time_s']==original['time_s'], 'Selected human must retain every exact body reference and source clock')
            value={key:request[key] for key in ('camera','geometry_prior','equipment','target_object_id')}
            value.update(schema_version=1,profile=CONFIG_PROFILE,source_sha256=snapshot['source_sha256'])
            if 'frame_translation_xy' in request:value['frame_translation_xy']=request['frame_translation_xy']
            config=validate_proposal_config(value)
            indexed={(r['object_id'],r['source_frame']):r for r in equipment['slots']}
            for chosen in config['equipment']:
                if 'frame_selection' in chosen:
                    continue
                row=indexed.get((chosen['object_id'],chosen['source_frame']))
                _require(row is not None and row['selection_eligible'],'Selected equipment frame is unavailable, excluded or not an original body reference')
            ids=[r['object_id'] for r in config['equipment']]
            binding,construction=build_shared_world_binding(documents,source_sha256=snapshot['source_sha256'],world_ids=['human',*ids],
                entity_refs={'human':(ref['document_sha256'],ref['object_id']),**{i:(equipment['document_sha256'],i) for i in ids}},
                source_equivalences=proofs,deadline=deadline)
            automatic=[row['object_id'] for row in config['equipment'] if 'frame_selection' in row]
            if automatic:
                from weft.equipment_box_consensus import select_equipment_box_frames

                frame_selection=select_equipment_box_frames(construction, source_binding=binding,
                    body_references=snapshot['body']['references'], object_ids=automatic, deadline=deadline)
                _require(set(frame_selection)==set(automatic), 'Automatic frame selection must retain every requested object')
                for name,selection in frame_selection.items():
                    if selection['status']!='selected':
                        errors.append(_issue('equipment_frame_selection_unavailable',
                            f'No consistent source frame was selected for {name}. Inspect frame-selection.json; no fallback frame was chosen.'))
                if errors:
                    config=None
        except (ValueError,TypeError,KeyError,OverflowError) as error:
            config=None;errors.append(_issue(getattr(error,'code','source_or_config_unavailable'),str(error)[:500]))
    if missing:
        errors.append(_issue('request_incomplete','Required choices are missing; supplied partial choices have not been accepted.'))
    return config,missing,errors,valid_options,frame_selection


def _resume_budget(files, snapshot_raw, proofs):
    # A successful setup must remain resumable without rereading its report or generator.
    source_names=[n for n in files if n.startswith('sources/') or n=='inputs/human-lifecycle.json']
    size=sum(len(files[n]) for n in source_names)+sum(len(p.raw_bytes('manifest.json')) for p in proofs)
    size+=len(snapshot_raw)+MAX_SETUP_MANIFEST_BYTES+MAX_CONFIG_BYTES
    count=len(source_names)+3
    _require(size<=MAX_INPUT_BYTES and count<=MAX_INPUT_FILES,
             'Complete sources plus bounded resume metadata exceed the input budget; no source members trimmed')
    return {'maximum_selected_input_bytes':size,'maximum_selected_input_files':count,
            'scope':'complete_sources_snapshot_prior_seal_edited_request_only'}


def _runtime(deadline):
    root=Path(__file__).parent; values={}
    for name,module in sorted(tuple(sys.modules.items())):
        if name=='weft' or name.startswith('weft.'):
            file=getattr(module,'__file__',None)
            if file is None:continue
            path=_path(file)
            _require(path.is_relative_to(root) and path.suffix=='.py','Loaded Weft source is outside the active package')
            relative=path.relative_to(root).as_posix()
            if relative not in values:values[relative]=read_package_file(root,relative,max_bytes=512<<10)
            _require(len(values)<=128 and sum(map(len,values.values()))<=8<<20,'Generator source archive exceeds bound')
            _check(deadline)
    brand=read_package_file(root,'assets/brand.css',max_bytes=16<<10);values['assets/brand.css']=brand
    target=io.BytesIO()
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_STORED) as archive:
        for name,raw in values.items():archive.writestr(zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0)),raw)
    return target.getvalue(),{name:_pin(raw) for name,raw in values.items()},brand


def prepare_shared_exercise_config(human_run, evidence_run, out, *, source_mappings=(), resume=None,
        request=None, expected_manifest_sha256=None, max_seconds=30., progress=None):
    """Write a new immutable preparation; only the existing builder may admit a world."""
    _require(type(max_seconds) in (int,float) and math.isfinite(max_seconds) and 1<=max_seconds<=30,'Use a1–30second cooperative bound')
    _require(type(source_mappings) in (tuple,list) and len(source_mappings)<=7,'At most seven explicit source mappings')
    resuming=resume is not None
    _require((resuming and human_run is None and evidence_run is None and not source_mappings and request is not None)
             or (not resuming and human_run is not None and evidence_run is not None and request is None
                 and expected_manifest_sha256 is None), 'Choose discovery sources OR resume+request without replacement sources')
    _require(expected_manifest_sha256 is None or _digest(expected_manifest_sha256),'Invalid expected resume seal')
    target=_path(out); originals=[_path(resume)] if resuming else [_path(human_run),_path(evidence_run),*map(_path,source_mappings)]
    _require(not target.exists() and all(not target.is_relative_to(p) and not p.is_relative_to(target) for p in originals),
             'Output must be new and outside all immutable inputs')
    if request is not None:_require(not _path(request).is_relative_to(target),'Edited request must remain outside new output')
    started=time.monotonic();deadline=started+max_seconds;reader=_Reader(deadline);files={};created=False;terminal=None;request_raw=None
    phase='inputs'
    def emit(stage,message):
        if progress is not None:progress(stage,message)
    try:
        target.mkdir(parents=True);created=True
        prior=None; prior_snapshot_raw=None; lifecycle=None
        if resuming:
            outer=_path(resume);raw,outer_meta,pins=reader.manifest(outer,expected_manifest_sha256)
            _require(outer_meta.get('profile')==PROFILE and outer_meta.get('status') in ('needs_input','config_ready','unavailable'),
                     'Resume requires an exact completed guided setup, not a failed or foreign package')
            reader.outer_root,reader.outer_pins=outer,pins
            _require(reader.watch_optional(outer/'failure.json') is None,'Prior setup has a failure marker')
            lifecycle_state=reader.watch_optional(outer/'inputs/human-lifecycle.json')
            _require(lifecycle_state is None or 'inputs/human-lifecycle.json' in pins,'Unsealed prior lifecycle marker')
            prior_snapshot_raw=reader.read(outer/'snapshot.json',MAX_SNAPSHOT_BYTES,pins.get('snapshot.json'))
            prior=_json(prior_snapshot_raw,MAX_SNAPSHOT_BYTES)
            _require(prior.get('profile')==SNAPSHOT_PROFILE,'Unsupported prior snapshot')
            inputs=prior.get('inputs',{});mappings=inputs.get('source_mappings')
            _require(type(inputs) is dict and inputs.get('human',{}).get('directory')=='sources/human'
                     and inputs.get('equipment',{}).get('directory')=='sources/equipment'
                     and type(mappings) is list and len(mappings)<=7,'Prior source directories must follow the exact snapshot layout')
            _require(all(type(item) is dict and _digest(item.get('manifest_sha256'))
                         and item.get('directory')=='sources/mappings/'+item['manifest_sha256'] for item in mappings),
                     'Prior mapping directories must remain exact seal-named copies')
            human_root=outer/'sources/human';equipment_root=outer/'sources/equipment'
            mapping_roots=[outer/item['directory'] for item in mappings]
            request_raw=reader.read(_path(request),MAX_CONFIG_BYTES)
            if 'inputs/human-lifecycle.json' in pins:lifecycle=reader.read(outer/'inputs/human-lifecycle.json',64<<10)
        else:
            human_root,equipment_root=originals[:2];mapping_roots=originals[2:]
            if (human_root/'run/manifest.json').is_file():
                if reader.watch_optional(human_root/'status.json') is not None:lifecycle=reader.read(human_root/'status.json',64<<10)
                human_root=_path(human_root/'run')
        reader.watch_optional(human_root/'failure.json');reader.watch_optional(equipment_root/'failure.json')
        emit('inputs','Copy exact complete saved parents and mappings; no decoder, inference or world construction')
        human_values,human_manifest=reader.package(human_root)
        equipment_values,equipment_manifest=reader.package(equipment_root)
        if (human_root/'failure.json').exists():_require('failure.json' in human_values,'Unsealed human failure marker prevents preparation')
        if (equipment_root/'failure.json').exists():_require('failure.json' in equipment_values,'Unsealed equipment failure marker prevents preparation')
        for prefix,values in [('sources/human/',human_values),('sources/equipment/',equipment_values)]:
            files.update({prefix+n:raw for n,raw in values.items()})
        descriptors={'human':_source_descriptor('sources/human',human_values,human_manifest,'legacy_mobile_run'),
                     'equipment':_source_descriptor('sources/equipment',equipment_values,equipment_manifest,'original_equipment_evidence'),
                     'source_mappings':[]}
        proofs=[]
        for root in mapping_roots:
            item,values,manifest=reader.proof(_path(root));proofs.append(item)
            directory='sources/mappings/'+item.manifest_sha256
            files.update({directory+'/'+n:raw for n,raw in values.items()})
            desc=_source_descriptor(directory,values,manifest,'whole-video-source-equivalence-v1')
            desc.update(original_sha256=item.original_sha256,derivative_sha256=item.derivative_sha256,
                        frame_count=len(item.frames),time_base=item.time_base,verification='full_saved_bundle_verified_no_decode')
            descriptors['source_mappings'].append(desc)
        if lifecycle is not None:files['inputs/human-lifecycle.json']=lifecycle
        phase='availability';emit(phase,'Retain every original source/entity slot; do not select objects or frames')
        snapshot,documents=_snapshot(human_values,equipment_values,descriptors,proofs,deadline)
        files['inputs/human-observations.json']=documents[snapshot['observation_sources'][0]['document_sha256']]
        if lifecycle is not None and _json(lifecycle,64<<10).get('status') in REJECTED:
            snapshot['issues'].append(_issue('human_lifecycle_failed','Exact saved outer human lifecycle remains failed.'))
        snapshot_raw=_encode(snapshot,MAX_SNAPSHOT_BYTES,deadline);snapshot_pin=_pin(snapshot_raw)
        resume_budget=_resume_budget(files,snapshot_raw,proofs)
        if resuming:_require(snapshot_raw==prior_snapshot_raw,'Copied source availability changed; rediscover rather than reuse stale choices')
        else:request_raw=_encode(_default_request(snapshot_pin),MAX_CONFIG_BYTES,deadline)
        requested=_json(request_raw,MAX_CONFIG_BYTES)
        _require(requested.get('snapshot_sha256')==snapshot_pin,'Edited request belongs to another exact source snapshot')
        config,missing,errors,options,frame_selection=_compile(requested,snapshot,documents,proofs,deadline)
        status='unavailable' if snapshot['issues'] else 'config_ready' if config is not None else 'needs_input'
        config_raw=None if config is None else _encode(config,MAX_CONFIG_BYTES,deadline)
        metadata={'schema_version':1,'profile':METADATA_PROFILE,'status':status,'snapshot_sha256':snapshot_pin,
            'request_sha256':_pin(request_raw),'config_sha256':None if config_raw is None else _pin(config_raw),
            'missing_fields':missing,'issues':snapshot['issues']+errors,'counts':snapshot['counts'],
            'build_options':options,'scope':SCOPE,'world_admission':'not_evaluated','contact':'unknown','grasp':'unknown','load':'unknown'}
        files.update({'snapshot.json':snapshot_raw,'request.json':request_raw,'metadata.json':_encode(metadata,256<<10,deadline)})
        if frame_selection is not None:
            files['frame-selection.json']=_encode(frame_selection,4<<20,deadline)
            metadata['frame_selection_sha256']=_pin(files['frame-selection.json'])
            files['metadata.json']=_encode(metadata,256<<10,deadline)
        if config_raw is not None:files['config.json']=config_raw
        phase='report';_check(deadline,2.);emit(phase,'Render saved choices and missing evidence; world admission remains separate')
        from weft.shared_exercise_config_report import build_shared_exercise_config_report
        report_inputs={name:files[name] for name in ('metadata.json','snapshot.json','request.json','config.json','frame-selection.json') if name in files}
        rendered=build_shared_exercise_config_report(report_inputs,max_seconds=min(5.,deadline-time.monotonic()-2.))
        _require(type(rendered) is dict and 'index.html' in rendered and len(rendered)<=32
                 and all(type(v) is bytes for v in rendered.values()) and sum(map(len,rendered.values()))<=16<<20,
                 'Offline report exceeds its fixed byte/file bound')
        validate_package_layout(rendered)
        files.update({'inspection/'+name:raw for name,raw in rendered.items()})
        archive,pins,brand=_runtime(deadline)
        _require(rendered.get('brand.css')==brand,'Report brand must match exact archived stylesheet')
        files['implementation-source.zip']=archive
        build_inputs=None if config is None else {'human_run':str(target/'sources/human'),
            'evidence_run':str(target/'sources/equipment'),'source_mappings':[str(target/d['directory']) for d in descriptors['source_mappings']],
            'config':str(target/'config.json')}
        result={'schema_version':1,'profile':PROFILE,'status':status,'sealed':True,'output_path':str(target),
            'metadata_path':str(target/'metadata.json'),'request_path':str(target/'request.json'),
            'config_path':None if config is None else str(target/'config.json'),'report_path':str(target/'inspection/index.html'),
            'build_inputs':build_inputs,'build_options':options,'full_reference':True,'snapshot_sha256':snapshot_pin,
            'world_admission':'not_evaluated','scope':SCOPE,'created_at_local':local_time(),
            'input_files':len(reader.raw),'input_read_operations':reader.operations,'input_bytes':reader.bytes,
            'runtime_sha256s':pins,'counts':snapshot['counts'],'work':{'video_decodes':0,'inference_calls':0,'native_model_calls':0,'physics_steps':0,'api_calls':0},
            'resume_input_budget':resume_budget,
            'bounds':{'input_bytes':MAX_INPUT_BYTES,'input_files':MAX_INPUT_FILES,'output_bytes':MAX_OUTPUT_BYTES,
                      'output_files':MAX_OUTPUT_FILES,'maximum_seconds':max_seconds,'time_limit_kind':'cooperative_not_hard_process_limit'}}
        if frame_selection is not None:
            result['work'].update(construction_frame_selection_calls=1, model_inference_calls=0)
        files['result.json']=_encode({**result,'sealed':False,'publication':'prepared_pending_terminal_seal'},256<<10,deadline)
        validate_package_layout(files);_require(len(files)+3<=MAX_OUTPUT_FILES and sum(map(len,files.values()))+(512<<10)<=MAX_OUTPUT_BYTES,
            'Complete output package exceeds its bound; no source parents trimmed')
        phase='write';emit(phase,'Write complete copied inputs, request and report to a new immutable package')
        for name,raw in files.items():
            _check(deadline);write_package_files(target,{name:raw})
        reader.recheck()
        terminal=_encode({'schema_version':1,'profile':PROFILE,'status':status,'files':{n:_pin(raw) for n,raw in files.items()}},256<<10,deadline)
        result['manifest_sha256']=_pin(terminal)
        _check(deadline);write_package_files(target,{'manifest.json':terminal})
        return result
    except Exception as error:
        if not created:raise
        failure={'schema_version':1,'profile':PROFILE,'status':'failed','sealed':False,'output_path':str(target),
            'report_path':None,'metadata_path':None,'request_path':None,'config_path':None,'build_inputs':None,'build_options':None,
            'full_reference':True,'phase':phase,'error':{'type':type(error).__name__,'message':str(error)[:1000]},
            'world_admission':'not_evaluated','input_bytes':reader.bytes,'input_files':len(reader.raw),'seconds':time.monotonic()-started}
        failure['input_pins']=[{'path':str(path),'sha256':_pin(raw),'bytes':len(raw)} for path,raw in reader.raw.items()]
        marker=target/'manifest.json'
        if marker.exists():
            existing=read_package_file(target,'manifest.json',max_bytes=256<<10)
            if terminal is not None and existing==terminal:
                sibling={**failure,'status':'caller_failed_after_terminal_publication','artifact_unchanged':True,'artifact_manifest_sha256':_pin(terminal)}
                write_package_files(target.parent,{target.name+'.caller-failure.json':(json.dumps(sibling,sort_keys=True)+'\n').encode()})
                raise
            # Invalid partial seals are not promoted or replaced with a successful setup.
            failure['manifest_sha256']=None
        raw=(json.dumps(failure,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        retained={'failure.json':raw}
        if request_raw is not None:retained['failure-inputs/request.json']=request_raw
        try:write_package_files(target,retained)
        except Exception:raise error
        return failure
