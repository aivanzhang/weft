"""One-shot local evidence copy; never reads provider credentials or changes source packs."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import signal
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MAX_BYTES = 10 * 1024**2
MAX_FILES = 350
MAX_SINGLE_BYTES = 1024**2
PARTITIONS = [f'companion-separation-v{i}' for i in range(1, 7)]
DANCES = [f'muse-panda-dance-probe-v{i}' for i in range(1, 4)]
COMBINED = ['muse-panda-dance-flight-probe-v1', 'muse-panda-dance-flight-v1']
MEDIA = 'muse-panda-dance-flight-media-v1'
NAMES = PARTITIONS + DANCES + COMBINED + [MEDIA]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def record(path):
    return {'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': digest(path)}


def write(path, value):
    if path.exists():
        raise ValueError('Refusing to overwrite evidence: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    signal.alarm(60)
    started = time.monotonic()
    if set(p.name for p in OUT.iterdir()) != {'build_receipt.py'}:
        raise ValueError('A fresh output directory containing only this script is required')
    copied, omitted_sensitive, selected, attempts = [], [], {}, []

    def select(path):
        if path.is_file():
            if path.is_symlink() or ROOT not in path.resolve().parents:
                raise ValueError('Evidence must be a regular in-workspace file')
            selected[str(path.relative_to(ROOT))] = path

    for name in NAMES:
        base = ROOT / 'outputs' / name
        for relative in ['manifest.json', 'job.json', 'config.json', 'finish-config.json', 'finish-report.json',
                         'native-report.json', 'partition-plan.json', 'rig-verification.json', 'rig-contract.json',
                         'render-attributes.json', 'tangent-repair.json', 'exported-nodes.json',
                         'resume-input-receipt.json', 'partition.log', 'finish.log', 'render.log',
                         'partition-handle.json', 'partition-monitor.json', 'finish-handle.json',
                         'finish-monitor.json', 'render-handle.json', 'render-monitor.json']:
            select(base / relative)
        for path in sorted((base / 'sources').iterdir()):
            if path.suffix in ('.py', '.json', '.txt'):
                select(path)
        if name in PARTITIONS:
            for image in ['open-gap-front.png', 'open-gap-back.png', 'companion-back-close.png']:
                select(base / image)
        elif name != MEDIA:
            state = 'dance' if name in DANCES else 'dance_flight'
            select(base / 'renders' / state / '0001.png')
        else:
            select(base / 'frames/dance_flight/0000.png')
            select(base / 'frames/dance_flight/0060.png')

    shared = ROOT / 'outputs/complex-motion-v1'
    for name in ['plan.json', 'closure-amendment-v1.json', 'closure-amendment-v2.json',
                 'dance-probe-amendment-v1.json', 'dance-weight-repair-v1.json',
                 'combined-probe-v1-plan.json', 'combined-candidate-v1-plan.json',
                 'normal-repair-plan-v1.json', 'media-v1-stopped.json',
                 'dance-probe-contact.png', 'dance-probe-v2-contact.png', 'dance-probe-v3-contact.png',
                 'combined-probe-v1-contact.png', 'combined-v1-contact.png']:
        select(shared / name)
    diagnostics = ROOT / 'outputs/companion-separation-diagnostic-v1'
    for name in ['unsupported-initial-method.json', 'boundary-diagnostics.json', 'dp-probe-v1.json',
                 'v5-export-audit.json', 'v6-export-audit.json', 'v7-export-audit.json']:
        select(diagnostics / name)
    select(ROOT / 'outputs/companion-separation-v7/config.json')
    select(ROOT / 'outputs/companion-separation-v7/partition-plan.json')
    if len(selected) > MAX_FILES:
        raise ValueError('Evidence selection exceeds the file-count cap')

    credential = re.compile(rb'(?i)(?:authorization|api[_-]?key|access[_-]?token|client[_-]?secret)\s*[:=]\s*[\"\x27]?(?:bearer\s+)?[A-Za-z0-9_./+\-]{8,}')
    total = 0
    for relative, source in sorted(selected.items()):
        size = source.stat().st_size
        if size > MAX_SINGLE_BYTES:
            raise ValueError('Selected evidence exceeds one MiB: ' + relative)
        data = source.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if source.suffix != '.png' and (re.search(rb'https?://', data) or credential.search(data)):
            omitted_sensitive.append({'source_path': relative, 'bytes': size, 'sha256': sha,
                'reason': 'Text contains a URL or possible credential assignment; bytes omitted, never printed.'})
            continue
        total += size
        if total > MAX_BYTES - 1024**2:
            raise ValueError('Evidence exceeds nine MiB copy envelope; one MiB reserved for receipts')
        target = OUT / 'retained' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
        if digest(source) != sha or digest(target) != sha:
            raise ValueError('Evidence changed during copying: ' + relative)
        copied.append({'path': str(target.relative_to(OUT)), 'source_path': relative, 'bytes': size,
                       'sha256': sha, 'byte_identical_source_copy': True})

    reasons = {
        PARTITIONS[0]: ('failed_cpu', 'Directed cut-boundary fans were open, branched or inconsistently wound; no model or native job.', ['manifest.json']),
        PARTITIONS[1]: ('failed_native', 'Blender 4.5 tessellate_polygon returned integer indices; the initial adapter treated them as vectors and raised TypeError before export.', ['partition.log', 'job.json']),
        PARTITIONS[2]: ('failed_native', 'The planar projected cap violated its area-overlap gate; later diagnostics showed genuinely nonplanar/self-crossing projections, not just a numerical tolerance issue.', ['partition.log', 'job.json']),
        PARTITIONS[3]: ('rejected_appearance', 'Two pinched loops remained unfilled; separated views exposed a panda-cheek notch and dark/open companion rear strips. 133 screened cap/source intersections also remained.', ['native-report.json', 'open-gap-front.png', 'open-gap-back.png']),
        PARTITIONS[4]: ('rejected_appearance', 'Pinch splitting filled the cheek, but companion hidden closures inherited charcoal body-contact pixels. Internal intersection/nonmanifold diagnostics remained unresolved.', ['native-report.json', 'open-gap-back.png']),
        PARTITIONS[5]: ('superseded_appearance', 'Dominant-channel palette and bounded seam repair removed broad dark contamination; pale rear-head contact strips remained and motivated explicit v7 repair regions.', ['native-report.json', 'companion-back-close.png']),
        DANCES[0]: ('superseded_partial_motion', 'Whole-body toy hops were stiff and did not include the clarified shoulder-dragon takeoff; not a completed combined dance-and-flight delivery.', ['finish-config.json', 'renders/dance/0001.png']),
        DANCES[1]: ('rejected_deformation', 'Added toe motion visibly creased the belly; stored sampled worst edge extension is 17.7465 mm.', ['finish-report.json', 'renders/dance/0001.png']),
        DANCES[2]: ('rejected_deformation', 'Tighter foot-region weights worsened ankle stretching; stored sampled worst edge extension is 25.3237 mm. Later choreography retained the original smooth weights and reduced toe motion.', ['finish-report.json', 'renders/dance/0001.png']),
        COMBINED[0]: ('motion_only_known_rejected_appearance', 'Deliberate rig/choreography probe using rejected partition-v5 appearance while seam-color repair proceeded; never a final appearance candidate.', ['finish-config.json', 'renders/dance_flight/0001.png']),
        COMBINED[1]: ('rejected_appearance_after_motion_checks', 'Airborne native poses exposed an angular/bald gray shoulder despite an exact part fur mask. Motion checks did not establish appearance acceptance; normal/fur continuity repair followed.', ['finish-report.json', 'renders/dance_flight/0001.png']),
        MEDIA: ('stopped_rejected_parent_appearance', 'Owned local render was deliberately stopped after the parent v1 shoulder appearance rejection; 80 completed PNG frames remain. Exit -15 is a stop, not a successful full media export.', ['job.json', 'manifest.json']),
    }
    retained_sources = {entry['source_path'] for entry in copied}
    for name in NAMES:
        base = ROOT / 'outputs' / name
        manifest = json.loads((base / 'manifest.json').read_text())
        config_files = [p for p in [base/'config.json', base/'finish-config.json', base/'sources/scene.json',
                                   base/'sources/output-spec.json'] if p.is_file()]
        disposition, reason, evidence = reasons[name]
        original_artifacts = manifest.get('artifacts', [])
        if len(original_artifacts) > 1000:
            raise ValueError('Original manifest inventory exceeds the bound')
        omission = [{'path': row['path'], 'bytes': row['bytes'], 'sha256': row['sha256'],
                     'hash_basis': 'recorded in original manifest; omitted bytes not reread in this compacting step'}
                    for row in original_artifacts if str((base / row['path']).relative_to(ROOT)) not in retained_sources]
        job = json.loads((base/'job.json').read_text()) if (base/'job.json').is_file() else None
        attempt = {'attempt': name, 'original_directory': str(base.relative_to(ROOT)),
            'original_manifest_status': manifest.get('status'), 'history_disposition': disposition,
            'rejection_or_supersession_reason': reason, 'reason_scope': 'Recorded implementation failure or local visual-review decision; not a new independent image/model evaluation.',
            'evidence_paths': [str((Path('retained')/base.relative_to(ROOT)/x)) for x in evidence],
            'original_manifest': record(base/'manifest.json'),
            'original_job': record(base/'job.json') if job else None,
            'job_absence_reason': None if job else 'CPU preflight failed before native job creation',
            'original_configs': [record(p) for p in config_files], 'omitted_original_manifest_artifacts': omission,
            'original_source_path': manifest.get('source_path'), 'original_source_sha256': manifest.get('source_sha256'),
            'native_job_summary': {key:job.get(key) for key in ['elapsed_seconds','exit_code','sampled_peak_rss_bytes','final_output_bytes','status','failure']} if job else None}
        if name in DANCES[1:]:
            finish = json.loads((base/'finish-report.json').read_text())
            outliers = [row for state in finish['states'] for row in state.get('deformation_outliers', [])]
            attempt['maximum_recorded_edge_excess_m'] = max(row['excess_length'] for row in outliers)
        if name == MEDIA:
            attempt['stop_receipt'] = 'retained/outputs/complex-motion-v1/media-v1-stopped.json'
            attempt['completed_png_frames_from_stop_receipt'] = 80
            attempt['selected_frame_copies'] = ['frames/dance_flight/0000.png', 'frames/dance_flight/0060.png']
        attempts.append(attempt)

    original = ROOT/'outputs/muse-panda-smooth-base-v2/stages/mesh/downloads/model.glb'
    v7 = ROOT/'outputs/companion-separation-v7/diagnostic-model.glb'
    source = record(original)
    partition = record(v7)
    if source['sha256'] != 'f3b8c6f48222ef2e588ef1798f788a79909a660f5467b92719d2242b7ad47188' or partition['sha256'] != '734ebdbbd06c0428a57f817d1a5deed8ff7078016ab262abb313513d94f293ca':
        raise ValueError('Pinned source or v7 model changed')
    lineage = {'schema_version': 1, 'original_source': source, 'v7_partition': partition,
        'v7_manifest': record(ROOT/'outputs/companion-separation-v7/manifest.json'),
        'v7_errata_audit_original_path': 'outputs/companion-separation-diagnostic-v1/v7-export-audit.json',
        'v7_errata_audit_copy': 'retained/outputs/companion-separation-diagnostic-v1/v7-export-audit.json',
        'changes': {'source_triangle_positions_preserved': 62117, 'source_corner_normals_preserved': True,
            'original_encoded_basecolor_sha256': '0355a39bdf4dd84b0cc4ab9f291231d39f94bce2dd81196bc075923cfde2481d',
            'original_image_bytes_preserved': True, 'authored_closure_triangles': 497,
            'original_face_uv_repairs': 209, 'body_uv_repaired_faces': 124, 'companion_uv_repaired_faces': 85,
            'repair_contact_band_faces': 999, 'exact_coincident_boundary_aliases': 2,
            'original_face_winding_reversals': [37445, 49491, 52411],
            'native_vertices': 31333, 'exported_vertices': 51055, 'total_triangles': 62614,
            'units': 'Authored 1.2-meter height; not measured anatomy',
            'axes_blender': '+Z up / -Y forward / +Y back'},
        'residual_limits': {'unfilled_loops': 0, 'cross_part_edges': 0,
            'four_valence_edges': 21, 'screened_cap_cap_intersections': 2, 'screened_cap_source_intersections': 208,
            'source_geometric_components_after_split': 18, 'watertight_verified': False,
            'physics_ready': False, 'weft_runtime_verified': False,
            'surface_scope': 'Locally authored stylized hidden closures and seam UVs; not recovered unseen geometry. Small rear contact marks and under-arm/neck seam remained in low-sample source diagnostics.'},
        'errata': ['The sealed v7 generic sentence that pinched loops remain open is stale: unrepaired_loops is empty.',
                   'The sealed generic claim that every original UV is retained is stale: 209 original-face UV repairs are explicitly inventoried.',
                   'Sealed packs are unchanged; this receipt and copied v7 audit clarify the stale narrative only.'],
        'rights': 'User-authorized local derivative. Original creator license and rights remain independently unverified; generated metadata is provenance, not rights verification.'}
    write(OUT/'attempts.json', {'schema_version': 1, 'attempts': attempts})
    write(OUT/'source-lineage.json', lineage)
    write(OUT/'copy-receipt.json', {'schema_version':1,'copies':copied,'omitted_sensitive_text':omitted_sensitive,
        'all_copies_byte_identical':True,'originals_modified':False,'complete_recreation':False,
        'omission_scope':'Large GLB/BLEND/NPZ binaries, textures, most image frames and some unselected files remain in their original local directories. Omitted hashes copied from original manifests are not fresh full-original verification.'})
    readme = '''# Compound motion rejected-attempt history

This is a compact provenance archive, not a playable asset pack or complete recreation.
The full original directories remain unchanged locally. No provider, native renderer,
network, credential access or new generation was used to build this archive.

- `attempts.json`: twelve failed, superseded, motion-only or appearance-rejected attempts;
  exact original manifest/job/config identities and reasons remain distinct from
  the historical statuses inside the byte-preserved manifests.
- `source-lineage.json`: pinned original/v7 source hashes, authored changes, v7 report
  errata and unresolved intersection/nonmanifold limits. The partition is not
  watertight-certified, physics-ready or Weft-runtime-verified.
- `copy-receipt.json`: byte-identical selected JSON/log/source/contact-image copies.
- `manifest.json`: portable hashes for every retained archive file except itself.
- `build_receipt.py`: exact one-shot bounded local copy procedure. It refuses to
  overwrite a populated archive. Source files are copied, never executed.

Selected PNGs are existing diagnostic evidence, not newly rendered frames.
The stopped media's two retained frames do not constitute a complete clip. Its
80 completed source PNGs remain local; no successful complete media claim is made.
Small saved-source snapshots are historical implementation evidence, not a
standalone runnable dependency bundle. Omitted binaries and frames are listed
from original manifests and must be obtained from those original local packs.

The accepted later candidate and full v7 source are saved separately by the parent
workstream; this archive does not promote rejected attempts or certify appearance.
'''
    (OUT/'README.md').write_text(readme)
    artifacts = []
    for path in sorted(OUT.rglob('*')):
        if path.is_file():
            artifacts.append({'path':str(path.relative_to(OUT)), 'bytes':path.stat().st_size,'sha256':digest(path)})
    payload = {'schema_version':1,'kind':'compact_rejected_attempt_provenance','status':'verified_selected_evidence',
        'recorded_at':datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
        'attempts':len(attempts),'complete_recreation':False,'originals_modified':False,
        'new_provider_calls':0,'new_provider_cost_usd':0,'native_jobs':0,'network_calls':0,
        'limits':{'total_bytes':MAX_BYTES,'copied_file_bytes':MAX_SINGLE_BYTES,'copied_files':MAX_FILES,'wall_seconds':60},
        'copied_artifacts':len(copied),'copied_bytes':total,'retained_artifacts_excluding_manifest':len(artifacts),
        'retained_bytes_excluding_manifest':sum(a['bytes'] for a in artifacts),
        'verification':'Every retained copy was hashed against its actual source bytes; selected source packs remain unchanged. This manifest excludes its own hash.',
        'elapsed_seconds':round(time.monotonic()-started,3),'artifacts':artifacts}
    write(OUT/'manifest.json',payload)
    files = [p for p in OUT.rglob('*') if p.is_file()]
    actual_total = sum(p.stat().st_size for p in files)
    if actual_total > MAX_BYTES:
        raise ValueError('Final archive exceeds ten MiB')
    for item in artifacts:
        path=OUT/item['path']
        if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise ValueError('Portable inventory mismatch')
    print(json.dumps({'status':payload['status'],'files':len(files),'bytes':actual_total,'copies':len(copied),
        'attempts':len(attempts),'omitted_sensitive_files':len(omitted_sensitive),
        'manifest_sha256':digest(OUT/'manifest.json'),'elapsed_seconds':round(time.monotonic()-started,3)},indent=2))


if __name__ == '__main__':
    main()
