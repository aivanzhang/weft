"""Read only the named readiness inputs; GLB census is not geometry validation."""
from __future__ import annotations

import hashlib
import json
import resource
import struct
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SMALL = [
    'TODO.md', 'README.md', 'docs/hands-off-studio.md', 'docs/prompt-pipeline.md',
    'docs/source-uv-control.md', 'docs/fur-workflow.md', 'docs/hd-quality-pass-v1.md',
    'avatar_assets/pipeline.py', 'avatar_assets/pipeline_prompts.py',
    'avatar_assets/quality_policy.py', 'avatar_assets/finish_generated.py',
    'avatar_assets/part_rigs.py', 'avatar_assets/assemble_generated.py',
    'scripts/blender_finish_generated.py', 'scripts/native_render_tasks.py',
    'outputs/fresh-acceptance-preparation-v1/requests.json',
    'outputs/fresh-acceptance-preparation-v1/estimate.json',
    'outputs/fresh-acceptance-preparation-v1/allowance-request.json',
    'outputs/fresh-acceptance-estimate-v2/estimate.json',
    'outputs/muse-panda-smooth-base-v2/plan.json',
    'outputs/muse-panda-smooth-fur-v3/manifest.json',
    'outputs/muse-panda-smooth-fur-v3/finish-report.json',
    'outputs/muse-panda-dance-probe-v2/manifest.json',
    'outputs/muse-panda-dance-probe-v2/finish-report.json',
    'outputs/agent-dragon-polish-v1/manifest.json',
    'outputs/agent-dragon-polish-v1/finish-report.json',
    'outputs/agent-robot-v3/request.json', 'outputs/agent-robot-v3/delivery.json',
    'outputs/lamp-hd-build-v2/manifest.json', 'outputs/lamp-hd-build-v2/rig-contract.json',
    'outputs/fox-hd-build-v2/manifest.json',
    'outputs/panda-uv2-appearance-review-v1/review.json',
    'outputs/panda-cheek-scar-selection-v1/manifest.json',
    'outputs/panda-cheek-curved-candidate-v1/result.json',
    'outputs/studio-rigged-robot-preparation-v1/request.json',
    'examples/meshy/muse-panda-smooth-fur-v3.json',
    'examples/meshy/muse-panda-dance-v4.json',
    'examples/meshy/agent-dragon-polish-v1.json',
    'examples/scenes/lamp-hd-v1.json', 'examples/scenes/fox-hd-v3.json',
]
MODELS = {
    'smooth_panda_raw': ('outputs/muse-panda-smooth-base-v2/stages/mesh/downloads/model.glb', 'f3b8c6f48222ef2e588ef1798f788a79909a660f5467b92719d2242b7ad47188'),
    'smooth_panda_fur_demo': ('outputs/muse-panda-smooth-fur-v3/model.glb', 'a5664da6dd59e4f9fed0e9154c7af75e198d9a9750edce74d258687a98c3ab77'),
    'smooth_panda_dance_probe': ('outputs/muse-panda-dance-probe-v2/model.glb', None),
    'dragon_raw': ('outputs/emerald-dragon-meshy-v1/stages/texture/downloads/model.glb', 'a5e034586961d2999846d185321e6b1b4289e8138ce4b11a4a19746700109efb'),
    'dragon_polished': ('outputs/agent-dragon-polish-v1/model.glb', None),
    'robot_raw': ('outputs/agent-robot-v3/inputs/source-model.glb', None),
    'robot_legacy_delivery': ('outputs/agent-robot-v3/model-1/model.glb', 'e2e1c4e73a83773673a9343d3ae0baeddea84dbace9ac61ac7ee203a9ec89d3e'),
    'robot_provider_rig': ('outputs/workshop-robot-meshy-v1/stages/rig/downloads/model.glb', 'ee7477ccf173f480db7fe41924451e4483fa5fc369247fba679942b52df510a7'),
    'authored_lamp': ('outputs/lamp-hd-build-v2/model.glb', '774eae69f51ae1a0cd49c61dd19abeab19b847ba8b72f393ad409b4582500949'),
    'authored_fox': ('outputs/fox-hd-build-v2/model.glb', '0a2a088e7c912a53221ed6a76a65006786d49e48cf64455ab0ba671a0c82da78'),
    'uv_repaired_panda_source': ('outputs/panda-shoulder-uv-repair-v2/model.glb', '736ad72d408a43295098f44a29919c6436b37f741b7335f35a2c3f5f2007fb3e'),
    'rejected_uv2_control': ('outputs/panda-uv2-portable-control-v1/model.glb', '82c05f20956226b90358ba67d582fe3964623d5ff146f12ef76c9e5247ac8f87'),
}
VIEWED = [
    'outputs/muse-panda-smooth-fur-v3/portrait.png',
    'outputs/muse-panda-smooth-fur-export-preview-v3/front.png',
    'outputs/panda-uv2-portable-preview-v1/renders/dance_flight/0001.png',
]


def hash_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def record(relative):
    path = ROOT / relative
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024**2:
        raise ValueError('Missing, aliased or oversized named input: ' + relative)
    return {'path': relative, 'bytes': path.stat().st_size, 'sha256': hash_file(path)}


def census(relative):
    path = ROOT / relative
    with path.open('rb') as stream:
        magic, version, size = struct.unpack('<4sII', stream.read(12))
        length, kind = struct.unpack('<II', stream.read(8))
        if (magic, version, size) != (b'glTF', 2, path.stat().st_size) or kind != 0x4E4F534A or length > 8 * 1024**2:
            raise ValueError('Unexpected bounded GLB header: ' + relative)
        doc = json.loads(stream.read(length))
    accessors = doc.get('accessors', [])
    primitives = [p for mesh in doc.get('meshes', []) for p in mesh.get('primitives', [])]
    routes = []
    for index, material in enumerate(doc.get('materials', [])):
        pbr = material.get('pbrMetallicRoughness', {})
        tex = pbr.get('baseColorTexture')
        routes.append({'material': index, 'name': material.get('name'),
                       'base_color_image': doc['textures'][tex['index']].get('source') if tex else None,
                       'base_color_uv': tex.get('texCoord', 0) if tex else None,
                       'base_color_factor': pbr.get('baseColorFactor', [1, 1, 1, 1])})
    return {
        'scope': 'JSON header counts and bindings only; no fresh loading, skin, geometry or appearance verification',
        'meshes': len(doc.get('meshes', [])), 'primitives': len(primitives),
        'attribute_vertices_including_primitive_duplication': sum(accessors[p['attributes']['POSITION']]['count'] for p in primitives),
        'triangles_from_index_counts': sum(accessors[p['indices']]['count'] // 3 for p in primitives),
        'skins': len(doc.get('skins', [])),
        'skin_bones': [[doc['nodes'][j].get('name') for j in skin['joints']] for skin in doc.get('skins', [])],
        'clips': [a.get('name') for a in doc.get('animations', [])],
        'material_base_color_routes': routes,
    }


def main():
    started = time.monotonic()
    result = {'schema_version': 1, 'status': 'running', 'recorded_local': datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
              'scope': 'Named-file readiness inventory only; no recursive output scan or generation/render/test call',
              'inputs': [], 'models': {}, 'viewed_images': [], 'missing_optional_metadata': []}
    try:
        total_snapshot_bytes = 0
        for relative in SMALL:
            if not (ROOT / relative).is_file():
                result['missing_optional_metadata'].append(relative)
                continue
            item = record(relative)
            if item['bytes'] > 1024**2:
                raise ValueError('Oversized small evidence file: ' + relative)
            data = (ROOT / relative).read_bytes()
            if hashlib.sha256(data).hexdigest() != item['sha256']:
                raise ValueError('Changed input during snapshot: ' + relative)
            total_snapshot_bytes += len(data)
            if total_snapshot_bytes > 8 * 1024**2:
                raise ValueError('Snapshot byte bound exceeded')
            target = OUT / 'snapshots' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(data)
            result['inputs'].append(item)
        for label, (relative, expected) in MODELS.items():
            item = record(relative)
            if expected and item['sha256'] != expected:
                raise ValueError('Named model no longer matches preserved evidence: ' + label)
            result['models'][label] = {**item, **census(relative), 'expected_sha256_matched': True if expected else None}
        result['viewed_images'] = [record(path) for path in VIEWED]
        result['snapshot_bytes'] = total_snapshot_bytes
        result['status'] = 'recorded'
    except Exception as error:
        result['status'] = 'failed'
        result['failure'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic() - started
        result['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['memory_scope'] = 'macOS ru_maxrss for this read-only Python process, not whole tree'
        with (OUT / 'inventory.json').open('x') as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write('\n')


if __name__ == '__main__':
    main()
