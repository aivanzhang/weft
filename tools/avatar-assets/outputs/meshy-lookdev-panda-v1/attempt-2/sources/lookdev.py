"""Render bounded local surface/fur comparisons from a verified generated pack."""

import json
from pathlib import Path, PurePosixPath
import re
import shutil
import struct

from PIL import Image

from avatar_assets.runner import run_bounded
from avatar_assets.safety import artifacts, new_directory, preserve_failure, read_json, require, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
MAX_INPUT = 512 * 1024 * 1024
MAX_OUTPUT = 256 * 1024 * 1024
MAX_BLEND_BLOCKS = 500000
VARIANTS = ('baseline_native', 'base_only', 'portable_only', 'base_matte_no_normal', 'native_parents_only')


def _regular(root, relative):
    require(isinstance(relative, str) and '\\' not in relative, 'Invalid pack artifact path')
    name = PurePosixPath(relative)
    require(not name.is_absolute() and name.parts and str(name) == relative
            and all(part not in ('.', '..') for part in name.parts), 'Unsafe pack artifact path')
    path = root
    for part in name.parts:
        path = path / part
        require(not path.is_symlink(), 'Pack artifact paths cannot contain symlinks')
    require(path.is_file(), 'Missing regular pack artifact: ' + relative)
    return path


def blend_header(path):
    """Reject linked libraries before Blender can follow their external paths."""
    size = path.stat().st_size
    require(12 <= size <= MAX_INPUT, 'Blend input exceeds the 512 MiB bound')
    with path.open('rb') as stream:
        header = stream.read(12)
        require(header[:7] == b'BLENDER' and header[7:8] in (b'-', b'_')
                and header[8:9] in (b'v', b'V') and header[9:].isdigit(),
                'Require an uncompressed native .blend; compressed or foreign files are unsupported')
        layout = struct.Struct(('<' if header[8:9] == b'v' else '>')
                               + ('4siQii' if header[7:8] == b'-' else '4siIii'))
        counts, blocks = {}, 0
        while stream.tell() < size:
            raw = stream.read(layout.size)
            require(len(raw) == layout.size, 'Truncated blend block header')
            code, length, _old, _dna, count = layout.unpack(raw)
            blocks += 1
            require(blocks <= MAX_BLEND_BLOCKS and 0 <= length <= size-stream.tell() and count >= 0,
                    'Invalid or oversized blend block')
            label = code.rstrip(b'\0').decode('ascii', errors='replace')
            counts[label] = counts.get(label, 0)+1
            require(label != 'LI', 'Linked blend libraries are unsupported; no external files may be loaded')
            stream.seek(length, 1)
            if code == b'ENDB':
                require(length == 0 and stream.tell() == size, 'Invalid blend terminator')
                break
        require(counts.get('ENDB') == 1 and counts.get('DNA1') == 1, 'Incomplete native blend structure')
    return {'version_header': header[9:].decode(), 'blocks': blocks,
            'linked_libraries': 0, 'text_blocks_not_executed': counts.get('TX', 0),
            'scope': 'Block-envelope inspection, not a complete Blender-format security verifier'}


def preflight(pack):
    pack = Path(pack).absolute()
    require(pack.is_dir() and not any(p.is_symlink() for p in (pack, *pack.parents)),
            'Require a non-symlink local pack directory')
    manifest_path = _regular(pack, 'manifest.json')
    manifest = read_json(manifest_path)
    require(manifest.get('schema_version') == 1 and manifest.get('kind') == 'generated_3d_avatar'
            and manifest.get('status') != 'failed', 'Require a generated_3d_avatar pack manifest')
    entries = manifest.get('artifacts')
    require(isinstance(entries, list) and 1 <= len(entries) <= 4096, 'Invalid pack artifact count')
    seen, declared = set(), manifest_path.stat().st_size
    for entry in entries:
        require(isinstance(entry, dict) and set(entry) == {'path', 'bytes', 'sha256'}, 'Invalid pack artifact record')
        name = entry['path']
        require(isinstance(name, str) and name not in seen and name != 'manifest.json', 'Duplicate pack artifact')
        seen.add(name)
        require(type(entry['bytes']) is int and 0 <= entry['bytes'] <= MAX_INPUT,
                'Invalid pack artifact byte count')
        declared += entry['bytes']
        require(declared <= MAX_INPUT, 'Pack input exceeds the aggregate 512 MiB bound')
        require(isinstance(entry['sha256'], str) and re.fullmatch('[0-9a-f]{64}', entry['sha256']),
                'Invalid pack artifact hash')
    require('source.blend' in seen, 'Pack has no inventoried source.blend')
    for entry in entries:
        path = _regular(pack, entry['path'])
        require(path.stat().st_size == entry['bytes'] and sha256(path) == entry['sha256'],
                'Pack artifact differs from manifest: ' + entry['path'])
    blend = pack/'source.blend'
    return {'pack': str(pack), 'manifest_sha256': sha256(manifest_path), 'source_blend_sha256': sha256(blend),
            'verified_input_bytes': declared, 'verified_artifact_count': len(entries),
            'artifacts': entries, 'blend_header': blend_header(blend)}, manifest


def compare(pack, output):
    receipt, manifest = preflight(pack)
    pack = Path(receipt['pack'])
    target = Path(output).absolute()
    require(not target.resolve().is_relative_to(pack.resolve()), 'Lookdev output must not modify the input pack')
    out = new_directory(output)
    try:
        (out/'sources').mkdir()
        for source in (Path(__file__), ROOT/'scripts/blender_lookdev.py'):
            shutil.copyfile(source, out/'sources'/source.name)
        shutil.copyfile(pack/'manifest.json', out/'source-manifest.json')
        write_json(out/'input-receipt.json', receipt)
        finish = manifest.get('finish', {})
        require(isinstance(finish, dict), 'Invalid pack finishing report')
        portable = [row['material']['material'] for row in finish.get('portable_fur', [])]
        config = {'schema_version': 1, 'source_blend': str(pack/'source.blend'),
                  'source_blend_sha256': receipt['source_blend_sha256'], 'out': str(out),
                  'portable_material_names': portable, 'resolution': 640, 'samples': 16, 'threads': 4,
                  'variants': list(VARIANTS), 'pose': 'saved', 'camera_lights': 'saved unchanged',
                  'denoised': True, 'new_provider_calls': 0,
                  'limits': {'wall_seconds': 300, 'rss_mb': 2048, 'output_mb': 256}}
        write_json(out/'config.json', config)
        exe = ROOT/'.tools/Blender.app/Contents/MacOS/Blender'
        require(exe.is_file(), 'Local Blender is missing')
        command = [str(exe), '--background', '--factory-startup', '--disable-autoexec', '--offline-mode',
                   '--python-exit-code', '1', '--threads', '4', '--python', str(out/'sources/blender_lookdev.py'),
                   '--', str(out/'config.json')]
        job = run_bounded(command, out/'lookdev.log', out, timeout=300, memory_mb=2048, output_mb=256)
        write_json(out/'job.json', job)
        require(job['exit_code'] == 0 and not job['failure'], 'Lookdev failed; inspect retained job evidence')
        native = read_json(out/'native-report.json')
        require(native.get('source_blend_sha256') == receipt['source_blend_sha256']
                and native.get('source_unchanged') is True, 'Native source hash verification failed')
        require([row['id'] for row in native['variants']] == list(VARIANTS), 'Incomplete lookdev variants')
        for variant in VARIANTS:
            with Image.open(out/(variant+'.png')) as image:
                require(image.format == 'PNG' and image.size == (640, 640) and image.mode in ('RGB', 'RGBA')
                        and getattr(image, 'n_frames', 1) == 1, 'Invalid lookdev render')
                image.verify()
        after, _ = preflight(pack)
        require(after == receipt, 'Original pack changed during comparison')
        output_artifacts = artifacts(out)
        require(sum(row['bytes'] for row in output_artifacts) <= MAX_OUTPUT, 'Lookdev output exceeds 256 MiB')
        result = {'schema_version': 1, 'kind': 'generated_model_lookdev', 'status': 'needs_visual_review',
                  'source': receipt, 'original_pack_unchanged': True, 'native': native, 'job': job,
                  'provider_calls': 0, 'new_provider_cost_usd': 0,
                  'scope': 'Diagnostic shading/visibility comparisons, not a repaired or improved avatar',
                  'artifacts': output_artifacts}
        write_json(out/'manifest.json', result)
        return {'status': result['status'], 'out': str(out), 'variants': list(VARIANTS), 'original_pack_unchanged': True}
    except Exception as error:
        preserve_failure(out, error)
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pack')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.pack, args.out), indent=2))
