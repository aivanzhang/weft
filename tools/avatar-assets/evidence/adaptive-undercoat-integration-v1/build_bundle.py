"""Copy an exact, pinned allowlist only; never execute retained scripts or models."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import time
from datetime import datetime
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
MAX_BYTES = 20*1024*1024
MAX_FILE_BYTES = 8*1024*1024
MAX_REFERENCE_BYTES = 512*1024*1024
MAX_ENTRIES = 512
PREPARATION_FILES = ('build_bundle.py', 'allowlist.json', 'preparation.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative(value):
    require(isinstance(value, str), 'Path must be a string')
    path = PurePosixPath(value)
    require(not path.is_absolute() and path.parts and all(p not in ('', '.', '..') for p in path.parts),
            'Path must be canonical and relative')
    require(str(path) == value and '\\' not in value and not any(ord(c) < 32 for c in value), 'Unsafe path')
    require(not any(p.startswith('.') or 'credential' in p.lower() for p in path.parts), 'Private path denied')
    return path


def source(value):
    path = relative(value)
    require(path.parts[0] in ('outputs', 'tests', 'scripts', 'avatar_assets', 'docs', 'evidence'), 'Origin outside allowlisted trees')
    result = ROOT.joinpath(*path.parts)
    current = ROOT
    for part in path.parts:
        current /= part
        require(not current.is_symlink(), 'Symlink origin denied')
    require(result.is_file(), 'Missing origin: '+value)
    return result


def identity(path):
    info = path.stat()
    require(stat.S_ISREG(info.st_mode), 'Only regular files are supported')
    return {'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns, 'inode': info.st_ino, 'device': info.st_dev}


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write_new(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    with path.open('xb') as handle:
        handle.write(data)


def inspect():
    require((HERE/'allowlist.json').stat().st_size <= 1024*1024, 'Oversized allowlist')
    selection = json.loads((HERE/'allowlist.json').read_text())
    entries = selection['retained']
    references = selection['omitted_references']
    require(0 < len(entries)+len(references) <= MAX_ENTRIES, 'Entry bound exceeded')
    destinations, origins = {}, set()
    total = sum((HERE/name).stat().st_size for name in PREPARATION_FILES)
    for row in entries+references:
        origin = source(row['origin'])
        actual = identity(origin)
        require(actual == row['source_identity'] and actual['bytes'] == row['bytes'], 'Pinned origin identity changed: '+row['origin'])
        require(len(row['sha256']) == 64 and all(c in '0123456789abcdef' for c in row['sha256']), 'Invalid hash')
        require(row['origin'] not in origins, 'Duplicate origin')
        origins.add(row['origin'])
        require(row['classification'] in ('analytic_fixture', 'test_receipt', 'prepared_actual_subject',
                'actual_generated_source_local_finish', 'production_source', 'independent_review'), 'Missing provenance class')
        if row in entries:
            destination = relative(row['path'])
            require(destination.parts[0] == 'retained', 'Invalid destination')
            saved = str(destination)
            fingerprint = (row['bytes'], row['sha256'])
            require(saved not in destinations or destinations[saved] == fingerprint, 'Conflicting deduplicated destination')
            require(0 <= row['bytes'] <= MAX_FILE_BYTES, 'Retained origin exceeds8MiB')
            if saved not in destinations:
                total += row['bytes']
                destinations[saved] = fingerprint
        else:
            require(0 <= row['bytes'] <= MAX_REFERENCE_BYTES, 'Omitted reference exceeds512MiB')
    require(total+1024*1024 <= MAX_BYTES, 'Insufficient20MiB envelope for bundle metadata')
    return selection, {'status': 'prepared_not_sealed', 'retained_origins': len(entries),
        'unique_retained_files': len(destinations),
        'omitted_references': len(references), 'selected_bytes_plus_preparation': total,
        'reserved_metadata_bytes': 1024*1024, 'limit_bytes': MAX_BYTES,
        'source_check': 'SHA256 pinned at preparation; current size/mtime/inode/device unchanged',
        'pending_groups': selection['pending_groups'], 'native_calls': 0, 'provider_calls': 0}


def inventory():
    rows, stack, visited = [], [HERE], 0
    while stack:
        directory = stack.pop()
        for path in directory.iterdir():
            visited += 1
            require(visited <= MAX_ENTRIES*2, 'Bundle inventory entry bound exceeded')
            require(not path.is_symlink(), 'Bundle symlink denied')
            if path.is_dir():
                stack.append(path)
            elif path.name != 'manifest.json' or path.parent != HERE:
                require(path.is_file(), 'Unexpected nonfile bundle entry')
                rows.append({'path': path.relative_to(HERE).as_posix(), 'bytes': path.stat().st_size,
                             'sha256': digest(path)})
    return sorted(rows, key=lambda row: row['path'])


def seal(status_path):
    require(not (HERE/'manifest.json').exists(), 'Never overwrite a sealed bundle')
    selection, preview = inspect()
    require(not selection['pending_groups'], 'Final evidence groups are still pending')
    status_origin = source(status_path)
    require(status_origin.stat().st_size <= 256*1024, 'Oversized final status')
    status = json.loads(status_origin.read_text())
    require(status.get('bundle_ready') is True and isinstance(status.get('final_actual_build_status'), str),
            'Root terminal actual-build status is required before sealing')
    require(status.get('native_or_paid_jobs_active') is False, 'Active job status prevents sealing')
    checked_targets = set()
    for row in selection['retained']:
        origin, target = source(row['origin']), HERE/row['path']
        if row['path'] in checked_targets:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            require(target.is_file() and not target.is_symlink() and target.stat().st_size == row['bytes']
                    and digest(target) == row['sha256'], 'Partial bundle file differs; do not overwrite')
            checked_targets.add(row['path'])
            continue
        hashed, copied = hashlib.sha256(), 0
        with origin.open('rb') as reader, target.open('xb') as writer:
            for chunk in iter(lambda: reader.read(1024*1024), b''):
                copied += len(chunk)
                require(copied <= row['bytes'], 'Origin grew while copying')
                hashed.update(chunk)
                writer.write(chunk)
        require(copied == row['bytes'] and hashed.hexdigest() == row['sha256']
                and identity(origin) == row['source_identity'], 'Origin changed while copying')
        checked_targets.add(row['path'])
    write_new(HERE/'integration-status.json', status)
    write_new(HERE/'omitted-originals.json', {'scope': 'Hash-bound references, not copied or decoded here',
        'references': selection['omitted_references'], 'complete_recreation': False})
    readme = ('# Adaptive undercoat integration evidence\n\n'
        'This bounded selection preserves exact failed fixtures, tests, configuration and terminal evidence. '
        'It is not a complete avatar pack or standalone recreation. Original manifests describe full original outputs, '
        'not this smaller bundle. See `integration-status.json` for the final actual-build result.\n\n'
        'Analytic fixtures are distinct from an actual generated source and its local finish. Short-pile normal detail '
        'is an optional surface component, not dense fur, recovered hairs, Muse parity, physics or Weft compatibility. '
        'No automatic quality promotion is implied.\n\n'
        '`allowlist.json` maps each retained origin to its exact saved bytes and records omitted binaries. '
        'Byte-identical origins may share one saved path; each origin keeps its own hash and provenance class. '
        'Do not execute retained scripts against sealed folders; new work requires a new bounded output directory.\n\n'
        'Verify locally from tools/avatar-assets: `python3 evidence/adaptive-undercoat-integration-v1/build_bundle.py verify`.\n')
    with (HERE/'README.md').open('x') as handle:
        handle.write(readme)
    rows = inventory()
    manifest = {'schema_version': 1, 'status': 'sealed_evidence_not_avatar_pack',
        'recorded_local_time': datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M:%S %Z'),
        'artifacts': rows, 'retained_origins': selection['retained'],
        'omitted_originals': selection['omitted_references'], 'complete_recreation': False,
        'final_status_origin': status_path, 'final_status_origin_sha256': digest(status_origin),
        'native_calls_by_bundler': 0, 'provider_calls_by_bundler': 0, 'limit_bytes': MAX_BYTES}
    encoded = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    require(sum(row['bytes'] for row in rows)+len(encoded) <= MAX_BYTES, 'Final bundle exceeds20MiB')
    with (HERE/'manifest.json').open('xb') as handle:
        handle.write(encoded)
    return verify()


def verify():
    path = HERE/'manifest.json'
    manifest = json.loads(path.read_text())
    rows = inventory()
    require(rows == manifest['artifacts'], 'Bundle inventory/hash mismatch')
    size = sum(row['bytes'] for row in rows)+path.stat().st_size
    require(size <= MAX_BYTES, 'Bundle exceeds20MiB')
    return {'status': 'verified', 'files': len(rows)+1, 'bytes': size,
            'manifest_sha256': digest(path), 'origin_recheck': False,
            'scope': 'Retained byte integrity only; no new rendering, tests, model or quality claim'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('inspect', 'seal', 'verify'))
    parser.add_argument('--status-file')
    args = parser.parse_args()
    began = time.monotonic()
    result = inspect()[1] if args.mode == 'inspect' else verify() if args.mode == 'verify' else seal(args.status_file)
    result['elapsed_seconds'] = round(time.monotonic()-began, 6)
    print(json.dumps(result, sort_keys=True))
