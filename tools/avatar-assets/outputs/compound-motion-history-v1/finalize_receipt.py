"""Restore only reviewed public-document citations rejected by the broad URL preflight."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import signal
from zoneinfo import ZoneInfo


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
ALLOWED_DOCUMENT = 'https://github.com/blender/blender/blob/v4.5.3/source/blender/makesrna/intern/rna_particle.cc'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def main():
    signal.alarm(30)
    if digest(OUT/'manifest.json') != 'aaceecf0aa403f3ed8463d2ec9cda4c479d6221b84a1c393845f500d08381434':
        raise ValueError('Require the exact unfinalized first receipt')
    copies = json.loads((OUT/'copy-receipt.json').read_text())
    attempts = json.loads((OUT/'attempts.json').read_text())
    inventory = json.loads((OUT/'manifest.json').read_text())
    reviewed = []
    credential = re.compile(rb'(?i)(?:authorization|api[_-]?key|access[_-]?token|client[_-]?secret)\s*[:=]\s*[\"\x27]?(?:bearer\s+)?[A-Za-z0-9_./+\-]{8,}')
    for row in copies['omitted_sensitive_text']:
        source = ROOT/row['source_path']
        data = source.read_bytes()
        urls = re.findall(rb'https?://[^\s\"\x27<>]+', data)
        if (hashlib.sha256(data).hexdigest() != row['sha256'] or credential.search(data)
                or not urls or any(url.decode('ascii') != ALLOWED_DOCUMENT for url in urls)):
            raise ValueError('Restoration contains something other than the reviewed public citation')
        target = OUT/'retained'/row['source_path']
        if target.exists():
            raise ValueError('Refusing to overwrite retained source evidence')
        reviewed.append((row, data, target))
    originals = ['manifest.json', 'copy-receipt.json', 'attempts.json']
    current_bytes = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    if current_bytes+sum(len(data) for _, data, _ in reviewed)+sum((OUT/n).stat().st_size for n in originals) > 9.5*1024**2:
        raise ValueError('Finalization exceeds reserved total-size envelope')
    draft = OUT/'receipt-draft-v1'
    draft.mkdir()
    for name in originals:
        (draft/name).write_bytes((OUT/name).read_bytes())
    for row, data, target in reviewed:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
        if digest(target) != row['sha256']:
            raise ValueError('Reviewed source copy mismatch')
        copies['copies'].append({'path':str(target.relative_to(OUT)), 'source_path':row['source_path'],
            'bytes':len(data),'sha256':row['sha256'],'byte_identical_source_copy':True})
    copies['omitted_sensitive_text'] = []
    copies['reviewed_public_document_url'] = ALLOWED_DOCUMENT
    copies['privacy_filter_correction'] = 'Initial all-URL exclusion rejected these public Blender implementation citations; exact original bytes were reviewed and restored. No provider URL or credential was copied.'
    copies['initial_receipt'] = 'receipt-draft-v1/copy-receipt.json'
    retained = {row['source_path'] for row in copies['copies']}
    for attempt in attempts['attempts']:
        attempt['omitted_original_manifest_artifacts'] = [row for row in attempt['omitted_original_manifest_artifacts']
            if str(Path(attempt['original_directory'])/row['path']) not in retained]
        for path in attempt['evidence_paths']:
            if not (OUT/path).is_file():
                raise ValueError('Missing rejection evidence')
        for row in [attempt['original_manifest'], *attempt['original_configs'],
                    *([attempt['original_job']] if attempt['original_job'] else [])]:
            if digest(OUT/'retained'/row['path']) != row['sha256']:
                raise ValueError('Missing original manifest/job/config copy')
    write(OUT/'copy-receipt.json', copies)
    write(OUT/'attempts.json', attempts)
    (OUT/'receipt-draft-v1/README.md').write_text(
        '# Retained initial compacting draft\n\nThese exact receipts record an overbroad all-URL filter, not an additional native or generation attempt.\nThe root receipts supersede their omission lists after restoring15files containing only the reviewed public Blender source citation.\nNo original avatar pack or model was edited.\n')
    inventory['recorded_at'] = datetime.now(ZoneInfo('America/Los_Angeles')).isoformat()
    inventory['copied_artifacts'] = len(copies['copies'])
    inventory['copied_bytes'] = sum(row['bytes'] for row in copies['copies'])
    inventory['privacy_filter_review'] = {'restored_files':len(reviewed),'provider_urls_or_credentials_copied':False,
        'public_source_reference':ALLOWED_DOCUMENT,'initial_receipt_preserved':'receipt-draft-v1/manifest.json'}
    artifacts = []
    for path in sorted(OUT.rglob('*')):
        if path.is_file() and path != OUT/'manifest.json':
            artifacts.append({'path':str(path.relative_to(OUT)),'bytes':path.stat().st_size,'sha256':digest(path)})
    inventory['artifacts'] = artifacts
    inventory['retained_artifacts_excluding_manifest'] = len(artifacts)
    inventory['retained_bytes_excluding_manifest'] = sum(row['bytes'] for row in artifacts)
    write(OUT/'manifest.json', inventory)
    total = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    if total > 10*1024**2:
        raise ValueError('Final archive exceeds ten MiB')
    for row in copies['copies']:
        if digest(ROOT/row['source_path']) != row['sha256'] or digest(OUT/row['path']) != row['sha256']:
            raise ValueError('Source or copy changed during finalization')
    for row in artifacts:
        if digest(OUT/row['path']) != row['sha256'] or (OUT/row['path']).stat().st_size != row['bytes']:
            raise ValueError('Final inventory mismatch')
    print(json.dumps({'status':'verified_selected_evidence','files':len(artifacts)+1,'bytes':total,
        'copies':len(copies['copies']),'attempts':len(attempts['attempts']),
        'manifest_sha256':digest(OUT/'manifest.json'),'all_original_manifest_job_config_copies_verified':True},indent=2))


if __name__ == '__main__':
    main()
