import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import unittest

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from avatar_assets.safety import read_json, sha256, write_json

start = time.monotonic()
suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_material_native_reuse')
result = unittest.TextTestRunner(verbosity=2).run(suite)
actual = ROOT/'outputs/fox-material-fur-finish-v1'
groom = read_json(actual/'material-groom/report.json')
coat = read_json(actual/'material-coat.json')
finished = read_json(actual/'finish-report.json')
checks = {
    'groom_equals_embedded': groom == finished.get('groom'),
    'coat_equals_embedded': coat == finished.get('material_coat'),
    'coat_policy_matches': isinstance(coat, dict) and coat.get('policy') == 'material-owned-tufts-v1',
    'groom_status_matches': groom.get('status') == 'native_grooms_pending_render_review',
    'original_body_unchanged_claim_present': groom.get('original_body_unchanged') is True,
    'correspondence_path_matches': groom.get('correspondence', {}).get('path') == 'material-groom/correspondence.npz',
    'correspondence_hash_matches': sha256(actual/'material-groom/correspondence.npz') == groom.get('correspondence', {}).get('sha256'),
}
write_json(OUT/'actual-receipt-comparison.json', {
    'source': str(actual), 'checks': checks, 'passed': all(checks.values()),
    'scope': 'One read-only receipt consistency check; no native reuse, render, skin or appearance revalidation.',
    'inputs': [{'path': str((actual/name).relative_to(ROOT)), 'sha256': sha256(actual/name)}
               for name in ('material-groom/report.json', 'material-groom/correspondence.npz',
                            'material-coat.json', 'finish-report.json')],
})
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
receipt = {'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
    'skipped': len(result.skipped), 'passed': result.wasSuccessful() and all(checks.values()),
    'test_failures': [{'name': str(test), 'traceback': detail[-6000:]} for test, detail in result.failures+result.errors],
    'actual_read_only_receipt_checks': 1, 'actual_receipts_passed': all(checks.values()),
    'elapsed_seconds': time.monotonic()-start,
    'self_peak_rss_bytes': int(peak if sys.platform == 'darwin' else peak*1024),
    'rss_scope': 'Direct-child getrusage high-water RSS, separate from monitor samples',
    'native_jobs': 0, 'provider_calls': 0, 'ai_role_calls': 0}
write_json(OUT/'test-result.json', receipt)
raise SystemExit(0 if receipt['passed'] else 1)
