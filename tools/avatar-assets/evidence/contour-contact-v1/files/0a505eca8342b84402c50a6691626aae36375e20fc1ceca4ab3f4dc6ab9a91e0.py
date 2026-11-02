"""Actual-source native rejection check; stop before any downstream finishing."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'sources'))
from scripts import blender_finish_generated as native
from avatar_assets.normal_continuity import NormalContinuityError

args = sys.argv[sys.argv.index('--')+1:]
if len(args) != 1:
    raise ValueError('Require one fresh output directory')
out = Path(args[0])
cfg = json.loads((HERE/'finish-config.json').read_text())
reached = []

def forbid_downstream(*args, **kwargs):
    reached.append('basecolor_decode')
    raise AssertionError('Guard failed to stop before downstream finishing')

native.shared_color_texture = forbid_downstream
try:
    native.finish(str(HERE/'finish-config.json'), str(out), prepare_only=True, preserve_skin_source=True)
except NormalContinuityError as error:
    report = json.loads((out/'normal-continuity-rejection.json').read_text())
    assert report == error.report and report['status'] == 'rejected'
    assert report['outside_hemisphere_corners'] > 0 and not reached
    assert not (out/'finish-report.json').exists() and not (out/'model.glb').exists()
    assert native.hash_file(Path(cfg['source'])) == cfg['source_sha256']
    native.write_json(out/'verification.json', {
        'schema_version': 1, 'status': 'actual_native_rejection_proved',
        'receipt_sha256': native.hash_file(out/'normal-continuity-rejection.json'),
        'outside_hemisphere_corners': report['outside_hemisphere_corners'],
        'checked_corners': report['checked_corners'], 'checked_faces': report['checked_faces'],
        'source_unchanged': True, 'downstream_sentinel_reached': False,
        'new_generation_calls': 0, 'rendered_images': 0, 'quality_approved': False,
        'scope': 'Real Blender import/rig setup/transfer rejection and receipt writer; not a completed model or appearance repair.'})
else:
    raise AssertionError('Expected normal transfer rejection')
