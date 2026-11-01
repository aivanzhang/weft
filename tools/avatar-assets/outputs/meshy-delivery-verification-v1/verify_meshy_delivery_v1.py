"""Read-only audit of the actual generated examples; fixtures are tested separately."""

import json
from pathlib import Path
import resource
import sys
import time

from avatar_assets.cli import verify_pack
from avatar_assets.media import image_info, video_info
from avatar_assets.meshy_access import Ledger
from avatar_assets.safety import read_json, require, sha256, write_json


def audit(out):
    start = time.monotonic()
    cases = [
        ('muse-panda-meshy-refined-v6', 'muse-panda-meshy-media-v2'),
        ('workshop-robot-meshy-refined-v2', 'workshop-robot-meshy-media-v1'),
    ]
    reports = []
    for model_name, media_name in cases:
        model, media = Path('outputs')/model_name, Path('outputs')/media_name
        portable = Path(str(media)+'-portable-v1')
        manifests = [read_json(p/'manifest.json') for p in (model, media, portable)]
        proofs = [verify_pack(p) for p in (model, media, portable)]
        for p, manifest in zip((model, media, portable), manifests):
            canonical = json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
            require((p/'manifest.json').read_text() == canonical, 'Manifest serialization changed')
        require(manifests[1]['parent_3d_manifest_sha256'] == sha256(model/'manifest.json'), 'Wrong parent')
        portrait = image_info(media/'portrait.webp')
        require(portrait['width'] == portrait['height'] == 1600 and not portrait['has_transparency'], 'Wrong portrait')
        states = []
        for clip in manifests[1]['clips']:
            report = video_info(media/clip['path'])
            require(report == clip['inspection'], 'Independent video decode differs')
            require(report['width'] == report['height'] == 720 and report['fps'] == 24
                    and report['frames'] == report['unique_thumbnails'] == 96
                    and report['duration_seconds'] == 4 and report['codec'] == 'h264'
                    and not report['audio'] and not report['has_transparency'], 'Wrong clip format')
            require(sha256(media/clip['fallback']) == sha256(media/'frames'/clip['id']/'0000.png'), 'Fallback differs')
            for name in (clip['path'], clip['fallback'], 'portrait.webp'):
                require(sha256(media/name) == sha256(portable/name), 'Portable bytes differ')
            states.append({'id': clip['id'], 'inspection': report, 'static_fallback_exact': True,
                           'loop_declaration': clip['loop_declaration'], 'endpoint_mae': clip['exact_endpoint_mae']})
        reports.append({'model': str(model), 'media': str(media), 'portable': str(portable),
                        'model_sha256': sha256(model/'model.glb'),
                        'manifest_sha256': [sha256(p/'manifest.json') for p in (model, media, portable)],
                        'proofs': proofs, 'portrait': portrait, 'states': states})
    current = Path('outputs/muse-panda-meshy-media-v2')
    reused = read_json(current/'reused-frames.json')
    prior = Path(reused['previous'])
    require(sha256(prior/'manifest.json') == reused['previous_manifest_sha256'], 'Failed attempt changed')
    require(len(reused['paths']) == 366, 'Unexpected continuation inventory')
    for name in reused['paths']:
        target = 'portrait-raw.png' if name == 'portrait.png' else name
        require(sha256(prior/name) == sha256(current/target), 'Reused rendered bytes changed')
    archive = Path('/Users/ivan/Downloads/Poo avatar assets.zip')
    archive_hash = sha256(archive)
    require(archive_hash == 'cb7eac375b5fcef0cd6c3f9457da7911f08f73337178c31fe33bfa090e6a54cb', 'Original ZIP changed')
    ledger = Ledger()
    try:
        cost = ledger.summary()
    finally:
        ledger.close()
    require(cost['paid_submissions'] == cost['settled_tasks'] == 13
            and cost['provider_reported_credits'] == cost['reserved_credits'] == 132
            and cost['unknown_submissions'] == cost['pricing_overruns'] == 0, 'Unexpected credit state')
    result = {'schema_version': 1, 'status': 'structural_and_media_checks_passed',
              'quality_target_met': False, 'semantic_acceptance': 'Crafting/working remain unaccepted; see visual review',
              'actual_generated_assets': True, 'analytic_fixtures': False, 'examples': reports,
              'original_archive_sha256': archive_hash, 'ledger': cost, 'new_provider_calls': 0,
              'reused_pngs_byte_equal': 366, 'canonical_manifest_serialization': True,
              'reproducibility_scope': 'Existing artifact bytes/configuration, not stochastic generation or byte-identical Blender rebuilds',
              'elapsed_seconds': round(time.monotonic()-start, 3),
              'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_json(out/'verification.json', result)
    print(json.dumps({k:result[k] for k in ('status','quality_target_met','elapsed_seconds','process_peak_rss_bytes','new_provider_calls')}))


if __name__ == '__main__':
    audit(Path(sys.argv[1]))
