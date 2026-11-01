"""Read-only audit of the real dragon; analytic fixtures remain separate."""

import json
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from avatar_assets.cli import verify_pack
from avatar_assets.media import image_info, video_info
from avatar_assets.meshy_access import Ledger
from avatar_assets.safety import read_json, require, sha256, write_json
from avatar_assets.three_d import validate_glb, validate_render_attributes


def audit(out):
    start = time.monotonic()
    model = Path('outputs/emerald-dragon-finished-v3')
    media = Path('outputs/emerald-dragon-media-v1')
    portable = Path(str(media)+'-portable-v1')
    review = Path('outputs/emerald-dragon-review-v1')
    packs = (model, media, portable, Path('outputs/emerald-dragon-reload-v1'))
    proofs = {str(p): verify_pack(p) for p in packs}
    for p in packs:
        canonical = json.dumps(read_json(p/'manifest.json'), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
        require((p/'manifest.json').read_text() == canonical, 'Noncanonical manifest')
    full, small = [read_json(p/'manifest.json') for p in (media, portable)]
    require(full['parent_3d_manifest_sha256'] == small['parent_3d_manifest_sha256'] == sha256(model/'manifest.json'), 'Wrong parent')
    require(sha256(model/'model.glb') == 'b87ee5f5fe0c3b9ab6b84d9a310104a75408ccc247d68c27fbb1d85685840d14', 'Model changed')
    raw = Path('outputs/emerald-dragon-meshy-v1/stages/texture/downloads/model.glb')
    require(sha256(raw) == 'a5e034586961d2999846d185321e6b1b4289e8138ce4b11a4a19746700109efb', 'Original provider model changed')
    corrected = validate_glb(model/'model.glb', read_json(review/'authored-rig-contract.json'))
    attributes = validate_render_attributes(model/'model.glb')
    portrait = image_info(media/'portrait.webp')
    require(portrait['width'] == portrait['height'] == 1600 and not portrait['has_transparency'], 'Wrong portrait')
    states = []
    require({c['id'] for c in full['clips']} == {'idle','look_around','wing_flap','milestone_level_up'}, 'Missing state')
    for clip in full['clips']:
        report = video_info(media/clip['path'])
        require(report == clip['inspection'], 'Independent decoding differs')
        require(report['width'] == report['height'] == 720 and report['fps'] == 24
                and report['frames'] == report['unique_thumbnails'] == 96
                and report['duration_seconds'] == 4 and report['codec'] == 'h264'
                and not report['audio'] and not report['has_transparency'], 'Wrong video format')
        require(sha256(media/clip['fallback']) == sha256(media/'frames'/clip['id']/'0000.png'), 'Fallback differs')
        for name in (clip['path'],clip['fallback'],'portrait.webp','portrait.png'):
            require(sha256(media/name) == sha256(portable/name), 'Portable pixels changed')
        require(len(list((media/'frames'/clip['id']).glob('*.png'))) == 97, 'Missing source or endpoint frame')
        states.append({'id':clip['id'],'inspection':report,'static_fallback_exact':True,
                       'loop_declaration':clip['loop_declaration'],'endpoint_mae':clip['exact_endpoint_mae']})
    archive_hash = sha256(Path('/Users/ivan/Downloads/Poo avatar assets.zip'))
    require(archive_hash == 'cb7eac375b5fcef0cd6c3f9457da7911f08f73337178c31fe33bfa090e6a54cb', 'Original archive changed')
    ledger = Ledger()
    try:
        cost = ledger.summary()
    finally:
        ledger.close()
    require(cost['paid_submissions'] == cost['settled_tasks'] == 15
            and cost['reserved_credits'] == cost['provider_reported_credits'] == 167
            and cost['unknown_submissions'] == cost['pricing_overruns'] == 0
            and cost['base_project_credit_limit'] == 140
            and cost['job_scoped_allowances'] == {'emerald-dragon-meshy-v1':35}, 'Unexpected credit state')
    result = {'schema_version':1,'status':'passed','actual_generated_asset':True,'analytic_fixture':False,
              'model_sha256':sha256(model/'model.glb'),'original_model_sha256':sha256(raw),
              'manifest_sha256':{str(p):sha256(p/'manifest.json') for p in packs},'pack_proofs':proofs,
              'corrected_rig_contract_proof':corrected,'attributes':attributes,'portrait':portrait,'states':states,
              'original_archive_sha256':archive_hash,'ledger':cost,'new_provider_calls':0,
              'canonical_manifests':True,'all_source_frames_preserved':True,
              'quality_equivalence_verified':False,'physical_motion_verified':False,
              'reproducibility':'Preserved bytes, recipes and manifests, not stochastic paid regeneration',
              'elapsed_seconds':round(time.monotonic()-start,3),
              'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_json(out/'verification.json',result)
    print(json.dumps({k:result[k] for k in ('status','elapsed_seconds','process_peak_rss_bytes','new_provider_calls')}))


if __name__ == '__main__':
    audit(Path(sys.argv[1]))
