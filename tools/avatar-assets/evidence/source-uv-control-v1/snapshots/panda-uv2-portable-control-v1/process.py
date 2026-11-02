"""Bounded derivative chain for the actual experimental UV2 control."""
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(OUT/'sources'))
from avatar_assets.assemble_generated import rig_contract
from avatar_assets.finish_generated import _undercoat_tangent_targets
from avatar_assets.fringe_export import omit_unused_fringe_tangents
from avatar_assets.generated_tangents import repair
from avatar_assets.safety import require, sha256, write_json
from avatar_assets.source_skin_fidelity import restore_source_skin
from avatar_assets.three_d import _GLB, validate_glb, validate_render_attributes

prepared = json.loads((OUT/'prepared.json').read_text())
for row in prepared['inputs']:
    require(sha256(ROOT/row['path']) == row['sha256'], 'Frozen native input changed')
for row in prepared['source_pins']:
    require(sha256(OUT/'sources'/row['path']) == row['sha256'], 'Frozen code changed')
native = ROOT/'outputs/panda-uv2-staged-control-v1'
try:
    report = json.loads((native/'prepared-finish-report.json').read_text())
    expected = {'source': sha256(native/'model.glb'), 'snapshot': sha256(native/'source-skin.npz'),
                'snapshot_report': sha256(native/'source-skin.json')}
    restored = restore_source_skin(native/'model.glb', OUT/'model-skin-restored.glb',
        native/'source-skin.npz', native/'source-skin.json', experimental=True, expected_hashes=expected)
    write_json(OUT/'skin-restoration.json', restored)
    omitted = omit_unused_fringe_tangents(OUT/'model-skin-restored.glb', OUT/'model-fringe-export.glb',
        [report['portable_fringe']['material']['material']])
    write_json(OUT/'unused-fringe-tangent-omission.json', omitted)
    targets = _undercoat_tangent_targets(report, True)
    corrected = repair(OUT/'model-fringe-export.glb', OUT/'model.glb', authored_normal_targets=targets)
    write_json(OUT/'authored-tangent-correction.json', corrected)
    glb = _GLB(OUT/'model.glb')
    contract = rig_contract(glb.doc, authored=True)
    write_json(OUT/'rig-contract.json', contract)
    rig = validate_glb(OUT/'model.glb', contract)
    write_json(OUT/'rig-verification.json', rig)
    surface = validate_render_attributes(OUT/'model.glb')
    write_json(OUT/'render-attributes.json', surface)
    require(rig['motion']['animation_verified'], 'Control must retain actual weighted skin motion')
    require(all(row['normal_map_uv_set'] != 1 for row in surface['primitives']),
            'A used original normal UV1 binding survived')
    write_json(OUT/'result.json', {'schema_version': 1, 'status': 'needs_visual_review',
        'source_sha256': expected['source'], 'model_sha256': sha256(OUT/'model.glb'),
        'vertices': surface['vertices'], 'authored_uv2_materials': len(targets),
        'actual_weighted_motion_verified': True, 'rendered_images': 0,
        'source_normals_not_transferred': True, 'original_uv1_normal_faces_intentionally_rebaked': True,
        'general_source_uv1_frame_preservation': False, 'experimental_public_fringe_enabled': False,
        'quality_accepted': False, 'fresh_text_generation': False, 'new_provider_calls': 0})
except Exception as error:
    write_json(OUT/'failure.json', {'status': 'rejected', 'type': type(error).__name__, 'error': str(error)})
    raise
