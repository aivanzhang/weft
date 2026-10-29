"""Check retained Meta proposals against unchanged source/model evidence; no API calls."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from weft.benchmark import seal_suite
from weft.human import HumanEnv
from weft.io import local_time, write_json
from weft.pipeline import verify_artifacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    if (out / 'report.json').exists() or (out / 'manifest.json').exists():
        raise FileExistsError('Choose a new unsealed diagnostic directory')
    out.mkdir(parents=True, exist_ok=True)
    source = Path('artifacts/mobile-exercise-e2e-10/squat/run')
    live = Path('artifacts/meta-body-contact-live-02')
    sealed = Path('artifacts/whole-body-feasibility-01')
    verification = verify_artifacts(sealed)
    def load(path):
        return json.loads(path.read_text())
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    paths = [live / name for name in ['hypotheses.json', 'context.json', 'advisor-call.json']]
    paths += [sealed / 'manifest.json', sealed / 'squat/feasibility.json', sealed / 'squat/human.json']
    paths += [source / name for name in ['source.webm', 'root-motion.json', 'pose-landmarks.json']]
    paths += [Path(__file__), Path('src/weft/human.py'), Path('src/weft/mobile_human.py')]
    hashes = {str(path): digest(path) for path in paths}
    for name in ['human.py', 'mobile_human.py']:
        assert digest(Path('src/weft') / name) == digest(sealed / 'implementation' / name)
    hypotheses = load(live / 'hypotheses.json')
    assert hypotheses['accepted'] is False and hypotheses['source'] == 'model_proposal'
    spec = load(sealed / 'squat/human.json')
    floor = load(sealed / 'squat/feasibility.json')
    root = load(source / 'root-motion.json')
    poses = {row['frame']: row for row in load(source / 'pose-landmarks.json')['frames']}
    observations = {row['frame']: row for row in root['observations']}
    assert digest(source / 'source.webm') == spec['source_sha256'] == root['source_sha256']
    indices = [0, 72, 110, 174, 212]
    rows = []
    def forbidden(*unused, **kwargs):
        raise AssertionError('No dynamics allowed in construction-only hypothesis checks')
    mujoco.mj_step = forbidden
    env = HumanEnv(spec)
    try:
        data = mujoco.MjData(env.model)
        for frame in indices:
            pose, image_root = poses[frame], observations[frame]
            time_s = pose['time_s']
            index = next(i for i, row in enumerate(spec['reference']) if row['time_s'] == time_s)
            saved = next(row for row in floor['checks'] if row['kind'] == 'source_pose'
                         and row['source_reference_index'] == index)
            data.qpos[env._q_array] = [spec['reference'][index]['joints'][name] for name in env.joint_names]
            data.qpos[env.mobility.q] = spec['root_reference'][index]['translation']
            mujoco.mj_forward(env.model, data)
            surfaces = []
            for geom in range(env.model.ngeom):
                if not int(env.model.geom_contype[geom]) & 2:
                    continue
                kind = int(env.model.geom_type[geom])
                lower = float(data.geom_xpos[geom, 2] - env.model.geom_size[geom, 0])
                if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                    lower -= abs(float(data.geom_xmat[geom].reshape(3, 3)[2, 2])) * env.model.geom_size[geom, 1]
                elif kind != int(mujoco.mjtGeom.mjGEOM_SPHERE):
                    raise ValueError('Unexpected collision shape')
                surfaces.append({'geom': mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom),
                                 'lowest_z': lower})
            lowest = min(surfaces, key=lambda row: row['lowest_z'])
            assert abs(max(0., -lowest['lowest_z']) - saved['max_penetration_model_units']) < 1e-10
            derived_z = -(image_root['pelvis_image_px'][1] - root['origin']['pelvis_image_px'][1]) * root['scale']['model_units_per_pixel']
            ankles = {side: {**pose['landmarks'][i],
                             'minimum_visibility_presence': min(pose['landmarks'][i]['visibility'], pose['landmarks'][i]['presence'])}
                      for side, i in [('left', 27), ('right', 28)]}
            rows.append({'frame': frame, 'time_s': time_s, 'evidence_id': pose['id'],
                         'pose_status': pose['status'], 'image_pelvis_px': image_root['pelvis_image_px'],
                         'root_translation_xyz': spec['root_reference'][index]['translation'],
                         'root_z_recomputed_from_image': derived_z,
                         'root_z_formula_absolute_error': abs(derived_z - data.qpos[env.mobility.q[2]]),
                         'joint_angles_rad': spec['reference'][index]['joints'], 'ankle_estimates': ankles,
                         'lowest_collision_surface': lowest, 'all_collision_surfaces': surfaces,
                         'sealed_floor_check': saved,
                         'visual_audit': {'method': 'Human-readable inspection of exact source PNG by assistant, not new tracking',
                                          'barbell_visible': True, 'complete_left_sole_contact_visible': False,
                                          'complete_right_sole_contact_visible': False,
                                          'reason': 'Feet/soles extend beyond lower image boundary; rack also obscures lower-body regions',
                                          'foot_support_state': 'unidentified', 'barbell_mass': 'unidentified'}})
    finally:
        env.close()
    by_frame = {row['frame']: row for row in rows}
    floor_support = by_frame[0]['lowest_collision_surface']['lowest_z'] > 0 and all(
        by_frame[f]['lowest_collision_surface']['lowest_z'] < -.025 for f in [72, 174])
    projection = [by_frame[f] for f in [0, 72, 110, 212]]
    ankle_min = min(a['minimum_visibility_presence'] for row in rows for a in row['ankle_estimates'].values())
    outcomes = [
        {'id': 'h_floor_anchor', 'outcome': 'supported_within_model' if floor_support else 'contradicted',
         'finding': 'First-pose alignment and later capsule-floor conflicts reproduced without changing geometry or poses',
         'physical_cause': 'unidentified; this does not isolate floor offset, scale, camera, pose, or support error'},
        {'id': 'h_pelvis_projection', 'outcome': 'unidentified',
         'coordinate_claim': {'outcome': 'contradicted', 'requested': 'inferred root y as vertical',
                              'actual': 'root_z is vertical; root_y is unobserved depth and stays zero'},
         'construction_mapping': {'outcome': 'supported_within_model',
                                  'maximum_formula_error': max(row['root_z_formula_absolute_error'] for row in projection)},
         'finding': 'Image y determines root_z algebraically under the authored scale; this is not independent evidence of camera motion or real vertical motion'},
        {'id': 'h_support_load_unknown', 'outcome': 'unidentified',
         'confidence_claim': {'outcome': 'contradicted', 'minimum_raw_ankle_confidence': ankle_min,
                              'finding': 'All requested raw ankle visibility/presence minima exceed .5; blanket low-confidence claim is unsupported'},
         'finding': 'Sole contacts are not fully visible; barbell is visible but its mass, contact forces, EMG and physiological activation are not supplied',
         'missing_evidence': ['measured EMG', 'force-plate data', 'measured contact forces', 'calibrated barbell mass', 'calibrated floor/camera']}]
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    x = [row['frame'] for row in projection]
    axes[0].plot(x, [row['image_pelvis_px'][1] for row in projection], 'o-')
    axes[0].set_ylabel('Image pelvis y (pixels)')
    for axis, name in enumerate(['x', 'y (depth)', 'z (up)']):
        axes[1].plot(x, [row['root_translation_xyz'][axis] for row in projection], 'o-', label=name)
    axes[1].set_ylabel('Root (model units)'); axes[1].legend()
    for name in ['left_hip', 'right_hip', 'left_knee', 'right_knee']:
        axes[2].plot(x, [row['joint_angles_rad'][name] for row in projection], 'o-', label=name)
    axes[2].set_ylabel('Inferred angle (rad)'); axes[2].legend(ncol=2)
    axes[3].plot(x, [row['sealed_floor_check']['max_penetration_model_units'] for row in projection], 'o-')
    axes[3].axhline(.025, color='red', linestyle='--', label='unchanged gate')
    axes[3].set_ylabel('Overlap (model units)'); axes[3].set_xlabel('Absolute source frame; lines connect only the four requested samples')
    fig.suptitle('Construction-only check: root z is derived from image y, not independently measured')
    fig.tight_layout(); fig.savefig(out / 'projection-check.png', dpi=140); plt.close(fig)
    for name in ['hypotheses.json', 'context.json', 'advisor-call.json']:
        shutil.copyfile(live / name, out / ('input-' + name))
    shutil.copyfile(sealed / 'squat/human.json', out / 'input-human.json')
    shutil.copyfile(Path(__file__), out / Path(__file__).name)
    assert all(digest(Path(path)) == expected for path, expected in hashes.items())
    report = {'schema_version': 1, 'created_at': local_time(), 'accepted': False, 'applied': False,
              'scope': 'Independent construction-only checks of untrusted model proposals; no physical validation or repair',
              'source_sha256': spec['source_sha256'], 'input_hashes': hashes,
              'sealed_feasibility_verification': verification, 'outcomes': outcomes, 'measurements': rows,
              'visual_audit_source_frames': indices, 'physics_steps': 0, 'new_api_calls': 0, 'external_api_usd': 0,
              'limitations': ['Only the requested five existing frames inspected',
                              'No hypothesis establishes camera motion, contact force, muscle activation, barbell mass or a causal repair']}
    write_json(out / 'report.json', report)
    print(json.dumps({'outcomes': outcomes, 'measurements': [{key:row[key] for key in
          ['frame','image_pelvis_px','root_translation_xyz','lowest_collision_surface']} for row in rows],
          'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
