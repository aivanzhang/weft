"""One CPU-only analytic screen; outputs are fixtures, never generated avatar assets."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from edge_fringe import build_edge_fringe, sample_color, two_tip_atlas


HERE = Path(__file__).resolve().parent
STARTED = time.monotonic()


def save(name, value):
    with (HERE/name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def fixture():
    columns, rows, radius = 40, 120, .12
    theta = np.linspace(-np.pi/2, np.pi/2, columns+1)
    z = np.linspace(-.54, .54, rows+1)
    points = np.array([(radius*np.sin(t), -radius*np.cos(t), h) for h in z for t in theta])
    normal = np.array([(np.sin(t), -np.cos(t), 0.) for _ in z for t in theta])
    uv = np.array([(x/columns, 1-y/rows) for y in range(rows+1) for x in range(columns+1)])
    faces = []
    for y in range(rows):
        for x in range(columns):
            a = y*(columns+1)+x
            faces.extend([(a, a+1, a+columns+2), (a, a+columns+2, a+columns+1)])
    faces = np.asarray(faces, dtype=np.int32)
    weights = np.zeros((len(points), 4))
    weights[:, 1] = np.clip((points[:, 2]+.04)/.08, 0, 1)
    weights[:, 0] = 1-weights[:, 1]
    joints = np.tile([0, 1, 0, 0], (len(points), 1)).astype(np.uint16)
    image = np.empty((rows+1, columns+1, 3), dtype=np.uint8)
    image[:, :, 0] = np.rint(np.linspace(0, 255, columns+1))[None, :]
    image[:, :, 1] = np.rint(np.linspace(255, 0, rows+1))[:, None]
    image[:, :, 2] = 128
    nf = len(faces)
    inputs = {'points': points, 'faces': faces, 'corner_normals': normal[faces], 'uv0': uv[faces],
              'materials': np.zeros(nf, dtype=np.int32), 'parts': np.zeros(nf, dtype=np.int32),
              'fur_faces': np.ones(nf, dtype=bool), 'non_fur_faces': np.zeros(nf, dtype=bool),
              'closure_faces': np.zeros(nf, dtype=bool), 'joints': joints, 'weights': weights,
              'flow_directions': np.tile([0., 0., -1.], (len(points), 1)),
              'textures': {0: {'pixels': image, 'factor': np.ones(3)}, 1: {'pixels': image, 'factor': np.ones(3)}},
              'design_views': np.array([[np.sin(t), -np.cos(t), .1] for t in np.linspace(-np.pi, np.pi, 8, endpoint=False)]),
              'settings': {'seed': 912, 'min_fins': 4800, 'max_fins': 4800, 'length_m': .008,
                           'min_spacing_m': .003, 'min_edge_m': .0085, 'max_edge_m': .010, 'comb': .15}}
    inputs['non_fur_faces'][0:20] = True
    inputs['closure_faces'][100:110] = True
    inputs['fur_faces'][200:210] = False
    inputs['materials'][300:310] = 1
    inputs['parts'][400:410] = 1
    inputs['uv0'][500, 0, 0] += .001
    return inputs


def posed(points, joints, weights, angle):
    c, s = np.cos(angle), np.sin(angle)
    matrix = np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
    rotated = points@matrix.T
    influence = np.sum(weights*(joints == 1), axis=1)
    return points*(1-influence[:, None])+rotated*influence[:, None]


def groups(mask):
    return int(np.count_nonzero(np.diff(np.r_[False, mask, False].astype(int)) == 1))


def main():
    inputs = fixture()
    original = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in inputs.items() if isinstance(value, np.ndarray)}
    result = build_edge_fringe(**inputs, source_export_vertices=len(inputs['points']))
    assert result['report']['selected_fins'] == 4800
    small = {**inputs, 'settings': {**inputs['settings'], 'min_fins': 24, 'max_fins': 24}}
    first = build_edge_fringe(**small, source_export_vertices=len(inputs['points']))
    second = build_edge_fringe(**small, source_export_vertices=len(inputs['points']))
    assert all(np.array_equal(value, second[key]) for key, value in first.items() if isinstance(value, np.ndarray))
    edge_counts = Counter(tuple(sorted(edge)) for face in inputs['faces']
                          for edge in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])))
    assert all(edge_counts[tuple(edge)] == 2 for edge in result['root_vertex_ids'])
    positions, triangles = result['positions'].astype(np.float64), result['triangles']
    normals = result['normals'].astype(np.float64)
    geometric = np.cross(positions[triangles[:, 1]]-positions[triangles[:, 0]], positions[triangles[:, 2]]-positions[triangles[:, 0]])
    geometric /= np.linalg.norm(geometric, axis=1)[:, None]
    assert np.all(np.sum(geometric*normals[triangles[:, 0]], axis=1) > .99999)
    assert np.isfinite(positions).all() and np.max(np.abs(np.linalg.norm(normals, axis=1)-1)) < 1e-6
    assert np.isfinite(result['fin_directions']).all()
    assert result['report']['min_outward_dot'] > .95
    assert np.max(np.abs(result['weights'].sum(axis=1)-1)) < 1e-6
    assert all(result['report']['exclusions'].get(name, 0) > 0 for name in
               ('non_fur', 'authored_closure', 'material_seam', 'moving_part_seam', 'UV_seam'))
    roots = result['root_centers']
    minimum = float('inf')
    for start in range(0, len(roots), 128):
        delta = roots[start:start+128, None]-roots[None]
        distance = np.sum(delta*delta, axis=2)
        distance[np.arange(len(distance)), np.arange(start, start+len(distance))] = np.inf
        minimum = min(minimum, float(distance.min()))
    minimum = float(np.sqrt(minimum))
    assert minimum >= inputs['settings']['min_spacing_m']-1e-12
    errors = []
    for degrees in (0, -25, 25, 45, 60):
        source = posed(inputs['points'], inputs['joints'], inputs['weights'], np.radians(degrees))
        fins = posed(positions, result['joints'], result['weights'], np.radians(degrees))
        error = float(np.linalg.norm(fins.reshape(-1, 4, 3)[:, :2]-source[result['root_vertex_ids']], axis=2).max())
        assert error < 1e-6 and np.isfinite(fins).all()
        errors.append({'degrees': degrees, 'max_root_attachment_error_m': error})
    root_ids = result['root_vertex_ids']
    image = inputs['textures'][0]['pixels']
    x, y = root_ids % 41, root_ids//41
    source_rgb = image[120-y, x].astype(float)/255
    expected = np.where(source_rgb <= .04045, source_rgb/12.92, ((source_rgb+.055)/1.055)**2.4)
    color_error = float(np.max(np.abs(result['colors'].reshape(-1, 4, 3)[:, :2]-expected)))
    assert color_error < 1e-6
    known = {'pixels': np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8),
             'factor': np.array([.8, .5, 1.])}
    np.testing.assert_allclose(sample_color(known, np.array([.25, .75])), [.3, .125, .75], atol=1e-12)
    invalid_cases = []
    for label, changes in (('too_many_fins', {'settings': {**inputs['settings'], 'max_fins': 4801}}),
                           ('aggregate_vertices', {'source_export_vertices': 67105}),
                           ('oversized_source', {'points': np.broadcast_to(np.zeros((1, 3)), (100001, 3))}),
                           ('invalid_weights', {'weights': np.full_like(inputs['weights'], .5)})):
        values = {**inputs, 'source_export_vertices': len(inputs['points']), **changes}
        try:
            build_edge_fringe(**values)
            raise AssertionError('Invalid input unexpectedly accepted: '+label)
        except ValueError as error:
            invalid_cases.append({'case': label, 'rejected': str(error)})
    assert all(hashlib.sha256(inputs[key].tobytes()).hexdigest() == value for key, value in original.items())

    atlas = two_tip_atlas(193)
    Image.fromarray(atlas).save(HERE/'analytic-two-tip-atlas.png')
    footprints = []
    for display in (720, 512, 256):
        for span in (.006, .008, .009, .010):
            width, height = max(1, round(display*span/1.43)), max(1, round(display*.008/1.43))
            rows, opacity = [], []
            for tile in range(16):
                y, x = tile//4*256, tile % 4*256
                reduced = np.asarray(Image.fromarray(atlas[y:y+256, x:x+256, 3]).resize((width, height), Image.Resampling.BOX))/255
                mask = reduced[height//2] > .18
                rows.append(groups(mask)); opacity.append(float((reduced > .18).mean()))
            footprints.append({'display_px': display, 'span_m': span, 'projected_span_px': display*span/1.43,
                               'integer_mask_size': [width, height], 'two_tip_tiles_at_middle_row': rows.count(2),
                               'zero_tip_tiles': rows.count(0), 'merged_tiles': rows.count(1),
                               'analytic_alpha_occupancy_mean': float(np.mean(opacity))})
    selected = [row for row in footprints if row['display_px'] == 512 and row['span_m'] == .009][0]
    face_on_ratio = .15/np.sqrt(.85**2+.15**2)
    screen = {'schema_version': 1, 'kind': 'analytic_two_bone_curved_patch_not_avatar',
        'status': 'geometry_contracts_passed_display_screen_pending_review', 'geometry': result['report'],
        'minimum_root_spacing_m': minimum, 'bend_attachment': errors, 'max_endpoint_color_error': color_error,
        'sources_unchanged': True, 'small_same_seed_array_determinism_verified': True,
        'all_roots_have_exactly_two_source_faces': True, 'invalid_inputs': invalid_cases, 'footprints': footprints,
        'nominal_9mm_512px_two_tip_tiles': selected['two_tip_tiles_at_middle_row'],
        'face_on_projection_bound': {'comb': .15, 'visible_fin_depth_m': .008*face_on_ratio,
            'at_512px': 512*.008*face_on_ratio/1.43,
            'note': 'Analytic planar local surface, no occlusion/light/GPU test; no wide face-on block guarantee.'},
        'display_screen_passed_at_nominal_512px': selected['two_tip_tiles_at_middle_row'] == 16,
        'elapsed_seconds': time.monotonic()-STARTED, 'native_provider_ai_role_calls': 0,
        'quality_approved': False,
        'limitations': ['Alpha occupancy and projected group counts are analytic, not visible fur quality.',
            'Only a source-independent curved patch was built; no panda derivative or GLB export.',
            'Two-sided GPU shading, occlusion, culling, AA and actual view allocation remain unverified.',
            'No guarantee at256px; merged tips are recorded, never relabeled resolved fibers.',
            'Endpoint attachment is sampled over five authored LBS poses, not simulated hair.']}
    np.savez_compressed(HERE/'analytic-fins.npz', **{key: value for key, value in result.items() if isinstance(value, np.ndarray)})
    save('result.json', screen)
    print(json.dumps({'status': screen['status'], 'fins': result['report']['selected_fins'],
                      'nominal_512_two_tip_tiles': selected['two_tip_tiles_at_middle_row'],
                      'elapsed_seconds': screen['elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save('failure.json', {'status': 'failed_preserved', 'error': type(error).__name__+': '+str(error),
                              'elapsed_seconds': time.monotonic()-STARTED, 'quality_approved': False})
        raise
