"""One bounded analytic allocation comparison; no avatar or renderer is involved."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'sources'))
import edge_fringe_frozen as old
import allocation_fringe as changed
from root_sampler import repeat_root_colors

STARTED = time.monotonic()


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError('Unsupported JSON value: '+type(value).__name__)


def save(name, value):
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=json_scalar)+'\n'
    with (HERE/name).open('x') as stream:
        stream.write(payload)


def fixture(counts, reverse_diagonal=False):
    # Extruding the same polygonal arc keeps all three surfaces' actual areas equal.
    columns, radius, band_height = 40, .12, .216
    theta = np.linspace(-np.pi/2, np.pi/2, columns+1)
    z = np.concatenate([np.linspace(-.324+i*band_height, -.324+(i+1)*band_height,
                                   count+1)[int(i > 0):] for i, count in enumerate(counts)])
    points = np.array([(radius*np.sin(t), -radius*np.cos(t), h) for h in z for t in theta])
    normal = np.array([(np.sin(t), -np.cos(t), 0.) for _ in z for t in theta])
    uv = np.array([(x/columns, 1-(h+.324)/.648) for h in z for x in range(columns+1)])
    faces = []
    for row in range(len(z)-1):
        for column in range(columns):
            a = row*(columns+1)+column
            b, c, d = a+1, a+columns+2, a+columns+1
            if reverse_diagonal and (row+column) % 2 == 0:
                faces.extend(((a, b, d), (b, c, d)))
            else:
                faces.extend(((a, b, c), (a, c, d)))
    faces = np.asarray(faces, dtype=np.int32)
    weights = np.zeros((len(points), 4))
    weights[:, 1] = np.clip((points[:, 2]+.04)/.08, 0, 1)
    weights[:, 0] = 1-weights[:, 1]
    pixels = np.array([[[255, 0, 0], [0, 255, 0]],
                       [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
    nf = len(faces)
    return {'points': points, 'faces': faces, 'corner_normals': normal[faces], 'uv0': uv[faces],
            'materials': np.zeros(nf, dtype=np.int32), 'parts': np.zeros(nf, dtype=np.int32),
            'fur_faces': np.ones(nf, dtype=bool), 'non_fur_faces': np.zeros(nf, dtype=bool),
            'closure_faces': np.zeros(nf, dtype=bool),
            'joints': np.tile([0, 1, 0, 0], (len(points), 1)).astype(np.uint16), 'weights': weights,
            'flow_directions': np.tile([0., 0., -1.], (len(points), 1)),
            'textures': {0: {'pixels': pixels, 'factor': np.ones(3)},
                         1: {'pixels': pixels, 'factor': np.ones(3)}},
            'design_views': np.array([[np.sin(t), -np.cos(t), .1]
                                     for t in np.linspace(-np.pi, np.pi, 8, endpoint=False)])}


def settings(new, maximum=4800):
    return {'seed': 912, 'min_fins': 1, 'max_fins': maximum, 'length_m': .008,
            'min_spacing_m': .003, 'min_edge_m': .0085,
            'max_edge_m': .020 if new else .010, 'comb': .15}


def posed(points, joints, weights, degrees):
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    matrix = np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
    influence = np.sum(weights*(joints == 1), axis=1)
    return points*(1-influence[:, None])+(points@matrix.T)*influence[:, None]


def measure(source, result, new):
    root_ids = result['root_vertex_ids']
    positions = result['positions'].astype(float).reshape(-1, 4, 3)
    assert np.isfinite(positions).all()
    assert np.max(np.abs(positions[:, :2]-source['points'][root_ids])) < 5e-8
    assert np.array_equal(result['joints'].reshape(-1, 4, 4)[:, :2], source['joints'][root_ids])
    assert np.max(np.abs(result['weights'].sum(axis=1)-1)) < 1e-6
    adjacency = defaultdict(list)
    for face_id, face in enumerate(source['faces']):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            adjacency[tuple(sorted((int(a), int(b))))].append(face_id)
    assert all(len(adjacency[tuple(edge)]) == 2 for edge in root_ids)
    for edge in root_ids:
        pair = adjacency[tuple(edge)]
        assert source['fur_faces'][pair].all() and not source['non_fur_faces'][pair].any()
        assert not source['closure_faces'][pair].any()
        assert source['materials'][pair[0]] == source['materials'][pair[1]]
        assert source['parts'][pair[0]] == source['parts'][pair[1]]
    triangles = result['triangles']
    points = positions.reshape(-1, 3)
    normals = result['normals'].astype(float)
    face_normal = np.cross(points[triangles[:, 1]]-points[triangles[:, 0]],
                           points[triangles[:, 2]]-points[triangles[:, 0]])
    face_normal /= np.linalg.norm(face_normal, axis=1)[:, None]
    normal_dot = np.sum(face_normal*normals[triangles[:, 0]], axis=1)
    assert normal_dot.min() > .99999 and np.isfinite(normals).all()
    assert np.max(np.abs(np.linalg.norm(normals, axis=1)-1)) < 1e-6
    assert result['report']['min_outward_dot'] > .95
    roots = result['root_centers']
    minimum = float('inf')
    for start in range(0, len(roots), 128):
        delta = roots[start:start+128, None]-roots[None]
        distances = np.einsum('ijk,ijk->ij', delta, delta)
        distances[np.arange(len(distances)), np.arange(start, start+len(distances))] = np.inf
        minimum = min(minimum, float(distances.min()))
    assert minimum >= .003**2-1e-12
    errors = []
    for angle in (0, -25, 25, 45, 60):
        expected = posed(source['points'], source['joints'], source['weights'], angle)[root_ids]
        actual = posed(points, result['joints'], result['weights'], angle).reshape(-1, 4, 3)[:, :2]
        error = float(np.linalg.norm(actual-expected, axis=2).max())
        errors.append({'degrees': angle, 'maximum_root_error_m': error})
        assert error < 1e-7
    if new:
        expected = repeat_root_colors(source['textures'][0]['pixels'],
                                      result['root_source_uv'].reshape(-1, 2), np.ones(3))
        color_error = float(np.max(np.abs(result['colors'].reshape(-1, 4, 3)[:, :2].reshape(-1, 3)-expected)))
        assert color_error < 6e-8
    else:
        color_error = None
    vertices = source['points'][source['faces']]
    areas = .5*np.linalg.norm(np.cross(vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0]), axis=1)
    face_band = np.clip(np.floor((vertices.mean(axis=1)[:, 2]+.324)/.216).astype(int), 0, 2)
    root_band = np.clip(np.floor((roots[:, 2]+.324)/.216).astype(int), 0, 2)
    area_by_band = np.bincount(face_band, weights=areas, minlength=3)
    counts = np.bincount(root_band, minlength=3)
    expected_fraction = area_by_band/area_by_band.sum()
    fraction = counts/counts.sum()
    density = counts/area_by_band
    spans = np.linalg.norm(source['points'][root_ids[:, 1]]-source['points'][root_ids[:, 0]], axis=1)
    assert spans.min() >= .0085-1e-12 and spans.max() <= (.020 if new else .010)+1e-12
    assert len(root_ids) <= 4800 and result['report']['projected_vertices'] <= 100000
    return {'fins': len(root_ids), 'area_by_band_m2': area_by_band.tolist(),
            'counts_by_equal_area_band': counts.tolist(), 'fins_per_square_meter': density.tolist(),
            'allocation_fraction': fraction.tolist(), 'area_fraction': expected_fraction.tolist(),
            'allocation_total_variation': float(.5*np.abs(fraction-expected_fraction).sum()),
            'density_cv': float(density.std()/density.mean()),
            'selected_span_m': [float(spans.min()), float(np.median(spans)), float(spans.max())],
            'minimum_root_spacing_m': float(np.sqrt(minimum)),
            'minimum_normal_winding_dot': float(normal_dot.min()), 'attachment': errors,
            'maximum_linear_endpoint_color_error': color_error, 'builder': result['report']}


def group_ranges(mask):
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def atlas_measurements(atlas):
    # A physical line through the middle of the tile isolates tip scale from edge length.
    tile = atlas[:256, :256, 3].astype(float)/255
    middle = (tile[127]+tile[128])*.5
    x = np.linspace(-.0104, .0104, 16385)
    atlas_x = (x/changed.ATLAS_DOMAIN_M+.5)*255
    base = np.interp(atlas_x, np.arange(256), middle)
    physical = []
    masks = []
    for span in (.0085, .010, .015, .020):
        alpha = base.copy()
        alpha[np.abs(x) > span/2] = 0
        mask = alpha >= .18
        ranges = group_ranges(mask)
        assert len(ranges) == 2
        widths = [(b-a)*(x[1]-x[0]) for a, b in ranges]
        centers = [float(x[a:b].mean()) for a, b in ranges]
        physical.append({'edge_span_m': span, 'tip_widths_at_mid_height_m': widths,
                         'tip_center_separation_m': centers[1]-centers[0],
                         'visible_extent_m': [float(x[ranges[0][0]]), float(x[ranges[-1][1]-1])],
                         'maximum_alpha_at_cropped_root_edge': float(np.interp(span/2, x, base))})
        masks.append(mask)
    assert all(np.array_equal(masks[0], mask) for mask in masks[1:])
    assert np.max(np.abs(x[masks[0]])) < .0033
    phase_records = []
    scale_m = 1.43
    for display in (720, 512, 256):
        for span in (.0085, .010, .015, .020):
            counts, max_runs, opaque_counts = [], [], []
            for phase in np.linspace(0, 1, 8, endpoint=False):
                centers_px = np.arange(-12, 13)+phase
                samples_px = centers_px[:, None]+(np.arange(128)+.5)/128-.5
                sample_x = samples_px*scale_m/display
                samples = np.interp(sample_x, x, base, left=0, right=0)
                samples[np.abs(sample_x) > span/2] = 0
                mask = samples.mean(axis=1) >= .18
                ranges = group_ranges(mask)
                counts.append(len(ranges))
                max_runs.append(max((b-a for a, b in ranges), default=0))
                opaque_counts.append(int(mask.sum()))
            phase_records.append({'display_px': display, 'edge_span_m': span,
                'group_counts_by_eighth_pixel_phase': counts, 'maximum_contiguous_opaque_run_px': max(max_runs),
                'opaque_pixel_counts': opaque_counts, 'source_edge_width_px': span*display/scale_m})
    for display in (720, 512, 256):
        rows = [r for r in phase_records if r['display_px'] == display]
        assert all(r['group_counts_by_eighth_pixel_phase'] == rows[0]['group_counts_by_eighth_pixel_phase']
                   and r['opaque_pixel_counts'] == rows[0]['opaque_pixel_counts'] for r in rows[1:])
    return {'physical_crop_checks': physical, 'box_filtered_midline': phase_records,
            'alpha_cutoff': .18, 'nominal_camera_scale_m': scale_m,
            'limits': ['CPU horizontal BOX integration at one height, not GPU alpha coverage or shading.',
                       'Tip merging/disappearance and phase dependence are recorded, never reclassified as resolved hair.',
                       'No visible-width benefit is inferred solely from alpha occupancy.']}


def main():
    cases = [('uniform', (24, 24, 24), False),
             ('coarse_dense_medium', (16, 32, 24), False),
             ('dense_medium_coarse_flipped', (32, 24, 16), True)]
    results, source_hashes, points_for_plot = {}, {}, {}
    for name, counts, reverse in cases:
        source = fixture(counts, reverse)
        hashes = {key: hashlib.sha256(value.tobytes()).hexdigest()
                  for key, value in source.items() if isinstance(value, np.ndarray)}
        source_hashes[name] = hashes
        pair = {}
        for new, label, module in ((False, 'frozen_old', old), (True, 'area_share', changed)):
            start = time.monotonic()
            result = module.build_edge_fringe(**source, settings=settings(new),
                                              source_export_vertices=len(source['points']))
            pair[label] = measure(source, result, new)
            pair[label]['build_and_audit_seconds'] = time.monotonic()-start
            points_for_plot[(name, label)] = result['root_centers'].copy()
            np.savez_compressed(HERE/(name+'-'+label+'-roots.npz'),
                                root_vertex_ids=result['root_vertex_ids'], root_centers=result['root_centers'],
                                root_face_ids=result['root_face_ids'], root_uv=result['root_source_uv'])
            assert hashes == {key: hashlib.sha256(source[key].tobytes()).hexdigest() for key in hashes}
        results[name] = {'axial_rows_by_equal_area_band': list(counts),
                         'source_vertices': len(source['points']), 'source_triangles': len(source['faces']),
                         'diagonal_pattern_alternates': reverse, 'comparison': pair}
    totals = [sum(r['comparison']['area_share']['area_by_band_m2']) for r in results.values()]
    assert max(totals)-min(totals) < 1e-12
    guard = fixture((24, 24, 24))
    guard['non_fur_faces'][:20] = True
    guard['closure_faces'][100:110] = True
    guard['fur_faces'][200:210] = False
    guard['materials'][300:310] = 1
    guard['parts'][400:410] = 1
    guard['uv0'][500, 0, 0] += .001
    guarded = changed.build_edge_fringe(**guard, settings=settings(True, 128),
                                       source_export_vertices=len(guard['points']))
    guard_report = measure(guard, guarded, True)
    assert all(guarded['report']['exclusions'].get(name, 0) > 0 for name in
               ('non_fur', 'authored_closure', 'material_seam', 'moving_part_seam', 'UV_seam'))
    rejected = []
    for label, kwargs in (
        ('oversized_fin_count', {'settings': {**settings(True), 'max_fins': 4801}}),
        ('aggregate_vertex_budget', {'source_export_vertices': 100000}),
        ('invalid_skin', {'weights': np.zeros_like(guard['weights'])})):
        inputs = {**guard, 'settings': settings(True), 'source_export_vertices': len(guard['points']), **kwargs}
        try:
            changed.build_edge_fringe(**inputs)
        except ValueError as error:
            rejected.append({'case': label, 'error': str(error)})
        else:
            raise AssertionError('Invalid input accepted: '+label)
    atlas = changed.two_tip_atlas(912)
    Image.fromarray(atlas).save(HERE/'physical-two-tip-atlas.png')
    atlas_report = atlas_measurements(atlas)
    plot = Image.new('RGB', (1040, 840), '#f3eee6')
    draw = ImageDraw.Draw(plot)
    for row, (name, _, _) in enumerate(cases):
        for column, label in enumerate(('frozen_old', 'area_share')):
            left, top = 20+column*520, 35+row*280
            draw.text((left, top-22), name+' / '+label, fill='#202020')
            draw.rectangle((left, top, left+480, top+240), fill='#ffffff', outline='#555555')
            for split in (80, 160):
                draw.line((left, top+split, left+480, top+split), fill='#ddd5c8')
            points = points_for_plot[(name, label)]
            u = (np.arctan2(points[:, 0], -points[:, 1])+np.pi/2)/np.pi
            v = (points[:, 2]+.324)/.648
            for x, y in zip(left+u*480, top+(1-v)*240):
                draw.point((int(x), int(y)), fill='#204f55' if column else '#965644')
    plot.save(HERE/'analytic-allocation-map.png')
    changed_cases = [results[name]['comparison'] for name, _, _ in cases[1:]]
    improved = all(pair['area_share']['allocation_total_variation'] < pair['frozen_old']['allocation_total_variation']
                   for pair in changed_cases)
    bounded_uniform = results['uniform']['comparison']['area_share']['density_cv'] < .15
    supported = improved and bounded_uniform
    result = {'schema_version': 1, 'kind': 'analytic_allocation_comparison_not_avatar',
              'status': 'passed_analytic_screen' if supported else 'rejected_allocation_hypothesis',
              'allocation_hypothesis_supported_on_these_fixtures': supported,
              'safety_checks_passed': True, 'actual_character_model_used': False,
              'renderer_used': False, 'new_provider_ai_native_calls': 0, 'quality_approved': False,
              'cases': results, 'source_array_hashes': source_hashes,
              'same_actual_surface_area_m2': totals, 'safety_mask_fixture': guard_report,
              'invalid_inputs_rejected': rejected, 'atlas': atlas_report,
              'elapsed_seconds': time.monotonic()-STARTED,
              'limits': ['Three controlled tessellations do not prove allocation on a generated character.',
                         'Exact endpoint LOD0 colors do not reproduce the texture between longer edge endpoints.',
                         'Root spacing does not establish disjoint fins, hidden-surface visibility or visual quality.',
                         'Six-mm spans are not promoted; prior screen found systematic merging at 512 px.',
                         '256-pixel footprints are not claimed to resolve two hairs.']}
    save('result.json', result)
    print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'],
                      'allocation': {name: {label: {'fins': item['fins'],
                          'counts': item['counts_by_equal_area_band'], 'cv': item['density_cv']}
                          for label, item in row['comparison'].items()} for name, row in results.items()}}, indent=2))
    if not supported:
        raise SystemExit(2)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save('failure.json', {'status': 'failed_analytic_screen', 'error_type': type(error).__name__,
                             'error': str(error), 'elapsed_seconds': time.monotonic()-STARTED})
        raise
