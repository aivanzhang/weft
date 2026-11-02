"""Tiny read-only analytic checks; never load a native module or source texture."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent


def function(path, name):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[name]


def main():
    started = time.monotonic()
    pins = json.loads((HERE/'source-pins.json').read_text())
    for row in pins['sources']:
        assert hashlib.sha256((HERE/row['saved_path']).read_bytes()).hexdigest() == row['sha256']
    source = HERE/'snapshots/fur_root_sampling.py'
    spec = importlib.util.spec_from_file_location('pinned_root_sampler', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    repeat = function(HERE/'snapshots/experimental_build_candidate_v2.py', 'repeat_root_colors')
    old = function(HERE/'snapshots/experimental_edge_fringe_v1.py', 'sample_color')
    shared = function(HERE/'snapshots/blender_finish_generated.py', 'shared_color_texture')
    nearest = function(HERE/'snapshots/blender_finish_generated.py', 'texture_colors')

    # Native rows/UV-V are both reversed relative to the glTF/image-row convention.
    pixels = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
    uv = np.array([[.25, .25], [.75, .25], [.25, .75], [.75, .75],
                   [0, 0], [1, 1], [.5, .25], [-.25, .25], [1.25, .25]])
    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],
                         [.5, .5, .5], [.5, .5, .5], [.5, .5, 0], [0, 1, 0], [1, 0, 0]])
    native_uv = uv.copy()
    native_uv[:, 1] = 1-native_uv[:, 1]
    corners = np.repeat(native_uv[:, None, :], 3, axis=1)
    sampler = module.RootColorSampler(pixels[::-1].astype(np.float32)/255, corners)
    actual = np.array([sampler(i, (1, 0, 0)) for i in range(len(uv))])
    assert np.array_equal(actual, expected)
    assert np.array_equal(repeat(pixels, uv, np.ones(3)), expected)
    legacy = np.array([old({'pixels': pixels, 'factor': np.ones(3)}, item) for item in uv[:7]])
    wrong_rows = module.RootColorSampler(pixels.astype(np.float32)/255, corners)(0, (1, 0, 0))
    assert wrong_rows != tuple(expected[0])

    gray = np.array([[[128]*3, [255]*3], [[64]*3, [0]*3]], dtype=np.uint8)
    gray_native = module.RootColorSampler(gray[::-1].astype(np.float32)/255,
                                         np.array([[[.5, .75]]*3]))
    gray_observed = gray_native(0, (1, 0, 0))[0]
    gray_expected = (((128/255+.055)/1.055)**2.4+1)/2
    decode_after = (((128/255+1)/2+.055)/1.055)**2.4
    assert abs(gray_observed-gray_expected) < 2e-7
    assert abs(gray_observed-decode_after) > .05
    linear = module.RootColorSampler(np.full((2, 2, 3), .5), np.array([[[.5, .75]]*3]),
                                     pixels_are_linear=True)(0, (1, 0, 0))
    assert linear == (.5, .5, .5)

    accepted = []
    image_id = object()
    for mode in ('Linear', 'Closest', 'Cubic', 'Smart'):
        texture = SimpleNamespace(type='TEX_IMAGE', image=image_id, projection='FLAT',
                                  extension='REPEAT', interpolation=mode,
                                  inputs={'Vector': SimpleNamespace(links=[])})
        shader = SimpleNamespace(type='BSDF_PRINCIPLED', inputs={
            'Base Color': SimpleNamespace(links=[SimpleNamespace(from_node=texture)])})
        material = SimpleNamespace(use_nodes=True, node_tree=SimpleNamespace(nodes=[shader, texture]))
        assert shared([material]) is texture
        accepted.append(mode)

    rgba = np.concatenate((pixels[::-1].astype(np.float32)/255, np.ones((2, 2, 1), dtype=np.float32)), axis=2)
    image = SimpleNamespace(size=(2, 2), pixels=SimpleNamespace(
        foreach_get=lambda target: target.__setitem__(slice(None), rgba.ravel())))
    obj = SimpleNamespace(data=SimpleNamespace(vertices=[None],
        loops=[SimpleNamespace(vertex_index=0, index=0)],
        uv_layers=SimpleNamespace(active=SimpleNamespace(data=[SimpleNamespace(uv=(.5, .75))]))))
    legacy_nearest = nearest(obj, image)[0]
    assert np.array_equal(legacy_nearest, [0, 1, 0])

    report = {'schema_version': 1, 'status': 'read_only_analytic_contract_checks_complete',
        'kind': '2x2_fixture_not_avatar_or_native_shader_proof',
        'texel_center_repeat_orientation_cases': len(uv), 'actual_rgb': actual.tolist(),
        'gltf_image_v_down_uv': uv.tolist(), 'blender_bottom_up_uv': native_uv.tolist(),
        'existing_root_sampler_matches_corrected_candidate_kernel': True,
        'legacy_endpoint_helper_maximum_fixture_difference': float(np.abs(legacy-expected[:7]).max()),
        'wrong_row_orientation_at_first_center': list(wrong_rows),
        'srgb_decode_before_filter': {'observed': gray_observed, 'expected': gray_expected,
                                      'incorrect_decode_after_filter': decode_after},
        'linear_input_bypass_rgb': list(linear),
        'shared_color_texture_accepts_interpolations': accepted,
        'closest_vs_root_bilinear_at_midpoint': {'nearest': legacy_nearest.tolist(),
                                               'root_sampler': actual[6].tolist()},
        'production_or_tests_changed': False, 'native_provider_ai_calls': 0,
        'source_texture_decodes': 0, 'quality_approved': False,
        'elapsed_seconds': time.monotonic()-started}
    with (HERE/'analytic-checks.json').open('x') as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': report['status'], 'elapsed_seconds': report['elapsed_seconds'],
                      'cases': len(uv), 'endpoint_difference': report['legacy_endpoint_helper_maximum_fixture_difference']}))


if __name__ == '__main__':
    main()
