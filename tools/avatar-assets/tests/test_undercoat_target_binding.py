"""Native receipt admission fixtures, not rendered fur or appearance acceptance."""

from copy import deepcopy
import unittest

from avatar_assets.finish_generated import _undercoat_tangent_targets
from avatar_assets.generated_tangents import NORMAL_UV1_FRAME_POLICY as POLICY


def report(uv=2):
    return {'fur_undercoat': {'frame_policy': POLICY, 'plan': {'frame_policy': POLICY},
        'authored_uv': {'index': uv, 'name': 'AuthoredUndercoatUV'+str(uv), 'sha256': 'b'*64},
        'original_uv_layers': [{'index': i, 'name': 'SourceUV'+str(i), 'sha256': 'c'*64,
                                'active_render': i == 0} for i in range(uv)],
        'materials': [{'material': 'source__short_pile_adaptive_v1', 'normal_texcoord': uv,
                       'normal_uv_map': 'AuthoredUndercoatUV'+str(uv), 'selected_faces': 4,
                       'texture_sha256': 'a'*64, 'frame_policy': POLICY},
                      {'material': 'healthy__short_pile_adaptive_v1', 'normal_texcoord': 0,
                       'normal_uv_map': 'SourceUV0', 'selected_faces': 2,
                       'texture_sha256': 'd'*64, 'frame_policy': None}]}}


class UndercoatTargetBindingTests(unittest.TestCase):
    def test_only_authored_atlas_rows_become_export_targets(self):
        for uv in (1, 2):
            native = report(uv)
            before = deepcopy(native)
            targets = _undercoat_tangent_targets(native, True)
            self.assertEqual(targets, [{key: native['fur_undercoat']['materials'][0][key] for key in
                              ('material', 'normal_texcoord', 'selected_faces', 'texture_sha256', 'frame_policy')}])
            self.assertEqual(native, before)
        self.assertIsNone(_undercoat_tangent_targets({}, False))

    def test_legacy_uv1_receipt_still_requires_exact_name_count_and_hash(self):
        native = report(1)
        native['fur_undercoat'].pop('authored_uv')
        native['fur_undercoat'].pop('original_uv_layers')
        self.assertEqual(_undercoat_tangent_targets(native, True)[0]['normal_texcoord'], 1)
        for key, value in (('material', None), ('selected_faces', True), ('selected_faces', 2049),
                           ('texture_sha256', 'missing'), ('frame_policy', 'legacy')):
            bad = deepcopy(native)
            bad['fur_undercoat']['materials'][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                _undercoat_tangent_targets(bad, True)

    def test_uv2_cannot_reinterpret_or_overwrite_existing_source_uv1(self):
        mutations = (
            lambda d: d.pop('authored_uv'),
            lambda d: d['authored_uv'].update(index=2.),
            lambda d: d['authored_uv'].update(sha256='invalid'),
            lambda d: d['original_uv_layers'].pop(),
            lambda d: d['original_uv_layers'][1].update(index=True),
            lambda d: d['original_uv_layers'][1].update(active_render=True),
            lambda d: d['original_uv_layers'][1].update(name='SourceUV0'),
            lambda d: d['materials'][0].update(normal_uv_map='SourceUV1'),
            lambda d: d['materials'].append({**d['materials'][0], 'material': 'another', 'normal_texcoord': 1}),
        )
        for index, mutation in enumerate(mutations):
            native = report()
            mutation(native['fur_undercoat'])
            with self.subTest(index=index), self.assertRaises(ValueError):
                _undercoat_tangent_targets(native, True)


if __name__ == '__main__':
    unittest.main()
