"""Analytic image/video fixtures test evidence fidelity, not generated-avatar quality."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from avatar_assets.media import encode_video, video_info
from avatar_assets.pipeline_evidence import build_evidence, decoded_poses, select_indices
from avatar_assets.quality_policy import evaluate_quality_evidence
from avatar_assets.safety import read_json, sha256, write_json
from test_quality_policy import policy


class FullEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.pack, self.exported, self.media = [self.root/n for n in ('native', 'exported', 'media')]
        for p in (self.pack, self.exported, self.media):
            p.mkdir()
        for representation, root in (('native', self.pack/'views'), ('exported', self.exported)):
            root.mkdir(exist_ok=True)
            for view in ('front', 'side', 'back', 'three_quarter'):
                name = 'three-quarter' if representation == 'exported' and view == 'three_quarter' else view
                Image.new('RGB', (720, 720), 'navy').save(root/(name+'.png'))
        for root in (self.pack, self.exported):
            folder = root/'renders/dance'
            folder.mkdir(parents=True)
            for i, color in enumerate(('red', 'blue', 'yellow', 'purple')):
                Image.new('RGB', (720, 720), color).save(folder/f'{i:04d}.png')
        video = self.media/'dance.mp4'
        encode_video((Image.new('RGB', (128, 128), c) for c in ('green', 'cyan', 'orange', 'white')),
                     video, 24, 128, 128, '#ffffff')
        self.facts = video_info(video)
        write_json(self.pack/'manifest.json', {'states': [{'id': 'dance'}]})
        write_json(self.pack/'finish-report.json', {'states': [{'id': 'dance', 'seconds': [0, 1, 2, 4]}]})
        write_json(self.exported/'manifest.json', {'provenance': 'analytic_fixture'})
        Image.new('RGB', (1600, 1600), 'silver').save(self.media/'portrait.webp', lossless=True)
        write_json(self.media/'manifest.json', {'parent_3d_manifest_sha256': sha256(self.pack/'manifest.json'),
                                               'clips': [{'id': 'dance', 'path': 'dance.mp4', 'inspection': self.facts}]})

    def test_full_resolution_and_actual_encoded_media_are_preserved(self):
        report = build_evidence(self.pack, self.exported, self.media, self.root/'evidence')
        self.assertTrue(evaluate_quality_evidence(policy(False), report['coverage'])['passed'])
        self.assertEqual(len(report['images']), 6)
        isolated = [image for image in report['images'] if image['kind'] == 'isolated_motion_pose']
        self.assertEqual([image['second'] for image in isolated], [2, 2])
        for image in isolated:
            source = self.pack if image['representation'] == 'native' else self.exported
            self.assertEqual(image['sha256'], sha256(source/'renders/dance/0002.png'))
            self.assertEqual((image['width'], image['height']), (720, 720))
        self.assertEqual(report['sources'][str(self.media/'dance.mp4')], sha256(self.media/'dance.mp4'))
        with Image.open(self.root/'evidence/motion-00.png') as sheet:
            self.assertEqual(sheet.size, (2880, 3740))
            self.assertEqual(sheet.getpixel((360, 388)), (255, 0, 0))
            # The last row is decoded green video, not the red native pose above it.
            color = sheet.getpixel((360, 3380))
            self.assertGreater(color[1], color[0]+80)
        self.assertEqual(report['model_manifest_sha256'], sha256(self.pack/'manifest.json'))

    def test_largest_recorded_deformation_selects_real_unmodified_pose(self):
        finish = read_json(self.pack/'finish-report.json')
        finish['states'][0]['deformation_outliers'] = [
            {'second': 1, 'excess_length': .002}, {'second': 4, 'excess_length': .001}]
        (self.pack/'finish-report.json').write_text(json.dumps(finish))
        report = build_evidence(self.pack, self.exported, self.media, self.root/'evidence')
        isolated = [image for image in report['images'] if image['kind'] == 'isolated_motion_pose']
        self.assertEqual([image['second'] for image in isolated], [1, 1])
        self.assertTrue(all(image['selection'] == 'greatest_recorded_edge_stretch' for image in isolated))

    def test_enlarged_contact_sheet_cannot_hide_low_resolution_input(self):
        Image.new('RGB', (128, 128), 'red').save(self.exported/'renders/dance/0000.png')
        with self.assertRaisesRegex(ValueError, 'actual720px'):
            build_evidence(self.pack, self.exported, self.media, self.root/'evidence')

    def test_modified_video_or_wrong_decode_count_is_rejected(self):
        wrong = deepcopy(self.facts)
        wrong['frames'] += 1
        with self.assertRaisesRegex(ValueError, 'frame count'):
            decoded_poses(self.media/'dance.mp4', wrong)
        media = read_json(self.media/'manifest.json')
        media['clips'][0]['inspection']['sha256'] = '0'*64
        (self.media/'manifest.json').unlink()
        write_json(self.media/'manifest.json', media)
        with self.assertRaisesRegex(ValueError, 'video changed'):
            build_evidence(self.pack, self.exported, self.media, self.root/'evidence')

    def test_sampling_bounds_and_encoded_dimensions_fail_closed(self):
        for count in (True, 1, 481, '4'):
            with self.assertRaises(ValueError):
                select_indices(count)
        self.assertEqual(select_indices(97), [0, 32, 64, 96])
        with self.assertRaisesRegex(ValueError, 'decode bounds'):
            decoded_poses(self.media/'dance.mp4', {**self.facts, 'width': 2000, 'height': 2000})


if __name__ == '__main__':
    unittest.main()
