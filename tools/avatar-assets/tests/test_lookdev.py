"""Analytic pack fixtures; no native rendering or generated-avatar claims."""

import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from avatar_assets import lookdev
from avatar_assets.safety import sha256


def blend_bytes(extra=()):
    def block(code, data=b''):
        return struct.pack('<4siQii', code, len(data), 0, 0, 1 if data else 0)+data
    return b'BLENDER-v405'+b''.join(block(*item) for item in extra)+block(b'DNA1', b'fixture')+block(b'ENDB')


class LookdevTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.pack = self.root/'pack'
        self.pack.mkdir()
        (self.pack/'source.blend').write_bytes(blend_bytes())
        (self.pack/'texture.bin').write_bytes(b'analytic fixture')
        self.manifest = {'schema_version': 1, 'kind': 'generated_3d_avatar', 'status': 'needs_visual_review',
                         'artifacts': [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha256(p)}
                                       for p in sorted(self.pack.iterdir())]}
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        (self.pack/'manifest.json').write_text(json.dumps(self.manifest))

    def test_verified_manifest_and_header(self):
        receipt, _ = lookdev.preflight(self.pack)
        self.assertEqual(receipt['verified_artifact_count'], 2)
        self.assertEqual(receipt['blend_header']['linked_libraries'], 0)
        self.assertEqual(receipt['source_blend_sha256'], sha256(self.pack/'source.blend'))

    def test_hash_and_size_tampering_rejected(self):
        path = self.pack/'texture.bin'
        path.write_bytes(b'changed contents')
        with self.assertRaisesRegex(ValueError, 'differs from manifest'):
            lookdev.preflight(self.pack)

    def test_paths_duplicates_and_symlink_parents_rejected(self):
        for name in ('../outside', '/absolute', 'nested/../source.blend', './source.blend', 'nested\\asset'):
            with self.subTest(name=name):
                old = self.manifest['artifacts'][0]['path']
                self.manifest['artifacts'][0]['path'] = name
                self.save()
                with self.assertRaises(ValueError):
                    lookdev.preflight(self.pack)
                self.manifest['artifacts'][0]['path'] = old
        self.manifest['artifacts'].append(dict(self.manifest['artifacts'][0]))
        self.save()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            lookdev.preflight(self.pack)
        self.manifest['artifacts'].pop()
        target = self.root/'external'
        target.mkdir()
        (target/'asset').write_bytes(b'x')
        (self.pack/'linked').symlink_to(target, target_is_directory=True)
        self.manifest['artifacts'].append({'path': 'linked/asset', 'bytes': 1, 'sha256': sha256(target/'asset')})
        self.save()
        with self.assertRaisesRegex(ValueError, 'symlinks'):
            lookdev.preflight(self.pack)

    def test_input_bounds_rejected_before_hashing(self):
        self.manifest['artifacts'][0]['bytes'] = lookdev.MAX_INPUT
        self.save()
        with patch.object(lookdev, 'sha256', side_effect=AssertionError('must reject sizes first')):
            with self.assertRaisesRegex(ValueError, 'aggregate 512 MiB'):
                lookdev.preflight(self.pack)
        oversized = self.root/'oversized.blend'
        with oversized.open('wb') as handle:
            handle.truncate(lookdev.MAX_INPUT+1)
        with self.assertRaisesRegex(ValueError, '512 MiB'):
            lookdev.blend_header(oversized)

    def test_external_library_compression_truncation_and_bad_blocks_rejected(self):
        path = self.root/'bad.blend'
        candidates = [blend_bytes([(b'LI\0\0', b'/external/library.blend')]), b'\x28\xb5\x2f\xfd'+b'0'*40,
                      blend_bytes()[:-3], b'BLENDER-v405'+struct.pack('<4siQii', b'DATA', -1, 0, 0, 1)]
        for data in candidates:
            with self.subTest(data=data[:20]):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    lookdev.blend_header(path)
        path.write_bytes(blend_bytes([(b'DATA', b'x')]))
        with patch.object(lookdev, 'MAX_BLEND_BLOCKS', 2):
            with self.assertRaisesRegex(ValueError, 'oversized blend block'):
                lookdev.blend_header(path)

    def test_missing_source_and_failed_pack_are_not_loaded(self):
        self.manifest['status'] = 'failed'
        self.save()
        with self.assertRaisesRegex(ValueError, 'pack manifest'):
            lookdev.preflight(self.pack)
        self.manifest['status'] = 'needs_visual_review'
        self.manifest['artifacts'] = [r for r in self.manifest['artifacts'] if r['path'] != 'source.blend']
        self.save()
        with self.assertRaisesRegex(ValueError, 'source.blend'):
            lookdev.preflight(self.pack)

    def test_output_inside_source_or_existing_output_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'must not modify'):
            lookdev.compare(self.pack, self.pack/'diagnostic')
        out = self.root/'existing'
        out.mkdir()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            lookdev.compare(self.pack, out)

    def test_invalid_variant_selections_rejected_before_native_work(self):
        for selection in ([], ['unknown'], ['base_only', 'base_only'], 'base_only', [1], list(lookdev.VARIANTS)*2):
            with self.subTest(selection=selection), self.assertRaisesRegex(ValueError, 'unique known'):
                lookdev.compare(self.pack, self.root/'unused', selection)

    def test_fixture_orchestration_flags_limits_variants_and_source_proof(self):
        output = self.root/'out'
        def fixture(command, log, out, **limits):
            self.assertIn('--disable-autoexec', command)
            self.assertIn('--offline-mode', command)
            self.assertIn('--factory-startup', command)
            self.assertNotIn('--python-use-system-env', command)
            self.assertEqual(limits, {'timeout': 300, 'memory_mb': 2048, 'output_mb': 256})
            config = json.loads((out/'config.json').read_text())
            self.assertEqual(config['threads'], 4)
            for name in config['variants']:
                Image.new('RGB', (640, 640), 'white').save(out/(name+'.png'))
            (out/'native-report.json').write_text(json.dumps({'execution_kind': 'analytic_fixture',
                 'source_blend_sha256': sha256(self.pack/'source.blend'), 'source_unchanged': True,
                 'variants': [{'id': name} for name in config['variants']]}))
            log.write_text('analytic fixture; Blender was not run\n')
            return {'exit_code': 0, 'failure': None, 'execution_kind': 'analytic_fixture'}
        before = (self.pack/'source.blend').read_bytes()
        with patch.object(lookdev, 'run_bounded', side_effect=fixture):
            result = lookdev.compare(self.pack, output)
        self.assertEqual(result['variants'], list(lookdev.VARIANTS))
        self.assertTrue(result['original_pack_unchanged'])
        self.assertEqual((self.pack/'source.blend').read_bytes(), before)
        self.assertTrue((output/'sources/blender_lookdev.py').is_file())
        manifest = json.loads((output/'manifest.json').read_text())
        self.assertEqual(manifest['native']['execution_kind'], 'analytic_fixture')
        selection = ['baseline_native', 'base_matte_no_normal']
        with patch.object(lookdev, 'run_bounded', side_effect=fixture):
            selected = lookdev.compare(self.pack, self.root/'selected', selection)
        self.assertEqual(selected['variants'], selection)
        self.assertFalse((self.root/'selected/portable_only.png').exists())
        with patch.object(lookdev, 'run_bounded', side_effect=fixture):
            clay = lookdev.compare(self.pack, self.root/'clay', ['base_clay'])
        self.assertEqual(clay['variants'], ['base_clay'])
        self.assertTrue((self.root/'clay/base_clay.png').is_file())
        self.assertFalse((self.root/'clay/baseline_native.png').exists())


if __name__ == '__main__':
    unittest.main()
