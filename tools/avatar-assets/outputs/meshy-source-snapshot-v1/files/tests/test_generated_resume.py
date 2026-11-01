"""Analytic PNG fixtures exercise continuation integrity without rendering a character."""

from pathlib import Path
import tempfile
import unittest

from PIL import Image

from avatar_assets.export2d import reuse_render_frames
from avatar_assets.safety import artifacts, sha256, write_json


class GeneratedResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name)
        self.prior, self.root, self.out = [base/name for name in ('prior', 'source', 'next')]
        for path in (self.prior/'frames/bend', self.root, self.out/'sources'):
            path.mkdir(parents=True)
        write_json(self.root/'manifest.json', {'kind': 'analytic-not-a-model'})
        (self.root/'source.blend').write_bytes(b'Analytic placeholder, never executed')
        self.spec = {'size':16,'portrait':16,'seconds':4,'samples':8}
        scene = {'schema_version':1,'kind':'generated_render','id':'analytic','subject':'Fixture','style':'Analytic',
                 'states':[{'id':'bend','meaning':'Test','duration':4,'loop':False}]}
        write_json(self.out/'sources/scene.json',scene)
        write_json(self.prior/'job.json',{'failure':'wall-time budget exceeded','process_exit_verified':True})
        write_json(self.prior/'resume-input-receipt.json',{'parent_manifest_sha256':sha256(self.root/'manifest.json'),
                   'source_blend_sha256':sha256(self.root/'source.blend'),'scene_sha256':sha256(self.out/'sources/scene.json'),
                   'spec':self.spec})
        Image.new('RGB',(16,16),'blue').save(self.prior/'frames/bend/0000.png')
        Image.new('RGB',(16,16),'white').save(self.prior/'portrait.png')

    def tearDown(self):
        self.temp.cleanup()

    def seal(self):
        write_json(self.prior/'manifest.json',{'status':'failed','artifacts':artifacts(self.prior)})

    def test_verified_frames_are_copied_without_relabeling_missing_frames(self):
        self.seal()
        reuse_render_frames(self.prior,self.root,self.out,self.spec)
        self.assertEqual((self.out/'frames/bend/0000.png').read_bytes(),(self.prior/'frames/bend/0000.png').read_bytes())
        self.assertFalse((self.out/'frames/bend/0001.png').exists())

    def test_changed_parent_cannot_reuse_other_geometry_frames(self):
        self.seal()
        (self.root/'source.blend').write_bytes(b'Changed')
        with self.assertRaisesRegex(ValueError,'source or settings differ'):
            reuse_render_frames(self.prior,self.root,self.out,self.spec)

    def test_incomplete_png_is_retained_but_not_used_as_a_completed_frame(self):
        (self.prior/'frames/bend/0001.png').write_bytes(b'partial PNG')
        self.seal()
        reuse_render_frames(self.prior,self.root,self.out,self.spec)
        self.assertTrue((self.prior/'frames/bend/0001.png').exists())
        self.assertFalse((self.out/'frames/bend/0001.png').exists())

    def test_a_second_continuation_is_rejected(self):
        write_json(self.prior/'reused-frames.json',{'paths':[]})
        self.seal()
        with self.assertRaisesRegex(ValueError,'Only one continuation'):
            reuse_render_frames(self.prior,self.root,self.out,self.spec)


if __name__ == '__main__':
    unittest.main()
