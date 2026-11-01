"""Small moving-square fixtures validate packaging; they are not avatar examples."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from avatar_assets.export2d import export_2d
from avatar_assets.packaging import package_frames
from avatar_assets.safety import read_json


class PackagingTests(unittest.TestCase):
    def inputs(self, root, moving=True):
        folder = root / "frames/test"
        folder.mkdir(parents=True)
        Image.new("RGB", (32, 32), "white").save(root / "portrait.png")
        for i in range(25):
            im = Image.new("RGB", (32, 32), "white")
            x = i % 24 if moving else 0
            ImageDraw.Draw(im).rectangle((x, 10, x+5, 15), fill="black")
            im.save(folder / f"{i:04d}.png")
        scene = {"states": [{"id": "test", "meaning": "Analytic moving-square fixture", "duration": 1, "loop": False}]}
        return scene, {"portrait": 32, "size": 32, "seconds": 1, "samples": 1}

    def test_streams_real_motion_and_static_fallback_without_claiming_loop(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, out = root / "source", root / "output"
            out.mkdir()
            scene, spec = self.inputs(source)
            result = package_frames(out, source, scene, spec)
            clip = result["clips"][0]
            self.assertEqual(clip["inspection"]["frames"], 24)
            self.assertTrue(clip["inspection"]["motion_detected"])
            self.assertEqual(clip["fallback_inspection"]["frames"], 1)
            self.assertFalse(clip["fallback_inspection"]["has_transparency"])
            self.assertIn("not_certified", clip["loop_declaration"])

    def test_repeated_stills_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, out = root / "source", root / "output"
            out.mkdir()
            scene, spec = self.inputs(source, moving=False)
            with self.assertRaisesRegex(ValueError, "Repeated stills"):
                package_frames(out, source, scene, spec)
            self.assertFalse((out / "packaging-result.json").exists())

    def test_missing_or_wrong_sized_frames_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, out = root / "source", root / "output"
            out.mkdir()
            scene, spec = self.inputs(source)
            (source / "frames/test/0012.png").rename(source / "frames/test/0050.png")
            with self.assertRaisesRegex(ValueError, "Noncontiguous"):
                package_frames(out, source, scene, spec)

    def test_postrender_failure_preserves_failed_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "attempt"
            def fail(*args):
                out.mkdir()
                (out / "partial.txt").write_text("Preserve this failed attempt")
                raise ValueError("Encoder failure")
            with patch("avatar_assets.export2d._export_2d", side_effect=fail):
                with self.assertRaisesRegex(ValueError, "Encoder failure"):
                    export_2d("unused", out)
            manifest = read_json(out / "manifest.json")
            self.assertEqual(manifest["status"], "failed")
            self.assertTrue(manifest["rejected_attempts"][0]["preserved"])
            self.assertEqual(manifest["artifacts"][0]["path"], "partial.txt")


if __name__ == "__main__":
    unittest.main()
