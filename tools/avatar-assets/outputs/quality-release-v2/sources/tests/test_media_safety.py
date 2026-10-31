"""Small analytic fixtures test validation; they are not generated-avatar examples."""

import io
import stat
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from avatar_assets.archive import inspect_archive
from avatar_assets.media import encode_video, fit, image_info, video_info
from avatar_assets.safety import LIMITS, contained, local_file, read_json, sha256, write_json


def png_bytes(mode="RGB", color="navy", size=(32, 32)):
    handle = io.BytesIO()
    Image.new(mode, size, color).save(handle, format="PNG")
    return handle.getvalue()


class LocalFixtureCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="avatar-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def archive(self, members, compression=zipfile.ZIP_STORED):
        source = self.root / "input.zip"
        with zipfile.ZipFile(source, "w", compression=compression) as archive:
            for name, content in members:
                archive.writestr(name, content)
        return source

    def rejected_archive(self, source, message):
        output = self.root / "out"
        before = sha256(source)
        with self.assertRaisesRegex(ValueError, message):
            inspect_archive(source, output)
        self.assertFalse(output.exists(), "Unsafe archive must fail before extraction")
        self.assertEqual(before, sha256(source))


class ArchiveSafetyTests(LocalFixtureCase):
    def test_reference_bytes_and_artifact_hashes_survive_inspection(self):
        data = png_bytes()
        source = self.archive([("portrait.png", data)])
        before = sha256(source)
        out = self.root / "verified"
        report = inspect_archive(source, out, before)
        self.assertEqual(before, sha256(source))
        self.assertEqual(data, (out / "originals/portrait.png").read_bytes())
        self.assertEqual(before, report["source_sha256"])
        self.assertEqual(1, len(report["media"]))
        self.assertEqual("pending", report["visual_review"])
        for artifact in report["artifacts"]:
            self.assertEqual(artifact["sha256"], sha256(out / artifact["path"]))

    def test_wrong_expected_hash_cannot_create_output(self):
        source = self.archive([("portrait.png", png_bytes())])
        output = self.root / "out"
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            inspect_archive(source, output, "0" * 64)
        self.assertFalse(output.exists())

    def test_traversal_and_absolute_paths_rejected_before_writes(self):
        for name in ["../portrait.png", "/portrait.png", "..\\portrait.png"]:
            with self.subTest(name=name):
                self.rejected_archive(self.archive([(name, png_bytes())]), "Unsafe archive path")
        self.assertFalse((self.root.parent / "portrait.png").exists())

    def test_nested_paths_are_explicitly_unsupported(self):
        self.rejected_archive(self.archive([("nested/portrait.png", png_bytes())]), "flat media")

    def test_symlink_and_special_entries_rejected(self):
        for kind in [stat.S_IFLNK, stat.S_IFIFO]:
            with self.subTest(kind=kind):
                info = zipfile.ZipInfo("portrait.png")
                info.create_system = 3
                info.external_attr = (kind | 0o600) << 16
                self.rejected_archive(self.archive([(info, b"target.png")]), "link or special")

    def test_case_collisions_rejected_on_any_filesystem(self):
        source = self.archive([("Avatar.PNG", png_bytes()), ("avatar.png", png_bytes())])
        self.rejected_archive(source, "Duplicate archive filename")

    def test_nonmedia_members_rejected(self):
        self.rejected_archive(self.archive([("run.py", b"raise RuntimeError('not data')")]), "nonmedia")

    def test_empty_and_excess_entry_archives_rejected(self):
        self.rejected_archive(self.archive([]), "entry limit")
        with patch.dict(LIMITS, {"archive_entries": 2}):
            source = self.archive([(f"p{i}.png", png_bytes()) for i in range(3)])
            self.rejected_archive(source, "entry limit")

    def test_high_compression_ratio_rejected_without_decoding(self):
        source = self.archive([("bomb.png", b"x" * 262_144)], zipfile.ZIP_DEFLATED)
        self.rejected_archive(source, "compression ratio")

    def test_declared_oversized_member_rejected_without_allocating_it(self):
        source = self.archive([("portrait.png", png_bytes())])
        raw = bytearray(source.read_bytes())
        central = raw.index(b"PK\x01\x02")
        struct.pack_into("<I", raw, central + 24, LIMITS["input_bytes"] + 1)
        source.write_bytes(raw)
        self.rejected_archive(source, "member too large")

    def test_aggregate_expansion_limit_rejected_before_extraction(self):
        data = png_bytes()
        source = self.archive([("one.png", data), ("two.png", data)])
        with patch.dict(LIMITS, {"archive_expanded_bytes": len(data) * 2 - 1}):
            self.rejected_archive(source, "expanded byte limit")

    def test_preexisting_output_is_never_replaced(self):
        source = self.archive([("portrait.png", png_bytes())])
        output = self.root / "out"
        output.mkdir()
        sentinel = output / "sentinel"
        sentinel.write_text("preserve")
        with self.assertRaisesRegex(ValueError, "already exists"):
            inspect_archive(source, output)
        self.assertEqual("preserve", sentinel.read_text())
        self.assertEqual([sentinel], list(output.iterdir()))


class ImageSafetyTests(LocalFixtureCase):
    def test_alpha_channel_is_not_confused_with_actual_transparency(self):
        opaque, transparent = self.root / "opaque.png", self.root / "transparent.png"
        opaque.write_bytes(png_bytes("RGBA", (20, 30, 40, 255)))
        transparent.write_bytes(png_bytes("RGBA", (20, 30, 40, 100)))
        self.assertFalse(image_info(opaque)["has_transparency"])
        self.assertTrue(image_info(transparent)["has_transparency"])
        self.assertEqual([100, 100], image_info(transparent)["alpha_extrema"])

    def test_oversized_image_rejected_before_pixel_load(self):
        path = self.root / "portrait.png"
        path.write_bytes(png_bytes(size=(32, 32)))
        with patch.dict(LIMITS, {"pixels": 512}):
            with self.assertRaisesRegex(ValueError, "pixel limit"):
                image_info(path)

    def test_multiframe_webp_is_not_a_static_fallback(self):
        path = self.root / "animated.webp"
        Image.new("RGB", (16, 16), "red").save(
            path, save_all=True, append_images=[Image.new("RGB", (16, 16), "blue")], duration=100
        )
        with self.assertRaisesRegex(ValueError, "still image"):
            image_info(path)

    def test_silent_aspect_ratio_crop_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Aspect ratio differs"):
            fit(Image.new("RGB", (32, 16)), 32, 32, "white")

    def test_preserved_alpha_and_explicit_opaque_composite(self):
        source = Image.new("RGBA", (16, 16), (255, 0, 0, 0))
        self.assertEqual((255, 0, 0, 0), fit(source, 16, 16, "transparent").getpixel((0, 0)))
        self.assertEqual((255, 255, 255), fit(source, 16, 16, "white").getpixel((0, 0)))


class VideoEvidenceTests(LocalFixtureCase):
    def make_video(self, moving=True, count=12):
        destination = self.root / ("motion.mp4" if moving else "repeated-still.mp4")
        def frames():
            for index in range(count):
                image = Image.new("RGB", (64, 64), "white")
                x = index * 3 if moving else 10
                ImageDraw.Draw(image).rectangle((x, 15, x + 14, 40), fill="black")
                yield image
        encode_video(frames(), destination, 24, 64, 64, "white")
        return destination

    def test_real_decode_counts_duration_motion_and_unverified_loop(self):
        path = self.make_video()
        before = sha256(path)
        contact = self.root / "contact.png"
        result = video_info(path, contact)
        self.assertEqual((64, 64, 12), (result["width"], result["height"], result["frames"]))
        self.assertEqual(24, result["fps"])
        self.assertAlmostEqual(0.5, result["duration_seconds"], places=5)
        self.assertTrue(result["motion_detected"])
        self.assertFalse(result["has_transparency"])
        self.assertFalse(result["audio"])
        self.assertEqual("unverified", result["loop"])
        self.assertGreater(result["boundary_mae"], 0)
        self.assertEqual(before, sha256(path))
        self.assertTrue(contact.is_file())
        self.assertTrue(any(0 < i < 11 for i in result["contact_sample_frames"]))
        self.assertLessEqual(len(result["contact_sample_frames"]), 12)

    def test_repeated_still_is_not_detected_as_motion(self):
        result = video_info(self.make_video(moving=False))
        self.assertFalse(result["motion_detected"])
        self.assertEqual("unverified", result["loop"])

    def test_input_pixel_and_frame_limits(self):
        path = self.make_video()
        with patch.dict(LIMITS, {"video_pixels": 2048}):
            with self.assertRaisesRegex(ValueError, "pixel limit"):
                video_info(path)
        with patch.dict(LIMITS, {"frames": 6}):
            with self.assertRaisesRegex(ValueError, "frame limit"):
                video_info(path)

    def test_input_duration_limit(self):
        path = self.make_video()
        with patch.dict(LIMITS, {"seconds": 0.25}):
            with self.assertRaisesRegex(ValueError, "duration limit"):
                video_info(path)

    def test_mp4_transparency_request_rejected_without_file(self):
        destination = self.root / "alpha.mp4"
        with self.assertRaisesRegex(ValueError, "cannot preserve alpha"):
            encode_video([], destination, 24, 64, 64, "transparent")
        self.assertFalse(destination.exists())


class ManifestSafetyTests(LocalFixtureCase):
    def test_duplicate_and_nonfinite_json_rejected(self):
        path = self.root / "input.json"
        for data in ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}']:
            with self.subTest(data=data):
                path.write_text(data)
                with self.assertRaises(ValueError):
                    read_json(path)

    def test_existing_manifest_not_overwritten(self):
        path = self.root / "manifest.json"
        write_json(path, {"schema_version": 1, "verified": False})
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            write_json(path, {"verified": True})
        self.assertEqual(before, path.read_bytes())

    def test_manifest_relative_path_cannot_escape(self):
        for path in ["../escape", "/tmp/escape"]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                contained(self.root, path)
        self.assertEqual(self.root / "asset.png", contained(self.root, "asset.png"))

    def test_linked_artifact_and_oversized_file_rejected(self):
        source = self.root / "source"
        source.write_bytes(b"12345")
        link = self.root / "link"
        link.symlink_to(source)
        with self.assertRaisesRegex(ValueError, "regular local file"):
            local_file(link)
        with self.assertRaisesRegex(ValueError, "too large"):
            local_file(source, max_bytes=4)


if __name__ == "__main__":
    unittest.main()
