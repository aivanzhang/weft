"""Byte-preserving distribution tests; fixture media bytes are analytic, not rendered assets."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from avatar_assets.cli import verify_pack
from avatar_assets.safety import artifacts, write_json
from scripts import package_portable_media as portable


class PortableMediaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / "analytic-source"
        self.source.mkdir()
        payloads = {
            "portrait.webp": b"ANALYTIC PORTRAIT; NOT AN IMAGE",
            "portrait.png": b"ANALYTIC PORTRAIT PNG",
            "animations/idle.mp4": b"ANALYTIC MOTION; NOT A VIDEO",
            "fallbacks/idle.png": b"ANALYTIC STATIC FALLBACK",
            "frames/idle/0000.png": b"ANALYTIC NATIVE FRAME",
            "render.log": b"ANALYTIC RENDER LOG",
            "sources/prompt.txt": b"An original test subject; no provider call.\n",
            "packaging.log": b"",
        }
        for name, content in payloads.items():
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        self.manifest = {
            "schema_version": 1, "kind": "rendered_2d_avatar", "status": "analytic_fixture",
            "id": "test-subject", "parent_3d_manifest_sha256": "a" * 64,
            "origin": "Analytic fixture; no genuine image, model or video generation",
            "clips": [{"id": "idle", "path": "animations/idle.mp4", "fallback": "fallbacks/idle.png",
                       "loop_declaration": "not_certified", "meaning": "Analytic motion test"}],
            "artifacts": artifacts(self.source),
        }
        write_json(self.source / "manifest.json", self.manifest)

    def replace_manifest(self):
        (self.source / "manifest.json").write_text(json.dumps(self.manifest, indent=1) + "\n")

    def assert_rejected(self, message, limits=None, target=None):
        target = target or self.base / "rejected-output"
        with self.assertRaisesRegex(ValueError, message):
            portable.package_portable_media(self.source, target, limits or portable.PortableLimits())
        self.assertFalse(target.exists())

    def test_preserves_bytes_provenance_parent_binding_and_all_static_fallbacks(self):
        original = (self.source / "manifest.json").read_bytes()
        source_hashes = {item["path"]: item["sha256"] for item in self.manifest["artifacts"]}
        output = self.base / "portable-one"
        result = portable.package_portable_media(self.source, output)
        self.assertEqual((output / "source-manifest.json").read_bytes(), original)
        self.assertEqual((self.source / "manifest.json").read_bytes(), original)
        self.assertEqual(result["parent_3d_manifest_sha256"], self.manifest["parent_3d_manifest_sha256"])
        self.assertEqual(result["clips"], self.manifest["clips"])
        self.assertFalse((output / "frames").exists())
        self.assertFalse((output / "render.log").exists())
        self.assertTrue((self.source / "frames/idle/0000.png").is_file())
        self.assertTrue((self.source / "render.log").is_file())
        self.assertEqual({item["path"] for item in result["distribution"]["omitted_artifacts"]},
                         {"frames/idle/0000.png", "render.log"})
        for item in result["artifacts"]:
            if item["path"] != "source-manifest.json":
                self.assertEqual(item["sha256"], source_hashes[item["path"]])
        self.assertEqual(verify_pack(output)["artifact_hashes_verified"], 7)

    def test_manifests_are_reproducible_across_output_folders(self):
        for name in ("one", "two"):
            portable.package_portable_media(self.source, self.base / name)
        self.assertEqual((self.base / "one/manifest.json").read_bytes(), (self.base / "two/manifest.json").read_bytes())

    def test_changed_retained_asset_rejected_before_output(self):
        path = self.source / "portrait.webp"
        path.write_bytes(b"X" * path.stat().st_size)
        self.assert_rejected("hash changed")

    def test_changed_omitted_frame_also_rejected(self):
        path = self.source / "frames/idle/0000.png"
        path.write_bytes(b"X" * path.stat().st_size)
        self.assert_rejected("hash changed")

    def test_file_count_and_byte_caps(self):
        cases = [
            ({"source_entries": 2}, "artifact count"),
            ({"file_bytes": 10}, "Artifact byte"),
            ({"source_bytes": 30}, "aggregate byte"),
            ({"output_bytes": 40}, "aggregate byte"),
            ({"retained_entries": 2}, "Portable artifact count"),
            ({"manifest_bytes": 30}, "manifest byte"),
            ({"tree_entries": 2}, "directory entry"),
        ]
        for index, (changes, error) in enumerate(cases):
            with self.subTest(changes=changes):
                self.assert_rejected(error, replace(portable.PortableLimits(), **changes), self.base / f"bad-{index}")

    def test_traversal_absolute_url_and_noncanonical_paths(self):
        for index, path in enumerate(("../outside", "/outside", "sources/../x", "sources//x", "./x", "a\\b", "https://x")):
            with self.subTest(path=path):
                self.manifest["artifacts"][0]["path"] = path
                self.replace_manifest()
                self.assert_rejected("Unsafe artifact path", target=self.base / f"bad-path-{index}")

    def test_duplicate_case_collision_and_reserved_paths(self):
        for index, path in enumerate(("portrait.webp", "PORTRAIT.WEBP", "SOURCE-MANIFEST.JSON")):
            with self.subTest(path=path):
                self.manifest["artifacts"][0]["path"] = path
                self.replace_manifest()
                self.assert_rejected("Duplicate, reserved, or case-colliding", target=self.base / f"bad-name-{index}")

    def test_symlink_component_and_leaf_rejected(self):
        outside = self.base / "outside"
        outside.mkdir()
        target = self.source / "animations/idle.mp4"
        target.rename(outside / "idle.mp4")
        target.symlink_to(outside / "idle.mp4")
        self.assert_rejected("symlink or special file")

    def test_parent_directory_symlink_rejected(self):
        outside = self.base / "outside"
        (self.source / "animations").rename(outside)
        (self.source / "animations").symlink_to(outside, target_is_directory=True)
        self.assert_rejected("symlink or special file")

    def test_unrecorded_file_rejected(self):
        (self.source / "unrecorded.txt").write_text("not inventoried")
        self.assert_rejected("Unrecorded or missing")

    def test_special_file_rejected_without_opening(self):
        os.mkfifo(self.source / "pipe")
        self.assert_rejected("symlink or special file")

    def test_existing_or_nested_outputs_are_not_overwritten(self):
        output = self.base / "existing"
        output.mkdir()
        (output / "keep.txt").write_text("keep")
        with self.assertRaisesRegex(ValueError, "Output already exists"):
            portable.package_portable_media(self.source, output)
        self.assertEqual((output / "keep.txt").read_text(), "keep")
        self.assert_rejected("inside the source", target=self.source / "nested")

    def test_wrong_pack_or_missing_fallback_rejected(self):
        self.manifest["kind"] = "authored_3d_avatar"
        self.replace_manifest()
        self.assert_rejected("rendered_2d_avatar")
        self.manifest["kind"] = "rendered_2d_avatar"
        self.manifest["clips"][0]["fallback"] = "frames/idle/0000.png"
        self.replace_manifest()
        self.assert_rejected("fallback is missing or excluded")

    def test_source_changed_between_verification_and_copy_preserves_failed_attempt(self):
        transfer = portable._transfer

        def changed(root_fd, item, limits, check, destination=None):
            if destination is not None:
                path = self.source / item["path"]
                path.write_bytes(b"X" * path.stat().st_size)
            return transfer(root_fd, item, limits, check, destination)

        output = self.base / "changed-during-copy"
        with patch.object(portable, "_transfer", changed), self.assertRaisesRegex(ValueError, "hash changed"):
            portable.package_portable_media(self.source, output)
        failed = json.loads((output / "manifest.json").read_text())
        self.assertEqual(failed["status"], "failed")

    def test_invalid_and_excessive_limits_rejected(self):
        for changes in ({"source_entries": 1.5}, {"source_bytes": 0}, {"output_bytes": 128*1024*1024}):
            with self.subTest(changes=changes):
                self.assert_rejected("Limits must be positive", replace(portable.PortableLimits(), **changes))

    def test_runtime_bound_is_enforced_before_output(self):
        with patch.object(portable.time, "monotonic", side_effect=[0, 100]):
            self.assert_rejected("runtime limit")


if __name__ == "__main__":
    unittest.main()
