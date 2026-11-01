"""End-to-end boundary checks with small analytic input; never avatar examples."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from avatar_assets.cli import verify_pack
from avatar_assets.runner import run_bounded
from avatar_assets.safety import artifacts, write_json


class WorkflowTests(unittest.TestCase):
    def test_nested_source_manifest_is_hashed_and_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "sources").mkdir()
            write_json(root / "sources/manifest.json", {"prompt": "Original source prompt"})
            write_json(root / "manifest.json", {"schema_version": 1, "artifacts": artifacts(root)})
            self.assertEqual(verify_pack(root)["artifact_hashes_verified"], 1)
            (root / "sources/manifest.json").write_text('{"prompt":"Changed"}')
            with self.assertRaises(ValueError):
                verify_pack(root)

    def test_extra_or_missing_artifacts_are_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "original.txt").write_text("original")
            write_json(root / "manifest.json", {"artifacts": artifacts(root)})
            (root / "extra.txt").write_text("untracked")
            with self.assertRaises(ValueError):
                verify_pack(root)

    def test_request_captures_prompt_without_claiming_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "request"
            run = subprocess.run([sys.executable, "-m", "avatar_assets", "request", "--subject", "A copper moth",
                                  "--prompt", "A copper moth unfolds its wings", "--states", "idle,unfold",
                                  "--out", str(out)], capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stderr)
            data = json.loads((out / "request.json").read_text())
            self.assertEqual(data["status"], "needs_agent_authoring")
            self.assertEqual(data["provider_calls"], 0)
            self.assertFalse(list(out.glob("*.glb")))

    def test_wall_time_limit_stops_owned_process_and_preserves_handle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = run_bounded([sys.executable, "-c", "import time; time.sleep(20)"],
                                 root / "bound.log", root, timeout=.4)
            self.assertLess(result["elapsed_seconds"], 6)
            self.assertNotEqual(result["exit_code"], 0)
            self.assertEqual(result["failure"], "wall-time budget exceeded")
            self.assertTrue((root / "bound-handle.json").exists())

    def test_resident_memory_limit_stops_large_owned_process(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = run_bounded([sys.executable, "-c", "import time; x=bytearray(64*1024*1024); time.sleep(20)"],
                                 root / "memory.log", root, timeout=10, memory_mb=32)
            self.assertLess(result["elapsed_seconds"], 6)
            self.assertEqual(result["failure"], "resident-memory budget exceeded")

    def test_fast_output_overrun_is_rejected_at_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = run_bounded([sys.executable, "-c", "import sys; from pathlib import Path; Path(sys.argv[1]).write_bytes(b'x'*2097152)",
                                  str(root / "oversized.bin")], root / "fast.log", root, output_mb=1)
            self.assertEqual(result["failure"], "output byte budget exceeded")

    def test_monitor_exception_does_not_orphan_owned_process(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch("avatar_assets.runner.resident_bytes", side_effect=RuntimeError("monitor failed")):
                with self.assertRaises(RuntimeError):
                    run_bounded([sys.executable, "-c", "import time; time.sleep(20)"],
                                root / "exception.log", root)
            pid = json.loads((root / "exception-handle.json").read_text())["pid"]
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_consumer_bundle_verifies_hashes_without_claiming_rig_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "model.glb").write_bytes(b"analytic placeholder, deliberately not a model")
            write_json(root / "manifest.json", {"kind": "weft_authored_model_and_mapping", "artifacts": artifacts(root)})
            result = verify_pack(root)
            self.assertEqual(result["artifact_hashes_verified"], 1)
            self.assertNotIn("rig", result)
            self.assertIn("Hashes only", result["rig_verification"])

    def test_reskin_preserves_source_prompt_and_records_actual_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scene_path = Path(__file__).resolve().parents[1] / "examples/scenes/articulated-lamp-v2.json"
            original = json.loads(scene_path.read_text())
            changes = {"body": {"color": [.1, .2, .8, 1], "roughness": .5, "metallic": .2}}
            write_json(root / "changes.json", changes)
            run = subprocess.run([sys.executable, "-m", "avatar_assets", "reskin", str(scene_path),
                                  str(root / "changes.json"), "--out", str(root / "variant"),
                                  "--subject", "Blue lamp", "--prompt", "Change only the body paint to blue"],
                                 capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stderr)
            variant = json.loads((root / "variant/scene.json").read_text())
            self.assertEqual(variant["source_prompt"], original["prompt"])
            self.assertEqual(variant["prompt"], "Change only the body paint to blue")
            self.assertEqual(variant["subject"], "Blue lamp")
            for name in ("parts", "bones", "states"):
                self.assertEqual(variant[name], original[name])
            self.assertEqual(variant["materials"]["body"], changes["body"])


if __name__ == "__main__":
    unittest.main()
