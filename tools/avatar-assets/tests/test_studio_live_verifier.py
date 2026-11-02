"""Analytic registry/receipt fixtures only; no server, agent or provider execution."""

from contextlib import closing
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from avatar_assets import meshy, pipeline
from avatar_assets.studio_jobs import _digest
from avatar_assets.studio_worker import MEDIA_SPECS, PIPELINE_LIMITS


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("studio_live_verifier", ROOT / "scripts/verify_studio_live.py")
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)
JOB = "a1" * 12


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StudioInspectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.state = self.root / ".local/studio"
        self.output = self.root / "outputs" / ("studio-" + JOB)
        self.job = self.state / "jobs" / JOB
        self.database = self.state / "jobs.sqlite3"
        self.sources = {name: ROOT / "avatar_assets" / name for name in ("meshy.py", "meshy_access.py")}
        self.addCleanup(patch.stopall)
        patch.object(verifier, "ROOT", self.root).start()
        patch.object(pipeline, "_pipeline_sources", return_value=self.sources).start()
        # Admission must not acquire execution ownership or touch account credentials.
        self.blocked = [patch(target, side_effect=AssertionError("Inspection attempted execution")).start()
                        for target in ("avatar_assets.studio_jobs.StudioJobs.__init__", "avatar_assets.pipeline.Pipeline.__init__",
                                       "avatar_assets.meshy_access.Client.__init__", "avatar_assets.meshy_access.Ledger.__init__")]
        self.request = {"prompt": "A soft panda with a green scarf", "style": "Source-colored plush fur",
                        "custom_actions": ["Wave the left paw."], "preset": "preview", "reference": None,
                        "reuse_model": None}
        self.plan = {"status": "ready", "reason": "", "subject": "A panda with a green scarf",
                     "style": self.request["style"], "generation_prompt": self.request["prompt"], "height_meters": 1.2,
                     "states": [{"id": "wave", "meaning": "Wave the left paw.", "user_request": "Wave the left paw.",
                                 "roles": [{"id": "paw", "description": "Left paw"}],
                                 "requirements": [{"id": "wave", "kind": "articulation", "role": "paw",
                                                   "description": "Left paw moves relative to the body"}]}],
                     "appearance_checks": ["Keep the scarf green"], "limitations": ["Unseen surfaces are authored"]}
        self.make_fixture()

    def pins(self, target):
        result = []
        for name, source in self.sources.items():
            saved = target / name
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, saved)
            result.append({"path": name, "bytes": saved.stat().st_size, "sha256": digest(saved)})
        return result

    def make_fixture(self, reuse=False):
        if reuse:
            original = self.root / "outputs/prior-model/model.glb"
            original.parent.mkdir(parents=True, exist_ok=True)
            original.write_bytes(b"Analytic pinned bytes; this inspector does not claim GLB validation")
            info = original.stat()
            self.request["reuse_model"] = {"path": str(original), "bytes": info.st_size,
                                            "device": info.st_dev, "inode": info.st_ino, "mtime_ns": info.st_mtime_ns}
        self.prepared = {"schema_version": 1, "id": self.output.name, **deepcopy(self.request), "states": [],
                         "media_spec": MEDIA_SPECS["preview"], "max_credits": 0 if reuse else 35,
                         **PIPELINE_LIMITS, "created": "2026-11-02T10:00:00-08:00"}
        if reuse:
            self.prepared["max_agent_calls"] = 10  # Historical jobs remain readable.
            original = Path(self.request["reuse_model"]["path"])
            self.prepared["reuse_model"] = {"path": "inputs/source-model.glb", "original_path": str(original),
                                            "bytes": original.stat().st_size, "sha256": digest(original)}
            copied = self.output / "inputs/source-model.glb"
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, copied)
        write(self.root / ".local/studio-registry.json", {"schema_version": 1, "state_dir": str(self.state)})
        write(self.job / "request.json", self.request)
        write(self.output / "request.json", self.prepared)
        write(self.output / "plan.json", self.plan)
        self.generation = None
        if not reuse:
            write(self.output / "input-receipt.json", {"request_sha256": digest(self.output / "request.json"),
                                                        "sources": self.pins(self.output / "sources")})
            write(self.job / "pipeline-request.json", {"schema_version": 1, "studio_request_digest": _digest(self.request),
                                                        "pipeline_request_sha256": digest(self.output / "request.json")})
            self.config = {"schema_version": 1, "id": self.output.name + "-mesh", "subject": self.plan["subject"],
                           "style": self.plan["style"], "prompt": self.plan["generation_prompt"], "height_meters": 1.2,
                           "rig": "none", "states": [], "max_credits": 35, "rights": "Analytic fixture; not generated assets",
                           "generation": {"mode": "text", "geometry_resolution": "4k", "texture_resolution": "4k",
                                          "target_polycount": 35000, "pose_mode": ""}}
            self.generation = {"schema_version": 1, "id": self.config["id"], "config": self.config,
                               "model": meshy.MODEL, "max_credits": 35, "estimated_credits": 35, "estimated_paid_calls": 2,
                               "automatic_paid_retries": 0, "reference_reconstruction": False, "usd_cost": None,
                               "sources": self.pins(self.output / "generation/sources")}
            self.save_generation()
        else:
            # Reuse fixtures intentionally lack fresh-only receipts and source inventories.
            generation_path = self.output / "generation/plan.json"
            if generation_path.exists():
                generation_path.unlink()
        self.plan_id = _digest({"request": _digest(self.request), "plan": self.plan, "generation": self.generation})
        self.data = {"plan_id": self.plan_id, "owner_pid": None, "operation": "plan", "invocations": 1,
                     "ended": 1793635200, "provenance": "analytic_fixture"}
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,request TEXT,data TEXT,request_digest TEXT,output TEXT,status TEXT,queued_at REAL)")
            db.execute("CREATE TABLE IF NOT EXISTS active(id INTEGER PRIMARY KEY,job TEXT)")
            db.execute("INSERT OR REPLACE INTO jobs VALUES(?,?,?,?,?,?,?)", (JOB, json.dumps(self.request), json.dumps(self.data),
                       _digest(self.request), self.output.name, "planned" if reuse else "awaiting_authorization", None))

    def save_generation(self):
        write(self.output / "generation-config.json", self.config)
        write(self.output / "generation/inputs/config.json", self.config)
        write(self.output / "generation/plan.json", self.generation)
        write(self.output / "generation-receipt.json", {"plan_sha256": digest(self.output / "generation/plan.json")})

    def update_row(self, **values):
        with closing(sqlite3.connect(self.database)) as db, db:
            for key, value in values.items():
                db.execute("UPDATE jobs SET " + key + "=? WHERE id=?", (value, JOB))

    def test_fresh_admission_is_read_only_and_grants_no_authority(self):
        before = {str(p.relative_to(self.root)): digest(p) for p in self.root.rglob("*") if p.is_file()}
        result = verifier.inspect(JOB)
        self.assertEqual(result["plan_id"], self.plan_id)
        self.assertEqual(result["generation_mode"], "new_meshy_generation")
        self.assertIsNone(result["source_reuse"])
        self.assertEqual(result["source_generation"]["estimated_credits"], 35)
        self.assertEqual(result["source_generation"]["estimated_paid_calls"], 2)
        self.assertEqual(result["source_generation"]["maximum_agent_calls"], 5)
        self.assertFalse(result["source_generation"]["authorization_checked"])
        self.assertEqual(result["source_generation"]["execution_gate"], "canonical_ledger_and_explicit_start")
        self.assertEqual(result["source_generation"]["plan_sha256"], digest(self.output / "generation/plan.json"))
        self.assertEqual(result["owner"]["queued_jobs"], [])
        self.assertEqual(result["request"]["custom_actions"], self.request["custom_actions"])
        self.assertEqual(before, {str(p.relative_to(self.root)): digest(p) for p in self.root.rglob("*") if p.is_file()})
        for blocked in self.blocked:
            blocked.assert_not_called()

    def test_historical_reuse_keeps_source_shape_and_prior_limits(self):
        self.make_fixture(reuse=True)
        result = verifier.inspect(JOB)
        self.assertEqual(result["generation_mode"], "reused_generated_source")
        self.assertIsNone(result["source_generation"])
        self.assertEqual(result["source_reuse"], {key: self.prepared["reuse_model"][key] for key in ("path", "bytes", "sha256")})

    def test_canonical_digest_and_saved_request_mismatches_reject(self):
        self.update_row(request_digest="f" * 64)
        with self.assertRaisesRegex(ValueError, "digest differs"):
            verifier.inspect(JOB)
        self.update_row(request_digest=_digest(self.request))
        write(self.job / "request.json", {**self.request, "prompt": "Different"})
        with self.assertRaisesRegex(ValueError, "saved request differs"):
            verifier.inspect(JOB)

    def test_prepared_identity_actions_limits_and_selection_are_bound(self):
        for field, value in (("id", "other"), ("prompt", "Different"), ("style", "Different"),
                             ("custom_actions", ["Jump"]), ("preset", "draft"), ("states", ["wave"]),
                             ("max_credits", 500), ("max_agent_calls", 10), ("agent_timeout", 600),
                             ("reference", {}), ("reuse_model", {})):
            with self.subTest(field=field):
                write(self.output / "request.json", {**self.prepared, field: value})
                with self.assertRaises(ValueError):
                    verifier.inspect(JOB)
        write(self.output / "request.json", self.prepared)

    def test_input_and_canonical_pipeline_request_hashes_are_required(self):
        for path, field in ((self.output / "input-receipt.json", "request_sha256"),
                            (self.job / "pipeline-request.json", "pipeline_request_sha256"),
                            (self.job / "pipeline-request.json", "studio_request_digest")):
            old = verifier.read_json(path)
            write(path, {**old, field: "0" * 64})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "hash differs|not bound"):
                verifier.inspect(JOB)
            write(path, old)

    def test_generation_config_plan_and_receipt_mismatches_reject(self):
        for relative in ("generation-config.json", "generation/inputs/config.json", "generation/plan.json"):
            path = self.output / relative
            old = verifier.read_json(path)
            modified = deepcopy(old)
            (modified["config"] if relative.endswith("plan.json") else modified)["prompt"] = "Changed creature"
            write(path, modified)
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                verifier.inspect(JOB)
            write(path, old)
        write(self.output / "generation-receipt.json", {"plan_sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "Generation plan hash differs"):
            verifier.inspect(JOB)

    def test_self_consistent_generation_cannot_change_planner_or_profile(self):
        for field, value in (("subject", "Robot"), ("style", "Metal"), ("prompt", "Robot"),
                             ("height_meters", 2), ("rig", "humanoid"), ("max_credits", 40)):
            old = self.config[field]
            self.config[field] = value
            self.save_generation()
            with self.subTest(field=field), self.assertRaises(ValueError):
                verifier.inspect(JOB)
            self.config[field] = old
        self.config["generation"]["target_polycount"] = 40000
        self.save_generation()
        with self.assertRaisesRegex(ValueError, "generation"):
            verifier.inspect(JOB)

    def test_changed_model_estimate_retry_policy_and_frozen_plan_reject(self):
        for field, value in (("model", "other-model"), ("estimated_credits", 34), ("estimated_paid_calls", 1),
                             ("automatic_paid_retries", 1), ("reference_reconstruction", True)):
            old = self.generation[field]
            self.generation[field] = value
            self.save_generation()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "plan field"):
                verifier.inspect(JOB)
            self.generation[field] = old
        self.save_generation()
        self.update_row(data=json.dumps({**self.data, "expected_plan_id": "b" * 64}))
        with self.assertRaisesRegex(ValueError, "plan ID differs"):
            verifier.inspect(JOB)

    def test_planner_custom_action_binding_is_validated(self):
        self.plan["states"][0]["user_request"] = "Jump instead"
        write(self.output / "plan.json", self.plan)
        with self.assertRaises(ValueError):
            verifier.inspect(JOB)

    def test_changed_copied_and_current_source_pins_reject(self):
        for relative in ("sources/meshy.py", "generation/sources/meshy.py"):
            path = self.output / relative
            original = path.read_bytes()
            path.write_bytes(original + b"\n# changed fixture\n")
            with self.subTest(relative=relative), self.assertRaisesRegex(ValueError, "source pin changed"):
                verifier.inspect(JOB)
            path.write_bytes(original)
        with patch.object(pipeline, "_pipeline_sources", return_value={**self.sources, "new.py": ROOT / "avatar_assets/pipeline.py"}):
            with self.assertRaisesRegex(ValueError, "lacks current source pins"):
                verifier.inspect(JOB)

    def test_source_inventory_paths_and_duplicate_pins_reject(self):
        path = self.output / "input-receipt.json"
        original = verifier.read_json(path)
        for name in ("../meshy.py", "/tmp/meshy.py", "scripts/../meshy.py"):
            changed = deepcopy(original)
            changed["sources"][0]["path"] = name
            write(path, changed)
            with self.subTest(name=name), self.assertRaises(ValueError):
                verifier.inspect(JOB)
        changed = deepcopy(original)
        changed["sources"].append(changed["sources"][0])
        write(path, changed)
        with self.assertRaisesRegex(ValueError, "duplicate source pin"):
            verifier.inspect(JOB)

    def test_job_output_and_registry_paths_reject_traversal_or_symlinks(self):
        for identifier in ("../" + JOB, JOB.upper(), "", JOB + "/"):
            with self.subTest(identifier=identifier), self.assertRaisesRegex(ValueError, "job ID"):
                verifier.inspect(identifier)
        for output in ("../escape", "/tmp/escape", "studio-" + "b" * 24):
            self.update_row(output=output)
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "output name"):
                verifier.inspect(JOB)
        self.update_row(output=self.output.name)
        path = self.output / "generation/inputs/config.json"
        path.unlink()
        path.symlink_to(self.output / "generation-config.json")
        with self.assertRaisesRegex(ValueError, "canonical regular file"):
            verifier.inspect(JOB)

    def test_reuse_hash_and_traversal_reject(self):
        self.make_fixture(reuse=True)
        (self.output / "inputs/source-model.glb").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "bytes/hash differ"):
            verifier.inspect(JOB)
        self.prepared["reuse_model"]["path"] = "../outside.glb"
        write(self.output / "request.json", self.prepared)
        with self.assertRaisesRegex(ValueError, "relative artifact path"):
            verifier.inspect(JOB)

    def test_bounded_json_and_database_rows_reject_before_large_parse(self):
        path = self.output / "plan.json"
        for content in (b" " * (verifier.MAX_JSON + 1), b'{"status":"ready","status":"unsupported"}', b'{"value":NaN}'):
            path.write_bytes(content)
            with self.subTest(size=len(content)), self.assertRaises(ValueError):
                verifier.inspect(JOB)
        write(path, self.plan)
        self.update_row(data=" " * (verifier.MAX_JSON + 1))
        with patch.object(verifier, "parse_json", side_effect=AssertionError("Oversized DB row was decoded")):
            # Registry JSON is needed before the DB size check.
            binding = {"schema_version": 1, "state_dir": str(self.state)}
            with patch.object(verifier, "read_json", return_value=binding):
                with self.assertRaisesRegex(ValueError, "row exceeds"):
                    verifier.inspect(JOB)

    def api_fixture(self, reuse):
        from avatar_assets.agent_backend import DEFAULT, pipeline_limits
        self.make_fixture(reuse=reuse)
        settings = {**DEFAULT, "max_calls": 3, "max_repairs": 0}
        self.prepared.update(agent_backend=settings, **pipeline_limits(settings))
        write(self.output / "request.json", self.prepared)
        write(self.output / "input-receipt.json", {"request_sha256": digest(self.output / "request.json"),
                                                  "sources": self.pins(self.output / "sources")})
        write(self.job / "pipeline-request.json", {"schema_version": 1, "studio_request_digest": _digest(self.request),
                                                  "pipeline_request_sha256": digest(self.output / "request.json")})

    def test_api_reuse_three_calls_zero_repairs_is_inspection_only(self):
        self.api_fixture(reuse=True)
        with patch("avatar_assets.openai_runner.OpenAILedger.__init__", side_effect=AssertionError("No ledger creation")), \
             patch("avatar_assets.openai_runner.read_api_key", side_effect=AssertionError("No key access")):
            result = verifier.inspect(JOB)
        self.assertEqual(result["generation_mode"], "reused_generated_source")
        self.assertEqual(result["agent_backend"]["max_calls"], 3)
        self.assertEqual(result["agent_backend"]["max_repairs"], 0)
        self.assertEqual(result["agent_backend"]["maximum_reservation_microdollars"], 16855920)
        self.assertFalse(result["agent_backend"]["authorization_checked"])
        self.assertFalse((self.root / ".local/openai").exists())

    def test_api_fresh_job_keeps_independent_meshy_credit_estimate(self):
        self.api_fixture(reuse=False)
        result = verifier.inspect(JOB)
        self.assertEqual(result["source_generation"]["maximum_agent_calls"], 3)
        self.assertEqual(result["source_generation"]["estimated_credits"], 35)
        self.assertEqual(result["agent_backend"]["currency"], "USD")

    def test_api_reuse_requires_bound_request_and_exact_limits(self):
        self.api_fixture(reuse=True)
        binding = self.job / "pipeline-request.json"
        original = verifier.read_json(binding)
        write(binding, {**original, "pipeline_request_sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "API request is not bound"):
            verifier.inspect(JOB)
        write(binding, original)
        self.prepared["max_agent_calls"] = 5
        write(self.output / "request.json", self.prepared)
        with self.assertRaisesRegex(ValueError, "max_agent_calls"):
            verifier.inspect(JOB)

    def test_api_runtime_revision_delegates_custody_and_hashes_its_evidence(self):
        self.api_fixture(reuse=True)
        pointer = self.output / "runtime-revision.json"
        manifest = self.root / "outputs/analytic-runtime-revision/manifest.json"
        write(pointer, {"provenance": "analytic_fixture"})
        write(manifest, {"provenance": "analytic_fixture"})
        verify = Mock(return_value={"sources": self.pins(self.output / "analytic-effective-sources"),
                                    "plan": self.plan, "revision_path": manifest})
        with patch.dict(sys.modules, {"avatar_assets.pipeline_recovery": SimpleNamespace(verify_runtime_revision=verify)}):
            result = verifier.inspect(JOB)
        verify.assert_called_once_with(self.output, self.sources)
        evidence = {item["path"]: item["sha256"] for item in result["files"]}
        self.assertEqual(evidence[str(pointer.relative_to(self.root))], digest(pointer))
        self.assertEqual(evidence[str(manifest.relative_to(self.root))], digest(manifest))

    def test_runtime_revision_manifest_must_match_sibling_policy(self):
        self.api_fixture(reuse=True)
        write(self.output / "runtime-revision.json", {"provenance": "analytic_fixture"})
        for manifest in (self.output / "revision/manifest.json", self.root / "outside/manifest.json",
                         self.root / "outputs/revision/not-manifest.json"):
            write(manifest, {"provenance": "analytic_fixture"})
            verify = Mock(return_value={"revision_path": manifest})
            with self.subTest(manifest=manifest), \
                 patch.dict(sys.modules, {"avatar_assets.pipeline_recovery": SimpleNamespace(verify_runtime_revision=verify)}), \
                 self.assertRaisesRegex(ValueError, "sibling output folder"):
                verifier.inspect(JOB)

    def test_runtime_revision_failure_never_falls_back_to_original_pins(self):
        self.api_fixture(reuse=True)
        write(self.output / "runtime-revision.json", {"provenance": "analytic_fixture"})
        verify = Mock(side_effect=ValueError("Revision custody differs"))
        with patch.dict(sys.modules, {"avatar_assets.pipeline_recovery": SimpleNamespace(verify_runtime_revision=verify)}), \
             self.assertRaisesRegex(ValueError, "Revision custody differs"):
            verifier.inspect(JOB)
        verify.assert_called_once()

    def test_sealed_revision_readback_retains_original_snapshot_policy(self):
        self.api_fixture(reuse=True)
        write(self.output / "runtime-revision.json", {"provenance": "analytic_fixture"})
        write(self.output / "delivery.json", {"provenance": "analytic_fixture"})
        write(self.output / "delivery-receipt.json", {"provenance": "analytic_fixture"})
        verify = Mock(side_effect=AssertionError("Historical readback must not require current runtime"))
        with patch.dict(sys.modules, {"avatar_assets.pipeline_recovery": SimpleNamespace(verify_runtime_revision=verify)}):
            result = verifier.inspect(JOB)
        self.assertEqual(result["agent_backend"]["max_calls"], 3)
        verify.assert_not_called()
        (self.output / "sources/meshy.py").write_bytes(b"changed snapshot")
        with self.assertRaisesRegex(ValueError, "source pin changed"):
            verifier.inspect(JOB)

    def seal_unsupported_fixture(self):
        self.api_fixture(reuse=True)
        self.data.update(token="b"*32, owner_pid=900001, owner_identity="analytic-exited-owner", operation="execute",
                         invocation_dir="jobs/"+JOB+"/run-0001")
        invocation = self.state/self.data["invocation_dir"]
        write(self.output/"unsupported.json", {"status":"unsupported", "stage":"repair_budget",
                                               "request_sha256":digest(self.output/"request.json")})
        write(self.output/"status.json", {"status":"unsupported", "stage":"repair_budget"})
        write(invocation/"invocation.json", {"job_id":JOB,"token":self.data["token"],"operation":"execute",
            "request_digest":_digest(self.request),"root":str(self.root),"invocation_dir":str(invocation)})
        write(invocation/"owner.json", {"pid":self.data["owner_pid"],"start_identity":self.data["owner_identity"],
                                        "token":self.data["token"]})
        write(invocation/"worker-job.json", {"status":"finished","exit_code":0,
                                            "manual_stop_required":False,"process_exit_verified":True})
        for name in ("result.json","supervisor-result.json"):
            write(invocation/name,{"status":"unsupported","safe_to_release":True})
        self.update_row(data=json.dumps(self.data),status="unsupported")
        return invocation

    def test_sealed_unsupported_uses_archived_sources_not_current_source(self):
        self.seal_unsupported_fixture()
        for name in list(self.sources):
            changed=self.root/"new-implementation"/name
            changed.parent.mkdir(exist_ok=True);changed.write_text("changed implementation")
            self.sources[name]=changed
        result=verifier.inspect(JOB)
        self.assertEqual(result["status"],"unsupported")
        self.assertIn("outputs/"+self.output.name+"/unsupported.json",[row["path"] for row in result["files"]])
        (self.output/"sources/meshy.py").write_text("changed archive")
        with self.assertRaisesRegex(ValueError,"source pin changed"):verifier.inspect(JOB)

    def test_unsupported_needs_bound_terminal_and_safe_exit_receipts(self):
        invocation=self.seal_unsupported_fixture()
        for path,field,value in ((self.output/"unsupported.json","request_sha256","0"*64),
                                 (invocation/"result.json","safe_to_release",False),
                                 (invocation/"worker-job.json","process_exit_verified",False),
                                 (invocation/"invocation.json","request_digest","0"*64)):
            original=verifier.read_json(path);write(path,{**original,field:value})
            with self.subTest(field=field),self.assertRaises(ValueError):verifier.inspect(JOB)
            write(path,original)

    def test_terminal_author_unsupported_is_authenticated_without_acceptance(self):
        self.seal_unsupported_fixture()
        terminal=verifier.read_json(self.output/"unsupported.json")
        terminal.update(stage="author",attempt=1,reason="The finishing profile cannot be repaired.")
        write(self.output/"unsupported.json",terminal)
        write(self.output/"status.json",{k:terminal[k] for k in ("stage","status","attempt","reason")})
        folder=self.output/"agents/author-1"
        answer={"schema_version":2,"status":"unsupported","design":None,"reason":terminal["reason"]}
        write(folder/"answer.json",answer)
        receipt={"status":"succeeded","execution_kind":"openai_responses_api","tool_activity":False,
                 "answer_sha256":digest(folder/"answer.json"),"artifacts":[
                     {"path":"answer.json","bytes":(folder/"answer.json").stat().st_size,"sha256":digest(folder/"answer.json")}]}
        write(folder/"receipt.json",receipt)
        result=verifier.inspect(JOB)
        self.assertEqual(result["status"],"unsupported")
        self.assertIn(str((folder/"receipt.json").relative_to(self.root)),[r["path"] for r in result["files"]])
        write(folder/"answer.json",{**answer,"reason":"Changed"})
        with self.assertRaisesRegex(ValueError,"terminal reason"):verifier.inspect(JOB)
        write(folder/"answer.json",answer)
        write(folder/"receipt.json",{**receipt,"answer_sha256":"0"*64})
        with self.assertRaisesRegex(ValueError,"hash-bound"):verifier.inspect(JOB)
        write(folder/"receipt.json",receipt)
        terminal["attempt"]=2;write(self.output/"unsupported.json",terminal)
        with self.assertRaisesRegex(ValueError,"attempt differs"):verifier.inspect(JOB)

    def continuation_fixture(self):
        self.api_fixture(reuse=True)
        proof={"receipts":[],"known_pids":[],"provenance":"analytic delegation fixture"}
        self.request.update(continuation_from="c"*24,continuation_policy_sha256="d"*64,
                            continuation_custody_sha256=_digest(proof))
        write(self.job/"request.json",self.request);write(self.job/"retry-parent-custody.json",proof)
        self.prepared["continuation"]={"parent":"outputs/studio-"+"c"*24}
        write(self.output/"request.json",self.prepared)
        write(self.output/"input-receipt.json",{"request_sha256":digest(self.output/"request.json"),
                                               "sources":self.pins(self.output/"sources")})
        write(self.job/"pipeline-request.json",{"schema_version":1,"studio_request_digest":_digest(self.request),
                                               "pipeline_request_sha256":digest(self.output/"request.json")})
        self.data["plan_id"]=_digest({"request":_digest(self.request),"plan":self.plan,"generation":None})
        self.update_row(request=json.dumps(self.request),request_digest=_digest(self.request),data=json.dumps(self.data))

    def test_continuation_delegates_production_receipt_verification(self):
        self.continuation_fixture()
        with patch("avatar_assets.pipeline_continuation.verify",return_value={}) as verify:
            result=verifier.inspect(JOB)
            verify.assert_called_once_with(self.root,self.output,self.prepared)
        self.assertEqual(result["plan_id"],self.data["plan_id"])
        with patch("avatar_assets.pipeline_continuation.verify",side_effect=ValueError("Continuation pins changed")), \
             self.assertRaisesRegex(ValueError,"Continuation pins changed"):
            verifier.inspect(JOB)
        write(self.job/"retry-parent-custody.json",{"changed":True})
        with self.assertRaisesRegex(ValueError,"custody receipt differs"):verifier.inspect(JOB)

    def test_continuation_fields_are_exact_not_arbitrary_request_extensions(self):
        self.continuation_fixture()
        for change in ({"continuation_from":"../escape"},{"continuation_policy_sha256":"invalid"},{"max_microdollars":15000000}):
            bad={**self.request,**change};write(self.job/"request.json",bad)
            self.update_row(request=json.dumps(bad),request_digest=_digest(bad))
            with self.subTest(change=change),self.assertRaises(ValueError):verifier.inspect(JOB)

    def test_queue_and_foreign_lease_still_block_server_before_creation(self):
        create_server = Mock(side_effect=AssertionError("Server must not start"))
        for owner in ({"queued_jobs": [JOB], "active_lease_job": None},
                      {"queued_jobs": [], "active_lease_job": "b" * 24}):
            with self.subTest(owner=owner), patch.object(verifier, "inspect", return_value={"owner": owner}), \
                    patch.object(sys, "argv", ["verify", "--job", JOB, "--serve"]), \
                    patch.dict(sys.modules, {"avatar_assets.studio": SimpleNamespace(create_server=create_server)}):
                with self.assertRaises(ValueError):
                    verifier.main()
        create_server.assert_not_called()


if __name__ == "__main__":
    unittest.main()
