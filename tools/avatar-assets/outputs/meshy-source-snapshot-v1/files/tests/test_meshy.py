"""Analytic provider fixtures: billing invariants, not generated model quality."""

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from avatar_assets.meshy import prepare, public_value, stage_request, status, submit, validate_config
from avatar_assets.meshy_access import Ledger, MeshyError, api_path, load_key
from avatar_assets.safety import sha256, write_json


def config():
    return {"schema_version": 1, "id": "analytic-robot", "subject": "A test robot", "style": "Painted blue metal",
            "prompt": "A friendly blue humanoid robot", "generation": {"mode": "text", "geometry_resolution": "4k",
            "texture_resolution": "4k", "target_polycount": 60000, "pose_mode": "a-pose"},
            "states": [{"id": "wave", "prompt": "Wave with the left arm", "duration": 4}],
            "rig": "humanoid", "height_meters": 1.2, "max_credits": 53, "rights": "Analytic fixture; no paid calls"}


class FakeClient:
    def __init__(self, unknown=False):
        self.posts = 0
        self.gets = 0
        self.unknown = unknown

    def balance(self):
        return {"balance": 1000}

    def request(self, method, path, payload=None):
        if method == "POST":
            self.posts += 1
            if self.unknown:
                raise MeshyError(None, "Simulated lost response")
            return {"result": "fixture-task"}
        self.gets += 1
        return {"id": "fixture-task", "status": "SUCCEEDED", "progress": 100, "consumed_credits": 25}


class MeshyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir="/private/tmp")
        self.root = Path(self.temp.name)
        self.config = config()
        write_json(self.root/"config.json", self.config)
        self.out = self.root/"job"
        self.ledger = Ledger(self.root)
        self.ledger.configure(100)

    def tearDown(self):
        self.ledger.close()
        self.temp.cleanup()

    def prepared(self):
        prepare(self.root/"config.json", self.out)
        return json.loads((self.out/"plan.json").read_text())

    def test_estimate_covers_all_dependent_calls(self):
        self.assertEqual(validate_config(self.config), 53)
        self.config["max_credits"] = 52
        with self.assertRaisesRegex(ValueError, "credit ceiling"):
            validate_config(self.config)

    def test_duplicate_submit_never_pays_again(self):
        self.prepared()
        client = FakeClient()
        first = submit(self.out, "mesh", client, self.ledger)
        second = submit(self.out, "mesh", client, self.ledger)
        self.assertTrue(first["new_paid_call"])
        self.assertFalse(second["new_paid_call"])
        self.assertEqual(client.posts, 1)

    def test_unknown_submission_keeps_reservation_and_blocks_retry(self):
        self.prepared()
        client = FakeClient(unknown=True)
        with self.assertRaises(MeshyError):
            submit(self.out, "mesh", client, self.ledger)
        with self.assertRaisesRegex(ValueError, "Uncertain submission"):
            submit(self.out, "mesh", client, self.ledger)
        self.assertEqual(client.posts, 1)
        self.assertEqual(self.ledger.summary()["reserved_credits"], 25)
        self.assertEqual(self.ledger.summary()["unknown_submissions"], 1)

    def test_poll_uses_existing_task_and_reuses_terminal_result(self):
        self.prepared()
        client = FakeClient()
        submit(self.out, "mesh", client, self.ledger)
        self.assertEqual(status(self.out, "mesh", client, self.ledger)["status"], "SUCCEEDED")
        status(self.out, "mesh", client, self.ledger)
        self.assertEqual(client.gets, 1)
        self.assertEqual(self.ledger.summary()["provider_reported_credits"], 25)

    def test_changed_plan_does_not_reuse_job_identity(self):
        self.prepared()
        client = FakeClient()
        submit(self.out, "mesh", client, self.ledger)
        plan = json.loads((self.out/"plan.json").read_text())
        plan["config"]["prompt"] = "Changed model"
        (self.out/"plan.json").write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "plan or location changed"):
            submit(self.out, "mesh", client, self.ledger)

    def test_dependencies_prevent_paying_for_unusable_motion(self):
        plan = self.prepared()
        for stage in ("rig", "motion_wave", "animate_wave"):
            with self.assertRaisesRegex(ValueError, "must succeed first"):
                stage_request(self.out, plan, stage, self.ledger)

    def test_shared_ledger_cannot_overspend_or_reset(self):
        plan = self.prepared()
        self.ledger.bind_job(plan, self.out)
        self.ledger.reserve(plan["id"], "one", "/fixture", {}, 40)
        other = Ledger(self.root)
        try:
            with self.assertRaisesRegex(ValueError, "allowance exhausted"):
                other.reserve(plan["id"], "two", "/fixture", {}, 14)
            with self.assertRaisesRegex(ValueError, "immutable"):
                other.configure(200)
            with self.assertRaisesRegex(ValueError, "already reserved"):
                other.reserve(plan["id"], "one", "/fixture", {}, 1)
        finally:
            other.close()

    def test_unexpected_provider_cost_persists_and_stops_new_paid_work(self):
        plan = self.prepared()
        self.ledger.bind_job(plan, self.out)
        self.ledger.reserve(plan['id'], 'mesh', '/fixture', {'estimated_credits':25}, 25)
        self.ledger.update(plan['id'], 'mesh', 'SUCCEEDED', 'fixture', {'consumed_credits':30})
        self.assertEqual(self.ledger.summary()['reserved_credits'],30)
        self.assertEqual(self.ledger.summary()['pricing_overruns'],1)
        self.ledger.update(plan['id'], 'mesh', 'FAILED', response={'consumed_credits':0})
        self.assertEqual(self.ledger.summary()['reserved_credits'],30)
        with self.assertRaisesRegex(ValueError,'reconcile pricing'):
            self.ledger.reserve(plan['id'], 'texture', '/fixture', {'estimated_credits':10}, 10)

    def test_reference_is_preserved_and_changed_bytes_rejected(self):
        image = self.root/"selected.png"
        Image.new("RGB", (8, 8), "red").save(image)
        cfg = config()
        cfg["generation"]["mode"] = "image"
        cfg.update(reference="selected.png", reference_sha256=sha256(image))
        write_json(self.root/"image-config.json", cfg)
        prepare(self.root/"image-config.json", self.out)
        copy = self.out/"inputs/reference.png"
        self.assertEqual(copy.read_bytes(), image.read_bytes())
        Image.new("RGB", (8, 8), "blue").save(copy)
        with self.assertRaisesRegex(ValueError, "reference changed"):
            submit(self.out, "mesh", FakeClient(), self.ledger)

    def test_disguised_reference_format_is_rejected_before_a_paid_call(self):
        image = self.root/'disguised.png'
        Image.new('RGB',(8,8),'red').save(image,format='GIF')
        cfg = config();cfg['generation']['mode']='image'
        cfg.update(reference=image.name,reference_sha256=sha256(image))
        write_json(self.root/'image-config.json',cfg)
        with self.assertRaisesRegex(ValueError,'content format'):
            prepare(self.root/'image-config.json',self.out)
        self.assertFalse(self.out.exists())

    def test_foreign_and_malformed_endpoints_rejected(self):
        for path in ("https://evil.invalid", "/openapi/v1/rigging/../balance", "/openapi/v1/image-to-3d?url=x",
                     "/openapi/v1/accounts", "/openapi/v1/image-to-3d/"+"a"*129):
            self.assertFalse(api_path(path))
        self.assertTrue(api_path("/openapi/v1/image-to-3d/task-1"))
        self.assertTrue(api_path("/openapi/v1/text-to-motion"))

    def test_signed_urls_not_in_public_receipts(self):
        self.assertEqual(public_value({"url": "https://assets.meshy.ai/m.glb?secret=signed#x"}),
                         {"url": "https://assets.meshy.ai/m.glb"})
        self.assertNotIn("base64", json.dumps(public_value("data:image/png;base64,secret")))

    def test_secret_file_permissions_rejected(self):
        folder = self.root/".local/credentials"
        folder.mkdir(mode=0o700)
        key = folder/"meshy-api-key"
        key.write_text("msy_fixturekeynotreal")
        key.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "owner-only"):
            load_key(self.root)
        key.chmod(0o600)
        self.assertEqual(load_key(self.root), "msy_fixturekeynotreal")

    def test_invalid_config_before_creating_output(self):
        for value in (float("nan"), 1, 2.1, 20):
            cfg = config()
            cfg["states"][0]["duration"] = value
            with self.assertRaises(ValueError):
                validate_config(cfg)
        cfg = config()
        cfg["generation"]["target_polycount"] = 100001
        with self.assertRaises(ValueError):
            validate_config(cfg)


if __name__ == "__main__":
    unittest.main()
