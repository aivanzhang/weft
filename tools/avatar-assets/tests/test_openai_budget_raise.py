"""Ceiling raises chain with reductions, bind their approvals and never rewrite history."""

import tempfile
import unittest
from pathlib import Path

from avatar_assets.openai_runner import OpenAILedger, budget_snapshot


def raise_approval(amount, reply="raise it"):
    return {"decision": "approved", "operation": "raise_project_ceiling",
            "max_microdollars": amount, "user_reply": reply}


def tighten_approval(amount):
    return {"decision": "approved", "operation": "tighten_project_ceiling",
            "max_microdollars": amount, "user_reply": "tighten"}


class BudgetRaiseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger = OpenAILedger(Path(self.tmp.name).resolve() / ".local/openai")
        self.ledger.configure(10_000_000, {"decision": "approved", "user_reply": "base ten dollars"})
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.ledger.close)

    def test_raises_and_reductions_interleave_in_recorded_order(self):
        self.ledger.tighten_budget(8_000_000, tighten_approval(8_000_000))
        snapshot = self.ledger.raise_budget(20_000_000, raise_approval(20_000_000, "raise it to $20"))
        self.assertEqual(snapshot["effective_ceiling_microdollars"], 20_000_000)
        self.ledger.tighten_budget(12_000_000, tighten_approval(12_000_000))
        snapshot = self.ledger.raise_budget(15_000_000, raise_approval(15_000_000))
        self.assertEqual(snapshot["effective_ceiling_microdollars"], 15_000_000)
        self.assertEqual((snapshot["reductions"], snapshot["raises"]), (2, 2))
        # A fresh read replays the whole chain and lands on the same effective value.
        reread = budget_snapshot(self.ledger.db)
        self.assertEqual(reread["effective_ceiling_microdollars"], 15_000_000)
        self.assertEqual(reread["original_ceiling_microdollars"], 10_000_000)

    def test_raise_must_strictly_increase_and_bind_its_amount(self):
        with self.assertRaisesRegex(ValueError, "strictly increase"):
            self.ledger.raise_budget(10_000_000, raise_approval(10_000_000))
        with self.assertRaisesRegex(ValueError, "bind the exact higher amount"):
            self.ledger.raise_budget(20_000_000, raise_approval(19_000_000))
        with self.assertRaisesRegex(ValueError, "approval is required"):
            self.ledger.raise_budget(20_000_000, {"decision": "approved", "user_reply": ""})

    def test_tampered_raise_record_is_refused_on_read(self):
        self.ledger.raise_budget(20_000_000, raise_approval(20_000_000))
        self.ledger.db.execute("UPDATE budget_raises SET ceiling=30000000")
        self.ledger.db.commit()
        with self.assertRaisesRegex(ValueError, "bind the exact higher amount"):
            budget_snapshot(self.ledger.db)
        self.ledger.db.execute("UPDATE budget_raises SET ceiling=20000000, previous_ceiling=9000000")
        self.ledger.db.commit()
        with self.assertRaisesRegex(ValueError, "Broken API budget raise chain"):
            budget_snapshot(self.ledger.db)

    def test_grants_can_use_the_raised_headroom(self):
        self.ledger.authorize_job("studio-aaaaaaaaaaaaaaaaaaaaaaaa", "a" * 64, 3, 9_500_000,
                                  {"decision": "approved", "user_reply": "grant most of base"})
        with self.assertRaisesRegex(ValueError, "already allocated"):
            self.ledger.authorize_job("studio-bbbbbbbbbbbbbbbbbbbbbbbb", "b" * 64, 3, 2_000_000,
                                      {"decision": "approved", "user_reply": "second grant"})
        self.ledger.raise_budget(50_000_000, raise_approval(50_000_000, "raise it to $50"))
        self.ledger.authorize_job("studio-bbbbbbbbbbbbbbbbbbbbbbbb", "b" * 64, 3, 2_000_000,
                                  {"decision": "approved", "user_reply": "second grant"})


if __name__ == "__main__":
    unittest.main()
