"""Real temporary SQLite accounting; canonical Studio custody is a mocked boundary."""

import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

from avatar_assets import openai_grant_closure as closure
from avatar_assets import openai_runner as api

APPROVAL = {'decision': 'approved', 'user_reply': 'Analytic authorization only; no actual spend.'}
USAGE = {'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120}


class GrantClosureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.ledger = api.OpenAILedger(self.root / '.local/openai')
        self.addCleanup(self.ledger.close)
        self.parent = 'studio-' + 'a' * 24
        self.digest = 'a' * 64
        self.ledger.configure(15_000_000, APPROVAL)
        self.ledger.authorize_job(self.parent, self.digest, 7, 11_063_773, APPROVAL)
        self.ledger.reserve(self.parent, self.digest, 7, 'planner', 'b' * 64, 5_618_640)
        self.ledger.settle(self.parent, 'planner', 'resp_planner', USAGE, 550)
        self.ledger.reserve(self.parent, self.digest, 7, 'author-0', 'c' * 64, 5_618_640)
        self.record = self.release_record()
        self.path = self.root / '.local/studio/jobs' / self.parent[7:] / ('uncertain-owner-release-' + 'd' * 32 + '.json')
        self.path.parent.mkdir(mode=0o700, parents=True)
        self.save_receipt()
        verifier = types.SimpleNamespace(verify_release=lambda *args, **kwargs: copy.deepcopy(self.record))
        self.mock_boundary = patch.dict('sys.modules', {'avatar_assets.studio_unknown_release': verifier})
        self.mock_boundary.start()
        self.addCleanup(self.mock_boundary.stop)

    def release_record(self):
        grant = dict(self.ledger.db.execute('SELECT * FROM jobs WHERE id=?', (self.parent,)).fetchone())
        return {'schema_version': 1, 'job_id': self.parent[7:], 'request_digest': self.digest,
                'requested_transition': 'release_local_owner_preserve_uncertain_charge',
                'no_dispatch': True, 'no_cost_reset': True, 'old_job_restart_allowed': False,
                'proof': {'unknown_provider': {'policy': 'unknown-api-author-custody-v1',
                    'job_id': self.parent, 'request_sha256': self.digest,
                    'grant': closure._grant(grant), 'call_rows': closure._inventory(self.ledger.db, self.parent),
                    'uncertain_call': {'role': 'author-0', 'request_digest': 'c' * 64,
                        'reserved_microdollars': 5_618_640, 'state': 'reserved_uncertain',
                        'response_id': None, 'actual_microdollars': None}}}}

    def save_receipt(self):
        self.path.write_bytes(closure._canonical(self.record))
        self.path.chmod(0o600)
        self.receipt_sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.approval = {**APPROVAL, 'operation': 'close_unused_grant', 'parent_job_id': self.parent,
                         'parent_request_sha256': self.digest, 'release_receipt_sha256': self.receipt_sha}

    def close_grant(self, ledger=None, **changes):
        args = {'root': self.root, 'parent_id': self.parent, 'parent_digest': self.digest,
                'release_receipt': self.path, 'release_sha256': self.receipt_sha, 'approval': self.approval}
        args.update(changes)
        return closure.close_unused_grant(ledger or self.ledger, **args)

    def history(self):
        return {name: [tuple(row) for row in self.ledger.db.execute('SELECT * FROM '+name+' ORDER BY 1,2')]
                for name in ('budget', 'budget_reductions', 'jobs', 'calls', 'retirements', 'call_profiles')}

    def test_closure_preserves_every_original_row_and_full_unknown_hold(self):
        before = self.history()
        result = self.close_grant()
        self.assertFalse(result['idempotent'])
        self.assertEqual(self.history(), before)
        self.assertEqual(result['exposure_microdollars'], 5_619_190)
        self.assertEqual(self.ledger._allocated(), 5_619_190)
        accounting = api.accounting_snapshot(self.ledger.db)
        self.assertEqual(accounting['jobs'][self.parent]['uncertain_reserved_microdollars'], 5_618_640)
        self.assertEqual(accounting['project_exposure_microdollars'], 5_619_190)
        self.assertEqual(self.ledger.call(self.parent, 'author-0')['state'], 'reserved_uncertain')

    def test_new_independent_grant_uses_only_unallocated_capacity(self):
        self.close_grant()
        self.ledger.authorize_job('independent', 'd' * 64, 3, 9_380_810, APPROVAL)
        self.assertEqual(self.ledger._allocated(), 15_000_000)
        with self.assertRaisesRegex(ValueError, 'already allocated'):
            self.ledger.authorize_job('overflow', 'e' * 64, 3, 1, APPROVAL)
        self.assertEqual(self.ledger.call(self.parent, 'author-0')['reserved'], 5_618_640)

    def test_closed_parent_reserve_and_continuation_stay_blocked(self):
        self.close_grant()
        with self.assertRaisesRegex(ValueError, 'Closed API grant'):
            self.ledger.reserve(self.parent, self.digest, 7, 'author-1', 'e' * 64, 1)
        self.ledger.settle(self.parent, 'author-0', 'resp_verified_later', USAGE, 550)
        approval = {**APPROVAL, 'operation': 'retire_and_continue', 'parent_job_id': self.parent,
                    'successor_job_id': 'child', 'additional_calls': 3}
        with self.assertRaisesRegex(ValueError, 'Closed API grant'):
            self.ledger.continue_job(self.parent, self.digest, 'child', 'd' * 64, 3, 1000, approval)
        self.assertEqual(self.ledger.db.execute('SELECT COUNT(*) FROM retirements').fetchone()[0], 0)

    def test_idempotency_has_no_duplicate_history_or_allocation(self):
        self.close_grant()
        before = self.history()
        record = self.ledger.db.execute('SELECT record FROM grant_closures').fetchone()[0]
        self.assertTrue(self.close_grant()['idempotent'])
        self.assertEqual(self.history(), before)
        self.assertEqual(self.ledger.db.execute('SELECT record FROM grant_closures').fetchall()[0][0], record)
        self.assertEqual(self.ledger.db.execute('SELECT COUNT(*) FROM grant_closures').fetchone()[0], 1)

    def test_existing_closure_requires_same_approval_and_receipt(self):
        self.close_grant()
        with self.assertRaisesRegex(ValueError, 'Existing grant closure'):
            self.close_grant(approval={**self.approval, 'user_reply': 'Different approval'})

    def test_legitimate_later_settlement_does_not_rewrite_closure(self):
        self.close_grant()
        record = self.ledger.db.execute('SELECT record FROM grant_closures').fetchone()[0]
        self.ledger.settle(self.parent, 'author-0', 'resp_verified_later', USAGE, 550)
        self.assertEqual(self.ledger._allocated(), 1100)
        self.assertEqual(self.ledger.db.execute('SELECT record FROM grant_closures').fetchone()[0], record)
        self.assertEqual(closure.grant_closures(self.ledger.db)[self.parent]['exposure_microdollars'], 5_619_190)

    def test_receipt_hash_permissions_and_symlink_fail_closed(self):
        for change in ('hash', 'mode', 'symlink'):
            with self.subTest(change=change):
                if change == 'hash':
                    kwargs = {'release_sha256': 'f' * 64,
                              'approval': {**self.approval, 'release_receipt_sha256': 'f' * 64}}
                elif change == 'mode':
                    self.path.chmod(0o644); kwargs = {}
                else:
                    target = self.path.with_suffix('.original')
                    self.path.rename(target); self.path.symlink_to(target); kwargs = {}
                with self.assertRaises(ValueError):
                    self.close_grant(**kwargs)
                if change == 'mode':
                    self.path.chmod(0o600)
                elif change == 'symlink':
                    self.path.unlink(); target.rename(self.path)
        self.assertEqual(self.ledger.db.execute('SELECT COUNT(*) FROM grant_closures').fetchone()[0], 0)

    def test_release_and_approval_must_bind_parent_and_nonrestart_transition(self):
        cases = [('parent_job_id', 'studio-' + 'b' * 24), ('operation', 'settle'),
                 ('parent_request_sha256', 'f' * 64), ('release_receipt_sha256', 'e' * 64)]
        for key, value in cases:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.close_grant(approval={**self.approval, key: value})
        self.record['old_job_restart_allowed'] = True
        self.save_receipt()
        with self.assertRaisesRegex(ValueError, 'prohibit restart'):
            self.close_grant()

    def test_changed_call_after_release_invalidates_initial_closure(self):
        self.ledger.reserve(self.parent, self.digest, 7, 'author-1', 'd' * 64, 100)
        before = self.history()
        with self.assertRaisesRegex(ValueError, 'call inventory or grant changed'):
            self.close_grant()
        self.assertEqual(self.history(), before)
        self.assertEqual(self.ledger.db.execute('SELECT COUNT(*) FROM grant_closures').fetchone()[0], 0)

    def test_settlement_after_release_requires_new_custody_proof(self):
        self.ledger.settle(self.parent, 'author-0', 'resp_verified', USAGE, 550)
        with self.assertRaisesRegex(ValueError, 'call inventory or grant changed'):
            self.close_grant()

    def test_post_insert_failure_rolls_back_without_rewriting_history(self):
        before = self.history()
        with patch.object(closure, 'allocated_grants', return_value=15_000_001):
            with self.assertRaisesRegex(ValueError, 'Other active allocations'):
                self.close_grant()
        self.assertEqual(self.history(), before)
        self.assertEqual(self.ledger.db.execute('SELECT COUNT(*) FROM grant_closures').fetchone()[0], 0)

    def test_history_or_closure_tampering_blocks_allocation(self):
        self.close_grant()
        self.ledger.db.execute('UPDATE calls SET reserved=reserved-1 WHERE job=? AND role=?',
                               (self.parent, 'author-0'))
        self.ledger.db.commit()
        with self.assertRaisesRegex(ValueError, 'reservation changed'):
            self.ledger._allocated()

    def test_legacy_readonly_database_needs_no_closure_table(self):
        self.ledger.db.execute('DROP TABLE grant_closures')
        self.ledger.db.commit()
        db = sqlite3.connect((self.ledger.folder/'ledger.sqlite3').as_uri()+'?mode=ro', uri=True)
        self.addCleanup(db.close)
        self.assertEqual(closure.grant_closures(db), {})
        self.assertEqual(closure.allocated_grants(db), 11_063_773)

    def test_reserve_close_race_cannot_use_a_stale_release_proof(self):
        barrier = threading.Barrier(2)
        outcomes = []
        def operation(close):
            ledger = api.OpenAILedger(self.root / '.local/openai')
            try:
                barrier.wait(timeout=2)
                if close:
                    self.close_grant(ledger)
                else:
                    ledger.reserve(self.parent, self.digest, 7, 'author-1', 'd' * 64, 100)
                outcomes.append(('close' if close else 'reserve', True))
            except ValueError:
                outcomes.append(('close' if close else 'reserve', False))
            finally:
                ledger.close()
        threads = [threading.Thread(target=operation, args=(flag,)) for flag in (True, False)]
        for thread in threads: thread.start()
        for thread in threads: thread.join(timeout=5); self.assertFalse(thread.is_alive())
        self.assertEqual(len(outcomes), 2)
        self.assertEqual(sum(passed for _, passed in outcomes), 1)
        self.assertEqual(self.ledger.call(self.parent, 'author-0')['reserved'], 5_618_640)

    def test_concurrent_new_grants_cannot_double_allocate_recovered_capacity(self):
        self.close_grant()
        barrier = threading.Barrier(2)
        outcomes = []
        def grant(index):
            ledger = api.OpenAILedger(self.root / '.local/openai')
            try:
                barrier.wait(timeout=2)
                ledger.authorize_job('child-'+str(index), str(index)*64, 3, 6_000_000, APPROVAL)
                outcomes.append(True)
            except ValueError:
                outcomes.append(False)
            finally:
                ledger.close()
        threads = [threading.Thread(target=grant, args=(index,)) for index in (1, 2)]
        for thread in threads: thread.start()
        for thread in threads: thread.join(timeout=5); self.assertFalse(thread.is_alive())
        self.assertEqual(sorted(outcomes), [False, True])
        self.assertEqual(self.ledger._allocated(), 11_619_190)


if __name__ == '__main__':
    unittest.main()
