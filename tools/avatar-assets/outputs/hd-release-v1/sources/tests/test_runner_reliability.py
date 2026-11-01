"""Fault injection exercises monitor/exit races without starting native tools."""

import errno
import json
import signal
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from avatar_assets.runner import run_bounded


class FakeProcess:
    pid = 424242

    def __init__(self):
        self.returncode = None
        self.wait_timeouts = []
        self.direct_signals = []
        self.deny_direct = False

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.wait_timeouts.append(timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired(["analytic-owned-process"], timeout)
        return self.returncode

    def send_signal(self, signum):
        self.direct_signals.append(signum)
        if self.deny_direct:
            raise PermissionError(errno.EPERM, "injected direct signal denial")
        self.returncode = -int(signum)


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class RunnerReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.process = FakeProcess()
        self.clock = FakeClock()
        self.stack.enter_context(patch("avatar_assets.runner.subprocess.Popen", return_value=self.process))
        self.stack.enter_context(patch("avatar_assets.runner.time.monotonic", self.clock.monotonic))
        self.stack.enter_context(patch("avatar_assets.runner.time.sleep", self.clock.sleep))
        self.group_signal = self.stack.enter_context(patch("avatar_assets.runner.os.killpg"))
        self.group_signal.side_effect = lambda pid, sig: setattr(self.process, "returncode", -int(sig))

    def run_job(self, **kwargs):
        return run_bounded(["analytic-owned-process"], self.root / "run.log", self.root, **kwargs)

    def checkpoint(self):
        return json.loads((self.root / "run-monitor.json").read_text())

    def test_rss_exit_race_is_not_a_monitor_failure_or_an_invented_rss_sample(self):
        def exiting(_pid):
            self.process.returncode = 0
            raise PermissionError(errno.EPERM, "proc inspection raced exit")

        with patch("avatar_assets.runner.resident_bytes", side_effect=exiting):
            result = self.run_job()
        self.assertIsNone(result["failure"])
        self.assertIsNone(result["sampled_peak_rss_bytes"])
        self.assertEqual(result["rss_samples"], 0)
        self.assertTrue(result["rss_exit_race_observed"])
        self.assertEqual(result["exit_code"], 0)
        self.group_signal.assert_not_called()
        self.assertEqual(self.checkpoint(), result)

    def test_recorded_memory_breach_survives_a_successful_exit_race(self):
        def exiting(_pid):
            self.process.returncode = 0
            return 3 * 1024 * 1024

        with patch("avatar_assets.runner.resident_bytes", side_effect=exiting):
            result = self.run_job(memory_mb=2)
        self.assertEqual(result["failure"], "resident-memory budget exceeded")
        self.assertEqual(result["sampled_peak_rss_bytes"], 3 * 1024 * 1024)
        self.assertEqual(result["exit_code"], 0)
        self.group_signal.assert_not_called()

    def test_metrics_are_durable_before_denied_group_signal_and_child_exit(self):
        def deny_while_exiting(_pid, _sig):
            saved = self.checkpoint()
            self.assertEqual(saved["status"], "stop_requested")
            self.assertEqual(saved["failure"], "resident-memory budget exceeded")
            self.assertEqual(saved["sampled_peak_rss_bytes"], 3 * 1024 * 1024)
            self.assertIsNone(saved["exit_code"])
            self.process.returncode = 0
            raise PermissionError(errno.EPERM, "injected group signal denial")

        self.group_signal.side_effect = deny_while_exiting
        with patch("avatar_assets.runner.resident_bytes", return_value=3 * 1024 * 1024):
            result = self.run_job(memory_mb=2)
        self.assertEqual(result["failure"], "resident-memory budget exceeded")
        self.assertEqual(result["cleanup_status"], "child_exit_verified")
        self.assertEqual(result["cleanup_events"][0]["errno"], errno.EPERM)
        self.assertFalse(result["manual_stop_required"])
        self.assertEqual(self.process.direct_signals, [])
        self.assertEqual(result, self.checkpoint())

    def test_denied_group_signal_falls_back_to_only_the_owned_child(self):
        self.group_signal.side_effect = PermissionError(errno.EPERM, "injected denial")
        with patch("avatar_assets.runner.resident_bytes", return_value=3 * 1024 * 1024):
            result = self.run_job(memory_mb=2)
        self.assertEqual(self.process.direct_signals, [signal.SIGTERM])
        self.assertEqual(self.process.wait_timeouts, [5])
        self.assertEqual(result["exit_code"], -signal.SIGTERM)
        self.assertEqual(result["cleanup_status"], "child_exit_verified")
        self.assertFalse(result["descendant_exit_verified"])

    def test_denied_cleanup_has_bounded_waits_and_never_claims_exit(self):
        self.process.deny_direct = True
        self.group_signal.side_effect = PermissionError(errno.EPERM, "injected denial")
        with patch("avatar_assets.runner.resident_bytes", return_value=3 * 1024 * 1024):
            result = self.run_job(memory_mb=2)
        self.assertEqual(self.process.wait_timeouts, [5, 2])
        self.assertEqual(self.process.direct_signals, [signal.SIGTERM, signal.SIGKILL])
        self.assertIsNone(result["exit_code"])
        self.assertFalse(result["process_exit_verified"])
        self.assertTrue(result["manual_stop_required"])
        self.assertEqual(result["status"], "cleanup_incomplete")
        self.assertEqual(result["failure"], "resident-memory budget exceeded")
        self.assertIn("owned process cleanup incomplete", result["failure_reasons"][-1])
        self.assertEqual(result, self.checkpoint())

    def test_successful_kill_signal_does_not_make_final_wait_unbounded(self):
        self.group_signal.side_effect = None
        with patch("avatar_assets.runner.resident_bytes", return_value=3 * 1024 * 1024):
            result = self.run_job(memory_mb=2)
        self.assertEqual(self.process.wait_timeouts, [5, 2])
        self.assertTrue(result["manual_stop_required"])
        self.assertIsNone(result["exit_code"])
        self.assertEqual([call.args[1] for call in self.group_signal.call_args_list],
                         [signal.SIGTERM, signal.SIGKILL])

    def test_live_rss_monitor_denial_stops_job_and_preserves_prior_peak(self):
        with patch("avatar_assets.runner.resident_bytes", side_effect=[12345, PermissionError(errno.EPERM, "denied")]):
            result = self.run_job()
        self.assertEqual(result["failure"], "resident-memory monitor unavailable")
        self.assertEqual(result["sampled_peak_rss_bytes"], 12345)
        self.assertEqual(result["rss_samples"], 1)
        self.assertEqual(result["monitor_errors"][0]["operation"], "resident_memory")
        self.assertEqual(result["cleanup_status"], "child_exit_verified")

    def test_unexpected_monitor_exception_preserves_metrics_then_reraises(self):
        with patch("avatar_assets.runner.resident_bytes", side_effect=[12345, RuntimeError("injected bug")]):
            with self.assertRaisesRegex(RuntimeError, "injected bug"):
                self.run_job()
        saved = self.checkpoint()
        self.assertEqual(saved["sampled_peak_rss_bytes"], 12345)
        self.assertEqual(saved["failure"], "runner exception: RuntimeError")
        self.assertTrue(saved["process_exit_verified"])
        self.assertEqual(self.process.wait_timeouts, [5])

    def test_final_output_scan_denial_preserves_real_exit_and_memory(self):
        def exiting(_pid):
            self.process.returncode = 0
            return 12345

        with patch("avatar_assets.runner.resident_bytes", side_effect=exiting), \
                patch("avatar_assets.runner._output_bytes", side_effect=PermissionError(errno.EPERM, "injected scan denial")):
            result = self.run_job()
        self.assertEqual(result["failure"], "output byte monitor unavailable")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["sampled_peak_rss_bytes"], 12345)
        self.assertIsNone(result["final_output_bytes"])
        self.assertEqual(result, self.checkpoint())

    def test_status_monitor_denial_preserves_peak_and_stops_owned_child(self):
        original_poll, calls = self.process.poll, 0

        def intermittent_poll():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise PermissionError(errno.EPERM, "injected waitpid denial")
            return original_poll()

        self.process.poll = intermittent_poll
        with patch("avatar_assets.runner.resident_bytes", return_value=12345):
            result = self.run_job()
        self.assertEqual(result["failure"], "process status monitor unavailable")
        self.assertEqual(result["sampled_peak_rss_bytes"], 12345)
        self.assertEqual(result["cleanup_status"], "child_exit_verified")
        self.assertEqual(result, self.checkpoint())

    def test_wait_permission_error_does_not_erase_limit_or_create_unbounded_retry(self):
        def group_signal(_pid, signum):
            if signum == signal.SIGKILL:
                self.process.returncode = -int(signum)

        self.group_signal.side_effect = group_signal
        with patch.object(self.process, "wait", side_effect=PermissionError(errno.EPERM, "injected wait denial")) as wait, \
                patch("avatar_assets.runner.resident_bytes", return_value=3 * 1024 * 1024):
            result = self.run_job(memory_mb=2)
        self.assertEqual([call.kwargs["timeout"] for call in wait.call_args_list], [5, 2])
        self.assertEqual(result["exit_code"], -signal.SIGKILL)
        self.assertEqual(result["failure"], "resident-memory budget exceeded")
        self.assertEqual(len([e for e in result["cleanup_events"] if e["result"] == "wait_error"]), 2)
        self.assertEqual(result, self.checkpoint())

    def test_existing_monitor_is_never_overwritten(self):
        original = '{"preserved":"prior attempt"}\n'
        (self.root / "run-monitor.json").write_text(original)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_job()
        self.assertEqual((self.root / "run-monitor.json").read_text(), original)

    def test_nonfinite_or_nonpositive_limits_are_rejected_before_launch(self):
        for value in (float("inf"), float("nan"), 0, -1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite positive"):
                self.run_job(timeout=value)
        self.assertFalse((self.root / "run.log").exists())


if __name__ == "__main__":
    unittest.main()
