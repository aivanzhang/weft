from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import artifacts, read_json, sha256, write_json

out = Path(__file__).resolve().parent
targets = [
    'test_repair_composition_continuation',
    'test_pipeline_continuation.ContinuationTests.test_paid_author_refusal_retains_valid_seed_and_native_failure_without_planner',
    'test_pipeline_continuation.ContinuationTests.test_replan_preserves_built_seed_and_all_rejected_answers',
    'test_pipeline_continuation.ContinuationTests.test_child_preserves_context_and_never_replans_or_resets_parent',
    'test_visual_repair_continuation.VisualContinuationTests.test_completed_visual_repair_retains_paid_design_critique_and_no_new_calls',
    'test_visual_repair_continuation.VisualContinuationTests.test_changed_action_gate_cannot_be_repaired_by_resealing_its_pack',
]
code = "import sys, unittest; sys.path.insert(0, 'tests'); unittest.main(module=None, argv=" + repr(['unittest', *targets]) + ", verbosity=2)"
command = [str(ROOT/'.venv/bin/python'), '-B', '-c', code]
write_json(out/'command.json', {'command': command, 'cwd': str(ROOT)})
job = run_bounded(command, out/'tests.log', out, timeout=30, memory_mb=512, output_mb=16)
write_json(out/'job.json', job)
prepared = read_json(out/'prepared.json')
changed = [name for name, pin in prepared['sources'].items() if sha256(ROOT/pin['path']) != pin['sha256']]
write_json(out/'result.json', {'passed': job['exit_code'] == 0 and not job['failure'] and not changed,
    'source_changes': changed, 'targets': targets, 'job': job, 'provider_calls': 0, 'native_calls': 0,
    'scope': 'Analytic receipt/authentication and continuation control flow; no appearance or actual native proof.'})
write_json(out/'manifest.json', {'schema_version': 1, 'kind': 'analytic_composed_continuation_tests', 'artifacts': artifacts(out)})
print({'exit_code': job['exit_code'], 'failure': job['failure'], 'seconds': job['elapsed_seconds'],
       'sampled_peak_rss_bytes': job['sampled_peak_rss_bytes'], 'rss_samples': job['rss_samples'],
       'source_changes': changed, 'manifest_sha256': sha256(out/'manifest.json')})
