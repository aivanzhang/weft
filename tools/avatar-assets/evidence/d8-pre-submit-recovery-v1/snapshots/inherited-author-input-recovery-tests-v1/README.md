# Zero-call inherited-plan recovery

Ready, not adopted. The API prompt limit failed before any provider submission. The guarded profile requires the exact error, zero call rows, only runner metadata, a verified non-replan continuation, the unchanged valid saved plan, original source snapshots/grant and completed source-bound inspection. Existing paid-planner recovery remains supported.

No actual lease, job, ledger or old artifact was changed. 30 focused tests passed. See `result.json` for exact hashes, commands, owner identity and source changes.

## Root actions after checkpoint

Run from `tools/avatar-assets` using `.venv/bin/python`. First release the retained lease through `StudioJobs.release_inspected`, then prepare and adopt the append-only revision. These actions do not dispatch an API/native job; website Start remains separate.

```python
from pathlib import Path
from avatar_assets.studio_jobs import StudioJobs
from avatar_assets.pipeline_recovery import prepare_revision
from avatar_assets.safety import read_json

root = Path.cwd()
registry = read_json(root / '.local/studio-registry.json')
jobs = StudioJobs(root, state_dir=Path(registry['state_dir']))
evidence = read_json(root / 'outputs/inherited-author-input-recovery-tests-v1/result.json')
job_id = 'd8a680ce6580429f9fdd016f'
jobs.release_inspected(job_id, expected_pid=64159,
    expected_identity='darwin-start:1793660115:477725',
    reason='Preserve zero-call inherited plan and verified inspection after the exact API prompt bound failure.')
revision = root / 'outputs/d8-inherited-author-input-runtime-revision-v1'
prepared = prepare_revision(root / ('outputs/studio-' + job_id), revision,
    expected_request_sha256=evidence['actual_read_only_check']['request_sha256'],
    expected_changes=evidence['expected_revision_changes'],
    reason='Compact JSON whitespace only; recover the unchanged inherited plan without new planning or budget changes.',
    profile='author_input_bound')
jobs.adopt_cached_planner_revision(job_id, revision, expected_revision_sha256=prepared['sha256'])
```

Do not overwrite an existing revision folder. If ownership, source hashes, calls, grants, inspection or inherited evidence changes, the guarded path must stop for inspection. No direct retry or counter reset is authorized by this helper.
