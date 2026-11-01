# Resume the original camera policy lab

The current-source command in `docs/local-lab.md` is stale for `artifacts/camera-shared-policy-play-01/lab`: four required runtime files changed (`recording.py`, `shared_exercise_lab.py`, `shared_world_evidence.py`, `shared_world_packaging.py`). The current loader correctly rejects those exact-pin mismatches. Do not repin the old lab or widen compatibility.

## Existing frozen route

The original reviewed runtime remains at `.local/camera-shared-policy-play-runtime-01`, based on commit `9dae370ff8ded889cd7185e013031b7111f6445f` plus its recorded overrides. All 254 files (5,220,532 bytes) match its saved manifest; all 54 lab runtime pins and 246 implementation pins join that same snapshot. Runtime manifest SHA: `2d649f20080497ef3d365e0af59f5c862b45ffa6149d361a0468d4f295603f36`.

From the repository root, this console invocation selects that verified source snapshot, not current source:

```sh
PYTHONPATH="$PWD/.local/camera-shared-policy-play-runtime-01/src"   .venv-lab/bin/python -B -P -m weft lab-console   --resume "$PWD/artifacts/camera-shared-policy-play-01/lab"
```

This uses the frozen-source import route retained in `.local/run_camera_shared_policy_play_01.py`. It is a console command derived from that historical launch configuration, not a newly executed acceptance. Do not rerun the historical acceptance worker: it creates the already-completed study. The console can create new session recordings while preserving the completed `run/` package.

The shared interpreter must still satisfy the saved versions: Python 3.14.7, MuJoCo 3.17.0, NumPy 2.5.4, Gymnasium 1.4.0. This review did not import native dependencies or run the console; its loader must recheck those versions on use. If the environment differs, prepare a new current-runtime manual/authored lab or review a separate migration. Do not claim this old policy is current-runtime compatible.

## Verification and limits

Saved-only receipt: `.local/camera-policy-runtime-resume-review-02.json`. The initial auditor-only project-path mapping failure is retained in `-01.json`; all already-passed frozen-file hashes were reused. No native execution, replay, source acquisition, media decode, policy change or sealed-lab change occurred. The controller remains experimental trained joint corrections with programmed body-position assistance; this does not verify physical balance or ground/equipment contact.
