# Named experimental equipment sandbox

## Current state

The equipment lab now has a verified named launcher preset. Implementation,
40 distinct authored/mocked checks, the real metadata-only launcher check and
one fresh full-source-checked read-only lab open pass.
The reference controller still passes **0/3 development starts**. This change
does not enable the experimental action projector or change that result.

```sh
.venv-lab/bin/python scripts/play_sandbox.py --example equipment
.venv-lab/bin/python scripts/play_sandbox.py --example equipment --console
.venv-lab/bin/python scripts/play_sandbox.py --example equipment --check
```

The first command prints a loopback link; the user opens it. The terminal command
uses the existing console and recording path. No server or browser was started
during this work. Recordings remain in
`artifacts/split-limb-feedback-play-05/lab/sessions`.

## Implementation and fixed inputs

- Changed `scripts/play_sandbox.py`, `tests/test_play_sandbox.py` and the
  [equipment section](../local-lab.md#experimental-equipment-sandbox).
- Fixed Git runtime: `fdab07bbdf8cad4c3c4362836ee6f1aef146bee2`, exported through
  the existing trusted-cache path. No mutable development-runtime fallback.
- Export: **292 files / 6,345,775 bytes**, largest file 405,810 bytes. Equipment-only
  limits: 292 files, 8 MiB total, 512 KiB/member; sealed lab: 1,191 members
  (1,192 including its manifest). Other presets keep their original bounds.
- Exact saved v5 policy: 25 source pins; lab: 17 source pins. Dependencies remain
  Python 3.14.7, MuJoCo 3.17.0, NumPy 2.5.4 and Gymnasium 1.4.0. Overrides must
  pass the same complete compatibility checks; the 0/3 notice explicitly refers
  to the reference example, not an arbitrary override.
- Policy SHA256: `47e4d2034233606d46bb1a94052a7ea35812bccf77d7b3449d3d3bb1dcc5c015`.
- Sealed run manifest: `36fefcb67333f51e212860555e1af95f4e594a52b3bf17688bcf11f324a372af`.
- Final launcher: `ecd0e540178bd358189ef4397b210e0d41387e3c73266d18f669088bb5b9f6bc`.
- Final tests: `6b25cd79fa79d4dc07ecd696172d86ee02ccbb7c3ec034ca371cfb86b1cb6534`.

## Completed verification

Scratch and receipts: `.local/equipment-lab-launcher-01/`.

1. Initial authored/mocked run: 39 cases, 0.495 seconds, 71,942,144-byte peak RSS.
   Includes seven exact pre-edit comparisons of successful old-preset output and
   launch commands. No subprocess, native, server, media or network work.
   Receipt `process-01.json`:
   `4fd07712e42fdf682ea9e6cd1c7c70f699e2c417840350bc9cf534c7d13ea283`.
2. The reference-result labeling fix reran four affected cases and added one
   help case: five pass, 0.215 seconds, 71,286,784-byte peak RSS. XML test-ID union
   is **40 distinct cases**, not 44. Unchanged passing checks were reused.
   Receipt `process-02.json`:
   `6dd791c940583d31f63386731afe96c508a08e2794cec3a689f272041febd882`.
3. Root executed the actual `--example equipment --console --check` once. Passed,
   exited 0 and reaped; 0.553207 seconds, 43,335,680-byte peak RSS; three output
   files / 4,225 bytes. It created the exact fixed Git cache using tree/blob
   children, then the fixed dependency-version child; read exactly four small
   saved metadata files. Zero native calls, source proofs, server or media work.
   Output `artifacts/equipment-lab-launcher-check-01`; root verified both sealed
   members. Seal:
   `9ebb35b67ab15a725b4c46ce5fe1208646d7914386accb480377abdf118bcbe8`.
   Frozen plan `metadata-plan-frozen-01.json`:
   `de6cdd2cabebe31ee11ef147f8e8fc43a52118b17edbd292620dc025db6dce65`.
4. Root executed the fresh read-only open once: passed, exited 0 and reaped;
   3.037478 seconds, 383,647,744-byte peak RSS (365.9 MiB); four output files /
   843,973 bytes. Exactly 28 models, one data object, two forwards and one reset;
   zero controls, steps, replay, decode or forbidden work. All 15 scoped summary
   hashes match the prior qualified open. The 1,192 run members / 173,860,491 bytes
   and nine saved-session members / 19,843,650 bytes remain unchanged.
   Output `artifacts/equipment-lab-launcher-open-01`; root verified all three
   sealed members once. Result:
   `64e59cc4e6a384b77268ac3cd59f7561016b602c32a796c63f37dd033697f863`;
   seal `1c530872abc0b4ed9116ba0e45cd5c4a89a5097ebca8ba6d2e2a1faa433d64ee`.
   Root's `open-saved-audit-01.json`:
   `72fd7d13b6f590ce03f0e3f2d7b7deb1befb7592cff389b646f010d1094e2b62`.

Metadata-only success is not full source admission. The ordinary loader still
performs the source, training and evaluation checks when the lab actually opens.

## Runtime comparison and fresh-open acceptance

The fixed commit contains the qualified lower-memory loader
`a721d4de760da115532ea4bb60c971318570a033d6903a6d9ec1275419e8260d`
and all 25 exact policy pins. Its full export is **not byte-equivalent** to the
prior retained-open runtime: two shared surface-report files differ, two modules
are added and the old copied `requirements.lock` is absent. One bounded Git-export
comparison recorded this; old payloads and the 173 MB run were not reread.
`runtime-comparison-01.json`:
`29b4268b9ac1cbf84d361e5ba5c10e4141f0afce0243e2213ac6373e37b4cdb3`.

Therefore the prior [two-retained-open proof](lab-loader-retention.md) establishes
the loader's implementation, not acceptance of this whole named runtime.
`open-plan-draft-01.json` prepared one fresh real `Lab.summary()` from the cache
already created above: 28 models, one data object, two forwards, one reset;
**zero controls, physics steps, replay, rendering, decode, export or training**.
It preserved ordinary source verification, compared 15 scoped public-summary
hashes to the prior accepted open and checked the original recording inventory.
One runtime hash pass; no extra full run/recording payload read beyond the loader.

Bounds: 40 CPU / 60 external wall seconds, 512 MiB observed RSS, 4 MiB output;
35 CPU / 50 wall cooperative budget leaves cleanup reserve. No retries, new API
spend or widened limits. A source/proof failure must remain a failure.
Draft SHA256:
`c803a9a372f19ce3ffba5136b51fdece3c8fb82c704ea5053a6084f0095e8f7a`;
worker:
`c2216a2955678237f6dce2cb5adab68b59a70650f996a40306ed91cd80251470`.
Root reviewed the worker and shared supervisor without findings. Frozen plan
`open-plan-frozen-01.json`:
`9c3b5d1db60fdee230b21a74da5bbef7e452cebca3fc2e3420dfa49a0f91a1f5`.
Only status/preparation metadata changed from the preserved draft. Root executed
it once; both acceptance children are terminal and reaped. No live handles remain.

Root also owns Git integration. No Git mutations, paid calls, source acquisition,
controller/scorer changes, avatar edits or native execution were performed here.
Root can publish this scoped launcher change. No additional acceptance run is
needed. The user can launch the named preset; general reconstruction, controller
success and complete build/export/reload capacity remain separate work.
