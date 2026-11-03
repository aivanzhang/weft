# Preserve competing-pose identity breaks

November 2, 2026 (PST). Additive diagnostic only; no body or pipeline default changed.

`src/weft/pose_identity_windows.py` exposes
`screen_identity_windows(evidence_bytes, raw_pose_bytes)`. It requires exact raw-mode
pose rows and their original artifact hash, complete declared stride inventory and
unchanged nominal frame times. Inputs have a 16-MiB aggregate bound and at most
1,800 samples. Both original documents remain in the result. No pose is selected,
rewritten, interpolated or silently assigned to another person.

Competing samples become explicit identity barriers. Separate windows screen only
original minimum count (8), usable fraction (.65) and duration (.4 seconds). Motion,
gaps, stable identity, native clock, camera/root, floor and physics remain untested.
A passing window is not a body, executable episode or full-clip success.

## Verified result

**22 focused tests pass** in `.local/pose-identity-windows-03-tests.xml`.
One saved-only source study `.local/pose-identity-windows-02` takes0.118 seconds /
51,101,696 bytes peak RSS. Limits:10-second wall/8-second CPU/128-MiB RSS sampled
at50ms/16-MiB output. No decoding, inference, network, API, simulation or training.
All original inputs remain unchanged; watchdog joined and alarm cleared.

All **153 sampled slots**, not verified identities for458decodedframes, remain.
Both raw hypotheses at frames111/123 are preserved with no chosen winner.

| Original source frames, inclusive | Usable/sampled | Usable span, nominal source time | Count/coverage/span only |
| --- | --- | --- | --- |
| 0–108 | 35/37 | 3.599843s | Pass |
| 114–120 | 2/3 | .099996s | Fail count and duration |
| 126–456 | 106/111 | 10.899524s | Pass |

Original full-clip quality refusal and incomplete decoder/native capture remain.
The helper cannot turn that failed package into completed body evidence. No
selected window or shortened task is produced. Root independently verifies all
8 sealed members, both complete raw documents, every sample's unique window/barrier
membership, unchanged helper implementation and all explicit unqualified flags in
0.055 seconds (`root-audit.json`).

- Result SHA256: `1a5d44663e80b3f6ffc70064ed644260a370faa3d796422317c90888507f0eaf`.
- Manifest: `9cf644f5402cc10c5a50a1db20fd7df80ebdf9f07f85c1c2af4611e4865dbe8b`.
- Source: the same new kettlebell-flip derivative `c289fc1afe0b60a418cc6ed9e4554c8414a4d970d2dfcaea1f6f82839204dac3`.
- Original evidence lives under `.local/portal-kettlebell-flip-01/portal/a85537e09e284facb10de22cf2a7dcbb/perception/preflight/body_source`.

Attempt01 failed because sandboxed process RSS inspection was unavailable. Preserve
its failure. A later exact-path process check found no active worker; this is not
retrospective proof of its exit code or reaping. Attempt02 used in-process RSS
checks. A test fixture's decimal-rounding assertion was corrected to compare the
exact original decoder value; original failure XML remains.

## Next action and ownership

All lanes are released; no live handle. Root owns publication. A separate
bounded study tested the locally pinned **Full** pose model on the same complete
video with unchanged two-hypothesis/visibility/gap/floor rules and v2 tracking hint.
It also fails body admission: **4 competing samples** at frames111/114/120/123,
versus Lite's2. Full has142/153 usable (92.81%),5low-confidence and2no-pose samples.
It takes5.388seconds/410,664,960bytes peak RSS; all339source/model/runtime pins
remain unchanged. No completed native body/seed/anchor, API, training or default
change. This different-model hypothesis did not resolve identity; do not repeat it.

`.local/kettlebell-flip-full-preflight-01/{plan,comparison}.json` and `HANDOFF.md`
retain exact settings, failures and output. Model SHA256
`5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1`;
39-file study seal `bc86005300777bb3c4b98725a60b7ebf29f47ba5f95237b191908972b6ed6836`.

If full-clip identity remains unresolved, a future source-window episode must
retain the original full source/refusal and barriers, obtain complete source/native
capture, apply all body gates independently, and label the shorter episode separately.
No such constructor is implemented. Continue the complete video-to-world/contact/
learning goal without treating these three screens as completion.

The tested portal checkpoint `a9a8fd808ba4f933651cec58579f0eaa1aa3f8d1` is published.
Inspect the retained local portal failure without rerunning inference using:

```sh
.venv-lab/bin/python -m weft portal --root .local/portal-kettlebell-flip-01/portal --port 8766
```

The user manages the server. No deployment or browser launch. Preserve avatar work.
