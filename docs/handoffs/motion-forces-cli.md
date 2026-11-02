# Motion-force check in the normal CLI

Verified November 2, 2026, PST. This exposes the existing source-force diagnostic
as `weft motion-forces`; it does not fix reconstruction or change a controller.
The user's current priority is substantive reconstruction/contact fixes, not
additional sandbox polish. Parallel agents own those fixes; root integrates.

## Use it

```sh
PYTHONPATH=src .venv-lab/bin/python -B -m weft motion-forces \
  --run artifacts/aspset-motion-native-01/lab/run \
  --out artifacts/my-motion-force-check
```

Use a new output directory. `index.html` explains sampled motor demands and links
all original targets, forces, source/task copies, installed implementation and
resource metadata. The saved lab stays unchanged. This is a separate companion,
not an automatically attached lab scorecard or measured force/contact evidence.

Current support: sealed human-only camera shared-world v6, complete nominal
22-axis interval and explicitly enabled three-axis root assistance. Other body,
equipment and disabled-root cases reject rather than silently changing controls.
Midpoints sample constant-speed linear interpolation; pose-change accelerations
are explicitly finite-window probes. Contact-constrained and constraint-free
results stay separate; no samples means unavailable, not zero error. Sampling
cannot certify the complete path. Mass/inertia/forces remain authored assumptions.

## Bounds and verification

The public command uses cooperative checks between bounded reads/native operations:
default 30 seconds, maximum 60, 512 MiB observed peak RSS, 512 input members/
256 MiB, 900 poses/1,798 inverse calls, and 64 MiB/160 output members. It is not
hard native-call preemption or an OS sandbox. No physics steps, training,
model inference or paid request is made. Post-preflight failures preserve their
new inputs, runtime and failed receipt; invalid inputs do not create an output.

**75 distinct checks pass**, including 34 new workflow/CLI cases:
`.local/motion-forces-tests-01.xml`. Coverage includes analytic mass/damping,
force balance, unknown constraint-free values, source/path/hash/control rejection,
cooperative cancellation, retained failures and native-state closure.

Actual public CLI acceptance used a frozen runtime with exactly the three new
CLI/diagnostic source files overlaid. All 54 loaded implementation modules equal
the working source. Control: `.local/motion-forces-cli-01`; worker PID5477 is
terminal/reaped. Supervisor: **0.937 s**, exit0, no stop reason. Worker:
**0.846 s /208,601,088 B**, four model compilations, two data allocations,
92 inverse calls, zero forwards/steps. External acceptance bounds were
30 wall/20 CPU seconds, 512 MiB RSS and the output bounds above.

Result: `artifacts/motion-forces-cli-01`, 63 sealed members, 64 files/
2,243,872 bytes including manifest. Seal:
`8d717a9e06a90c8e3494f4f8371a9619cd5b2b916dcc15e3492296ab515bca0f`.
The entire native diagnostic equals the previous motion-prior diagnostic exactly:
14/44 constraint-free midpoints exceed caps (up to2.129×), as do all42 interior
transition probes (up to9.866×). No improvement is claimed for exposing this data.

Saved-only verifier `.local/motion-forces-cli-01/verify_saved.py` checks all528
members across source/output/previous diagnostic, 317 frozen runtime members,
2,024 scalar force balances, exact source/task copies and HTML links. Receipt:
`saved-verification.json`. Source seal remains
`e08fb8aceca34fcfdeeaec0f94dd452de95eed674884dd9133f8a23f22813b3f`;
previous diagnostic seal remains
`61ac8ac5f4a7dab61cf819fa0d54577696bcb8aa61d2e738a5ead69f63cb56f1`.

## Next work and ownership

Root owns the CLI integration, checks and scoped publication. Separate agents own
`limb_twist_gauge.py` (source-constrained orientation),
`equipment_temporal_silhouette.py` (multi-frame geometry under explicit static
priors), and a bounded unfamiliar exercise source acquisition. Their files and
separate avatar edits must not be staged in this checkpoint. Integrate qualified
changes through the original world/task, then compare complete-interval controls
and exact recording; preserve source-image regressions and every failed gate.
No API/key/ledger access, browser/server/deployment or new physical claim.
