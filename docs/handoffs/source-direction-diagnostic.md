# Source-only simultaneous motion directions

## November 2, 2026 (PST)

Follow-up to the rejected [root/leg source fit](source-ground-contact.md).
`root_leg_direction_diagnostic.py` analyzes simultaneous camera/root/leg changes
without changing a body, source observation, clock, floor, task or evaluator.
It is a diagnostic API, not an installed repair or a new portal mode.

## Result and interpretation

The unchanged native squat inputs and rejected candidate were revalidated through
the original source/decoder bindings. One bounded geometry pass retains all
34 source references and all source/30 Hz samples. The image-only derivative
matrix has **748 rows, 374 variables and numerical rank 356**: 18 directions are
locally unconstrained by image data under the fixed authored camera/body gauge.
Rank uses a relative tolerance of `1e-10` and documented angle/root column scales;
floor constraints, priors and solver regularization do not contribute rank.

The bounded eight-direction search finds **no verified improving direction**.
Its original solver reports dependent constraints. Saved-matrix-only analyses
handle redundant rows and conservatively restrict dependent directions; the
final search still has no verified improvement. Every original inequality remains
part of normalized-direction verification. A negative result in this selected
subspace is not global infeasibility, proof of uniqueness or a repaired motion.
No finite correction, nonlinear candidate trial, rollout or promotion ran.

## Reproducible evidence

Study: `.local/root-leg-direction-01`.

- Original plan SHA256:
  `21a2562fb6de9bfa277018809d7527137e04b7033b47c096cb57c510439e2f40`.
- One geometry analysis: **0.329 s /102 MB RSS**, 214 kinematic calls,
  470 original constraints and 29 failing-floor targets. Result:
  `3b8c9c81425c8880f001d68bad790e2098eb48c8e8ec566c95ba19ce71752060`.
- `cone-01` and `cone-02` retain earlier saved-matrix outcomes; neither repeats
  geometry. `cone-03` preserves a NumPy scalar JSON-serialization failure.
  Corrected `cone-04` completes in **0.079 s /44 MB**, three solver rounds,
  eight reduced directions conservatively restricted to five, zero direction gain.
  Result: `53d06b5a7c239c64a908e5f9361979d2a67e7f8f3e4f6e34c0cc580820aa189c`.
- Complete original constraints, mode basis, normalized direction checks, singular
  values, implementation snapshots, input hashes and process results remain saved.
- Limits: 64 complete references, 256 source/grid samples, 1,200 geometry calls,
  4,096 constraints, eight search variables, 20-second geometry analysis and
  16 MiB output. Each saved-cone analysis is at most 32 rounds/five seconds.
  Parent supervisors enforce wall/memory limits; API and physics calls are absent.
- Study seal: `19acd3b747787d7bdd1dc6915878a3efc476be900aedf7eb6062e7083afa321f`.

Seventeen focused cases cover coupled-versus-separate directions, source-only
rank, underdetermined image matrices, normalized constraint checks, dependencies,
redundancy, native inputs and source/clock/gauge/protected-joint rejection.
Independent final review found no acceptance blocker; unchanged passing checks
were not rerun by the reviewer.

## Resume without repeating failed work

Keep the rejected source fit and all diagnostic outcomes. Do not rerun identical
fits, widen the floor/image gates or treat this local tangent search as a proof
that all possible reconstructions fail. Weak selected image modes mostly affect
the right leg while remaining floor failures affect the left shin; numerical
ambiguity does not itself supply a contact repair.

A materially different next hypothesis is source-constrained limb shape/camera
estimation shared across time, with explicit uncertainty and capable simultaneous
constraint handling. This diagnosis does not establish that shape is wrong.
Any proposal needs predeclared bounds, unchanged original task/gates and an
independent finite-step audit; no per-clip floor shifting. More saved-matrix
variations alone do not finish the product.

The [portal checkpoint](portal-inventory-and-import.md) is already usable and
published as `2a79f4f`; its imported equipment candidate remains unpromoted.
Prioritize automatic mixed input reconstruction and verified Muse construction
attachments alongside the physical-method work. No API budget or deployment
change is included. All diagnostic workers are terminal; separate avatar work
remains untouched. Publication receipt:
`.local/source-direction-publication-01.json` after remote verification.
