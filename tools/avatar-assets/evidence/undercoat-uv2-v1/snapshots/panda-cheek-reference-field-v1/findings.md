# Cheek reference field diagnosis

**The failed normal gate is mainly inherited contact geometry, not a custom-normal encoding problem.** No fit, geometry change or render was performed.

The exact old support mask is reproduced: 304 corners and normal-error p95 **63.67240187106602°**. Replacing the supplied custom normals with the actual triangle normals still gives p95 **64.24°**, above the unchanged 45° gate. Of the 24 supported outlier faces, 23 have an original-body edge crease over 30°; only seven have custom/geometric disagreement over 15°.

For all 132 original reference faces, the worst example is source primitive 0 / triangle 5416 (original partition face 31467). Its custom and geometric normals differ by only 0.003°, yet both point about 154° away from the smooth fitted field. This face was excluded by the old hemisphere-support rule; it demonstrates why small positional residuals do not imply healthy smooth reference geometry. Its adjacent original-body dihedral reaches 104.66°. Exact IDs and measures for every face are in `face-diagnostics.json`.

## Independent reference-only selection

The diagnostic reconstructed v7's original body/companion interface using its recorded 1e-6 m rounded-position policy. Only original body surfaces participate. A bounded two-edge-ring reference band, inside the fixed cheek bounds plus 20 mm, contains 196 candidate faces. No replacement-region expansion occurred.

Predetermined support rules retained 44 faces:

- At least two original-body edges from the contact interface.
- Maximum original-body neighbor dihedral at most 30°.
- Custom/geometric normal disagreement at most 15°.
- No original-body nonmanifold edge or near-degenerate triangle.

These rules do **not** use error against the failed fit to select faces. The 44 faces cover all eight coarse angular sectors. Their custom/geometric p95 is 5.35°. Their geometric-normal error against the **unchanged failed coefficients** is p95 23.57°; this is a diagnostic comparison, not a new passing fit or a reason to relax a gate. The maximum remains 65.82° and is retained. Exact positions and proposed geometric normals are in `proposed-reference-field.json`.

## Why refitting alone is not enough

Only four of the 78 fixed boundary vertices have exact support in this cleaner field. The current remaining-body boundary normal means differ from the smooth fixed field by p95 103.74°. Simply using the cleaner outer references while preserving the old scarred join would leave a major boundary problem.

The next justified change is a **targeted contact-scar ownership adjustment**, using the exact 24 supported outlier IDs and connected high-dihedral contact faces. Keep the 44 cleaner faces as protected reference evidence and keep unrelated body/companion surfaces protected. Test the resulting removal region and retained normal join before any new fit. This is different from another unqualified ring expansion or discarding normals merely because they failed the desired answer.

The exact 24 source primitive-0 triangle IDs are:

`4717, 4725, 4751, 4780, 4912, 4946, 4947, 4948, 4953, 4962, 4991, 5006, 5008, 5445, 5452, 5460, 5461, 5462, 5511, 5646, 5659, 5680, 5937, 6389`.

They are witnesses, not automatically an accepted new deletion mask. The protected cleaner field has zero overlap with them.

## Custody and resources

All source-face correspondences are exact within the declared 2e-7 m allowance; observed maximum error is zero. Source v7 SHA256 remains `734ebdbbd06c0428a57f817d1a5deed8ff7078016ab262abb313513d94f293ca`. The failed fit and its coefficients/gates remain unchanged.

The main bounded diagnostic completed in 0.783 s with sampled peak RSS 148,914,176 B; PID 57710 exited. Limits were 30 s / 256 MiB / 16 MiB. `reconcile.py` subsequently reaggregated only already-written JSON inputs, each capped at 1 MiB, to reproduce the exact old mask and boundary support; it read no mesh or image and performed no new fit. No native, provider, network or AI-role call occurred. No quality acceptance is claimed.
