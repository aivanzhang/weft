# Contour and contact diagnosis

Recorded November 1, 2026, 23:22 PST. These are local development results, not a
new accepted avatar. Approved panda/dragon assets remain unchanged. The public
portable-fringe option stays unsupported. No new provider or pipeline-role AI
calls, deployment, server or browser session occurred.

## Reproducible evidence

Every directory below retains its controller/configuration snapshots, receipts
and outputs. Do not rerun into existing folders. Native work used the saved
exported model, 720px EEVEE/16 samples and bounded single-worker execution.

| Output under `outputs/` | Result |
| --- | --- |
| `panda-contact-ablation-v1` | Full control pixels exactly match the prior exported preview. Coat-hidden, constant-gray and unlit-basecolor controls isolate different causes. Gray retains custom normals/shadows; it is not geometry-only. 2.857 s, 1,039,089,664 B sampled peak RSS. |
| `panda-contact-pixel-attribution-v1` | Seventeen fixed pixel-center rays hit panda body materials: fifteen body-surface and two authored closure faces. None hit the companion. This is not full antialias/mipmap coverage. 1.036 s, 285,835,264 B sampled RSS. |
| `panda-contact-defects-diagnosis-v1` | Exact source correspondence separates original, repainted and closure faces. A cap material is not proof of authored-only geometry. Initial census exceeded 512 MiB at 591,101,952 B and was terminated; preserve that failure. Smaller finalizers succeeded. |
| `panda-cheek-source-normal-payload-v1` | Hash-bound, unique position/UV correspondence for 198 faces / 594 corners, not a guessed nearest match. |
| `panda-cheek-normal-ablation-v1` | Baseline pixels match the prior gray control. Restoring source normals changes only the selected corners. Dark contrast improves, but the raised angular patch remains. 2.079 s, 846,905,344 B sampled RSS. |
| `portable-fringe-profile-diagnosis-v1` | Actual upright fins explain bristled outlines. Shorter regional flow reduces contour height but also reduces projected coverage; it is not a density solution. |
| `fringe-flow-tests-v1` | Ten analytic tests pass. The 4,800-fin maximum fixture runs in 0.01244 s; self-reported high-water RSS 53,493,760 B, monitor peak unsampled. |
| `panda-regional-fringe-candidate-v1` | Actual GLB derivative preserves roots, body, topology, normals, UVs, skin, textures and actions. Only fur tips and position bounds change. 0.523 s, 251,576,320 B sampled RSS. |
| `panda-regional-fringe-preview-v1` | Four views and four sampled poses in 3.376 s, 1,029,079,040 B sampled RSS. Shorter plush improves outlines but leaves cheek/shoulder defects. Camera auto-fit uses new bounds, so this is not a pixel-registered comparison. |
| `panda-regional-fringe-independent-review-v2` | Independent byte audit passes: 21,600 tip rows / 183,918 bytes and six bounds change; all other JSON/binary bytes remain exact. 0.262 s; self high-water RSS 111,460,352 B, monitor unsampled. Initial wrong-interpreter import failure remains in v1. No new motion evaluation. |
| `panda-cheek-patch-topology-v1` | No healthy connected disk within rings 0–3. Two smaller patches have sound attachments; the 124-face patch has three bad body-side junctions. Main screen 0.519 s, 78,954,496 B sampled RSS; attachment check 0.263 s, RSS unsampled. |

All owned workers exited. Stills and unchanged animation bytes do not establish
a new full-animation, collision, seamless-loop or appearance acceptance.

## Reusable component and exact identities

`scripts/fringe_flow.py::reshape_fringe` samples the existing regional groom
field at each root midpoint, after unchanged root selection. It preserves the
six-array mesh contract and rejects invalid inward displacements. Bounds:
4,800 fins, 1–40 mm regional length and 100k export vertices/triangles. It is an
experimental helper, not a new public preset or automatic finisher route.

- Helper SHA256: `b7d60d5e9f152edf063ed9ce140b187dafe018e82470c66eb695aaccf8940ed9`.
- Test SHA256: `5e5cc333ccd9018ef1a864020bc483843f08888a3e3b8ea90c63ccd20df1b41a`.
- Candidate `panda-regional-fringe-candidate-v1/model.glb` SHA256:
  `257ff11efdf31eaa3a7ab77b527b7b45a48f97bcf45c498985e9936b1069c5dd`.
- Independent review manifest SHA256:
  `f6ef40ad768b2c7ef381a4ab6a35dec7b207fd4ba3637287a70c0dffcee0b9c2`.
- Topology manifest SHA256:
  `3eeef25a55eba0e57f060457edd820db5132efe3fcc4490f76d260bfa23e7fe9`.

Focused test command, already run against the recorded sources:

```sh
cd /Users/ivan/kintsugi/weft/tools/avatar-assets
.venv/bin/python -B -m unittest discover -s tests -p test_fringe_flow.py -v
```

## Next safe actions

1. The [normal-transfer guard](normal-transfer-guard.md) is now implemented and
   verified on the actual source and native Blender path. It rejects 172 unsafe
   transferred corners before normal writes and retains bounded witnesses. This
   is prevention, not repair; do not suppress it for the old panda recipe.
2. Use `panda-cheek-patch-topology-v1/boundary-attachment.json` for bounded local
   fan/attachment resolution at the three named junctions before a curved patch.
   Do not merge coincident body/companion vertices or repeat failed flat caps.
3. Treat shoulder color separately: explicit protected-white/body selectors and
   a padded UV/palette repair. Current constant-material corrections exclude
   repaired vertices from fur and would recreate a bald patch.
4. Review actual new native/exported output and preserve failures. Compact
   evidence is now in `evidence/contour-contact-v1`; the scoped publication result
   is recorded separately in `outputs/contour-contact-checkpoint-v1/publication.json`.

Fresh text-prompt Studio acceptance still needs its pending allowance. The
existing 8 Meshy credits cannot cover a newly estimated 35-credit model; no
allowance was silently expanded. The prior 13-call AI-role study is exhausted.

## Source-repair follow-up

The targeted topology refinement distinguishes old whole-body links from the
body that remains after replacement. Two junctions become healthy when their
old cap faces are replaced. The third requires explicit retirement of original
body primitive 0 / triangle 4961: a 0.205 mm² dangling flap. With that retirement,
the 124-face loop-27 patch has 78 fixed, consistently oriented boundary edges and
healthy remaining-body attachment. The flap and all selections are retained in
`outputs/panda-cheek-targeted-selection-v2`.

The subsequent curved candidate is **unsupported**. Its three-iteration robust
Hermite fit has 1.328 mm positional RMS error but 63.67° 95th-percentile reference
normal disagreement, above the unchanged 45° gate. It stops before displacement,
intersection and export checks; no `model.glb` exists. Preserve
`outputs/panda-cheek-curved-candidate-v1`. Next: diagnose contradictory reference
normals/crease faces or select a justified cleaner reference band, not more blind
replacement-region expansion or a relaxed acceptance threshold.

The separate UV candidate is `outputs/panda-shoulder-uv-repair-v1/model.glb`, SHA
`53aff0bba281367f4a7a81f2038adb2484409bc3440eb172e0884ef7a0ba4528`.
Its 242 selected faces include all 17 center-ray witnesses. Original triangle
positions, normals, tangents and image bytes are retained; selected textured
material clones use new color UV0 and preserve old normal/metallic-roughness
lookup through copied UV1. The plan protects 244 white candidates and all
companion faces. Planning and production take 2.609 s total, with maximum
419,774,464 B sampled direct-child RSS and one texture decode.

Actual matched color-only review is in `outputs/panda-shoulder-uv-ablation-v1`.
Two 720px renders take 2.588 s / 878,428,160 B sampled RSS. Baseline pixels exactly
equal the previous unlit control. Only selected UV0 corners change in the
transient posed scene; geometry, normals, UV1, skin and pose are unchanged.
Thirteen of sixteen fixed green witnesses become dark neutral pixels. Three
remain nearly unchanged; the bright witness drops from RGB `[194,191,190]` to
`[117,113,112]` but remains visible. **Partial improvement, not completed repair.**
Center-ray selection does not prove full antialias/filter coverage. Further
attribution must preserve protected white regions rather than broadly paint the
shoulder dark.

This derivative is unrigged source work. The transient posed-scene control does
not give it a verified rig or animation. Combining a future geometry patch with
the UV repair requires explicit retained-face correspondence, fresh skin/motion
checks and regenerated fur/undercoat. Disjoint face selections do not prove
disjoint attribute storage; triangle ordinals change after replacement. No
source repair is promoted into automatic finishing or the approved gallery.

## Subpixel-guided UV follow-up

`outputs/panda-shoulder-subpixel-attribution-v1` traces 305 rays with the actual
material culling flags. The two corrected controls have 49/49 selected hits;
the three residual green pixels have 27/49, 25/49 and 28/49 selected hits. Eight
unselected neighboring triangles contribute; no sampled near-depth overlap is
found. These regular grids are not the engine's exact antialiasing pattern.
The trace takes 2.345 s / 311,574,528 B sampled RSS and produces no render.

Exact correspondence identifies seven dark contributors and one protected white
face. Candidate v2 adds only v7 primitive-0 triangles
`1579,2384,2395,2571,3128,3336,3411`; triangle 2395 shares an edge with newly
attributed 2571 and is explicitly admitted from measured local evidence. Keep
triangle 5167 protected: 29/91 saved samples meet the unchanged white rule.
No larger spatial region, texture decode or threshold change is introduced.

V2 model: `outputs/panda-shoulder-uv-repair-v2/model.glb`, SHA
`736ad72d408a43295098f44a29919c6436b37f741b7335f35a2c3f5f2007fb3e`.
Independent review verifies all 62,581 oriented triangles/P/N/T, 62,332
unselected color-UV faces and exact original material/image routes. All 493
selected/protected current faces resolve uniquely to v7; all 244 protected
white faces remain unchanged. Review manifest:
`f824be233c6bda5720499f8b44bbb35e41c87ad8739c9ee34fdecaedd5483903`.
Its 1.279 s CPU check samples 331,415,552 B RSS. The first v1 checker schema
failure is preserved; correcting ID-space correspondence did not change gates.

`outputs/panda-shoulder-uv-ablation-v2` renders one new image and reuses the
byte-exact baseline only after all protected state matches. Fifteen of the 16
fixed green witnesses are dark; the grazing sixteenth is neutral gray
`[66,68,67]`, not black. The bright witness stays `[117,113,112]`. Sixty pixels
change from v1. Runtime: 2.598 s / 866,467,840 B sampled RSS. Both reviews agree:
color improves, but small flecks/bright marks remain and full quality is not
accepted. No model rig, fur or animation is inferred from this one unlit view.

The compact checkpoint contains 647 original records / 491 exact snapshots,
17,181,529 B, at `evidence/contour-contact-v1`. Manifest SHA:
`0039e838229601975246c06c0f48423eb5c5d0f47ff7c05fc6a1521c70ea6445`.
Large models/scenes remain local with hashes, not remote binary backups.
All owned jobs ended; no new Meshy credits or pipeline-role calls were used.
