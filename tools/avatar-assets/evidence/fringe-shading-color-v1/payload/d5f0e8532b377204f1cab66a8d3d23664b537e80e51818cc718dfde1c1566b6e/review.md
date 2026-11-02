# Fin shading review — measured hypothesis, not acceptance

**The fin shading normals are a plausible independent source of hard patches.** Their median angle to the original surface endpoint normals is **90.27°**; the original endpoint normals differ by only **4.72°** median. Source and fin materials both use metallic 0 and roughness approximately 0.92. The known long-edge color errors remain a separate, unresolved defect.

## Evidence

The one CPU pass read the unchanged source `e0488074…e5d5` and actual area-weighted candidate `8faa6bc2…00492d`. It inspected all 4,800 fins without decoding textures or rendering.

- 99.99% of the 9,600 fin/source endpoint comparisons lie between 60° and 120°. Median best-sign angle is 83.48°, so merely reversing all fin normals cannot align them to the surface.
- The fin normals agree with their actual geometric triangle normals. This is intentional side-facing plane shading, not corrupt or nonunit normal data.
- Source average normals differ from their associated source face geometry by only 2.04° median. The large discontinuity is introduced by the upright fin plane.
- A unitless Lambert-cosine diagnostic, using preserved preview light centers and geometric backface classification, differs by more than 0.25 between source and fin normals for 76.0% of the front-view source-facing subset under the key-light center. This excludes albedo, intensity, light area, shadows, environment, roughness, specular response and occlusion. It is not rendered brightness or a visibility result.
- Both materials remain lit PBR. Matching roughness/metallic constants does not make different normals shade alike. The base also has an undercoat normal texture; the fin does not. That texture was not decoded, so this comparison concerns serialized vertex normals only.
- The existing report still records **194/9,600 nominal tip colors with linear RGB component error >0.1**, maximum 0.6494. This audit does not repair or independently resample that defect. Normal changes cannot establish correct colors.

**Metric-label erratum:** `angle_after_best_global_sign_choice_degrees` in the sealed measurement uses `abs(dot)` independently for each endpoint comparison. It is a permissive per-comparison lower bound, not one globally achievable sign assignment. The numeric result is unchanged; no global-flip success is claimed.

## Double-sided reversal prevents a naive fix

glTF specifies backface culling for single-sided materials; double-sided materials disable culling and reverse back-facing normals before lighting. [Official glTF §3.9.5](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#double-sided), verified by the parent session.

A hypothetical normalized blend of 75% source-outward normal and 25% current fin normal is 18.46° from the source on the front side, but points inward after the required backface reversal. Among source-normal-facing fins, this occurs for **1,134/2,184 in front** and **1,144/2,191 in three-quarter**. These subsets are not confirmed visible pixels: body occlusion and alpha footprint are excluded.

The proposed reversed-winding duplicate can avoid that particular flip in a conforming renderer: use two opposite index triangles per original triangle, shared source-outward shading normals and `doubleSided=false`. One winding should survive culling from each side. Shared positions/UVs/weights/vertices can remain unchanged; total triangles increase from 72,181 to **81,781**, within the existing 100,000 bound. This is an authored shading workaround, not extra hair thickness, recovered surface geometry, or physical fur.

Important renderer boundary: the actual preview uses **Cycles**. The installed glTF importer only assigns `mat.use_backface_culling` from the glTF flag. Reading that assignment does not prove Cycles culls both coincident windings as intended, nor establish shadow/normal-hemisphere behavior. A small renderer control must establish that behavior before a full comparison. Otherwise the material/culling change is a confound, not an isolated shading-normal experiment.

## Contracts requiring an explicit new policy

| Existing contract | What it currently enforces | Required experimental distinction |
| --- | --- | --- |
| `avatar_assets/three_d.py:25`, `validate_render_attributes` | Finite, count-matched, unit normals; unit/orthogonal tangents and handedness when present. No geometric-face agreement test. | Preserve these checks. The new fin has no tangent or normal map; do not change the undercoat UV1 frame policy. |
| `tests/test_hd_uv1_attributes.py`, `test_uv1_still_requires_valid_frames_and_handedness` | Invalid normals/tangents remain rejected. These fixtures do not establish geometric-normal equality. | Keep the existing tests and bounds unchanged. |
| `outputs/edge-fringe-area-candidate-preparation-v1/sources/edge_fringe.py:160–164,222,238` | Fin normals are `cross(edge,direction)` and the report explicitly promises geometric fin normals. | Introduce a separately named, opt-in authored shading-normal policy; preserve the old default and describe the formula, source IDs, blend and side handling. |
| `outputs/fringe-allocation-analytic-v2/screen.py:105–111` and `outputs/edge-fringe-analytic-v1/screen.py:90–94` | Normal/geometric-face dot >0.99999 plus finite/unit normals. | Keep legacy geometric policy assertions. A new policy must verify its exact source-normal formula separately, not silently loosen the old assertion. |
| `outputs/panda-edge-fringe-area-audit-v1/audit.py:243–246` | Actual exported fin normals agree with triangle winding. | Future audit separates geometric winding/area from intentionally authored shading normals, retaining exact source/body/UV/skin/action preservation. |

All inspected code/test sources are hash-bound in `prepared.json`. No tests or production files were changed or rerun.

Minimum meaningful experimental guard: preserve source and existing fin positions, UVs, weights and actions exactly; validate finite unit shading normals against the pinned formula; retain nondegenerate geometric triangles; verify any reversed-index pairs and material scope; and prove the target renderer's front/back behavior on a tiny control. If normal and culling policy change together, label the comparison accordingly. An append-only new normal/color accessor and cloned experimental material can preserve original bytes and make the exact change reviewable.

The controlled next comparison should keep lighting, camera, alpha shape and animation fixed, retain a color-only control, and distinguish its result from color plus authored-normal/backside policy. The current numerical evidence does not select a final blend, accept an appearance, resolve merged alpha tips, or prove that each white speckle is shading rather than color.

## Resource and artifact receipt

One pass: **0.263 s**, PID 16526 exited; RSS **unsampled/null**, not zero. Limits: 15 s / 256 MiB / 8 MiB. All 14 pinned inputs and frozen worker sources remained unchanged. No native/provider/AI-role calls, model mutations, retry, or live handle.

`measurement.json` SHA256 `2df90d950256c5d893020a6d437656e280d7f87eef3f6efb60409f8bc27f4861`.
Execution manifest SHA256 `5f3d2e255e4806c027109c4559f067eb00d00c5c2cbaae726522d60db6888bcc` was sealed before this review. `review-receipt.json` binds this append-only interpretation and metric-label erratum to it.
