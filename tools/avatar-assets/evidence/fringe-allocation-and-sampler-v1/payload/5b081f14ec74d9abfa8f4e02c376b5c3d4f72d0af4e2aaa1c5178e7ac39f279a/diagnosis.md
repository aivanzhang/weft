# Root color sampler review

**No production texel-center or gamma correction is needed.** `RootColorSampler` already uses `mod(uv,1)*[W,H] - 0.5`, wraps neighboring texels, and decodes sRGB before bilinear filtering. The endpoint-grid `uv*(size-1)` error belongs to the frozen experimental edge helper. Its corrected candidate changed sampled linear color by up to 0.38430118560791016.

## Concrete production gap

`shared_color_texture` rejects non-REPEAT addressing and custom UV routes, but accepts every image interpolation mode. The exact tuft path then always samples bilinear LOD0. A tiny extracted-code fixture admitted `Closest`, `Cubic` and `Smart`; at one midpoint, nearest gives `(0,1,0)` while the root sampler gives `(.5,.5,0)`. The actual panda source uses linear-filtered REPEAT, so this is a generic input-contract defect, not the cause of its observed coat quality.

## Consumer boundaries

| Path | Actual behavior | Keep/change |
| --- | --- | --- |
| Tuft `RootColorSampler` | REPEAT, normalized texel centers, bilinear LOD0, per-texel sRGB decode | Keep numeric behavior; validate source interpolation |
| Mask/band `texture_colors` | Nearest repeated texel from each vertex's first UV loop | Preserve legacy approximation |
| Legacy/combed card color | Barycentric interpolation of decoded vertex samples | Do not relabel exact root sampling |
| Native hair | New image node; source wrap/filter not explicitly copied | Defaults not independently probed here |
| Portable atlas | Explicit CLIP; exported CLAMP_TO_EDGE, linear mip sampler | Separate from root-source sampling |
| Undercoat | Explicit REPEAT/Linear Non-Color normal maps; basecolor graph retained | No root/gamma change |
| HD surface generator | Procedural authored textures, not source-image sampling | Unrelated |

Blender bottom-up image rows pair with Blender UVs; PIL top-down rows pair with glTF image-V-down UVs. Do not insert a second V flip. Root samples are LOD0, not GPU mip/derivative equivalence.

## Smallest versioned fix

Use policy `repeat-texel-centers-linear-lod0-srgb-v1` in exact-root reports. Keep the `RootColorSampler` API and math unchanged. Add strict Linear/REPEAT/direct-UV validation at the exact-root adapter boundary, including agreement between sampled and render-active UV layers. An optional strict flag or separate helper preserves the legacy admission path. Unsupported modes should reject, not be silently changed. Supported existing Linear/REPEAT tuft output remains numerically identical; legacy/combed profiles retain their outputs.

Do not edit prior sealed edge helpers. If that experiment becomes reusable, opt it into the verified policy and share its math instead of maintaining a second endpoint-grid sampler. This does not promote a geometry profile or change any quality threshold.

## Verified and proposed tests

The saved read-only fixture checked nine 2x2 centers/seams/negative and integer-repeat coordinates plus paired row orientation. Existing production sampling exactly matched the corrected candidate kernel and independent expected values. Separate gray/white checks verify decode-before-filter and linear bypass. The old endpoint helper differs by up to 0.5 in this analytic fixture.

For a released implementation, add these to the production sampler tests and add strict-mode rejection tests for Closest/Cubic/Smart and UV-layer mismatch. Preserve existing legacy/default byte fixtures. No native render, actual texture decode, or production/test edit occurred.

Exact sources/hashes: `source-pins.json`; numeric evidence: `analytic-checks.json`; full consumer/compatibility assessment: `report.json`. Command: `.venv/bin/python -B outputs/root-color-sampler-review-v1/check_contract.py`.
