# Portable fur visibility audit

**Result: sparse projected geometry and unresolved fibers dominate; back-face culling is secondary in these two views.** This is a CPU diagnostic, not a new render or appearance approval.

Source GLB: `e0488074965a89a9f7eda7788ebb26e7a8f14f98bbb44e1d9856fa091915e5d5`. The saved exported preview uses those same bytes. Its front and three-quarter rest-camera recipes were reconstructed from the immutable preview code; CPU rest bounds match the saved native bounds within 0.2µm. No Blender file was executed.

## Measured separation

| Quantity at 720px | Front | Three-quarter |
| --- | ---: | ---: |
| Visible fur-selected body pixels | 173,683 | 147,093 |
| Visible opaque front-card pixels over that body | 4,148 (2.39%) | 3,752 (2.55%) |
| Actual atlas, bilinear LOD0 alpha pixels | 1,110 (0.64%) | 990 (0.67%) |
| BOX-mip/trilinear modeled alpha pixels | 3,522 (2.03%) | 3,242 (2.20%) |
| Additional visible LOD0 alpha pixels if cards were two-sided, including silhouette | 35 | 41 |
| Additional modeled mip-alpha pixels if two-sided, including silhouette | 111 | 115 |

Although back-face removal discards about 46–47% of summed card projected area before occlusion, disabling it recovers only about 3–4% of visible alpha pixels. Most back-facing cards are already behind the opaque body. Even removing alpha entirely leaves only the opaque-card coverage shown above; filtering cannot supply missing surface geometry.

The 2,400 ribbons have median maximum row widths of 1.53–1.63px across the two material bands/views. Ten fibers occupy each ribbon: a whole-width/fiber-count spacing proxy is only 0.15–0.16px. At 360px, 70–74% of ribbons are narrower than one pixel; at 180px, all are. This proxy does not count resolved GPU strands.

The actual embedded atlas has 27.09% alpha occupancy at cutoff 0.18. The modeled mip chain increases the visible masked footprint roughly threefold rather than removing it. Fine fibers merge into small opaque-looking patches. Actual GPU mip generation, anisotropy, alpha antialiasing and lighting remain unmeasured.

## Bounded next decision

Do not prioritize disabling back-face culling, increasing atlas resolution, or adding native groom children: none addresses these measured limits. Screen the separately proposed representation change against these same cameras and measurements before another expensive appearance run. It should change where meaningful projected geometry survives and provide resolved features, rather than only increase the summed hidden card area. This audit does not set a new quality threshold or approve a representation.

## Scope and evidence

- `audit.json`: continuous projected-area upper bounds, per-band footprints, actual source material/sampler data and both camera reconstructions.
- `front.json`, `three-quarter.json`: per-view measurements. Corresponding `*-analytic-masks.npz` files are CPU diagnostic masks, not rendered avatars.
- Pixel-center depth uses the actual double-sided opaque body. Card culling follows the single-sided glTF contract. Cycles need not enforce that flag identically; no renderer behavior is inferred.
- Pixel-center sampling can miss subpixel geometry and depends on phase. Continuous summed area is also reported but overcounts overlap and ignores occlusion.
- Only rest views are measured. Animation, contact defects, undercoat lighting and perceived softness are not verified.
- The first attempt stopped before projection because its parser incorrectly required VEC4 colors. Actual cards use VEC3, whose alpha is implicitly one. V1 remains unchanged; V2 only accepts that exact case or explicit constant-one VEC4.

V2 completed in 2.603s with 187,318,272B sampled direct-child peak RSS and 361,722B output at job completion; the process exited. Combined attempted CPU wall time was 2.865s. Limits stayed 60s aggregate, 512MiB memory and 32MiB output. All 14 bound source inputs stayed unchanged. No native, provider or model calls occurred.

Audit SHA256: `08001baf3cc86d12f67fbac463dd88f86033677f10c54d52e0f2b870fd9558c2`.
Execution manifest SHA256: `d77cd85c12522ab13045aeb132d4ea9299c9d86a63e452c2859a9bf111227943`.
