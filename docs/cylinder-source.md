# Real cylinder: depth-motion tracking test

A new real non-ball source exposes a fixed-size tracking limitation. The unchanged appearance tracker returns only one of four scored source centers, with 5-pixel error against a frozen 4-pixel tolerance. The other three remain missing. No simulator or learned policy was created from this result.

## Source and review

Source: [Aufwärtsrollender Doppelkegel](https://commons.wikimedia.org/wiki/File:Aufw%C3%A4rtsrollender_Doppelkegel.webm), by Rhetos. The Commons page declares CC BY-SA 4.0; the video's end credits visibly say CC BY 4.0. Both statements are preserved, not silently reconciled. The source sidecar records the Commons license and attribution. The demonstration includes a yellow cylinder and blue double cone, hand manipulation, camera changes, explanatory overlays and edited sections.

The official complete 240p VP9 transcode is retained unchanged as `data/general/bicone-cylinder-240p.webm`: 2,015,504 bytes, SHA-256 `b50fa16f49425f7c2fc959f425bc008d978daa8c9472db616d82747c35fc941d`. Page plus video acquisition used 2,111,503 bytes, within the predeclared 30 MB bound, with no paid services. An initial sandbox DNS failure is retained; the narrowly authorized public download subsequently succeeded.

Sequential decoding returns 3,936 frames at 320×240 and nominal 25 fps, matching the metadata count. Decoder timestamps run from 0.006 to 157.406 seconds. This is decoder coverage, not independent encoded-frame certification. The [whole-source overview](../artifacts/general-cylinder-source-review-01/overview.png) preserves cuts, altered viewpoints and the video's labeled slow-motion section; that slow-motion section is not used for this tracking trial.

The explicitly selected segment is source frames **2630–2805 inclusive**, with initialization at **2695**. It retains 65 pre-initialization frames, the released cylinder's motion, retrieval and hand overlap. It is a segment of the first cylinder demonstration, not the complete setup approach or entire 157-second video. The complete original remains available.

In this front view, the cylinder moves substantially in depth: its projected width shrinks while its center shifts relatively little. The two rails define support geometry that is not frontoparallel. A hand later lifts the cylinder. These are not equivalent to a freely coasting constant-size 2D box. No gravity, friction, force, rigid 3D shape or camera calibration is inferred here.

## Frozen source labels and unchanged baseline

Five approximate body-envelope centers were authored from source-only images before tracking; the first initializes the appearance rectangle and is excluded from scoring. Radial uncertainty is 4 pixels. These are assistant estimates, not expert labels or centers of mass. Hand overlap remains visible at the last slot.

| Source frame | Source center | Baseline result |
|---|---|---|
| 2695 | (164.5, 107.0) | Authored initialization, unscored |
| 2720 | (165.5, 108.0) | Returned (160.5, 108.0); 5.0-pixel error |
| 2745 | (165.5, 111.5) | Lost |
| 2770 | (166.5, 114.0) | Lost |
| 2805 | (159.5, 96.5) | Lost; hand overlap retained |

The default independent tracker retains all 176 rows: 65 pre-initialization, one initialization, 26 tracked, nine not detected and 75 lost. Appearance score falls below the unchanged 0.7 threshold at frame 2722; loss follows the existing gap limit. Its translated rectangle stays 177×16 pixels despite visible shrinkage. This is evidence for testing scale-aware appearance matching, not proof that scaling alone will resolve depth, occlusion or physical reconstruction.

The background screen reports `not_verified_fixed`, not a verified fixed camera. Its nominal row times are source-frame/fps; the separate decoder audit preserves the 6 ms media offset. Frame identities, not silently rebased times, link the labels.

## Evidence and next step

- Source review: 72-file seal `cca75a72350ff7bd995b323e63117bfbf4597e37d635db1e3361e9dfcb9b7e49`.
- Unchanged tracking: 14-file seal `839fd309e855137245cb4edd52b09b29ec26af501119e870b84a95a8c9454c16`; 0.154-second tracking runtime, one local observation rather than a performance guarantee.
- All source and implementation hashes remain unchanged. No paid tracking, physical parameter fit, environment generation or training ran.

The [bounded scale-template experiment](scale-tracking.md) is now implemented after synthetic shrink/grow, rotation, occlusion, distractor and missing-data controls. It returns two of four scored centers, only one within tolerance: frame 2720 worsens from 5.00 to 14.04 pixels, while newly recovered frame 2745 has 2.06-pixel error. No physical model is generated. Independent extent/boundary evidence is the next requirement, not more accepted rectangles. Keep original identity/color evidence and the four frozen scored slots. Do not reduce thresholds, turn hand forces into zero-force dynamics, reinterpret image width as a changing physical body, or call this partial source interval whole-video reconstruction.
