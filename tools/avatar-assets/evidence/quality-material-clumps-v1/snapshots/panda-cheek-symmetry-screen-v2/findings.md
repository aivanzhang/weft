# Opposite-cheek reference screen

## Result

The unobstructed side supplies a nearby smooth shape reference, but reflection alone cannot satisfy the frozen patch boundary. No repair was built.

The corrected screen examines 14,518 original-body donor triangles; 11,442 meet the fixed local dihedral/custom-normal criteria. All nearest matches in this test already belong to that cleaner subset. Companion/cap triangles and authored eye/nose/muzzle exclusion regions are excluded. No texture was decoded.

| Independently defined reflection plane | Stopped-boundary distance p95 | Custom/geometric normal p95 | Fully supported edges |
|---|---:|---:|---:|
| Eye midpoint, world X normal | 19.792 mm | 81.13° / 85.39° | 19/65 |
| Nose center, world X normal | 18.293 mm | 81.14° / 85.39° | 25/65 |
| Head bone X plane | 6.842 mm | 81.25° / 82.95° | 52/65 |
| Eye-pair separation normal | 20.146 mm | 79.68° / 83.00° | 9/65 |

Support means all three samples on an edge satisfy 15 mm distance and 45° custom/geometric normal limits with a clean donor. The p95 values are this screen's 195 edge-endpoint/midpoint samples, not the earlier fit's 304 corner samples. No plane was fitted to the failed patch or promoted.

## Coordinate and anatomy limits

The preserved source uses glTF Y-up; the recorded authoring guides use Blender Z-up with conversion `[x,-z,y]`. Paired eye-exclusion centers place the primary plane at X=0.053584719 m; the nose center is X=0.052676504 m and Head pivot X=0.045410779 m. Whole-model bbox X=0 is not used as the facial midline. These are authored feature annotations, not recovered anatomical measurements. The full eye-pair plane accounts for their slight asymmetric Y/Z positions without searching for a better match.

At the Head plane every stopped-boundary sample is within 7.289 mm, but 13 of 65 edges retain excessive normal disagreement. The original 78-edge disk similarly has only 39 fully supported edges despite all samples being within 8.249 mm. `remaining-boundary-disagreement.json` records exact source/current face and edge IDs. Geometric normals fail as well as custom normals, so changing shading normals alone is not established as a solution.

The stopped 215-face scar domain still has its recorded multiple boundary cycles and invalid vertex links. Reflection does not fix topology, authorize moving protected faces, or recover the hidden contact surface. The forbidden diagonal, eye/nose/face texture, all 244 protected white faces, and every source byte remain unchanged.

## Preserved first attempt and bounded execution

The first screen used the authored head-groom ellipsoid as a donor cutoff. All 429 primary nearest points landed at that cutoff; its large distances were therefore inconclusive. That run is sealed unchanged in `../panda-cheek-symmetry-screen-v1`. The parent authorized this single changed-hypothesis pass using the full opposite original-body half, not a larger replacement domain.

- First run: 2.601 s, 166,543,360 B sampled peak RSS.
- Corrected run: 7.026 s, 209,649,664 B sampled peak RSS; PID 69832 exited.
- Aggregate: 9.627 s within 60 s; no native, model, provider, or paid calls.
- Source hashes, scripts, exact commands, logs, job monitor, and all per-sample correspondences are retained.

Next safe action: parent reviews the 13 exact failing edge witnesses before choosing a different boundary/shape authoring strategy. Do not repeat scar-ring expansion, optimize a plane against the damaged boundary, weaken 45°, or treat the closer Head-plane positions as a completed repair.
