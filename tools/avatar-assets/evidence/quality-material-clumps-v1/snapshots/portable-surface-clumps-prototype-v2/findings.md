# Surface-clump prototype v2: fixed-lift rejection

The exact float32 V-flip admission passed without an epsilon change. Actual final GLB positions, normals, UVs and unpruned four-weight skin supply the retained 900 contour fins; all 3,600 oriented retained triangles and duplicate corner attributes match exactly.

The unchanged 1.2 mm lift failed the predeclared 15-degree geometric-turn gate on 163 of 5,400 source triangles, affecting 163 of 2,700 surface cards. Maximum turn is 83.581248 degrees (fin 296, native source face 24069). Failing triangle altitude ranges 0.270118–4.495377 mm, median 3.156426 mm; the selected population median is 14.996191 mm. The offset is too large for these thin source triangles, not an opposing source-normal or root-correspondence failure. All selected source corner normals were outward, with minimum geometric dot 0.817147.

No intersection screen, source texture decode, atlas paint, component-array write, model integration or render ran. This is an unsupported candidate, not an appearance result. All inputs remain unchanged. The candidate child exited in 0.262 s; the separate read-only diagnosis exited in 0.264 s. Both exited before an RSS sample, so sampled peak memory is null/unknown.

## Exact next hypothesis, not an executed repair

With both root endpoints fixed, let h be source triangle altitude; a the source corner normal's component along the transverse altitude direction; b its component along the original geometric normal. The same 15-degree gate implies the continuous ceiling `L = h*tan(15°)/(b-a*tan(15°))`. Use the smaller ceiling of the card's two source triangles and retain the 1.2 mm outer cap. Only 163 cards need a smaller lift; 45 ceilings are below 0.6 mm, 7 below 0.2 mm and 1 below 0.1 mm. Minimum ceiling is 0.070694 mm. All 2,700 selected cards and 900 original fins can therefore remain selected under a distinct topology-adaptive hypothesis.

A separately approved candidate could use a small inward margin below each derived ceiling, then run the unchanged actual-float32 15-degree and intersection gates. This would not relax the gate or prune source cards. The continuous formula is not proof of actual intersection safety, visible density, source palette fidelity, animation or Muse quality. No such candidate has been run here.

The exact rejected source IDs, raw GLB primitive/face IDs, source/offset coordinates, normals and analytical ceilings are in normal-turn-diagnosis.json. The earlier controller mismatch remains preserved in portable-surface-clumps-prototype-v1. Both analytic test manifests had an invalid empty self-inventory entry; surface-clumps-test-seal-audit-v1 preserves that erratum and independently hashes their actual saved files without rerunning tests.
