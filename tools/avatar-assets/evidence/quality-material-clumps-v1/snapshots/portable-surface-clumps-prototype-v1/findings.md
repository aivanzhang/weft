# Surface-clump prototype v1: pre-geometry rejection

The one bounded CPU invocation stopped before calling the geometry helper. Exact helper-to-final UV equality was too strict for the known native image-V conversion: all 4,864 differing rows match the exact float32 V-flip round trip, with maximum component difference 2.9802322387695312e-8. Positions are exact. The raw and final GLB POSITION/UV bytes match. No tolerance, selection, geometry or normal gate was changed.

The separate read-only contour check proves the retained 900 fins have the same 3,600 oriented triangles and exact duplicated position, normal, UV and four-influence skin attributes. It does not prove the proposed surface geometry is safe; no offset/intersection or texture-paint stage ran.

Child PID 73609 exited after 0.262 seconds. No RSS sample completed, so peak memory is unknown/null rather than zero. No model, atlas, rendered image, provider or AI-role call was produced. All original inputs remain hash-identical.

Next safe action: a newly versioned controller may admit only the exact documented float32 V-flip conversion and then use final GLB UV bytes. Require the same retained-contour oriented correspondence. The fixed 2,700/900 selection, 1.2 mm lift, 15-degree normal-turn gate and intersection checks remain unchanged; parent release is required before the next bounded candidate invocation.
