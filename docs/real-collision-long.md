# Longer real collision acquisition — bounded result

**No qualifying longer fixed-camera collision source was established in this acquisition pass.** One real, permissively licensed candidate was downloaded and source-only labels were frozen; its full physical interval visibly contains camera motion. No tracker, reconstruction scorer or full pipeline was run on it.

The statement above describes the frozen acquisition pass. A later [full pipeline and unchanged-label evaluation](appearance-guards.md) detects camera movement, accepts 8/19 visible centers and rejects reconstruction. Source-only acquisition artifacts remain unchanged.

## Retained source

- **Elastischer Stoß**, by **Julian11.08**, from [the original Wikimedia Commons file page](https://commons.wikimedia.org/wiki/File:Elastischer_Sto%C3%9F.webm).
- License: **[CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/)**, as declared on that file page.
- Local original: `data/balls/elastic-table-experiment.webm`, unchanged.
- SHA-256: `b4f9f4a919f665cc94cd9a40c4bf085961304ca8f51eeda5a9710bd3611abe2e`.
- Size: **4,993,777 bytes**; fully decoded **1,299 frames**, **1,920 × 1,080**, **24 fps**, approximately **54.13 seconds**.
- Contents: real unequal-sized balls colliding on a rail, hand release, title/formula slides, and handheld framing changes. It is not a free 2D air-table scene.

The original includes apparent source-reported mass/velocity calculations, but this acquisition did not verify or use those numbers to calibrate a world.

## Frozen before tracking

The source-only inspection selected **frames 300–420 inclusive** as a physical release/contact/exit interval. This retains the hand-held initialization, collision, both balls leaving view and camera drift. It was not shortened to make a tracker or reconstruction pass.

Files beside the source:

- `.source.json`: source rights, URLs, hash, format and retrieval time.
- `.manual.json`: **22 object/frame visibility labels**, including **19 approximate centers**, ±16 original pixels; closest-approach/contact candidate **frames 337–340**. This interval is inferred visually, not instrumented contact ground truth.
- `.initialization.json`: two manually placed projected boxes at frame 300 and the frozen interval.
- `.scene.json`: the same initialization plus explicitly authored placeholder controls/dynamics. **This is not an approved accurate physical model or a qualifying benchmark success.**

The labels were created by assistant visual inspection of original decoded frames/contact sheets, without predicted coordinates. They are development annotations, not independent expert labels or final holdouts. No missing/out-of-view centers were invented.

Inspection artifacts, acquisition decisions and the label-freezing script are retained in `artifacts/real-collision-long-01`. The contact sheets cover the complete source, the physical interval and closest approach. No existing air-table or pool source, labels, settings or runs were modified.

## Other candidates screened without downloads

- [OSP 2-D Collision of Pucks](https://www.compadre.org/OSP/items/detail.cfm?ID=10579): the teaching resource has a noncommercial license and separately credits the video to Flashmedia; permissive video reuse was not established.
- [OSP Explosive Collision Lab](https://www.compadre.org/Portal/items/detail.cfm?ID=16672): noncommercial/no-derivatives video license; not acquired for this lane.
- [LivePhoto puck/bar collisions](https://livephoto.sciencetutorials.net/LPVideos/puckbar/): custom teaching-use terms, not broad project reuse permission.
- [Snooker break](https://commons.wikimedia.org/wiki/File:Snooker_break.ogv): explicitly a 3D rendering, not a physical video.
- [Carom billiards](https://commons.wikimedia.org/wiki/File:Carom_billiards.webm) and [Dreifachstoß](https://commons.wikimedia.org/wiki/File:Dreifachsto%C3%9F_(Pilotversuch).webm): original files exceed this lane's 20 MB per-download bound; not downloaded.

Search/acquisition stayed below the requested 10-minute search envelope. **One download, 4.99 MB total**, against limits of three downloads, 20 MB each and 40 MB aggregate. No paid services or account actions.

## Remaining requirement

A longer real fixed-camera ball/disk collision with a sufficiently long complete, unobscured pre-contact/contact/post-contact interval is still needed. The retained rail experiment can later test camera compensation and explicit rail constraints, but must not be presented as filling that fixed-camera requirement.
