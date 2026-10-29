# Appearance-tracking development sources

`elastic-pendulum.webm` is the unchanged original [ElasticPendulum video](https://commons.wikimedia.org/wiki/File:ElasticPendulum.webm), by Tomasz59, licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Attribution, original URL, retrieval time, size, and SHA-256 are pinned in `elastic-pendulum.source.json`. The video is 16.8 seconds, 504 frames, 1080×1920, approximately 4.6 MB. No account, model, or paid service was used to obtain it.

The `annotations/pendulum.*` files and rendered pendulum previews are Weft adaptations, distributed under the same CC BY-SA 4.0 license. The video pixels are unchanged. The separate electrostatic annotations refer to the original `../balls/bouncing-ball.webm`; that source remains credited to ScienceCOLA under CC BY 3.0 in its existing sidecar.

Initialization files contain authored object regions and identity labels. Manual files contain a small set of visual center/visibility labels written before running the appearance tracker, with annotation uncertainty. These were assistant visual inspections, not independent human expert measurements. They are development checks, never a final holdout. Initialization frames are excluded from the scored labels.

The electrostatic scene contains a large charging sphere, a hand-held metal sphere on a rod, and a small suspended white pellet. The authored regions select the two larger spheres; **the small pellet is not tracked**. A source title is not sufficient to infer the physical object or driving force.
