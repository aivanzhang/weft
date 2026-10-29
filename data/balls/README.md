# Additional real-ball development input

`bouncing-ball.webm` is the unchanged **Bouncing Ball** video by **ScienceCOLA**, downloaded from [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Bouncing_Ball.webm). The page attributes it to [ScienceCOLA's original video](https://www.youtube.com/watch?v=A5z0i2SdqDk) and records **[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)**. The `.source.json` sidecar pins the original media URL, retrieval time, SHA-256 and byte size.

The title is misleading for a gravity-bounce benchmark: this is a suspended metal-covered ball interacting with a charged sphere, after explanatory text frames. It is included as a **real unsupported scene**, not an ordinary bouncing-ball success. Its near-gray reflective objects, string constraint and electrostatic forcing exceed the saturated-color planar tracker. The preview contact sheet is a Weft-generated adaptation; retain the attribution/license when sharing it.

The original WebM reports 661 frames at 25 fps through OpenCV metadata but decoded 656 frames in the current environment. The untouched source fails the strict planar decode gate; `artifacts/ball-real-electrostatic-01/failure.json` preserves that result. That is a container/decode rejection, **not** proof that the system correctly understood the electrostatic mechanism. No final holdout or physical transfer claim is made.

The existing, independently sourced ordinary rolling-ball clip remains documented in [the parent README](../README.md).
