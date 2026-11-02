# Independent assembly audit — CPU passed

The frozen checker passed once against the released actual public-CLI assembly. It independently verified every component transform, complete primitive/index/UV coverage, original embedded image bytes, PBR/sampler semantics and the unapplied canonical rig-space prediction. It did not import the assembler or accept its status as an oracle.

- Model: `outputs/robot-dragon-source-assembly-v1/model.glb`, SHA `a2bdca0698d3ef777323889f8476ca80219bcde03c4e56a761c270fc97e44127`.
- 81,091 vertices; 87,821 triangles; 50,393,860 bytes; six images / 75,497,472 pixels.
- Maximum P error: 5.95925e-8 m (gate1e-6); maximum rotated N component error: 2.98019e-8 (gate2e-7). Robot normals, all UV0/index rows and embedded image bytes are exact.
- Canonical receipt offset/translation/scale errors: zero; the transform is explicitly **not applied**. No rig or clips were created.
- One CPU run, PID81725 exited0; bounded runner0.262s, checker0.1509s. No periodic RSS sample because it completed early; separately reported macOS process high-water RSS248,479,744B.

`result.json` contains independent measurements, `review.json` their limits, and `job.json` process custody. Source/checker/config copies and immutable large-model references are retained.

This is reused-source composition evidence, not fresh generation, the original shoulder-dragon identity, animation or visual acceptance. No native/provider/AI-role/browser call ran. The parent owns the separate native preview.
