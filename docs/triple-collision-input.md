# Triple-collision lab input mapping

`data/balls/triple-collision-240p.scene.json` adapts the frozen manual initialization into the generic appearance lab schema. It does not modify the media, source-only labels, source sidecars or sealed acquisition manifest.

- **Source:** complete official Wikimedia 240p transcode, Rhetos, CC BY-SA 4.0; SHA-256 `f2d9f8125a197ec135dcc4d95e5b1b615c875f097c12ee272d83f8b97baa0887`.
- **Full selected interval:** source frames 2580–2760 inclusive, 181 frames, spanning 43.0–46.0 media seconds at the recorded 60 fps. The same interval populates `start_frame`, `end_frame` and `decode_interval`; it has not been narrowed to improve tracking.
- **Initialization:** frame 2620 (43.6667 media seconds), the first inspected frame with all three centers visible. Frames 2580–2619 and their incoming-ball occlusion remain part of the selected interval; initialization is not permission to discard them.
- **Boxes:** unchanged 8×8 source-pixel boxes from `.initialization.json`, centered exactly on the frozen frame-2620 manual centers. Upper passive `[187,104,8,8]`, lower passive `[189,111,8,8]`, incoming active `[270,113,8,8]`. The source resolution is 426×240; no resizing or enhancement was performed. Manual center tolerance remains ±5 pixels, large relative to each ball.
- **Simulation:** only `incoming-active` is controlled. Target `[0.5,0.7]`, gravity `[0,0]`, damping `1.0` and restitution `0.8` are authored placeholders. Physical forces, scale and restitution remain unknown. Circle shapes and roles are explicit assumptions, not calibrated properties.

The first selected shot is presented in real time by the source. A later repeat is explicitly tenfold slowed, enlarged and annotated; it is not used as another physical trial or assigned real-time velocities. Independent physical timing remains uncalibrated. Camera drift, hand occlusion, contact and exits stay in the selected evidence.

## Verification and remaining integration

`artifacts/triple-collision-input-01/verification.json` records exact scene/source hashes, equality to frozen manual boxes, metadata and verification of all 15 acquisition-manifest files. The three frozen sidecars still match their sealed copies.

The metadata-only check reports 60 fps, 426×240 and 12,759 estimated frames. Earlier complete decoding recorded 12,758 frames. A fresh bounded container-block audit stopped with **“Container elements exceed the audit bound”** at the existing 20,000-element ceiling; this failure was retained, not relaxed or mistaken for proof of corruption. No frames were decoded again for this adaptation.

The full source exceeds the existing 60-second/1,800-frame ingestion limit. Integration requires the bounded source-hash-pinned interval decoder and explicit initialization-frame handling owned by the main implementation. This preparation does not demonstrate a lab pass, valid fixed camera, accurate physics or successful object tracking. No tracker, inference or full worker pipeline was run.
