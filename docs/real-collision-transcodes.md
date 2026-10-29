# Real collision video: bounded official transcode follow-up

Source review completed October 28, 2026, 1:58 p.m. PDT, before any tracker or physics fit. The earlier `artifacts/real-collision-long-01` acquisition remains unchanged.

**Later implementation:** [bounded interval decoding now processes the selected real-time event](source-intervals.md). The original review below remains preserved; the first actual tracking evaluation fails coverage and includes a large position/identity error. It is not a qualified physical reconstruction.

## Acquired media

| Complete official derivative | Rights / attribution | Bytes | Decoded frames | Duration from metadata |
| --- | --- | ---: | ---: | ---: |
| [Carom billiards, 480p](https://commons.wikimedia.org/wiki/File:Carom_billiards.webm) | billclub, CC BY 3.0 | 6,821,405 | 2,399 | 80.08 s |
| [Dreifachstoß, 240p](https://commons.wikimedia.org/wiki/File:Dreifachsto%C3%9F_(Pilotversuch).webm) | Rhetos, CC BY-SA 4.0 | 3,597,273 | 12,758 | 212.65 s |

These are complete Wikimedia-hosted transcodes, not byte-identical original uploads. Exact URLs, hashes, attribution and retrieval time are pinned in `data/balls/carom-billiards-480p.source.json` and `data/balls/triple-collision-240p.source.json`. Official source-page HTML is preserved in `artifacts/real-collision-transcodes-01`. Container frame estimates exceed actual decoded counts by one for each file; no strict decoder check was bypassed and completeness is not independently certified.

The two follow-up downloads total 10,418,678 bytes. Including the earlier rail experiment, the three media downloads total 15,412,455 bytes, below the 25 MB cumulative limit. No original exceeding the download limit was fetched. No paid services or accounts were used.

## Suitability

Neither source establishes a clean fixed-camera benchmark.

- **Carom:** real balls and repeated interactions, but oblique view, changing framing, edited shots and player occlusion. The full source is preserved, not cropped into a passing case.
- **Dreifachstoß:** a useful real overhead three-ball event, but small roughly six-pixel balls, hand occlusion, visible camera drift and later frame exits. The source also repeats this event at tenfold slow motion with annotation overlays. Those repeated frames are not independent evidence and their media timing is not physical timing.

For Dreifachstoß, manual source-only labels freeze the entire first real-time shot, frames **2580–2760 inclusive** (43–46 media seconds). The interval includes release, collision, camera movement and exits; it was selected before any tracker. It has 24 visibility records, 22 approximate centers (±5 derivative pixels), and an inferred contact interval at frames 2640–2650. Initial object boxes are pinned at frame 2620, the first inspected frame with all three centers visible; the preceding occlusion remains in the full labeled interval. Post-contact identities are inferred continuity, not ground truth. No masks, contact-force facts, physical calibration or measured restitution are supplied.

Both full derivatives exceed the current ingestion limit of **60 seconds / 1,800 frames**. Using the independently selected event requires a source-hash-pinned, bounded interval decoder. No crop, silent trim, repeated-frame workaround, low-resolution pipeline run or tracker run was performed here. The original derivatives and unsuccessful qualification evidence remain available for further work.

## Sealed artifacts

`artifacts/real-collision-transcodes-01/manifest.json` pins 15 review files, including source-page snapshots, full-duration contact sheets, collision-detail sheets, manual labels, initialization settings and acquisition outcome. Its SHA-256 is `7a18fee96dd790c316f1d14d5aa53b3926d5049efdaed535af4e6154ad0e2c9c`.

Media SHA-256:

- Carom: `8dc3cc9b095ee04ed300d5ca44ada83a4b20d2aab73967a2a39eba0934a6ea1a`
- Dreifachstoß: `f2d9f8125a197ec135dcc4d95e5b1b615c875f097c12ee272d83f8b97baa0887`
