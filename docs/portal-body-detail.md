# More detailed body extraction in the portal

When an exercise/equipment project's **Check video locally** step fails during body extraction, Weft can offer **Try detailed body extraction · no API charge**. This creates a separate project; the original video, failure and raw body estimates stay unchanged.

1. Open the failed project and inspect its body-extraction explanation.
2. Choose the detailed attempt if it is available. It uses the local Full pose model, every frame, and a fixed region derived from all retained credible body estimates—not a manually selected person.
3. Follow the new project's progress. Its artifacts show the original source binding and the exact region, model and sampling recipe.
4. If local checks pass, choose separately whether to send sampled images to OpenAI and the original video to Meta. The local attempt does not authorize or charge either API.

The same identity, pose-quality, body, floor and timing checks still apply. A more detailed attempt can fail. Neither a completed body nor its illustrative muscles establishes physical accuracy. Passing local checks prepares evidence for the existing equipment workflow; it does not guarantee a playable reconstructed world.

## Supported input and limits

This action currently requires an original unresolved exercise upload with a sealed failed **body** preflight, retained raw pose alternatives, an MP4 within 37 MB and the existing 12–500-frame envelope. Failures in later stages, saved-body imports, recovery chains and projects with existing paid preparation or equipment candidates are not eligible. Exact source/model/region checks run in the worker before pose acquisition.

There is one attempt per parent and recipe. Repeated clicks reopen the same child; failed, cancelled or interrupted children are not silently retried. Interrupted partial output stays available for diagnosis. The existing shared processing queue and 2 GB portal storage limit apply; Weft does not delete older projects to make room.

Start a separate local project directory for new videos:

```sh
.venv-lab/bin/python -m weft portal --root .local/portal-new-videos
```

The [existing playable examples](moving-placement.md) use a different root that is close to its storage limit. New attempts can be refused when a complete child cannot be reserved; this is not a pose or geometry failure. No hosted deployment or persistent server was started for this feature.

## Verification boundary

Focused tests cover source/model tampering, all-hypothesis region derivation, old preflight compatibility, one-attempt queueing, cancellation/restart behavior, separate paid consent, safe downloads and UI actions. Complete worker orchestration tests use explicit inference fixtures. The real-source verification stages preserved evidence without repeating the already-run 458-frame Full-model inference. It is not a newly qualified unfamiliar-video reconstruction. See the [handoff](handoffs/portal-body-detail.md) for exact results and remaining work.
