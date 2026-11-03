# Restore body sandbox actions for AI-routed exercise uploads

Recorded November 3, 2026 (PST). The user uploaded the provided squat MP4 with
**Let AI identify the scene**, **I'm not sure**, and **Interactive simulation**.
The local body check succeeded, but no body-sandbox action appeared. A subsequent
consented equipment analysis refused its empty equipment-region list. That refusal
is preserved and is not evidence of an invalid video or failed body extraction.

## Fix

`portal_saved_body._metadata` checked the original upload selector (`mode=exercise`)
instead of accepting `mode=auto` after the saved route resolved to exercise. It now
accepts both selectors only for original video jobs with `status.route=exercise`.
All source seals, explicit body/ground approximation scope, original unknown
inventory, active-job fences and no-retry rules remain unchanged. The preparation
heading now says **Prepare this video**, with the local body route explained before
optional equipment consent.

## Verification

- 58 saved-body, portal integration and Muse-binding checks passed, including AI
  routing after both successful local checks and failed equipment preparation.
- Negative cases reject unresolved/object routes, object-mode uploads and imported
  labs. Queue tests retain original metadata and authorize zero API calls.
- Existing portal DOM checks pass: `.local/portal-auto-body-fix-01/ui.txt`.
- Actual user projects `1f96afd8b62d4766b782386e0e75cc64` and
  `545374838071429ca80c3e4139924cf4` now advertise the body action; complete saved
  preflights revalidated unchanged, including original unknown inventory.
  `.local/portal-auto-body-fix-01/verification.json` records exact seals.
- No API request, repeated pose inference, physics step or live-server restart.

## Apply in the current portal

The existing server has the old Python module loaded. The user manages servers:
stop that terminal's portal and rerun its command, then refresh the page. Select
an existing checked project and **Turn this body into a sandbox**. Confirm the
explicit body-and-ground approximation; do not upload again, relabel equipment
absent, or repeat paid equipment preparation. The earlier failed preparation stays
inspectable. Restart recovery must not rewrite completed source artifacts.

Relevant files only are published on main under standing authorization. Preserve
separate dirty avatar authoring work and the untested research draft; no stash
cleanup or data deletion is needed.
