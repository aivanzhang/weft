# Avatar preview quality correction

Recorded November 3, 2026 (PST), after the user reported the linked avatar video
was too short and glitchy. The previous seven-frame, three-avatar acceptance clip
is an integration test, **not a useful motion demo or visual quality pass**.

## Verified

- Original clip: six actions, seven frames at 30 fps (0.233 seconds), with avatar
  switches anchored at events 0, 2 and 4. Rapid switching explains its flashing.
- New continuous robot: 234 controls / 235 frames, 7.8 seconds of simulation
  (7.833 seconds encoded), one avatar, no switches or artificial slow-down.
  Full original interval completed; exact physical replay and appearance export
  verified. **Task success is false:** mean and peak body-position tracking fail;
  angle and floor gates pass. No runtime failure, new inference or API charge.
- Normal recording: primary demo child
  `1a5b41db5aa34654a84c83a9dbf5e316/lab/sessions/session-fafeb2ffe374` under
  `.local/portal-unfamiliar-source-acceptance-01/portal`.
  Evidence `.local/portal-avatar-continuous-01/{result,media,seal}.json`; actual
  frame contact sheet inspected. Robot stays recognizable/connected, but the
  fixed camera crops portions as the body drifts.
- UI now reports the current exported video duration, warns below one second,
  and explains that paused wall-clock time does not add simulated video time.
  Existing `node tests/lab_ui_smoke.mjs` passes; receipt
  `.local/portal-avatar-continuous-01/lab-ui-check.txt`.

## Unresolved Studio deformation

The earlier “rough surface from simplification” explanation was insufficient.
Saved-only diagnosis finds severe edge stretching in the original, undecimated
opaque Studio body under the same actual RL pose, not just in reduced fur/cards.
Neutral/rest transforms are clean and fur cards stay rigid. This is a real
skin/retarget-compatibility problem; do not label the Studio avatar visually
qualified merely because importer bounds, exact physics replay and HTML export
pass. The original records and source remain unchanged.

The pipeline plumbing works; generalized visual motion compatibility is not
finished. Independent saved-pose diagnosis identifies bad source skin weights as the
primary cause: an original 0.00322-world-unit foot edge stretches to 0.09561
because neighboring vertices mix opposite legs/feet; a derivative chest/hand
edge stretches 0.02265→0.44657. Applying one common torso rotation keeps all
edges rigid. This is not a root/bind numerical explosion or simply alpha/LOD.
Studio remains visually unqualified. The next action is a separate part-aware,
continuous skin-weight derivative plus canonical joint-pose deformation checks,
without changing RL physics or patching individual flight/hopping clips. Preserve unrelated authoring work and failed artifacts.
