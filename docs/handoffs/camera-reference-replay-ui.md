# Camera fitted-reference replay UI

## Released scope

The existing offline replay now supports the camera fitted-reference controller without changing the scene, task, body, camera, metrics, or gates. This is saved-state presentation, not a new simulator, source admission, action replay, or physical validation.

Owned changes:

- src/weft/assets/shared-world-replay.html — SHA256 64acc86c8b4859cd5157a2506f1783c0a4e8a234e9f45091d898e6b6f8c46d13
- tests/camera_reference_ui_smoke.mjs — SHA256 44d06c2b5c60cd757c6766f1be5d8c111811d77f601537e4d706d22ace8ea813
- This handoff. No lab.js, avatar, scorer, backend, or Git changes in this lane.

## Saved-view contract

The recording backend supplies reference_controller with exactly eight string fields: profile, sha256, kind, scene_sha256, task_sha256, gates_sha256, physical_validation, and scope.

Required values:

- profile: camera-source-fit-controller-view-v1
- kind: source_fit_authored_controller
- physical_validation: unverified
- scope: Fitted joint controls; original camera, body, task and scoring remain unchanged

The worker sends only this compact view, never the archived recipe or reference arrays. Missing fields, extra fields, arrays, empty strings, and strings longer than 512 characters become a tiny invalid marker. Main-thread validation joins the scene, task, and gate pins to the existing camera identity and task. Policy and reference summaries cannot coexist.

A reference_proposal step must carry exactly one reference-prefixed field: reference_controller_sha256, equal to the summary pin. Reference metadata on start, reset, manual, coast, authored, learned, or unknown modes is rejected. Existing camera state, clock, metric, and gate checks remain in force. The UI calls the controller programmed, not learned, and retains the distinction between simulated movement, source video, and physical evidence.

These are metadata consistency checks. The full journal and backend exact-action/full-state replay remain authoritative; a displayed hash is not independent authentication.

## Evidence

Two pre-fix regressions reproduced the real integration defect: the valid reference step paused playback, and the worker omitted its compact summary. Both failures remain in .local/camera-reference-ui-red-01.log. The second assertion reported JSON parsing of undefined because the summary was absent; it was not a product JSON decoder failure. Pre-fix source/test bytes were not archived.

Commands:

    node tests/camera_reference_ui_smoke.mjs referenceReplayPreservesCameraIdentityAndGates workerForwardsOnlyTheCompactReferenceView > .local/camera-reference-ui-red-01.log 2>&1
    node tests/camera_reference_ui_smoke.mjs > .local/camera-reference-ui-green-01.log 2>&1
    node tests/camera_shared_ui_smoke.mjs offlineReplayRequiresActualProfilesAndExplicitBundleIdentity > .local/camera-reference-ui-legacy-camera-01.log 2>&1
    node tests/camera_shared_policy_ui_smoke.mjs offlinePolicyKeepsLearnedMetadataAndAuthoredRoots offlineResidualRoundoffMatchesRecorderTolerance > .local/camera-reference-ui-legacy-policy-01.log 2>&1

Six new scripted check groups and three affected existing replay groups passed. Cases cover roots off/on, exact summaries, missing or forged pins, mixed controllers, invalid event modes, unchanged camera metric checks, and bounded worker projection. A 1 MiB extra recipe field is rejected in the worker rather than forwarded. One thousand unchanged mocked view rotations took 29.233291 ms and caused zero camera diagnostic text writes.

The fixtures use the saved payload shape, including actual engine profile fields, but are authored analytical data. This is not a browser responsiveness measurement or a native recording acceptance. No source decoding, fitting, model/physics work, server, network, or API call ran. Unchanged broad suites were not repeated.

## Backend review and next step

Read-only review found an unbounded scene/recording-metadata reread in the common loader. The parent fixed it with bounded no-follow reads before adapter/native validation; the corrected section was reread and the finding closed. Runtime/recipe authentication, unchanged camera identity, event allowlists, exact controller-action replay, root-action preservation, cleanup, and dependency closure showed no other concrete blocker in this scoped review.

Root owns backend tests, publication, and any native acceptance. Keep these asset bytes frozen during a combined native run. Use its actual exported replay bundle for a later saved-payload UI check; do not relabel these analytical fixtures as native evidence. No live handles or jobs remain in this lane.
