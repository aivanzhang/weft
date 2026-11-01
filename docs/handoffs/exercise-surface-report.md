# Exercise surface collection report

## State and ownership

Implemented and tested locally on October 31, 2026 (PDT). The exact observation
time, commands and hashes are in `.local/exercise-surface-report-checks-01.json`.
No deployment, browser interaction, native/model/media work, paid request or Git
mutation occurred in this lane. No live handles remain.

This lane owns only `src/weft/exercise_surface_report.py`,
`tests/test_exercise_surface_report.py` and this note. Product and test files are
frozen for the collection-workflow owner's one actual filesystem/report
integration check. Root owns the common CLI, common documentation and release.
Do not repeat completed collector, camera fitter or native studies to test this
metadata-only report.

## Interface and input boundary

```python
from weft.exercise_surface_report import build_exercise_surface_report

outputs = build_exercise_surface_report(saved_files, max_seconds=5)
# The filesystem workflow exclusively writes outputs under inspection/ and seals them.
```

`saved_files` is an exact-byte snapshot containing:

- `inventory.json`, `selection.json`, `workflow-state.json`;
- sorted `surface-observation-00.json` through at most `07`, plus optional
  `surface-camera.json` and `original-human.json`;
- after selection, the existing collector's request, report, complete JSONL,
  manifest and any qualified surface documents/configuration.

The backend filters out runtime, plan and parent-manifest packaging members before
calling the reporter. The reporter neither executes saved runtime code nor
requalifies observation documents or geometry. It verifies exact selected hashes,
profiles, source bindings, selection/request identity, complete frame/channel
counts, source-clock declarations and annotation-template joins. Input schema
qualification remains distinct from fitter or simulator admission.

The only filesystem read is the existing generated `src/weft/assets/brand.css`,
once and at most 16 KiB. It is returned byte-for-byte as `brand.css`; the backend
must include `assets/brand.css` in its archived/rechecked runtime closure. No other
assets, libraries, fonts or network resources are loaded. Layout CSS and scripts
are literal in the reporter; no palette redesign or avatar changes were made.

## User journey

1. **Needs selection:** show the exact supplied source and construction-filtered
   entity claims. Entity, channel and body identity start unselected. A source
   acknowledgement plus explicit channel fields is required to download the
   existing `exercise-surface-selection-v1` schema. The optional body entity is
   chosen separately or remains null. Surface names imply neither ground nor
   stationary equipment. Entity choices are searched and paginated 50 at a time.
2. **Draft or unavailable inputs:** list exact missing-field paths and remaining
   explicit inputs. Link `collection-request.json` and show the real
   `surface-inputs --resume ... --request ... --out ...` route with placeholders.
   Frame pages retain every original slot, show separate source/body clocks,
   saved envelope/body/image-admission statuses, supplied ranges and evidence
   references. Downloads preserve exact original-frame annotation identity and
   null pixels/uncertainty; they never modify the saved request.
3. **Schema-qualified input files:** link the actual `surface-config.json`, while
   stating that body/camera/mapping admission can still fail. Show the existing
   `body-reference-propose` command with a separately qualified body-run
   placeholder. Camera-linked bodies use `--objective camera-surface-intervals`
   and require an explicit `--camera-image-scale AUTHORED_VALUE`; no numeric
   default or body-run path is guessed. No-body collections are explicitly
   described as source envelopes without a body-to-surface join.

An edited request can explicitly change its body entity under the existing
collector contract. The report displays that identity separately from the
original selection rather than relabelling it or inventing a match.

Height/extents in world metres, pixel uncertainty, endpoint correspondence,
authored fitting weight and contact are kept distinct. Unknowns remain unknown,
not zero. A conditional range is not contact, load, grasp, installed geometry,
physical accuracy or successful reconstruction.

## Bounds and rendering

- At most 32 selected input files / 32 MiB; complete collector JSONL at most
  8 MiB, 1,800 frames and 12 channels.
- At most 32 returned files / 16 MiB; every HTML file at most 2 MiB.
- `index.html`, exact `brand.css`, compact `report.json`, and fixed
  `frames-001.html` onward. Frame pages contain at most 100 original frames and
  1.5 MiB of projected row data. Bounds reject explicitly; slots are never dropped.
- Only the selected frame/channel renders. Repeated unchanged selections return
  immediately; updates replace text, not screen roots or all-frame tables.
- Embedded JSON escapes `<`; user data is written with text nodes. All evidence
  links point to fixed existing root files. No fetch, timers, server or browser
  automation is used. Download object URLs are limited to one live URL and
  revoked on replacement/page exit.

The five-second deadline and byte caps are cooperative trusted-process bounds,
not hard CPU/RSS isolation. The report is not an independent source authenticator
or a replacement for the existing collector validators.

## Verification

**17 distinct cases passed.** The first run had 14 passes and one test assertion
expecting a later missing-member message. The product correctly rejected the
missing camera earlier at the original-input hash check. Only that expected
message was corrected, then that case passed. Two additional template-download
and output-cap cases passed. Unchanged passing cases were not rerun.

Checks use real pure collector draft/qualified/unavailable/no-body snapshots,
saved-byte corruption cases, explicit selection and annotation downloads, local
link resolution, unknown preservation, escaping and bounded output rejection.
Node runs a mock DOM only; no browser QA or browser responsiveness claim follows.

The large synthetic saved-metadata case retained **1,800 × 12 = 21,600 slots**:

- 7,692,194 input bytes; 21 report files / 5,732,500 bytes;
- report preparation: **0.156545 s**;
- 1,000 scripted frame/channel changes: **0.007569 s**, **zero new DOM nodes**;
- 100 frame options and 12 channel options on the loaded page.

These are bounded projection/mock-DOM observations, not real-source qualification,
physical recovery, sustained browser performance or a hard memory guarantee.

| Frozen file or receipt | SHA256 |
| --- | --- |
| `src/weft/exercise_surface_report.py` | `e6ed5dbf76fe42687c077f92b5324bffafb1b6d5cf44afb484b8ba290c81f5d5` |
| `tests/test_exercise_surface_report.py` | `346c24f98c21cadf35dd7ef7c01ad44c4aa341de898f5d1dca55317e9251f562` |
| `.local/exercise-surface-report-checks-01.json` | `983f0a7ba846b926e822fa1456834160e3652c13ea90a4e3732a1649761c2bc6` |

Exact commands and full logs/JUnit pins are retained in the receipt. The three
JUnit files are `.local/exercise-surface-report-tests-01.xml`, `-02.xml` and
`-03.xml`; the initial failure is preserved.

## Next safe action

- [ ] Collection-workflow owner completes its one actual filesystem/report
  integration check with the frozen reporter and archived `assets/brand.css`.
- [ ] Root independently reviews the UI/contracts and connects the common CLI.
  Unknown/incomplete collections are useful completed outputs, not failed
  reconstructions.
- [ ] Root records integration evidence and publishes only under the applicable
  repository rules. This lane has not inspected or changed Git state.

Completion means the existing collection CLI produces a sealed offline report,
explicitly chosen selection, exact-frame templates and honest unavailable/qualified
states without a second fitter product, hidden default assumptions or dropped
evidence. It does not complete unfamiliar-video reconstruction or contact recovery.
