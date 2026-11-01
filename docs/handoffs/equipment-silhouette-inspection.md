# Saved equipment-evidence inspection

## Current work

Root owns `equipment_silhouette_inspection.py`, the
`equipment-evidence-report` CLI route and their two test files.
`/root/support_ui` owns the separate report renderer, its tests and
[renderer handoff](equipment-silhouette-report.md).
`/root/exercise_reconstruction` owns the actual saved-source composition;
`/root/support_learning` reviews its bounded runner and this wrapper.
No source study is repeated to generate an inspection page.

The public command and actual saved-source acceptance are complete. Open the
[existing report](../../artifacts/pose-equipment-inspection-01/index.html) without
rerunning it. To create a separate report from another completed study:

```sh
.venv-lab/bin/python -m weft equipment-evidence-report \
  --run path/to/completed-equipment-study \
  --out artifacts/my-equipment-inspection-01
```

The destination must be new and outside the original study. The command starts
no server and opens no browser. Optional `--manifest-sha256` pins the exact
study seal; `--max-seconds` defaults to a ten-second cooperative deadline.

## Evidence boundary

Read only the completed root `manifest.json` and its two exact pinned members:
`silhouette/metadata.json` and `silhouette/frames.jsonl`. Preserve their bytes in
the report as `equipment-metadata.json` and `equipment-frames.jsonl`, along with
the original seal. This verifies a selected saved snapshot, **not** all source
media, masks, clocks, reconstructed geometry or physical contact. Other source
members are neither read nor silently claimed verified.

The renderer validates the complete original frame/pair records. It must keep
missing and excluded slots, source and derivative clocks, authored pixel ranges
and unknown contact/load/grasp. Conditional image intersection is not physical
contact; image exclusion is not proof of physical noncontact.

Bounds: at most 384 source-manifest entries; one-MiB seal; combined evidence
eight MiB plus at most 1,801 JSONL separator bytes; three source-member reads
within nine MiB plus separators. Selected-file identity is checked again before
publication without rereading source bytes. Four generator modules are archived
separately, each at most 512 KiB, plus the at-most-32-KiB approved stylesheet.
The rendered stylesheet must match those exact archived bytes.
The report is at most 16 MiB/32 flat files;
the complete output is at most 20 MiB/48 files. These are byte/work limits and a
cooperative deadline, not a hard process memory guarantee.

Publication uses exclusive package writes and places the terminal manifest last.
An interrupted write retains unsealed partial output; the destination is never
overwritten or reused. A completed report remains terminal. The inspector never
imports the silhouette producer or calls a model, raster decoder or simulator.

## Completed verification

- `.local/equipment-silhouette-inspection-tests-01.xml`: **31 new passes** across
  selected-byte custody, missing/changed seals, unsafe paths, unchanged source,
  deadline/read budgets, partial output, and CLI routing. Rendering is stubbed in
  this wrapper suite; these are not actual-report or native tests.
- `.local/equipment-silhouette-inspection-legacy-01.xml`: **five affected passes**
  for surface-input discovery/help, surface-anchor routing/help and fitted-lab
  CLI options. No unchanged broad suite was repeated.
- Independent review identified a missing stylesheet snapshot. The inspector
  now archives the exact bounded asset and checks equality with the rendered CSS.
  One new mismatch regression and three affected cases pass in
  `.local/equipment-silhouette-inspection-review-fix-01.xml`; **32 distinct new
  wrapper/CLI cases** are covered in total, not 35.
- The renderer's **20 distinct cases** include one actual analytical producer
  composition: seven original frames/14 pairs, exact bytes preserved. Its
  product is unchanged after root's independent read-through.
- Five later renderer cases add the optional selected-frame scope while keeping
  all six no-field outputs byte-identical: **25 renderer cases** in total. One
  new missing-seal guidance case plus an affected normal-path case pass in
  `.local/equipment-silhouette-inspection-guidance-01.xml`: **33 distinct
  wrapper/CLI cases** now pass, plus the five affected legacy checks.
- Actual public CLI rejection on the failed/unsealed source attempt passes in
  0.335489 seconds with 84,115,456 bytes observed child peak RSS and no output
  directory. Receipt `.local/equipment-evidence-report-cli-rejection-01.json`,
  SHA `5fca095d205dc6ecccebbe29eb41e27f91227de8b21ac7afe5539794ca311151`.
  No source processing was repeated. This is negative admission evidence, not
  successful rendering of a completed real-source result.
- Current base: `f1e61e6625e94df766838a1efac098ba09a9860e`. This inspector work is
  not yet committed. Root owns the later scoped publication; preserve all avatar
  files, source artifacts, failed attempts and the empty stash.

## Actual saved-source acceptance — October 31, 2026, 9:22 PM PDT

Composition 02 completed after the independently reviewed restrict-only export
fix. Its root seal is
`455665c0aaa6d71e582801033c502bbf35d4bcb3046cbb28041aa2d39ff444b4`;
the original failed 01 remains unchanged. Metadata pin
`7b83945b884c9022b25756fb447ee187a65273f2f1a44c2f29f687bc967a6cc9`
and frame pin `f3fbceb46092ea26ef38ab8fb482d99e31bfe39f01cd2e57e42c5d2a4a440138`
are retained exactly. No source processing was repeated to render the report.

The actual CLI passed in **0.382827 seconds**, with 92,815,360-byte observed child
peak RSS, 16 files and 2,621,027 output bytes. Output:
`artifacts/pose-equipment-inspection-01`, seal
`4eba1a16087850001109c3b14e847fdf6f1660941b6dcb9542d14fd49f2ef0a9`.
Acceptance `.local/equipment-evidence-inspection-acceptance-01.json`, SHA
`d00c9fa067a4201097c989cc4c73ea328de83592a4301fec64442071ade5efa6`.
Runtime `.local/equipment-evidence-inspection-runtime-02`, seal
`67b0225ea9af00780ca429174348fb0881111be5a6cbb5256eab70ba3141ac34`:
committed-base archive plus only reviewed inspector/CLI changes, not a dirty tree.

Independent saved audit verifies every output pin, all 259 source frames/1,036
pair slots, exact raw downloads, 20 local links and each scripted pair selection.
It reads 2,621,027 cached-output bytes plus 1,172,085 HTML bytes for the Node check;
no original source, raster, model or native work. All three pages retain selected-
only consistency checking and excluded labels explicitly not reconciled. Receipt
`.local/equipment-evidence-inspection-saved-audit-01.json`, SHA
`3165be98a0078199304963a791b8d1ff70ae94899715345fba0c29bca28ea4cd`.
The 1,036 scripted selections take 10.60 ms, no new DOM nodes or repeated-selection
writes. This is not browser QA or a sustained resource guarantee.

Root will publish this reviewed milestone; actual Git outcome is recorded in
`.local/equipment-evidence-publication-01.json`. Ground/equipment reconstruction
and unfamiliar-video physical accuracy remain open. The next source task applies
qualified image evidence to the existing support selector without contact claims.
