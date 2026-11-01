# Recover the completed camera fit report

## Current result

The fit itself completed and is not rerun. Its original failed inspection remains at `artifacts/camera-reference-real-01/fit-02`; the completed proposal is its `proposal/` directory. Recovery must use a new report output directory, never overwrite that study.

The offline reporter now preserves the existing 16 MiB input/output and 2 MiB HTML limits. Ordinary small and legacy reports retain their prior layout. Only an oversized camera report uses this sequence:

1. Share artifact bytes only when their original hashes, lengths **and bytes** agree. Every logical selected member and raw pin remains present.
2. If still oversized, gzip only the copied source `report.json`, using a fixed header and compression level. Bounded decompression must reproduce the exact original bytes before any output is written.
3. Keep generated `report.json`, inline HTML data, all source/grid/unknown rows, gate results and original evidence unchanged. Reject any remaining overflow.

`selected_members` retains logical raw SHA256/length. The optional `artifact_storage` map records each logical member's stored path, encoding, SHA256 and length. Two logical names may resolve to one artifact. Gzip links say “decompress to read”; the interval-trial link follows the actual compressed filename. No arbitrary evidence compression, threshold change, physics admission or source promotion was added.

## Exact failure diagnosis

Saved metadata only, with native/model/network/report writes denied:

| Quantity | Bytes |
|---|---:|
| Original output before seal | 20,026,151 |
| Allowed before reserved seal | 16,711,680 |
| Duplicate original/derived evidence | 1,903,436 |
| Duplicate original/surface camera | 127,993 |
| Excess remaining after those aliases | 1,283,042 |
| Original HTML | 1,778,392 |

The selected source report is 7,856,096 bytes. Diagnosis read 19 cached package files / 15,709,083 bytes in 0.397414 seconds, plus the small local palette. It retained 120 source rows, 239 grid rows and all 240 camera/surface slots. No source video was read.

Reproducible diagnostic script: `.local/run_camera_reference_report_size_probe_01.py`; the original overflow receipt is `.local/camera-reference-report-size-diagnosis-01.json` (SHA `74666fcfd45903cfd77eec69f0dadbcbf11349f8f9b73acf911d960e33bef27f`). This script targets the old failure and must not be rerun against the fixed reporter as an unchanged check.

## Verification and release

- Reporter SHA: `5688e2cbee1029bbcabe4b35d604f9b207671f63af68853434582caca605b64a`.
- New test SHA: `6b2233d872f963b12ba04982b889dfae2fd3c3a1c8a4ec41d3e3740ad4ad136e`.
- Five distinct new analytical cases and two affected legacy/navigation cases pass. The first run's test-only `sample_kind`/`kind` assertion typo remains in XML01; only that failed case was corrected and rerun, with the two affected regressions.
- Cases cover ordinary layout, alias-only success, the recorded overflow size envelope, deterministic gzip/exact roundtrip, all logical pins/link targets, preserved failed/unknown states, alias mismatch and continued rejection below the required size.
- The size-envelope test uses repeated analytical bytes at recorded sizes; it is **not** a measured compression ratio for the actual source report.
- Existing scripted camera navigation performed 200 updates in 0.017083 seconds with 109 retained point nodes. This is mocked DOM evidence, not browser performance.

Receipts: `.local/camera-reference-report-storage-tests-01.xml`, `-02.xml`, `-03.xml` and `.local/camera-reference-report-storage-checks-01.json`. No native, fitter, media, browser, source acquisition or Git work was performed in this lane. Root owns actual saved-report recovery and the independent matched controller trial; this release does not claim either has run.
