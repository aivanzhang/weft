# Automatic-equipment contact diagnosis

## Completed — November 1, 2026, 5:40 PM PST

The saved-only inspection completed: **0.691901 seconds /156,319,744 B** child
high-water RSS, three files/42,650 B. Five model compilations and ten quaternion
matrix calls validate saved evidence; zero simulation data, controls, substeps,
replays or media decodes. PID 2743 exited zero and was reaped; 303 pinned input
identities stayed unchanged. The blocked import-time subprocess probe is retained.

The complete post-reset authored episode has 254 controls, seed 19/no jitter.
Its first equipment crossing is control 57, between 1.866667 and 1.900000 seconds:

| Original term | First control | Value | Unchanged limit |
| --- | ---: | ---: | ---: |
| `sam-2` speed | 57 | 0.291410 | 0.15 |
| `sam-2` angular speed | 57 | 1.740077 | 0.75 |
| `sam-2` displacement | 59 | 0.040816 | 0.03 |
| `sam-2` orientation | 59 | 0.200667 | 0.15 |

The same saved boundary contains `right_elbow_geom`/`sam-2` contact, identified
as the right forearm; distance −0.000869 and normal force 16.602918 in simulation
units. This is boundary co-occurrence, not proof of the exact causal substep or
physical contact. Reference frames bracket 58–62, with 59–61 missing. This is the
declared reference clock, **not native-video synchronization**. Do not combine the
earlier nine-action manual/coast/authored prefix with the complete authored episode.

The full interval still fails all four `sam-2` stability terms; body/floor and
`sam-3` pass. Different geometry and controllers prevent calling its later crossing
an improvement over the old-world result. No hidden cavity/grasp was introduced.
The follow-up [temporal-arm study](temporal-arm-control.md) also remains unsuccessful;
do not repeat the same fixed coefficients or promote the weak single-start rank gain.

Output: `artifacts/auto-equipment-contact-diagnosis-01`.
Timeline SHA `8d5c85a5e919768ebe73af34302e10d34f7a68156ee55522d9dfb661ecbed87b`;
verification `ecca1e08efe316e55bf2a9086d01c592534347324ce51c96edb2b594bf32d911`;
seal `460016608e8178340b98454155ebc0f96806e813f15cca9a09eb2dc61fe8057d`;
process `c38929ba0ca4705da0aef55d2d7e852670f43aebeb615f39263cf07d8be86696`.
Root reused the already-hashed `.local/auto-equipment-contact-diagnosis-01/saved-metadata-01.json`
for interpretation; no second physics/source/media pass. No paid API or deployment.

## Original preparation — November 1, 2026, 5:20 PM PST

Root owned one bounded saved-recording inspection, now completed above. Previous
turn made progress: actual terminal interaction/recording/export/replay/resume
passed at published `91ec429`, but full authored task still fails equipment.
The old-world right-forearm finding cannot be assumed for different automatic
frame geometry. Inspect the newly recorded boundaries before another control
or reconstruction change. No hidden cavity or grasp will be invented.

Input: `artifacts/auto-equipment-console-01/lab/sessions/session-0f8f6d170f69`,
recording seal `09686e8509262fe0277f8edda093c1684c954818d7000007a6f1ae3c5f027cf1`.
Expected265events,one reset/two episodes,no learnedpolicy. Reuse the exact299-file
trustedGit runtime/manifest from the completed acceptance; no recopy or override.
Existing `inspect_recording_failure` may compile at most20models for validation
and use at most10,000quaternion-matrix arithmetic calls. It may not construct
simulation data, reset/step/replay, decode media, infer pixels or make API calls.
Limits:10-second diagnostic,30CPU/60external,512MiB observedRSS,2MiB/3outputfiles,
1MiB perlog. Cost:$0external/$0recurring. No source/task/scorer/default changes.
Output: `artifacts/auto-equipment-contact-diagnosis-01`. Scratch uses the same
suffix beneath `.local/`. The existing descriptor/output and process guards
remain; source/runtime stat identities are checked after the saved read.
Root reviewed only the stated metadata/path/count changes from the already-used
inspector harness; agents remain at usage limits, no peer review claim.

Frozen plan `4da4774f9f7ca5fe3351d64fa4664e2f84cbcca85209c48b50d9dd38b66a9501`; worker `f494eedc76ff34317548e170829b241bf27552f5db14caeb3f17b5b34581cb9d`;
supervisor `49c375fa9f40a9fac3ec29872f52198b2a662b51d7ec97044263d1a54c369073`.
One execution only; first crossings, boundary contacts and missing source-clock
coverage are retained above. Boundary coincidence is not exact substep cause or physical contact.
