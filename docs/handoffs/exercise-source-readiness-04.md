# New body–ground source screen

## Result — November 1, 2026, 4:52 PM PST

**No acquisition-ready unit qualified among four new dataset/project leads.**
BioCV has the strongest relevant independent measurements, but requires an
access request and explicit terms agreement. No such request or agreement was
made. This is a bounded negative result, not an exhaustive dataset finding or
a blocker for other product work.

Owner: `/root/reconstruction_next_step`. Published numerical/report checkpoint
`7cbec9179a73aa8652bab59d507f83c5a0114785` was not changed. This lane created
only this note and `.local/exercise-source-readiness-04/`. No accounts, forms,
messages, downloads of data/code/models, media processing, native calls, paid
work or Git operations. No source pixels were viewed; no new holdout exists.

## Four-source decision

| Official source | Relevant evidence | Current acquisition boundary |
| --- | --- | --- |
| [BioCV, University of Bath](https://researchdata.bath.ac.uk/1258/) — DOI `10.15125/BATH-01258` | Synchronized camera video, optical marker tracks and floor-embedded force plates; numeric intrinsics, world-to-camera transforms and distortion files. A new jump/hop case would test body–ground reconstruction using separate measurement systems. | Restricted access; the publisher requires explicit agreement. Participant archives are typically 15 GB. No accessible individual clip, byte count or media hash was established. |
| [SportsPose, DTU](https://orbit.dtu.dk/en/datasets/the-sportspose-dataset/) — DOI `10.11583/dtu.29382803` | Seven views of dynamic sports, with a published comparison to a marker-based system. | Custom academic license, academic use only, access by request. No request. A dataset-level accuracy comparison is not independent per-frame contact evidence. |
| [UMPM, Utrecht](https://www.projects.science.uu.nl/umpm/download.shtml) | [Official packaging](https://www2.projects.science.uu.nl/umpm/data_description.html) describes synchronized video, separately supplied camera calibration and C3D marker tracks; its example `p4_ball_2` joins these files. | Name/email submission and release agreement required. Modification/commercial use and redistribution need prior approval. No request. Exact floor/equipment geometry, contact channels and small file sizes remain unverified. |
| TUM Kitchen — [official author thesis](https://mediatum.ub.tum.de/doc/973175/973175.pdf), [former project endpoint](http://kitchendata.cs.tum.edu) | Calibrated table-setting video, full-body tracks, RFID/object and door/drawer readings. Body tracks are markerless estimates, not independent measured anatomy. | The former endpoint was inaccessible through the metadata tool. No current dataset reuse license, accessible bounded unit or measured floor/support transform qualified. Household activity is also not an exercise case; it is not substituted for one. |

The four names were absent from the specifically checked prior readiness
records. That does **not** certify absence from every historical project
artifact. Any future acquisition must first check the source/exposure registry
and assign its development/evaluation partition before reading pixels.

## BioCV: useful lead, not permission

The [authors' data descriptor](https://pmc.ncbi.nlm.nih.gov/articles/PMC11604968/)
describes primary counter-movement jump and hopping trials with cameras,
markers and force recordings. Camera calibration is spatially aligned to the
motion-capture system; a ground-calibration L-frame is documented. The files
distinguish raw marker trajectories from processed joint centres. Some trials
lack the auxiliary LED synchronization file and have approximate timing; listed
missing/corrupted trials must not be silently repaired.

The article's CC BY 4.0 license is **not** a license for the separately restricted
dataset. Bath explicitly states: “We ask that users explicitly agree to suitable
terms before downloading.” Its linked
[LICENSE.md](https://researchdata.bath.ac.uk/1258/24/LICENSE.md) could not be read
because the tool rejected its Markdown content type; no alternate download was
attempted. The public restriction is already sufficient to stop acquisition.

Preserve the documentation discrepancy: the paper says 1920×1080 video; Bath's
technical description says 1920×1280. No file was inspected to resolve it.
Numeric floor/force-plate corner coordinates, calibration uncertainty, per-trial
force channel definitions and their exact clock join remain unverified here.
Force-plate data would test ground loading, not handheld-equipment cavities,
grasp or contact forces. Processed joints and photogrammetry remain estimates.

## Next action, conditional on changed access

There is **no authorized acquisition command, nominated clip or media SHA**.
Do not repeat these requests under unchanged conditions or open the large
participant archives. The next concrete measurement candidate is one complete
primary BioCV sub-maximal counter-movement-jump trial, but only if the user
separately authorizes its terms/access or the publisher provides suitable open
rights and no-agreement access.

After that prerequisite, root may review a metadata-only inventory plan: identify
one exact trial/camera, its updated calibration, raw marker/force record,
auxiliary LED timing and missing-marker notes. Require a stable individual-file
delivery route and explicit hashes/sizes before acquisition. Prefer **≤25 MiB
video +1 MiB sidecars/measurements**. If complete measurements exceed that bound,
re-plan before fetching; do not drop channels, observations or timestamps to
fit it. Reserve another untouched trial before any pixel/model exposure.

This would advance independent body–ground generalization, not establish moving
equipment geometry or full-task success. No existing per-clip placement, exposed
benchmark or new pinhole setting was used to make a source appear qualified.

## Bounds and receipt

Prior limits were read from `exercise-contact-source-readiness-03.md`,
`exercise-source-readiness-02.md`, `contact-reconstruction-resume.md` and
`.local/fresh-source-readiness-01.json`; targeted name checks also covered
`exercise-floor-calibration-gap.md` and `exercise-reconstruction.md`.
No blocked OpenCap/Fit3D/MoVi/Panoptic/MM-Fit/UCOPhyRehab/BEHAVE/RICH/
OpenBiomechanics/AddBiomechanics/GroundLink route was reopened. Incidental search
results were not selected or acquired.

The bounded screen used **four dataset/project candidates, six search queries
and seven page/resource opens** (including three unsuccessful responses).
The selected Nature page failed on an identity-provider redirect; the official
article mirror appeared in search. The retired TUM endpoint and BioCV Markdown
license also failed. No retry bypass or protected access followed.
Tool-return metadata totaled **132,347 serialized bytes**, not measured network
transfer. Only the selected findings/URLs and operation accounting are retained;
unrelated search results are not copied into the repository.

Receipt: `.local/exercise-source-readiness-04/receipt.json`. It records exact
queries/URLs, current access gaps, start/stop times, the ten-minute cap and $0
external spend. No source tree, dataset archive or saved media scan occurred.
