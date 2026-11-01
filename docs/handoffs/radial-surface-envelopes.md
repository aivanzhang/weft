# Certified positive-radial surface envelopes

## Frozen — October 31, 2026, 5:19:58 PM PDT

Owner `/root/exercise_reconstruction`; root approved edits only to
`src/weft/source_surface_envelopes.py`, new
`tests/test_radial_surface_envelopes.py`, this note, and the single obsolete
radial-unavailable fixture coefficient `+.1`→`-.1` in the old envelope test.
No fitter, CLI, camera model, report, scorer, source-data or Git edits. No native,
media/model inference, network/API call or real source study; no live handles.

The prior integrated surface fit was published at `259cfc2`. Its single authored
native worker completed with all twelve gates, but its outer acceptance failed
on one blocked nested-process attempt. Root owns that diagnosis; this lane does
not rerun or relabel it.

## Approved domain and enclosure argument

Preserve the exact supplied forward model
`d = u * (1 + k * ||u||²)` and raw K/R/t/k bytes. For `0 < k <= 1` (the positive
half of the existing camera validator's domain), invert only if the entire
authored distorted pixel rectangle has normalized radius <=4, matching the
existing inverse routine's domain. Negative k stays unavailable: no ambiguous
branch is silently selected. Zero k keeps its exact previous serialized result.

1. Exact-rational `K^-1` maps the pixel rectangle to a normalized distorted
   parallelogram. Its coordinate extrema come from affine corners. The maximum
   squared radius is at a corner by convexity; distance from zero to its outer
   coordinate box gives a conservative minimum squared radius.
2. Write the inverse as `u = alpha*d`. For squared distorted radius `s`, alpha
   satisfies `alpha + k*s*alpha³ = 1`. This function has derivative
   `1 + 3*k*s*alpha² > 0` on `[0,1]`, so its root is unique. Implicit monotonicity
   gives decreasing alpha with increasing s. Two exact-rational bisection
   brackets therefore enclose all inverse scales over the whole pixel set.
3. Interval products of that positive scale range and each distorted coordinate
   range enclose every undistorted ray, including interior pixels. Correlation
   loss makes this an outer box, not a set of reconstructed point measurements.
4. Apply the unchanged exact `R^-1` and original horizontal height interval.
   Over the outer ray rectangle, a constant-sign affine plane denominator gives
   linear-fractional coordinate extrema at its corners. Height is affine, so
   its two endpoints suffice. Reject a horizon crossing, nonforward/out-of-bound
   depth, or support-domain overflow. Final float bounds round outward.

Numerical bisection probes are not chosen geometric targets. No midpoint is
returned, no coefficient is zeroed/refitted, and sampled interior checks are
supplements to this proof rather than the justification for an enclosure.

## Bounds / compatibility checkpoint

At most 48 exact bisections per scalar bracket /96 per uncached pixel box;
cache at most 256 boxes. Every iterative path checks the existing ten-second
cooperative deadline; overflow/time exhaustion cannot return a partial source
ledger. Existing <=1,800 source slots, <=900 annotated frames, <=12 channels,
4 MiB additional inputs, 128 KiB per row and 8 MiB complete cache remain.

Before edits, three purely analytical zero-k cases were frozen under existing
native/model/media/network/subprocess denials: ordinary camera, skew/oblique
camera, and missing camera. Full metadata/frames and their byte hashes are saved
once in `.local/radial-surface-zero-baseline-01.json` (35,179B), SHA
`8c3ac84d632fe5a672e24bebc08a9fcf488bcd8e5d2c9a41832a65f092a60399`.
This is a byte-compatibility fixture, not a source study. All three cases now
pass frozen metadata and every frame hash comparisons, not only numeric bounds.

## Implemented public boundary

The existing `prepare_source_surface_envelopes(...)` API automatically uses the
certified path for qualified positive coefficients; no new caller argument or
profile substitution is required. Negative k keeps the original unavailable
reason, `radial_distortion_has_no_certified_interval_bound`. A positive pixel
rectangle outside radius four is unavailable with
`radial_pixel_interval_exceeds_inverse_domain`; it is never clipped. Original
slot admission order, timing, missing/occluded/excluded/unsampled states and all
source identities remain intact.

Positive results add `radial_inverse` metadata and per-anchor diagnostics under
`certified-nonnegative-radial-inverse-v1`, retaining raw k, distorted normalized
intervals, radius bounds, exact-bisection scale brackets and the outer
undistorted ray box. Their bound certificate is
`exact_rational_monotone_radial_outer_box_plane_extrema_outward_binary64`.
The zero-k branch adds no keys or changed certificate text. Only the exact
internal fractions feed the geometry; published float diagnostics do not feed
back into the enclosure.

The bounded cache is shared within one preparation and keyed by the exact
pixel-box endpoints, with fixed K/k. Public rows are detached from its internal
records; cached queries neither recompute geometry nor permit result mutation
to change future rows. Cache eviction can increase work, not change an answer.

## Verification / frozen identities

Executed once, with native/model/media/network/subprocess operations denied:

```sh
PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_radial_surface_envelopes.py 'tests/test_source_surface_envelopes.py::test_uncertified_geometry_abstains_without_clipping_or_fabricated_height[radial-radial_distortion_has_no_certified_interval_bound]' -o junit_family=legacy --junitxml=.local/radial-surface-envelope-tests-01.xml
```

Result: **39 passed**, no failures/skips, 0.822 seconds in the JUnit suite.
These are 38 new checks plus the one authorized old negative-radial case; no
unchanged full envelope/fitter suite or prior native acceptance was rerun.

The cases cover exact polynomial bracket inequalities/uniqueness, monotone
scale, independently authored forward points in every sign quadrant, exact
axis/origin cases, and independent high-precision inverse/plane checks for
skewed/oblique cameras at corners and interior pixels. They also cover negative
branches, radius boundaries, horizon/forward-depth/support-domain guards,
measured-calibration receipt admission, non-nominal clocks, full slot retention,
immutable raw bytes and query results, cache eviction, deadline interruption,
complete-result byte caps, and the three byte-identical zero-k cases.

Representative analytical workload: 600 original frames, two channels and
**1,200 distinct pixel boxes**, 839,170 exact input bytes. Preparation took
0.659638667 seconds for 115,200 exact bisections; twenty cached queries took
0.000292583 seconds. Complete retained cache: 2,241,196 bytes. These are local
fixture timings, not a universal latency/memory guarantee; the ten-second
deadline remains cooperative, not process isolation. Every bounded query uses
the already-prepared cache. No source image, decoder, detector, native physics,
provider, training or fitting work ran; paid cost $0. No live handles.

| Frozen path | Bytes | SHA-256 |
| --- | ---: | --- |
| `src/weft/source_surface_envelopes.py` | 24,596 | `d93935cb9be517a549fdf8f9c9b65dcf76dc45696d87f6ad11c45baa84709adc` |
| `tests/test_radial_surface_envelopes.py` | 18,116 | `812ee7d3015075f98d455244571a44ae416720c777c21d756545441329cf59ad` |
| `tests/test_source_surface_envelopes.py` (one coefficient only) | 15,168 | `d416aef4f77f0c2fd5ee4c42eba58252cd3a71c45999b05b574eb97cbe3be3c7` |
| `.local/radial-surface-envelope-tests-01.xml` | 9,298 | `25364b70dc7254494a4dfb001b98b8a03bdf63a24492bc0645bb17b0ea90d883` |

Scoped tracked-file whitespace check passed. The new test's no-index
whitespace check emitted no diagnostics (exit one denotes its new-file diff).
No Git mutations were performed; root owns review/publication and preservation
of other lanes' work.

## Remaining evidence boundary

The existing jogging coefficient `+0.00047528394168603385` is mathematically in
the implemented positive domain, not automatically qualified for every annotated
pixel box. Missing real intervals/correspondences/body-world mapping and the
whole-body fitter's camera-reference rejection remain separate. This extension
does not recover a heel, contact/load/grasp, anatomy, physical accuracy, or an
unfamiliar-video reconstruction. No saved source is invoked in this milestone.

Root separately owns a single saved-camera metadata/domain check; this agent
does not duplicate that read. Such a check cannot supply missing real intervals,
correspondences or body bindings, and does not admit a reconstruction.

Next: root independently reviews these four released paths and integrates the
change. No new source/native study is authorized or needed for this isolated
mathematical extension. Any later real construction must still qualify the
original camera/body/source/world/clock bindings, explicitly supplied ranges,
all original source slots, and existing independent admission/scoring gates.
