"""Offline display of saved silhouette proposals; rendering never fits or admits geometry."""

import base64
from collections import Counter
import hashlib
from html import escape
from io import BytesIO
import math
import re

import numpy as np
from PIL import Image


PROFILE = "conditional-silhouette-bands-v1"
MAX_ITEMS = 12
MAX_BANDS_PER_OBJECT = 8
MAX_TOTAL_BANDS = 32
MAX_PNG_BYTES = 4 * 1024 * 1024
MAX_TOTAL_PNG_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 1_000_000
MAX_TOTAL_PIXELS = 8_000_000
MAX_HTML_BYTES = 16 * 1024 * 1024


def _mapping(value):
    if not isinstance(value, dict):
        raise ValueError("Report fields must be mappings")
    return value


def _text(value, maximum=256):
    if not isinstance(value, str) or len(value) > maximum:
        raise ValueError("Report text must be a bounded string")
    return escape(value, quote=True)


def _number(value, low=0, high=1_000_000_000):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError("Report numbers must be finite and bounded")
    return value


def _count(value, maximum=1_000_000_000):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("Report counts must be bounded nonnegative integers")
    return value


def _sha(value):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("Report requires exact SHA256 identities")
    return value


def _fmt(value):
    return format(value, ".12g")


def _vertices(value):
    if not isinstance(value, list) or len(value) != 8:
        raise ValueError("Each projected box requires its exact eight vertices")
    points = []
    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("Projected vertices require image-space pairs")
        points.append(tuple(_number(v, -20000, 20000) for v in point))
    return points


def _hull(points):
    # The eight box corners are not perimeter-ordered; sorting avoids crossing outlines.
    unique = sorted(set(points))
    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    halves = []
    for ordered in (unique, unique[::-1]):
        half = []
        for point in ordered:
            while len(half) >= 2 and cross(half[-2], half[-1], point) <= 0:
                half.pop()
            half.append(point)
        halves.extend(half[:-1])
    if len(halves) < 3:
        raise ValueError("Projected box has no visible area")
    return " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in halves)


def _mask(raw, row, hashes):
    if type(raw) is not bytes or not 1 <= len(raw) <= MAX_PNG_BYTES:
        raise ValueError("Selected PNG exceeds the exact-byte bound")
    dimensions = row.get("raster_dimensions")
    if (not isinstance(dimensions, list) or len(dimensions) != 2
            or any(type(n) is not int or not 1 <= n <= 4096 for n in dimensions)
            or math.prod(dimensions) > MAX_PIXELS):
        raise ValueError("Selected mask dimensions exceed the native-grid bound")
    file_hash = hashlib.sha256(raw).hexdigest()
    path = row.get("raster_path")
    _text(path, 1024)
    if not isinstance(hashes, dict) or hashes.get(path) != file_hash:
        raise ValueError("Selected PNG differs from the proposal's exact file identity")
    try:
        with Image.open(BytesIO(raw)) as image:
            if image.format != "PNG" or image.mode not in ("P", "L", "1") or list(image.size) != dimensions[::-1]:
                raise ValueError("Selected PNG format or dimensions differ")
            raster = np.asarray(image, dtype=np.uint8)
    except (OSError, SyntaxError) as error:
        raise ValueError("Selected PNG cannot be read") from error
    binary_hash = hashlib.sha256(raster.tobytes()).hexdigest()
    if np.any(raster > 1) or binary_hash != _sha(row.get("raster_sha256")):
        raise ValueError("Selected binary mask differs from the saved native cells")
    foreground = _count(_mapping(row.get("silhouette")).get("foreground_pixels"), raster.size)
    if not foreground or foreground != int(raster.sum()):
        raise ValueError("Selected mask foreground accounting differs")
    # Only display colors change; each saved binary cell keeps its original position and value.
    display = Image.fromarray(raster).convert("P")
    display.putpalette([255, 253, 248, 91, 111, 108] + [0] * (768 - 6))
    stream = BytesIO()
    display.save(stream, format="PNG", optimize=False)
    return base64.b64encode(stream.getvalue()).decode("ascii"), dimensions, foreground, file_hash, binary_hash


def _scores(value, pixels, foreground):
    if not isinstance(value, dict):
        raise ValueError("Missing saved silhouette scores")
    names = ("foreground_cells", "covered_foreground_cells", "projected_cells",
             "conditional_exterior_cells", "uncertain_projected_cells")
    counts = {name: _count(value.get(name), pixels) for name in names}
    if (counts["foreground_cells"] != foreground or counts["covered_foreground_cells"] > foreground
            or counts["conditional_exterior_cells"] > counts["projected_cells"]
            or counts["uncertain_projected_cells"] > counts["projected_cells"]):
        raise ValueError("Saved silhouette score accounting differs")
    score = value.get("conditional_clear_iou")
    if score is not None:
        _number(score, 0, 1)
    return counts | {"conditional_clear_iou": score}


def _clock(row):
    if row is None:
        return "Not supplied; no original/submitted clock equivalence is inferred."
    if not isinstance(row, dict):
        raise ValueError("Source clock must be explicit or absent")
    frame = _count(row.get("frame"), 499)
    stamp = _number(row.get("time_s"))
    digest = _sha(row.get("sha256"))
    return f"Frame {frame}, {_fmt(stamp)} s; video <code>{digest}</code>"


def _slots(report, selected):
    rows = report.get("all_source_slots")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 4000 or report.get("source_slots_filled") != 0:
        raise ValueError("Report must retain its bounded source slots without filling them")
    identities, counts, frames, objects = set(), Counter(), set(), set()
    selected_slots = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid source slot")
        frame, obj = _count(row.get("frame"), 499), row.get("object_id")
        _text(obj)
        identity = frame, obj
        if identity in identities:
            raise ValueError("Duplicate source slot")
        identities.add(identity)
        frames.add(frame)
        objects.add(obj)
        phase, status = row.get("phase"), row.get("mask_status")
        _text(phase)
        _text(status)
        counts["pre_initialization"] += int(phase == "pre_initialization")
        counts["unavailable"] += int(status != "decoded")
        counts["withheld"] += int(row.get("admission") == "withheld")
        if identity == (selected["frame"], selected["object_id"]):
            selected_slots.append(row)
    if (len(objects) > 8 or frames != set(range(max(frames) + 1))
            or len(rows) != len(frames) * len(objects) or len(selected_slots) != 1
            or selected_slots[0].get("submitted_time_s") != selected["time_s"]
            or selected_slots[0].get("original_source") != selected.get("original_source")):
        raise ValueError("Source registry or selected source clock differs")
    return (f"{len(rows)} entity/frame slots across {len(frames)} frames; "
            f"{counts['pre_initialization']} pre-initialization slots; {counts['unavailable']} without decoded masks; "
            f"{counts['withheld']} withheld slots; "
            "0 filled. The complete registry remains in the saved JSON, not hidden by this selected-frame view.")


def _prepare(item):
    if not isinstance(item, dict):
        raise ValueError("Each report item must be a mapping")
    object_id = item.get("object_id")
    _text(object_id)
    result, row = item.get("result"), item.get("selected_mask_row")
    if not isinstance(result, dict) or not isinstance(row, dict) or not isinstance(result.get("report"), dict):
        raise ValueError("Each item requires a saved result and selected mask row")
    report = result["report"]
    status = result.get("status")
    if (report.get("profile") != PROFILE or status not in ("conditional_proposal", "no_qualified_improvement")
            or report.get("status") != status or report.get("physical_validation") != "unverified"):
        raise ValueError("Unsupported silhouette result or physical-validation claim")
    settings, ref = _mapping(report.get("settings")), _mapping(report.get("source_reference"))
    frame = _count(row.get("frame"), 499)
    submitted = _number(row.get("time_s"))
    if (settings.get("object_id") != object_id or settings.get("source_frame") != frame
            or ref.get("frame") != frame or ref.get("object_id") != row.get("object_id")
            or ref.get("evidence_id") != row.get("id") or report.get("submitted_time_s") != submitted
            or report.get("original_source") != row.get("original_source") or row.get("mask_status") != "decoded"):
        raise ValueError("Selected source identity or clock differs from the proposal")
    _text(row.get("object_id"))
    source_hash = _sha(ref.get("document_sha256"))
    hashes = _mapping(report.get("input_hashes"))
    if source_hash not in _mapping(hashes.get("source_documents")):
        raise ValueError("Selected source document is not pinned by the proposal")
    identity = _mapping(report.get("identity"))
    if (identity != row.get("identity") or identity.get("physical_identity_verified") is not False):
        raise ValueError("Selected model identity must remain physically unverified")
    _text(identity.get("status"))
    raster, dims, foreground, file_hash, binary_hash = _mask(item.get("selected_png_bytes"), row, hashes.get("raster_files"))
    box = _mapping(row.get("box")).get("box_xyxy_halfopen")
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("Selected mask needs its original image box")
    box = [_number(v, 0, 4096) for v in box]
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("Selected image box has no area")
    candidate = _mapping(report.get("candidate"))
    bands = candidate.get("bands")
    geometry = _mapping(candidate.get("geometry"))
    if (not isinstance(bands, list) or not 1 <= len(bands) <= MAX_BANDS_PER_OBJECT or settings.get("resolved_parts") != len(bands)
            or geometry.get("profile") != "rigid-compound-v1" or geometry.get("origin") != "authored_hypothesis"
            or not isinstance(geometry.get("parts"), list) or len(geometry["parts"]) != len(bands)):
        raise ValueError("Silhouette display requires one to eight authored solid box bands")
    seen, projections, lost_hulls = set(), [], 0
    for band, part in zip(bands, geometry["parts"]):
        _mapping(band)
        _mapping(part)
        name = band.get("part_id")
        _text(name)
        if name in seen or name != part.get("id") or part.get("shape") != "box" or type(band.get("frozen_original_width")) is not bool:
            raise ValueError("Band and primitive identities differ")
        seen.add(name)
        vertices = _vertices(band.get("projected_vertices_px"))
        projections.append((name, vertices, band["frozen_original_width"]))
        lost_hulls += _count(band.get("lost_required_cells"), math.prod(dims))
    baseline = item.get("baseline_vertices_px")
    if baseline is not None:
        baseline = _vertices(baseline)
        extents = [min(p[0] for p in baseline), min(p[1] for p in baseline),
                   max(p[0] for p in baseline), max(p[1] for p in baseline)]
        if any(abs(a - b) > 1e-6 for a, b in zip(box, extents)):
            raise ValueError("Original box projection differs from the selected source bounds")
    uncertain = report.get("uncertain_rows")
    if (not isinstance(uncertain, list) or len(uncertain) > dims[0]
            or any(type(n) is not int or not 0 <= n < dims[0] for n in uncertain)
            or uncertain != sorted(set(uncertain))):
        raise ValueError("Uncertain rows must be unique native row indices")
    pixels = math.prod(dims)
    scores = [_scores(_mapping(report.get("baseline")).get("scores"), pixels, foreground), _scores(candidate.get("scores"), pixels, foreground)]
    losses = [_count(report.get(name), pixels) for name in
              ("lost_original_foreground_cells", "lost_original_uncertain_cells", "new_outside_original_cells")]
    qualification = report.get("qualification", {})
    expected = {"conditional_exterior_opt_in": settings.get("conditional_exterior"),
                "original_foreground_preserved": losses[0] == 0, "uncertain_envelope_preserved": losses[1] == 0,
                "within_original_envelope": losses[2] == 0, "all_band_row_hulls_preserved": lost_hulls == 0,
                "strict_conditional_image_improvement": (scores[1]["conditional_clear_iou"] is not None
                    and scores[0]["conditional_clear_iou"] is not None
                    and scores[1]["conditional_clear_iou"] > scores[0]["conditional_clear_iou"] + 1e-12)}
    if (any(type(v) is not bool for v in expected.values()) or qualification != expected
            or (status == "conditional_proposal") != all(expected.values())
            or report.get("rejection_reasons") != [name for name, ok in expected.items() if not ok]):
        raise ValueError("Result status, qualification and saved counts conflict")
    holes = _count(report.get("filled_internal_hole_cells"), pixels)
    original_clock = _clock(report.get("original_source"))
    slots = _slots(report, row)
    return {"object_id": object_id, "status": status, "row": row, "report": report,
            "raster": raster, "dims": dims, "pixels": pixels, "png_bytes": len(item["selected_png_bytes"]),
            "file_hash": file_hash, "binary_hash": binary_hash, "box": box, "baseline": baseline,
            "projections": projections, "uncertain": uncertain, "scores": scores, "losses": losses,
            "holes": holes, "original_clock": original_clock, "slots": slots}


def _svg(item, index, candidate):
    box, bands = item["box"], item["projections"]
    points = [(box[0], box[1]), (box[2], box[3]), *(item["baseline"] or [])]
    points.extend(point for _, vertices, _ in bands for point in vertices)
    left, top = min(x for x, _ in points), min(y for _, y in points)
    width, height = max(x for x, _ in points) - left, max(y for _, y in points) - top
    padding = max(width, height) * .05
    stroke = max(width, height) * .0045
    view = " ".join(_fmt(n) for n in (left - padding, top - padding, width + padding * 2, height + padding * 2))
    label = "Candidate solid band projections" if candidate else "Original proxy projection" if item["baseline"] else "Original image-box bounds only"
    suffix = f"{index}-{'candidate' if candidate else 'baseline'}"
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title-{suffix}" viewBox="{view}">',
           f'<title id="title-{suffix}">{label}: {_text(item["object_id"])}</title>',
           f'<rect x="{_fmt(box[0])}" y="{_fmt(box[1])}" width="{_fmt(box[2]-box[0])}" height="{_fmt(box[3]-box[1])}" fill="#fffdf8"/>',
           f'<image x="{_fmt(box[0])}" y="{_fmt(box[1])}" width="{_fmt(box[2]-box[0])}" height="{_fmt(box[3]-box[1])}" preserveAspectRatio="none" style="image-rendering:pixelated" href="data:image/png;base64,{item["raster"]}"/>']
    # Consecutive rows share one rectangle, avoiding a DOM node per native cell.
    runs = []
    for row in item["uncertain"]:
        if runs and runs[-1][1] == row:
            runs[-1][1] = row + 1
        else:
            runs.append([row, row + 1])
    dy = (box[3] - box[1]) / item["dims"][0]
    for start, end in runs:
        svg.append(f'<rect class="uncertain-row" data-native-start="{start}" data-native-end="{end}" x="{_fmt(box[0])}" y="{_fmt(box[1]+start*dy)}" width="{_fmt(box[2]-box[0])}" height="{_fmt((end-start)*dy)}" fill="#5b91b4" fill-opacity=".24" stroke="#356b8e" stroke-dasharray="{_fmt(stroke*2)} {_fmt(stroke*2)}" stroke-width="{_fmt(stroke/2)}"/>')
    outlines = bands if candidate else [("original-proxy", item["baseline"], False)] if item["baseline"] else []
    for name, vertices, frozen in outlines:
        color = "#9a5e19" if frozen else "#965644" if candidate else "#214f50"
        dash = f' stroke-dasharray="{_fmt(stroke*3)} {_fmt(stroke*1.5)}"' if frozen else ""
        svg.append(f'<polygon data-part="{_text(name)}" data-frozen="{str(frozen).lower()}" points="{_hull(vertices)}" fill="none" stroke="{color}" stroke-width="{_fmt(stroke)}"{dash}><title>{_text(name)}; {"original width frozen" if frozen else "saved projection"}</title></polygon>')
    if not candidate and item["baseline"] is None:
        svg.append(f'<rect class="source-box-only" x="{_fmt(box[0])}" y="{_fmt(box[1])}" width="{_fmt(box[2]-box[0])}" height="{_fmt(box[3]-box[1])}" fill="none" stroke="#214f50" stroke-dasharray="{_fmt(stroke)} {_fmt(stroke*1.5)}" stroke-width="{_fmt(stroke)}"/>')
    return "".join(svg) + "</svg>"


def _score(value):
    return "Unavailable (no clear-cell union)" if value is None else f"{value:.6f}"


def _item_html(item, index):
    report, row, before, after = item["report"], item["row"], *item["scores"]
    frozen = sum(part[2] for part in item["projections"])
    title = "Conditional image proposal — not admitted by this report" if item["status"] == "conditional_proposal" else "No qualified improvement — candidate retained, not selected"
    explanations = {
        "conditional_exterior_opt_in": "Conditional exterior assumption was not enabled",
        "original_foreground_preserved": "Previously covered foreground cells were lost",
        "uncertain_envelope_preserved": "Uncertain original-envelope cells were lost",
        "within_original_envelope": "New projected cells extend outside the original envelope",
        "all_band_row_hulls_preserved": "Required row-hull cells were lost",
        "strict_conditional_image_improvement": "No strict conditional image-overlap improvement",
    }
    reason = "" if not report["rejection_reasons"] else "<ul>" + "".join(f"<li>{explanations[name]}</li>" for name in report["rejection_reasons"]) + "</ul>"
    table_rows = [("Conditional clear-cell IoU", _score(before["conditional_clear_iou"]), _score(after["conditional_clear_iou"]))]
    for key, label in (("covered_foreground_cells", "Covered foreground cells"), ("projected_cells", "Projected cells"),
                       ("conditional_exterior_cells", "Conditional exterior cells"), ("uncertain_projected_cells", "Uncertain projected cells")):
        table_rows.append((label, str(before[key]), str(after[key])))
    table = "".join(f"<tr><th scope=\"row\">{label}</th><td>{old}</td><td>{new}</td></tr>" for label, old, new in table_rows)
    loss_rows = " · ".join(f"{label}: <strong>{count}</strong>" for label, count in zip(
        ("Lost original foreground cells", "Lost uncertain envelope cells", "New outside-envelope cells"), item["losses"]))
    baseline_label = "Original box projection" if item["baseline"] else "Original image-box bounds (proxy projection not supplied)"
    return f'''<section class="object" aria-labelledby="object-{index}">
<h2 id="object-{index}">{_text(item['object_id'])}</h2><p class="status">{title}</p>{reason}
<div class="views"><figure>{_svg(item,index,False)}<figcaption>{baseline_label}</figcaption></figure><figure>{_svg(item,index,True)}<figcaption>Saved candidate: {len(item['projections'])} solid bands; {frozen} retain their full original width</figcaption></figure></div>
<p class="legend"><span class="mask-key">■</span> Native mask foreground · <span class="uncertain-key">▧</span> Blue/dotted rows: competing foreground or empty-row uncertainty · <span class="frozen-key">– –</span> Dashed amber: frozen band width</p>
<p>Same native {item['dims'][1]} × {item['dims'][0]} mask cells in both views; display colors only, no cell resampling. Outlines use saved projected corners, not a newly fitted shape.</p>
<table><thead><tr><th>Saved image comparison</th><th>Original</th><th>Candidate</th></tr></thead><tbody>{table}</tbody></table>
<p class="losses">{loss_rows}</p>
<p>IoU means intersection divided by union on clear native mask cells. It depends on the explicit conditional exterior assumption, not measured empty space. {item['holes']} internal hole cells were filled in the solid row hull; no handle, cavity, grasp or support is recovered.</p>
<dl><dt>Submitted/model clock</dt><dd>Frame {row['frame']}, {_fmt(row['time_s'])} s</dd><dt>Original-source clock</dt><dd>{item['original_clock']}</dd><dt>Construction-observation clock</dt><dd>{_fmt(_number(report['source_reference'].get('time_s')))} s (retained separately)</dd><dt>Model entity</dt><dd>{_text(row['object_id'])}; {_text(report['identity']['status'])}. Physical identity is unverified.</dd></dl>
<p>{item['slots']}</p><details><summary>Input identities and interpretation limits</summary><dl><dt>Observation document SHA256</dt><dd><code>{_sha(report['source_reference']['document_sha256'])}</code></dd><dt>Exact saved PNG SHA256</dt><dd><code>{item['file_hash']}</code></dd><dt>Native binary-cell SHA256</dt><dd><code>{item['binary_hash']}</code></dd><dt>Settings SHA256</dt><dd><code>{_sha(report.get('settings_sha256'))}</code></dd><dt>Camera SHA256</dt><dd><code>{_sha(report['settings'].get('camera_sha256'))}</code></dd></dl><p>The mask producer's pixels are preserved, not independently verified against the video. Object pose, depth, mass, materials and camera stay fixed. Original centered-box inertia remains an authored prior, not a measured center of mass. Missing, pre-initialization and withheld evidence is not filled.</p></details>
</section>'''


def _clearance(value):
    caveat = "Frozen-reference overlap only, with equipment fixed at its declared poses. Not observed contact, continuous-time clearance, dynamic stability, task success or physical validation."
    if value is None:
        return f'<section><h2>Full-motion clearance</h2><p>Not attached to this report. No clearance result is inferred from image overlap.</p><p>{caveat}</p></section>'
    if not isinstance(value, dict) or value.get("status") not in ("complete", "incomplete", "unsupported_input"):
        raise ValueError("Unsupported clearance-summary status")
    summary, comparisons = value.get("summary"), value.get("comparison_summary", [])
    if not isinstance(summary, dict) or not isinstance(comparisons, list) or len(comparisons) > 512:
        raise ValueError("Clearance summary exceeds its bounded comparison contract")
    statuses = summary.get("status_counts", {})
    allowed = {"checked", "unavailable", "partially_checked", "not_checked"}
    if not isinstance(statuses, dict) or set(statuses) - allowed:
        raise ValueError("Unknown clearance sample status")
    scheduled = _count(summary.get("scheduled"), 1_000_000)
    counts = {name: _count(count, scheduled) for name, count in statuses.items()}
    if sum(counts.values()) != scheduled:
        raise ValueError("Clearance sample accounting differs")
    changes = _count(summary.get("pointwise_increases"))
    new = _count(summary.get("new_penetrations"))
    before, after = [], []
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            raise ValueError("Invalid clearance pair summary")
        for key, values in (("baseline_max_penetration_world", before), ("alternative_max_penetration_world", after)):
            depth = comparison.get(key)
            if depth is not None:
                values.append(_number(depth, 0, 10000))
    maximum = lambda values: _fmt(max(values)) + " world units" if values else "Unavailable"
    label = {"complete": "Comparison computation complete", "incomplete": "Comparison incomplete", "unsupported_input": "Comparison input unsupported"}[value["status"]]
    return f'''<section><h2>Full-motion clearance</h2><p class="status">{label} — not a physical-validity verdict</p><p>Scheduled samples: {scheduled}; checked: {counts.get('checked',0)}; unavailable: {counts.get('unavailable',0)}; partially checked: {counts.get('partially_checked',0)}; not checked: {counts.get('not_checked',0)}.</p><p>Pointwise penetration increases: {changes}; new penetrations: {new}. Maximum reported penetration: original {maximum(before)}; candidate {maximum(after)}.</p><p>{caveat} Full pair and source-clock details remain in the saved clearance JSON.</p></section>'''


def render_silhouette_report(items, clearance_summary=None) -> bytes:
    """Render bounded verified-result views, without filesystem, network, fit or physics work.

    Items supply object_id, result, selected_mask_row and selected_png_bytes. Optional
    baseline_vertices_px supplies the exact original eight projected box corners.
    """
    if not isinstance(items, (list, tuple)) or not 1 <= len(items) <= MAX_ITEMS:
        raise ValueError("Use one to twelve bounded silhouette report items")
    prepared, names, pixels, raw_bytes, bands = [], set(), 0, 0, 0
    for item in items:
        row = _prepare(item)
        if row["object_id"] in names:
            raise ValueError("Duplicate report object identity")
        names.add(row["object_id"])
        pixels += row["pixels"]
        raw_bytes += row["png_bytes"]
        bands += len(row["projections"])
        if pixels > MAX_TOTAL_PIXELS or raw_bytes > MAX_TOTAL_PNG_BYTES or bands > MAX_TOTAL_BANDS:
            raise ValueError("Aggregate report images or bands exceed their bound")
        prepared.append(row)
    content = "".join(_item_html(item, index) for index, item in enumerate(prepared)) + _clearance(clearance_summary)
    html = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"><title>Weft · Conditional silhouette comparison</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5eadc;color:#214f50;font:16px/1.5 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:32px 20px}h1{font-size:2rem;line-height:1.15}h2{margin:0 0 12px}section{background:#fffdf8;border:1px solid #cebcaa;border-radius:10px;padding:22px;margin:24px 0}.status{font-weight:650;color:#79452f}.views{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}figure{margin:0;min-width:0}svg{display:block;width:100%;height:300px;border:1px solid #d6cbc0;background:#f5f0e9}figcaption{font-weight:600;margin-top:8px}.legend{font-size:.875rem}.mask-key{color:#5b6f6c}.uncertain-key{color:#356b8e}.frozen-key{color:#9a5e19}table{border-collapse:collapse;width:100%;font-size:.95rem}th,td{padding:8px;text-align:left;border-bottom:1px solid #d6cbc0}thead{background:#f5eadc}dt{font-weight:650}dd{margin:0 0 10px}code{overflow-wrap:anywhere;font-size:.83rem}.losses{padding:10px;background:#f5eadc}summary{cursor:pointer;font-weight:650}details{margin-top:18px}details dl{padding-top:12px}.intro{max-width:84ch}.scope{border-left:4px solid #965644;padding-left:14px}@media(max-width:660px){.views{grid-template-columns:1fr}main{padding:20px 12px}section{padding:16px}svg{height:280px}th,td{padding:6px;font-size:.85rem}}@media print{body{background:white}section{break-inside:avoid}details{display:block}main{max-width:none}}
</style></head><body><main><h1>Conditional silhouette comparison</h1><p class="intro">A saved image-based alternative to the original box, not a recovered physical object. Compare solid outlines against the selected native mask, then inspect uncertainty and the separate motion-clearance results.</p><p class="scope">Visible-silhouette extrusion is an explicit authored assumption. Absent mask pixels are not measured empty space. No camera/depth recovery, measured anatomy, handle, cavity, grasp or physical/task-pass claim is made.</p>''' + content + "</main></body></html>"
    encoded = html.encode("utf-8")
    if len(encoded) > MAX_HTML_BYTES:
        raise ValueError("Rendered silhouette report exceeds 16 MiB")
    return encoded
