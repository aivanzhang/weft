"""Offline inspection of sealed reference proposals; no geometry is recomputed."""

from bisect import bisect_left
from collections import Counter
from datetime import datetime
from hashlib import sha256
import base64
import html
import json
import math
import os
from pathlib import Path
import re
import time
from zoneinfo import ZoneInfo

from weft.shared_world_packaging import read_package_file, write_package_files


PROFILE = "whole-body-reference-report-v1"
METHODS = {"blocked-whole-body-floor-reference-v1", "blocked-whole-body-floor-first-reference-v1"}
LIMIT = .025
MEMBERS = {"candidate-human.json": "candidate-human.json.txt", "report.json": "proposal-report.json.txt",
           "inputs/human.json": "source-human.json.txt", "inputs/evidence-derived.json": "evidence-derived.json.txt"}
MAX_BYTES, MAX_HTML = 16 * 1024 * 1024, 2 * 1024 * 1024
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_TARGET_HASHES = ('original_observation_sha256', 'proposal_sha256', 'decision_sha256')
_CROP_METHOD = 'single-view-selective-crop-v1'
_COUNTERS = {'floor_qp_calls', 'image_qp_calls', 'floor_descent_accepts', 'image_tiebreak_accepts',
             'invalid_directions', 'zero_directions', 'backtrack_trials', 'rejected_backtracks', 'qp_status_counts'}
_QP_STATUSES = {'optimal', 'iteration_limit', 'time_limit', 'unsupported_infeasible_initial', 'dependent_active_set',
                'kkt_verification_failed', 'solution_bound', 'numerical_feasibility_loss', 'dependent_blocker', 'other'}
_PRIMITIVES = {'torso_geom': 'Torso', 'head_geom': 'Head', 'shoulder_frame': 'Shoulder frame', 'pelvis_frame': 'Pelvis frame',
               **{f'{side}_{joint}_geom': f'{side.capitalize()} {segment} capsule' for side in ('left', 'right')
                  for joint, segment in (('knee', 'shin'), ('hip', 'thigh'), ('shoulder', 'upper arm'), ('elbow', 'forearm'))}
              }


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _canonical(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate report JSON key")
            result[key] = value
        return result
    try:
        result = json.loads(raw, object_pairs_hook=pairs)
        json.dumps(result, allow_nan=False)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ValueError('Report evidence must be bounded strict JSON') from error
    _require(isinstance(result, dict), "Report evidence must be a JSON object")
    return result


def _text(value, maximum=512):
    _require(isinstance(value, str) and len(value) <= maximum, "Unbounded report text")
    return value


def _number(value, *, low=-10000, high=10000):
    if value is not None:
        _require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high,
                 "Invalid bounded report number")
    return value


def _count(value, maximum=10_000_000):
    if value is not None:
        _require(type(value) is int and 0 <= value <= maximum, "Invalid report count")
    return value


def _support_label(name, primitive):
    label = _PRIMITIVES.get(primitive, 'Unknown saved primitive')
    endpoint = name[len(primitive) + 1:] if name.startswith(primitive + ':') else None
    suffix = {'-1.0': 'endpoint −1', '1.0': 'endpoint +1', '0.0': 'center sample'}.get(endpoint, 'unmapped sample')
    return f'{label} · {suffix}'


def _target_info(frame, reference):
    descriptor = reference.get('target_origin')
    selected = 'image_refinement' in frame or 'image_refinement_raw' in frame
    path = 'identity_raw.landmarks' if 'identity_raw' in frame else 'landmarks'
    if selected:
        meta, raw = frame.get('image_refinement'), frame.get('image_refinement_raw')
        _require('identity_raw' not in frame and isinstance(meta, dict) and isinstance(raw, dict)
                 and not any(key in raw for key in ('identity_raw', 'image_refinement', 'image_refinement_raw')),
                 'Selected crop targets cannot mix identity repair or omit preserved raw evidence')
        _require(meta.get('method') == _CROP_METHOD and meta.get('anatomical_identity_verified') is False,
                 'Unsupported selected crop provenance')
        _require(all(isinstance(meta.get(key), str) and _SHA.fullmatch(meta[key]) for key in _TARGET_HASHES),
                 'Crop provenance requires exact SHA256 identifiers')
        _require(all(key in frame and key in raw and frame[key] == raw[key]
                     for key in ('id', 'frame', 'time_s', 'decoder_timestamp_s')), 'Crop source identities or clocks differ')
        _require(_canonical(raw) == meta['original_observation_sha256']
                 and _canonical({k: v for k, v in frame.items() if k not in {'image_refinement', 'image_refinement_raw'}})
                 == meta['proposal_sha256'], 'Crop preserved/raw proposal identity differs')
        expected = {'kind': 'selected_crop_estimate_mapped_to_source', 'method': _CROP_METHOD,
                    **{key: meta[key] for key in _TARGET_HASHES}, 'anatomical_identity_verified': False}
        _require(type(descriptor) is dict and descriptor.get('anatomical_identity_verified') is False and descriptor == expected,
                 'Selected crop source requires its exact target_origin descriptor')
        label = 'Selected crop estimate, mapped to source image'
    else:
        _require(descriptor is None, 'Target origin claims a crop absent from its source frame')
        if 'identity_raw' in frame:
            raw = frame['identity_raw']
            _require(isinstance(raw, dict) and not any(key in raw for key in ('image_refinement', 'image_refinement_raw'))
                     and all(raw.get(key) == frame.get(key) for key in ('id', 'frame', 'time_s')),
                     'Preserved whole-frame target identity differs or mixes crop provenance')
        label = ('Inferred whole-frame estimate (preserved before identity processing)' if 'identity_raw' in frame
                 else 'Inferred whole-frame estimate')
    for point in reference['points']:
        recorded = point.get('landmark_source')
        _require(recorded == path if selected else recorded is None or recorded == path,
                 'Endpoint landmark_source must name its actual selected JSON path')
    return {'target_origin': descriptor, 'target_label': label, 'target_landmark_source': path}


def _model(documents, hashes, check):
    report, candidate = documents["report.json"], documents["candidate-human.json"]
    source, evidence = documents["inputs/human.json"], documents["inputs/evidence-derived.json"]
    method = report.get("profile")
    _require(type(report.get("schema_version")) is int and report['schema_version'] == 1 and method in METHODS,
             "Unsupported reference proposal profile")
    _require(report.get("status") in {"rejected_construction_proposal", "unadmitted_construction_proposal", "partial_budget_exhausted"}
             and report.get("promoted") is False and report.get("physical_validation") == "unverified",
             "Only explicitly unpromoted, physically unverified proposals are supported")
    _require(report.get("candidate_sha256") == _canonical(candidate), "Canonical candidate identity differs")
    for key, name in (("input_human_sha256", "inputs/human.json"), ("input_evidence_sha256", "inputs/evidence-derived.json")):
        _require(report.get(key) == hashes[name], "Proposal input byte identity differs")
    marker = candidate.get("whole_body_reference_refinement")
    _require(isinstance(marker, dict) and marker.get("method") == method and marker.get("status") == report["status"]
             and marker.get("promoted") is False and all(marker.get(k) == report[k]
             for k in ("input_human_sha256", "input_evidence_sha256")), "Candidate provenance differs from its report")
    source_id = report.get("source_sha256")
    _require(isinstance(source_id, str) and _SHA.fullmatch(source_id)
             and all(value.get("source_sha256") == source_id for value in (source, candidate, evidence)),
             "Source identifiers disagree across selected evidence")
    _require({k: v for k, v in candidate.items() if k not in {"reference", "whole_body_reference_refinement"}}
             == {k: v for k, v in source.items() if k != "reference"}, "Nonreference body, root, shape, floor or clocks changed")
    contract = report.get("contract")
    _require(isinstance(contract, dict) and contract.get("method") == method
             and contract.get("floor_penetration_limit_model_units") == LIMIT, "The fixed 0.025 floor limit changed")
    tolerance = _number(contract.get("numerical_tolerance"), low=0, high=.000001)
    _require(tolerance is not None, "Missing numerical tolerance")
    gates = report.get("gates")
    _require(isinstance(gates, dict) and len(gates) <= 32 and all(v is None or type(v) is bool for v in gates.values()),
             "Invalid saved gate outcomes")
    for name in gates: _text(name, 100)
    _require(not (report["status"] == "unadmitted_construction_proposal" and any(v is False for v in gates.values())),
             "Proposal status contradicts its rejected gates")
    failures = report.get('failures')
    _require(failures is None or isinstance(failures, list) and len(failures) <= 32 and all(isinstance(name, str) for name in failures)
             and set(failures) == {name for name, value in gates.items() if value is False}, 'Saved failed gate names differ')
    _require(report['status'] != 'rejected_construction_proposal' or any(v is False for v in gates.values()),
             'Rejected proposal is missing its failed gate')
    references, old_refs, new_refs = report.get("references"), source.get("reference"), candidate.get("reference")
    _require(isinstance(references, list) and isinstance(old_refs, list) and isinstance(new_refs, list)
             and 1 <= len(references) == len(old_refs) == len(new_refs) <= 900, "Reference counts differ or exceed bounds")
    times = [r.get("time_s") for r in old_refs]
    _require(all(_number(t, low=0) is not None for t in times) and all(a < b for a, b in zip(times, times[1:])),
             "Reference clocks must increase")
    _require([r.get("time_s") for r in new_refs] == times == [r.get("time_s") for r in references],
             "Source/candidate/report clocks differ")
    origins = source.get("reference_evidence")
    _require(isinstance(origins, list) and len(origins) == len(references), "Missing source reference identities")
    slots, frames = report.get("retained_pose_slots"), evidence.get("frames")
    _require(isinstance(slots, list) and isinstance(frames, list) and 1 <= len(slots) == len(frames) <= 900,
             "Retained source-slot counts differ")
    _require(slots == [{k: row.get(k) for k in ("id", "frame", "time_s", "decoder_timestamp_s", "status")} for row in frames],
             "Retained source slots differ from the exact input evidence")
    _require(len({row.get("id") for row in slots}) == len(slots), "Duplicate retained source identity")
    by_id, target_info = {}, {}
    frames_by_id = {row['id']: row for row in frames}
    for index, ref in enumerate(references):
        check()
        identity = _text(ref.get("evidence_id"))
        _require(identity == origins[index].get("evidence_id") and origins[index].get('time_s') == times[index]
                 and identity not in by_id, "Reference evidence identities differ")
        _text(ref.get('status'), 100)
        points = ref.get("points")
        _require(isinstance(points, list) and len(points) <= 32, "Endpoint details exceed their bound")
        names = set()
        for point in points:
            joint = _text(point.get("joint"), 100); _require(joint not in names, "Duplicate endpoint"); names.add(joint)
            _text(point.get("status"), 100)
            for field in ("source_confidence", "weight"): _number(point.get(field), low=0, high=1)
            for field in ("error_before_px", "error_after_px"): _number(point.get(field), low=0, high=1_000_000)
            if point.get("weight") == 0:
                _require(point.get("error_before_px") is None and point.get("error_after_px") is None,
                         "Missing or low-confidence endpoint errors must stay unknown")
        by_id[identity] = (index, ref)
        _require(identity in frames_by_id, 'A target reference has no source frame')
        target_info[identity] = _target_info(frames_by_id[identity], ref)
    support_names, primitive_names = report.get("support_names"), report.get("support_primitive_names")
    _require(isinstance(support_names, list) and isinstance(primitive_names, list)
             and 1 <= len(support_names) == len(primitive_names) <= 64 and len(set(support_names)) == len(support_names),
             "Invalid bounded support identities")
    for name in support_names + primitive_names: _text(name, 100)
    checks = report.get("geometry_checks")
    _require(isinstance(checks, list) and len(checks) <= 2701, "Geometry sample count exceeds its bound")
    source_checks, grid, unique = {}, [], {}
    for row in checks:
        check()
        kind, stamp = row.get("kind"), _number(row.get("time_s"), low=0)
        _require(kind in {"source_pose", "control_grid"} and stamp is not None, "Invalid saved sample clock or kind")
        for side in ("before", "after"):
            gaps = row.get(side + "_floor_support_gaps")
            _require(gaps is None or isinstance(gaps, list) and len(gaps) == len(support_names), "Support gap count differs")
            if gaps is not None:
                for value in gaps: _number(value)
            peak = _number(row.get(side + "_max_penetration_model_units"), low=0)
            if gaps is not None and all(v is not None for v in gaps) and peak is not None:
                _require(math.isclose(peak, max(0., -min(gaps)), abs_tol=1e-10), "Reported penetration differs from saved gaps")
        _number(row.get("maximum_depth_change_model_units"), low=0)
        _require(row.get("no_worse_floor") is None or type(row["no_worse_floor"]) is bool, "Invalid floor comparison flag")
        pair = [row.get(side + "_floor_support_gaps") for side in ("before", "after")]
        if stamp in unique:
            _require(unique[stamp] == pair, "Coincident source/grid support records disagree")
        unique[stamp] = pair
        if kind == "source_pose":
            index = row.get("source_reference_index")
            _require(type(index) is int and 0 <= index < len(references) and index not in source_checks
                     and stamp == times[index], "Source geometry sample identity differs")
            source_checks[index] = row
        else:
            _require(row.get("source_reference_index") is None, "Grid samples cannot claim a source-reference index")
            _require(times[0] <= stamp <= times[-1], 'Control-grid clock is outside the source reference span')
            grid.append(row)
    _require(len(grid) <= 1801 and all(a['time_s'] < b['time_s'] for a, b in zip(grid, grid[1:])), "Grid clocks must increase")

    def sample(slot, geometry, ref=None):
        return {"time_s": slot.get("time_s"), "decoder_timestamp_s": slot.get("decoder_timestamp_s"),
                "frame": slot.get("frame"), "evidence_id": slot.get("id"), "source_status": slot.get("status"),
                "geometry": geometry, "points": ref.get("points") if ref else None,
                "reference_status": ref.get("status") if ref else "No body reference for this retained source slot",
                **(target_info[ref['evidence_id']] if ref else {'target_origin': None, 'target_landmark_source': None,
                    'target_label': 'Unavailable — no saved source endpoint targets for this sample'})}

    source_rows = []
    for slot in slots:
        _text(slot.get("id")); _text(slot.get("status"), 100)
        _count(slot.get("frame")); _number(slot.get("time_s"), low=0); _number(slot.get("decoder_timestamp_s"), low=0)
        found = by_id.get(slot["id"])
        if found:
            index, ref = found
            _require(slot['frame'] == ref.get('frame') and slot['time_s'] == ref['time_s'], "Retained slot/reference identity differs")
            source_rows.append(sample(slot, source_checks.get(index), ref))
        else: source_rows.append(sample(slot, None))
    slot_times = [row['time_s'] for row in slots]
    _require(all(t is not None for t in slot_times) and all(a < b for a, b in zip(slot_times, slot_times[1:])),
             'Retained nominal clocks must increase; missing decoder clocks remain unknown')
    _require(set(by_id) <= {s['id'] for s in slots}, "A body reference has no retained source slot")
    grid_rows = []
    for row in grid:
        index = bisect_left(times, row['time_s'])
        neighbors = ([index] if index < len(times) and times[index] == row['time_s'] else
                     sorted({max(0, index - 1), min(len(times) - 1, index)}))
        value = sample({"time_s": row['time_s'], "status": "Authored control-grid sample"}, row)
        value['bracketing_references'] = [{"evidence_id": references[i]['evidence_id'], "time_s": times[i]} for i in neighbors]
        value['reference_status'] = "No source-image endpoint target is authored for control-grid samples"
        grid_rows.append(value)
    counts = report.get("source_counts", {})
    _require(isinstance(counts, dict), "Invalid source counts")
    for key, actual in (("retained_pose_slots", len(slots)), ("body_reference_knots", len(references))):
        _require(counts.get(key) == actual, "Saved source counts differ")
    for key in ("declared_decoded_frames", "declared_frame_count"): _count(counts.get(key))
    _require(counts.get('declared_decoded_frames') == evidence.get('decoded_frames')
             and counts.get('declared_frame_count') == evidence.get('frame_count'), "Declared source frame counts differ")
    floor = report.get('floor_summary', {})
    _require(isinstance(floor, dict), "Invalid floor summary")
    for key, actual in (('checks', len(checks)), ('source_pose_checks', len(source_checks)), ('control_grid_checks', len(grid))):
        _require(floor.get(key) is None or _count(floor[key]) == actual, "Saved geometry counts differ")
    displayed_counts = {}
    for kind, rows in (('source_pose', list(source_checks.values())), ('control_grid', grid)):
        value = {'checks': len(rows)}
        for side in ('before', 'after'):
            values = [row.get(side + '_max_penetration_model_units') for row in rows]
            value[side + '_failures'] = sum(v is not None and v > LIMIT for v in values)
            value[side + '_unknown'] = sum(v is None for v in values)
        displayed_counts[kind] = value
    for side in ('before', 'after'):
        values = [row.get(side + '_max_penetration_model_units') for row in checks]
        reported = _count(floor.get(side + '_failures'))
        maximum = _number(floor.get(side + '_maximum_penetration_model_units'), low=0)
        if values and all(v is not None for v in values) and reported is not None:
            _require(reported == sum(v > LIMIT for v in values), "Saved floor failure total differs")
        if values and all(v is not None for v in values) and maximum is not None:
            _require(math.isclose(maximum, max(values), abs_tol=1e-10), 'Saved maximum penetration differs')
    _number(floor.get('maximum_penetration_reduction_model_units'))
    _count(floor.get('samples_with_reduced_penetration'))
    after = [row.get('after_max_penetration_model_units') for row in checks]
    gate = gates.get('floor_feasibility')
    if gate is True:
        _require(not any(value is not None and value > LIMIT for value in after), 'Saved floor gate contradicts known penetration')
    if (gate is False and after and all(value is not None for value in after) and len(source_checks) == len(references)
            and grid and floor.get('checks') == len(checks) and floor.get('control_grid_checks') == len(grid)):
        _require(any(value > LIMIT for value in after), 'Saved floor gate contradicts complete saved penetration values')
    objective = report.get('objective')
    if method.endswith('floor-first-reference-v1'):
        _require(isinstance(objective, dict) and objective.get('profile') == 'floor_first_v1'
                 and objective.get('units') == 'model_units_squared' and objective.get('proposal_target_gap_model_units') == 0.,
                 "Missing supported floor-first objective")
        _require(_count(objective.get('unique_samples')) == len(unique), "Floor-first unique time count differs")
        for side, name in enumerate(('primary_before', 'primary_after')):
            value = _number(objective.get(name), low=0, high=1_000_000)
            if all(pair[side] is not None and all(v is not None for v in pair[side]) for pair in unique.values()):
                expected = .5 * sum(min(0., gap + LIMIT)**2 for pair in unique.values() for gap in pair[side])
                _require(value is not None and math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-12),
                         "Saved squared excess objective differs from unique-time gaps")
        counter = objective.get('counters')
        _require(isinstance(counter, dict) and set(counter) == _COUNTERS, "Missing supported floor-first counters")
        for key, value in counter.items():
            _text(key, 100)
            if key == 'qp_status_counts':
                _require(isinstance(value, dict) and set(value) == _QP_STATUSES, "Invalid QP status counts")
                for name, count in value.items(): _text(name, 100); _require(_count(count) is not None, "Missing QP count")
            else: _require(_count(value) is not None, "Missing objective counter")
        _require(counter.get('floor_qp_calls', -1) + counter.get('image_qp_calls', -1) == report.get('qp_calls')
                 == sum(counter.get('qp_status_counts', {}).values()), "QP counters differ")
        _require(counter.get('floor_descent_accepts', -1) + counter.get('image_tiebreak_accepts', -1) == report.get('accepted_updates')
                 and counter.get('backtrack_trials') == report.get('accepted_updates', -1) + counter.get('rejected_backtracks', -1),
                 "Completed-trial counters differ")
    else: _require(objective is None, "Image-first profile cannot silently claim a floor-first objective")
    image = report.get('image_summary', {})
    _require(isinstance(image, dict) and image.get('grid_image_targets') in (None, 0), "Grid image targets must remain unavailable")
    for name in ('mean_error_before_px', 'mean_error_after_px'): _number(image.get(name), low=0, high=1_000_000)
    _count(image.get('confident_source_endpoint_pairs'))
    points = [p for ref in references for p in ref['points']]
    if all(p.get('weight') is not None for p in points):
        pairs = [p for p in points if p['weight'] > 0]
        _require(image.get('confident_source_endpoint_pairs') in (None, len(pairs)), 'Confident endpoint counts differ')
        for side in ('before', 'after'):
            errors = [p.get(f'error_{side}_px') for p in pairs]
            average = image.get(f'mean_error_{side}_px')
            if errors and all(e is not None for e in errors) and average is not None:
                _require(math.isclose(average, sum(errors) / len(errors), rel_tol=1e-9, abs_tol=1e-10), 'Saved mean endpoint error differs')
    return {'method': method, 'status': report['status'], 'promoted': False, 'physical_validation': 'unverified',
            'geometry_admission': report.get('geometry_admission'), 'gates': gates, 'failures': report.get('failures'),
            'source_sha256': source_id, 'candidate_canonical_sha256': report['candidate_sha256'], 'floor_limit': LIMIT,
            'source_counts': counts, 'source_status_counts': dict(Counter(slot['status'] for slot in slots)),
            'saved_floor_summary': floor, 'sample_counts': displayed_counts, 'unique_sample_times': len(unique),
            'source_slots_without_body_reference': len(slots) - len(references), 'image_summary': image, 'objective': objective,
            'source_references_without_geometry': len(references) - len(source_checks),
            'support_names': support_names, 'support_primitive_names': primitive_names,
            'support_labels': [_support_label(name, primitive) for name, primitive in zip(support_names, primitive_names)],
            'source': source_rows, 'grid': grid_rows,
            'scope': 'Saved source image estimates and authored-floor sample checks, not measured truth, physical poses, contact, anatomy or task/policy proof.'}


_SCRIPT = r"""'use strict';
const data=JSON.parse(document.getElementById('data').textContent),mode=document.getElementById('mode'),select=document.getElementById('sample');
const el=id=>document.getElementById(id),num=v=>v===null||v===undefined?'Unknown':Number(v).toPrecision(6),ns='http://www.w3.org/2000/svg';
let rows=[];
function cell(tr,value,head=false){const node=document.createElement(head?'th':'td');if(head)node.scope='row';node.textContent=value;tr.append(node);}
function textRow(table,values){const tr=document.createElement('tr');values.forEach((v,i)=>cell(tr,v,i===0));table.append(tr);return tr;}
function chart(){const svg=el('chart'),layer=el('lines');layer.replaceChildren();const values=rows.flatMap(r=>['before','after'].map(k=>r.geometry?.[k+'_max_penetration_model_units'])).filter(v=>v!==null&&v!==undefined);
 const ymax=Math.max(.03,...values)*1.1,first=rows[0]?.time_s??0,last=rows.at(-1)?.time_s??first;
 const x=t=>50+850*(t-first)/Math.max(.001,last-first),y=v=>250-210*v/ymax;
 svg.dataset.ymax=ymax;svg.dataset.first=first;svg.dataset.last=last;
 function line(d,klass){const p=document.createElementNS(ns,'path');p.setAttribute('d',d);p.setAttribute('class',klass);layer.append(p);}
 line(`M50 ${y(data.floor_limit)}H900`,'limit');
 for(const side of ['before','after']){let path='',open=false;for(const r of rows){const v=r.geometry?.[side+'_max_penetration_model_units'];if(v===null||v===undefined||r.time_s===null){open=false;continue;}path+=(open?' L':' M')+x(r.time_s)+' '+y(v);open=true;}line(path,side);}
 el('ymax').textContent=num(ymax);el('tfirst').textContent=num(first)+' s';el('tlast').textContent=num(last)+' s';
}
function detail(){const row=rows[Number(select.value)],geometry=row?.geometry;el('gaps').replaceChildren();el('points').replaceChildren();el('target-metadata').replaceChildren();el('target-details').hidden=true;el('target-label').textContent='Unavailable — no saved source endpoint targets for this sample';
 if(!row){el('identity').textContent='No saved samples available.';return;}
 el('identity').textContent=`${mode.value==='source'?'Retained source slot':'Control-grid check'} ${Number(select.value)+1}/${rows.length} · nominal time ${num(row.time_s)} s · decoder time ${num(row.decoder_timestamp_s)} s · frame ${row.frame??'unavailable'}`;
 el('source-id').textContent=row.evidence_id||'No direct source observation ID for this interpolated control sample';
 el('context').textContent=[row.source_status,row.reference_status,...(row.bracketing_references||[]).map(v=>`Body reference ${v.evidence_id} at ${num(v.time_s)} s`)].join(' · ');
 el('penetration').textContent=`Input body: ${num(geometry?.before_max_penetration_model_units)}; Proposed body: ${num(geometry?.after_max_penetration_model_units)} model units. Fixed limit: 0.025. Unknown is not zero.`;
 el('target-label').textContent=row.target_label;
 if(row.target_origin){el('target-details').hidden=false;for(const [label,value]of[['Method',row.target_origin.method],['Preserved whole-frame observation SHA256',row.target_origin.original_observation_sha256],['Selected crop proposal SHA256',row.target_origin.proposal_sha256],['Saved decision SHA256',row.target_origin.decision_sha256],['Landmark JSON path',row.target_landmark_source],['Anatomical identity','Not independently verified']]){const term=document.createElement('dt'),detail=document.createElement('dd');term.textContent=label;detail.textContent=value;el('target-metadata').append(term,detail);}}
 let shown=0;for(let i=0;i<data.support_names.length;i++){const a=geometry?.before_floor_support_gaps?.[i],b=geometry?.after_floor_support_gaps?.[i],known=a!==null&&a!==undefined&&b!==null&&b!==undefined;
 if(el('affected').checked&&known&&a===b&&a>=0&&b>=0)continue;
 const tr=textRow(el('gaps'),[data.support_labels[i],'',num(a),num(b),!known?'Unknown':b>a?'Gap increased':b<a?'Gap decreased':'Unchanged']);
 const details=document.createElement('details'),label=document.createElement('summary'),ids=document.createElement('span');label.textContent='Exact saved IDs';ids.textContent=data.support_names[i]+' · '+data.support_primitive_names[i];details.append(label,ids);tr.children[1].append(details);shown++;}
 if(!shown)textRow(el('gaps'),['No changed or below-floor support gaps in this sample.','','','','']);
 if(mode.value==='grid'||!row.points)textRow(el('points'),['Unavailable — no source endpoint targets for this sample.','','','','']);
 else for(const p of row.points)textRow(el('points'),[p.joint,p.status,num(p.source_confidence),num(p.error_before_px),num(p.error_after_px)]);
 const svg=el('chart'),ymax=Number(svg.dataset.ymax),first=Number(svg.dataset.first),last=Number(svg.dataset.last);for(const side of ['before','after']){const dot=el(side+'-dot'),v=geometry?.[side+'_max_penetration_model_units'];dot.style.display=v===null||v===undefined?'none':'';if(v!==null&&v!==undefined){dot.setAttribute('cx',50+850*(row.time_s-first)/Math.max(.001,last-first));dot.setAttribute('cy',250-210*v/ymax);}}
 el('previous').disabled=Number(select.value)<=0;el('next').disabled=Number(select.value)>=rows.length-1;
}
function changeMode(){rows=data[mode.value];select.replaceChildren();rows.forEach((row,i)=>{const option=document.createElement('option');option.value=i;option.textContent=`${i+1} · ${num(row.time_s)} s${row.frame===null||row.frame===undefined?'':' · frame '+row.frame}`;select.append(option);});select.value='0';chart();detail();}
mode.addEventListener('change',changeMode);select.addEventListener('change',detail);el('affected').addEventListener('change',detail);
for(const [id,d]of[['previous',-1],['next',1]])el(id).addEventListener('click',()=>{select.value=Math.max(0,Math.min(rows.length-1,Number(select.value)+d));detail();});changeMode();
"""


def _render(summary, palette):
    esc = lambda value: html.escape(str(value), quote=True)
    fmt = lambda value: "Unknown" if value is None else esc(value)
    labels = {'rejected_construction_proposal': 'Rejected proposal — not promoted',
              'unadmitted_construction_proposal': 'Unadmitted proposal — not promoted',
              'partial_budget_exhausted': 'Incomplete proposal — not promoted'}
    counts = ''.join(f'<tr><th scope="row">{label}</th><td>{row["checks"]}</td><td>{row["before_failures"]}</td><td>{row["after_failures"]}</td><td>{row["before_unknown"]} / {row["after_unknown"]}</td></tr>'
                     for key, label in [('source_pose', 'Source-reference checks'), ('control_grid', 'Control-grid checks')]
                     for row in [summary['sample_counts'][key]])
    gates = ''.join(f'<tr><th scope="row">{esc(name)}</th><td>{"Unknown" if value is None else "Reported satisfied" if value else "Reported failure"}</td></tr>'
                    for name, value in summary['gates'].items())
    objective = summary['objective']
    objective_html = ('<p>Image-first proposal. No floor-first objective was reported.</p>' if objective is None else
        f'<p>Saved squared <strong>excess above 0.025</strong>: {fmt(objective.get("primary_before"))} → {fmt(objective.get("primary_after"))} model units² over {fmt(objective.get("unique_samples"))} unique sampled times. Lower is not a pass; zero would mean no saved excess above the limit, not zero penetration or contact.</p><details><summary>Saved floor-first counters</summary><pre>{esc(json.dumps(objective.get("counters"),indent=2))}</pre><p>Backtrack counts cover completed trials only. Interrupted trials may still add geometry calls. No per-update ledger was saved; these aggregates do not establish each optimization step.</p></details>')
    data = json.dumps(summary, separators=(',', ':'), allow_nan=False).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    script_hash = base64.b64encode(sha256(_SCRIPT.encode()).digest()).decode()
    links = ''.join(f'<li><a href="artifacts/{name}">{esc(source)}</a></li>' for source, name in MEMBERS.items())
    css = """body{margin:0;background:var(--weft-canvas);color:var(--weft-ink);font:16px/1.5 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:30px 20px}h1{font:700 2.4rem/1.1 Georgia,serif}section{background:var(--weft-surface);border:1px solid var(--weft-border);border-radius:14px;padding:22px;margin:20px 0}a{color:var(--weft-primary)}.notice{border-left:4px solid var(--weft-primary);padding:10px 16px}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:8px;border-bottom:1px solid var(--weft-border)}.scroll{overflow-x:auto}code,pre,.identity{overflow-wrap:anywhere}pre{white-space:pre-wrap}label{display:inline-block;margin:8px 14px 8px 0}select,button{font:inherit;padding:7px;max-width:100%}button:focus-visible,select:focus-visible{outline:3px solid var(--weft-primary)}svg{width:100%;height:auto}.before{fill:none;stroke:var(--weft-ink);stroke-width:2}.after{fill:none;stroke:var(--weft-primary);stroke-width:2.5}.limit{fill:none;stroke:var(--weft-muted);stroke-dasharray:7 5;stroke-width:1.5}svg text{fill:var(--weft-ink);font-size:13px}.dot-before{fill:var(--weft-ink)}.dot-after{fill:var(--weft-primary)}small{color:var(--weft-muted)}@media(max-width:600px){main{padding:16px 10px}section{padding:14px}td,th{padding:6px;font-size:.85rem}}"""
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'sha256-{script_hash}'; base-uri 'none'; form-action 'none'; object-src 'none'">
<title>Weft · Reference proposal</title><style>{palette}{css}</style></head><body><main>
<header><p>weft · saved reference proposal</p><h1>{labels[summary['status']]}</h1><p class="notice">The floor penetration limit remains <strong>0.025 model units</strong>. Fewer failures or smaller image error do not make a proposal pass. Source image estimates are <strong>not measured truth</strong>. No geometry, physics, policy, model or media processing runs in this report.</p><p>Method: <code>{esc(summary['method'])}</code></p></header>
<section><h2>What changed, and what did not</h2><p>Source slots: {fmt(summary['source_counts'].get('retained_pose_slots'))}; body references: {fmt(summary['source_counts'].get('body_reference_knots'))}; retained slots without a body reference: {summary['source_slots_without_body_reference']}. Declared decoded frames: {fmt(summary['source_counts'].get('declared_decoded_frames'))}; container frame count: {fmt(summary['source_counts'].get('declared_frame_count'))}.</p>
<div class="scroll"><table><thead><tr><th>Sampling view</th><th>Checks</th><th>Input body failures</th><th>Proposed body failures</th><th>Unknown input / proposed</th></tr></thead><tbody>{counts}</tbody></table></div><p>Counts cover saved checks only. Body references without saved geometry: {summary['source_references_without_geometry']}. The grid schedule is not re-created here; missing checks do not imply success. Source and grid checks overlap in time: {summary['unique_sample_times']} unique nominal sampled times. They are <strong>not distinct physical poses</strong> and not a continuous-time collision check. Root, shape, floor, clocks and nonreference fields match the supplied input body (after any pose recovery), not a body from before crop selection.</p>
<p>Input body → Proposed body. Reported worst floor penetration: {fmt(summary['saved_floor_summary'].get('before_maximum_penetration_model_units'))} → {fmt(summary['saved_floor_summary'].get('after_maximum_penetration_model_units'))} model units. Reported confident endpoint error: {fmt(summary['image_summary'].get('mean_error_before_px'))} → {fmt(summary['image_summary'].get('mean_error_after_px'))} pixels; reported pairs: {fmt(summary['image_summary'].get('confident_source_endpoint_pairs'))}. Both bodies are compared against the <strong>same selected source-image targets</strong>. These are body-fitting errors, not before/after crop-detector accuracy. Missing targets remain unknown.</p>{objective_html}<details><summary>Saved gate outcomes — not a new evaluation</summary><table><tbody>{gates}</tbody></table></details></section>
<section><h2>Floor penetration by saved sample</h2><label>Sampling view <select id="mode"><option value="source">Retained source slots</option><option value="grid">Control grid</option></select></label>
<p>Input body: teal line. Proposed body: clay line. Dashed line: unchanged 0.025 limit. Missing values break the line; they are not plotted as zero.</p>
<svg id="chart" viewBox="0 0 950 290" role="img" aria-labelledby="chart-title chart-description"><title id="chart-title">Input body and proposed body maximum floor penetration</title><desc id="chart-description">Model-unit penetration by nominal time. Use the sample selector for exact values and unknowns.</desc><path d="M50 35V250H900" fill="none" stroke="currentColor"/><text x="5" y="20">model units</text><text id="ymax" x="5" y="43"></text><text x="28" y="254">0</text><text id="tfirst" x="50" y="278"></text><text id="tlast" x="800" y="278"></text><g id="lines"></g><circle id="before-dot" r="4" class="dot-before"/><circle id="after-dot" r="4" class="dot-after"/></svg>
<label>Sample <select id="sample"></select></label><button id="previous" type="button">Previous</button> <button id="next" type="button">Next</button>
<p id="identity" aria-live="polite"></p><p id="source-id" class="identity"></p><p id="context" class="identity"></p><p id="penetration"></p>
<h3>Signed support gaps</h3><p>Negative means below the authored floor; positive means above it. A larger signed gap alone does not establish support or contact. Capsule endpoints −1 and +1 and the center sample 0 are points on saved shapes, not extra body parts. Labels describe authored collision shapes, not recovered anatomy.</p><label><input id="affected" type="checkbox" checked> Show changed, below-floor or unknown gaps</label>
<div class="scroll"><table><thead><tr><th>Authored shape / support point</th><th>Saved identity</th><th>Input body gap</th><th>Proposed body gap</th><th>Change</th></tr></thead><tbody id="gaps"></tbody></table></div>
<h3>Source endpoint agreement</h3><p>Only confident source-knot estimates supply targets. Control-grid samples have no source-image targets, including times that coincide with a source slot.</p><p id="target-label"></p><details id="target-details" hidden><summary>Selected target provenance</summary><dl id="target-metadata" class="identity"></dl><p>Preserved observation and selected proposal hashes match the copied input. The decision hash is a saved identifier; decision bytes and image pixels are not read or independently verified here. Original whole-frame estimates remain provenance, not alternative fitting targets.</p></details><div class="scroll"><table><thead><tr><th>Endpoint</th><th>Saved status</th><th>Source confidence</th><th>Input body error (px)</th><th>Proposed body error (px)</th></tr></thead><tbody id="points"></tbody></table></div><noscript>Enable local JavaScript for sample selection, or inspect the exact supporting JSON below. No backend is used.</noscript></section>
<section><h2>Exact copied evidence</h2><p>Only four selected members were hash-checked and copied. Other manifest members were not read. External manifest hash supplied: {str(summary['external_manifest_pin_supplied']).lower()}. Without one, the local manifest identity is frozen for consistency, not externally authenticated. The stated video source ID is not a new hash of video bytes.</p><ul>{links}<li><a href="source-manifest.json.txt">Original manifest</a></li><li><a href="report.json">Report data and provenance</a></li></ul><p class="identity">Source manifest SHA256: <code>{summary['source_manifest_sha256']}</code></p><p>{esc(summary['scope'])}</p><small>Generated {esc(summary['created_at'])} (America/Los_Angeles).</small></section>
</main><script id="data" type="application/json">{data}</script><script>{_SCRIPT}</script></body></html>'''.encode()


def prepare_whole_body_reference_report(source, out, *, expected_manifest_sha256=None, max_seconds=10,
                                        max_input_bytes=MAX_BYTES, max_output_bytes=MAX_BYTES):
    """Copy selected sealed JSON into a new bounded, offline inspection report."""
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 0 < max_seconds <= 30, "Report time bound must be 0–30 seconds")
    for value in (max_input_bytes, max_output_bytes):
        _require(type(value) is int and 1024 <= value <= MAX_BYTES, "Report byte bounds must be 1 KiB–16 MiB")
    if expected_manifest_sha256 is not None:
        _require(isinstance(expected_manifest_sha256, str) and _SHA.fullmatch(expected_manifest_sha256), "Expected manifest must be an exact SHA256")
    source, out = Path(source).absolute(), Path(out).absolute()
    _require('..' not in source.parts + out.parts and not source.is_symlink() and not out.is_symlink(), "Report paths must not traverse parents or links")
    source, out = source.resolve(strict=True), out.resolve(strict=False)
    _require(len(source.parts) <= 64 and len(out.parts) <= 64, 'Report paths exceed the depth bound')
    _require(not out.exists() and out != source and not out.is_relative_to(source) and not source.is_relative_to(out),
             "Report output must be new and outside its source")
    started = time.monotonic(); deadline = started + max_seconds

    def check():
        if time.monotonic() > deadline:
            raise TimeoutError("Offline reference report exceeded its cooperative time bound")

    check()
    manifest_raw = read_package_file(source, 'manifest.json', max_bytes=min(128 * 1024, max_input_bytes), expected_sha256=expected_manifest_sha256)
    manifest_sha = sha256(manifest_raw).hexdigest(); manifest = _json(manifest_raw); files = manifest.get('files')
    _require(type(manifest.get('schema_version')) is int and manifest['schema_version'] == 1
             and manifest.get('profile') == 'whole-body-reference-candidate-study-v1'
             and manifest.get('status') == 'completed' and isinstance(files, dict) and 1 <= len(files) <= 1024,
             "Expected a bounded completed study seal")
    for name, identity in files.items():
        _require(isinstance(name, str) and 0 < len(name) <= 512 and '\\' not in name and ':' not in name and all(ord(c) >= 32 for c in name)
                 and not Path(name).is_absolute() and all(part not in ('', '.', '..') for part in name.split('/'))
                 and name != 'manifest.json' and isinstance(identity, str) and _SHA.fullmatch(identity), "Unsafe manifest member or identity")
    _require(set(MEMBERS) <= set(files), "Study lacks required sealed report inputs")
    _require(not any(parent.as_posix() in files for name in files for parent in Path(name).parents), 'Manifest file/directory paths overlap')
    raw, total = {}, len(manifest_raw)
    for name in MEMBERS:
        check(); remaining = max_input_bytes - total
        _require(remaining > 0, "Selected report input byte bound exhausted")
        raw[name] = read_package_file(source, name, max_bytes=remaining, expected_sha256=files[name]); total += len(raw[name])
    model = _model({name: _json(value) for name, value in raw.items()}, {name: files[name] for name in MEMBERS}, check)
    check()
    palette_path = Path(__file__).parent / 'assets/brand.css'
    with palette_path.open('rb') as stream: palette_raw = stream.read(16385)
    _require(len(palette_raw) <= 16384 and b'</' not in palette_raw, "Invalid report palette")
    implementation = read_package_file(Path(__file__).parent, Path(__file__).name, max_bytes=256 * 1024)
    model.update(schema_version=1, profile=PROFILE, created_at=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M:%S %Z'),
        source_manifest_sha256=manifest_sha, external_manifest_pin_supplied=expected_manifest_sha256 is not None,
        selected_members={name: {'sha256': files[name], 'bytes': len(value)} for name, value in raw.items()},
        selected_input_bytes=total, source_read_byte_bound=2 * max_input_bytes,
        local_asset_read_byte_bound=16384 + 256 * 1024,
        implementation_sha256=sha256(implementation).hexdigest(), palette_sha256=sha256(palette_raw).hexdigest())
    page = _render(model, palette_raw.decode())
    _require(len(page) <= MAX_HTML, "Offline report HTML exceeds its 2 MiB bound")
    output = {'index.html': page, 'report.json': _bytes(model), 'source-manifest.json.txt': manifest_raw,
              **{'artifacts/' + MEMBERS[name]: value for name, value in raw.items()}}
    _require(sum(map(len, output.values())) <= max_output_bytes - 65536, "Offline report output exceeds its byte bound")
    check(); out.mkdir(parents=True, exist_ok=False)
    for name, value in output.items():
        check(); write_package_files(out, {name: value})
    for name in MEMBERS:
        check(); read_package_file(source, name, max_bytes=len(raw[name]), expected_sha256=files[name])
    check(); read_package_file(source, 'manifest.json', max_bytes=128 * 1024, expected_sha256=manifest_sha)
    for directory, expected in ((out, {'index.html', 'report.json', 'source-manifest.json.txt', 'artifacts'}),
                                (out / 'artifacts', set(MEMBERS.values()))):
        with os.scandir(directory) as entries:
            actual = set()
            for entry in entries:
                check()
                _require(entry.name in expected and not entry.is_symlink(), 'Report output inventory changed')
                actual.add(entry.name)
        _require(actual == expected, 'Report output inventory changed')
    seal = _bytes({'schema_version': 1, 'profile': PROFILE, 'status': 'report_generated', 'source_manifest_sha256': manifest_sha,
                   'proposal_status': model['status'], 'promoted': False, 'files': {name: sha256(value).hexdigest() for name, value in output.items()}})
    _require(sum(map(len, output.values())) + len(seal) <= max_output_bytes, "Report seal exceeds its output byte bound")
    check(); write_package_files(out, {'manifest.json': seal})
    return {'directory': str(out), 'manifest_sha256': sha256(seal).hexdigest(), 'source_manifest_sha256': manifest_sha,
            'status': 'report_generated', 'proposal_status': model['status'], 'promoted': False, 'files': len(output) + 1,
            'bytes': sum(map(len, output.values())) + len(seal), 'seconds': time.monotonic() - started}
