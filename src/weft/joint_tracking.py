"""Experimental joint ROI association; a temporal preference is not a measurement."""

from copy import deepcopy
import hashlib
import math
from types import MappingProxyType

import cv2
import numpy as np

from weft.appearance_consistency import color_signature, compare_color


# Frozen analytic controls; real-source grades must not tune this contract.
JOINT_CONTRACT = MappingProxyType({
    "method": "experimental-joint-immutable-roi-v2",
    "confusable_search": "full_configured_radius",
    "displacement_memory": "bounded_maximum_observed_hop_no_decay",
    "maximum_objects": 8,
    "maximum_peaks": 6,
    "maximum_assignment_nodes": 20000,
    "maximum_analysis_dimension": 960,
    "minimum_visual_score": .7,
    "local_radius_object_sizes": 1.5,
    "maximum_temporal_bonus": .12,
    "contact_object_sizes": 2.,
    "source_similarity": .9,
    "conflict_overlap": .5,
})


def _overlap(a, b):
    width = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    height = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return width * height / min(a[2] * a[3], b[2] * b[3])


def _distance(a, b):
    return math.hypot(a[0] + a[2] / 2 - b[0] - b[2] / 2,
                      a[1] + a[3] / 2 - b[1] - b[3] / 2)


class JointAppearance:
    """Locate without mutation, then commit exactly once; templates never adapt."""

    def __init__(self, gray, boxes, settings, *, color=None, shapes=None):
        self._validate_frame(gray, color)
        if not 1 <= len(boxes) <= JOINT_CONTRACT["maximum_objects"]:
            raise ValueError("Joint tracking requires 1–8 initialized objects")
        if shapes is not None and (len(shapes) != len(boxes) or any(
                shape not in {"circle", "box", "unknown"} for shape in shapes)):
            raise ValueError("Shapes must match the initialized object count")
        self._settings = dict(settings)
        for key, default, lower, upper in (("search_radius_px", 60, 8, 200),
                                           ("max_gap_frames", 8, 0, 30)):
            value = self._settings.setdefault(key, default)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"Invalid {key}")
        for key, default, lower, upper in (("minimum_score", .7, .5, .99),
                                           ("minimum_margin", .06, .02, .5)):
            value = self._settings.setdefault(key, default)
            if type(value) not in (float, int) or not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"Invalid {key}")
        self._minimum_score = max(JOINT_CONTRACT["minimum_visual_score"], self._settings["minimum_score"])
        self._shape = gray.shape
        self._tracks = []
        signatures = []
        for box in boxes:
            if (len(box) != 4 or any(type(v) not in (int, np.int64, np.int32) for v in box)
                    or min(box[2:]) < 6 or min(box[:2]) < 0
                    or box[0] + box[2] > gray.shape[1] or box[1] + box[3] > gray.shape[0]):
                raise ValueError("Initialized region exceeds image or analysis resolution")
            x, y, w, h = map(int, box)
            # A bytes owner prevents callers from re-enabling ndarray writes.
            pixels = gray[y:y+h, x:x+w].tobytes()
            template = np.frombuffer(pixels, dtype=np.uint8).reshape(h, w)
            if template.std() < 8:
                raise ValueError("Initialized object lacks contrast/texture")
            source_color = color_signature(color[y:y+h, x:x+w]) if color is not None else None
            self._tracks.append({"box": (x, y, w, h), "template": template,
                                 "color": source_color, "gap": 0, "lost": False,
                                 "velocity": np.zeros(2), "observed_hop_max_px": 0., "observations": 1})
            color_bytes = color[y:y+h, x:x+w].tobytes() if color is not None else b""
            signatures.append(hashlib.sha256(str((w, h)).encode() + pixels + color_bytes).hexdigest())
        if any(_overlap(a, b) > JOINT_CONTRACT["conflict_overlap"]
               for i, a in enumerate(boxes) for b in boxes[i+1:]):
            raise ValueError("Initialized objects overlap ambiguously")
        self._source_signatures = tuple(signatures)
        self._confusable = set()
        for i, first in enumerate(self._tracks):
            for j in range(i + 1, len(self._tracks)):
                second = self._tracks[j]
                resized = cv2.resize(second["template"], first["template"].shape[::-1])
                similarity = float(cv2.matchTemplate(resized, first["template"], cv2.TM_CCOEFF_NORMED)[0, 0])
                distinct_color = (first["color"] is not None and second["color"] is not None and
                                  (compare_color(first["color"], second["color"])["passes"] is False or
                                   compare_color(second["color"], first["color"])["passes"] is False))
                if similarity >= JOINT_CONTRACT["source_similarity"] and not distinct_color:
                    self._confusable.add((i, j))
        for i, track in enumerate(self._tracks):
            track["confusable_source"] = any(i in pair for pair in self._confusable)
        self._pending = None
        self._frame_number = 0

    @property
    def source_signatures(self):
        return self._source_signatures

    @staticmethod
    def _validate_frame(gray, color):
        if (not isinstance(gray, np.ndarray) or gray.dtype != np.uint8 or gray.ndim != 2
                or not gray.size or max(gray.shape) > JOINT_CONTRACT["maximum_analysis_dimension"]):
            raise ValueError("Joint tracking requires bounded uint8 analysis frames")
        if color is not None and (color.dtype != np.uint8 or color.shape != (*gray.shape, 3)):
            raise ValueError("Color evidence must be matching uint8 BGR")

    def _proposals(self, gray, color, track):
        if track["lost"]:
            return [], {"proposal_status": "lost", "score": None, "peak_margin": None, "search_pixels": 0,
                        "search_rect_xywh": None, "best_visual_hypothesis": None}
        x, y, w, h = track["box"]
        radius = self._settings["search_radius_px"] if track["confusable_source"] else min(
            self._settings["search_radius_px"], max(8, math.ceil(
                JOINT_CONTRACT["local_radius_object_sizes"] * max(w, h)
                + track["observed_hop_max_px"] * min(track["gap"] + 1, 3))))
        # Search includes the last observed location, including abrupt reversals.
        left, top = max(0, x - radius), max(0, y - radius)
        right, bottom = min(gray.shape[1], x + w + radius), min(gray.shape[0], y + h + radius)
        scores = cv2.matchTemplate(gray[top:bottom, left:right], track["template"], cv2.TM_CCOEFF_NORMED)
        np.nan_to_num(scores, copy=False, nan=-1, posinf=-1, neginf=-1)
        peaks, best, second, color_rejected = [], None, None, False
        hypotheses = []
        best_hypothesis = None
        truncated = False
        for index in range(JOINT_CONTRACT["maximum_peaks"] + 1):
            _, score, _, (sx, sy) = cv2.minMaxLoc(scores)
            if index == 0:
                best = float(score)
                best_hypothesis = {"box": [left + sx, top + sy, w, h], "visual_score": best,
                                   "visual_gate_passes": best >= self._minimum_score,
                                   "status": "visual_gate_passed_not_identity_verified" if best >= self._minimum_score else "below_visual_gate",
                                   "coordinate_scope": "hypothesis_only_not_an_observed_position"}
            elif index == 1:
                second = float(score)
            if score < self._minimum_score:
                break
            if index == JOINT_CONTRACT["maximum_peaks"]:
                truncated = True
                break
            box = (left + sx, top + sy, w, h)
            appearance = {"status": "unavailable_color_evidence", "passes": None}
            if track["color"] is not None and color is not None:
                appearance = compare_color(track["color"], color_signature(color[box[1]:box[1]+h, box[0]:box[0]+w]))
            hypotheses.append({"box": list(box), "visual_score": float(score),
                               "appearance_evidence": appearance,
                               "status": "proposed" if appearance["passes"] is not False else "rejected_color"})
            if appearance["passes"] is not False:
                peaks.append({"box": box, "score": float(score), "appearance_evidence": appearance})
            else:
                color_rejected = True
            dx, dy = max(3, w // 3), max(3, h // 3)
            scores[max(0, sy-dy):sy+dy+1, max(0, sx-dx):sx+dx+1] = -1
        return peaks, {"proposal_status": "proposal_limit" if truncated else (
            "candidates_ready" if peaks else "appearance_changed" if color_rejected else "not_detected"),
            "score": best, "peak_margin": None if second is None else best - second,
            "radius_px": radius, "search_pixels": (right-left)*(bottom-top),
            "search_rect_xywh": [left, top, right-left, bottom-top],
            "search_policy": "full_configured_confusable_radius" if track["confusable_source"] else "local_with_bounded_observed_hop_memory",
            "observed_hop_max_px": track["observed_hop_max_px"],
            "best_visual_hypothesis": best_hypothesis,
            "response_positions": int(scores.size), "proposal_limit_reached": truncated,
            "candidate_hypotheses": hypotheses}

    def locate(self, gray, color=None):
        self._validate_frame(gray, color)
        if gray.shape != self._shape:
            raise ValueError("Frame dimensions changed")
        if self._pending is not None:
            raise ValueError("Commit the previous located frame before locating another")
        proposals, audits = [], []
        for track in self._tracks:
            peaks, audit = self._proposals(gray, color, track)
            proposals.append(peaks)
            audits.append(audit)
        contact = set()
        for i, j in self._confusable:
            first, second = self._tracks[i], self._tracks[j]
            size = max(*first["box"][2:], *second["box"][2:])
            competing = any(_overlap(a["box"], b["box"]) > JOINT_CONTRACT["conflict_overlap"]
                            for a in proposals[i] for b in proposals[j])
            nearby_peaks = any(JOINT_CONTRACT["conflict_overlap"] >= _overlap(a["box"], b["box"])
                               and _distance(a["box"], b["box"]) <= JOINT_CONTRACT["contact_object_sizes"] * size
                               for a in proposals[i] for b in proposals[j])
            if competing and (first["gap"] or second["gap"] or nearby_peaks or
                              _distance(first["box"], second["box"]) <= JOINT_CONTRACT["contact_object_sizes"] * size):
                contact.update((i, j))
        for i, (track, peaks) in enumerate(zip(self._tracks, proposals)):
            for peak in peaks:
                temporal = 0.
                if i not in contact and track["gap"] == 0:
                    scale = max(track["box"][2:])
                    predicted = tuple(np.array(track["box"][:2]) + track["velocity"]) + track["box"][2:]
                    # Bounded tie-breaking cannot turn a sub-threshold visual match into evidence.
                    distance = min(_distance(peak["box"], track["box"]), _distance(peak["box"], predicted))
                    temporal = JOINT_CONTRACT["maximum_temporal_bonus"] * math.exp(-distance / scale)
                peak["utility"] = peak["score"] + temporal
                peak["temporal_bonus"] = temporal
        budget = [0]
        exhausted = [False]
        order = sorted(range(len(proposals)), key=lambda i: len(proposals[i]))

        def solve(excluded=None):
            options = {i: sorted([(k, p["utility"]) for k, p in enumerate(proposals[i])
                                  if excluded != (i, k)], key=lambda item: -item[1]) + [(None, 0.)]
                       for i in order}
            upper = [0.] * (len(order) + 1)
            for depth in range(len(order)-1, -1, -1):
                upper[depth] = upper[depth+1] + options[order[depth]][0][1]
            best_value, best_assignment = -math.inf, None
            assignment = [None] * len(proposals)
            occupied = []

            def visit(depth, value):
                nonlocal best_value, best_assignment
                if budget[0] >= JOINT_CONTRACT["maximum_assignment_nodes"]:
                    exhausted[0] = True
                    return
                budget[0] += 1
                if value + upper[depth] <= best_value + 1e-10:
                    return
                if depth == len(order):
                    best_value, best_assignment = value, assignment.copy()
                    return
                i = order[depth]
                for k, utility in options[i]:
                    if exhausted[0]:
                        return
                    box = proposals[i][k]["box"] if k is not None else None
                    if box is not None and any(_overlap(box, other) > JOINT_CONTRACT["conflict_overlap"] for other in occupied):
                        continue
                    assignment[i] = k
                    if box is not None:
                        occupied.append(box)
                    visit(depth+1, value+utility)
                    if box is not None:
                        occupied.pop()
            visit(0, 0.)
            return best_value, best_assignment

        value, assignment = solve()
        margins = [None] * len(proposals)
        if assignment is not None and not exhausted[0]:
            for i, chosen in enumerate(assignment):
                if chosen is not None:
                    alternative, _ = solve((i, chosen))
                    margins[i] = max(0., value - alternative)
                    if exhausted[0]:
                        break
        results = []
        limited = any(audit.get("proposal_limit_reached") for audit in audits)
        for i, audit in enumerate(audits):
            chosen = assignment[i] if assignment is not None else None
            peak = proposals[i][chosen] if chosen is not None else None
            status = audit["proposal_status"]
            if self._tracks[i]["lost"]:
                status = "lost"
            elif exhausted[0]:
                status = "association_limit"
            elif limited:
                status = "proposal_limit"
            elif peak is None and proposals[i]:
                status = "ambiguous_overlap"
            elif peak is not None:
                status = "tracked" if margins[i] >= self._settings["minimum_margin"] else "ambiguous"
            for hypothesis in audit.get("candidate_hypotheses", []):
                if hypothesis["status"] == "proposed":
                    hypothesis["status"] = ("accepted_visual_measurement" if status == "tracked"
                        and hypothesis["box"] == list(peak["box"]) else "rejected_association")
            association = dict(audit, method=JOINT_CONTRACT["method"], final_status=status,
                               frame_sequence=self._frame_number + 1, source_signature=self.source_signatures[i],
                               candidate_count=len(proposals[i]), assignment_nodes=budget[0],
                               assignment_complete=not exhausted[0], association_margin=margins[i],
                               assignment_score=value if assignment is not None else None,
                               assignment_score_scope="sum_of_visual_scores_plus_bounded_temporal_preferences",
                               temporal_assumption="local_continuity_only_outside_similar_appearance_contact_and_gaps",
                               temporal_disabled_contact=i in contact,
                               temporal_bonus=peak["temporal_bonus"] if peak else None,
                               coordinate_evidence="immutable_template_visual_match_only")
            results.append({"status": status, "score": peak["score"] if peak else audit["score"],
                            "margin": margins[i], "box": list(peak["box"]) if status == "tracked" else None,
                            "appearance_evidence": peak["appearance_evidence"] if peak else {
                                "status": "not_checked_no_candidate", "passes": None},
                            "association_evidence": association})
        self._pending = deepcopy(results)
        return results

    def commit(self, candidates):
        if self._pending is None or candidates != self._pending:
            raise ValueError("Commit requires the unchanged results of the latest locate")
        for track, candidate in zip(self._tracks, self._pending):
            if candidate["status"] == "tracked":
                new = candidate["box"]
                hop = np.array(new[:2]) - track["box"][:2]
                # Repeated frames and small jitter cannot erase previously observed search reach.
                track["observed_hop_max_px"] = min(self._settings["search_radius_px"], max(
                    track["observed_hop_max_px"], float(np.linalg.norm(hop))))
                track["velocity"] = hop / (track["gap"] + 1)
                track["box"] = tuple(new)
                track["gap"] = 0
                track["observations"] += 1
            else:
                track["gap"] += 1
                if track["gap"] > self._settings["max_gap_frames"]:
                    track["lost"] = True
        self._frame_number += 1
        self._pending = None
