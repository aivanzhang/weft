"""Bounded source-template scales; projected extent is not physical body size."""

import math

import cv2
import numpy as np

from weft.appearance_consistency import color_signature, compare_color
from weft.generic_vision import AppearanceTrack


SCALE_CONTRACT = {
    'method': 'immutable-source-template-pyramid-v1',
    'scales': [round(.6 + i * .05, 2) for i in range(17)],
    'maximum_template_bytes': 4_000_000, 'maximum_score_cells': 1_000_000,
    'minimum_template_side': 6,
    'scope': 'Image extent hypothesis only; no segmentation, metric depth, rotation or physical geometry recovery',
}


class ScaleAppearanceTrack(AppearanceTrack):
    def __init__(self, gray, box, settings, *, color=None):
        super().__init__(gray, box, settings, color=color)
        self.original_size = tuple(box[2:])
        sizes = {}
        for scale in SCALE_CONTRACT['scales']:
            width, height = (round(value * scale) for value in self.original_size)
            if min(width, height) >= SCALE_CONTRACT['minimum_template_side'] and width <= gray.shape[1] and height <= gray.shape[0]:
                sizes.setdefault((width, height), scale)
        if sum(width * height for width, height in sizes) > SCALE_CONTRACT['maximum_template_bytes']:
            raise ValueError('Scale template bank exceeds its memory bound')
        self.templates = [(scale, cv2.resize(self.template, size, interpolation=cv2.INTER_LINEAR))
                          for size, scale in sizes.items()]
        self.current_scale = 1.

    def locate(self, gray, *, color=None):
        if self.lost:
            return {'status': 'lost', 'score': None, 'margin': None, 'box': None}
        x, y, width, height = self.box
        center = np.array([x + width / 2, y + height / 2]) + self.velocity * min(self.gap + 1, 3)
        radius = self.settings['search_radius_px']
        searches, cells = [], 0
        for scale, template in self.templates:
            h, w = template.shape
            left, top = max(0, math.floor(center[0] - radius - w / 2)), max(0, math.floor(center[1] - radius - h / 2))
            right, bottom = min(gray.shape[1], math.ceil(center[0] + radius + w / 2)), min(gray.shape[0], math.ceil(center[1] + radius + h / 2))
            if right - left < w or bottom - top < h:
                continue
            cells += (right - left - w + 1) * (bottom - top - h + 1)
            searches.append((scale, template, left, top, right, bottom))
        if cells > SCALE_CONTRACT['maximum_score_cells']:
            return {'status': 'search_budget_exceeded', 'score': None, 'margin': None, 'box': None,
                    'scale_evidence': {'method': SCALE_CONTRACT['method'], 'requested_score_cells': cells}}
        maps, best = [], None
        for scale, template, left, top, right, bottom in searches:
            scores = cv2.matchTemplate(gray[top:bottom, left:right], template, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(scores)
            h, w = template.shape
            item = {'score': float(score), 'scale': scale, 'box': [left + location[0], top + location[1], w, h]}
            maps.append((scores, left, top, w, h))
            # Ties stay closest to the last accepted size; templates never learn from predictions.
            if best is None or (score, -abs(scale - self.current_scale)) > (best['score'], -abs(best['scale'] - self.current_scale)):
                best = item
        if best is None:
            return {'status': 'out_of_frame', 'score': None, 'margin': None, 'box': None}
        bx, by, bw, bh = best['box']
        second = -1.
        for scores, left, top, w, h in maps:
            cx, cy = bx + bw / 2 - left - w / 2, by + bh / 2 - top - h / 2
            dx, dy = max(3, min(bw, w) // 3), max(3, min(bh, h) // 3)
            x0, x1 = max(0, math.ceil(cx - dx)), min(scores.shape[1], math.floor(cx + dx) + 1)
            y0, y1 = max(0, math.ceil(cy - dy)), min(scores.shape[0], math.floor(cy + dy) + 1)
            if x0 < x1 and y0 < y1:
                scores[y0:y1, x0:x1] = -1
            second = max(second, float(scores.max()))
        margin = best['score'] - second
        status = ('tracked' if best['score'] >= self.settings['minimum_score'] and margin >= self.settings['minimum_margin'] else
                  'ambiguous' if best['score'] >= self.settings['minimum_score'] else 'not_detected')
        appearance = {'status': 'not_checked_no_candidate', 'passes': None}
        if status == 'tracked':
            appearance = {'status': 'unavailable_color_evidence', 'passes': None}
            if self.source_color is not None and color is not None:
                appearance = compare_color(self.source_color, color_signature(color[by:by+bh, bx:bx+bw]))
                if appearance['passes'] is False:
                    status = 'appearance_changed'
        return {'status': status, 'score': best['score'], 'margin': margin,
                'box': best['box'] if status == 'tracked' else None, 'appearance_evidence': appearance,
                'scale_evidence': {'method': SCALE_CONTRACT['method'], 'candidate_scale': best['scale'],
                    'at_bound': best['scale'] in (SCALE_CONTRACT['scales'][0], SCALE_CONTRACT['scales'][-1]),
                    'tested_scales': len(searches), 'score_cells': cells,
                    'scope': SCALE_CONTRACT['scope']}}

    def commit(self, candidate):
        if candidate['status'] != 'tracked':
            super().commit(candidate)
            return
        old, new = self.box, candidate['box']
        previous_center = np.array(old[:2]) + np.array(old[2:]) / 2
        center = np.array(new[:2]) + np.array(new[2:]) / 2
        self.velocity = .5 * self.velocity + .5 * (center - previous_center) / (self.gap + 1)
        self.box, self.gap = tuple(new), 0
        self.current_scale = candidate['scale_evidence']['candidate_scale']
