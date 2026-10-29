"""Bounded appearance tracklets; image continuity is not verified physical identity."""

from copy import deepcopy
import math

import numpy as np

from weft.surface_features import CONTRACT as FEATURE_CONTRACT, _validate, match_surface_features


CONTRACT = {
    'method': 'adjacent-appearance-tracklets-v1', 'maximum_frames': 500,
    'maximum_time_step_s': .2, 'maximum_features_per_frame': 128,
    'maximum_track_ids': 64000,
    'reference': 'Unchanged initialization; never refresh, interpolate, merge or bridge missing frames',
    'association': 'Two-way adjacent descriptor match; split conflicting original-reference claims',
    'scope': 'Appearance hypotheses; object masks, physical identities, rigidity and source-edit continuity unverified',
}


def _stamp(frame, time_s):
    if (type(frame) is not int or not 0 <= frame <= 1_000_000_000
            or type(time_s) not in (int, float) or not math.isfinite(time_s) or not 0 <= time_s <= 1e7):
        raise ValueError('Use a finite nonnegative source clock and integer frame')


def _copy_features(features):
    _validate(features)
    return {key: deepcopy(features[key]) for key in
        ('method', 'keypoints', 'descriptors', 'selected_indices', 'aliases')}


class AppearanceTracklets:
    """Retain only original/previous features and active links; callers journal history."""

    def __init__(self, features, *, frame, time_s):
        _stamp(frame, time_s)
        self._reference = _copy_features(features)
        self._previous = self._reference
        self._frame, self._time = frame, time_s
        self._count, self._next_id = 1, 0
        self._active = {}
        rows = []
        for index, point in enumerate(features['keypoints']):
            if index not in features['selected_indices']:
                rows.append(self._duplicate(index, point, features['aliases'][index]))
                continue
            row = {'current_id': index, 'point_px': point[:2].tolist(), 'duplicate_of': None,
                'track_id': self._new_id(), 'age_frames': 1, 'previous_current_id': None,
                'previous_track_id': None, 'link_status': 'initialization',
                'reference_id': index, 'anchor_status': 'declared_initialization',
                'reference_candidates': [index]}
            rows.append(row); self._active[index] = row
        self._initial = self._result(rows, [], None, None, 'initialization')

    def _new_id(self):
        if self._next_id >= CONTRACT['maximum_track_ids']:
            raise ValueError('Track identity budget exhausted')
        value = self._next_id; self._next_id += 1
        return value

    @staticmethod
    def _duplicate(index, point, alias):
        return {'current_id': index, 'point_px': point[:2].tolist(), 'duplicate_of': alias,
            'track_id': None, 'age_frames': None, 'previous_current_id': None,
            'previous_track_id': None, 'link_status': 'duplicate_location', 'reference_id': None,
            'anchor_status': 'duplicate_location', 'reference_candidates': []}

    def _result(self, rows, ended, adjacent, direct, continuity):
        return {'method': CONTRACT['method'], 'frame': self._frame, 'time_s': self._time,
            'continuity': continuity, 'rows': deepcopy(rows), 'ended_tracks': ended,
            'adjacent_matches': adjacent, 'original_matches': direct,
            'active_tracks': len(self._active), 'total_track_ids': self._next_id,
            'linked_tracks': sum(row['link_status'] == 'adjacent_pair' for row in rows),
            'anchored_tracks': sum(row['reference_id'] is not None for row in rows),
            'conflicted_tracks': sum(row['link_status'] == 'reference_conflict' for row in rows),
            'physical_point_identity_verified': False, 'rigidity_verified': False}

    @property
    def initialization(self):
        return deepcopy(self._initial)

    def advance(self, features, *, frame, time_s):
        _stamp(frame, time_s)
        if frame <= self._frame or time_s <= self._time:
            raise ValueError('Frame and source clock must both increase')
        if self._count >= CONTRACT['maximum_frames']:
            raise ValueError('Feature tracklet frame budget exhausted')
        current = _copy_features(features)
        continuity = ('frame_gap' if frame != self._frame+1 else
            'time_gap' if time_s-self._time > CONTRACT['maximum_time_step_s']+1e-12 else 'adjacent')
        adjacent = match_surface_features(self._previous, current) if continuity == 'adjacent' else None
        direct = match_surface_features(self._reference, current)
        links = {row['current_id']: row['reference_id'] for row in adjacent['rows']
            if row['status'] == 'appearance_pair'} if adjacent else {}
        anchors = {row['current_id']: row['reference_id'] for row in direct['rows']
            if row['status'] == 'appearance_pair'}
        proposals = {}; claims = {}
        for index in current['selected_indices']:
            previous_index = links.get(index); previous = self._active.get(previous_index)
            carried = previous['reference_id'] if previous else None
            observed = anchors.get(index)
            candidates = sorted({value for value in (carried, observed) if value is not None})
            proposal = {'previous_index': previous_index, 'previous': previous,
                'carried': carried, 'observed': observed, 'candidates': candidates,
                'conflict': len(candidates) > 1}
            proposals[index] = proposal
            for candidate in candidates:claims.setdefault(candidate, []).append(index)
        # A direct match and a propagated claim cannot silently assign one anchor twice.
        for indices in claims.values():
            if len(indices) > 1:
                for index in indices:proposals[index]['conflict'] = True
        rows = []; active = {}; continued = set(); reasons = {}
        for index, point in enumerate(current['keypoints']):
            if index not in proposals:
                rows.append(self._duplicate(index, point, current['aliases'][index])); continue
            proposal = proposals[index]; previous = proposal['previous']; conflict = proposal['conflict']
            linked = previous is not None and not conflict
            if linked:continued.add(previous['track_id'])
            elif previous is not None:reasons[previous['track_id']] = 'reference_conflict'
            reference_id = proposal['candidates'][0] if len(proposal['candidates']) == 1 and not conflict else None
            row = {'current_id': index, 'point_px': point[:2].tolist(), 'duplicate_of': None,
                'track_id': previous['track_id'] if linked else self._new_id(),
                'age_frames': previous['age_frames']+1 if linked else 1,
                'previous_current_id': proposal['previous_index'],
                'previous_track_id': previous['track_id'] if previous else None,
                'link_status': 'reference_conflict' if conflict else 'adjacent_pair' if linked else
                    continuity if continuity != 'adjacent' else 'new_appearance',
                'reference_id': reference_id,
                'anchor_status': 'reference_conflict' if conflict else 'direct_reference' if proposal['observed'] is not None
                    else 'propagated_reference' if reference_id is not None else 'unanchored',
                'reference_candidates': proposal['candidates']}
            rows.append(row); active[index] = row
        ended = [{'track_id': row['track_id'], 'last_current_id': index, 'age_frames': row['age_frames'],
            'reference_id': row['reference_id'], 'reason': reasons.get(row['track_id'], continuity if continuity != 'adjacent' else 'no_adjacent_pair')}
            for index, row in self._active.items() if row['track_id'] not in continued]
        self._active, self._previous = active, current
        self._frame, self._time = frame, time_s; self._count += 1
        return self._result(rows, ended, adjacent, direct, continuity)
