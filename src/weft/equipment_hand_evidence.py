"""Named source hand points versus equipment masks; no filled hand or contact inference."""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import time

from weft.equipment_silhouette_evidence import (
    MAX_CACHE_BYTES, MAX_SECONDS, POINT_PROFILE, _check, _prepare_equipment_point_evidence,
)
from weft.pose_surface_annotations import PoseSurfaceAnnotations
from weft.source_surface_interval_construction import _encode


PROFILE = 'equipment-hand-pointset-evidence-v1'
PARTS = ('wrist', 'pinky', 'index', 'thumb')
HAND_POINTS = {
    'left_hand': {'wrist': 15, 'pinky': 17, 'index': 19, 'thumb': 21},
    'right_hand': {'wrist': 16, 'pinky': 18, 'index': 20, 'thumb': 22},
}
KNOWN = ('conditional_image_intersection', 'conditional_image_exclusion')


def _require(value, message):
    if not value:
        raise ValueError('Equipment hand evidence: ' + message)


def _selection(value):
    _require(type(value) is dict and 1 <= len(value) <= 2 and set(value) <= set(HAND_POINTS),
             'explicit left_hand/right_hand channel selection required')
    result, channels, roles, indices = [], [], {}, {}
    for hand, point_indices in HAND_POINTS.items():
        if hand not in value:
            continue
        supplied = value[hand]
        _require(type(supplied) is dict and set(supplied) == set(PARTS),
                 'every selected hand requires wrist/pinky/index/thumb channels')
        points = []
        side = hand.removesuffix('_hand')
        for part in PARTS:
            channel, name = supplied[part], side + '_' + part
            _require(type(channel) is str and 0 < len(channel) <= 80 and channel not in channels,
                     'bounded unique explicitly named channels required')
            channels.append(channel); roles[channel] = name; indices[name] = point_indices[part]
            points.append({'part': part, 'point_name': name, 'landmark_index': point_indices[part],
                           'channel_id': channel})
        result.append({'hand_id': hand, 'wrist_name': side + '_wrist', 'points': points})
    return result, channels, roles, indices


@dataclass(frozen=True)
class EquipmentHandEvidence:
    pose_annotations: PoseSurfaceAnnotations
    equipment_document: bytes
    mask_document: bytes
    source_equivalence: object | None
    _metadata: bytes
    _rows: tuple[bytes, ...]

    def describe(self):
        return json.loads(self._metadata)

    def frame(self, index):
        _require(type(index) is int and 0 <= index < len(self._rows), 'original source frame outside range')
        return json.loads(self._rows[index])


def prepare_equipment_hand_evidence(pose_annotations, *, hand_channel_ids, equipment_document,
        mask_document, equipment_ids, raster_loader, source_equivalence=None, construction_frames=None):
    """Require four qualified named rectangles; any missing point keeps a hand group unknown.

    The raster callback and source joins are the existing silhouette contract.
    Existing co-motion candidates may later be restricted, never created by this evidence.
    """
    deadline = time.monotonic() + MAX_SECONDS
    groups, channels, roles, indices = _selection(hand_channel_ids)
    points = _prepare_equipment_point_evidence(pose_annotations, equipment_document=equipment_document,
        mask_document=mask_document, equipment_ids=equipment_ids, channel_ids=channels,
        point_indices=indices, channel_roles=roles, profile=POINT_PROFILE, raster_loader=raster_loader,
        source_equivalence=source_equivalence, construction_frames=construction_frames, deadline=deadline)
    point_metadata = points.describe()
    _check(deadline)
    rows, cache_bytes, statuses, point_statuses = [], 0, Counter(), Counter()
    group_queries = 0
    for encoded in points._rows:
        _check(deadline)
        row = json.loads(encoded)
        by_pair = {(point['equipment_id'], point['point_channel_id']): point for point in row['pairs']}
        _require(len(by_pair) == len(row['pairs']) == len(equipment_ids) * len(channels),
                 'complete unique equipment/named-point slots required')
        hands = []
        for equipment_id in equipment_ids:
            for group in groups:
                selected = [by_pair[(equipment_id, item['channel_id'])] for item in group['points']]
                _require(all(point['point_name'] == item['point_name']
                             and point['landmark_index'] == item['landmark_index']
                             for point, item in zip(selected, group['points'], strict=True)),
                         'source point names or indices differ')
                counts = {name: sum(point['status'] == status for point in selected)
                          for name, status in zip(('intersection', 'exclusion'), KNOWN, strict=True)}
                counts['unavailable'] = 4 - counts['intersection'] - counts['exclusion']
                unavailable = [{'point_name': point['point_name'], 'reason': point['reason']}
                               for point in selected if point['status'] not in KNOWN]
                status = ('unavailable' if unavailable else 'conditional_pointset_image_intersection'
                          if counts['intersection'] else 'conditional_pointset_image_exclusion')
                hands.append({'equipment_id': equipment_id, 'hand_id': group['hand_id'],
                    'wrist_name': group['wrist_name'], 'human_entity_ref': selected[0]['human_entity_ref'],
                    'status': status, 'reason': 'required_hand_point_unavailable' if unavailable else None,
                    'intersects': None if unavailable else bool(counts['intersection']),
                    'point_counts': counts, 'unavailable_points': unavailable, 'points': selected,
                    'contact': 'unknown', 'physical_noncontact': 'unknown', 'grasp': 'unknown', 'load': 'unknown'})
                statuses[status] += 1
                point_statuses.update(point['reason'] or point['status'] for point in selected)
                group_queries += 1
        row = {key: value for key, value in row.items() if key != 'pairs'} | {'hands': hands}
        encoded = _encode(row, 128 << 10, deadline); cache_bytes += len(encoded)
        _require(cache_bytes <= MAX_CACHE_BYTES, 'complete grouped cache exceeds eight MiB; no slots dropped')
        rows.append(encoded)
    point_cache_bytes = point_metadata['cached_row_bytes'] + len(points._metadata)
    # Raw point records survive inside each group; do not retain a second whole row cache.
    del points
    metadata = {key: point_metadata[key] for key in (
        'schema_version', 'status', 'source_sha256', 'equipment_source_sha256', 'equipment_document_sha256',
        'mask_document_sha256', 'pose_provenance', 'source_equivalence', 'source_clocks',
        'construction_frames', 'selection_policy', 'pixel_cell_mapping', 'contact', 'physical_noncontact',
        'load', 'grasp', 'physical_validation', 'body_or_floor_installed', 'native_calls', 'image_decodes_in_adapter')}
    counts = point_metadata['counts']
    metadata.update(profile=PROFILE, hand_groups=groups, point_channels=point_metadata['point_channels'],
        equipment_ids=list(equipment_ids),
        counts={key: counts[key] for key in ('source_frames', 'decoded_frames', 'metadata_frames', 'pose_slots')} | {
            'hand_pair_slots': group_queries, 'point_slots': counts['pair_slots'],
            'hand_status_counts': dict(statuses), 'point_status_counts': dict(point_statuses)},
        work=point_metadata['work'] | {'grouped_source_rows': len(rows), 'hand_group_queries': group_queries,
            'grouping_raster_calls': 0, 'grouping_pose_queries': 0},
        cached_row_bytes=cache_bytes, preparation_point_cache_bytes=point_cache_bytes,
        point_preparation_metadata_sha256=sha256(_encode(point_metadata, 256 << 10, deadline)).hexdigest(),
        bounds=point_metadata['bounds'] | {'combined_preparation_grouping_seconds': MAX_SECONDS,
            'maximum_transient_encoded_row_caches_bytes': 2 * MAX_CACHE_BYTES, 'hands': 2, 'points_per_hand': 4},
        decision_rule='All four source points must qualify; any rectangle intersection gives point-set image consistency, otherwise point-set exclusion',
        geometry_convention='Union of four separately qualified closed named-point rectangles versus exact half-open foreground cells; no hull, dilation or filled hand region',
        consumer_policy='May only restrict original co-motion candidates after exact source/clock/window joins; not installed in the selector by this producer',
        scope='Model keypoint-set/image-mask consistency only; not a complete hand surface, contact, noncontact, grasp, depth, load or authenticated identity')
    if 'canonical_frame_selection' in point_metadata:
        metadata['canonical_frame_selection'] = point_metadata['canonical_frame_selection']
    encoded = _encode(metadata, 256 << 10, deadline)
    _require(cache_bytes + len(encoded) <= MAX_CACHE_BYTES,
             'complete grouped metadata/row cache exceeds eight MiB')
    _check(deadline)
    return EquipmentHandEvidence(pose_annotations, equipment_document, mask_document, source_equivalence, encoded, tuple(rows))
