"""Partition an explicit total tuft budget by material without sharing color images."""

from copy import deepcopy
import hashlib
import json
import math

import numpy as np

from material_fur import allocate_counts


def add_material_coat(obj, rig, mask, colors, cfg, sources, out, build_coat):
    obj.data.calc_loop_triangles()
    triangles = list(obj.data.loop_triangles)
    points = np.asarray([tuple(v.co) for v in obj.data.vertices], dtype=np.float64)
    names = cfg['material_fur']['materials']
    slots = sorted(slot for slot, source in sources.items() if source['material_name'] in names)
    eligible = [t for t in triangles if t.material_index in slots and all(mask[i] for i in t.vertices)]
    areas = {t.index: float(np.linalg.norm(np.cross(points[t.vertices[1]]-points[t.vertices[0]],
                                                   points[t.vertices[2]]-points[t.vertices[0]]))*.5)
             for t in eligible}
    if not eligible or any(not math.isfinite(area) or area <= 0 for area in areas.values()):
        raise ValueError('Material-owned coat requires nondegenerate eligible source faces')
    allocations, groups = [], {slot: [] for slot in slots}
    for band_index, band in enumerate(cfg['portable_fur']['bands']):
        selected = [t for t in eligible
                    if band['brightness'][0] <= float(colors[list(t.vertices)].mean()) < band['brightness'][1]]
        active_slots = [slot for slot in slots if any(t.material_index == slot for t in selected)]
        if not active_slots:
            raise ValueError('A requested portable-fur brightness band has no owned source faces')
        weights = [sum(areas[t.index] for t in selected if t.material_index == slot) for slot in active_slots]
        counts = allocate_counts(weights, band['count'], minimum=1, max_total=5000)
        row = {'source_band': band_index, 'requested_cards': band['count'], 'parts': []}
        for slot, count in zip(active_slots, counts):
            face_ids = [t.index for t in selected if t.material_index == slot]
            row['parts'].append({'material_slot': slot, 'material_name': sources[slot]['material_name'],
                                 'cards': count, 'eligible_triangles': len(face_ids),
                                 'source_triangle_ids_sha256': hashlib.sha256(
                                     np.asarray(face_ids, dtype='<i4').tobytes()).hexdigest()})
            if count:
                groups[slot].append((band_index, {**band, 'count': count}, face_ids))
        allocations.append(row)
    meshes, reports = [], []
    for slot in slots:
        if not groups[slot]:
            continue
        part_mask = np.zeros(len(mask), dtype=bool)
        for triangle in eligible:
            if triangle.material_index == slot:
                part_mask[list(triangle.vertices)] = True
        part_mask &= mask
        config = deepcopy(cfg['portable_fur'])
        config['bands'] = [band for _, band, _ in groups[slot]]
        # Stable material seeds prevent repeated root patterns without changing the legacy path.
        config['settings']['seed'] = (config['settings']['seed']+slot*1009) % (2**31-4)
        built, part_reports = build_coat(obj, rig, part_mask, colors, config, out,
            tuft_source=sources[slot], native_groom=cfg['groom'], groom_flow=cfg['groom_flow'])
        if len(part_reports) != len(groups[slot]):
            raise ValueError('Material coat report differs from its requested bands')
        for local_band, (report, (source_band, _, triangle_ids)) in enumerate(zip(part_reports, groups[slot])):
            report.update(source_material_slot=slot, source_material=sources[slot]['material_name'],
                          source_band=source_band, material_owned=True,
                          resolved_seed=config['settings']['seed']+local_band)
            for sample in report.get('root_samples', []):
                ordinal = sample.get('triangle_index')
                if ordinal is None:
                    raise ValueError('Tuft root evidence has no source triangle identity')
                if type(ordinal) is not int or not 0 <= ordinal < len(triangle_ids):
                    raise ValueError('Tuft root evidence escaped its owned material faces')
                triangle = triangles[triangle_ids[ordinal]]
                sample.update(source_material_slot=slot, source_polygon=triangle.polygon_index,
                              source_loops=list(triangle.loops), source_vertices=list(triangle.vertices))
        meshes.extend(built)
        reports.extend(part_reports)
    requested = sum(band['count'] for band in cfg['portable_fur']['bands'])
    if sum(report['cards'] for report in reports) != requested:
        raise ValueError('Material splitting changed the global portable-fur count')
    summary = {'schema_version': 1, 'policy': 'material-owned-tufts-v1', 'allocations': allocations,
               'requested_cards': requested, 'created_cards': requested,
               'pixel_lifetime': 'One material buffer per build call, released before the next material',
               'root_color_scope': 'Exact source-owned UV roots or constant linear material RGB',
               'body_materials_or_uvs_changed': False}
    with (out/'material-coat.json').open('x') as stream:
        json.dump(summary, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return meshes, reports, summary
