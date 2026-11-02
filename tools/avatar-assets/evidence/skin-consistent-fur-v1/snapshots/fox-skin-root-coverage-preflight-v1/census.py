"""Classify preserved fox faces; do not select or generate hybrid roots."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import time
import zipfile

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
START = time.monotonic()
spec = importlib.util.spec_from_file_location('previous_census_helpers', OUT/'sources/reference_census.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
require, sha = old.require, old.sha


def save(name, value):
    with (OUT/name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def canonical(joints, weights, bone_names):
    pairs = sorted((bone_names[int(j)], float(w)) for j, w in zip(joints, weights) if w > 0)
    require(1 <= len(pairs) <= 4 and len({k for k, _ in pairs}) == len(pairs), 'Invalid positive named support')
    total = math.fsum(value for _, value in pairs)
    require(math.isfinite(total) and abs(total-1) <= 1e-5, 'Source weight sum outside existing tolerance')
    return tuple((name, value/total) for name, value in pairs)


def domain_counts(total, areas):
    exact = [total*area/math.fsum(areas) for area in areas]
    counts = [math.floor(value) for value in exact]
    for index in sorted(range(2), key=lambda i: (-(exact[i]-counts[i]), i))[:total-sum(counts)]:
        counts[index] += 1
    adjustments = []
    if total >= 2:
        for index in range(2):
            if areas[index] > 0 and counts[index] == 0:
                donor = 1-index
                require(counts[donor] > 1, 'Insufficient roots for both nonempty domains')
                counts[donor] -= 1; counts[index] += 1
                adjustments.append({'from_domain': donor, 'to_domain': index, 'roots': 1})
    require(sum(counts) == total and all(a > 0 or n == 0 for a, n in zip(areas, counts)), 'Invalid allocation')
    return counts, exact, adjustments


def aliases(points, uv, tuples, global_ids):
    _, inverse, sizes = np.unique(points, axis=0, return_inverse=True, return_counts=True)
    repeated = np.flatnonzero(sizes > 1)
    examples, uv_seams, skin_seams = [], 0, 0
    for label in repeated:
        rows = np.flatnonzero(inverse == label)
        uv_count = len(np.unique(uv[rows], axis=0))
        skin_count = len({tuples[int(i)] for i in rows})
        uv_seams += uv_count > 1; skin_seams += skin_count > 1
        if len(examples) < 8:
            examples.append({'global_source_vertices': global_ids[rows].tolist(),
                             'uv_variants': uv_count, 'normalized_named_skin_variants': skin_count})
    return {'distinct_positions': len(sizes), 'coincident_groups': len(repeated),
        'duplicate_position_excess': len(points)-len(sizes),
        'coincident_instances': int(sizes[repeated].sum()),
        'max_multiplicity': int(sizes.max()) if len(sizes) else 0,
        'uv_seam_groups': int(uv_seams), 'skin_seam_groups': int(skin_seams), 'examples': examples}


def main():
    prepared = json.loads((OUT/'prepared.json').read_text())
    for item in prepared['inputs']:
        p = ROOT/item['path']
        require(p.is_file() and not p.is_symlink() and p.stat().st_size == item['bytes']
                and sha(p.read_bytes()) == item['sha256'], 'Input changed: '+item['path'])
    base = ROOT/'outputs/fox-vertex-root-coverage-preflight-v1'
    binding = json.loads((base/'source-binding.json').read_text())
    previous = json.loads((base/'report.json').read_text())
    require(binding['exact_native_array_binding'] is True, 'Previous immutable body-ID proof unavailable')
    source = ROOT/'outputs/fox-material-fur-finish-v1'
    cfg = json.loads((source/'finish-config.json').read_text())
    groom = json.loads((source/'material-groom/report.json').read_text())
    coat = json.loads((source/'material-coat.json').read_text())
    doc, binary = old.glb(Path(cfg['source']))
    require(len(doc['skins']) == len(doc['meshes']) == 1, 'Changed source mesh/skin')
    bone_names = [doc['nodes'][i]['name'] for i in doc['skins'][0]['joints']]
    require(len(bone_names) <= 64 and len(set(bone_names)) == len(bone_names), 'Ambiguous joint names')
    cp = source/'material-groom/correspondence.npz'
    with zipfile.ZipFile(cp) as z:
        require(len(z.namelist()) <= 16 and len(set(z.namelist())) == len(z.namelist())
                and sum(x.file_size for x in z.infolist()) <= 8*1024**2, 'Correspondence exceeds bound')
    with np.load(cp, allow_pickle=False) as z:
        correspondence = {k:z[k] for k in z.files if k.endswith('_face_ids')}
    reports, arrays = [], {}
    for part, allocation, former in zip(groom['parts'], coat['allocations'][0]['parts'], previous['materials']):
        material = part['source_material']
        ranges = [x for x in binding['primitive_ranges'] if x['material'] == material]
        require(len(ranges) == 1, 'Expected one proven source primitive per owned material')
        bound = ranges[0]; primitive = doc['meshes'][0]['primitives'][bound['primitive']]
        attrs = primitive['attributes']
        p = old.read_array(doc, binary, attrs['POSITION'])
        p = np.column_stack((p[:, 0], -p[:, 2], p[:, 1])).astype('<f4')
        uv = old.read_array(doc, binary, attrs['TEXCOORD_0'])
        j = old.read_array(doc, binary, attrs['JOINTS_0'])
        w = old.read_array(doc, binary, attrs['WEIGHTS_0'])
        f = old.read_array(doc, binary, primitive['indices']).ravel().astype('<i4').reshape(-1, 3)
        require(len(p) == bound['vertex_count'] and len(f) == bound['face_count']
                and len(p) == len(uv) == len(j) == len(w) and f.min() >= 0 and f.max() < len(p)
                and j.max() < len(bone_names) and np.all(w >= 0), 'Primitive differs from proven source ranges')
        selected_global = correspondence[f"part_{part['index']:02d}_face_ids"]
        selected = selected_global-bound['face_start']
        require(selected.min() >= 0 and selected.max() < len(f), 'Selected face IDs escape source primitive')
        f = f[selected]
        tri = p[f].astype(float)
        area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)*.5
        require(np.all(area > 0) and len(f) == part['faces']
                and abs(area.sum()-part['selected_area_m2']) < 1e-7*part['selected_area_m2'], 'Eligible domain changed')
        tuples = [canonical(x, y, bone_names) for x, y in zip(j, w)]
        classes, vocabulary = [], {}
        for value in tuples:
            if value not in vocabulary:
                vocabulary[value] = len(vocabulary)
            classes.append(vocabulary[value])
        classes = np.asarray(classes)
        same = (classes[f[:, 0]] == classes[f[:, 1]]) & (classes[f[:, 0]] == classes[f[:, 2]])
        uniform_area, varying_area = float(area[same].sum()), float(area[~same].sum())
        counts, quotas, adjustments = domain_counts(allocation['cards'], [uniform_area, varying_area])
        varying_vertices = np.unique(f[~same])
        varying_ids = varying_vertices+bound['vertex_start']
        alias = aliases(p[varying_vertices], uv[varying_vertices],
                        [tuples[i] for i in varying_vertices], varying_ids)
        labels = old.components(f, len(p))
        components = []
        for label in np.unique(labels[np.unique(f)]):
            member = labels[f[:, 0]] == label
            uniform = member & same; varying = member & ~same
            v = np.unique(f[varying]); unique_positions = len(np.unique(p[v], axis=0))
            varea = float(area[varying].sum()); uarea = float(area[uniform].sum())
            vquota = counts[1]*varea/varying_area if varying_area else 0
            uquota = counts[0]*uarea/uniform_area if uniform_area else 0
            named_support = sorted({name for vertex in np.unique(f[member]) for name, _ in tuples[vertex]})
            global_label = int(label+bound['vertex_start'])
            old_row = next(x for x in former['indexed_components'] if x['source_vertex_component'] == global_label)
            require(old_row['faces'] == int(member.sum()), 'Old component correspondence changed')
            components.append({'source_vertex_component': global_label, 'authored_bone_support': named_support,
                'faces': int(member.sum()), 'uniform_faces': int(uniform.sum()), 'varying_faces': int(varying.sum()),
                'uniform_area_m2': uarea, 'varying_area_m2': varea,
                'uniform_fractional_quota': uquota, 'varying_fractional_quota': vquota,
                'varying_candidate_vertices': len(v), 'varying_distinct_positions': unique_positions,
                'varying_index_capacity_covers_fractional_quota': len(v) >= vquota,
                'old_vertex_only_fractional_quota': old_row['fractional_area_quota'],
                'old_vertex_only_capacity': old_row['vertices'],
                'old_vertex_only_saturated': old_row['fractional_area_quota'] > old_row['vertices']})
        components.sort(key=lambda x:(-(x['uniform_area_m2']+x['varying_area_m2']),x['source_vertex_component']))
        lump = np.bincount(f[~same].ravel(), weights=np.repeat(area[~same]/3,3), minlength=len(p))[varying_vertices]
        nominal = counts[1]*lump/varying_area if varying_area else np.zeros(len(lump))
        prefix = f"part_{part['index']:02d}"
        arrays[prefix+'_uniform_source_faces'] = selected_global[same].astype('<i4')
        arrays[prefix+'_varying_source_faces'] = selected_global[~same].astype('<i4')
        arrays[prefix+'_varying_source_vertices'] = varying_ids.astype('<i4')
        skin_digest = sha(json.dumps(tuples, separators=(',',':'), allow_nan=False).encode())
        reports.append({'material': material, 'material_slot': part['source_material_slot'],
            'source_primitive': bound['primitive'], 'requested_tufts': allocation['cards'],
            'uniform_faces': int(same.sum()), 'varying_faces': int((~same).sum()),
            'uniform_area_m2': uniform_area, 'varying_area_m2': varying_area,
            'uniform_area_fraction': uniform_area/float(area.sum()),
            'domain_order': ['uniform_interior','varying_vertex'], 'fractional_domain_quotas': quotas,
            'domain_counts': counts, 'minimum_one_transfers': adjustments,
            'varying_candidate_vertices': len(varying_vertices), 'varying_aliases': alias,
            'varying_index_capacity_pass': len(varying_vertices) >= counts[1],
            'varying_distinct_position_capacity_pass': alias['distinct_positions'] >= counts[1],
            'varying_nominal_inclusion_mass_over_one': int((nominal>1).sum()),
            'varying_maximum_nominal_inclusion_mass': float(nominal.max()) if len(nominal) else 0,
            'normalized_named_tuples_sha256': skin_digest, 'normalized_named_tuple_classes': len(vocabulary),
            'all_original_eligible_faces_retained': int(same.sum()+(~same).sum()) == len(selected_global),
            'components': components})
        require(time.monotonic()-START < 25, 'Internal census deadline')
    require(sum(x.nbytes for x in arrays.values()) <= 4*1024**2, 'Classification arrays exceed bound')
    with (OUT/'classification.npz').open('xb') as stream:
        np.savez(stream, **arrays)
    report = {'schema_version':1, 'status':'measured',
        'policy_under_review':'skin-consistent-v1', 'source_sha256':cfg['source_sha256'],
        'original_body_proof_reused':{'path':str(base.relative_to(ROOT)/'source-binding.json'),
            'sha256':sha((base/'source-binding.json').read_bytes()), 'original_body_proof_rerun':False},
        'normalization':'Positive named influences, sort exact names, divide each by math.fsum; exact tuple equality, no cutoff/tolerance.',
        'domain_allocation':'Hamilton area quotas, stable uniform-first ties; transfer one only for a nonempty zero domain when count >=2.',
        'material_counts_unchanged':sum(x['requested_tufts'] for x in reports)==3000,
        'materials':reports,'classification_sha256':sha((OUT/'classification.npz').read_bytes()),
        'limits':{'seconds':30,'memory_mb':512,'output_mb':16,'native_jobs':0,'provider_calls':0,'role_calls':0},
        'limitations':['No root selector or model was executed; these are domain capacity measurements.',
            'Retaining all mixed-weight triangles in the candidate domain does not give every triangle a root.',
            'Distinct source indices can still coincide; no welding, identity substitution or count reallocation occurred.',
            'Fractional component quotas diagnose area pressure, not exact weighted-without-replacement inclusion probabilities.',
            'Authored bone names identify rig roles, not recovered anatomy; no visual density or Muse-parity verdict.'],
        'elapsed_seconds':time.monotonic()-START,
        'self_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)}
    save('report.json',report)
    print(json.dumps({'status':'measured','materials':[{k:x[k] for k in ['material','uniform_faces','varying_faces',
        'uniform_area_fraction','domain_counts','varying_candidate_vertices','varying_index_capacity_pass']}
        for x in reports]},sort_keys=True))


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        save('failure.json',{'status':'failed','type':type(exc).__name__,'message':str(exc),
                             'elapsed_seconds':time.monotonic()-START})
        raise
