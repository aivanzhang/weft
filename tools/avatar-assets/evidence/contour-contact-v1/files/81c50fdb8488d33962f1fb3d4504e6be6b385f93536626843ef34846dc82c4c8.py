"""Resolve selected/protected current faces to original v7 by bounded oriented P+UV evidence."""
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(OUT/'sources'))
from three_d import _GLB, MAX_TRIANGLE_INDICES

START = time.monotonic()
TOLERANCE = 2e-7  # Frozen planner correspondence tolerance; not changed by this checker.


def require(value, message):
    if not value:
        raise ValueError(message)
    if time.monotonic()-START > 27:
        raise TimeoutError('Identity supplement deadline')


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def faces(glb, mesh):
    values = []
    for primitive in glb.doc['meshes'][mesh]['primitives']:
        attr = primitive['attributes']
        p = np.asarray(glb.accessor(attr['POSITION'], 'VEC3', (5126,)), dtype=float)
        uv = np.asarray(glb.accessor(attr['TEXCOORD_0'], 'VEC2', (5126,)), dtype=float)
        idx = np.asarray(glb.accessor(primitive['indices'], 'SCALAR', (5121,5123,5125),
                                    count_limit=MAX_TRIANGLE_INDICES), dtype=np.int64).reshape(-1,3)
        values.append((p[idx], uv[idx]))
    return values


def main():
    pins = json.loads((OUT/'identity-prepared.json').read_text())
    for row in pins['inputs']:
        require(sha(ROOT/row['path']) == row['sha256'], 'Identity input pin')
    source = _GLB(ROOT/'outputs/companion-separation-v7/diagnostic-model.glb')
    current = _GLB(ROOT/'outputs/panda-staged-portable-model-v1/model.glb')
    original, now = faces(source, 0), faces(current, 0)
    original_p = np.concatenate([v[0] for v in original])
    original_uv = np.concatenate([v[1] for v in original])
    offsets = np.cumsum([0]+[len(v[0]) for v in original])
    centers = original_p.mean(axis=1)
    selector = json.loads((ROOT/'outputs/panda-shoulder-uv-repair-plan-v1/face-selector.json').read_text())
    census = json.loads((ROOT/'outputs/panda-shoulder-uv-repair-plan-v1/candidate-census.json').read_text())
    records = [(True, r) for r in selector['rows']] + [(False, r) for r in census['rows'] if r['protected_white']]
    require(len(records) == 486, 'Expected 242 selected plus 244 protected white rows')
    resolved, selected, protected = [], set(), set()
    for is_selected, row in records:
        cp, ct = row['current_primitive'], row['current_triangle']
        p, uv = now[cp][0][ct], now[cp][1][ct]
        nearby = np.flatnonzero(np.max(np.abs(centers-p.mean(axis=0)), axis=1) <= TOLERANCE)
        matches = []
        for fi in nearby:
            for rotation in range(3):
                rp, ru = np.roll(original_p[fi], rotation, axis=0), np.roll(original_uv[fi], rotation, axis=0)
                pe, ue = float(np.max(np.abs(rp-p))), float(np.max(np.abs(ru-uv)))
                if pe <= TOLERANCE and ue <= TOLERANCE:
                    sp = int(np.searchsorted(offsets, fi, side='right')-1)
                    matches.append((sp, int(fi-offsets[sp]), rotation, pe, ue))
        require(len(matches) == 1, f'Ambiguous/missing oriented P+UV match for current {cp}/{ct}: {len(matches)}')
        sp, st, rotation, pe, ue = matches[0]
        if is_selected:
            require((sp,st) == (row['source_primitive'],row['source_triangle']), 'Frozen selected source identity mismatch')
            selected.add((sp,st))
        else:
            protected.add((sp,st))
        resolved.append({'selected': is_selected, 'current': [cp,ct], 'source': [sp,st],
                         'source_to_current_rotation': rotation, 'position_max_error_m': pe, 'uv_max_error': ue})
    require(len(selected) == 242 and len(protected) == 244 and not selected.intersection(protected),
            'Selected/protected source face alias or overlap')
    for row in pins['inputs']:
        require(sha(ROOT/row['path']) == row['sha256'], 'Identity input changed')
    return {'schema_version': 1, 'status': 'selected_and_protected_source_identity_proved',
            'source_sha256': source.hash, 'current_sha256': current.hash,
            'selected_faces': len(selected), 'protected_white_faces': len(protected),
            'protected_source_intersection': [], 'all_unique_matches': True,
            'position_uv_tolerance': TOLERANCE, 'matching': 'All original triangles searched by conservative center bound, then all three oriented rotations; no normal/color/desired-label tie breaking.',
            'maximum_position_error_m': max(r['position_max_error_m'] for r in resolved),
            'maximum_uv_error': max(r['uv_max_error'] for r in resolved), 'resolved': resolved,
            'elapsed_seconds': time.monotonic()-START,
            'process_high_water_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'image_decodes': 0, 'native_calls': 0, 'quality_approved': False}


try:
    result = main()
except Exception as error:
    with (OUT/'identity-failure.json').open('x') as handle:
        handle.write(json.dumps({'status':'failed','type':type(error).__name__,'error':str(error)},indent=2)+'\n')
    raise
else:
    with (OUT/'identity-result.json').open('x') as handle:
        handle.write(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='resolved'},sort_keys=True))
