"""CPU-only code-path replay; no Blender process, asset edits, or acceptance."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def normalized(value):
    return value / np.linalg.norm(value, axis=-1, keepdims=True)


def main():
    extract = HERE/'sources/installed/primitive_extract.py'
    tree = ast.parse(extract.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'PrimitiveCreator')
    func = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__get_bone_data')
    # Execute only the reviewed installed pure-Python weight method on tiny witnesses.
    module = ast.Module(body=[func], type_ignores=[])
    scope = {}
    exec(compile(module, str(extract), 'exec'), scope)
    panda = json.loads((ROOT/'outputs/panda-fringe-attribute-diagnosis-v1/position-uv-material-00.json').read_text())
    weights = []
    for row in panda['weight_comparison']['worst_violations']:
        source = np.asarray(row['source_dense_weights'], dtype=np.float32)
        target = np.asarray(row['target_dense_weights'], dtype=np.float32)
        groups = [SimpleNamespace(group=i, weight=float(w)) for i, w in enumerate(source) if w > 0]
        names = [str(i) for i in range(len(source))]
        owner = SimpleNamespace(skin=SimpleNamespace(joints=[SimpleNamespace(name=n) for n in names]),
            blender_vertex_groups=[SimpleNamespace(name=n) for n in names],
            blender_mesh=SimpleNamespace(vertices=[SimpleNamespace(groups=groups)]))
        scope['__get_bone_data'](owner)
        kept = owner.vert_bones[0]
        # This is the installed float32 sum/divide at primitive_attributes.py149-161.
        four = np.zeros(4, dtype=np.float32)
        for i, (_, w) in enumerate(kept):
            four[i] = w
        four /= four.sum()
        predicted = np.zeros_like(source)
        for i, (joint, _) in enumerate(kept):
            predicted[joint] = four[i]
        weights.append({'target_vertex': row['target_vertex'], 'helper_vertex': row['closest_source_vertex'],
            'source_positive_influences': int(np.count_nonzero(source)), 'kept_positive_influences': len(kept),
            'removed_positive_weights': source[(source > 0) & (source <= .0001)].tolist(),
            'predicted_dense_weights': predicted.tolist(), 'actual_dense_weights': target.tolist(),
            'predicted_vs_actual_component_max': float(np.max(abs(predicted-target))),
            'predicted_vs_actual_float32_bytes_equal': predicted.tobytes() == target.tobytes()})

    rows = []
    for name, folder, diagnosis_path in (
        ('panda', 'panda-portable-fringe-finish-v1', 'panda-fringe-attribute-diagnosis-v1/position-uv-material-00.json'),
        ('fox', 'fox-fringe-component-v2', 'fox-fringe-export-diagnosis-v2/material-16.json')):
        path = ROOT/'outputs'/folder/'fringe-correspondence.npz'
        with zipfile.ZipFile(path) as archive:
            assert len(archive.infolist()) <= 20 and sum(i.file_size for i in archive.infolist()) <= 16*1024**2
        with np.load(path, allow_pickle=False) as saved:
            points = saved['positions'].copy()
            normal = saved['normals'].copy()
            faces = saved['triangles'].copy()
            corner_map = saved['native_vertex_to_helper_vertex'].copy()
        assert len(points) <= 19200 and len(faces) <= 19200 and np.array_equal(corner_map, faces.ravel())
        rounding = normalized(np.round(normal.astype(np.float32), 4))
        diagnosis = json.loads((ROOT/'outputs'/diagnosis_path).read_text())
        assert diagnosis['material'] == 'SourceColoredFringe'
        examples = []
        for witness in diagnosis['normal_comparison']['worst_violations']:
            vertex = witness['target_vertex']; source_id = witness['closest_source_vertex']
            index_match = int(corner_map[vertex]) == source_id
            row = {'target_vertex': vertex, 'helper_vertex': source_id, 'native_corner_index_agrees': index_match}
            if index_match:
                triangle = points[faces[vertex//3]].astype(np.float64)
                offset = vertex % 3
                geom = normalized(np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0]))
                edge = normalized(triangle[(offset+1)%3]-triangle[offset])
                ref = normalized(edge-geom*np.dot(edge,geom))
                intended = normalized(normal[source_id].astype(np.float64))
                projected = normalized(intended-geom*np.dot(geom,intended))
                cosine = float(np.dot(ref,projected))
                # Isolate beta snapping only; not a full C++ short2/libm emulation.
                predicted = normalized(geom*np.dot(geom,intended)+ref*np.sqrt(max(0.,1-float(np.dot(geom,intended))**2)))
                predicted = predicted[[0,2,1]]; predicted[2] *= -1
                actual = np.asarray(witness['target_normal'])
                row.update({'projected_beta_cosine':cosine, 'snap_threshold':float(np.float32(1)-np.float32(1e-4)),
                    'snapping_condition': cosine >= float(np.float32(1)-np.float32(1e-4)),
                    'projected_beta_degrees':float(np.degrees(np.arccos(np.clip(cosine,-1,1)))),
                    'snap_only_prediction_gltf_normal':predicted.tolist(),
                    'snap_only_vs_export_component_max':float(np.max(abs(predicted-actual))),
                    'original_export_component_error':witness['component_error']})
            examples.append(row)
        rows.append({'subject':name,'npz_path':str(path.relative_to(ROOT)),'npz_sha256':sha(path),
            'model_sha256_from_sealed_diagnosis': ('af387843524ba560f2b8f0dd9d5c75f5d923e8f2e8f0ed7db62d60f1dcdf9026' if name=='panda' else 'ac8c0d905564861cefe75a08c64bdc8d659242f34b08a269f4aeeb303f304c92'),
            'diagnosis_path':'outputs/'+diagnosis_path,'diagnosis_sha256':sha(ROOT/'outputs'/diagnosis_path),
            'helper_normals':len(normal),'export_round4_only_max_component_change':float(np.max(abs(rounding-normal))),
            'export_round4_only_above_original_normal_limit':int(np.count_nonzero(np.max(abs(rounding-normal),axis=1)>.0002)),
            'normal_examples':examples})
    result = {'status':'CPU_only_cause_witness_replay','installed_weight_method_sha256':sha(extract),
        'weight_replay':weights,'normal_replay':rows,'accepted':False,'native_calls':0,
        'limitations':['Weight branch replays actual installed Python on saved actual witness rows.',
            'Normal beta-snap experiment uses version-matched public C++ formula and actual triangle/helper data, not native instrumentation or full short2 emulation.',
            'No final attribute correction, pose test, threshold relaxation, or appearance acceptance.']}
    with (HERE/'witness-replay.json').open('x') as handle:
        json.dump(result,handle,indent=2,sort_keys=True,allow_nan=False);handle.write('\n')
    print(json.dumps({'weight_witnesses':len(weights),'normal_subjects':len(rows),
        'weight_all_float32_exact':all(r['predicted_vs_actual_float32_bytes_equal'] for r in weights),
        'normal_beta_snap_matches':[{'subject':r['subject'],'conditions':[w.get('snapping_condition') for w in r['normal_examples']]} for r in rows]}))


if __name__ == '__main__':
    main()
