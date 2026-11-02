"""Bind the saved ASPset right view to the existing, frozen manual-ground frame."""

import argparse
import json
from pathlib import Path

import numpy as np

from weft.camera_pose import validate_camera
from weft.io import digest, write_json
from weft.pipeline import verify_artifacts


SOURCE = '02cb72e5e893fc37cb8ac823ee7d7f4f04b4b65b3008796ddfc6a0ea08d9c67b'
IMAGES = 'bce514c9790825dca3961f7379202e5130f7b6f2294d520fface4aac06170a21'
GROUND = 'bdf11ee7ddd7b74fce3420c6ee36902acc35b9c57d15117536671bf6745f23f2'


def verify_source(root, expected):
    if digest(root/'manifest.json') != expected:
        raise ValueError('Frozen source manifest changed')
    manifest = json.loads((root/'manifest.json').read_text())
    for name, row in manifest['files'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != row['sha256'] or path.stat().st_size != row['bytes']:
            raise ValueError('Source package member changed: '+name)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/aspset/04ac-0003-right-01'))
    parser.add_argument('--images', type=Path, default=Path('artifacts/aspset-third-person-region-01'))
    parser.add_argument('--ground', type=Path, default=Path('artifacts/aspset-manual-ground-01'))
    parser.add_argument('--clock', type=Path, default=Path('.local/aspset-third-media-inspection-01/right/report.json'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or not args.out.parent.is_dir():
        raise ValueError('Choose a new output under an existing parent')
    source = verify_source(args.source, SOURCE)
    verify_source(args.images, IMAGES)
    if verify_artifacts(args.ground)['manifest_sha256'] != GROUND:
        raise ValueError('Frozen ground changed')
    raw = json.loads((args.source/'camera-right.json').read_text())
    clock = json.loads(args.clock.read_text())
    evidence = json.loads((args.images/'right/evidence.json').read_text())
    source_hash = source['files']['right.mkv']['sha256']
    if (clock['source_sha256'] != source_hash or clock['decoded_frames'] != 188 or not clock['decoded_to_eof']
            or len(clock['frames']) != 188 or clock['native_minus_decoder_seconds'] != 0
            or clock['metadata'] != {'frame_count': 188., 'fps': 50., 'width': 3840., 'height': 2160.}
            or any(row['frame'] != i or row['time_s'] != i/50 or abs(row['decoder_time_s']-i/50) > 1e-7
                   for i, row in enumerate(clock['frames']))):
        raise ValueError('Right-view clock differs from its complete original-source qualification')
    nominal = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
    transform = json.loads((args.ground/'estimate.json').read_text())['world_transform']
    basis = np.array(transform['R']) @ nominal
    intrinsic = np.array(raw['intrinsic_matrix']).reshape(3, 4)
    extrinsic = np.array(raw['extrinsic_matrix']).reshape(4, 4)
    if not np.array_equal(intrinsic[:, 3], [0, 0, 0]) or not np.array_equal(extrinsic[3], [0, 0, 0, 1]):
        raise ValueError('Unsupported publisher camera convention')
    camera = {'schema_version': 1, 'source_sha256': source_hash, 'width': 3840, 'height': 2160,
        'fps': 50., 'frame_count': 188, 'world_axes': 'right_handed_z_up', 'units': 'm', 'quality': 'reconstructed',
        'provenance': 'Publisher right camera, declared mm-to-m scale, original six-decimal rotation retained. '
                      'Same frozen manual-ground basis as left/mid; no body, contact, reference pose or score fitting. '
                      'Scale, shared coordinates and gravity remain conditional, not independently measured.',
        'K': intrinsic[:, :3].tolist(), 'radial_k1': 0.,
        'frames': [{'frame': i, 'time_s': i/50, 'R': (extrinsic[:3, :3] @ basis.T).tolist(),
                    't': (extrinsic[:3, 3]*.001).tolist()} for i in range(188)]}
    validate_camera(camera, evidence)
    args.out.mkdir()
    for name, path in {'evidence.json': args.images/'right/evidence.json', 'source-clock.json': args.clock,
                       'publisher-camera.json': args.source/'camera-right.json'}.items():
        (args.out/name).write_bytes(path.read_bytes())
    write_json(args.out/'camera.json', camera)
    write_json(args.out/'view.json', {'schema_version': 1, 'name': 'right', 'evidence': 'evidence.json',
        'camera': 'camera.json', 'clock_provenance': 'Matching ASPset subject/clip, complete 188-frame native and decoder '
        'clocks at 50 Hz and zero origin offset; physical synchronization is publisher-supplied, not independently measured.'})
    write_json(args.out/'provenance.json', {'source_manifest_sha256': SOURCE, 'images_manifest_sha256': IMAGES,
        'ground_manifest_sha256': GROUND, 'clock_sha256': digest(args.clock),
        'publisher_world_to_ground_world_rotation': basis.tolist(), 'translation_scale_mm_to_m': .001,
        'existing_ground_or_cameras_modified': False, 'reference_inputs': [], 'source_video_decodes': 0,
        'inference_calls': 0, 'api_calls': 0})
    (args.out/'worker.py').write_bytes(Path(__file__).read_bytes())
    write_json(args.out/'manifest.json', {'schema': 1, 'files': {p.name: digest(p) for p in sorted(args.out.iterdir())}})
    print(json.dumps(verify_artifacts(args.out)))


if __name__ == '__main__':
    main()
