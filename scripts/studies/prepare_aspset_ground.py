"""Package frozen manual ground labels without reading human/reference poses."""

import argparse
import json
from pathlib import Path

import numpy as np

from weft.io import canonical_hash, digest, write_json
from weft.pipeline import verify_artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data/aspset/04ac-0003-01')
    parser.add_argument('--markers', default='artifacts/aspset-ground-markers-01')
    parser.add_argument('--media-inspection', default='.local/aspset-media-inspection-01')
    parser.add_argument('--images', default='artifacts/aspset-person-region-01')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    source, markers, media, images, out = map(Path, (args.source, args.markers, args.media_inspection, args.images, args.out))
    if out.exists() or not out.parent.is_dir():
        raise ValueError('Choose a new output directory under an existing parent')
    source_pin = 'f7a57a345529fff170977b6f9fc1db04bc66af7a5241213013f90663875c46e1'
    marker_pin = '4ebee242abe4b9ccae7f287ac503adde5802dee34f90b8d5c2442399730fc19c'
    if digest(source/'manifest.json') != source_pin or verify_artifacts(markers)['manifest_sha256'] != marker_pin:
        raise ValueError('Source or frozen manual-marker package changed')
    manifest = json.loads((source/'manifest.json').read_text())
    plan = json.loads((markers/'plan.json').read_text())
    nominal = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
    cameras, views, inputs = [], [], {}
    for name in ('left', 'mid'):
        camera_path, image_path, clock_path = source/f'camera-{name}.json', images/name/'background.jpg', media/name/'report.json'
        camera_hash, image_hash = digest(camera_path), digest(image_path)
        if camera_hash != manifest['files'][camera_path.name]['sha256']:
            raise ValueError('Publisher camera bytes changed')
        image_key = f'artifacts/aspset-person-region-01/{name}/background.jpg'
        camera_key = f'data/aspset/04ac-0003-01/camera-{name}.json'
        if plan['inputs'].get(image_key) != image_hash or plan['inputs'].get(camera_key) != camera_hash:
            raise ValueError('Source images/calibrations differ from the frozen marker labels')
        clock = json.loads(clock_path.read_text())
        source_hash = manifest['files'][f'{name}.mkv']['sha256']
        if (clock['source_sha256'] != source_hash or clock['decoded_frames'] != 188
                or clock['decoded_to_eof'] is not True or len(clock['frames']) != 188
                or clock['native_minus_decoder_seconds'] != 0
                or clock['metadata'] != {'frame_count': 188., 'fps': 50., 'width': 3840., 'height': 2160.}):
            raise ValueError('Saved original-source clock qualification does not match this source')
        raw = json.loads(camera_path.read_text())
        intrinsic, extrinsic = np.array(raw['intrinsic_matrix']).reshape(3, 4), np.array(raw['extrinsic_matrix']).reshape(4, 4)
        if not np.array_equal(intrinsic[:, 3], [0, 0, 0]) or not np.array_equal(extrinsic[3], [0, 0, 0, 1]):
            raise ValueError('Unsupported publisher camera convention')
        camera = {'schema_version': 1, 'source_sha256': source_hash, 'width': 3840, 'height': 2160, 'fps': 50., 'frame_count': 188,
            'world_axes': 'right_handed_z_up', 'units': 'm', 'quality': 'reconstructed',
            'provenance': 'Publisher pinhole extrinsics, declared mm-to-m scale. Nominal x=camera right, y=forward, z=image up; not measured gravity or ground. Original six-decimal rotations retained.',
            'K': intrinsic[:, :3].tolist(), 'radial_k1': 0.,
            'frames': [{'frame': i, 'time_s': i/50., 'R': (extrinsic[:3, :3] @ nominal.T).tolist(),
                        't': (extrinsic[:3, 3]*.001).tolist()} for i in range(188)]}
        cameras.append(camera)
        views.append({'name': name, 'source_sha256': source_hash, 'calibration_sha256': canonical_hash(camera),
            'image_sha256': image_hash, 'frame': 0, 'time_s': 0.,
            'provenance': 'Original-raster first-frame JPEG retained in the sealed explicit-person-region study; source association from that study and saved native clock audit. JPEG is lossy.'})
        inputs.update({str(camera_path): camera_hash, str(image_path): image_hash, str(clock_path): digest(clock_path)})
    annotations = {'schema_version': 1, 'profile': 'stereo-ground-annotations-v1', 'views': views,
        'world_provenance': 'Publisher shared camera coordinates and metre scale; manually matched marker bases assumed static and on one local plane. Left far marker is partly hidden. No human pose, C3D reference, floor score or contact input. Gravity alignment remains an authored physical assumption.',
        'points': [{'id': row['id'], 'partition': 'construction', 'pixels': row['pixels'], 'uncertainty_px': row['uncertainty_px'],
                    'provenance': row['note']+'. Pixel ranges are reviewer sensitivity assumptions, not calibrated error bounds.'} for row in plan['labels']]}
    out.mkdir()
    for index, camera in enumerate(cameras):
        write_json(out/f'camera-{index}.json', camera)
    write_json(out/'annotations.json', annotations)
    write_json(out/'source-binding.json', {'source_manifest_sha256': source_pin, 'marker_manifest_sha256': marker_pin,
        'inputs': inputs, 'nominal_camera_world_rotation': nominal.tolist(), 'source_video_decodes': 0, 'pose_inputs': [], 'reference_inputs': []})
    (out/'worker.py').write_bytes(Path(__file__).read_bytes())
    write_json(out/'manifest.json', {'schema': 1, 'files': {p.name: digest(p) for p in sorted(out.iterdir())}})
    print(json.dumps(verify_artifacts(out)))


if __name__ == '__main__':
    main()
