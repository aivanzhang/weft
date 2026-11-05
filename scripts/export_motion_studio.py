"""Make read-only studio exhibits for the existing motion-showcase exports."""
import argparse
import json
import math
from pathlib import Path

from export_motion_showcase import EXAMPLES, load_example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hosted-catalog', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError('Choose an empty output directory')
    for identity, *_ in EXAMPLES:
        frames, _, _, provenance = load_example(identity, args.hosted_catalog)
        receipt = json.loads((root / 'assets/demo/motion-showcase' / identity / 'receipt.json').read_text())
        if receipt['reference_sha256'] != provenance['reference_sha256']:
            raise ValueError('Presentation and reference evidence differ')
        start = frames[0]['source_time_s']
        end = start + receipt['duration_s']
        selected = [f for f in frames if f['source_time_s'] <= end]
        # Retain one bracketing sample so the display reaches the same terminal time as the video.
        following = next((f for f in frames if f['source_time_s'] > end), None)
        if following is not None:
            selected.append(following)
        stride = max(1, math.ceil(len(selected) / 360))
        sampled = selected[::stride]
        if sampled[-1] is not selected[-1]:
            sampled.append(selected[-1])
        shoulders = next(p for p in sampled[0]['human']['segments'] if p['id'] == 'shoulders')
        axis = [b-a for a, b in zip(shoulders['a'], shoulders['b'])]
        model = {'segment_ids': [part['id'] for part in sampled[0]['human']['segments']], 'kind': 'skeleton', 'playback_kind': 'kinematic reference', 'duration': receipt['duration_s'],
                 'camera_yaw': round(-math.degrees(math.atan2(axis[1], axis[0]))),
                 'frames': [{'time': round(f['source_time_s'] - start, 6),
                             'segments': [[[round(v, 5) for v in part[k]] for k in ('a', 'b')]
                                          for part in f['human']['segments']]} for f in sampled],
                 'source_sha256': provenance['reference_sha256'], 'source_frames': len(selected),
                 'preview_stride': stride, 'coordinate_precision_decimals': 5,
                 'scope': 'Saved kinematic reference with linear display interpolation. Not a new simulation, learned policy or validated biomechanics.'}
        evidence = {'profile': 'weft-studio-reference-evidence-v1', **receipt,
                    'scope': model['scope'], 'omissions': ['Equipment geometry', 'Contact forces', 'Unassisted balance']}
        def asset(name, label):
            return {'name': name, 'label': label, 'kind': 'json'}
        manifest = {'model': identity + '-model.json', 'model_label': 'Reconstructed skeleton', 'steps': [
            {'id': 'intake', 'title': 'Start with the observed movement.',
             'description': 'Keep a supported window of the original footage. The initial input and the recording above use the same movement interval.',
             'assets': [asset(identity + '-evidence.json', 'Source window and evidence receipt')]},
            {'id': 'world', 'title': 'Reconstruct the body in motion.',
             'description': 'Save the estimated body segments and their positions over time. Inspect the skeleton above from another angle; equipment and contact forces are outside this reconstruction.',
             'assets': [asset(identity + '-model.json', 'Skeleton and motion samples')]},
            {'id': 'replay', 'title': 'Make the motion inspectable.',
             'description': 'Render the saved motion on a neutral body. The final video uses a fixed camera and interpolated display frames. It is a kinematic reference replay, not a learned controller or a new physics result.',
             'assets': [asset(identity + '-evidence.json', 'Render settings and output hashes')]},
        ]}
        for name, data in [(identity, manifest), (identity + '-model', model), (identity + '-evidence', evidence)]:
            (args.out / (name + '.json')).write_text(json.dumps(data, separators=(',', ':'), ensure_ascii=False) + '\n')
    print(json.dumps({'files': len(list(args.out.iterdir())), 'bytes': sum(p.stat().st_size for p in args.out.iterdir())}))


if __name__ == '__main__':
    main()
