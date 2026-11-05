"""Prepare presentation media from saved videos and admitted character assets."""
import argparse
import json
from pathlib import Path

from weft.portal_studio_media import prepare_run, poster


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path)
    parser.add_argument('--run')
    parser.add_argument('--all-runs', action='store_true')
    parser.add_argument('--public', action='store_true')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if args.public:
        seeds = json.loads((project / 'src/weft/assets/demo_examples.json').read_text())
        output = project / 'assets/demo/studio'
        for seed in seeds:
            for kind, key in [('source', 'source_video'), ('result', 'video')]:
                poster(project / seed[key], output / f'{seed["id"]}-{kind}-poster.jpg')
        print(json.dumps({'public_posters': len(seeds) * 2}))
    if args.run or args.all_runs:
        if not args.root:
            parser.error('--root is required for saved runs')
        identities = [args.run] if args.run else [p.parent.name for p in sorted((args.root / 'demo').glob('run-*/record.json'))[:64]]
        for identity in identities:
            result = prepare_run(args.root, identity)
            print(json.dumps({'run': identity, 'stages': list(result.get('steps', {})), 'notes': result.get('notes', [])}), flush=True)
    if not args.public and not args.run and not args.all_runs:
        parser.error('Choose --public or --root with --run')


if __name__ == '__main__':
    main()
