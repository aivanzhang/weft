"""Run one explicit conditional silhouette proposal and independent clearance comparison."""

import argparse
import json

from weft.silhouette_pipeline import prepare_silhouette_world


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--masks-run", required=True)
    parser.add_argument("--task")
    parser.add_argument("--object-id", action="append", required=True)
    parser.add_argument("--source-frame", type=int, required=True)
    parser.add_argument("--conditional-exterior", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-seconds", type=float, default=60)
    args = parser.parse_args()
    result = prepare_silhouette_world(args.run, args.masks_run, args.out, task_path=args.task,
        object_ids=args.object_id, source_frame=args.source_frame,
        conditional_exterior=args.conditional_exterior, max_seconds=args.max_seconds)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
