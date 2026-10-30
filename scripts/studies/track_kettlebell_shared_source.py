"""One bounded source-linked SAM request; preserve complete input and mapping evidence."""

import argparse
from pathlib import Path
import shutil

from weft.io import digest, local_time, write_json
from weft.pipeline import verify_artifacts
from weft.segmentation_lab import track_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    prepared = root / "artifacts/kettlebell-tracking-input-02"
    video = prepared / "kettlebell-lossless.mp4"
    verified = verify_artifacts(prepared)
    expected = "34e74d9ab854a474ba647395f232e61dfde5cfa166bb449fd3057084c6dd5416"
    if digest(video) != expected:
        raise ValueError("Prepared derivative identity changed")
    if args.out.exists():
        raise FileExistsError("Use a new tracking study output")
    args.out.mkdir(parents=True)
    shutil.copyfile(__file__, args.out / Path(__file__).name)
    boxes = [[1, 8, 215, 218, 175, 195], [2, 8, 223, 369, 47, 45], [3, 8, 125, 364, 46, 51]]
    write_json(args.out / "pre-run.json", {
        "created_at": local_time(), "hypothesis": "Track the person and both kettlebells from one source without losing opening frames.",
        "source_mapping_bundle": str(prepared), "source_mapping_verification": verified,
        "source_sha256": "4664b16bab5fbefd66f3dc9116ced8dfeb6f3b979635403b715bbc5848b0afbc",
        "submitted_sha256": expected, "frames": 257, "dimensions": [640, 480], "boxes": boxes,
        "object_labels": {"1": "person", "2": "working kettlebell", "3": "idle kettlebell"},
        "prompt_scope": "Manually inspected frame-8 initialization boxes, not independent labels or recovered geometry.",
        "opening_frames": "Frames 0–7 retained as pre-initialization; no backward propagation assumed.",
        "request_count": 1 if args.execute else 0, "automatic_retries": 0,
        "application_request_allowance_usd": .16, "conservative_estimate_usd": .1542,
        "shared_project_budget_usd": 5, "provider_hard_cap": False,
        "pricing_url": "https://dev.meta.ai/docs/pricing-rate-limits", "pricing_checked_on": "2026-10-29",
        "cost_assumption": "257 frames × 3 objects × $0.20/1000 as conservative envelope; provider usage determines settlement.",
        "deadline_seconds": 120, "scope": "Source tracking proposals only; no physical validation or model promotion."})
    report = track_video(video, args.out / "tracking", boxes, execute=args.execute,
                         max_cost_usd=.16 if args.execute else None, project_root=root)
    print({"status": report["status"], "out": str(args.out), "execute": args.execute})


if __name__ == "__main__":
    main()
