"""Execute an explicit bounded qualification plan; no server, training or video decoding."""

import argparse
from pathlib import Path
import sys

from weft.runtime_qualification import _json, qualify_runtimes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="JSON list of pinned sealed-input cases")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--max-output-mib", type=int, default=64)
    args = parser.parse_args()
    with args.cases.open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        parser.error("Case plan exceeds 1 MiB")
    cases = _json(raw)
    if not isinstance(cases, list):
        parser.error("Case plan must be a JSON list")
    for case in cases:
        if isinstance(case, dict) and isinstance(case.get("input"), str):
            case["input"] = str(args.cases.parent / case["input"])
    result = qualify_runtimes(cases, args.out, max_seconds=args.max_seconds, max_steps=args.max_steps,
                              max_output_bytes=args.max_output_mib * 1024 * 1024)
    print(f"{result['status']}: {result['operationally_qualified']}/{len(result['cases'])} operationally qualified")
    print(f"Report: {Path(result['directory']) / 'report.json'}")
    print("Task outcomes remain separate. Physical validation: unverified. No learned-policy evaluation.")
    return 0 if result["status"] == "completed" and result["operationally_qualified"] == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
