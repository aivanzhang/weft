import argparse
import json
from pathlib import Path

from weft.compare import compare_runs, save_comparison
from weft.env import RollingEnv
from weft.experiments import run_experiment
from weft.meta import status
from weft.pipeline import run, verify_artifacts
from weft.source import fetch_sample
from weft.world import World


def main():
    parser = argparse.ArgumentParser(description="Weft: bounded video-to-environment evidence pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch-sample", help="Download the pinned CC0 video once")
    fetch.add_argument("--out", default="data/rolling-ball.mp4")
    execute = commands.add_parser("run", help="Run A–E locally; output directory must be new")
    execute.add_argument("--video", default="data/rolling-ball.mp4")
    execute.add_argument("--out", required=True)
    execute.add_argument("--episodes", type=int, default=15000)
    execute.add_argument("--eval-episodes", type=int, default=300)
    execute.add_argument("--seed", type=int, default=1729)
    execute.add_argument("--eval-seed", type=int, default=938401)
    execute.add_argument("--no-repair", action="store_true")
    execute.add_argument("--provider", choices=("local", "meta"), default="local")
    execute.add_argument("--max-meta-cost-usd", type=float)
    experiment = commands.add_parser("experiment", help="Run a hash-pinned experiment config")
    experiment.add_argument("config")
    experiment.add_argument("--out", required=True)
    compare = commands.add_parser("compare", help="Compare verified runs and disclose incompatible benchmarks")
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    compare.add_argument("--out", help="Optional new JSON file outside both runs")
    verify = commands.add_parser("verify", help="Verify the complete artifact manifest")
    verify.add_argument("directory")
    commands.add_parser("meta-status", help="Report credential presence without making a request")
    interact = commands.add_parser("interact", help="Step the Python simulator manually")
    interact.add_argument("--world", required=True)
    args = parser.parse_args()
    try:
        if args.command == "fetch-sample":
            result = {"path": str(fetch_sample(args.out))}
        elif args.command == "run":
            result = run(args.video, args.out, args.episodes, args.eval_episodes, args.seed,
                         args.provider, args.max_meta_cost_usd, eval_seed=args.eval_seed,
                         repair_enabled=not args.no_repair)
        elif args.command == "experiment":
            result = run_experiment(args.config, args.out)
        elif args.command == "compare":
            result = compare_runs(args.baseline, args.candidate)
            if args.out:
                save_comparison(args.out, result)
        elif args.command == "verify":
            result = verify_artifacts(args.directory)
        elif args.command == "meta-status":
            result = status()
        else:
            env = RollingEnv(World.from_dict(json.loads(Path(args.world).read_text())))
            print("l = left, c = coast, r = right, reset = new trial, q = quit")
            print(env.reset(seed=0))
            while True:
                command = input("> ").strip()
                if command == "q":
                    break
                if command == "reset":
                    print(env.reset(seed=0)); continue
                if command not in {"l", "c", "r"}:
                    print("Use l, c, r, reset, or q"); continue
                try:
                    observation, reward, terminated, truncated, info = env.step({"l": 0, "c": 1, "r": 2}[command])
                    print(json.dumps({"observation": observation.tolist(), "reward": reward,
                                      "terminated": terminated, "truncated": truncated, **info}))
                except RuntimeError as error:
                    print(error)
            result = {"status": "closed"}
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, RuntimeError, TimeoutError, OSError) as error:
        parser.exit(1, f"weft: {error}\n")


if __name__ == "__main__":
    main()
