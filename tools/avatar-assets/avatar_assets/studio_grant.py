"""Record explicit user approvals as the targeted allowances one studio job needs.

The studio never creates or raises an allowance itself; these are the owner-side steps
that turn a verbatim user approval into the canonical grants its gates check: an OpenAI
dollar grant bound to the job's immutable pipeline request (planning and agent calls),
and a Meshy credit grant bound to the prepared generation plan. Both are immutable per
job, refuse once conflicting records exist, and never touch the base budgets.
"""

import argparse
import json
from pathlib import Path

from avatar_assets.agent_backend import request_digest, server_settings
from avatar_assets.meshy_access import Ledger, digest
from avatar_assets.openai_runner import OpenAILedger, accounting_snapshot, budget_snapshot
from avatar_assets.safety import require

ROOT = Path(__file__).resolve().parents[1]


def grant_meshy_by_output(output_dir, *, additional_credits, user_reply, project=None):
    """Authorize the generation plan saved under one studio output directory."""
    output_dir = Path(output_dir).absolute()
    require(output_dir.is_dir() and not output_dir.is_symlink(), "Studio output directory required")
    generation = output_dir / "generation"
    plan_path = generation / "plan.json"
    require(plan_path.is_file(), "This output has no prepared generation plan to authorize")
    plan = json.loads(plan_path.read_text())
    require(isinstance(plan, dict) and isinstance(plan.get("id"), str), "Generation plan is malformed")
    require(type(additional_credits) is int and 1 <= additional_credits <= plan.get("max_credits", 0),
            "Approval credits must fit the plan's immutable ceiling")
    require(isinstance(user_reply, str) and 0 < len(user_reply) <= 4000,
            "Preserve the user's verbatim approval text")
    ledger = Ledger(project=Path(project) if project else ROOT.parent.parent)
    approval = {"decision": "approved", "job_id": plan["id"], "plan_digest": digest(plan),
                "additional_credits": additional_credits, "user_reply": user_reply}
    summary = ledger.authorize_job(plan, generation, approval)
    return {"granted_credits": additional_credits, "job_id": plan["id"], "ledger_summary": summary}


def grant_openai_by_job(job_id, *, max_microdollars, user_reply):
    """Authorize the bounded agent calls for one saved studio pipeline request.

    The grant binds to the exact pipeline-request bytes and the configured call count;
    its dollar ceiling is clipped to the project's unallocated allowance so no approval
    can oversubscribe the administrator budget.
    """
    require(isinstance(job_id, str) and 1 <= len(job_id) <= 64, "Studio job id required")
    request_path = ROOT / ".local/studio/jobs" / job_id / "pipeline-request.json"
    require(request_path.is_file() and not request_path.is_symlink(),
            "This job has no saved pipeline request to bind")
    require(isinstance(user_reply, str) and 0 < len(user_reply) <= 4000,
            "Preserve the user's verbatim approval text")
    bound_digest = request_digest(request_path)
    settings = server_settings(ROOT)
    max_calls = settings["max_calls"]
    ledger = OpenAILedger()
    budget = budget_snapshot(ledger.db)
    require(budget is not None, "Configure the administrator OpenAI allowance first")
    from avatar_assets.openai_grant_closure import allocated_grants
    headroom = budget["effective_ceiling_microdollars"] - allocated_grants(ledger.db)
    require(type(max_microdollars) is int and 1 <= max_microdollars, "Positive dollar ceiling required")
    amount = min(max_microdollars, headroom)
    require(amount >= 1, "The project allowance is fully allocated; close or retire a grant first")
    approval = {"decision": "approved", "job_id": job_id, "request_digest": bound_digest,
                "max_calls": max_calls, "max_microdollars": amount, "user_reply": user_reply}
    ledger.authorize_job(job_id, bound_digest, max_calls, amount, approval)
    exposure = accounting_snapshot(ledger.db)["project_exposure_microdollars"]
    return {"granted_microdollars": amount, "max_calls": max_calls, "job_id": job_id,
            "project_exposure_microdollars": exposure}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    meshy = commands.add_parser("meshy", help="Grant Meshy credits to one prepared generation plan")
    meshy.add_argument("output", help="Studio output directory containing generation/plan.json")
    meshy.add_argument("--credits", type=int, required=True)
    meshy.add_argument("--reply", required=True)
    openai = commands.add_parser("openai", help="Grant bounded agent dollars to one saved request")
    openai.add_argument("job")
    openai.add_argument("--microdollars", type=int, required=True)
    openai.add_argument("--reply", required=True)
    args = parser.parse_args()
    if args.command == "meshy":
        result = grant_meshy_by_output(args.output, additional_credits=args.credits,
                                       user_reply=args.reply)
    else:
        result = grant_openai_by_job(args.job, max_microdollars=args.microdollars,
                                     user_reply=args.reply)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
