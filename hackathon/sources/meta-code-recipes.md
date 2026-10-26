# Source snapshot: meta-code-recipes

- Source: https://github.com/meta-models/meta-model-cookbook/tree/main/04_muse_code
- Retrieved: 2026-10-26 03:00:24 PM PDT
- Method: read-only HTTP GET; HTML converted to Markdown.
- Status: captured
- SHA-256 of source bytes: `3e92015ca39956fda3d109d9a07a2427e8d040630ede0d0dcf7b07a66ae6158f`

> Source snapshot, not Loom-authored guidance. Original source wording and timezone labels are retained for fidelity. Use the Los Angeles schedule in the analysis for planning. Navigation, scripts, styling, and interactive behavior are not preserved. Source rights remain with their owners.

---

1.  [meta-models](https://github.com/meta-models)
2.  [meta-model-cookbook](https://github.com/meta-models/meta-model-cookbook)

## Repository navigation

- 
- [Code](https://github.com/meta-models/meta-model-cookbook)
- [Issues17 (17)](https://github.com/meta-models/meta-model-cookbook/issues)
- [Pull requests26 (26)](https://github.com/meta-models/meta-model-cookbook/pulls)
- [Actions](https://github.com/meta-models/meta-model-cookbook/actions)
- [Projects](https://github.com/meta-models/meta-model-cookbook/projects)
- [Security and quality](https://github.com/meta-models/meta-model-cookbook/security)
- [Insights](https://github.com/meta-models/meta-model-cookbook/pulse)

More items

[](https://github.com/meta-models/meta-model-cookbook)

## FilesExpand file tree

main

## Breadcrumbs

1.  [meta-model-cookbook](https://github.com/meta-models/meta-model-cookbook/tree/main)

/

# 04_muse_code

/

Copy path

## Directory actions

## More options

More options

## Directory actions

## More options

More options

## Latest commit

 

## History

[History](https://github.com/meta-models/meta-model-cookbook/commits/main/04_muse_code)

[](https://github.com/meta-models/meta-model-cookbook/commits/main/04_muse_code)History

main

## Breadcrumbs

1.  [meta-model-cookbook](https://github.com/meta-models/meta-model-cookbook/tree/main)

/

# 04_muse_code

/

Copy path

Top

## Folders and files

[TABLE]

## [README.md](https://github.com/meta-models/meta-model-cookbook/tree/main/04_muse_code#readme)

Outline

# Muse Code

[](https://github.com/meta-models/meta-model-cookbook/tree/main/04_muse_code#muse-code)

Recipes for building with [Muse Code](https://dev.meta.ai/docs/cookbook#building-with-muse-code) when you need sessions you can audit, safety that fails closed, and an agent team you can steer. Each recipe walks a task end to end and finishes with a proof point you can replay.

## Recipes

[](https://github.com/meta-models/meta-model-cookbook/tree/main/04_muse_code#recipes)

| \# | Recipe | What it does |
|----|----|----|
| [01](https://github.com/meta-models/meta-model-cookbook/blob/main/04_muse_code/01_append_only_audit_log) | Append-only audit log and crash-safe resume | Export the append-only session log and walk the exact sequence of actions and approvals as replayable evidence, then kill the harness mid-task and resume where it stopped with no duplicate side effects. |
| [02](https://github.com/meta-models/meta-model-cookbook/blob/main/04_muse_code/02_deterministic_replay) | Deterministic Replay in CI | Turn recorded Muse Code events into a focused golden fixture and replay its model-context projection as a merge-blocking check. |
| [03](https://github.com/meta-models/meta-model-cookbook/blob/main/04_muse_code/03_staged_approvals) | Staged approvals for compound shell commands | Decompose a compound shell command into stages, auto-clear the safe ones, and hold the dangerous or unparseable ones for review. Trust is scoped per action and per workspace, and fails closed. |
| [04](https://github.com/meta-models/meta-model-cookbook/blob/main/04_muse_code/04_contained_execution) | Contained execution you can prove | Run shell commands inside an OS-level sandbox that live-probes its own enforcement and refuses to run when containment isn't enforcing. |
| [05](https://github.com/meta-models/meta-model-cookbook/blob/main/04_muse_code/05_immutable_guardrails) | The agent can't rewrite its own rules | Ask the agent to edit the guardrail file it runs under; the write stops for human review with no standing grant on offer, and a shell write to the same path fails read-only at the sandbox. |
| [06](https://github.com/meta-models/meta-model-cookbook/blob/main/04_muse_code/06_subagent_fanout) | Subagent fanout across isolated worktrees | Distribute one job across parallel subagents, each in its own git worktree, steered or stopped from a single control point. |
| [07](https://github.com/meta-models/meta-model-cookbook/blob/main/04_muse_code/07_goal_tracking) | Goal tracking with a pinned objective | Pin the objective once with `/goal`; the harness carries it across turns and audits the work against the acceptance checks before the goal can close. |
| [08](https://github.com/meta-models/meta-model-cookbook/blob/main/04_muse_code/08_bundled_skills) | Bundled skills: plan, pressure-test, and build | Drive the built-in `/plan`, `/grilling`, `/grill-with-docs`, and `/taste` skills end to end: plan a change, pressure-test it, record the decisions, and build UI that doesn't look AI-made. |
| [09](https://github.com/meta-models/meta-model-cookbook/blob/main/04_muse_code/09_loop_and_cron) | Loop and cron: scheduled agent work | Schedule recurring or one-time agent work in natural language, then view, change, or cancel it from the same chat surface. |
| [10](https://github.com/meta-models/meta-model-cookbook/blob/main/04_muse_code/10_side_chats) | Multi-turn side chats | Branch a multi-turn side conversation off a busy main thread with `/side` (or its alias `/btw`), isolated from the main transcript. |
| [11](https://github.com/meta-models/meta-model-cookbook/blob/main/04_muse_code/11_mcp_security_tooling) | Muse Code for Security Research | Wire Burp Suite, Meta’s bug bounty research toolkit, headless Ghidra and LLDB into Muse Code over MCP, then point it at PortSwigger’s deliberately vulnerable demo site, a real bounty programme, and a real CVE buried in a stripped binary. |
