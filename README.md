# Weft

Generate executable training environments from evidence. Test them against observed outcomes. Repair the environment when the evidence disagrees.

Weft generates and repairs reinforcement-learning environments using real-world evidence. It is a Loom-adjacent project with its own identity. This **separate hackathon repository** is adjacent to, not a fork of, the existing `../loom` data-ingestion product.

The original application used the name Loom. Historical application text and source snapshots retain that name; the project is now called **Weft**.

## Current state

Research and reference archive only. No simulator, agent trainer, repair engine, deployed service, or submission has been built here. No existing Loom code, data, credentials, or Git history was copied.

Intended entry: **Indie Developer / Push the Frontier** at the Meta Global AI Developer Hackathon. The proposed demonstration is one bounded tabletop sorting task, not a general robotics platform.

## Read first

1. [Hackathon reference hub](hackathon/README.md)
2. [What the judges want and how Loom fits](hackathon/analysis/fit-and-strategy.md)
3. [Requirements, dates, and unresolved conflicts](hackathon/analysis/requirements.md)
4. [Proposed demo and evaluation design](hackathon/analysis/demo-plan.md)
5. [Prior work versus hackathon work](hackathon/analysis/originality.md)
6. [Application context](hackathon/analysis/application-context.md)

The submission deadline is **November 4, 2026, at 6:00 p.m. PST**. Source snapshots retain the organizers' original timezone wording; planning documents use America/Los_Angeles.

## Repository boundaries

- Private GitHub repository: `kintsugi-software/weft`. No public release, deployment, or hackathon submission.
- `hackathon/sources/`: source snapshots and explicit unavailable-content records.
- `hackathon/analysis/`: interpretation and proposed project direction, not official rules.
- `scripts/`: deterministic read-only archival tooling.
- `.local/`: ignored raw HTTP responses and supporting artifacts; never publish blindly.
- Source documents retain their original ownership and terms. A public-code license is not selected yet; do not apply a future project license to third-party reference material.

The event requires a public repository at submission. This repository has **not** been made public. Publication and submission remain separate future actions.
