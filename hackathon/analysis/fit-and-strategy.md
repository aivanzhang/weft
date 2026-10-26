# What the hackathon is actually looking for

Analysis prepared October 26, 2026. This document is interpretation, not an organizer statement. Official evidence: [participant guide](../sources/participant-guide.md), [rules](../sources/official-rules.md), [submission form](../sources/submission-form.md), and [event page](../sources/event.md).

## Conclusion

Loom fits **Indie Developer / Push the Frontier** if it demonstrates a working new capability: Meta models change an executable environment in response to real evidence, and an independent measurement shows that agents trained in the repaired environment perform better on the actual task.

A simulator generator alone is weaker. A dashboard with a scripted repair animation is weaker. A large vision statement without observed improvement does not satisfy the operational parts of the rubric.

## Admission is not the competition rubric

Application selection weighs idea clarity/feasibility/theme fit and demonstrated technical skill equally. Your project description supplies a concrete loop and a bounded task. Your prior-project answer supplies specific engineering techniques and measured development results. Those claims were supplied by you; this review did not independently audit the earlier experiments.

The favorable API-opinion answers are not a substitute for technical evidence. The event says its developer-program questions 13–16 are not used for selection. The submitted application and visible participant resources are evidence of the application/participant interface, not an independently verified prize-eligibility determination.

## Competition scoring

Stage one is pass/fail: plausible theme fit plus reasonable use of Meta Model API and/or Meta open-weight models in the hackathon project. Stage two scores each category from 0–10. The five highest-scoring projects per track advance to live final-round viewing and ranking under the same rubric.

| Criterion | Weight | What Loom needs to make observable |
| --- | --- | --- |
| Impact | 30% | Name a user: someone building or calibrating task-training environments. Show the consequence of a wrong simulator and the cost/time or failure reduction from correcting it. |
| Use of Meta AI tooling | 25% | Show the model interpreting evidence, diagnosing a causal mismatch, proposing a constrained code/parameter change, using tools, and responding to validation feedback. A decorative chat panel is not enough. |
| Technical depth and execution | 20% | Executable generation, restricted execution, bounded repair, immutable scoring, held-out data, reproducibility, regression rejection, and clear failure handling. |
| Demo | 25% | A comprehensible, working three-minute story: confident simulator prediction → visible real failure → diagnosis and patch → evaluator decision → improved performance. |

The public description emphasizes creativity, ambition, and working software for indie entrants; the participant guide supplies the numeric rubric. Do not replace the numeric rubric with the shorter marketing description.

## What they are signaling beyond the weights

Interpretations grounded in the guide, not additional official criteria:

- **A new capability beats a familiar wrapper.** Their anti-project examples explicitly include basic retrieval chatbots and simple image analyzers. Loom should execute and improve a model of a task, not merely describe a video.
- **The model must be central.** The submission separately asks what Meta products you used and why Meta models matter. Record specific model-driven decisions and evidence; do not claim exclusivity or superiority without a comparison.
- **Real-world grounding matters for this theme.** The theme names robotics, manufacturing, science, and control of physical systems. Showing physical observations makes your chosen angle legible, but the rubric does not require a robot or a particular modality.
- **The demo is a major engineering deliverable.** Meta tooling and demo together make up half the score. Do not spend the entire window on invisible infrastructure.
- **Technical depth should support a small, reliable story.** No rubric requires a marketplace, fleet scheduler, auth, billing, multiple environments, or expensive model training.

## Strengths of your proposed Loom

- Explicit generate–test–repair loop rather than a generic assistant.
- Clear use of reasoning plus sandboxed tools.
- Explicit independent evaluator and version history.
- Concrete physical task and three sensible metric families.
- Prior experience is directly relevant to isolation, patch validation, multimodal evidence, bounded compute, and regression testing.

## Main weaknesses to resolve

1. **Scope:** Generating simulators, perception, policy learning, reality capture, autonomous repair, experiments, and a polished interface is too many independent research bets for one demo. Limit the state/action space and task family before choosing infrastructure.
2. **What “real” means:** Define whether actions are performed by a robot, a human following instructions, or only a software oracle. Human-actuated physical trials can demonstrate environment modeling, but do not establish autonomous robotic control.
3. **What is generated:** A fixed simulator with a manually chosen parameter is not the full proposed capability. Show which state rules, constraints, or parameters were generated from evidence and what was provided as scaffolding.
4. **Independent evaluation:** A second language model is not sufficient independence if it shares the same assumptions or can change the task. Use immutable checks and held-out observations as the authority.
5. **Evidence leakage:** Separate generation data, repair feedback, and final held-out trials. Do not optimize against the final score repeatedly and then call it a holdout result.
6. **Trivial failure injection:** An intentionally planted bug is valid as a labeled controlled test, but stronger evidence is a genuine mismatch from withheld conditions or observations. Do not conceal how a bug was introduced.
7. **Originality:** Existing Loom already contains ingestion, provenance, and storage work. That is background, not the hackathon deliverable. Maintain an explicit contribution ledger.

## Positioning to use

> Loom turns observed task behavior into a trainable environment, then repairs that environment when reality proves it wrong.

The primary result should be a measured, reproducible correction, not “we solved sim-to-real” or “we can generate any environment.”

## Bonus-award fit

- **Best Agentic Use-Case ($5,000):** natural fit for a bounded autonomous repair workflow that completes a meaningful task.
- **Best Build on Meta Open Weight Models ($10,000):** plausible only if an eligible open-weight model actually runs in the submitted system, with provider/runtime evidence. Hosted Muse Wonder alone does not establish eligibility for this bonus.
- **Most Viral ($20,000) / Community Vote ($10,000):** optional distribution paths, not substitutes for technical judging. No posting or submission is authorized now.
- **Best Creative Build ($5,000):** requires the relevant creative use of Muse Image or Muse Wonder. Do not distort the core build just to chase an extra category.
