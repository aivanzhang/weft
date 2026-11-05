# Loom application — user-provided record

- Source: text provided directly by Ivan in this conversation; also visible in the “Your Application” panel at https://cerebralvalley.ai/e/meta-ai-developer-hackathon
- Observed: October 26, 2026, approximately 2:55 p.m. PDT
- Method: user-provided text transcribed to Markdown; question/answer content retained. This is a historical application record, not a new submission or independent verification of project claims.
- No application was submitted, edited, or withdrawn during this task. Historical consent answers do not authorize new actions.

> Public-copy edit: the former employer name is omitted; the project description is unchanged.

## What's your LinkedIn?

https://www.linkedin.com/in/aivanzhang/

## What's your Twitter/X?

https://x.com/aivanzhang

## What's your Github?

https://github.com/aivanzhang

## Where do you live (City, State, Country)?

SF, CA, USA

## Are you applying for the startup track or indie dev track? (you can change your track anytime)

Indie Dev Track

## Where do you study or work?

Pinnacle

## What do you want to build for this hackathon?

I want to build Loom: a platform that generates reinforcement-learning environments from real-world evidence and automatically corrects them as it discovers errors.

Given task specifications, demonstrations, and observed outcomes, Loom would generate an executable simulator with state transitions, actions, constraints, and rewards. Agents could train inside it, explore strategies, and practice completing the task.

Its core capability is a continuous generate–test–repair loop. Loom would run agent rollouts, compare simulated behavior with real observations, and identify incorrect dynamics, missing constraints, or exploitable simulator bugs. Meta’s reasoning models would diagnose these discrepancies and generate targeted code or parameter changes through sandboxed tools. An independent evaluator would validate each repair against fixed tests and task-success criteria. When evidence is insufficient, Loom would propose a targeted real-world experiment.

Every environment version and training run would be recorded, making improvements reproducible and regressions detectable. The objective is to reduce the gap between what agents learn in simulation and what actually works in reality.

For the ten-day hackathon, I would demonstrate the complete loop on a tabletop sorting task with a small action space: generate the environment, train an agent, expose a mismatch with reality, automatically repair the simulator, and evaluate the updated policy. I would measure prediction error on held-out real trials, simulator exploit frequency, and the difference between simulated and real task performance.

Loom targets Push the Frontier by making accurate training environments easier to create and continuously improve. It builds on my work in model evaluation, constrained code repair, and multimodal processing. The long-term ambition is an expanding library of self-generating, self-correcting environments that enables reliable learning across robotics, manufacturing, and other real-world systems.

## Tell us about an AI project you recently built that showcases your technical skills. (hint: Include what you built, the techniques you used)

I built an AI research platform for evaluating coding agents, multimodal models, and efficient GPU inference.

Agent engineering. I integrated Harbor and mini-SWE-agent to run coding tasks in isolated environments, capture tool calls and patches, and re-evaluate patches in clean environments. A SWE-bench Pro pilot passed both initial grading and independent regrading.

Test-guided code repair. Built a bounded repair loops that takes exact code edits and uses public-test feedback.

Multimodal model integration. I connected ObjectStream’s entity-based memory compression to Qwen2.5-VL using PyTorch and Transformers. I handled visual embeddings, attention masks, positional encodings, and decoding caches. Experiments measured which evidence compression preserved or lost, while preventing access to future video frames.

GPU inference. I deployed a pretrained, quantized 27B Qwen model on a single 24 GB NVIDIA A10G using CUDA-enabled llama.cpp. I configured Flash Attention, KV-cache precision, context limits, and reasoning budgets. Controlled probes measured approximately 34 generated tokens per second.

Evaluation and regression testing. I built tooling that pins model hashes, datasets, prompts, runtimes, and grading protocols, then compares correctness and resource usage across runs. A fresh ten-task development baseline solved nine tasks without assisted repairs or truncated outputs. This was a development result, not a full-benchmark score.

ML infrastructure. I built disposable AWS GPU workers with restricted permissions, encrypted storage, private results, and independent shutdown deadlines. The system preserves predictions, logs, checksums, and failed attempts, verifies worker termination, and recovers completed results without rerunning inference.

## Have you used Meta Model API before?

No

## How likely are you to build with Meta Model API?

Very likely

## How would you describe your overall opinion of Meta Model API?

Very favorable

## How did you hear about this hackathon?

Friend

## Marketing communications consent

By selecting Yes, you agree to receive marketing related electronic communications from Meta, including news, events, updates, and promotional emails. You may withdraw your consent and unsubscribe from these at any time, for example, by clicking the unsubscribe link included on our emails.

For more information about how Meta handles your data please read our [Meta Data Policy](https://www.facebook.com/privacy/policy/).

Yes
