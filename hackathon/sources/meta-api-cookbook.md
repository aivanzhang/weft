# Source snapshot: meta-api-cookbook

- Source: https://github.com/meta-models/meta-model-cookbook
- Retrieved: 2026-10-26 03:00:24 PM PDT
- Method: read-only HTTP GET; HTML converted to Markdown.
- Status: captured
- SHA-256 of source bytes: `e13cc01680845852cb0556653f8aace83fed04c483ea5db28063cf6c20b641aa`

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

meta-models

**[meta-model-cookbook](https://github.com/meta-models/meta-model-cookbook)**

Public

- [Notifications](https://github.com/login?return_to=%2Fmeta-models%2Fmeta-model-cookbook)You must be signed in to change notification settings
- [Fork49 (49)](https://github.com/login?return_to=%2Fmeta-models%2Fmeta-model-cookbook)
- [Star186 (186)](https://github.com/login?return_to=%2Fmeta-models%2Fmeta-model-cookbook)You must be signed in to star a repository

## About

Developer recipes for Meta Model API — agentic workflows, multi-agent orchestration patterns, and use case examples with cost comparisons. Colab notebooks + standalone Python files.

### Resources

[Readme](https://github.com/meta-models/meta-model-cookbook#readme-ov-file)

[MIT license](https://github.com/meta-models/meta-model-cookbook#MIT-1-ov-file)

### Code of conduct

[Code of conduct](https://github.com/meta-models/meta-model-cookbook#coc-ov-file)

### Contributing

[Contributing](https://github.com/meta-models/meta-model-cookbook#contributing-ov-file)

[Activity](https://github.com/meta-models/meta-model-cookbook/activity)

[Custom properties](https://github.com/meta-models/meta-model-cookbook/custom-properties)

### Stars

**186** stars

### Watchers

**7** watching

### Forks

[**49** forks](https://github.com/meta-models/meta-model-cookbook/forks)

[Report repository](https://github.com/contact/report-content?content_url=https%3A%2F%2Fgithub.com%2Fmeta-models%2Fmeta-model-cookbook&report=meta-models+%28user%29)

main

[Branches](https://github.com/meta-models/meta-model-cookbook/branches)[Tags](https://github.com/meta-models/meta-model-cookbook/tags)

[](https://github.com/meta-models/meta-model-cookbook/branches)[](https://github.com/meta-models/meta-model-cookbook/tags)

Go to file

Code

Open more actions menu

## Latest commit

 

## History

[83 Commits](https://github.com/meta-models/meta-model-cookbook/commits/main/)

[](https://github.com/meta-models/meta-model-cookbook/commits/main/)83 Commits

## Folders and files

[TABLE]

## Repository files navigation

- 
- [README](https://github.com/meta-models/meta-model-cookbook#)
- [Code of conduct](https://github.com/meta-models/meta-model-cookbook#)
- [Contributing](https://github.com/meta-models/meta-model-cookbook#)
- [MIT license](https://github.com/meta-models/meta-model-cookbook#)

More items

# Meta Model API cookbook

[](https://github.com/meta-models/meta-model-cookbook#meta-model-api-cookbook)

Recipes for building agents and coding tools on the Meta Model API, covering API fundamentals, agent patterns, and end-to-end use cases.

Meta Model API lets you build with Muse Spark using the tools you already run: it's drop-in compatible with the OpenAI SDK, the Anthropic SDK, and agent CLIs like OpenCode and Claude Code.

Point your client at the Model API base URL, set your key, and keep the rest of your code. Each recipe is a self-contained, copy-paste starting point that proves a capability and gives you something to build on. The default model is **Muse Spark** (`muse-spark-1.3`), which has a 1,048,576-token context window; the preview is free.

## Getting started

[](https://github.com/meta-models/meta-model-cookbook#getting-started)

You need a [Model API account](https://dev.meta.ai/) and an API key. Store the key as an environment variable so it stays out of your code:

export MODEL_API_KEY="LLM\|{numeric_id}\|{secret}"\
pip install openai

import os\
\
from openai import OpenAI\
\
\# The OpenAI SDK does not auto-read MODEL_API_KEY, so pass it explicitly.\
client = OpenAI(\
    base_url="https://api.meta.ai/v1",\
    api_key=os.environ\["MODEL_API_KEY"\],\
)\
\
response = client.chat.completions.create(\
    model="muse-spark-1.3",\
    messages=\[{"role": "user", "content": "Hello, world!"}\],\
)\
\
print(response.choices\[0\].message.content)

## Recipes

[](https://github.com/meta-models/meta-model-cookbook#recipes)

This cookbook mirrors the three sections of the Meta Model API cookbook website. Each recipe is a self-contained, copy-paste starting point. Start with **API fundamentals** to learn the API primitives, then move into **agent patterns** and end-to-end **use cases**.

### [1. API fundamentals](https://github.com/meta-models/meta-model-cookbook/blob/main/01_api_fundamentals)

[](https://github.com/meta-models/meta-model-cookbook#1-api-fundamentals)

Prove each API primitive works and get a starting point you can build on.

| \# | Recipe | What it does |
|----|----|----|
| 01 | [Quickstart: chat completions](https://github.com/meta-models/meta-model-cookbook/blob/main/01_api_fundamentals/01_chat_completions.ipynb) | Make your first Muse Spark call by pointing the OpenAI SDK at one new base URL. |
| 02 | [Streaming responses](https://github.com/meta-models/meta-model-cookbook/blob/main/01_api_fundamentals/02_streaming.ipynb) | Render tokens as they generate and read the final usage chunk. |
| 03 | [Tool and function calling](https://github.com/meta-models/meta-model-cookbook/blob/main/01_api_fundamentals/03_tool_calling.ipynb) | Detect tool calls and run the execute-and-feed-back loop. |
| 04 | [Structured output](https://github.com/meta-models/meta-model-cookbook/blob/main/01_api_fundamentals/04_structured_output.ipynb) | Get schema-guaranteed JSON that parses on the first try. |
| 05 | [Prompt caching](https://github.com/meta-models/meta-model-cookbook/blob/main/01_api_fundamentals/05_prompt_caching.ipynb) | Reuse a stable prompt prefix and track cached tokens. |
| 06 | [Reasoning and thinking tokens](https://github.com/meta-models/meta-model-cookbook/blob/main/01_api_fundamentals/06_reasoning_tokens.ipynb) | Control reasoning effort and replay reasoning across turns. |
| 07 | [Vision input](https://github.com/meta-models/meta-model-cookbook/blob/main/01_api_fundamentals/07_vision_input.ipynb) | Send images by URL or base64 and get structured analysis back. |
| 08 | [Long context](https://github.com/meta-models/meta-model-cookbook/blob/main/01_api_fundamentals/08_long_context.ipynb) | Pack repo-scale context into the context window. |
| 09 | [Error handling and retry](https://github.com/meta-models/meta-model-cookbook/blob/main/01_api_fundamentals/09_error_handling.ipynb) | Back off with jitter and skip retries on client errors. |
| 10 | [Search grounding](https://github.com/meta-models/meta-model-cookbook/blob/main/01_api_fundamentals/10_search_grounding.ipynb) | Ground responses in live web search results with inline citations. |

### [2. Agent patterns](https://github.com/meta-models/meta-model-cookbook/blob/main/02_agent_patterns)

[](https://github.com/meta-models/meta-model-cookbook#2-agent-patterns)

Build the loops that turn a model into an agent: planning, parallel work, and self-correction.

| \# | Recipe | What it does |
|----|----|----|
| 01 | [Basic agent loop](https://github.com/meta-models/meta-model-cookbook/blob/main/02_agent_patterns/01_agent_loop_basics) | Wire up the core perceive-decide-act agent loop. |
| 02 | [Interleaved reasoning and tool use](https://github.com/meta-models/meta-model-cookbook/blob/main/02_agent_patterns/02_interleaved_reasoning) | Interleave reasoning with tool calls in a single turn. |
| 03 | [Multi-turn context management](https://github.com/meta-models/meta-model-cookbook/blob/main/02_agent_patterns/03_managing_context) | Manage growing context across a long agent run. |
| 04 | [Validated in-place edits](https://github.com/meta-models/meta-model-cookbook/blob/main/02_agent_patterns/04_validated_in_place_edits) | Validated search-and-replace edits with a coding agent. |
| 05 | [Alert fatigue copilot](https://github.com/meta-models/meta-model-cookbook/blob/main/02_agent_patterns/05_alert_fatigue_copilot) | Extract grounded patterns from a noisy alert feed, then probe → chat → self-assess with strict-JSON output. |

### [3. Use cases](https://github.com/meta-models/meta-model-cookbook/blob/main/03_use_cases)

[](https://github.com/meta-models/meta-model-cookbook#3-use-cases)

End-to-end patterns: multimodal perception, orchestration, and full applications.

| \# | Recipe | What it does |
|----|----|----|
| 01 | [Chart analysis](https://github.com/meta-models/meta-model-cookbook/blob/main/03_use_cases/01_chart-explainer) | Read charts and extract structured data from images. |
| 02 | [Error screenshot fix](https://github.com/meta-models/meta-model-cookbook/blob/main/03_use_cases/02_screenshot_bugfix) | Diagnose a bug from an error screenshot and fix it. |
| 03 | [Smart glasses with OpenClaw](https://github.com/meta-models/meta-model-cookbook/blob/main/03_use_cases/03_visionclaw) | Hands-free look-and-ask on Ray-Ban Meta glasses. |
| 04 | [Generating slides](https://github.com/meta-models/meta-model-cookbook/blob/main/03_use_cases/04_doc_generation) | Generate a slide deck from a prompt or source content. |
| 05 | [Browser-verified web design](https://github.com/meta-models/meta-model-cookbook/blob/main/03_use_cases/05_web_design) | Build a website with a coding agent that checks its own work in a real browser. |
| 06 | [Iterative game dev](https://github.com/meta-models/meta-model-cookbook/blob/main/03_use_cases/06_iterative_game_dev) | Build a browser game end-to-end with a coding agent, verified in a real browser. |
| 07 | [Sandboxed execution](https://github.com/meta-models/meta-model-cookbook/blob/main/03_use_cases/07_sandbox_execution) | Execute model-generated code in a sandbox. |
| 08 | [Multi-agent product studio](https://github.com/meta-models/meta-model-cookbook/blob/main/03_use_cases/08_multi_agent_orchestration) | Orchestrate a four-profile Hermes team (PM, backend, frontend, tech writer) that coordinates through a shared Kanban board. |
| 09 | [One-shot game dev](https://github.com/meta-models/meta-model-cookbook/blob/main/03_use_cases/09_one-shot-game_dev) | Build a complete 3D browser game in a single pass — an AGENTS.md plus one structured prompt, no iteration loop. |
| 10 | [Perception grounding](https://github.com/meta-models/meta-model-cookbook/blob/main/03_use_cases/10_perception_grounding) | Identify food items in a fridge photo, pin interactive health-score dots at their pixel locations, and generate a self-contained HTML overlay. |
| 11 | [GitHub repo agent](https://github.com/meta-models/meta-model-cookbook/blob/main/03_use_cases/11_github_repo_agent) | Autonomous GitHub Actions bot — triage, PR review, AI-slop detection, and bug-fix PRs — built on OpenCode + Muse Spark. |
| 12 | [Computer use](https://github.com/meta-models/meta-model-cookbook/blob/main/03_use_cases/12_computer_use) | Drive a Linux desktop from screenshots — the agent finds an app, opens it, and plays it, clicking through a Cua sandbox. |
| 13 | [macOS computer use](https://github.com/meta-models/meta-model-cookbook/blob/main/03_use_cases/13_macos_cua) | Drive a real Mac from screenshots with `metacua`, a native computer-use agent (Swift + Python) that clicks, types, and works out a GUI app on its own. |
| 14 | [Muse robotics](https://github.com/meta-models/meta-model-cookbook/blob/main/03_use_cases/14_muse_robotics) | Orchestrate Muse Voice, Muse Spark, a full-mesh MuJoCo Panda, hosted SAM preview, Reachy Mini camera/audio/reactions, Muse Image, a kiosk, and file-first QR postcards. |
| 15 | [Local agent that asks before calling Muse Spark](https://github.com/meta-models/meta-model-cookbook/blob/main/03_use_cases/15_glimmer_api_agent) | Run Muse Glimmer locally with Hermes and require approval before selected questions are sent to Muse Spark. |

### [4. Muse Code](https://github.com/meta-models/meta-model-cookbook/blob/main/04_muse_code)

[](https://github.com/meta-models/meta-model-cookbook#4-muse-code)

Build agents with sessions you can audit, safety that fails closed, and an agent team you can steer — each recipe ends with a proof point you can replay.

| \# | Recipe | What it does |
|----|----|----|
| 01 | [Append-only audit log and crash-safe resume](https://github.com/meta-models/meta-model-cookbook/blob/main/04_muse_code/01_append_only_audit_log) | Export the append-only session log and walk the exact sequence of actions and approvals as replayable evidence, then kill the harness mid-task and resume where it stopped with no duplicate side effects. |
| 02 | [Deterministic Replay in CI](https://github.com/meta-models/meta-model-cookbook/blob/main/04_muse_code/02_deterministic_replay) | Turn recorded Muse Code events into a focused golden fixture and replay its model-context projection as a merge-blocking check. |
| 03 | [Staged approvals for compound shell commands](https://github.com/meta-models/meta-model-cookbook/blob/main/04_muse_code/03_staged_approvals) | Decompose a compound shell command into stages, auto-clear the safe ones, and hold the dangerous or unparseable ones for review — trust scoped per action and per workspace, failing closed. |
| 04 | [Contained execution you can prove](https://github.com/meta-models/meta-model-cookbook/blob/main/04_muse_code/04_contained_execution) | Run shell commands inside an OS-level sandbox that live-probes its own enforcement and refuses to run when containment is not actually enforcing. |
| 05 | [The agent can't rewrite its own rules](https://github.com/meta-models/meta-model-cookbook/blob/main/04_muse_code/05_immutable_guardrails) | Ask the agent to edit the guardrail file it runs under; the write stops for human review with no standing grant on offer, and a shell write to the same path fails read-only at the sandbox. |
| 06 | [Subagent fanout across isolated worktrees](https://github.com/meta-models/meta-model-cookbook/blob/main/04_muse_code/06_subagent_fanout) | Fan one job out to parallel subagents, each in its own git worktree, steered or stopped from a single command center. |
| 07 | [Goal tracking with a judge](https://github.com/meta-models/meta-model-cookbook/blob/main/04_muse_code/07_goal_tracking) | Declare a goal once; a judge decides whether the work is actually done and refuses to close the turn until every acceptance check passes. |
| 08 | [Bundled skills: a paved road from idea to shipped](https://github.com/meta-models/meta-model-cookbook/blob/main/04_muse_code/08_bundled_skills) | Drive the built-in `/plan`, `/grilling`, `/grill-with-docs`, and `/taste` skills end to end — plan a change, stress-test it, record the decisions, and build UI that does not look AI-made. |
| 09 | [Loop and cron: scheduled agent work](https://github.com/meta-models/meta-model-cookbook/blob/main/04_muse_code/09_loop_and_cron) | Schedule recurring or one-time agent work in natural language, then view, change, or cancel it from the same chat surface. |
| 10 | [Multi-turn side chats](https://github.com/meta-models/meta-model-cookbook/blob/main/04_muse_code/10_side_chats) | Branch a multi-turn side conversation off a busy main thread with `/side` (or its alias `/btw`), isolated from the main transcript. |
| 11 | [Muse Code for Security Research](https://github.com/meta-models/meta-model-cookbook/blob/main/04_muse_code/11_mcp_security_tooling) | Wire Burp Suite, Meta’s bug bounty research toolkit, headless Ghidra and LLDB into Muse Code over MCP, then point it at PortSwigger’s deliberately vulnerable demo site, a real bounty programme, and a real CVE buried in a stripped binary. |

### [5. Muse Image](https://github.com/meta-models/meta-model-cookbook/blob/main/05_muse_image)

[](https://github.com/meta-models/meta-model-cookbook#5-muse-image)

Generate and edit images through the conversational Responses API, driving each task end to end in one chained conversation.

| \# | Recipe | What it does |
|----|----|----|
| 01 | [Generate, edit, and compose images](https://github.com/meta-models/meta-model-cookbook/blob/main/05_muse_image/01_image_api_fundamentals) | Drive Muse Image through the Responses API: generate an image from a prompt, refine it over chained turns, and attach reference images to steer or compose several into one scene. |
| 02 | [Ground image generation in web search](https://github.com/meta-models/meta-model-cookbook/blob/main/05_muse_image/02_backyard_restyle) | Let the model search the live web mid-generation, so it renders from furniture actually sold today, then place a coordinated outdoor set into a photo of a real backyard. |
| 03 | [Consistency across an image series](https://github.com/meta-models/meta-model-cookbook/blob/main/05_muse_image/03_anchored_generation) | Anchor a whole series on a small reference set and carry it through the conversation, so the character, style, and setting stay on-model from one image to the next. |
| 04 | [Editing with image understanding and reasoning](https://github.com/meta-models/meta-model-cookbook/blob/main/05_muse_image/04_image_editing_with_reasoning) | Edit a specific part of an image by describing it, letting the model reason over the frame and refine the same image across turns, changing what you ask for while preserving the rest. |

### [6. Muse Voice Transcribe](https://github.com/meta-models/meta-model-cookbook/blob/main/06_muse_voice)

[](https://github.com/meta-models/meta-model-cookbook#6-muse-voice-transcribe)

Turn speech into text as it is spoken. Muse Voice Transcribe is the one part of the Model API that is not an HTTP endpoint — real-time transcription needs a bidirectional connection, so these recipes use a WebSocket client rather than the OpenAI SDK.

| \# | Recipe | What it does |
|----|----|----|
| 01 | [Speech to text](https://github.com/meta-models/meta-model-cookbook/blob/main/06_muse_voice/01_voice_api_fundamentals) | Transcribe a recording or a live microphone over the streaming WebSocket, get speaker-attributed turns with diarization, or post a whole recording in one HTTP request. |
| 02 | [Voice-controlled Apple Chess](https://github.com/meta-models/meta-model-cookbook/blob/main/06_muse_voice/02_voice_chess_cua) | Turn exact spoken chess moves into locally validated Apple Chess actions with a passive HUD, calibrated grid, dry-run mode, and fail-closed native input. |

### [7. SAM 3](https://github.com/meta-models/meta-model-cookbook/blob/main/07_sam)

[](https://github.com/meta-models/meta-model-cookbook#7-sam-3)

Turn segmentation masks and video tracks into finished applications.

| \# | Recipe | What it does |
|----|----|----|
| 01 | [SAM API basics](https://github.com/meta-models/meta-model-cookbook/blob/main/07_sam/01_api_basics) | Segment images and videos in Python or TypeScript with the official `meta-sam` parsers. |
| 02 | [Video to animated sticker pack](https://github.com/meta-models/meta-model-cookbook/blob/main/07_sam/02_video_to_sticker_pack) | Track an animal through a video and export transparent, messaging-ready animated stickers. |
| 03 | [Multi-speaker headlocked captions](https://github.com/meta-models/meta-model-cookbook/blob/main/07_sam/03_headlocked_speech_bubble_captions) | Upload a conversation, map diarized voices to SAM-tracked people, and download the captioned video. |

## License

[](https://github.com/meta-models/meta-model-cookbook#license)

See [LICENSE](https://github.com/meta-models/meta-model-cookbook/blob/main/LICENSE) for details.

## About

Developer recipes for Meta Model API — agentic workflows, multi-agent orchestration patterns, and use case examples with cost comparisons. Colab notebooks + standalone Python files.

### Resources

[Readme](https://github.com/meta-models/meta-model-cookbook#readme-ov-file)

[MIT license](https://github.com/meta-models/meta-model-cookbook#MIT-1-ov-file)

### Code of conduct

[Code of conduct](https://github.com/meta-models/meta-model-cookbook#coc-ov-file)

### Contributing

[Contributing](https://github.com/meta-models/meta-model-cookbook#contributing-ov-file)

[Activity](https://github.com/meta-models/meta-model-cookbook/activity)

[Custom properties](https://github.com/meta-models/meta-model-cookbook/custom-properties)

### Stars

**186** stars

### Watchers

**7** watching

### Forks

[**49** forks](https://github.com/meta-models/meta-model-cookbook/forks)

[Report repository](https://github.com/contact/report-content?content_url=https%3A%2F%2Fgithub.com%2Fmeta-models%2Fmeta-model-cookbook&report=meta-models+%28user%29)

## Releases

## Packages

## Used by

## Contributors

## Languages
