# Weft demo runbook

Everything needed to record the walkthrough video. All figures here were measured on
2026-11-04; the provenance of each is stated so a stale number is recognisable as stale.

The access code is **not** in this repository. It lives in `.local/demo-access-code`, which is
gitignored, and the hosted Worker's `ACCESS_CODE` secret was set from the same string. Read it
from that file; keep it out of the recording frame.

## The two surfaces

| | Local portal | Hosted |
| --- | --- | --- |
| URL | `http://127.0.0.1:8765/demo` | `https://weft.kintsugi.software` |
| Has an upload UI | yes, the guided walkthrough | API only; the site is a landing page |
| Full run | **~50 s** | ~88 s cold, ~69 s warm |
| Costs money | ~$0.035 Meta per run (Muse step) | none |

Record against the **local portal**. The hosted landing page's "Build your world" link points
at `/demo`, which only the portal serves — the Worker returns 404 for it. Do not click it on
camera.

## Setting up on a fresh machine

The source clips ship on this branch in `assets/demo/clips/`, with their attribution in
`assets/demo/clips/ATTRIBUTION.md`. The portal reads them from `.local/demo-clips`, which is
gitignored, so copy them across once:

```sh
mkdir -p .local/demo-clips
cp assets/demo/clips/*.mp4 assets/demo/clips/manifest.json .local/demo-clips/
```

You also need the access code at `.local/demo-access-code`, which is not in the repository.
Copy it from the machine that has it, or let the portal generate a fresh one on first demo
access — a generated code works for the local walkthrough but will not match the hosted
service's secret.

Eight clips ship here. Three catalogued clips are omitted because their licence is unrecorded;
see the attribution file.

## Starting the portal

```sh
weft portal --root .local/demo-portal-04 --port 8765
```

`--root` matters: the demo's finished runs, featured list and generation budget live under
that root, so a different root shows an empty gallery. `.local/demo-clips` and
`.local/demo-access-code` are resolved against the repository root instead, not the portal
root.

Then open `http://127.0.0.1:8765/demo` and enter the code.

## Recording steps

1. Open the demo page and unlock it with the code.
2. The gallery shows finished runs plus curated examples. **Know which is which**: entries
   labelled "your run" are real executions with their own artifacts and recordings; the
   curated set is pre-rendered source/result pairs from `assets/demo/motion-showcase/`.
3. Upload a clip, or pick one from the catalogue. Upload is the honest demo: it is the path a
   reviewer takes. Any MP4 up to 100 MB.
4. **Leave the prompt field empty** unless you want live character generation. A prompt spends
   Meshy credits and is bounded to 2 attempts across the walkthrough; empty imports an
   already-generated character instead and costs nothing.
5. Let all seven steps run. Open a step's evidence zip on camera rather than filming a
   progress bar.

### Recommended clip

`waiters-squat.mp4` — shipped in `assets/demo/clips/`, 3,629,209 bytes, sha256 `28eb5ddd…`.
CC BY-SA 4.0, Taco Fleur / Cavemantraining via Wikimedia Commons, so it is safe to publish
with attribution. Admits on the first attempt with no retry ladder, 100/100 learned, tracked
fraction 1.000.

Alternatives: `bodyweight-squats.mp4` has the largest learning contrast in the pack;
`jumping-jacks.mp4` demonstrates automatic windowing on a 45-second multi-person clip but
takes about 185 s hosted. Avoid `overhead-squat`, `sots-press` and `sots-press-window` for
anything published — their licence reads "unverified - user-provided local video".

### Clip constraints, stated before upload

One person, upright, head to feet in frame throughout. Plain background, fixed camera, one
continuous shot. A moving exercise. MP4 up to 40 MB hosted, 100 MB locally. Hanging and
lying-down exercises are refused by design: the pose estimator cannot see them confidently and
a more accurate model measured worse.

## What the seven steps show

Measured on an uploaded run that completed in ~50 s, all seven `done`.

| Step | What it does | Talking point |
| --- | --- | --- |
| `intake` | Local pose tracking → joint trajectories with per-sample confidence, competing hypotheses, explicit gaps. No API calls. | Evidence is retained and inspectable, never asserted. |
| `world` | Evidence becomes a running MuJoCo simulation: scaled articulated body, joint limits, contacts, authored ground. | Every assumption is labelled as an assumption. |
| `task` | An RL task attaches with success gates frozen before the run. | The pipeline cannot edit the gates to flatter its own result. |
| `rollout` | A reference controller drives continuous control steps while the frozen gates score tracking error live; the session is recorded for exact replay. | Working software, not a scripted animation. |
| `avatar` | Appearance swaps **mid-recording**. Simulation state, controls and score untouched; the character keeps its own proportions rather than being stretched onto the body. | Measured: bone lengths move by at most 1.7e-15 relative across 103 posed frames, no non-finite vertex. |
| `muse` | Meta Muse reads construction evidence and source frames, then proposes missing constraints and alternative explanations. | An independent scorer, not the model, accepts or rejects. Skips and says so plainly when unavailable. |
| `create` | A generated character is imported and superimposed on the same recorded physics. Marked experimental. | Generated assets face the same gates. |

Each step packs a bounded evidence zip. A completed run is about 9 MB across seven zips, with
`intake.zip` the largest at ~5 MB.

## Video script, under three minutes

About 360 spoken words, roughly 2:25 at a normal pace, leaving room for visual beats.

| Time | On screen | Say |
| --- | --- | --- |
| 0:00 | Source clip playing | "This is an ordinary video of someone doing an overhead squat. Watch what happens to it." |
| 0:08 | Demo page | "Weft turns a clip like this into a reinforcement-learning environment an agent can actually train in. No studio, no rig, no hand-authoring." |
| 0:20 | Drop the file in | "I upload the file. Everything from here is the real pipeline — no pre-baked result." |
| 0:28 | Step 1, open `intake.zip` | "First it reads the video as evidence: joint trajectories with per-sample confidence, competing hypotheses, and explicit gaps. Nothing is asserted, nothing is discarded — and you can open it." |
| 0:48 | Step 2, world loads | "That evidence becomes a running MuJoCo world — a scaled articulated body, joint limits, contacts, an authored ground plane. Every assumption is labelled as one." |
| 1:05 | Step 3 | "Then the task attaches, and this is the part that matters: the success gates are frozen *before* the run. The pipeline cannot edit them to flatter its own result." |
| 1:25 | Step 4, rollout video | "An agent runs inside it. 246 control steps. The learned policy succeeds 100 times out of 100 — where random gets 1, and doing nothing gets 0." |
| 1:40 | Limits panel | "To be precise, that's a single-joint target, not a reproduction of the whole exercise. The pipeline states that itself rather than letting you assume otherwise." |
| 1:50 | **The swap** | "Now the part worth watching. Mid-recording, the character changes. Simulation state, controls and score are untouched — only appearance. And it keeps its own proportions instead of being stretched onto the body: bone lengths move by about one part in ten to the fifteenth." |
| 2:15 | Step 6, Muse text | "Meta's model then reads the construction evidence and proposes what the simulation got wrong. An independent scorer — not the model — accepts or rejects the change." |
| 2:32 | A refusal | "And when a clip is out of scope — hanging, or lying down — it refuses with the reason, in seconds. It doesn't quietly hand you a worse answer." |
| 2:45 | Finished run in gallery | "Video in, trainable environment out, in under a minute — with evidence at every step you can check yourself." |

### Why this order

The swap sits after the viewer understands there is real physics underneath, so it reads as
decoupling rather than a costume change. The frozen gates come before the 100/100, because
they are what make that number worth reading.

The two honesty beats are load-bearing, not hedging. "It refuses and tells you why" is a
stronger claim to a technical reviewer than implying nothing ever fails.

### Claims to keep accurate

- Task success is a **single-joint** target. It is not learned reproduction of the whole
  exercise, and the run's own limits list says so.
- The body is an anchored articulated model fitted to monocular pose estimates. It is not
  recovered anatomy, measured muscle activation, full-body balance, or validated biomechanics.
- Muscle bands, where shown, are illustrative actuator proxies.
- A simulation result is a controlled software experiment, not evidence of physical transfer.

## Spending

| | State on 2026-11-04 |
| --- | --- |
| Meta allowance | $5.00 cap, ~$1.04 charged, ~$3.96 remaining, all reservations settled |
| Per demo run | ~$0.035 at the Muse step |
| Muse calls | capped at 8 per run |
| Live character generation | only with a prompt; 2 attempts across the walkthrough, ~35 Meshy credits each |

Roughly 110 rehearsal runs fit in the remaining allowance. For zero spend, run without a
prompt and accept that Muse skips honestly if it is unavailable.

## Hosted lane, if you want it on camera

```sh
curl -X POST https://weft.kintsugi.software/api/jobs \
  -H "Content-Type: video/mp4" \
  -H "Origin: https://weft.kintsugi.software" \
  -H "X-Weft-Access-Code: $(cat .local/demo-access-code)" \
  --data-binary @.local/demo-clips/waiters-squat.mp4
```

Returns 100/100 learned against 1/100 random and 0/100 coast, three avatars with matching
render digests, and a ~9.2 MB archive. Submit **one at a time**: a single container serves all
jobs, and a second concurrent submission is told an environment is already generating.

Refusal timings, from 19 uncurated production inputs: wrong file type rejected in 0.3–3.0 s,
undecodable in 7.4–9.7 s, refused by the motion or person gates in 32.9–41.8 s.

## Gotchas

- The hosted `/demo` link 404s. The page is served by the local portal only.
- A first hosted submission within about four minutes of a deploy can hit the previous
  container image; wait it out before measuring.
- The portal keeps four finished runs and evicts the oldest, so a long rehearsal session will
  retire early runs.
- Three clips stay refused on purpose — hanging and lying-down exercises. That is the published
  envelope, not a bug, and the refusal names the reason.
