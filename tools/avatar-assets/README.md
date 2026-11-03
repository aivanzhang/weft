# Avatar assets

Local authoring for rigged 3D characters, portraits and motion previews.
This package creates assets; Weft's separate consumer handles simulation, controls
and recording. See [project status](../../PROJECT.md) and [next steps](TODO.md).

## Start locally

From the Weft repository root:

```sh
./tools/avatar-assets/run studio --port 8766
./tools/avatar-assets/run gallery --port 8767
```

Run only the service needed, then open its printed link. Studio creates jobs;
the gallery only previews saved assets and makes no paid calls. Port 8767 avoids
Weft's sandbox on 8765. The user starts/stops persistent servers.

Studio flow: **Save request → Prepare plan → Review estimate → Start creation**.
Saving is local. Planning and creation require server-side API credentials and
an exact-request spending grant. Use **Two separate assets** for independent
characters; a fused model is not two independently controllable assets.

## Current status

| Result | Scope |
| --- | --- |
| Saved-source panda dance | Completed appearance-preserving Studio delivery; shoulder dragon stays perched |
| Fresh owl and rabbit | Both source models generated; final animation rejected for rig/action failures |
| Built-in and authored examples | Rigged models and compact media exist; not universal prompt success |
| Weft export | Bounded package conversion and importer checks exist; each model still needs motion/visual qualification |

The current priority is general motion/rig repair and separate characters, **not
more fur tuning**. Unknown geometry, arbitrary interactions, automatic fused-model
decomposition, seamless loops and Muse-equivalent appearance are not established.
Native hair and browser fur use different representations. Passing numeric tests
or importing a GLB does not prove visual quality or physical accuracy.

## Setup

Reuse the existing `.venv`; its Pillow version conflicts with Weft's `.venv-lab`.
Fresh setup, tested on Python 3.14 / macOS arm64:

```sh
cd tools/avatar-assets
python3 -m venv .venv
.venv/bin/python -m pip install --require-hashes --only-binary=:all: -r requirements.lock
```

Native authoring uses `.tools/Blender.app` (Blender 4.5.13); other machines need
an explicit compatible Blender executable. The gallery needs no Blender.
Selected portable catalogs/assets are tracked; full renders and rejected local
outputs are not guaranteed to exist in a fresh clone.

## Prepare a prompt without spending

```sh
./tools/avatar-assets/run generate \
  --prompt 'A friendly emerald dragon with small wings' \
  --action 'Flap both wings while standing still.' \
  --preset preview --max-credits 35 --prepare-only \
  --out outputs/my-dragon-001
```

`--prepare-only` saves the exact request and estimate without provider calls.
`--plan-only` makes AI calls; `--execute` may generate, author and render.
`--reuse-model PATH --max-credits 0` avoids new Meshy geometry, not AI charges.
Use a new output directory and preserve every failed attempt.

Detailed commands and input contracts: [authoring skill](skills/avatar-asset-workflow/SKILL.md).
For manual scene authoring, use `request`, `build3d`, `reload3d`, `export2d` and
`verify`; `request` records intent, not a fabricated finished model.

## Budgets and preservation

- Avatar OpenAI has its own **$15 total cap**, including prior charges and unknown
  holds, in `.local/openai/ledger.sqlite3` inside this package. Recheck it before
  paid work; historical remaining balances are not current authorization.
- Meshy uses Weft-root `.local/meshy/ledger.sqlite3`. The approved additional
  62 credits funded the owl/rabbit pair alongside 8 remaining credits; that is
  not a reusable allowance or permission to buy credits/retry generation.
- Keep server keys private. Do not borrow Weft's Meta/OpenAI portal budgets or
  use a personal ChatGPT login as Studio's AI runtime. A key is not spending approval.
- Keep `outputs/`, `evidence/`, source ZIPs and failed jobs unchanged. Budget
  reservations, source identities and independent quality gates cannot be reset
  or weakened to make a run pass.

## Saved assets and verification

- Accepted saved-source Studio delivery: `outputs/studio-4701879a91cf42efbeebf16c/delivery.json`.
- Read-only gallery catalogs: `browser/gallery-compound-motion-v1.json`,
  `browser/gallery-panda-fur-v1.json`, `browser/gallery-agent-v1.json`.
- Original Weft mappings: `outputs/weft-muse-panda-v1` and
  `outputs/weft-humanoid-robot-v1`; proof: `outputs/weft-import-proof-v1/results.json`.
- Run the test suite with `.venv/bin/python -m unittest discover -s tests -v`
  from this package. Analytic tests do not replace actual motion/media review.

Original reference: `/Users/ivan/Downloads/Poo avatar assets.zip`, SHA-256
`cb7eac375b5fcef0cd6c3f9457da7911f08f73337178c31fe33bfa090e6a54cb`.
It contains 2D media, not a recovered rig or verified license; never execute
archive content. Authored/generated models are new assets, not recovered anatomy.

The [prior complete guide](https://github.com/kintsugi-software/weft/blob/1dbc204efaafc8e0697e654086c75fc465df1eba/tools/avatar-assets/README.md) retains detailed recipes, historical
results, catalogs and limitations. [TODO.md](TODO.md) is the current operational handoff.
