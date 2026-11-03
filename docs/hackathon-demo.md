# Run the local Weft demo

Video → estimated body → interactive 3D simulation → optional character → recorded
controls and replay → Muse review. This is a rough local product demo, not validated
biomechanics or reliable reconstruction of every unfamiliar scene.

## Start

From the repository root, run the existing portal yourself:

```sh
.venv-lab/bin/python -m weft portal --root .local/portal-unfamiliar-source-acceptance-01/portal
```

Open the authenticated localhost link printed in the terminal. This saved demo
root contains the licensed single-leg-squat video, its original unresolved scene,
and **Body and ground approximation** (`1a5b41db5aa34654a84c83a9dbf5e316`).
Do not move this root: saved source seals retain their original locations.
The normal `.venv-lab/bin/python -m weft portal` command uses `.local/portal` instead.
No hosted service is started or restored.

## Demo sequence

1. Select **Body and ground approximation**. Inspect its processing artifacts.
   The reference has geometry warnings; this remains a valid manual sandbox,
   not successful exercise tracking. Original equipment inventory remains unknown.
2. **Open interactive sandbox** once and wait for the body to load. Use Reset,
   manual actions, coast or authored reference controls. Reference control is not
   a learned policy. The simulation exposes actions, observations and rewards.
3. Return to the portal. **Prepare Muse panda + robot**, then **Add characters to
   this sandbox**. In the sandbox's Appearance controls, select a character or
   return to the original body. Start with the robot for the clearest preview;
   keep the skeleton overlay to compare proportions. The saved generated Studio
   panda is available too, but its reduced-detail surface is visibly rough. Large
   avatars can be cropped in fixed front views; rotate the 3D view to inspect them.
4. **Record & start**, interact and switch characters, then **Stop & save**.
   Inspect the saved action/state journal, exact-action replay and appearance
   playback/export. Starting a recording resets to its recorded seed.
5. Return to the portal's **Review the physics with Muse** section. The saved demo
   has a real Muse Spark review and cost receipt. It proposes explanations and
   tests; it does not alter the environment or claim measured muscle activity.
   A new project's review uses one explicit bounded request from the existing
   Meta allowance. Character switching/import makes no AI request.

## Your own video

Upload a fixed-camera exercise clip, at most 60 seconds / 1,800 frames / 100 MB.
Choose Exercise for local processing or explicitly enable server-side OpenAI
analysis. Choose **Interactive scene simulation** as the exercise output.
Declare the scene truthfully; do not label equipment absent just to get
an environment. For an unresolved exercise scene, **Check video locally** prepares
body evidence. If that succeeds, **Turn this body into a sandbox** offers an
explicit body-and-authored-ground approximation. Confirm that omitted equipment,
obstacles, other people and contacts remain unknown. The original evidence and
failed/uncertain attempts remain separate. Unusable body evidence is refused.

## Avatar creation and import

The existing Avatar Studio lives in `tools/avatar-assets/`; start it separately:

```sh
./tools/avatar-assets/run studio --port 8766
```

Studio generation uses its own API credentials and budget, not the portal's
allowance. Reusing an existing asset avoids another generation charge.
For the reusable conversion command and mapping settings, see
[Studio → Weft delivery](../tools/avatar-assets/docs/weft-delivery-v2.md).
The portal accepts a ZIP containing **only `model.glb` and `mapping.json`** at its
root (32 MiB maximum). Use a Weft delivery, not a raw Studio output ZIP. A rigged
humanoid model needs an explicit compatible mapping; arbitrary animal topology
is not automatically converted into a human skeleton. Two built-ins and up to two
custom deliveries fit one local library.

## Generalized motion fix

`weft-avatar-retarget-v2` maps the current simulated joint rotations and root
motion onto the character's original bone hierarchy and proportions. It does not
play an unrelated walk, hop or flight clip. Simulation timestamps drive both live
appearance and recorded playback. Unsupported rig mappings are rejected.

Different character proportions can put visual hands/feet away from physical
collision sites. The skeleton remains authoritative for ground contact, rewards
and task checks; this is appearance substitution, not a newly calibrated physical
body. No individual flight/hopping animation was repaired.

## Boundaries

- Approximate body/ground geometry and illustrative muscles, not recovered anatomy.
- No universal equipment/ground-contact recovery or successful learned balance claim.
- Muse avatar artwork and Muse Spark model review are separate features.
- Project API keys stay server-side; no ChatGPT login is used for inference.
- Interrupted jobs and unknown API charges remain recorded; no automatic paid retry.
