// Does a substituted avatar keep its OWN bone lengths when driven by the recovered body?
//
// The complaint substitution has to answer is a character stretched onto the estimated
// skeleton. This drives the product's real retargeting module over a finished run and
// reports the worst relative bone-length change, so the claim stays reproducible.
//
//   node scripts/verify/bone_length_preservation.mjs \
//     <run>/lab/run/scene.json <run>/lab/run/human-reference.json <avatar>/render.json ...
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const here = new URL(".", import.meta.url).pathname;
const R = require(`${here}../../src/weft/assets/avatar_retarget.js`);
const read = (p) => JSON.parse(readFileSync(p, "utf8"));
const [scenePath, refPath, ...renders] = process.argv.slice(2);
const config = R.compileScene(read(scenePath));
const ref = read(refPath);
const frames = ref.frames ?? ref.samples ?? ref.reference ?? [];
const parentOf = (role) => role === "torso" ? null
  : role.includes("elbow") ? role.replace("elbow", "shoulder")
  : role.includes("knee") ? role.replace("knee", "hip") : "torso";
const pos = (m) => m.slice(12, 15);
const dist = (a, b) => Math.hypot(a[0]-b[0], a[1]-b[1], a[2]-b[2]);

for (const rp of renders) {
  const asset = R.prepareAsset(read(rp));
  const roles = R.BODY_FRAMES;
  let worst = 0, pairs = 0, posed = 0, nonFinite = 0;
  for (const snap of frames) {
    const body = R.bodyFrames(config, snap);
    const out = R.retarget(asset, body);
    posed++;
    for (const role of roles) {
      const parent = parentOf(role);
      if (!parent) continue;
      const id = asset.roles.get(role), pid = asset.roles.get(parent);
      if (id === undefined || pid === undefined) continue;
      const restLen = dist(pos(asset.rest.get(id)), pos(asset.rest.get(pid)));
      const poseLen = dist(pos(out.nodes.get(id)), pos(out.nodes.get(pid)));
      if (!Number.isFinite(poseLen)) { nonFinite++; continue; }
      if (restLen > 1e-9) { worst = Math.max(worst, Math.abs(poseLen - restLen) / restLen); pairs++; }
    }
    for (const m of out.palette) for (const v of m) if (!Number.isFinite(v)) nonFinite++;
  }
  const name = rp.split("/").slice(-3, -1).join("/");
  console.log(`${name}\n  frames posed: ${posed}  bone comparisons: ${pairs}`
    + `\n  worst relative bone-length change: ${worst.toExponential(2)}`
    + `\n  non-finite values: ${nonFinite}`);
}
