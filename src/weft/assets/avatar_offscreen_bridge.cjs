"use strict";
const math = require("./avatar_retarget.js");
const MAX_INPUT = 36 * 1024 * 1024, MAX_OUTPUT = 16 * 1024 * 1024;
let config = null, world = null, assets = new Map(), width = 0, height = 0, requests = 0;
let chunks = [], bytes = 0;
const fail = message => { throw new Error(message); };
const vector = (value, size, label, limit = 100) => {
  if (!Array.isArray(value) || value.length !== size || value.some(v => typeof v !== "number" || !Number.isFinite(v) || Math.abs(v) > limit)) fail(`Invalid ${label}`);
  return value;
};
function send(value) {
  const raw = JSON.stringify(value);
  if (Buffer.byteLength(raw) > MAX_OUTPUT) fail("Avatar bridge output exceeds its byte bound");
  process.stdout.write(raw + "\n");
}
function settings(value) {
  if (!value || Object.keys(value).sort().join() !== "asset,camera,overlay,show_muscles"
      || typeof value.overlay !== "boolean" || typeof value.show_muscles !== "boolean"
      || (value.asset !== null && !assets.has(value.asset))) fail("Invalid avatar render settings");
  const camera = value.camera;
  if (!camera || Object.keys(camera).sort().join() !== "follow,yaw,zoom" || typeof camera.follow !== "boolean"
      || !Number.isFinite(camera.yaw) || Math.abs(camera.yaw) > 100
      || !Number.isFinite(camera.zoom) || camera.zoom < 1 || camera.zoom > 3) fail("Invalid recorded avatar camera");
  return value;
}
function handle(message) {
  if (++requests > 18002) fail("Avatar bridge frame count exceeded");
  if (message.type === "init") {
    if (config) fail("Avatar bridge cannot replace its scene or asset catalog");
    if (!Number.isInteger(message.width) || !Number.isInteger(message.height)
        || message.width < 3 || message.height < 1 || message.width > 2048 || message.height > 2048) fail("Invalid avatar viewport");
    if (!message.assets || Array.isArray(message.assets) || Object.keys(message.assets).length > 8) fail("Invalid bounded avatar catalog");
    config = math.compileScene(message.scene); world = math.prepareWorld(config);
    width = message.width; height = message.height;
    for (const [key, value] of Object.entries(message.assets)) {
      if (!/^[0-9a-f]{64}$/.test(key)) fail("Avatar catalog keys must be exact hashes");
      assets.set(key, math.prepareAsset(value));
    }
    send({type: "ready", world: world.meshes.map(mesh => ({id: mesh.id, color: mesh.color,
      positions: Array.from(mesh.positions), normals: Array.from(mesh.normals), indices: Array.from(mesh.indices)}))});
    return;
  }
  if (message.type !== "frame" || !config) fail("Avatar bridge expects initialization followed by frames");
  const selected = settings(message.settings), body = math.bodyFrames(config, message.snapshot);
  const pose = selected.asset === null ? null : math.retarget(assets.get(selected.asset), body);
  const matrices = math.worldMatrices(world, body.state);
  const segments = body.state.human.segments || [], muscles = body.state.human.muscles || [];
  if (segments.length > 32 || muscles.length > 64) fail("Avatar overlay exceeds its element bound");
  const segmentRows = segments.map(segment => {
    if (!Number.isFinite(segment.radius) || segment.radius <= 0 || segment.radius > 1) fail("Invalid human segment radius");
    return {a: vector(segment.a, 3, "segment start"), b: vector(segment.b, 3, "segment end"), radius: segment.radius};
  });
  const muscleRows = selected.show_muscles ? muscles.map(muscle => {
    if (!Number.isFinite(muscle.activation) || muscle.activation < 0 || muscle.activation > 1) fail("Invalid illustrative muscle drive");
    return {a: vector(muscle.a, 3, "muscle start"), b: vector(muscle.b, 3, "muscle end"), activation: muscle.activation};
  }) : [];
  send({type: "frame", nodes: pose ? Array.from(pose.nodes.entries()) : [], palette: pose ? pose.palette : [],
    world_matrices: matrices,
    views: math.cameraViews(config, body.state, selected.camera, width, height, pose ? math.posedBounds(assets.get(selected.asset), pose, world, matrices,body.state) : null),
    segments: segmentRows, muscles: muscleRows, controlled_human: body.state.controlled_id === "human"});
}
process.stdin.on("data", chunk => {
  let start = 0;
  try {
    while (start < chunk.length) {
      const end = chunk.indexOf(10, start), stop = end < 0 ? chunk.length : end;
      const part = chunk.subarray(start, stop); bytes += part.length;
      if (bytes > MAX_INPUT) fail("Avatar bridge input exceeds its byte bound");
      chunks.push(part);
      if (end < 0) break;
      const raw = Buffer.concat(chunks, bytes); chunks = []; bytes = 0;
      handle(JSON.parse(raw.toString("utf8")));
      start = end + 1;
    }
  } catch (error) {
    send({type: "error", message: String(error.message).slice(0, 2000)});
    process.stdin.pause(); process.exitCode = 1; process.stdin.destroy();
  }
});
process.stdin.on("end", () => { if (bytes) process.exitCode = 1; });
