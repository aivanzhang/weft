import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const source = fs.readFileSync(new URL("../src/weft/assets/lab_appearance.js", import.meta.url), "utf8");
const tick = () => new Promise(resolve => setTimeout(resolve, 0));
const pending = () => { let resolve; const promise = new Promise(done => { resolve = done; }); return {promise, resolve}; };
class Element {
  constructor() { this.hidden = false; this.checked = true; this.value = ""; this.textContent = ""; this.children = []; this.listeners = {}; }
  append(value) { this.children.push(value); }
  replaceChildren() { this.children = []; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  removeEventListener(name) { delete this.listeners[name]; }
}
function setup() {
  const elements = new Map(), $ = id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
  const sends = [], loads = [], frames = [], requests = [], messages = [], pauses = [];
  let resolveLoad = null, failFetch = false, pendingRead = null, viewerStatus;
  const viewer = {configure() {}, setVisible(value) { messages.push(value); }, update(value) { frames.push(value); }, dispose() {},
    load(value) { loads.push(value); if (resolveLoad) return resolveLoad.promise; return Promise.resolve({status: "ready"}); }};
  const context = {document: {getElementById: $, createElement: () => new Element()}, setTimeout, clearTimeout,
    AbortController, Uint8Array, ArrayBuffer, WeftAvatarViewer: {create(_canvas, {onStatus}) { viewerStatus = onStatus; return viewer; }},
    fetch: async (url, options) => {
      requests.push({url, options});
      if (failFetch) return {ok: false};
      let used = false;
      return {ok: true, headers: {get: () => "4"}, body: {getReader: () => ({
        read: async () => { if (pendingRead) await pendingRead.promise; if (options.signal.aborted) throw new Error("aborted"); if (used) return {done: true}; used = true; return {done: false, value: new Uint8Array([1, 2, 3, 4])}; },
        cancel: async () => {}, releaseLock() {},
      })}};
    },
  };
  vm.runInNewContext(source, context);
  const settings = {asset: null, overlay: true, show_muscles: true, camera: {yaw: .5, zoom: 1, follow: false}};
  const entries = ["a", "b"].map(key => ({key, display_name: key.toUpperCase(), render_sha256: "0".repeat(64), textures: 1}));
  const api = context.WeftLabAppearance.create({token: "local-only", pause: () => pauses.push(true), redraw() {},
    camera: () => ({yaw: .5, zoom: 1, follow: false}), visible: () => true, loadModules: async () => {},
    send: async value => { sends.push(value); return {settings: value}; }});
  api.configure({settings, choices: entries}, {profile: "shared-world-v2"});
  const choose = async key => { $("avatar-select").value = key; $("avatar-select").onchange(); await tick(); };
  return {$, api, choose, sends, loads, frames, requests, pauses,
    deferLoad() { resolveLoad = pending(); return resolveLoad; },
    deferRead() { pendingRead = pending(); return pendingRead; },
    failFetch() { failFetch = true; }, status(value) { viewerStatus(value); }};
}

{
  const h = setup();
  assert.equal(h.requests.length, 0);
  assert.equal(h.$("live").hidden, false);
  await h.choose("a"); await h.api.flush();
  assert.equal(h.$("avatar-live").hidden, false);
  assert.equal(h.$("live").hidden, true);
  assert.equal(h.requests.length, 2);
  assert.equal(h.requests[0].options.headers.Authorization, "Bearer local-only");
  h.api.update({human: {id: "same-snapshot"}});
  assert.equal(h.frames[0].snapshot.human.id, "same-snapshot");
  const reads = h.requests.length;
  h.$("avatar-overlay").checked = false; h.$("avatar-overlay").onchange(); await h.api.flush();
  assert.equal(h.requests.length, reads);
  h.status({status: "fallback", reason: "Body frames do not match"});
  assert.equal(h.$("live").hidden, false);
  assert.match(h.$("avatar-status").textContent, /Body frames do not match/);
  h.status({status: "ready", reason: "stale acknowledgment"});
  assert.equal(h.$("live").hidden, false);
  h.api.dispose();
  console.log("PASS verified load, same-state updates, overlay reuse and explicit fallback");
}
{
  const h = setup(), delayed = h.deferLoad();
  await h.choose("a");
  h.$("avatar-original").onclick(); await tick();
  assert.equal(h.$("live").hidden, false);
  assert.equal(h.sends.at(-1).asset, null);
  delayed.resolve({status: "ready"}); await h.api.flush(); await tick();
  assert.equal(h.$("avatar-live").hidden, true);
  assert.equal(h.$("live").hidden, false);
  h.api.dispose();
  console.log("PASS immediate original-body return cancels stale avatar reveal");
}
{
  const h = setup(), delayed = h.deferRead();
  await h.choose("a");
  h.$("avatar-original").onclick(); await tick();
  assert.equal(h.requests[0].options.signal.aborted, true);
  delayed.resolve(); await h.api.flush(); await tick();
  assert.equal(h.loads.length, 0);
  assert.equal(h.$("avatar-live").hidden, true);
  h.api.dispose();
  console.log("PASS obsolete byte transfers are aborted before worker load");
}
{
  const h = setup(); h.failFetch(); await h.choose("a"); await h.api.flush();
  assert.equal(h.$("live").hidden, false);
  assert.match(h.$("avatar-status").textContent, /resource is unavailable/);
  assert.equal(h.sends.length, 1);
  h.api.dispose();
  console.log("PASS missing media stays explicit without resetting the simulation");
}
