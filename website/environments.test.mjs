import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const source = await readFile(new URL("./environments.js", import.meta.url), "utf8");
const catalog = await readFile(new URL("./environments/catalog.js", import.meta.url), "utf8");
const mountEnd = source.indexOf("  (globalThis.WeftEnvironments");
assert.ok(mountEnd > 0);

function viewer() {
  function element() {
    return {
      attributes: {}, listeners: {}, children: [], hidden: false,
      setAttribute(key, value) { this.attributes[key] = value; },
      addEventListener(event, listener) { this.listeners[event] = listener; },
      append(child) { this.children.push(child); },
      removeAttribute(key) { delete this[key]; },
      load() {}, remove() {}, style: {},
    };
  }
  const nodes = new Map();
  const node = (selector) => {
    if (!nodes.has(selector)) nodes.set(selector, element());
    return nodes.get(selector);
  };
  const controls = [".scene-left", ".scene-right", ".scene-reset"].map(node);
  const card = {
    ...element(),
    querySelector: node,
    querySelectorAll: () => controls,
  };
  node(".world-screen > svg").querySelector = node;
  const video = node("video");
  video.paused = true;
  video.currentTime = 2;
  video.pause = () => {
    if (video.paused) return;
    video.paused = true;
    video.listeners.pause();
  };
  video.play = () => {
    video.paused = false;
    video.listeners.play();
    return Promise.resolve();
  };
  const frames = new Map();
  let nextFrame = 0, visibility;
  const context = {
    document: {
      hidden: false, querySelector: node,
      createElement: (tag) => tag === "article" ? card : element(),
    },
    requestAnimationFrame: (callback) => { frames.set(++nextFrame, callback); return nextFrame; },
    cancelAnimationFrame: (id) => frames.delete(id),
    IntersectionObserver: class {
      constructor(callback) { visibility = callback; }
      observe() {} disconnect() {}
    },
  };
  vm.runInNewContext(catalog, context);
  vm.runInNewContext(source.slice(0, mountEnd) + "globalThis.mountWorld = mountWorld;})();", context);
  const record = context.WeftEnvironments[0];
  const cleanup = context.mountWorld(element(), record);
  return { node, card, controls, video, frames, record, cleanup,
    visibility: (visible) => visibility([{ isIntersecting: visible }]) };
}

test("landing shows only the reconstructed model while preserving interaction", () => {
  const { node, record, card } = viewer();
  assert.doesNotMatch(card.innerHTML, /world-version|version-initial|version-reconstructed/);
  assert.match(card.innerHTML, /world-toggle/);
  assert.equal(node('.world-ball').attributes.r, record.models[0].world.radius * record.width);
  node('.world-toggle').listeners.click();
  assert.equal(node('.world-toggle').attributes['aria-pressed'], 'true');
});

test("comparison and interaction expose only their controls and stop hidden playback", () => {
  const { node, controls, video, frames, visibility, cleanup } = viewer();
  assert.equal(node(".world-transport").hidden, false);
  assert.ok(controls.every((control) => control.hidden));
  node(".world-play").listeners.click();
  assert.equal(video.paused, false);
  assert.equal(frames.size, 1);
  node(".world-toggle").listeners.click();
  assert.equal(video.paused, true);
  assert.equal(frames.size, 0);
  assert.equal(node(".world-transport").hidden, true);
  assert.ok(controls.every((control) => !control.hidden));
  assert.match(node(".world-note").textContent, /authored additions/);
  node(".scene-left").listeners.click();
  assert.equal(frames.size, 1);
  visibility(false);
  assert.equal(frames.size, 0);
  node(".scene-left").listeners.click();
  assert.equal(frames.size, 0, "offscreen controls cannot start animation");
  visibility(true);
  video.play();
  assert.equal(node(".world-toggle").attributes["aria-pressed"], "false", "video playback returns to comparison");
  assert.equal(node(".world-toggle").title, "Try controls");
  node(".world-toggle").listeners.click();
  assert.equal(node(".world-toggle").title, "Back to video comparison");
  node(".world-toggle").listeners.click();
  assert.equal(node(".world-transport").hidden, false);
  assert.ok(controls.every((control) => control.hidden));
  cleanup();
  assert.equal(frames.size, 0);
  assert.equal(video.src, undefined);
});


test("rounded media timestamps still display the first observed ball position", () => {
  const { video, node, record } = viewer();
  video.currentTime = record.segments[0].start - 0.000001;
  video.listeners.seeked();
  assert.equal(node(".world-body").style.opacity, "1");
  assert.equal(node(".world-ball").attributes.cx, record.segments[0].x0 * record.width);
  video.currentTime = record.segments[0].start - 0.1;
  video.listeners.seeked();
  assert.equal(node(".world-body").style.opacity, "0", "unobserved gaps remain empty");
});
