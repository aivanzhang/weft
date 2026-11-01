const $ = (id) => document.getElementById(id);
const model = $("model");
const video = $("video");
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
const state = { assets: [], asset: null, clip: null, mode: "model", ready: false, loadedModel: null, version: 0 };
let raf = 0;
let lastPaint = 0;
let loadTimer;

const label = (name) => ({ idle: "Idle", making_something: "Crafting", milestone_level_up: "Celebrate", working: "Working" }[name] || name.replaceAll("_", " "));
const player = () => state.mode === "model" ? model : video;
const duration = () => Number.isFinite(player().duration) ? player().duration : 0;
const playing = () => state.mode !== "portrait" && state.ready && !player().paused;
const initialView = () => state.asset?.initial_view || { azimuth_degrees: 25, elevation_degrees: 75, radius_percent: 125 };

function resetCamera() {
  const view = initialView();
  model.cameraOrbit = `${view.azimuth_degrees}deg ${view.elevation_degrees}deg ${view.radius_percent}%`;
  model.cameraTarget = "auto auto auto";
}

function setError(message = "") {
  $("error").textContent = message;
  $("error").hidden = !message;
  if (message) $("loading").hidden = true;
}

function stopAll() {
  model.pause?.();
  video.pause();
  cancelAnimationFrame(raf);
  raf = 0;
  paintTransport();
}

function seekModel(seconds) {
  // v4.3.1 scales seeks by mixer speed and retains a finished action's paused flag.
  model.timeScale = 1;
  model.play({ repetitions: 1 });
  model.pause();
  model.currentTime = Math.max(0, Math.min(seconds, model.duration));
  model.timeScale = Number($("speed").value);
}

function paintTransport() {
  const active = playing();
  const seconds = state.mode === "portrait" ? 0 : duration();
  const position = state.mode === "portrait" ? 0 : Math.min(Number(player().currentTime) || 0, seconds);
  $("play-label").textContent = active ? "Pause" : "Play";
  $("play-icon").textContent = active ? "Ⅱ" : "▶";
  $("play").setAttribute("aria-label", active ? "Pause animation" : "Play animation");
  $("play").disabled = !state.ready || !state.clip || state.mode === "portrait";
  $("seek").disabled = $("play").disabled;
  $("seek").max = String(seconds || 1);
  $("seek").value = String(position);
  $("seek").setAttribute("aria-valuetext", `${position.toFixed(1)} of ${seconds.toFixed(1)} seconds`);
  $("time").textContent = `${position.toFixed(1)} / ${seconds.toFixed(1)} s`;
  $("duration-note").textContent = seconds ? `${seconds.toFixed(1)} seconds · ${state.mode === "model" ? "Skeletal animation" : "Video"}` : state.mode === "model" ? "Static 3D model" : "Static image";
}

function tick(now) {
  if (now - lastPaint > 100) {
    paintTransport();
    lastPaint = now;
  }
  raf = playing() && !document.hidden ? requestAnimationFrame(tick) : 0;
}

async function togglePlay() {
  if (!state.ready || !state.clip || state.mode === "portrait") return;
  if (playing()) return stopAll();
  const version = state.version;
  try {
    if (player().currentTime >= duration() - .02) {
      if (state.mode === "model") seekModel(0);
      else video.currentTime = 0;
    }
    if (state.mode === "model") model.play({ repetitions: $("repeat").checked ? Infinity : 1 });
    else await video.play();
    if (version !== state.version) return;
    paintTransport();
    if (!raf) raf = requestAnimationFrame(tick);
  } catch (error) {
    if (version !== state.version || error.name === "AbortError") return;
    setError("Playback could not start. Try another state or the portrait view.");
  }
}

function beginLoading(message) {
  clearTimeout(loadTimer);
  state.ready = false;
  $("loading").hidden = false;
  $("loading").textContent = message;
  setError();
  paintTransport();
  loadTimer = setTimeout(() => setError("Loading is taking longer than expected. Try the portrait or another asset."), 20000);
}

async function chooseState(id) {
  stopAll();
  const version = ++state.version;
  state.clip = state.asset.states.find((clip) => clip.id === id) || state.asset.states[0] || null;
  for (const button of $("state-list").children) button.setAttribute("aria-pressed", String(button.dataset.state === state.clip?.id));
  $("state-meaning").textContent = state.clip?.meaning || "No animation is included in this asset.";
  if (state.mode === "model" && model.loaded && state.loadedModel === state.asset.model) {
    if (!state.clip) {
      state.ready = true;
      setError();
      clearTimeout(loadTimer);
      $("loading").hidden = true;
    } else if (!model.availableAnimations.includes(state.clip.id)) {
      state.ready = false;
      setError("This state is not present in the model. Choose another asset or view its saved media.");
    } else {
      state.ready = false;
      paintTransport();
      model.animationName = state.clip.id;
      await model.updateComplete;
      if (version !== state.version || state.mode !== "model") return;
      seekModel(0);
      state.ready = true;
      setError();
      clearTimeout(loadTimer);
      $("loading").hidden = true;
    }
  } else if (state.mode === "video") {
    const path = state.clip?.video;
    if (path) {
      beginLoading("Loading rendered clip…");
      video.poster = state.clip.poster || state.asset.poster;
      video.src = path;
      video.load();
    } else {
      state.ready = false;
      setError("No rendered video is included for this state. Use the interactive 3D model.");
    }
  }
  paintTransport();
}

function setMode(mode) {
  if (!state.asset) return;
  stopAll();
  ++state.version;
  clearTimeout(loadTimer);
  setError();
  state.mode = mode;
  model.hidden = mode !== "model";
  video.hidden = mode !== "video";
  $("portrait").hidden = mode !== "portrait";
  $("camera-tools").hidden = $("orbit-hint").hidden = mode !== "model";
  $("preview-badge").textContent = { model: "INTERACTIVE 3D", video: "RENDERED VIDEO", portrait: "STATIC PORTRAIT" }[mode];
  for (const button of document.querySelectorAll("[data-mode]")) button.setAttribute("aria-pressed", String(button.dataset.mode === mode));
  if (mode !== "video") {
    video.removeAttribute("src");
    video.load();
  }
  if (mode === "model") {
    if (model.getAttribute("src") !== state.asset.model) {
      beginLoading("Loading 3D model…");
      model.setAttribute("src", state.asset.model);
    } else if (!model.loaded || state.loadedModel !== state.asset.model) beginLoading("Loading 3D model…");
    chooseState(state.clip?.id);
  } else if (mode === "video") chooseState(state.clip?.id);
  else {
    state.ready = false;
    $("loading").hidden = true;
    $("portrait").src = state.asset.portrait || state.asset.poster;
    $("portrait").alt = `${state.asset.title} portrait`;
    chooseState(state.clip?.id);
  }
}

function chooseAsset(id) {
  const asset = state.assets.find((item) => item.id === id);
  if (!asset) return;
  stopAll();
  state.asset = asset;
  state.ready = false;
  if (model.getAttribute("src") !== asset.model) {
    state.loadedModel = null;
    model.removeAttribute("src");
  }
  model.alt = `${asset.title}, interactive 3D model. Drag to rotate and scroll to zoom.`;
  resetCamera();
  $("asset-title").textContent = asset.title;
  $("asset-kind").textContent = { reference_2d: "ORIGINAL REFERENCE · 2D ONLY", generated_3d: "GENERATED 3D CANDIDATE", authored_3d: "AUTHORED 3D COLLECTION" }[asset.kind];
  $("description").textContent = asset.description;
  $("notice").textContent = asset.notice;
  for (const button of $("asset-list").children) button.setAttribute("aria-pressed", String(button.dataset.asset === id));
  for (const [key, element] of [["model", $("download-model")], ["manifest", $("download-manifest")]]) {
    element.hidden = !asset.downloads[key];
    if (asset.downloads[key]) {
      element.href = asset.downloads[key];
      element.download = key === "model" ? `${asset.id}.glb` : `${asset.id}-manifest.json`;
    }
    else element.removeAttribute("href");
  }
  $("facts").replaceChildren();
  for (const [key, title] of [["vertices", "Vertices"], ["bones", "Rig joints"], ["animations", "States"]]) {
    if (asset.facts[key] == null) continue;
    const group = document.createElement("div");
    const term = document.createElement("dt");
    const value = document.createElement("dd");
    term.textContent = title;
    value.textContent = Number(asset.facts[key]).toLocaleString();
    group.append(term, value);
    $("facts").append(group);
  }
  $("state-list").replaceChildren();
  for (const clip of asset.states) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.state = clip.id;
    button.textContent = asset.kind === "reference_2d" ? clip.label : label(clip.id);
    button.setAttribute("aria-pressed", "false");
    button.addEventListener("click", () => chooseState(clip.id));
    $("state-list").append(button);
  }
  const hasVideo = asset.states.some((clip) => clip.video);
  document.querySelector('[data-mode="model"]').disabled = !asset.model;
  document.querySelector('[data-mode="video"]').disabled = !hasVideo;
  state.clip = asset.states[0] || null;
  let mode = state.mode;
  if (reducedMotion.matches) mode = "portrait";
  else if (!asset.model) mode = hasVideo ? "video" : "portrait";
  else if (mode === "video" && !hasVideo) mode = "model";
  setMode(mode);
  history.replaceState(null, "", `#${encodeURIComponent(id)}`);
}

function renderLibrary() {
  $("asset-count").textContent = state.assets.length;
  for (const asset of state.assets) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "asset-card";
    button.dataset.asset = asset.id;
    button.setAttribute("aria-pressed", "false");
    const image = document.createElement("img");
    image.src = asset.poster;
    image.alt = "";
    image.loading = "lazy";
    const copy = document.createElement("span");
    const title = document.createElement("strong");
    const subtitle = document.createElement("small");
    title.textContent = asset.title;
    subtitle.textContent = asset.model ? `3D · ${asset.states.length} states` : "Reference · 2D media";
    copy.append(title, subtitle);
    button.append(image, copy);
    button.addEventListener("click", () => chooseAsset(asset.id));
    $("asset-list").append(button);
  }
}

model.addEventListener("load", async (event) => {
  if (!state.asset?.model || new URL(event.detail.url, location.href).pathname !== state.asset.model) return;
  const asset = state.asset;
  state.loadedModel = state.asset.model;
  resetCamera();
  await model.updateComplete;
  if (state.asset !== asset) return;
  model.jumpCameraToGoal();
  if (state.mode === "model") chooseState(state.clip?.id);
});
model.addEventListener("error", () => {
  if (state.mode === "model") {
    state.ready = false;
    clearTimeout(loadTimer);
    setError("The 3D model could not load. Portraits and saved videos are still available.");
    paintTransport();
  }
});
model.addEventListener("finished", stopAll);
video.addEventListener("loadedmetadata", () => {
  if (state.mode !== "video") return;
  state.ready = true;
  setError();
  clearTimeout(loadTimer);
  $("loading").hidden = true;
  video.playbackRate = Number($("speed").value);
  paintTransport();
});
video.addEventListener("error", () => {
  if (state.mode === "video" && video.getAttribute("src")) {
    state.ready = false;
    clearTimeout(loadTimer);
    setError("This video could not load. Try the portrait or 3D view.");
    paintTransport();
  }
});
video.addEventListener("ended", stopAll);
$("play").addEventListener("click", togglePlay);
$("seek").addEventListener("input", () => {
  const position = Number($("seek").value);
  stopAll();
  if (state.mode === "model") seekModel(position);
  else video.currentTime = position;
  paintTransport();
});
$("speed").addEventListener("change", () => {
  model.timeScale = video.playbackRate = Number($("speed").value);
});
$("repeat").addEventListener("change", () => {
  video.loop = $("repeat").checked;
  if (state.mode === "model" && playing()) model.play({ repetitions: $("repeat").checked ? Infinity : 1 });
});
for (const button of document.querySelectorAll("[data-mode]")) button.addEventListener("click", () => setMode(button.dataset.mode));
for (const button of document.querySelectorAll("[data-view]")) button.addEventListener("click", () => {
  model.cameraOrbit = `${{ front: 0, side: 90, back: 180 }[button.dataset.view]}deg 80deg ${initialView().radius_percent}%`;
  if (reducedMotion.matches) model.jumpCameraToGoal();
});
$("reset-camera").addEventListener("click", () => {
  resetCamera();
  if (reducedMotion.matches) model.jumpCameraToGoal();
});
$("fullscreen").addEventListener("click", async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await document.querySelector(".preview-shell").requestFullscreen();
  } catch { setError("Fullscreen is unavailable in this browser."); }
});
$("search").addEventListener("input", () => {
  const query = $("search").value.toLocaleLowerCase();
  for (const button of $("asset-list").children) button.hidden = !button.textContent.toLocaleLowerCase().includes(query);
  $("empty-search").hidden = [...$("asset-list").children].some((button) => !button.hidden);
});
document.addEventListener("visibilitychange", () => { if (document.hidden) stopAll(); });
new IntersectionObserver(([entry]) => { if (!entry.isIntersecting) stopAll(); }).observe($("stage"));
reducedMotion.addEventListener("change", () => { if (reducedMotion.matches && state.asset) setMode("portrait"); });
document.addEventListener("keydown", (event) => {
  if (event.code === "Space" && !event.target.closest("input, button, select, a, model-viewer")) {
    event.preventDefault();
    togglePlay();
  }
});
$("motion-note").textContent = reducedMotion.matches ? "Reduced motion: portrait selected. Playback is always your choice." : "Playback starts paused. Repeating does not certify a seamless loop.";

try {
  const response = await fetch("/api/catalog");
  if (!response.ok) throw new Error("Catalog unavailable");
  const catalog = await response.json();
  state.assets = catalog.assets;
  if (!state.assets.length) throw new Error("No assets configured");
  await import("/vendor/model-viewer.min.js");
  await customElements.whenDefined("model-viewer");
  // Keep one GPU scene and no background playback when switching the collection.
  customElements.get("model-viewer").modelCacheSize = 1;
  renderLibrary();
  chooseAsset(decodeURIComponent(location.hash.slice(1)) || state.assets[0].id);
  if (!state.asset) chooseAsset(state.assets[0].id);
} catch (error) {
  $("asset-title").textContent = "Library unavailable";
  setError(`Could not load the local viewer. ${error.message}. Check the terminal running the gallery.`);
}
