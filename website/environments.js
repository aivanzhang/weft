(() => {
  "use strict";
  const lab = document.querySelector("#experiment");
  const viewers = new Set();
  const number = (value) => Number(value).toFixed(1);
  const icons = { play: "▷ Play pair", pause: "Ⅱ Pause" };

  function mountWorld(host, record, index = "01") {
    const card = document.createElement("article");
    card.className = "world-card";
    // This template is static; submitted names and server data only enter text nodes.
    card.innerHTML = `<header class="world-caption"><span class="world-number"></span><h3></h3><span>VIDEO → ENVIRONMENT</span></header>
      <div class="world-pair">
        <div class="world-panel"><div class="world-screen"><video controls muted playsinline preload="metadata" aria-label="Input video"></video></div><div class="world-panel-label"><span>Input video</span><span>Observed</span></div></div>
        <div class="world-panel"><div class="world-screen">
          <svg viewBox="0 0 640 360" role="img" aria-label="Reconstructed rolling environment"><defs><radialGradient id="ball-${index}" cx="30%" cy="25%" r="75%"><stop stop-color="#fae486"/><stop offset=".5" stop-color="#eac34c"/><stop offset="1" stop-color="#b58c23"/></radialGradient></defs><rect width="640" height="360" fill="#eee9e0"/><path class="world-ground" fill="#e1dacd"/><path class="world-surface" fill="none" stroke="#bcb5a7" stroke-width="1.2"/><g class="world-body"><ellipse class="world-shadow" fill="#214f50" opacity=".09"/><circle class="world-ball" fill="url(#ball-${index})"/></g></svg>
          <span class="scene-empty" hidden>Between observed passes</span>
          <button class="scene-control scene-left" type="button" aria-label="Push left" title="Push left" hidden>←</button><button class="scene-control scene-right" type="button" aria-label="Push right" title="Push right" hidden>→</button><button class="scene-control scene-reset" type="button" aria-label="Reset environment" title="Reset" hidden>↺</button>
        </div><div class="world-panel-label"><label><span class="sr-only">Simulation model</span><select class="model-select"></select></label><span class="world-state">Simulated</span></div></div>
      </div>
      <div class="world-transport"><button type="button" class="world-play">▷ Play pair</button><input class="world-seek" type="range" min="0" step="0.01" value="0" aria-label="Replay position"/><span class="world-time"></span><div class="world-modes" role="group" aria-label="Environment mode"><button type="button" data-mode="replay" aria-pressed="true">Replay</button><button type="button" data-mode="interact" aria-pressed="false">Interact</button></div></div>
      <p class="world-note"></p><div class="world-source"><span class="world-attribution"></span><a class="world-download" hidden>Download environment ↓</a></div>`;
    host.append(card);
    const $ = (selector) => card.querySelector(selector);
    $(".world-number").textContent = index;
    $("h3").textContent = record.title;
    const video = $("video");
    video.src = record.video;
    if (record.poster) video.poster = record.poster;
    const seek = $(".world-seek");
    seek.max = record.duration;
    const select = $(".model-select");
    record.models.forEach((model, i) => {
      const option = new Option(model.label, i);
      select.add(option);
    });
    const attribution = $(".world-attribution");
    if (record.attribution) {
      const a = document.createElement("a");
      a.href = "https://hubbs.schule/mediathek/rollender-ball";
      a.textContent = "Ecole Science · Rollender Ball";
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      attribution.append(a, " · CC0");
    } else attribution.textContent = "Your video · private result";
    if (record.download) {
      $(".world-download").hidden = false;
      $(".world-download").href = record.download;
    }
    let model = record.models[0].world;
    let mode = "replay",
      visible = true,
      raf = null,
      last = null;
    let x = model.x0,
      velocity = 0,
      force = 0,
      forceLeft = 0,
      elapsed = 0;
    const controls = card.querySelectorAll(".scene-control");
    const stop = () => {
      if (raf !== null) cancelAnimationFrame(raf);
      raf = null;
      last = null;
    };
    function pause() {
      video.pause();
      stop();
      force = forceLeft = velocity = 0;
    }
    const scene = $(".world-screen > svg");
    const body = $(".world-body"),
      circle = $(".world-ball"),
      shadow = $(".world-shadow"),
      empty = $(".scene-empty");
    scene.setAttribute("viewBox", `0 0 ${record.width} ${record.height}`);
    scene.querySelector("rect").setAttribute("width", record.width);
    scene.querySelector("rect").setAttribute("height", record.height);
    function geometry() {
      const width = record.width,
        height = record.height;
      const a = model.support_intercept * width;
      const b = (model.support_intercept + model.support_slope) * width;
      $(".world-ground").setAttribute(
        "d",
        `M0 ${a}L${width} ${b}V${height}H0Z`,
      );
      $(".world-surface").setAttribute("d", `M0 ${a}L${width} ${b}`);
      const radius = model.radius * width;
      circle.setAttribute("r", radius);
      shadow.setAttribute("rx", radius * 1.05);
      shadow.setAttribute("ry", Math.max(2, radius * 0.09));
    }
    function draw(position, show = true) {
      const cx = position * record.width;
      const cy =
        (model.support_intercept +
          model.support_slope * position -
          model.radius) *
        record.width;
      body.style.opacity = show ? "1" : "0";
      circle.setAttribute("cx", cx);
      circle.setAttribute("cy", cy);
      shadow.setAttribute("cx", cx);
      shadow.setAttribute("cy", cy + model.radius * record.width);
      empty.hidden = show;
    }
    function replay() {
      const t = video.currentTime;
      seek.value = t;
      $(".world-time").textContent =
        `${number(t)} / ${number(record.duration)}s`;
      const segment = record.segments.find((s) => t >= s.start && t <= s.end);
      if (!segment) {
        draw(model.x0, false);
        return;
      }
      const v = segment.primary
        ? model.v0
        : segment.v0 + 0.5 * model.drag * segment.initialization_dt;
      const dt = Math.min(
        t - segment.start,
        model.drag > 0 ? Math.abs(v) / model.drag : Infinity,
      );
      draw(segment.x0 + v * dt - 0.5 * Math.sign(v) * model.drag * dt * dt);
    }
    function frame(now) {
      raf = null;
      if (document.hidden || !visible) {
        pause();
        return;
      }
      if (mode === "replay") {
        replay();
        if (!video.paused) raf = requestAnimationFrame(frame);
        return;
      }
      const dt = last === null ? 0 : Math.min((now - last) / 1000, 0.05);
      last = now;
      if (dt > 0) {
        const driven = Math.min(dt, forceLeft);
        if (driven > 0) {
          [x, velocity] = WeftPhysics.advance(
            model,
            x,
            velocity,
            force,
            driven,
          );
          forceLeft -= driven;
        }
        if (dt > driven)
          [x, velocity] = WeftPhysics.advance(
            model,
            x,
            velocity,
            0,
            dt - driven,
          );
        elapsed += dt;
        if (x <= model.radius || x >= 1 - model.radius || elapsed >= 8) {
          x = Math.max(model.radius, Math.min(1 - model.radius, x));
          velocity = forceLeft = 0;
        }
        draw(x);
      }
      if (Math.abs(velocity) > 0.0001 || forceLeft > 0)
        raf = requestAnimationFrame(frame);
    }
    function setMode(next) {
      pause();
      mode = next;
      geometry();
      card
        .querySelectorAll("[data-mode]")
        .forEach((b) =>
          b.setAttribute("aria-pressed", String(b.dataset.mode === mode)),
        );
      controls.forEach((c) => (c.hidden = mode !== "interact"));
      seek.disabled = mode !== "replay";
      $(".world-play").disabled = mode !== "replay";
      $(".world-note").textContent =
        mode === "replay"
          ? `Same source, two model stages. ${record.models[select.value].label}: ${number(record.models[select.value].error_px)} px error on development frames. One-dimensional reconstruction; not recovered 3D geometry.`
          : "Use the arrows in the environment to change the motion. Push controls and the parking task are authored additions, not actions recovered from the video.";
      $(".world-state").textContent =
        mode === "interact" ? "Your actions" : "Simulated";
      if (mode === "replay") replay();
      else {
        x = 0.5;
        draw(x);
      }
    }
    $(".world-play").addEventListener("click", () => {
      if (!video.paused) video.pause();
      else {
        if (video.ended) video.currentTime = 0;
        video.play().catch(() => {
          $(".world-note").textContent =
            "Playback could not start. Use the video controls to try again.";
        });
      }
    });
    video.addEventListener("play", () => {
      if (mode !== "replay") {
        setMode("replay");
        video.play().catch(() => {});
      }
      $(".world-play").textContent = icons.pause;
      if (raf === null) raf = requestAnimationFrame(frame);
    });
    video.addEventListener("pause", () => {
      $(".world-play").textContent = icons.play;
      if (mode === "replay") {
        stop();
        replay();
      }
    });
    video.addEventListener("loadedmetadata", () => {
      video.currentTime = record.segments[0].start;
      replay();
    });
    video.addEventListener("seeked", () => {
      if (mode === "replay") replay();
    });
    video.addEventListener("error", () => {
      $(".world-note").textContent =
        "This video could not load. Reload the page or choose a browser-compatible MP4.";
    });
    seek.addEventListener("input", () => {
      video.currentTime = Number(seek.value);
      replay();
    });
    select.addEventListener("change", () => {
      model = record.models[select.value].world;
      setMode(mode);
    });
    card
      .querySelectorAll("[data-mode]")
      .forEach((b) =>
        b.addEventListener("click", () => setMode(b.dataset.mode)),
      );
    function push(direction) {
      if (document.hidden || !visible) return;
      force = direction;
      forceLeft = 0.18;
      elapsed = 0;
      if (raf === null) raf = requestAnimationFrame(frame);
    }
    $(".scene-left").addEventListener("click", () => push(-1));
    $(".scene-right").addEventListener("click", () => push(1));
    $(".scene-reset").addEventListener("click", () => setMode("interact"));
    const observer = new IntersectionObserver(
      ([entry]) => {
        visible = entry.isIntersecting;
        if (!visible) pause();
      },
      { threshold: 0.05 },
    );
    observer.observe(card);
    viewers.add(pause);
    setMode("replay");
    return () => {
      pause();
      observer.disconnect();
      viewers.delete(pause);
      video.removeAttribute("src");
      video.load();
      card.remove();
    };
  }

  (globalThis.WeftEnvironments || []).forEach((record, index) =>
    mountWorld(
      document.querySelector("#environment-gallery"),
      record,
      String(index + 1).padStart(2, "0"),
    ),
  );
  const disclosure = document.querySelector("#create-environment");
  document.querySelector("#upload-link").addEventListener("click", () => {
    disclosure.open = true;
  });
  const input = document.querySelector("#video-file"),
    form = document.querySelector("#video-submission");
  const consent = document.querySelector("#upload-consent"),
    button = document.querySelector("#generate-environment");
  const status = document.querySelector("#upload-status"),
    preview = document.querySelector("#upload-preview");
  const drop = document.querySelector("#video-drop"),
    progress = document.querySelector("#upload-progress");
  let file = null,
    objectURL = null,
    valid = false,
    ready = false,
    busy = false,
    token = null;
  let pollTimer = null,
    pollAbort = null,
    jobId = null,
    resultCleanup = null,
    selection = 0;
  function message(text, error = false) {
    status.textContent = text;
    status.dataset.error = String(error);
  }
  function enable() {
    button.disabled = !valid || !consent.checked || !ready || busy;
    input.disabled = busy;
  }
  async function api(path, options = {}) {
    const response = await fetch(`/api/${path}`, {
      ...options,
      credentials: "same-origin",
    });
    if (!response.ok) {
      let body;
      try {
        body = await response.json();
      } catch {}
      throw new Error(
        body?.error ||
          `Service unavailable (${response.status}). Try again later.`,
      );
    }
    return response.json();
  }
  let serviceChecked = false;
  async function checkService() {
    if (serviceChecked) return;
    serviceChecked = true;
    if (location.protocol === "file:") {
      message(
        "Video generation is available on the hosted site once its service is connected. You can inspect a clip here; nothing will be uploaded.",
      );
      return;
    }
    try {
      const config = await api("session", {
        signal: AbortSignal.timeout(10000),
      });
      ready = config.enabled;
      token = config.csrf;
      if (!ready)
        message(
          "Video submissions are not open yet. You can still explore the environments above.",
        );
      else message(file ? "Ready when you are." : "Choose a video to begin.");
      if (config.job) {
        jobId = config.job;
        busy = true;
        progress.hidden = false;
        poll();
      }
    } catch {
      message(
        "Video generation is temporarily unavailable. Your video has not been uploaded.",
      );
    }
    enable();
  }
  disclosure.addEventListener("toggle", () => {
    if (disclosure.open) checkService();
    else preview.pause();
  });
  if (disclosure.open) checkService();
  async function choose(next) {
    if (busy) return;
    const currentSelection = ++selection;
    valid = false;
    file = null;
    consent.checked = false;
    enable();
    preview.pause();
    preview.hidden = true;
    preview.removeAttribute("src");
    preview.load();
    if (objectURL) URL.revokeObjectURL(objectURL);
    objectURL = null;
    if (!next) return;
    if (
      !/\.mp4$/i.test(next.name) ||
      next.size <= 0 ||
      next.size > 10 * 1024 * 1024
    ) {
      message("Choose an MP4 smaller than 10 MB.", true);
      return;
    }
    objectURL = URL.createObjectURL(next);
    preview.src = objectURL;
    document.querySelector("#file-label").textContent = next.name;
    try {
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => finish(false), 10000);
        const loaded = () => finish(true),
          failed = () => finish(false);
        function finish(ok) {
          clearTimeout(timeout);
          preview.removeEventListener("loadedmetadata", loaded);
          preview.removeEventListener("error", failed);
          ok ? resolve() : reject();
        }
        preview.addEventListener("loadedmetadata", loaded, { once: true });
        preview.addEventListener("error", failed, { once: true });
      });
      if (currentSelection !== selection) return;
      if (
        !Number.isFinite(preview.duration) ||
        preview.duration <= 0 ||
        preview.duration > 15 ||
        preview.videoWidth > 1920 ||
        preview.videoHeight > 1080
      )
        throw new Error();
      preview.hidden = false;
      valid = true;
      file = next;
      message(
        ready
          ? "Ready. We’ll check the motion before building the environment."
          : "Clip selected. Hosted generation is not connected here; nothing has been uploaded.",
      );
    } catch {
      if (currentSelection === selection)
        message("Use a playable MP4, up to 15 seconds and 1920 × 1080.", true);
    }
    enable();
  }
  input.addEventListener("change", () => choose(input.files[0]));
  consent.addEventListener("change", enable);
  ["dragenter", "dragover"].forEach((name) =>
    drop.addEventListener(name, (event) => {
      event.preventDefault();
      if (!busy) drop.classList.add("dragging");
    }),
  );
  ["dragleave", "drop"].forEach((name) =>
    drop.addEventListener(name, (event) => {
      event.preventDefault();
      drop.classList.remove("dragging");
    }),
  );
  drop.addEventListener("drop", (event) => choose(event.dataTransfer.files[0]));
  function stopPolling() {
    clearTimeout(pollTimer);
    pollTimer = null;
    pollAbort?.abort();
    pollAbort = null;
  }
  async function poll() {
    if (!jobId || document.hidden || pollAbort) return;
    const controller = new AbortController();
    pollAbort = controller;
    try {
      const job = await api(`jobs/${jobId}`, {
        signal: AbortSignal.any([pollAbort.signal, AbortSignal.timeout(12000)]),
      });
      if (job.status === "complete") {
        busy = false;
        progress.hidden = true;
        message(
          "Your environment is ready. Compare it with your input, then try changing an action.",
        );
        const output = document.querySelector("#submitted-environment");
        resultCleanup?.();
        output.hidden = false;
        resultCleanup = mountWorld(output, job.result, "yours");
        jobId = null;
        enable();
        if (disclosure.open)
          output.scrollIntoView({
            behavior: matchMedia("(prefers-reduced-motion: reduce)").matches
              ? "instant"
              : "smooth",
            block: "nearest",
          });
      } else if (job.status === "failed" || job.status === "expired") {
        busy = false;
        progress.hidden = true;
        jobId = null;
        message(
          job.message ||
            "This clip could not produce a supported environment. Try another clip.",
          true,
        );
        enable();
      } else {
        message(
          job.status === "queued"
            ? "Your video is queued. You can leave this page open."
            : "Reconstructing motion, checking the world, and training the task…",
        );
      }
    } catch (error) {
      if (error.name !== "AbortError")
        message(
          "Connection interrupted. Reconnecting to your existing job; no duplicate upload.",
        );
    } finally {
      if (pollAbort === controller) {
        pollAbort = null;
        if (jobId && !document.hidden) pollTimer = setTimeout(poll, 4000);
      }
    }
  }
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (button.disabled || !file) return;
    busy = true;
    enable();
    progress.hidden = false;
    preview.pause();
    message("Uploading your video…");
    try {
      const job = await api("jobs", {
        method: "POST",
        headers: {
          "Content-Type": "video/mp4",
          "X-Weft-CSRF": token,
          "X-Weft-Consent": "yes",
          "Idempotency-Key": crypto.randomUUID(),
        },
        body: file,
        signal: AbortSignal.timeout(60000),
      });
      jobId = job.id;
      poll();
    } catch (error) {
      // A lost response is ambiguous; recover the owned job before allowing a second upload.
      try {
        const session = await api("session", {
          signal: AbortSignal.timeout(10000),
        });
        if (session.job && session.active) {
          jobId = session.job;
          poll();
          return;
        }
      } catch {}
      busy = false;
      progress.hidden = true;
      message(error.message, true);
      enable();
    }
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      viewers.forEach((pause) => pause());
      preview.pause();
      stopPolling();
    } else if (jobId) poll();
  });
  window.addEventListener("pagehide", () => {
    viewers.forEach((pause) => pause());
    preview.pause();
    stopPolling();
  });
  window.addEventListener("pageshow", () => {
    if (jobId) poll();
  });
})();
