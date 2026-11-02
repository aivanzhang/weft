"use strict";
((root) => {
  const importViewer = async () => { await import("/avatar_retarget.js"); await import("/avatar_viewer.js"); };
  function create({send, token, pause, redraw, camera, visible, basePath = "", loadModules = importViewer}) {
    const $ = id => document.getElementById(id);
    const select = $("avatar-select"), overlay = $("avatar-overlay"), canvas = $("avatar-live");
    let scene = null, choices = [], settings = null, viewer = null, latest = null;
    let ready = false, disposed = false, generation = 0, controller = null, cameraTimer = null;
    let queued = null, saving = null, loadTask = null, priorStatus = "";
    const status = message => { if ($("avatar-status").textContent !== message) $("avatar-status").textContent = message; };
    const show = () => { canvas.hidden = !(ready && settings?.asset); $("live").hidden = !canvas.hidden; };
    function viewerStatus(result) {
      // Only the matching load promise may reveal an avatar; stale frame acknowledgments cannot.
      if (result.status === "ready" || result.status === "idle") return;
      const key = `${result.status}:${result.reason || ""}`;
      if (key === priorStatus) return;
      priorStatus = key;
      const wasReady = ready;
      ready = false;
      status(result.status === "fallback" ? `Original body shown: ${result.reason || "avatar unavailable"}. Choose Original body to clear the avatar.`
        : "Loading avatar. The original body remains visible.");
      show();
      if (ready !== wasReady) redraw();
    }
    async function binary(url, signal, maximum) {
      const response = await fetch(url, {headers: {Authorization: `Bearer ${token}`}, signal});
      if (!response.ok) throw new Error("Avatar resource is unavailable");
      const declared = Number(response.headers.get("Content-Length"));
      if (!Number.isInteger(declared) || declared <= 0 || declared > maximum) throw new Error("Avatar response exceeds its byte bound");
      const reader = response.body.getReader(), chunks = [];
      let size = 0;
      try {
        while (true) {
          const part = await reader.read();
          if (part.done) break;
          size += part.value.byteLength;
          if (size > declared) throw new Error("Avatar response exceeds its declared size");
          chunks.push(part.value);
        }
      } catch (error) { await reader.cancel(); throw error; }
      finally { reader.releaseLock(); }
      if (size !== declared) throw new Error("Avatar response is incomplete");
      const bytes = new Uint8Array(size);
      let offset = 0;
      for (const part of chunks) { bytes.set(part, offset); offset += part.length; }
      return bytes.buffer;
    }
    async function load() {
      const version = ++generation;
      controller?.abort(); controller = null;
      ready = false; show();
      if (!settings?.asset) {
        viewer?.setVisible(false);
        status("Original simulated body. Illustrative muscles are not measured anatomy.");
        redraw(); return;
      }
      const choice = choices.find(item => item.key === settings.asset);
      if (!choice) { status("Original body shown: the selected avatar is missing."); redraw(); return; }
      const abort = new AbortController(); controller = abort;
      const timeout = setTimeout(() => abort.abort(), 15000);
      status("Loading avatar. The original body remains visible.");
      try {
        if (!viewer) {
          await loadModules();
          if (disposed || version !== generation) return;
          viewer = root.WeftAvatarViewer.create(canvas, {onStatus: viewerStatus});
          viewer.configure(scene);
        }
        const base = `${basePath}/api/avatar/${encodeURIComponent(choice.key)}/`;
        const renderBytes = await binary(base + "render.json", abort.signal, 8 * 1024 * 1024);
        const textures = [];
        let total = 0;
        // Sequential texture reads bound peak memory and avoid eight concurrent image transfers.
        for (let i = 0; i < choice.textures; i++) {
          const bytes = await binary(base + `texture-${i}`, abort.signal, 32 * 1024 * 1024 - total);
          total += bytes.byteLength; textures.push(bytes);
        }
        if (disposed || version !== generation) return;
        viewer.setVisible(visible());
        const result = await viewer.load({renderBytes, renderSha256: choice.render_sha256, textures});
        if (disposed || version !== generation) return;
        if (result.status !== "ready") throw new Error(result.reason || "Avatar loading did not complete");
        ready = true;
        status("Avatar view · same simulated body and equipment. Appearance is saved with your trial.");
        show(); redraw();
      } catch (error) {
        if (!disposed && version === generation) viewerStatus({status: "fallback", reason: error.message});
      } finally { clearTimeout(timeout); }
    }
    async function drain() {
      while (queued && !disposed) {
        const proposed = queued; queued = null;
        const previous = settings?.asset;
        try {
          const result = await send(proposed);
          if (disposed) return;
          settings = result.settings;
          select.value = settings.asset || ""; overlay.checked = settings.overlay;
          if (settings.asset !== previous) loadTask = load();
          else redraw();
        } catch (error) {
          status(`Appearance was not changed: ${error.message}`);
          select.value = settings?.asset || ""; overlay.checked = settings?.overlay ?? true;
          throw error;
        }
      }
    }
    function save(proposed) {
      queued = proposed;
      if (!saving) saving = drain().finally(() => { saving = null; });
      return saving;
    }
    const proposal = () => ({...settings, asset: select.value || null, overlay: overlay.checked,
      show_muscles: $("show-muscles").checked, camera: camera()});
    function choose() {
      pause(); generation++; controller?.abort(); ready = false; viewer?.setVisible(false); show();
      save(proposal()).catch(() => {});
    }
    select.onchange = choose;
    overlay.onchange = () => { save(proposal()).catch(() => {}); };
    $("avatar-original").onclick = () => { select.value = ""; choose(); };
    function cameraChanged() {
      if (!settings || !choices.length || disposed) return;
      clearTimeout(cameraTimer);
      cameraTimer = setTimeout(() => { cameraTimer = null; save(proposal()).catch(() => {}); }, 200);
    }
    for (const id of ["spatial-yaw", "spatial-zoom", "spatial-follow", "spatial-world-yaw", "spatial-world-zoom", "spatial-world-follow", "show-muscles"]) {
      $(id).addEventListener("input", cameraChanged);
    }
    return {
      configure(appearance, value) {
        if (!appearance) return;
        scene = value; choices = appearance.choices; settings = appearance.settings;
        $("avatar-controls").hidden = false; select.replaceChildren();
        for (const item of [{key: "", display_name: "Original body"}, ...choices]) {
          const option = document.createElement("option"); option.value = item.key; option.textContent = item.display_name; select.append(option);
        }
        select.value = settings.asset || ""; overlay.checked = settings.overlay;
        select.disabled = !choices.length; overlay.disabled = !choices.length;
        $("avatar-original").disabled = !choices.length;
        $("avatar-import-help").hidden = Boolean(choices.length);
        loadTask = load();
      },
      update(snapshot) {
        latest = snapshot;
        if (ready && settings?.asset) viewer.update({snapshot, camera: camera(), overlay: settings.overlay,
          showMuscles: $("show-muscles").checked});
        return ready && Boolean(settings?.asset);
      },
      setVisible(value) { viewer?.setVisible(value); if (value && latest && ready) this.update(latest); },
      async flush() {
        if (cameraTimer !== null) { clearTimeout(cameraTimer); cameraTimer = null; await save(proposal()); }
        if (saving) await saving;
        if (loadTask) await loadTask;
      },
      dispose() {
        disposed = true; generation++; controller?.abort(); clearTimeout(cameraTimer); viewer?.dispose();
        for (const id of ["spatial-yaw", "spatial-zoom", "spatial-follow", "spatial-world-yaw", "spatial-world-zoom", "spatial-world-follow", "show-muscles"]) $(id).removeEventListener("input", cameraChanged);
      },
    };
  }
  root.WeftLabAppearance = Object.freeze({create});
})(globalThis);
