/* A run owns the screen; switching views releases hidden media and simulation work. */
((root) => {
  let token = "";
  const query = new URLSearchParams(location.search);
  const accessCode = (query.get("access_code") || "").trim().slice(0, 64);
  const session = { token: "", stored: false, accessCode,
    setToken(value) {
      token = value;
      session.token = value;
      session.stored = false;
      try {
        sessionStorage.setItem("weft-portal-token", value);
        session.stored = Boolean(value);
      } catch { /* Access links still work when browser storage is disabled. */ }
    },
  };
  let initial = new URLSearchParams(location.hash.slice(1)).get("token") || "";
  try { initial ||= sessionStorage.getItem("weft-portal-token") || ""; } catch { /* Storage is optional. */ }
  if (initial) session.setToken(initial);
  if (query.has("access_code")) query.delete("access_code");
  const cleanQuery = query.toString();
  if (initial || accessCode) history.replaceState(null, "", location.pathname + (cleanQuery ? "?" + cleanQuery : ""));
  root.WeftDemoSession = session;

  root.WeftElapsedClock = (paint, {
    now = () => performance.now(), schedule = setTimeout, cancel = clearTimeout,
  } = {}) => {
    let id = null, base = 0, anchor = 0, running = false, timer = null;
    const elapsed = () => base + (running ? Math.max(0, now() - anchor) / 1000 : 0);
    const pause = () => { if (timer !== null) cancel(timer); timer = null; };
    const tick = () => {
      timer = null;
      if (document.hidden) return;
      paint(elapsed(), running);
      if (running) timer = schedule(tick, 100);
    };
    const visibility = () => { pause(); if (!document.hidden) tick(); };
    document.addEventListener("visibilitychange", visibility);
    root.addEventListener?.("pagehide", pause);
    return {
      update(state) {
        pause();
        const reported = Math.max(0, Number(state.elapsed) || 0);
        // Polling corrects the anchor, but stale samples must not rewind a live clock.
        base = state.running && running && state.id === id ? Math.max(reported, elapsed()) : reported;
        id = state.id;
        running = state.running;
        anchor = now();
        tick();
      },
      dispose() {
        pause(); running = false;
        document.removeEventListener("visibilitychange", visibility);
        root.removeEventListener?.("pagehide", pause);
      },
    };
  };

  root.WeftRunResults = (record) => {
    const allowed = new Set(["task success", "failed gates", "control steps", "tracking error",
      "physics", "scope", "validation", "imported", "model", "API cost"]);
    const metrics = new Map();
    for (const step of record.steps || []) {
      for (const [key, value] of Object.entries(step.metrics || {})) {
        if (allowed.has(key)) metrics.set(key, value);
      }
    }
    if (!metrics.size && record.metrics) metrics.set("result", record.metrics);
    return [...metrics];
  };

  root.WeftRunFocus = () => {
    const $ = (id) => document.getElementById(id);
    const tabs = [...document.querySelectorAll("[data-run-view]")];
    let world = null, video = null, opener = null, background = [], activeView = null;
    const release = () => {
      $("world-view").removeAttribute("src");
      $("studio-video").pause();
      $("studio-video").removeAttribute("src");
      $("studio-video").load();
    };
    const select = (view) => {
      if (view === activeView) return;
      activeView = view;
      release();
      $("world-frame").hidden = view !== "world" || !world;
      $("play").hidden = view !== "world" || !world;
      $("result-media").hidden = view !== "recording" || !video;
      $("studio-results").hidden = view !== "results";
      for (const tab of tabs) tab.setAttribute("aria-pressed", String(tab.dataset.runView === view));
      if (view === "world" && world) {
        $("world-view").src = `/labs/${world}/#token=${encodeURIComponent(token)}&embed=1`;
      }
      if (view === "recording" && video) $("studio-video").src = video;
    };
    for (const tab of tabs) tab.addEventListener("click", () => select(tab.dataset.runView));
    $("studio-modal").addEventListener("keydown", (event) => {
      if (event.key !== "Tab") return;
      const controls = [...$("studio-modal").querySelectorAll('button:not(:disabled), a[href], video[controls], iframe, [tabindex="0"]')]
        .filter((element) => element.getClientRects().length);
      const first = controls[0], last = controls.at(-1);
      if (event.shiftKey && (document.activeElement === first || document.activeElement === $("studio-title"))) {
        event.preventDefault(); last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault(); first?.focus();
      }
    });
    return {
      get active() { return activeView; },
      open(record) {
        opener = document.activeElement;
        background = [...document.querySelectorAll("body > .site-head, main > :not(.modal)")]
          .filter((element) => !element.inert);
        for (const element of background) element.inert = true;
        activeView = null;
        world = record.world;
        video = record.video;
        $("demo-video").pause();
        for (const tab of tabs) tab.disabled = tab.dataset.runView === "world" ? !world
          : tab.dataset.runView === "recording" ? !video : false;
        select(record.view || (world ? "world" : video ? "recording" : "results"));
        $("studio-title").focus();
      },
      close() {
        release();
        activeView = null;
        world = video = null;
        for (const element of background) element.inert = false;
        background = [];
        opener?.focus();
      },
    };
  };
})(window);
