/* A run owns the screen; switching views releases hidden media and simulation work. */
((root) => {
  let stored = false;
  let token = new URLSearchParams(location.hash.slice(1)).get("token") || "";
  try {
    token ||= sessionStorage.getItem("weft-portal-token") || "";
    if (token) {
      sessionStorage.setItem("weft-portal-token", token);
      stored = true;
    }
  } catch { /* Access links still work when browser storage is disabled. */ }
  if (token) history.replaceState(null, "", location.pathname + location.search);
  root.WeftDemoSession = { token, stored };

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
      $("studio-steps").hidden = view !== "evidence";
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
        select(record.view || (world ? "world" : video ? "recording" : "evidence"));
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
