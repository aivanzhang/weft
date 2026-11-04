/* Local demo only: one bounded request at a time, none while the page is hidden. */
(() => {
  const initial = document.currentScript.dataset.version;
  let timer, controller, disposed = false;
  function stop() { clearTimeout(timer); controller?.abort(); }
  async function poll() {
    if (disposed || document.hidden || controller) return;
    controller = new AbortController();
    const timeout = setTimeout(() => controller?.abort(), 3000);
    try {
      const response = await fetch('/__weft_reload', { cache: 'no-store', signal: controller.signal });
      if (response.ok) {
        const state = await response.json();
        if (state.ready && state.revision !== initial) {
          disposed = true;
          location.reload();
        }
      }
    } catch { /* Keep the current page while the local server restarts. */ }
    finally {
      clearTimeout(timeout); controller = null;
      if (!disposed && !document.hidden) timer = setTimeout(poll, 1000);
    }
  }
  document.addEventListener('visibilitychange', () => { stop(); if (!document.hidden) poll(); });
  window.addEventListener('pagehide', () => { disposed = true; stop(); });
  window.addEventListener('pageshow', () => { disposed = false; poll(); });
  poll();
})();
