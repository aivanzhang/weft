(() => {
  "use strict";
  const contents = document.querySelector("#page-contents");
  if (!contents) return;
  const trigger = contents.querySelector("summary");
  contents.addEventListener("click", (event) => {
    if (event.defaultPrevented || event.button > 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const link = event.target.closest("a[href^='#']");
    if (!link) return;
    const section = document.getElementById(link.hash.slice(1));
    if (!section) return;
    contents.open = false;
    // Move keyboard focus to the destination without replacing native anchor scrolling.
    section.setAttribute("tabindex", "-1");
    section.focus({ preventScroll: true });
  });
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || !contents.open) return;
    event.preventDefault();
    contents.open = false;
    trigger.focus();
  });
  document.addEventListener("pointerdown", (event) => {
    if (contents.open && !contents.contains(event.target)) contents.open = false;
  });
  window.addEventListener("pagehide", () => { contents.open = false; });
})();
