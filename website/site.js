(() => {
  "use strict";

  const entries = [...document.querySelectorAll(".research-entry")];
  const list = document.querySelector(".research-list");
  const triggers = entries.map((entry) =>
    entry.querySelector(".research-trigger"),
  );
  const artwork = entries.map((entry) => entry.querySelector(".research-art"));
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const closingTimers = new Map();
  let pinned = null;
  let hovered = null;
  let focused = null;
  let active = null;
  let dismissed = null;
  let hoverTimer = null;
  let lastPointer = null;

  function cancelHoverIntent() {
    if (hoverTimer !== null) clearTimeout(hoverTimer);
    hoverTimer = null;
  }

  function reveal(index) {
    cancelHoverIntent();
    if (active === index) return;
    active = index;
    entries.forEach((entry, i) => {
      const open = i === index;
      if (closingTimers.has(i)) {
        clearTimeout(closingTimers.get(i));
        closingTimers.delete(i);
      }
      entry.dataset.active = String(open);
      triggers[i].setAttribute("aria-expanded", String(open));
      artwork[i].setAttribute("aria-hidden", String(!open));
      if (open) {
        entry.dataset.motion = "true";
      } else if (entry.dataset.motion === "true") {
        // Preserve the illustration's current pose until its exit fade finishes.
        if (reducedMotion.matches) entry.dataset.motion = "false";
        else
          closingTimers.set(
            i,
            setTimeout(() => {
              entry.dataset.motion = "false";
              closingTimers.delete(i);
            }, 560),
          );
      }
    });
  }

  function scheduleReveal(index, delay) {
    cancelHoverIntent();
    if (index === active) return;
    hoverTimer = setTimeout(() => {
      hoverTimer = null;
      reveal(index);
    }, delay);
  }

  list.addEventListener("pointermove", (event) => {
    if (event.pointerType !== "mouse") return;
    // Layout changes must not select the row sliding under a stationary pointer.
    if (lastPointer?.x === event.clientX && lastPointer?.y === event.clientY)
      return;
    lastPointer = { x: event.clientX, y: event.clientY };
    const entry = event.target.closest(".research-entry");
    const index = entry ? entries.indexOf(entry) : null;
    if (index === hovered) return;
    hovered = index;
    if (dismissed !== index) dismissed = null;
    if (dismissed !== null) return;
    scheduleReveal(index ?? focused ?? pinned, index === null ? 150 : 70);
  });
  list.addEventListener("pointerleave", (event) => {
    if (event.pointerType !== "mouse") return;
    hovered = null;
    dismissed = null;
    lastPointer = null;
    scheduleReveal(focused ?? pinned, 150);
  });

  triggers.forEach((trigger, index) => {
    trigger.addEventListener("focus", () => {
      if (!trigger.matches(":focus-visible")) return;
      focused = index;
      dismissed = null;
      reveal(index);
    });
    trigger.addEventListener("blur", () => {
      if (focused === index) focused = null;
      reveal(hovered === dismissed ? pinned : (hovered ?? pinned));
    });
    trigger.addEventListener("click", () => {
      pinned = pinned === index ? null : index;
      dismissed = pinned === null ? index : null;
      reveal(pinned);
    });
    trigger.addEventListener("keydown", (event) => {
      if (event.key !== "Escape") return;
      pinned = null;
      focused = null;
      dismissed = index;
      reveal(null);
    });
  });
  function clearPendingMotion() {
    cancelHoverIntent();
    closingTimers.forEach((timer, index) => {
      clearTimeout(timer);
      entries[index].dataset.motion = "false";
    });
    closingTimers.clear();
  }
  window.addEventListener("pagehide", clearPendingMotion);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) clearPendingMotion();
  });

  const heroStageList = document.querySelector(".hero-stages");
  const heroStages = [...heroStageList.querySelectorAll(".hero-stage")];
  const stageButtons = heroStages.map((stage) =>
    stage.querySelector(".stage-trigger"),
  );
  const stageCards = heroStages.map((stage) =>
    stage.querySelector(".stage-card"),
  );
  const stageExitTimers = new Map();
  let stageActive = null;
  let stagePinned = null;
  let stageHovered = null;
  let stageFocused = null;
  let stageIntent = null;

  function cancelStageIntent() {
    if (stageIntent !== null) clearTimeout(stageIntent);
    stageIntent = null;
  }
  function showStage(index) {
    cancelStageIntent();
    if (stageActive === index) return;
    stageActive = index;
    // Measure once on opening, before writes, so cards stay inside the viewport.
    const placement =
      index === null
        ? null
        : stageButtons[index].getBoundingClientRect().top <
            stageCards[index].offsetHeight + 24
          ? "below"
          : "above";
    heroStages.forEach((stage, i) => {
      if (stageExitTimers.has(i)) {
        clearTimeout(stageExitTimers.get(i));
        stageExitTimers.delete(i);
      }
      const open = i === index;
      stage.dataset.active = String(open);
      stageButtons[i].setAttribute("aria-expanded", String(open));
      stageCards[i].setAttribute("aria-hidden", String(!open));
      stageCards[i].inert = !open;
      if (open) {
        stage.dataset.placement = placement;
        stage.dataset.motion = "true";
      } else if (stage.dataset.motion === "true") {
        if (reducedMotion.matches) stage.dataset.motion = "false";
        else
          stageExitTimers.set(
            i,
            setTimeout(() => {
              stage.dataset.motion = "false";
              stageExitTimers.delete(i);
            }, 300),
          );
      }
    });
  }
  function scheduleStage(index, delay) {
    cancelStageIntent();
    if (index === stageActive) return;
    stageIntent = setTimeout(() => {
      stageIntent = null;
      showStage(index);
    }, delay);
  }
  function dismissStages() {
    stagePinned = null;
    stageHovered = null;
    stageFocused = null;
    showStage(null);
  }
  heroStages.forEach((stage, index) => {
    const button = stageButtons[index];
    stage.addEventListener("pointerenter", (event) => {
      if (event.pointerType !== "mouse") return;
      stageHovered = index;
      scheduleStage(index, 80);
    });
    stage.addEventListener("pointerleave", (event) => {
      if (event.pointerType !== "mouse") return;
      stageHovered = null;
      scheduleStage(stageFocused ?? stagePinned, 180);
    });
    button.addEventListener("focus", () => {
      if (!button.matches(":focus-visible")) return;
      stageFocused = index;
      showStage(index);
    });
    button.addEventListener("blur", () => {
      if (stageFocused === index) stageFocused = null;
      scheduleStage(stageHovered ?? stagePinned, 100);
    });
    button.addEventListener("click", () => {
      stagePinned = stagePinned === index ? null : index;
      if (stagePinned === null) stageFocused = null;
      showStage(stagePinned);
    });
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && stageActive !== null) dismissStages();
  });
  document.addEventListener("pointerdown", (event) => {
    if (stageActive !== null && !event.target.closest(".hero-stage"))
      dismissStages();
  });
  function clearStageMotion() {
    dismissStages();
    stageExitTimers.forEach((timer, index) => {
      clearTimeout(timer);
      heroStages[index].dataset.motion = "false";
    });
    stageExitTimers.clear();
  }
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) clearStageMotion();
  });
  window.addEventListener("pagehide", clearStageMotion);

  const heroChart = document.querySelector(".hero-figure svg");
  const heroSlide = document.querySelector("#hero-ball-slide");
  function finishHeroMotion() {
    heroChart.classList.add("hero-motion-complete");
    heroChart.pauseAnimations();
  }
  heroSlide.addEventListener("endEvent", finishHeroMotion, { once: true });
  if (reducedMotion.matches) finishHeroMotion();
  reducedMotion.addEventListener("change", (event) => {
    if (event.matches) finishHeroMotion();
  });

  function pauseHiddenMotion() {
    document.documentElement.classList.toggle("page-hidden", document.hidden);
    // Keep SVG motion and the CSS line reveal on the same visibility clock.
    if (
      document.hidden ||
      heroChart.classList.contains("hero-motion-complete")
    ) {
      heroChart.pauseAnimations();
    } else {
      heroChart.unpauseAnimations();
    }
  }
  document.addEventListener("visibilitychange", pauseHiddenMotion);

  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(
      (observations) => {
        observations.forEach((observation) => {
          if (observation.target === heroStageList) {
            if (!observation.isIntersecting) dismissStages();
          } else {
            observation.target.classList.toggle(
              "in-view",
              observation.isIntersecting,
            );
          }
        });
      },
      { threshold: 0.05 },
    );
    [heroStageList, ...entries].forEach((element) => observer.observe(element));
  } else {
    entries.forEach((entry) => entry.classList.add("in-view"));
  }
  pauseHiddenMotion();
})();
