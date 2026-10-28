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

  const experiment = document.querySelector("#experiment");
  const simulation = document.querySelector("#simulation");
  const ball = document.querySelector("#simulation-ball");
  const leftButton = document.querySelector("#push-left");
  const rightButton = document.querySelector("#push-right");
  const resetButton = document.querySelector("#reset");
  const status = document.querySelector("#simulation-status");
  const engineReady = typeof globalThis.WeftPhysics?.advance === "function";

  // Illustration settings, not parameters fitted to the source clip.
  const world = { drag: 0.3, max_acceleration: 2 };
  let position = 0.5;
  let velocity = 0;
  let action = 0;
  let pushRemaining = 0;
  let elapsed = 0;
  let running = false;
  let visible = true;
  let frame = null;
  let previousTime = null;

  function announce(message) {
    if (status.textContent !== message) status.textContent = message;
  }

  function draw() {
    ball.style.transform = "translateX(" + (position * 100).toFixed(3) + "%)";
  }

  function stop(message) {
    running = false;
    velocity = 0;
    action = 0;
    pushRemaining = 0;
    previousTime = null;
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
    simulation.dataset.running = "false";
    if (message) announce(message);
  }

  function schedule() {
    if (
      frame === null &&
      running &&
      experiment.open &&
      visible &&
      !document.hidden
    ) {
      frame = requestAnimationFrame(tick);
    }
  }

  function tick(now) {
    frame = null;
    if (!running || !experiment.open || !visible || document.hidden) {
      stop("Stopped. Give the ball another push to continue.");
      return;
    }
    if (previousTime === null) previousTime = now;
    const dt = Math.min((now - previousTime) / 1000, 0.05);
    previousTime = now;
    if (dt > 0) {
      let remaining = dt;
      // Finish the short push exactly, then let the shared engine handle coasting.
      if (pushRemaining > 0) {
        const poweredTime = Math.min(remaining, pushRemaining);
        [position, velocity] = WeftPhysics.advance(
          world,
          position,
          velocity,
          action,
          poweredTime,
        );
        pushRemaining -= poweredTime;
        remaining -= poweredTime;
      }
      if (remaining > 0) {
        action = 0;
        [position, velocity] = WeftPhysics.advance(
          world,
          position,
          velocity,
          0,
          remaining,
        );
      }
      elapsed += dt;
      if (position <= 0.02 || position >= 0.98) {
        position = Math.max(0.02, Math.min(0.98, position));
        stop("At the edge. Push the other way or reset.");
      } else if (elapsed >= 8) {
        stop("At rest. Give the ball another push.");
      } else if (Math.abs(velocity) < 0.0001 && pushRemaining <= 0) {
        stop("At rest. Give the ball another push.");
      }
      draw();
    }
    schedule();
  }

  function push(direction) {
    if (!engineReady || !experiment.open || document.hidden || !visible) return;
    action = direction;
    pushRemaining = 0.18;
    elapsed = 0;
    velocity = Math.max(-1, Math.min(1, velocity));
    if (!running) previousTime = null;
    running = true;
    simulation.dataset.running = "true";
    announce(direction < 0 ? "Pushed left." : "Pushed right.");
    schedule();
  }

  experiment.addEventListener("toggle", () => {
    if (!experiment.open) stop();
  });
  document.querySelectorAll('a[href="#experiment"]').forEach((link) => {
    link.addEventListener("click", () => {
      experiment.open = true;
    });
  });
  function openLinkedExperiment() {
    if (location.hash === "#experiment") experiment.open = true;
  }
  window.addEventListener("hashchange", openLinkedExperiment);
  openLinkedExperiment();

  leftButton.addEventListener("click", () => push(-1));
  rightButton.addEventListener("click", () => push(1));
  resetButton.addEventListener("click", () => {
    stop();
    position = 0.5;
    elapsed = 0;
    draw();
    announce("Reset. Choose a direction to begin.");
  });

  if (engineReady) {
    [leftButton, rightButton, resetButton].forEach((control) => {
      control.disabled = false;
    });
  } else {
    status.className = "small muted";
    announce(
      "The simulation could not load. Reopen this page from the Weft repository.",
    );
  }

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
    if (document.hidden && running) stop("Stopped while this page was hidden.");
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
  window.addEventListener("pagehide", () => stop());

  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(
      (observations) => {
        observations.forEach((observation) => {
          if (observation.target === simulation) {
            visible = observation.isIntersecting;
            if (!visible && running) stop("Stopped outside the viewport.");
          } else if (observation.target === heroStageList) {
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
    [simulation, heroStageList, ...entries].forEach((element) =>
      observer.observe(element),
    );
  } else {
    entries.forEach((entry) => entry.classList.add("in-view"));
  }
  draw();
  pauseHiddenMotion();
})();
