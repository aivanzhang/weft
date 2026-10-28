(() => {
  "use strict";
  const root = document.querySelector("#footer-secret");
  if (!root) return;
  const $ = (s) => root.querySelector(s);
  const $$ = (s) => [...root.querySelectorAll(s)];
  const trigger = $(".thread");
  const film = $("#footer-film");
  const theatre = $(".theatre");
  const scene = $(".stage");
  const rotations = new Set();
  let presenting = false;
  let presentationFrame = null;
  let pendingScroll = false;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const animations = new Set();
  const duration = 14800;
  const lengths = [];
  let visible = false;
  let hintInView = false;
  let hintReady = false;
  let hintTimer = null;
  let opened = false;
  let hold = null;
  let bounds = null;
  let tension = null;

  function cancelHintDelay() {
    clearTimeout(hintTimer);
    hintTimer = null;
  }

  function syncHintDelay() {
    if (hintReady) return;
    if (document.hidden || !hintInView) {
      cancelHintDelay();
      return;
    }
    if (hintTimer !== null) return;
    hintTimer = setTimeout(() => {
      hintTimer = null;
      if (document.hidden || !hintInView) return;
      hintReady = true;
      theatre.hidden = false;
      root.classList.add("thread-ready");
      trigger.inert = false;
      trigger.removeAttribute("aria-hidden");
    }, 10000);
  }

  function track(
    selector,
    property,
    values,
    easing = "cubic-bezier(.4,0,.2,1)",
  ) {
    $$(selector).forEach((element) => {
      // SVG rotations stay in the rig's local coordinate system.
      if (
        property === "transform" &&
        values.every(([, value]) => /^rotate\(-?[\d.]+deg\)$/.test(value))
      ) {
        if (reduced.matches) {
          element.setAttribute(
            "transform",
            `rotate(${parseFloat(values.at(-1)[1].slice(7))})`,
          );
          return;
        }
        const motion = document.createElementNS(
          "http://www.w3.org/2000/svg",
          "animateTransform",
        );
        motion.setAttribute("attributeName", "transform");
        motion.setAttribute("type", "rotate");
        motion.setAttribute(
          "values",
          values.map(([, value]) => parseFloat(value.slice(7))).join(";"),
        );
        motion.setAttribute(
          "keyTimes",
          values.map(([time]) => time / 14.8).join(";"),
        );
        motion.setAttribute("calcMode", "spline");
        motion.setAttribute(
          "keySplines",
          values
            .slice(1)
            .map(() => ".4 0 .2 1")
            .join(";"),
        );
        motion.setAttribute("dur", "14.8s");
        motion.setAttribute("begin", "indefinite");
        motion.setAttribute("fill", "freeze");
        element.append(motion);
        rotations.add(motion);
        motion.beginElement();
        return;
      }
      const keyframes = values.map(([time, value]) => ({
        offset: time / 14.8,
        [property]: value,
        easing,
      }));
      const animation = element.animate(keyframes, {
        duration: reduced.matches ? 1 : duration,
        fill: "both",
        easing: "linear",
      });
      animations.add(animation);
      if (document.hidden || !visible || presenting) animation.pause();
    });
  }

  function show(selector, start, end, opacity = 1) {
    track(selector, "opacity", [
      [0, 0],
      [start, 0],
      [end, opacity],
      [14.8, opacity],
    ]);
  }

  function play() {
    if (!lengths.length) {
      for (let index = 0; index < 3; index++) {
        const stitch = $(`#stitch-${index}`);
        const length = stitch.getTotalLength();
        stitch.style.strokeDasharray = `${length * 0.045} ${length}`;
        lengths.push(length);
      }
    }
    animations.forEach((a) => a.cancel());
    animations.clear();
    rotations.forEach((motion) => motion.remove());
    rotations.clear();
    scene.setCurrentTime(0);
    if (presenting || document.hidden || !visible) scene.pauseAnimations();
    else scene.unpauseAnimations();
    track("#actor-position", "transform", [
      [0, "translate(-145px,260px)"],
      [0.4, "translate(-100px,260px)"],
      [2.1, "translate(102px,260px)"],
      [3.3, "translate(102px,260px)"],
      [4.1, "translate(80px,260px)"],
      [4.6, "translate(102px,260px)"],
      [14.8, "translate(102px,260px)"],
    ]);
    track("#actor-position", "opacity", [
      [0, 0],
      [0.3, 1],
      [14.8, 1],
    ]);
    track("#actor-bounce", "transform", [
      [0, "translateY(0) scale(1)"],
      [0.4, "translateY(-5px) scale(1)"],
      [0.7, "translateY(0) scale(1.02,.98)"],
      [1, "translateY(-5px) scale(1)"],
      [1.3, "translateY(0) scale(1.02,.98)"],
      [1.6, "translateY(-5px) scale(1)"],
      [1.9, "translateY(0) scale(1.02,.98)"],
      [2.2, "translateY(0) scale(1)"],
      [3.8, "translateY(2px) scale(1.03,.97)"],
      [4.25, "translateY(-6px) scale(.98,1.02)"],
      [4.7, "translateY(0) scale(1)"],
      [11.2, "translateY(0) scale(1)"],
      [11.45, "translateY(3px) scale(1.05,.95)"],
      [11.8, "translateY(-23px) scale(.97,1.03)"],
      [12.15, "translateY(0) scale(1.05,.95)"],
      [12.4, "translateY(0) scale(1)"],
      [14.8, "translateY(0) scale(1)"],
    ]);
    track("#actor-mascot", "transform", [
      [0, "rotate(0deg)"],
      [2.3, "rotate(0deg)"],
      [2.8, "rotate(8deg)"],
      [3.4, "rotate(5deg)"],
      [4.1, "rotate(-13deg)"],
      [4.5, "rotate(4deg)"],
      [4.9, "rotate(0deg)"],
      [7.2, "rotate(3deg)"],
      [10.7, "rotate(3deg)"],
      [11.2, "rotate(0deg)"],
      [14.8, "rotate(0deg)"],
    ]);
    for (const [side, sign] of [
      ["left", 1],
      ["right", -1],
    ]) {
      track(`#actor-foot-${side}`, "transform", [
        [0, "rotate(0deg)"],
        [0.35, `rotate(${sign * 19}deg)`],
        [0.65, `rotate(${-sign * 19}deg)`],
        [0.95, `rotate(${sign * 19}deg)`],
        [1.25, `rotate(${-sign * 19}deg)`],
        [1.55, `rotate(${sign * 19}deg)`],
        [1.85, `rotate(${-sign * 12}deg)`],
        [2.2, "rotate(0deg)"],
        [3.8, `rotate(${-sign * 10}deg)`],
        [4.7, "rotate(0deg)"],
        [14.8, "rotate(0deg)"],
      ]);
    }
    track("#actor-arm-right", "transform", [
      [0, "rotate(0deg)"],
      [0.5, "rotate(-15deg)"],
      [1, "rotate(12deg)"],
      [1.5, "rotate(-15deg)"],
      [2, "rotate(0deg)"],
      [2.8, "rotate(-58deg)"],
      [4.2, "rotate(-65deg)"],
      [4.8, "rotate(-15deg)"],
      [6, "rotate(-55deg)"],
      [7.2, "rotate(-28deg)"],
      [10.8, "rotate(-28deg)"],
      [11.6, "rotate(-100deg)"],
      [12.1, "rotate(-80deg)"],
      [12.55, "rotate(-45deg)"],
      [12.8, "rotate(-72deg)"],
      [13.05, "rotate(-45deg)"],
      [13.3, "rotate(-72deg)"],
      [13.8, "rotate(0deg)"],
      [14.8, "rotate(0deg)"],
    ]);
    track("#actor-arm-left", "transform", [
      [0, "rotate(0deg)"],
      [0.5, "rotate(15deg)"],
      [1, "rotate(-12deg)"],
      [1.5, "rotate(15deg)"],
      [2, "rotate(0deg)"],
      [3.8, "rotate(28deg)"],
      [4.6, "rotate(0deg)"],
      [11.2, "rotate(0deg)"],
      [11.7, "rotate(50deg)"],
      [12.2, "rotate(30deg)"],
      [13, "rotate(0deg)"],
      [14.8, "rotate(0deg)"],
    ]);
    track("#actor-eyes-open", "transform", [
      [0, "translate(0,0) scaleY(1)"],
      [1.1, "translate(0,0) scaleY(1)"],
      [1.18, "translate(0,0) scaleY(.06)"],
      [1.28, "translate(0,0) scaleY(1)"],
      [2.6, "translate(7px,-4px) scaleY(1)"],
      [4.1, "translate(7px,-4px) scaleY(.7)"],
      [4.5, "translate(10px,-8px) scaleY(1.1)"],
      [5, "translate(10px,-8px) scaleY(1)"],
      [8, "translate(10px,-3px) scaleY(1)"],
      [8.08, "translate(10px,-3px) scaleY(.06)"],
      [8.2, "translate(10px,-3px) scaleY(1)"],
      [11, "translate(10px,0) scaleY(1)"],
      [13.5, "translate(0,0) scaleY(1)"],
      [14.8, "translate(0,0) scaleY(1)"],
    ]);
    track("#actor-eyes-open", "opacity", [
      [0, 1],
      [11.5, 1],
      [11.65, 0],
      [12.5, 0],
      [12.65, 1],
      [14.8, 1],
    ]);
    track("#actor-eyes-closed", "opacity", [
      [0, 0],
      [11.5, 0],
      [11.65, 1],
      [12.5, 1],
      [12.65, 0],
      [14.8, 0],
    ]);
    track("#actor-shadow", "opacity", [
      [0, 0],
      [1.8, 0.065],
      [11.45, 0.065],
      [11.8, 0.025],
      [12.15, 0.065],
      [14.8, 0.065],
    ]);
    show("#ribbon-tail", 1.6, 2.5);
    track("#ribbon-tail", "transform", [
      [0, "scaleX(1)"],
      [3.3, "scaleX(1)"],
      [4.1, "scaleX(1.2)"],
      [4.6, "scaleX(1)"],
      [14.8, "scaleX(1)"],
    ]);
    for (let i = 0; i < 3; i++) {
      $$(`#reveal-${i} .thread-reveal`).forEach((path) => {
        const start = 4.05 + i * 0.2;
        const length = lengths[i];
        path.removeAttribute("pathLength");
        const animation = path.animate(
          [
            {
              strokeDasharray: `${length} ${length}`,
              strokeDashoffset: length,
              opacity: 0,
              offset: 0,
            },
            {
              strokeDasharray: `${length} ${length}`,
              strokeDashoffset: length,
              opacity: 0,
              offset: start / 14.8,
            },
            {
              strokeDasharray: `${length} ${length}`,
              strokeDashoffset: length,
              opacity: 1,
              offset: (start + 0.02) / 14.8,
            },
            {
              strokeDasharray: `${length} ${length}`,
              strokeDashoffset: 0,
              opacity: 1,
              offset: (start + 2.35) / 14.8,
            },
            {
              strokeDasharray: `${length} ${length}`,
              strokeDashoffset: 0,
              opacity: 1,
              offset: 1,
            },
          ],
          {
            duration: reduced.matches ? 1 : duration,
            fill: "both",
            easing: "linear",
          },
        );
        animations.add(animation);
        if (document.hidden || !visible || presenting) animation.pause();
      });
      track(`.branch-${i}`, "opacity", [
        [0, 1],
        [10.8, 1],
        [12.1, i === 1 ? 1 : 0.45],
        [14.8, i === 1 ? 1 : 0.45],
      ]);
      const start = 6.8 + i * 0.18;
      const end = start + 3.8;
      track(`#stitch-${i}`, "strokeDashoffset", [
        [0, 0],
        [start, 0],
        [end, -lengths[i] * 0.955],
        [14.8, -lengths[i] * 0.955],
      ]);
      track(`#stitch-${i}`, "opacity", [
        [0, 0],
        [6.1 + i * 0.12, 0],
        [6.7 + i * 0.12, i === 1 ? 1 : 0.65],
        [10.8, i === 1 ? 1 : 0.65],
        [11.6, 0],
        [14.8, 0],
      ]);
    }
    track(".arrival", "opacity", [
      [0, 0],
      [10.7, 0],
      [11, 1],
      [11.5, 1],
      [12, 0],
      [14.8, 0],
    ]);
    show(".closing p, .closing > span", 11.5, 12.6);
    $("#footer-story-status").textContent =
      "Weft pulls three ribbon worlds from a thread. Small stitches trace different actions along the ribbons. An illustrative animated story.";
  }

  function cancelHold() {
    clearTimeout(hold);
    hold = null;
    bounds = null;
    if (tension) tension.cancel();
    tension = null;
    if (!opened) $(".hint").textContent = "Pull gently. Stay curious.";
  }
  function reveal() {
    cancelHold();
    opened = true;
    presenting = true;
    document.documentElement.classList.add("weft-presenting");
    trigger.setAttribute("aria-expanded", "true");
    film.classList.add("open");
    film.setAttribute("aria-hidden", "false");
    film.inert = false;
    play();
    // Reserve the complete scene before measuring a single scroll destination.
    presentationFrame = requestAnimationFrame(finishPresentation);
  }
  function completePresentation() {
    presentationFrame = null;
    pendingScroll = false;
    presenting = false;
    document.documentElement.classList.remove("weft-presenting");
    film.focus({ preventScroll: true });
    syncVisibility();
  }
  function finishPresentation() {
    presentationFrame = null;
    if (document.hidden) {
      pendingScroll = true;
      return;
    }
    pendingScroll = false;
    document.documentElement.classList.add("weft-presenting");
    const rect = film.getBoundingClientRect();
    const room = Math.max(16, (window.innerHeight - rect.height) / 2);
    const maxScroll = Math.max(
      0,
      document.documentElement.scrollHeight - window.innerHeight,
    );
    const target = Math.max(
      0,
      Math.min(maxScroll, window.scrollY + rect.top - room),
    );
    window.scrollTo({
      top: target,
      behavior: reduced.matches ? "instant" : "smooth",
    });
    if (reduced.matches) {
      completePresentation();
      return;
    }
    const started = performance.now();
    function settle(now) {
      if (document.hidden) {
        presentationFrame = null;
        pendingScroll = true;
        return;
      }
      if (Math.abs(window.scrollY - target) < 2 || now - started > 1200) {
        window.scrollTo({ top: target, behavior: "instant" });
        completePresentation();
        return;
      }
      presentationFrame = requestAnimationFrame(settle);
    }
    presentationFrame = requestAnimationFrame(settle);
  }
  function beginHold() {
    if (!hintReady || hold !== null || opened || document.hidden || !visible)
      return;
    bounds = trigger.getBoundingClientRect();
    $(".hint").textContent = "Something is on the other end…";
    if (!reduced.matches)
      tension = $(".thread-fill").animate(
        [{ strokeDashoffset: 1 }, { strokeDashoffset: 0 }],
        { duration: 5000, fill: "forwards" },
      );
    hold = setTimeout(reveal, 5000);
  }
  trigger.addEventListener("pointerdown", (event) => {
    if (!event.isPrimary || event.button !== 0) return;
    event.preventDefault();
    trigger.focus({ preventScroll: true });
    trigger.setPointerCapture(event.pointerId);
    beginHold();
  });
  trigger.addEventListener("pointermove", (event) => {
    if (
      bounds &&
      (event.clientX < bounds.left ||
        event.clientX > bounds.right ||
        event.clientY < bounds.top ||
        event.clientY > bounds.bottom)
    )
      cancelHold();
  });
  ["pointerup", "pointercancel", "lostpointercapture", "blur"].forEach(
    (event) => trigger.addEventListener(event, cancelHold),
  );
  trigger.addEventListener("contextmenu", (event) => event.preventDefault());
  trigger.addEventListener("keydown", (event) => {
    if (event.key === " " || event.key === "Enter") {
      event.preventDefault();
      if (!event.repeat) beginHold();
    }
    if (event.key === "Escape") cancelHold();
  });
  trigger.addEventListener("keyup", (event) => {
    if (event.key === " " || event.key === "Enter") {
      event.preventDefault();
      cancelHold();
    }
  });
  function syncVisibility() {
    syncHintDelay();
    if (pendingScroll && !document.hidden) {
      finishPresentation();
      return;
    }
    if (document.hidden || !visible || presenting) {
      scene.pauseAnimations();
      cancelHold();
      animations.forEach((a) => {
        if (a.playState === "running") a.pause();
      });
    } else {
      scene.unpauseAnimations();
      animations.forEach((a) => {
        if (a.playState === "paused") a.play();
      });
    }
  }
  const observer = new IntersectionObserver(
    ([entry]) => {
      visible = entry.isIntersecting;
      hintInView = entry.isIntersecting && entry.intersectionRatio >= 0.5;
      syncVisibility();
    },
    { threshold: [0, 0.5] },
  );
  observer.observe(root);
  document.addEventListener("visibilitychange", syncVisibility);
  window.addEventListener("pagehide", () => {
    cancelHintDelay();
    cancelHold();
    if (presentationFrame !== null) {
      cancelAnimationFrame(presentationFrame);
      presentationFrame = null;
    }
    if (presenting) pendingScroll = true;
    document.documentElement.classList.remove("weft-presenting");
    scene.pauseAnimations();
    animations.forEach((a) => {
      if (a.playState === "running") a.pause();
    });
  });
  window.addEventListener("pageshow", syncVisibility);
  reduced.addEventListener("change", () => {
    if (opened) play();
  });
})();
