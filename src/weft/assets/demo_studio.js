/* Saved evidence is read-only. Camera and playback never submit simulation actions. */
((root) => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const node = (tag, className, text) => {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text != null) element.textContent = text;
    return element;
  };
  const icon = (label, path) => {
    const button = node('button', 'studio-icon');
    button.type = 'button'; button.title = label; button.setAttribute('aria-label', label);
    button.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true">${path}</svg>`;
    return button;
  };
  const exhibit = (name) => `/api/demo/exhibit/${encodeURIComponent(name)}`;
  const stageCopy = {
    intake: ['Read the video.', 'Track the movement and keep the frames that explain it.'],
    world: ['Build the environment.', 'Turn the tracked movement into a body with joints and a ground plane.'],
    task: ['Set the task.', 'Define what the controller should do and how its result will be scored.'],
    rollout: ['Run the movement.', 'A reference controller follows the recorded motion. Watch what the simulated body does.'],
    avatar: ['Try another character.', 'Change the appearance while keeping the same movement, controls and score.'],
    muse: ['Review the result.', 'Meta reviews the result and suggests improvements.'],
    replay: ['Watch the replay.', 'Replay the saved movement.'],
    create: ['Try another character.', 'Change the appearance while keeping the same movement, controls and score.'],
  };
  root.WeftPipelineCopy = stageCopy;
  root.WeftPresentationSteps = (steps) => {
    const appearance = steps.find(step => step.id === 'avatar');
    return steps.map(step => step.id !== 'create' ? step : {
      ...step, experimental: false, evidenceStep: 'avatar',
      title: stageCopy.avatar[0], description: stageCopy.avatar[1],
      detail: appearance?.detail || '', metrics: appearance?.metrics || {},
      visuals: appearance?.visuals || [], assets: appearance?.assets || [],
      artifact: appearance?.artifact,
    });
  };
  let dispose = () => {};
  let lastOptions = null;

  async function readJSON(url, signal) {
    const response = await fetch(url, { signal: AbortSignal.any([signal, AbortSignal.timeout(15000)]) });
    if (!response.ok) throw new Error('This saved asset is not available.');
    return response.json();
  }

  function poseExperiment(segments, reference, ids, options) {
    if (!ids?.includes('torso')) return segments;
    const byID = new Map(ids.map((id, index) => [id, index]));
    const get = (list, id) => list[byID.get(id)];
    const add = (a, b) => a.map((v, i) => v + b[i]);
    const sub = (a, b) => a.map((v, i) => v - b[i]);
    const mul = (v, n) => v.map(x => x * n);
    const norm = v => Math.hypot(...v);
    const unit = v => mul(v, 1 / (norm(v) || 1));
    const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
    const rotate = (v, axis, degrees) => {
      const theta = degrees * Math.PI / 180, c = Math.cos(theta), s = Math.sin(theta), dot = v.reduce((n, x, i) => n + x * axis[i], 0);
      return add(add(mul(v, c), mul(cross(axis, v), s)), mul(axis, dot * (1-c)));
    };
    const out = segments.map(line => line.map(point => [...point]));
    const motion = options.motion ?? 1;
    const vector = id => {
      const current = get(segments, id), initial = get(reference, id);
      const v = sub(current[1], current[0]), base = sub(initial[1], initial[0]);
      const blended = add(mul(unit(base), 1-motion), mul(unit(v), motion));
      return mul(norm(blended) > 1e-6 ? unit(blended) : unit(v), norm(v));
    };
    const set = (id, a, b) => { if (byID.has(id)) out[byID.get(id)] = [a, b]; };
    const base = get(reference, 'torso')[0];
    const root = add(base, mul(sub(get(segments, 'torso')[0], base), motion));
    const neck = add(root, vector('torso')); set('torso', root, neck);
    if (byID.has('head')) set('head', neck, add(neck, vector('head')));
    for (const [id, center] of [['pelvis', root], ['shoulders', neck]]) {
      if (byID.has(id)) { const half = mul(vector(id), .5); set(id, sub(center, half), add(center, half)); }
    }
    for (const side of ['left', 'right']) for (const limb of ['arm', 'leg']) {
      const upper = `${side}_${limb === 'arm' ? 'upper_arm' : 'thigh'}`, lower = `${side}_${limb === 'arm' ? 'forearm' : 'shin'}`;
      const attach = limb === 'arm' ? 'shoulders' : 'pelvis';
      if (![upper, lower, attach].every(id => byID.has(id))) continue;
      const originalAnchor = get(segments, upper)[0], anchorLine = get(segments, attach);
      const endpoint = norm(sub(originalAnchor, anchorLine[0])) < norm(sub(originalAnchor, anchorLine[1])) ? 0 : 1;
      const a = get(out, attach)[endpoint], factor = limb === 'arm' ? options.arms ?? 1 : options.legs ?? 1;
      const u = mul(vector(upper), factor), l = mul(vector(lower), factor);
      const normal = cross(u, l), axis = norm(normal) > 1e-5 ? unit(normal) : [0, 1, 0];
      const shoulder = options.joints?.[`${side}_${limb === 'arm' ? 'shoulder' : 'hip'}`] || 0;
      const elbow = options.joints?.[`${side}_${limb === 'arm' ? 'elbow' : 'knee'}`] || 0;
      const b = add(a, rotate(u, axis, shoulder));
      set(upper, a, b); set(lower, b, add(b, rotate(l, axis, shoulder + elbow)));
    }
    return out;
  }

  function modelViewer(host, model) {
    const skeleton = model.kind === 'skeleton';
    const canvas = node('canvas', 'studio-canvas');
    canvas.width = 960; canvas.height = 760;
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', skeleton ? 'Recorded simulated skeleton; use playback and camera controls below' : 'Reconstructed rolling motion; use playback and push controls below');
    const context = canvas.getContext('2d');
    if (!context) throw new Error('Canvas is unavailable. Download the model to inspect it.');
    const toolbar = node('div', 'studio-transport');
    const play = icon('Play model', '<path d="m9 5 11 7-11 7Z"/>');
    play.setAttribute('aria-pressed', 'false');
    const seek = node('input'); seek.type = 'range'; seek.min = '0'; seek.step = '.001'; seek.value = '0';
    seek.setAttribute('aria-label', 'Model playback position');
    const time = node('output', 'studio-time');
    const reset = icon('Reset model view', '<path d="M4 10a8 8 0 1 1 1 8M4 4v6h6"/>');
    const knobs = node('div', 'studio-knobs');
    knobs.setAttribute('role', 'group'); knobs.setAttribute('aria-label', 'Model controls');
    function slider(label, min, max, value, step) {
      const row = node('label', '', label); const input = node('input');
      Object.assign(input, { type: 'range', min, max, value, step }); row.append(input); knobs.append(row); return input;
    }
    const initialYaw = String(model.camera_yaw ?? -12);
    const rotation = skeleton ? slider('Camera angle', '-180', '180', initialYaw, '1') : null;
    const zoom = slider('Zoom', '.7', '1.5', '1', '.05');
    const speed = slider('Playback speed', '.25', '1.5', '1', '.25');
    toolbar.append(play, seek, time, reset);
    const modelControls = node('div', 'studio-body-controls');
    const values = { motion: 1, arms: 1, legs: 1, joints: {} };
    const modelNote = node('p', 'studio-edit-note', 'Original pose');
    const bodyInputs = [];
    if (skeleton && model.segment_ids?.includes('torso')) {
      modelControls.setAttribute('role', 'group'); modelControls.setAttribute('aria-label', 'Model pose controls');
      for (const [key, title, min, max] of [['motion', 'Motion range', '0', '1.5'], ['arms', 'Arm length', '.75', '1.25'], ['legs', 'Leg length', '.75', '1.25']]) {
        const row = node('label', '', title), input = node('input');
        Object.assign(input, { type: 'range', min, max, step: '.05', value: '1' });
        input.setAttribute('aria-label', title); input.setAttribute('aria-valuetext', '100%');
        input.addEventListener('input', () => { values[key] = Number(input.value); input.setAttribute('aria-valuetext', `${Math.round(values[key]*100)}%`); edited(); });
        row.append(input); modelControls.append(row); bodyInputs.push(input);
      }
      const jointRow = node('label', 'studio-joint-choice', 'Joint');
      const joint = node('select'); joint.setAttribute('aria-label', 'Joint to adjust');
      for (const side of ['left', 'right']) for (const part of ['elbow', 'shoulder', 'knee', 'hip']) {
        const option = node('option', '', `${side[0].toUpperCase()+side.slice(1)} ${part}`); option.value = `${side}_${part}`; joint.append(option);
      }
      joint.value = 'left_elbow'; jointRow.append(joint);
      const bendRow = node('label', 'studio-joint-bend', 'Bend offset'), bend = node('input');
      Object.assign(bend, { type: 'range', min: '-45', max: '45', step: '1', value: '0' }); bend.setAttribute('aria-label', 'Joint bend offset'); bend.setAttribute('aria-valuetext', '0 degrees');
      joint.addEventListener('change', () => { bend.value = String(values.joints[joint.value] || 0); bend.setAttribute('aria-valuetext', `${bend.value} degrees`); });
      bend.addEventListener('input', () => { values.joints[joint.value] = Number(bend.value); bend.setAttribute('aria-valuetext', `${bend.value} degrees`); edited(); });
      bendRow.append(bend); modelControls.append(jointRow, bendRow, modelNote); bodyInputs.push(bend);
    }
    host.replaceChildren(canvas, toolbar, ...(modelControls.children.length ? [modelControls] : []), knobs);
    function edited() {
      const changed = values.motion !== 1 || values.arms !== 1 || values.legs !== 1 || Object.values(values.joints).some(Boolean);
      modelNote.textContent = changed ? 'Modified pose' : 'Original pose';
      draw();
    }
    let frame = 0, elapsed = 0, raf = null, last = null, playing = false, visible = true;
    const frames = model.frames || [];
    const duration = skeleton ? (model.duration ?? frames.at(-1).time - frames[0].time) : (model.segment.end - model.segment.start) * 4;
    seek.max = String(duration);
    let center = [0, 0, 0], extent = 1, x = model.segment?.x0 || 0, velocity = model.world?.v0 || 0, interactive = false, impulse = 0;
    if (skeleton) {
      const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (const f of frames) for (const line of f.segments) for (const p of line) for (let i = 0; i < 3; i++) {
        lo[i] = Math.min(lo[i], p[i]); hi[i] = Math.max(hi[i], p[i]);
      }
      center = lo.map((v, i) => (v + hi[i]) / 2); extent = Math.max(hi[2] - lo[2], hi[0] - lo[0], hi[1] - lo[1], .1);
    }
    const project = (p) => {
      const yaw = Number(rotation?.value || 0) * Math.PI / 180;
      const dx = p[0] - center[0], dy = p[1] - center[1], dz = p[2] - center[2];
      const scale = 530 / extent * Number(zoom.value);
      return [480 + (dx * Math.cos(yaw) - dy * Math.sin(yaw)) * scale,
        365 - (dz * .96 - (dx * Math.sin(yaw) + dy * Math.cos(yaw)) * .28) * scale];
    };
    function line(a, b, color, width) {
      context.beginPath(); context.moveTo(...a); context.lineTo(...b); context.strokeStyle = color; context.lineWidth = width; context.stroke();
    }
    function draw() {
      context.clearRect(0, 0, 960, 760); context.fillStyle = '#efe9df'; context.fillRect(0, 0, 960, 760);
      context.lineCap = 'round';
      if (skeleton) {
        const floor = 0;
        for (let i = -3; i <= 3; i++) {
          const n = i * extent / 4, span = extent * .85;
          line(project([center[0] + n, center[1] - span, floor]), project([center[0] + n, center[1] + span, floor]), '#dbd2c5', 1);
          line(project([center[0] - span, center[1] + n, floor]), project([center[0] + span, center[1] + n, floor]), '#dbd2c5', 1);
        }
        const target = elapsed + frames[0].time;
        while (frame < frames.length - 1 && frames[frame + 1].time <= target) frame++;
        while (frame > 0 && frames[frame].time > target) frame--;
        const current = frames[frame], next = frames[Math.min(frame + 1, frames.length - 1)];
        const fraction = next.time > current.time ? Math.min(1, (target - current.time) / (next.time - current.time)) : 0;
        let segments = current.segments.map((line, i) => line.map((p, end) => p.map((value, axis) => value + fraction * (next.segments[i][end][axis] - value))));
        if (values.motion !== 1 || values.arms !== 1 || values.legs !== 1 || Object.values(values.joints).some(Boolean)) segments = poseExperiment(segments, frames[0].segments, model.segment_ids, values);
        for (const segment of segments) {
          const a = project(segment[0]), b = project(segment[1]);
          line(a, b, '#214f50', 10);
          for (const p of [a, b]) {
            context.beginPath(); context.arc(...p, 6, 0, Math.PI * 2); context.fillStyle = '#fbf7f1'; context.fill();
            context.strokeStyle = '#965644'; context.lineWidth = 2; context.stroke();
          }
        }
      } else {
        const w = model.world, t = elapsed / 4;
        if (!interactive) x = model.segment.x0 + w.v0 * t - .5 * Math.sign(w.v0) * w.drag * t * t;
        const scale = Number(zoom.value), px = (value) => 480 + (value - .5) * 850 * scale;
        const py = (value) => 380 + (value - w.support_intercept) * 850 * scale;
        context.beginPath(); context.moveTo(px(0), py(w.support_intercept)); context.lineTo(px(1), py(w.support_intercept + w.support_slope)); context.lineTo(960, 760); context.lineTo(0, 760); context.fillStyle = '#e1dacd'; context.fill();
        context.beginPath(); context.arc(px(x), py(w.support_intercept + w.support_slope * x - w.radius), w.radius * 850 * scale, 0, Math.PI * 2);
        context.fillStyle = '#eac34c'; context.fill(); context.strokeStyle = '#b58c23'; context.lineWidth = 2; context.stroke();
      }
      seek.value = String(elapsed); time.textContent = `${elapsed.toFixed(1)} / ${duration.toFixed(1)}s`;
    }
    function pause() {
      playing = false; last = null;
      if (raf !== null) cancelAnimationFrame(raf); raf = null;
      play.setAttribute('aria-pressed', 'false'); play.setAttribute('aria-label', 'Play model'); play.title = 'Play model';
      play.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m9 5 11 7-11 7Z"/></svg>';
    }
    function tick(now) {
      raf = null;
      if (!playing || document.hidden || !visible) { pause(); return; }
      const dt = last === null ? 0 : Math.min((now - last) / 1000, .05) * Number(speed.value); last = now;
      elapsed = Math.min(duration, elapsed + dt);
      if (interactive && dt > 0) {
        [x, velocity] = root.WeftPhysics.advance(model.world, x, velocity, impulse, dt);
        impulse = 0;
        if (x < model.world.radius || x > 1 - model.world.radius) { x = Math.max(model.world.radius, Math.min(1 - model.world.radius, x)); velocity = 0; }
      }
      draw();
      if (elapsed >= duration) pause(); else raf = requestAnimationFrame(tick);
    }
    function start() {
      if (playing || document.hidden || !visible) return;
      if (elapsed >= duration) elapsed = 0;
      playing = true; play.setAttribute('aria-pressed', 'true'); play.setAttribute('aria-label', 'Pause model'); play.title = 'Pause model';
      play.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14M16 5v14"/></svg>';
      raf = requestAnimationFrame(tick);
    }
    play.addEventListener('click', () => playing ? pause() : start());
    seek.addEventListener('input', () => { pause(); interactive = false; elapsed = Number(seek.value); draw(); });
    zoom.addEventListener('input', draw); rotation?.addEventListener('input', draw);
    reset.addEventListener('click', () => { pause(); elapsed = frame = 0; interactive = false; zoom.value = '1'; speed.value = '1'; if (rotation) rotation.value = initialYaw; values.motion = values.arms = values.legs = 1; values.joints = {}; bodyInputs.forEach(input => { input.value = input.min === '-45' ? '0' : '1'; input.setAttribute('aria-valuetext', input.min === '-45' ? '0 degrees' : '100%'); }); edited(); });
    if (!skeleton) {
      const pushes = node('div', 'studio-pushes');
      for (const direction of [-1, 1]) {
        const button = icon(direction < 0 ? 'Try an authored push left' : 'Try an authored push right', direction < 0 ? '<path d="M19 12H5m6-6-6 6 6 6"/>' : '<path d="M5 12h14m-6-6 6 6-6 6"/>');
        button.addEventListener('click', () => { pause(); interactive = true; elapsed = 0; velocity = direction * .5; impulse = direction; start(); }); pushes.append(button);
      }
      knobs.append(pushes);
    }
    const visibility = () => { if (document.hidden) pause(); };
    document.addEventListener('visibilitychange', visibility);
    const observer = new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; if (!visible) pause(); }, { threshold: .05 });
    observer.observe(host); draw();
    return () => { pause(); observer.disconnect(); document.removeEventListener('visibilitychange', visibility); };
  }

  root.WeftStageVisuals = (host, visuals, signed) => {
    const controller = new AbortController(), cleanups = [], videos = [];
    let alive = true;
    for (const asset of (visuals || []).slice(0, 4)) {
      const figure = node('figure', 'stage-output');
      figure.append(node('figcaption', '', asset.label || 'Saved stage output'));
      if (asset.kind === 'facts') {
        const grid = node('dl', 'stage-facts');
        for (const [label, value] of (asset.items || []).slice(0, 8)) {
          const row = node('div'); row.append(node('dt', '', label), node('dd', '', value)); grid.append(row);
        }
        figure.append(grid);
      } else if (asset.kind === 'image' && asset.url) {
        const image = node('img'); image.src = signed(asset.url); image.alt = asset.label;
        image.loading = 'lazy'; image.decoding = 'async'; figure.append(image);
      } else if (asset.kind === 'video' && asset.url) {
        const film = node('video'); film.src = signed(asset.url); film.controls = true;
        film.muted = true; film.playsInline = true; film.preload = 'metadata';
        if (asset.poster) film.poster = signed(asset.poster);
        else film.addEventListener('loadedmetadata', () => { if (film.paused) film.currentTime = Math.min(.1, film.duration / 2); }, { once: true });
        film.setAttribute('aria-label', asset.label); figure.append(film); videos.push(film);
        const observer = new IntersectionObserver(([entry]) => { if (!entry.isIntersecting) film.pause(); });
        observer.observe(film); cleanups.push(() => observer.disconnect());
      } else if (asset.kind === 'model' && (asset.url || asset.data)) {
        const canvasHost = node('div', 'studio-model-host');
        canvasHost.append(node('p', 'hint', 'Opening the saved model…')); figure.append(canvasHost);
        let loaded = false;
        const observer = new IntersectionObserver(async ([entry]) => {
          if (!entry.isIntersecting || loaded) return;
          loaded = true;
          try {
            const model = asset.data || await readJSON(signed(asset.url), controller.signal);
            if (alive) cleanups.push(modelViewer(canvasHost, model));
          } catch (error) {
            if (alive && error.name !== 'AbortError') canvasHost.replaceChildren(node('p', 'hint', 'Preview unavailable. The saved bundle remains below.'));
          }
        });
        observer.observe(canvasHost); cleanups.push(() => observer.disconnect());
      } else continue;
      host.append(figure);
    }
    const pause = () => { if (document.hidden) videos.forEach(film => film.pause()); };
    document.addEventListener('visibilitychange', pause);
    return () => {
      alive = false; controller.abort(); cleanups.forEach(cleanup => cleanup());
      document.removeEventListener('visibilitychange', pause);
      videos.forEach(film => { film.pause(); film.removeAttribute('src'); film.load(); });
    };
  };

  root.WeftCharacterPreference = () => {
    const picker = $('character-picker'), input = $('prompt'), trigger = $('character-trigger');
    const indicator = $('character-indicator'), note = $('character-note');
    let suggestion = '';
    const update = () => {
      const value = input.value.trim();
      trigger.setAttribute('data-filled', String(Boolean(value)));
      trigger.setAttribute('aria-label', value ? `Character: ${value}. Edit character` : 'Optional character');
      indicator.textContent = value ? 'Character selected' : 'No character selected';
      note.textContent = value && value === suggestion ? 'Measured with this clip. Reuses the existing character.'
        : value ? 'Custom character generation is optional and experimental.' : 'Leave blank to use the standard characters.';
    };
    input.addEventListener('input', update);
    picker.addEventListener('toggle', () => { if (picker.open) input.focus({ preventScroll: true }); });
    document.addEventListener('pointerdown', event => { if (picker.open && !picker.contains(event.target)) picker.open = false; });
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape' && picker.open) { picker.open = false; trigger.focus(); }
    });
    update();
    return { prefill(character) {
      const replace = !input.value.trim() || input.value.trim() === suggestion;
      suggestion = character?.prompt || '';
      if (replace) input.value = suggestion;
      update();
    } };
  };

  // Symbolic movement cues, not reconstructed joint data.
  root.WeftExampleMotions = picker => {
    const standing = [[40,12],[40,23],[40,43],[29,33],[24,43],[51,33],[56,43],[31,54],[26,66],[49,54],[54,66]];
    const squat = [[40,25],[40,36],[40,53],[28,37],[23,29],[52,37],[57,29],[22,54],[26,66],[58,54],[54,66]];
    const overhead = pose => pose.map((point, i) => ({3:[29,pose[1][1]-8],4:[26,pose[1][1]-22],5:[51,pose[1][1]-8],6:[54,pose[1][1]-22]})[i] || point);
    const waiter = pose => pose.map((point, i) => ({3:[29,pose[1][1]-8],4:[26,pose[1][1]-22]})[i] || point);
    const frames = {
      'bodyweight-squats': [standing, squat],
      'overhead-squat': [overhead(standing), overhead(squat)],
      'waiters-squat': [waiter(standing), waiter(squat)],
      'deadlift': [
        [[41,12],[39,23],[32,43],[42,35],[44,48],[35,36],[37,49],[34,54],[40,66],[31,55],[36,66]],
        [[59,28],[51,36],[29,44],[51,49],[49,61],[47,49],[45,61],[36,55],[40,66],[32,55],[36,66]],
      ],
      'incline-press': [
        [[52,21],[46,31],[30,48],[34,32],[40,20],[56,40],[62,28],[44,53],[49,66],[30,58],[35,66]],
        [[52,21],[46,31],[30,48],[45,19],[48,7],[55,22],[60,10],[44,53],[49,66],[30,58],[35,66]],
      ],
      'jumping-jacks': [standing,
        [[40,8],[40,19],[40,39],[27,12],[18,4],[53,12],[62,4],[26,51],[14,65],[54,51],[66,65]],
        standing, [[48,36],[43,45],[27,51],[48,55],[53,66],[51,54],[58,66],[36,56],[28,66],[22,60],[16,66]]],
      'sots-press': [squat, overhead(squat)],
      'lunge-twist': [
        [[41,15],[40,26],[38,46],[29,33],[39,36],[51,33],[41,36],[23,48],[21,66],[53,56],[66,66]],
        [[33,17],[35,28],[38,46],[22,29],[14,25],[43,35],[30,37],[23,48],[21,66],[53,56],[66,66]],
      ],
      'single-leg-squat': [
        [[38,12],[37,23],[35,43],[46,29],[55,27],[42,32],[54,30],[33,54],[35,66],[49,45],[63,49]],
        [[36,28],[34,39],[27,52],[45,40],[57,37],[42,43],[55,40],[40,55],[35,66],[46,52],[66,53]],
      ],
    };
    let active = null, raf = null;
    const svgNode = (tag, attrs = {}) => {
      const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
      for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
      return element;
    };
    function stop() {
      if (raf !== null) cancelAnimationFrame(raf);
      raf = null;
      active?.setAttribute('data-motion', 'false'); active = null;
    }
    function decorate(button, id) {
      if (id !== 'rolling-motion' && !frames[id]) return;
      const svg = svgNode('svg', { viewBox: '0 -4 80 76', class: 'example-motion', 'aria-hidden': 'true', focusable: 'false' });
      const ground = svgNode('path', { d: 'M12 68H68', class: 'motion-ground' });
      const body = svgNode('path'), head = svgNode('circle', { r: id === 'rolling-motion' ? 10 : 5 });
      svg.append(ground, body, head); button.append(svg);
      button.setAttribute('data-motion', 'false');
      const draw = progress => {
        if (id === 'rolling-motion') {
          const x = 22 + progress * 36, angle = progress * 3.6;
          head.setAttribute('cx', x); head.setAttribute('cy', 56);
          body.setAttribute('d', `M${x} 56l${Math.sin(angle)*8} ${-Math.cos(angle)*8}`);
          return;
        }
        const poses = frames[id], position = progress * poses.length;
        const index = Math.floor(position) % poses.length, t = (1-Math.cos((position%1)*Math.PI))/2;
        const points = poses[index].map((p, i) => p.map((v, axis) => v + (poses[(index+1)%poses.length][i][axis]-v)*t));
        head.setAttribute('cx', points[0][0]); head.setAttribute('cy', points[0][1]);
        body.setAttribute('d', [[1,2],[1,3,4],[1,5,6],[2,7,8],[2,9,10]].map(chain => chain.map((i,j) => `${j?'L':'M'}${points[i].join(' ')}`).join('')).join(''));
      };
      draw(.25);
      const start = () => {
        if (!picker.open || document.hidden || active === button) return;
        stop(); active = button; button.setAttribute('data-motion', 'true');
        if (root.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches) { draw(.25); return; }
        let began;
        const tick = now => {
          raf = null;
          if (!picker.open || document.hidden) { stop(); return; }
          began ??= now;
          const progress = Math.min(1, (now-began)/2400);
          draw(progress);
          if (progress < 1) raf = requestAnimationFrame(tick);
        };
        raf = requestAnimationFrame(tick);
      };
      const leave = () => { if (active === button) stop(); };
      button.addEventListener('pointerenter', event => { if (event.pointerType !== 'touch') start(); });
      button.addEventListener('pointerleave', leave);
      button.addEventListener('focus', start); button.addEventListener('blur', leave);
    }
    picker.addEventListener('toggle', () => { if (!picker.open) stop(); });
    document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
    root.addEventListener('pagehide', stop);
    return { decorate, stop };
  };

  root.WeftGallery = () => {
    const original = $('demo-source-video'), result = $('demo-video');
    const play = $('gallery-play'), label = $('gallery-play-label'), seek = $('gallery-seek');
    const time = $('gallery-time'), reset = $('gallery-reset'), toggle = $('gallery-interact');
    const host = $('gallery-model'), transport = $('gallery-transport'), status = $('gallery-status');
    let media = [], selected = null, sign = value => value, playing = false, pending = false, buffering = false;
    let raf = null, epoch = 0, selection = 0, interactionEpoch = 0, duration = 0, interactive = false, visible = true;
    let modelAbort = null, modelDispose = null, modelCache = null;
    const cancelFrame = () => { if (raf !== null) cancelAnimationFrame(raf); raf = null; };
    const update = () => {
      const position = Math.min(duration, Math.max(0, result.currentTime || 0));
      seek.value = String(position); time.textContent = `${position.toFixed(1)} / ${duration.toFixed(1)}s`;
      play.setAttribute('aria-pressed', String(playing));
      play.setAttribute('aria-label', playing ? 'Pause comparison' : media.length > 1 ? 'Play both videos' : 'Play recording');
      label.textContent = playing ? 'Pause' : media.length > 1 ? 'Play together' : 'Play recording';
      play.firstElementChild.textContent = playing ? 'Ⅱ' : '▷';
    };
    function pause() {
      epoch++; playing = pending = buffering = false; cancelFrame(); media.forEach(video => video.pause()); status.textContent = ''; update();
    }
    function metadata() {
      if (!media.length) return;
      const lengths = media.map(video => video.duration);
      duration = lengths.every(value => Number.isFinite(value) && value > 0) ? Math.min(...lengths) : 0;
      seek.max = String(duration || 1); play.disabled = seek.disabled = reset.disabled = !duration; update();
    }
    function sync() {
      raf = null;
      if (!playing || buffering || document.hidden || !visible || interactive) return;
      const position = result.currentTime;
      if (position >= duration - .025 || media.some(video => video.ended)) { pause(); return; }
      if (media.length > 1 && !original.seeking && Math.abs(original.currentTime - position) > .065) original.currentTime = position;
      update(); raf = requestAnimationFrame(sync);
    }
    async function resume() {
      if (pending || !playing || document.hidden || !visible || interactive) return;
      const request = epoch; pending = true; buffering = false;
      try {
        await Promise.all(media.map(video => video.play()));
        if (request !== epoch) { if (!playing) media.forEach(video => video.pause()); return; }
        pending = false;
        if (buffering && !media.every(item => item.readyState >= 3)) return;
        buffering = false; status.textContent = ''; cancelFrame(); raf = requestAnimationFrame(sync); update();
      } catch {
        if (request !== epoch) return;
        pause(); status.textContent = 'Playback could not start. Press play to try again.';
      }
    }
    function start() {
      if (!duration || playing || interactive || document.hidden || !visible) return;
      media.forEach(video => { video.preload = 'auto'; });
      if (result.currentTime >= duration - .05) media.forEach(video => { video.currentTime = 0; });
      else media.forEach(video => { video.currentTime = result.currentTime; });
      playing = true; update(); resume();
    }
    const playPause = () => playing ? pause() : start();
    play.addEventListener('click', playPause);
    for (const video of [original, result]) {
      video.addEventListener('loadedmetadata', metadata);
      video.addEventListener('ended', () => { if (!interactive) pause(); });
      video.addEventListener('error', () => { if (media.includes(video)) { pause(); status.textContent = 'This video could not load. Choose another example or reload the page.'; } });
      video.addEventListener('waiting', () => {
        if (!playing || interactive || !media.includes(video)) return;
        // Pausing rejects pending play promises; invalidate them without losing play intent.
        epoch++; pending = false; buffering = true; cancelFrame();
        media.forEach(item => item.pause()); status.textContent = 'Buffering the comparison…';
        if (media.every(item => item.readyState >= 3)) resume();
      });
      video.addEventListener('canplay', () => { if (buffering && media.every(item => item.readyState >= 3)) resume(); });
      video.addEventListener('click', () => { if (!interactive) playPause(); });
    }
    seek.addEventListener('input', () => { const position = Math.min(duration, Number(seek.value)); pause(); media.forEach(video => { video.currentTime = position; }); update(); });
    reset.addEventListener('click', () => { pause(); media.forEach(video => { video.currentTime = 0; }); update(); });
    function exitInteraction() {
      interactionEpoch++;
      modelAbort?.abort(); modelAbort = null; modelDispose?.(); modelDispose = null;
      interactive = false; toggle.setAttribute('aria-pressed', 'false'); toggle.setAttribute('aria-label', 'Inspect the model'); toggle.title = 'Inspect the model';
      host.hidden = true; host.replaceChildren(); result.hidden = false; original.controls = false; transport.hidden = false;
      status.textContent = ''; pause();
    }
    toggle.addEventListener('click', async () => {
      if (interactive) { exitInteraction(); return; }
      pause(); interactive = true;
      toggle.setAttribute('aria-pressed', 'true'); toggle.setAttribute('aria-label', 'Return to paired playback'); toggle.title = 'Return to paired playback';
      result.hidden = true; host.hidden = false; original.controls = true; transport.hidden = true;
      host.replaceChildren(node('p', 'studio-placeholder', 'Opening the model…'));
      const request = selection, visit = ++interactionEpoch; modelAbort = new AbortController();
      try {
        if (!modelCache) {
          const manifest = selected.model_url ? null : await readJSON(exhibit(`${selected.id}.json`), modelAbort.signal);
          const model = await readJSON(manifest ? exhibit(manifest.model) : sign(selected.model_url), modelAbort.signal);
          if (request !== selection || visit !== interactionEpoch || !interactive) return;
          modelCache = model;
        }
        if (request !== selection || visit !== interactionEpoch || !interactive) return;
        modelDispose = modelViewer(host, modelCache);
      } catch (error) {
        if (request === selection && visit === interactionEpoch && interactive && error.name !== 'AbortError') host.replaceChildren(node('p', 'studio-placeholder', 'Model inspection is unavailable. Use the toggle to return to the recording.'));
      }
    });
    const visibility = () => { if (document.hidden) pause(); };
    document.addEventListener('visibilitychange', visibility);
    const observer = new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; if (!visible) pause(); }, { threshold: .05 });
    observer.observe($('finished-content'));
    const hint = $('gallery-interact-hint');
    let hintTimer = null, hintSeen = false;
    try { hintSeen = sessionStorage.getItem('weft-model-hint-seen') === 'yes'; } catch { /* Optional storage never blocks the controls. */ }
    function dismissHint() {
      clearTimeout(hintTimer); hintTimer = null; hint.hidden = true; toggle.removeAttribute('aria-describedby');
    }
    const hintObserver = new IntersectionObserver(([entry]) => {
      if (!entry.isIntersecting || toggle.hidden || document.hidden || hintSeen) return;
      hintSeen = true; hint.hidden = false; toggle.setAttribute('aria-describedby', 'gallery-interact-hint');
      try { sessionStorage.setItem('weft-model-hint-seen', 'yes'); } catch { /* Show once per page when storage is unavailable. */ }
      hintTimer = setTimeout(dismissHint, 4500); hintObserver.disconnect();
    }, { threshold: 1 });
    hintObserver.observe(toggle);
    toggle.addEventListener('click', dismissHint);
    document.addEventListener('visibilitychange', () => { if (document.hidden) dismissHint(); });
    root.addEventListener('pagehide', () => { dismissHint(); hintObserver.disconnect(); });
    const picker = $('example-picker');
    document.addEventListener('pointerdown', event => { if (picker.open && !picker.contains(event.target)) picker.open = false; });
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape' && picker.open) { picker.open = false; picker.querySelector('summary').focus(); }
    });
    root.addEventListener('pagehide', () => { pause(); exitInteraction(); });
    return {
      pause,
      select(record, signed) {
        selection++; selected = record; exitInteraction(); sign = signed; modelCache = null;
        media = record.source_video_url ? [original, result] : [result];
        duration = 0; play.disabled = seek.disabled = reset.disabled = true;
        toggle.hidden = /^run-/.test(record.id) && !record.model_url;
        update(); metadata();
      },
    };
  };

  root.WeftStudioPage = {
    modelViewer, poseExperiment,
    open(options) {
      const { record, entry, world, signed } = options;
      lastOptions = options;
      dispose();
      const controller = new AbortController(), cleanups = [];
      const body = $('modal-body');
      for (const child of [...body.children]) child.hidden = true;
      $('studio-tour')?.remove();
      const tour = node('div', 'studio-tour'); tour.id = 'studio-tour'; body.append(tour);
      const videos = [];
      let alive = true;
      dispose = () => { alive = false; controller.abort(); cleanups.forEach(fn => fn()); videos.forEach(video => { video.pause(); video.removeAttribute('src'); video.load(); }); };
      const pauseVideos = () => { if (document.hidden) videos.forEach(video => video.pause()); };
      document.addEventListener('visibilitychange', pauseVideos);
      cleanups.push(() => document.removeEventListener('visibilitychange', pauseVideos));
      const mediaObserver = new IntersectionObserver(entries => entries.forEach(item => { if (!item.isIntersecting) item.target.pause(); }));
      cleanups.push(() => mediaObserver.disconnect());
      const video = (url, label, deferred = false) => {
        const media = node('video'); media.controls = true; media.playsInline = true; media.preload = deferred ? 'none' : 'metadata'; media.src = signed(url);
        const poster = url === entry.video_url ? (entry.poster_url || record.poster_url) : (entry.source_poster_url || record.source_poster_url);
        if (poster) media.poster = signed(poster);
        else { media.preload = 'metadata'; media.addEventListener('loadedmetadata', () => { if (media.paused) media.currentTime = Math.min(.1, media.duration / 2); }, { once: true }); }
        media.setAttribute('aria-label', label); videos.push(media); mediaObserver.observe(media); return media;
      };
      const intro = node('div', 'studio-result-heading');
      intro.append(node('p', 'eyebrow', 'THE RESULT'));
      const result = node('section', 'studio-result-pair'); result.id = 'finished-result'; result.setAttribute('aria-label', 'Initial input, model and recording');
      const initial = node('figure', 'studio-initial'); initial.id = 'initial-input';
      const initialCaption = node('figcaption');
      initialCaption.append(node('span', 'media-origin', '00 / INITIAL'), node('strong', '', entry.input_label || 'Input footage'));
      const sourceURL = entry.source_video_url || record.source_video_url;
      initial.append(initialCaption, sourceURL ? video(sourceURL, 'Initial input video') : node('p', 'studio-placeholder', 'The input video is not available for this saved run.'));
      const modelPanel = node('figure', 'studio-model-panel');
      const modelCaption = node('figcaption'); modelCaption.append(node('span', 'media-origin', '01 / MODEL'), node('strong', '', 'Reconstructed skeleton'));
      const modelHost = node('div', 'studio-model-host'); const modelMessage = node('p', 'studio-placeholder', 'Opening the saved model…'); modelMessage.setAttribute('role', 'status'); modelHost.append(modelMessage);
      modelPanel.append(modelCaption, modelHost);
      const film = node('figure'); const filmCaption = node('figcaption');
      filmCaption.append(node('span', 'media-origin', '02 / RECORDING'), node('strong', '', entry.result_label || 'Generated result'));
      film.append(filmCaption, entry.video_url ? video(entry.video_url, 'Final recorded simulation') : node('p', 'studio-placeholder', 'No final recording was saved for this run.'));
      result.append(initial, modelPanel, film); tour.append(intro, result);
      if (world) {
        const link = node('a', 'studio-lab-link', 'Try new actions in the live sandbox ↗');
        link.href = `/labs/${world}/#token=${encodeURIComponent(root.WeftDemoSession.token)}`;
        link.target = '_blank'; link.rel = 'noopener'; tour.append(link);
      }
      const walk = node('section', 'studio-walkthrough'); walk.id = 'pipeline';
      const heading = node('header', 'studio-walkthrough-head');
      heading.append(node('p', 'eyebrow', 'HOW IT WAS MADE'), node('h2', '', 'From video to a world you can test.'), node('p', '', 'Scroll through the steps to see what each one produced.'));
      const layout = node('div', 'studio-pipeline-layout'), nav = node('nav', 'studio-step-nav'); nav.setAttribute('aria-label', 'Generation steps');
      const stages = node('div', 'studio-stages'); layout.append(nav, stages); walk.append(heading, layout); tour.append(walk);
      const jump = node('a', 'studio-tour-link', 'Explore the pipeline ↓'); jump.href = '#pipeline'; intro.append(jump);

      function assetCard(asset, privateStep) {
        const box = node('div', 'studio-asset');
        const url = privateStep ? signed(`/api/demo/artifact/${privateStep.evidenceStep || privateStep.id}/${record.id}`) : exhibit(asset.name);
        const link = node('a', 'studio-asset-link', asset.label + ' ↓'); link.href = url; link.download = asset.name;
        box.append(link, node('span', 'studio-asset-format', privateStep ? 'ZIP · SAVED BUNDLE' : asset.kind === 'image' ? 'IMAGE · SAVED EVIDENCE' : 'JSON · SAVED EVIDENCE'));
        if (asset.kind === 'image') {
          const image = node('img'); image.src = url; image.loading = 'lazy'; image.alt = 'Saved joint-tracking contact sheet from the source video'; box.append(image);
        } else if (!privateStep && !asset.name.endsWith('-model.json')) {
          const details = node('details', 'studio-asset-inspect'); details.append(node('summary', '', 'Inspect this asset'));
          const preview = node('pre', '', 'Opening saved evidence…'); details.append(preview); box.append(details);
          let loaded = false;
          details.addEventListener('toggle', async () => {
            if (!details.open || loaded) return; loaded = true;
            try { const data = await readJSON(url, controller.signal); if (alive) { const text = JSON.stringify(data, null, 2); preview.textContent = text.length > 16000 ? text.slice(0, 16000) + '\n… Download the asset for the full record.' : text; } }
            catch (error) { if (alive) { preview.textContent = error.message; loaded = false; } }
          });
        }
        return box;
      }
      let visualModel = null;
      const visualHosts = [];
      function showStageModel(slot) {
        if (!visualModel || !slot.visible || slot.mounted || !alive) return;
        slot.mounted = true;
        try { cleanups.push(modelViewer(slot.host, visualModel)); }
        catch { slot.host.replaceChildren(node('p', 'studio-placeholder', 'This model preview is unavailable. The saved assets remain below.')); }
      }
      function renderStages(steps) {
        steps = root.WeftPresentationSteps(steps);
        nav.replaceChildren(); stages.replaceChildren();
        if (!steps.length) { stages.append(node('p', 'hint', 'No stage records were saved with this example. The available recording remains above.')); return; }
        const items = [];
        steps.forEach((step, index) => {
          const number = String(index + 1).padStart(2, '0');
          const section = node('article', 'studio-step'); section.id = `pipeline-${step.id}`;
          const link = node('a', '', `${number}  ${stageCopy[step.id]?.[0].replace(/\.$/, '') || step.title}`); link.href = '#' + section.id; nav.append(link);
          const overline = node('p', 'eyebrow', `STEP ${number}`);
          if (step.status && !['done', 'complete'].includes(step.status)) overline.append(node('span', 'studio-stage-status', step.status));
          section.append(overline, node('h3', '', stageCopy[step.id]?.[0] || step.title), node('p', 'studio-step-description', stageCopy[step.id]?.[1] || step.description));
          if (['failed', 'skipped'].includes(step.status)) section.append(node('p', 'hint', step.error || step.detail));
          if (index === 0 && sourceURL && !step.visuals?.some(asset => asset.kind === 'video')) {
            const source = node('figure', 'studio-step-visual');
            source.append(video(sourceURL, 'Input footage for this stage', true));
            section.append(source);
          }
          if (step.visuals?.length) {
            const outputs = node('div', 'stage-outputs'); section.append(outputs);
            cleanups.push(root.WeftStageVisuals(outputs, step.visuals, signed));
          }
          if (step.id === 'world' && !step.visuals?.some(asset => asset.kind === 'model')) {
            const preview = node('figure', 'studio-step-visual');
            const host = node('div', 'studio-model-host');
            host.append(node('p', 'studio-placeholder', 'The saved model appears here when this step comes into view.'));
            preview.append(host);
            section.append(preview);
            const slot = { host, mounted: false, visible: false }; visualHosts.push(slot);
            const observer = new IntersectionObserver(([entry]) => { slot.visible = entry.isIntersecting; showStageModel(slot); });
            observer.observe(host); cleanups.push(() => observer.disconnect());
          }
          if (!privateRun && !step.visuals?.length && ['rollout', 'replay'].includes(step.id) && entry.video_url) {
            const preview = node('figure', 'studio-step-visual');
            preview.append(video(entry.video_url, 'Recorded output for this stage', true));
            section.append(preview);
          }
          const evidence = node('details', 'step-inspect'); evidence.append(node('summary', '', 'Details and files'));
          if (step.detail) evidence.append(node('p', 'hint', step.detail));
          if (step.metrics && Object.keys(step.metrics).length) {
            const metrics = node('dl', 'studio-step-metrics');
            for (const [key, value] of Object.entries(step.metrics)) { const row = node('div'); row.append(node('dt', '', key), node('dd', '', String(value))); metrics.append(row); } evidence.append(metrics);
          }
          const assets = node('div', 'studio-assets');
          for (const asset of step.assets || []) assets.append(assetCard(asset));
          if (step.artifact?.name) assets.append(assetCard({ name: step.artifact.name, label: 'Download this stage’s assets' }, step));
          const stageVideo = step.visuals?.find(asset => asset.kind === 'video')?.url
            || (!privateRun && ['rollout', 'replay'].includes(step.id) ? entry.video_url : null);
          if (stageVideo) {
            const savedVideo = node('a', 'studio-asset studio-asset-link', 'Download the recorded result ↓');
            savedVideo.href = signed(stageVideo); savedVideo.download = 'weft-result.mp4'; assets.append(savedVideo);
          }
          evidence.append(assets); section.append(evidence);

          if (index < steps.length - 1) { const next = node('a', 'studio-next', `Next: ${steps[index + 1].title} →`); next.href = `#pipeline-${steps[index + 1].id}`; section.append(next); }
          stages.append(section); items.push({ section, link });
        });
        items[0]?.link.setAttribute('aria-current', 'step');
        const observer = new IntersectionObserver(entries => {
          const active = entries.find(entry => entry.isIntersecting); if (!active) return;
          for (const item of items) { if (item.section === active.target) item.link.setAttribute('aria-current', 'step'); else item.link.removeAttribute('aria-current'); }
        }, { rootMargin: '-15% 0px -60% 0px' });
        items.forEach(item => observer.observe(item.section)); cleanups.push(() => observer.disconnect());
      }
      const privateRun = /^run-/.test(record.id || '');
      const steps = (record.steps || []).map(step => ({ ...step,
        title: stageCopy[step.id]?.[0] || step.title || step.id,
        description: stageCopy[step.id]?.[1] || step.explainer || '' }));
      renderStages(steps);
      (async () => {
        try {
          const manifest = privateRun ? null : await readJSON(exhibit(`${entry.id}.json`), controller.signal);
          if (!alive) return;
          if (manifest) { renderStages(manifest.steps); modelCaption.lastChild.textContent = manifest.model_label; }
          const url = manifest ? exhibit(manifest.model) : record.model_url ? signed(record.model_url) : null;
          if (!url) throw new Error('No inspectable model was saved. The recording and stage assets remain available.');
          const model = await readJSON(url, controller.signal); if (!alive) return;
          visualModel = model;
          cleanups.push(modelViewer(modelHost, model));
          visualHosts.forEach(showStageModel);
        } catch (error) {
          if (alive) { modelHost.replaceChildren(node('p', 'studio-placeholder', error.message)); visualHosts.forEach(slot => slot.host.replaceChildren(node('p', 'studio-placeholder', 'No model preview was saved. The available evidence is below.'))); }
        }
      })();
    },
  };
  root.addEventListener('pagehide', () => dispose());
  root.addEventListener('pageshow', event => {
    if (event.persisted && lastOptions) root.WeftStudioPage.open(lastOptions);
  });
})(window);
