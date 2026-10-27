(function (root) {
  'use strict';
  function advance(world, x, velocity, action, dt) {
    if (![-1, 0, 1].includes(action) || !(dt > 0 && dt <= 0.1)) throw new Error('Invalid action or timestep');
    const steps = Math.ceil(dt / 0.005), h = dt / steps;
    for (let i = 0; i < steps; i++) {
      const drive = action * world.max_acceleration;
      if (Math.abs(velocity) < 1e-10 && Math.abs(drive) <= world.drag) { velocity = 0; continue; }
      const direction = Math.sign(Math.abs(velocity) > 1e-10 ? velocity : drive);
      const acceleration = drive - direction * world.drag;
      const nextV = velocity + acceleration * h;
      if (velocity * nextV < 0) {
        const stop = -velocity / acceleration, remaining = h - stop;
        x += velocity * stop + 0.5 * acceleration * stop * stop;
        const newA = drive ? Math.sign(drive) * Math.max(0, Math.abs(drive) - world.drag) : 0;
        velocity = newA * remaining;
        x += 0.5 * newA * remaining * remaining;
      } else { x += velocity * h + 0.5 * acceleration * h * h; velocity = nextV; }
    }
    return [x, velocity];
  }
  root.WeftPhysics = { advance };
})(globalThis);
