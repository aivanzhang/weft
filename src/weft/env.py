from dataclasses import dataclass, asdict
import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from weft.world import advance


@dataclass(frozen=True)
class Task:
    name: str = "park_the_ball"
    position_tolerance: float = 0.04
    velocity_tolerance: float = 0.15
    max_steps: int = 100
    dt: float = 0.05
    provenance: str = "Authored push/brake actuator and parking objective; not inferred from the video"

    def to_dict(self):
        return asdict(self)


def task_success(x, velocity, target, task):
    return abs(x - target) <= task.position_tolerance and abs(velocity) <= task.velocity_tolerance


class RollingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, world, task=Task(), render_mode=None):
        self.world = world.validate()
        self.task = task
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(np.array([-2.0, -8.0], dtype=np.float32),
                                            np.array([2.0, 8.0], dtype=np.float32))
        self._done = True

    def _observation(self):
        return np.array([self.target - self.x, self.velocity], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if set(options) - {"x", "velocity", "target"}:
            raise ValueError("Unknown reset options")
        self.x = float(options.get("x", self.np_random.uniform(0.15, 0.35)))
        self.velocity = float(options.get("velocity", self.np_random.uniform(0.0, 0.7)))
        self.target = float(options.get("target", self.np_random.uniform(0.65, 0.85)))
        if not (all(math.isfinite(v) for v in (self.x, self.velocity, self.target))
                and self.world.radius <= self.x <= 1 - self.world.radius
                and self.world.radius <= self.target <= 1 - self.world.radius and abs(self.velocity) <= 2):
            raise ValueError("Reset state lies outside the task envelope")
        self.steps, self._done = 0, False
        return self._observation(), {"is_success": False}

    def step(self, action):
        if self._done:
            raise RuntimeError("Reset is required before stepping or after termination")
        if not self.action_space.contains(action):
            raise ValueError("Action must be 0 (left), 1 (coast), or 2 (right)")
        old_distance = abs(self.target - self.x)
        self.x, self.velocity = advance(self.world, self.x, self.velocity, int(action) - 1, self.task.dt)
        self.steps += 1
        success = task_success(self.x, self.velocity, self.target, self.task)
        out_of_bounds = not self.world.radius <= self.x <= 1 - self.world.radius
        terminated = success or out_of_bounds
        truncated = self.steps >= self.task.max_steps and not terminated
        self._done = terminated or truncated
        reward = 3 * (old_distance - abs(self.target - self.x)) - 0.02 - 0.01 * abs(self.velocity)
        reward += 10.0 if success else -3.0 if out_of_bounds else 0.0
        return self._observation(), float(reward), bool(terminated), bool(truncated), {
            "is_success": success, "out_of_bounds": out_of_bounds, "x": self.x,
            "velocity": self.velocity, "target": self.target, "steps": self.steps,
        }

    def render(self):
        from PIL import Image, ImageDraw
        image = Image.new("RGB", (640, 240), "#F4F4F0")
        draw = ImageDraw.Draw(image)
        draw.line((0, 180, 640, 180), fill="#2B2B2B", width=3)
        target = int(self.target * 640)
        draw.rectangle((target - 26, 90, target + 26, 185), outline="#C5A059", width=3)
        r = int(self.world.radius * 640)
        x = int(self.x * 640)
        draw.ellipse((x - r, 180 - 2 * r, x + r, 180), fill="#C5A059", outline="#2B2B2B", width=2)
        draw.text((12, 12), f"SIMULATED CONTROL | x={self.x:.3f} v={self.velocity:.3f} target={self.target:.3f}", fill="#2B2B2B")
        return np.asarray(image)
