from dataclasses import replace
from pathlib import Path
import math

import numpy as np

from weft.io import digest
from weft.world import advance, passive_position

REPLAY_MAX_RMSE = 0.03
MIN_RELATIVE_IMPROVEMENT = 0.05


def evaluator_identity():
    return {"version": "1.0", "sha256": digest(Path(__file__)),
            "max_replay_rmse_image_width": REPLAY_MAX_RMSE,
            "minimum_relative_improvement": MIN_RELATIVE_IMPROVEMENT}


def replay_metrics(world, rows, origin_time, x0=None, v0=None):
    if not rows:
        raise ValueError("Scoring requires observations")
    times = np.array([r["time_s"] - origin_time for r in rows])
    if np.any(times < 0):
        raise ValueError("Evaluation cannot precede initialization")
    x = passive_position(world, times, x0=x0, v0=v0)
    y = world.support_intercept + world.support_slope * x - world.radius
    observed_x = np.array([r["x_px"] / world.width_px for r in rows])
    observed_y = np.array([r["y_px"] / world.width_px for r in rows])
    distance = np.sqrt((x - observed_x) ** 2 + (y - observed_y) ** 2)
    rmse = float(np.sqrt(np.mean(distance ** 2)))
    return {"observations": len(rows), "rmse_image_width": rmse,
            "rmse_px": rmse * world.width_px,
            "max_error_px": float(np.max(distance) * world.width_px),
            "passes": rmse <= REPLAY_MAX_RMSE,
            "evidence_ids": [r["id"] for r in rows]}


def physics_checks(world):
    world.validate()
    checks = {}
    frictionless = replace(world, drag=0.0)
    x, v = advance(frictionless, 0.2, 0.3, 0, 0.1)
    checks["unforced_constant_velocity"] = abs(x - 0.23) < 1e-10 and abs(v - 0.3) < 1e-10
    x, v = advance(frictionless, 0.2, 0.0, 1, 0.1)
    checks["known_constant_force_solution"] = (abs(x - (0.2 + 0.5 * world.max_acceleration * 0.1 ** 2)) < 1e-10
                                               and abs(v - world.max_acceleration * 0.1) < 1e-10)
    x, v = 0.2, 0.4
    dissipative = replace(world, drag=0.2)
    velocities = []
    for _ in range(100):
        x, v = advance(dissipative, x, v, 0, 0.05)
        velocities.append(v)
    checks["passive_drag_stops_without_reversal"] = min(velocities) >= -1e-10 and abs(v) < 1e-10
    checks["passive_drag_does_not_add_energy"] = all(b <= a + 1e-10 for a, b in zip(velocities, velocities[1:]))
    checks["passive_stopping_distance"] = abs(x - (0.2 + 0.4 ** 2 / (2 * 0.2))) < 1e-8
    plus = advance(world, 0.5, 0.0, 1, 0.1)
    minus = advance(world, 0.5, 0.0, -1, 0.1)
    checks["left_right_symmetry"] = abs(plus[0] + minus[0] - 1) < 1e-10 and abs(plus[1] + minus[1]) < 1e-10
    checks["action_changes_state"] = plus[0] > 0.5 and minus[0] < 0.5
    checks["finite_state"] = all(math.isfinite(n) for n in (*plus, *minus))
    return {"passes": all(checks.values()), "checks": checks,
            "scope": "Analytic consistency of the restricted 1D engine; not proof of real-world interventions"}


def audit_episode(world, rows):
    initial = rows[:3]
    x0 = initial[0]["x_px"] / world.width_px
    dt = initial[-1]["time_s"] - initial[0]["time_s"]
    # Initial observations set the trial state; subsequent positions are scored, never fitted.
    v0 = (initial[-1]["x_px"] - initial[0]["x_px"]) / world.width_px / dt
    v0 += 0.5 * math.copysign(world.drag, v0) * dt
    return {"initialization_ids": [r["id"] for r in initial],
            "initial_velocity": v0,
            "metrics": replay_metrics(world, rows[3:], initial[0]["time_s"], x0, v0),
            "claim": "Same-video withheld-pass check, conditioned on three initialization frames"}
