from dataclasses import asdict, dataclass, replace
import math

import numpy as np


@dataclass(frozen=True)
class World:
    x0: float
    v0: float
    drag: float
    radius: float
    support_slope: float
    support_intercept: float
    width_px: int
    height_px: int
    source_frame: int
    max_acceleration: float = 2.0

    def validate(self):
        numeric = asdict(self)
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
               for v in numeric.values()):
            raise ValueError("World fields must be finite numbers")
        if not (0 <= self.x0 <= 1 and -4 <= self.v0 <= 4 and 0 <= self.drag <= 3
                and 0.005 <= self.radius <= 0.25 and abs(self.support_slope) < 0.5
                and 0 < self.support_intercept < 2 and 1 <= self.max_acceleration <= 4
                and 0 < self.width_px <= 4096 and 0 < self.height_px <= 2160
                and 0 <= self.source_frame <= 1800):
            raise ValueError("World violates the bounded rolling-body contract")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if set(value) != set(cls.__dataclass_fields__):
            raise ValueError("World schema contains unknown or missing fields")
        return cls(**value).validate()

    def patch(self, changes):
        if not changes or set(changes) - {"v0", "drag"}:
            raise ValueError("Repairs may only change v0 and drag")
        return replace(self, **changes).validate()

    def center_y(self, x):
        return self.support_intercept + self.support_slope * x - self.radius


def generate(evidence):
    rows = evidence["generation"]
    width = evidence["width"]
    x = np.array([r["x_px"] / width for r in rows])
    y = np.array([r["y_px"] / width for r in rows])
    radius = float(np.median([r["radius_px"] / width for r in rows]))
    slope, intercept = np.polyfit(x, y, 1)
    world = World(x0=float(x[0]), v0=float((x[2] - x[0]) / (rows[2]["time_s"] - rows[0]["time_s"])),
                  drag=0.0, radius=radius, support_slope=float(slope),
                  support_intercept=float(intercept + radius), width_px=evidence["width"],
                  height_px=evidence["height"], source_frame=rows[0]["frame"]).validate()
    graph = {
        "coordinate_system": "image x-right/y-down; one unit = source image width; time in seconds",
        "objects": [
            {"id": "ball-0", "kind": "dynamic_rigid_circle", "state": ["x", "velocity"],
             "display_rotation": "derived no-slip illustration; not measured or independently simulated",
             "geometry": {"radius": radius, "evidence": "generation masks"},
             "identity_across_passes": "assumed; not independently verified"},
            {"id": "support-0", "kind": "static_support", "state": "fixed",
             "geometry": {"slope": float(slope), "intercept": float(intercept + radius)},
             "evidence": "inferred from ball contact envelope; not a measured 3D surface"},
            {"id": "backdrop-0", "kind": "appearance_only", "state": "fixed",
             "evidence": "first source frame", "collisions": False},
        ],
        "relationships": [{"subject": "ball-0", "relation": "supported_by", "object": "support-0",
                           "status": "hypothesis"}],
        "assumptions": ["Fixed camera and planar motion", "Circular rigid body; no deformation",
                        "Effective image-plane drag conflates physical resistance and perspective",
                        "Push/brake actuator and parking goal are authored task extensions, not observed controls"],
        "unknown": ["metric scale", "mass", "force calibration", "depth", "unseen interactions"],
        "evidence_ids": [r["id"] for r in rows],
    }
    return world, graph


def passive_position(world, times, x0=None, v0=None):
    x0, v0 = world.x0 if x0 is None else x0, world.v0 if v0 is None else v0
    t = np.asarray(times, dtype=float)
    if world.drag > 0:
        t = np.minimum(t, abs(v0) / world.drag)
    return x0 + v0 * t - 0.5 * math.copysign(world.drag, v0) * t ** 2


def advance(world, x, velocity, action, dt):
    if action not in (-1, 0, 1) or not 0 < dt <= 0.1:
        raise ValueError("Invalid bounded action or timestep")
    # Substeps handle rest and braking without Coulomb friction reversing motion.
    steps = math.ceil(dt / 0.005)
    h = dt / steps
    for _ in range(steps):
        drive = action * world.max_acceleration
        if abs(velocity) < 1e-10 and abs(drive) <= world.drag:
            velocity = 0.0
            continue
        direction = math.copysign(1, velocity if abs(velocity) > 1e-10 else drive)
        acceleration = drive - direction * world.drag
        next_v = velocity + acceleration * h
        if velocity * next_v < 0:
            stop_t = -velocity / acceleration
            x += velocity * stop_t + 0.5 * acceleration * stop_t ** 2
            remainder = h - stop_t
            new_a = math.copysign(max(0, abs(drive) - world.drag), drive) if drive else 0.0
            velocity = new_a * remainder
            x += 0.5 * new_a * remainder ** 2
        else:
            x += velocity * h + 0.5 * acceleration * h ** 2
            velocity = next_v
    return x, velocity
