"""A shared, bounded 2D rigid-body world; image widths are not metric units."""

from copy import deepcopy
import json
import math
import re

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image, ImageDraw
import pymunk

from weft.brand import COLORS


ACTIONS = ("coast", "left", "right", "up", "down", "brake")
MAX_SPEED = 4.0
MAX_ANGULAR_SPEED = 20.0
SCENE_UNITS = {"length": "scene_width", "mass": "relative", "time": "second",
               "angle": "radian", "axes": "x_right_y_down"}


def _number(value, name, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and between {low} and {high}")
    return float(value)


def _vector(value, name, low, high):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must have two coordinates")
    return [_number(v, name, low, high) for v in value]


def _local_extent(obj):
    return ([obj["radius"], obj["radius"]] if obj["shape"] == "circle"
            else [v / 2 for v in obj["size"]])


def _extent(obj):
    half = _local_extent(obj)
    if obj["shape"] == "circle" or not obj.get("angle", 0):
        return half
    c, s = abs(math.cos(obj["angle"])), abs(math.sin(obj["angle"]))
    return [half[0] * c + half[1] * s, half[0] * s + half[1] * c]


def _axes(obj):
    c, s = math.cos(obj.get("angle", 0)), math.sin(obj.get("angle", 0))
    return ((c, s), (-s, c))


def _overlap(a, b, padding=0.0):
    ax, ay = a["position"]
    bx, by = b["position"]
    if a["shape"] == b["shape"] == "circle":
        return math.hypot(ax - bx, ay - by) < a["radius"] + b["radius"] + padding
    if a["shape"] == "circle" or b["shape"] == "circle":
        circle, box = (a, b) if a["shape"] == "circle" else (b, a)
        if box.get("angle", 0):
            offset = [x - y for x, y in zip(circle["position"], box["position"])]
            delta = [abs(sum(v * axis for v, axis in zip(offset, direction))) - half
                     for direction, half in zip(_axes(box), _local_extent(box))]
        else:
            delta = [abs(x - y) - half for x, y, half in zip(circle["position"], box["position"], _extent(box))]
        return math.hypot(max(0, delta[0]), max(0, delta[1])) < circle["radius"] + padding
    if a.get("angle", 0) or b.get("angle", 0):
        offset = [ax - bx, ay - by]
        axes_a, axes_b = _axes(a), _axes(b)
        for direction in axes_a + axes_b:
            distance = abs(sum(v * axis for v, axis in zip(offset, direction)))
            radii = [sum(half * abs(sum(x * y for x, y in zip(axis, direction)))
                         for half, axis in zip(_local_extent(obj), axes))
                     for obj, axes in ((a, axes_a), (b, axes_b))]
            if distance >= sum(radii) + padding - 1e-12:
                return False
        return True
    return all(abs(x - y) < h1 + h2 + padding for x, y, h1, h2 in
               zip(a["position"], b["position"], _extent(a), _extent(b)))


def validate_scene(scene):
    if not isinstance(scene, dict):
        raise ValueError("Scene must be an object")
    result = deepcopy(scene)
    version = result.get("schema_version")
    if type(version) is not int or version not in (1, 2) or result.get("profile") != f"planar-rigid-v{version}":
        raise ValueError("Unsupported scene schema/profile")
    required = {"schema_version", "profile", "width", "height", "dt", "gravity", "damping", "restitution",
                "objects", "controlled_id", "target", "task", "relationships", "assumptions"}
    if version == 2:
        required |= {"units", "boundary_friction"}
    if set(result) != required:
        raise ValueError(f"Scene fields differ from schema: {sorted(set(result) ^ required)}")
    if version == 2:
        if result["units"] != SCENE_UNITS:
            raise ValueError("Scene units must declare normalized length, relative mass, seconds and radians")
        result["boundary_friction"] = _number(result["boundary_friction"], "boundary_friction", 0, 2)
    if _number(result["width"], "width", 1, 1) != 1:
        raise ValueError("Scene width must be one image width")
    result["height"] = _number(result["height"], "height", 0.2, 2)
    result["dt"] = _number(result["dt"], "dt", 1 / 240, 1 / 20)
    result["gravity"] = _vector(result["gravity"], "gravity", -20, 20)
    result["damping"] = _number(result["damping"], "damping", 0.01, 1)
    result["restitution"] = _number(result["restitution"], "restitution", 0, 1)
    if not isinstance(result["objects"], list) or not 1 <= len(result["objects"]) <= 12:
        raise ValueError("Scene requires 1–12 objects")
    identifiers = set()
    for obj in result["objects"]:
        common = {"id", "label", "role", "shape", "position", "velocity", "color", "mass", "provenance"}
        if version == 2:
            common |= {"angle", "angular_velocity", "friction"}
        if not isinstance(obj, dict) or not isinstance(obj.get("shape"), str) or obj["shape"] not in {"circle", "box"}:
            raise ValueError("Each object must be a circle or box")
        geometry = "radius" if obj["shape"] == "circle" else "size"
        if set(obj) != common | {geometry}:
            raise ValueError("Object fields differ from schema")
        if not isinstance(obj["id"], str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", obj["id"]) or obj["id"] in identifiers:
            raise ValueError("Object IDs must be unique short identifiers")
        identifiers.add(obj["id"])
        if not isinstance(obj["label"], str) or not 1 <= len(obj["label"]) <= 100:
            raise ValueError("Object labels must have 1–100 characters")
        if not isinstance(obj["role"], str) or obj["role"] not in {"active", "passive", "static"}:
            raise ValueError("Invalid object role")
        if not isinstance(obj["color"], str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", obj["color"]):
            raise ValueError("Object color must be #RRGGBB")
        obj["position"] = _vector(obj["position"], "position", 0, 2)
        obj["velocity"] = _vector(obj["velocity"], "velocity", -MAX_SPEED, MAX_SPEED)
        if math.hypot(*obj["velocity"]) > MAX_SPEED:
            raise ValueError("Object velocity exceeds the speed envelope")
        if obj["role"] == "static" and obj["velocity"] != [0, 0]:
            raise ValueError("Static objects cannot have velocity")
        if version == 2:
            obj["angle"] = _number(obj["angle"], "angle", -math.pi, math.pi)
            obj["angular_velocity"] = _number(obj["angular_velocity"], "angular_velocity", -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)
            obj["friction"] = _number(obj["friction"], "friction", 0, 2)
            if obj["role"] == "static" and obj["angular_velocity"] != 0:
                raise ValueError("Static objects cannot have angular velocity")
        obj["mass"] = _number(obj["mass"], "mass", 0.1, 100)
        if obj["shape"] == "circle":
            obj["radius"] = _number(obj["radius"], "radius", 0.008, 0.3)
        else:
            obj["size"] = _vector(obj["size"], "size", 0.016, 0.99 if version == 2 and obj["role"] == "static" else 0.6)
        if not all(extent + 0.002 <= coordinate <= limit - extent - 0.002 for coordinate, extent, limit in
                   zip(obj["position"], _extent(obj), [1, result["height"]])):
            raise ValueError("Object geometry must lie inside the assumed boundary walls")
        if not isinstance(obj["provenance"], dict):
            raise ValueError("Object provenance must be explicit")
    active = [o for o in result["objects"] if o["role"] == "active"]
    if len(active) != 1 or active[0]["id"] != result["controlled_id"]:
        raise ValueError("Exactly one active object must match controlled_id")
    for i, obj in enumerate(result["objects"]):
        if any(_overlap(obj, other) for other in result["objects"][i + 1:]):
            raise ValueError("Initial objects overlap; correct measurements or geometry before simulation")
    result["target"] = _vector(result["target"], "target", 0, 2)
    if not all(extent + 0.002 <= coordinate <= limit - extent - 0.002 for coordinate, extent, limit in
               zip(result["target"], _extent(active[0]), [1, result["height"]])):
        raise ValueError("Target must fit the controlled object inside the world")
    target_body = active[0] | {"position": result["target"]}
    if any(_overlap(target_body, obj) for obj in result["objects"] if obj["role"] == "static"):
        raise ValueError("Target overlaps a static obstacle")
    task = result["task"]
    if not isinstance(task, dict) or set(task) != {"position_tolerance", "speed_tolerance", "max_steps", "acceleration"}:
        raise ValueError("Task fields differ from schema")
    task["position_tolerance"] = _number(task["position_tolerance"], "position_tolerance", 0.01, 0.2)
    task["speed_tolerance"] = _number(task["speed_tolerance"], "speed_tolerance", 0.01, 0.5)
    task["acceleration"] = _number(task["acceleration"], "acceleration", 0.1, 10)
    if type(task["max_steps"]) is not int or not 10 <= task["max_steps"] <= 900:
        raise ValueError("max_steps must be an integer between 10 and 900")
    if not isinstance(result["relationships"], list) or len(result["relationships"]) > 100:
        raise ValueError("Relationships must be a bounded list")
    for relationship in result["relationships"]:
        if (not isinstance(relationship, dict) or set(relationship) != {"type", "objects", "status"}
                or not isinstance(relationship["objects"], list) or not relationship["objects"]
                or any(not isinstance(identifier, str) or identifier not in identifiers for identifier in relationship["objects"])
                or not isinstance(relationship["status"], str)
                or relationship["status"] not in {"observed", "inferred", "assumed", "unknown"}
                or not isinstance(relationship["type"], str)):
            raise ValueError("Relationship must reference existing objects and state its evidence status")
    if not isinstance(result["assumptions"], list) or len(result["assumptions"]) > 100 or any(
            not isinstance(item, str) or len(item) > 2000 for item in result["assumptions"]):
        raise ValueError("Assumptions must be bounded text records")
    try:
        if len(json.dumps(result, allow_nan=False)) > 250_000:
            raise ValueError("Scene metadata exceeds 250 KB")
    except (TypeError, OverflowError) as error:
        raise ValueError("Scene metadata must be finite JSON values") from error
    return result


def task_success(scene, position, velocity):
    return (math.dist(position, scene["target"]) <= scene["task"]["position_tolerance"]
            and math.hypot(*velocity) <= scene["task"]["speed_tolerance"])


class SceneEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, scene, render_mode=None):
        self.scene = validate_scene(scene)
        self.oriented = self.scene["schema_version"] == 2
        self._minimum_extent = min(min(_local_extent(obj)) for obj in self.scene["objects"])
        self._radii = {obj["id"]: (obj["radius"] if obj["shape"] == "circle" else math.hypot(*obj["size"]) / 2)
                       for obj in self.scene["objects"] if obj["role"] != "static"}
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(np.array([-2, -2, -MAX_SPEED, -MAX_SPEED], dtype=np.float32),
                                            np.array([2, 2, MAX_SPEED, MAX_SPEED], dtype=np.float32))
        self._done = True
        self.space = None
        self.bodies = {}

    def _observation(self):
        body = self.bodies[self.scene["controlled_id"]]
        return np.array([self.scene["target"][0] - body.position.x,
                         self.scene["target"][1] - body.position.y,
                         body.velocity.x, body.velocity.y], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if set(options) - {"jitter"} or type(options.get("jitter", True)) is not bool:
            raise ValueError("Reset accepts only boolean jitter")
        objects = deepcopy(self.scene["objects"])
        if options.get("jitter", True):
            for obj in objects:
                if obj["role"] == "static":
                    continue
                original = obj["position"][:]
                for _ in range(20):
                    proposal = np.array(original) + self.np_random.uniform(-0.06, 0.06, 2)
                    extent = np.array(_extent(obj)) + 0.003
                    obj["position"] = np.clip(proposal, extent, np.array([1, self.scene["height"]]) - extent).tolist()
                    if not any(_overlap(obj, other, 0.002) for other in objects if obj is not other):
                        break
                else:
                    obj["position"] = original
        self.space = pymunk.Space(threaded=False)
        self.space.gravity = tuple(self.scene["gravity"])
        self.space.damping = self.scene["damping"]
        self.space.iterations = 20
        self.space.collision_slop = 0.0005
        height = self.scene["height"]
        corners = [(0, 0), (1, 0), (1, height), (0, height)]
        for a, b in zip(corners, corners[1:] + corners[:1]):
            wall = pymunk.Segment(self.space.static_body, a, b, 0.002)
            wall.elasticity = math.sqrt(self.scene["restitution"])
            wall.friction = self.scene.get("boundary_friction", 0.5)
            self.space.add(wall)
        self.bodies = {}
        self.shapes = {}
        self.contacts = set()
        for obj in objects:
            if obj["role"] == "static":
                body = pymunk.Body(body_type=pymunk.Body.STATIC)
            else:
                moment = (pymunk.moment_for_circle(obj["mass"], 0, obj["radius"]) if obj["shape"] == "circle"
                          else pymunk.moment_for_box(obj["mass"], tuple(obj["size"])))
                body = pymunk.Body(obj["mass"], moment)
            body.position = tuple(obj["position"])
            body.velocity = tuple(obj["velocity"])
            body.angle = obj.get("angle", 0)
            body.angular_velocity = obj.get("angular_velocity", 0)
            shape = (pymunk.Circle(body, obj["radius"]) if obj["shape"] == "circle"
                     else pymunk.Poly.create_box(body, tuple(obj["size"])))
            shape.elasticity = math.sqrt(self.scene["restitution"])
            shape.friction = obj.get("friction", 0.5)
            self.space.add(body, shape)
            self.bodies[obj["id"]] = body
            self.shapes[shape] = obj["id"]
        self.space.on_collision(None, None, post_solve=self._contact)
        self.steps = 0
        self._done = self.terminated = self.truncated = False
        self.success = False
        self.speed_limited = False
        self.angular_speed_limited = self.substep_limited = False
        self.substeps = 0
        return self._observation(), {"is_success": False, "reset_jitter": options.get("jitter", True)}

    def _contact(self, arbiter, space, data):
        identifiers = tuple(sorted(self.shapes.get(shape, "boundary") for shape in arbiter.shapes))
        self.contacts.add(identifiers)

    def step(self, action):
        if self._done or self.space is None:
            raise RuntimeError("Reset before stepping and after an episode ends")
        return self._advance(action, evaluate_task=True)

    def advance_physics(self, action=0):
        if self.space is None or self.steps >= 1800:
            raise RuntimeError("Physics rollout requires reset and at most 1800 steps")
        self._advance(action, evaluate_task=False)
        return self.snapshot()

    def _advance(self, action, *, evaluate_task):
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError("Action must be an integer from 0 to 5")
        action = int(action)
        body = self.bodies[self.scene["controlled_id"]]
        old_distance = math.dist(body.position, self.scene["target"])
        acceleration = self.scene["task"]["acceleration"]
        direction = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))[action]
        dt = self.scene["dt"]
        max_velocity = max(b.velocity.length for b in self.bodies.values())
        predicted_speed = min(MAX_SPEED, max_velocity + (acceleration + math.hypot(*self.scene["gravity"])) * dt)
        if self.oriented:
            predicted_speed += max(abs(self.bodies[key].angular_velocity) * radius for key, radius in self._radii.items())
        # Use local thickness, not rotated bounds, so thin supports cannot reduce integration accuracy.
        required_substeps = max(4, math.ceil(predicted_speed * dt / (0.4 * self._minimum_extent)))
        substeps = min(256 if self.oriented else 64, required_substeps)
        self.substeps = substeps
        self.substep_limited = substeps < required_substeps
        self.contacts.clear()
        self.speed_limited = False
        self.angular_speed_limited = False
        for _ in range(substeps):
            if action == 5 and body.velocity.length > 0:
                speed = body.velocity.length
                body.velocity *= max(0, speed - acceleration * dt / substeps) / speed
            elif action != 0:
                body.apply_force_at_world_point((direction[0] * acceleration * body.mass,
                                                 direction[1] * acceleration * body.mass), body.position)
            self.space.step(dt / substeps)
            for other in self.bodies.values():
                if not all(math.isfinite(v) for v in (*other.position, *other.velocity, other.angle, other.angular_velocity)):
                    self._done = True
                    raise RuntimeError("Simulation produced nonfinite state")
                if other.velocity.length > MAX_SPEED:
                    other.velocity *= MAX_SPEED / other.velocity.length
                    self.speed_limited = True
                if self.oriented and abs(other.angular_velocity) > MAX_ANGULAR_SPEED:
                    other.angular_velocity = math.copysign(MAX_ANGULAR_SPEED, other.angular_velocity)
                    self.angular_speed_limited = True
        self.steps += 1
        distance = math.dist(body.position, self.scene["target"])
        self.success = evaluate_task and task_success(self.scene, body.position, body.velocity)
        escaped = any(not -0.02 <= b.position.x <= 1.02 or not -0.02 <= b.position.y <= self.scene["height"] + 0.02
                      for b in self.bodies.values())
        self.terminated = bool(self.success or escaped)
        self.truncated = evaluate_task and self.steps >= self.scene["task"]["max_steps"] and not self.terminated
        self._done = self.terminated or self.truncated
        reward = 20 * (old_distance - distance) - 0.025 - 0.015 * body.velocity.length
        reward += 10 if self.success else -10 if escaped else 0
        return self._observation(), float(reward), self.terminated, self.truncated, {
            "is_success": self.success, "escaped": escaped, "distance": distance,
            "speed": body.velocity.length, "steps": self.steps, "speed_limited": self.speed_limited,
            "contacts": [list(pair) for pair in sorted(self.contacts)]}

    def snapshot(self):
        if self.space is None:
            raise RuntimeError("Reset before requesting state")
        objects = []
        for obj in self.scene["objects"]:
            body = self.bodies[obj["id"]]
            objects.append({key: deepcopy(value) for key, value in obj.items() if key != "provenance"} | {
                "position": list(body.position), "velocity": list(body.velocity), "angle": body.angle})
            if self.oriented:
                objects[-1]["angular_velocity"] = body.angular_velocity
        return {"objects": objects, "target": self.scene["target"][:], "controlled_id": self.scene["controlled_id"],
                "width": 1, "height": self.scene["height"], "steps": self.steps, "time_s": self.steps * self.scene["dt"],
                "success": self.success, "terminated": self.terminated, "truncated": self.truncated,
                "speed_limited": self.speed_limited, "contacts": [list(pair) for pair in sorted(self.contacts)]} | (
                    {"angular_speed_limited": self.angular_speed_limited, "substep_limited": self.substep_limited,
                     "substeps": self.substeps, "units": deepcopy(self.scene["units"])} if self.oriented else {})

    def render(self):
        state = self.snapshot()
        scale = min(640, 720 / self.scene["height"])
        image = Image.new("RGB", (round(scale), round(self.scene["height"] * scale)), COLORS["surface"])
        draw = ImageDraw.Draw(image)
        draw.rectangle((1, 1, image.width - 2, image.height - 2), outline=COLORS["ink"], width=2)
        x, y = [v * scale for v in self.scene["target"]]
        radius = self.scene["task"]["position_tolerance"] * scale
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=COLORS["primary"], width=2)
        draw.line((x - 5, y, x + 5, y), fill=COLORS["primary"])
        draw.line((x, y - 5, x, y + 5), fill=COLORS["primary"])
        for obj in state["objects"]:
            x, y = [v * scale for v in obj["position"]]
            if obj["shape"] == "circle":
                radius = obj["radius"] * scale
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=obj["color"], outline=COLORS["ink"], width=2)
                draw.line((x, y, x + math.cos(obj["angle"]) * radius, y + math.sin(obj["angle"]) * radius), fill=COLORS["ink"])
            else:
                hw, hh = [v * scale / 2 for v in obj["size"]]
                c, s = math.cos(obj["angle"]), math.sin(obj["angle"])
                points = [(x + px * c - py * s, y + px * s + py * c) for px, py in
                          [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]]
                draw.polygon(points, fill=obj["color"], outline=COLORS["ink"], width=2)
            draw.text((x + 8, y + 8), obj["id"], fill=COLORS["ink"])
        draw.text((8, 8), f"SIMULATION | step {self.steps} | authored controls + target", fill=COLORS["ink"])
        return np.asarray(image)

    def close(self):
        self.space = None
        self.bodies.clear()
        self._done = True


def rollout(scene, actions, *, seed=0, jitter=False):
    if not isinstance(actions, (list, tuple)) or len(actions) > 900:
        raise ValueError("Rollout must contain at most 900 actions")
    env = SceneEnv(scene)
    env.reset(seed=seed, options={"jitter": jitter})
    states = [env.snapshot()]
    try:
        for action in actions:
            _, reward, terminated, truncated, _ = env.step(action)
            states.append(env.snapshot() | {"action": int(action), "reward": reward})
            if terminated or truncated:
                break
    finally:
        env.close()
    return states
