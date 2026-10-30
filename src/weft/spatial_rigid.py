"""Explicit 3D rigid-body hypotheses with bounded virtual forces and full-state observations."""

from copy import deepcopy
import math
import re
import xml.etree.ElementTree as ET

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from PIL import Image, ImageDraw


PROFILE = 'spatial-rigid-v1'
UNITS = {'length': 'scene_width', 'mass': 'relative', 'time': 'second',
         'axes': 'x_right_y_away_z_up', 'orientation': 'quaternion_wxyz', 'angular_velocity': 'world_radians_per_second'}
ACTIONS = ('coast', 'left', 'right', 'forward', 'back', 'up', 'down', 'brake')
DIRECTIONS = np.array([[0, 0, 0], [-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [0, 0, 0]])
MAX_SPEED, MAX_SPIN, MAX_CONTACTS = 4., 200., 256
STATE_EPS = 1e-8
FACES = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))


def _number(value, name, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be finite and between {low} and {high}')
    return float(value)


def _vector(value, count, name, low, high):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f'{name} requires {count} coordinates')
    return [_number(v, name, low, high) for v in value]


def rotation(quaternion):
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, np.asarray(quaternion, dtype=float))
    return result.reshape(3, 3)


def corners(obj):
    half = np.asarray(obj['size']) / 2
    local = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half
    return local @ rotation(obj['quaternion']).T + obj['position']


def validate_scene(scene):
    keys = {'schema_version', 'profile', 'units', 'bounds', 'dt', 'gravity', 'objects', 'controlled_id',
            'target', 'task', 'contact', 'assumptions'}
    if not isinstance(scene, dict) or set(scene) != keys or type(scene['schema_version']) is not int or scene['schema_version'] != 1 or scene['profile'] != PROFILE:
        raise ValueError('Unsupported spatial rigid scene contract')
    result = deepcopy(scene)
    if result['units'] != UNITS:
        raise ValueError('Spatial scenes require explicit normalized units and world axes')
    result['bounds'] = _vector(result['bounds'], 3, 'bounds', .2, 2)
    if result['bounds'][0] != 1:
        raise ValueError('Spatial scene width must be one normalized unit')
    result['dt'] = _number(result['dt'], 'dt', 1/240, 1/30)
    result['gravity'] = _vector(result['gravity'], 3, 'gravity', -20, 20)
    if np.linalg.norm(result['gravity']) > 20:
        raise ValueError('Gravity exceeds the integration envelope')
    contact = result['contact']
    if not isinstance(contact, dict) or set(contact) != {'time_constant', 'damping_ratio'}:
        raise ValueError('Contact solver assumptions must be explicit')
    contact['time_constant'] = _number(contact['time_constant'], 'contact time constant', .005, .05)
    contact['damping_ratio'] = _number(contact['damping_ratio'], 'contact damping ratio', .5, 2)
    if not isinstance(result['objects'], list) or not 1 <= len(result['objects']) <= 12:
        raise ValueError('Spatial scene requires 1–12 rigid objects')
    identifiers = set()
    for obj in result['objects']:
        required = {'id', 'label', 'role', 'shape', 'position', 'quaternion', 'velocity', 'angular_velocity',
                    'mass', 'friction', 'color', 'provenance'}
        shape = obj.get('shape') if isinstance(obj, dict) else None
        if shape not in {'box', 'sphere'} or set(obj) != required | ({'size'} if shape == 'box' else {'radius'}):
            raise ValueError('Spatial object fields differ from the primitive contract')
        if not isinstance(obj['id'], str) or not re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', obj['id']) or obj['id'] in identifiers:
            raise ValueError('Spatial object IDs must be unique safe identifiers')
        identifiers.add(obj['id'])
        if obj['role'] not in {'active', 'passive', 'static'} or not isinstance(obj['label'], str) or not 1 <= len(obj['label']) <= 120:
            raise ValueError('Invalid spatial object role or label')
        if not isinstance(obj['color'], str) or not re.fullmatch(r'#[0-9A-Fa-f]{6}', obj['color']):
            raise ValueError('Object colors require hexadecimal RGB')
        obj['position'] = _vector(obj['position'], 3, 'position', 0, 2)
        obj['quaternion'] = _vector(obj['quaternion'], 4, 'quaternion', -1, 1)
        if not math.isclose(np.linalg.norm(obj['quaternion']), 1, rel_tol=0, abs_tol=1e-8):
            raise ValueError('Object orientation must be a unit quaternion')
        obj['velocity'] = _vector(obj['velocity'], 3, 'velocity', -MAX_SPEED, MAX_SPEED)
        obj['angular_velocity'] = _vector(obj['angular_velocity'], 3, 'angular velocity', -MAX_SPIN, MAX_SPIN)
        if np.linalg.norm(obj['velocity']) > MAX_SPEED or np.linalg.norm(obj['angular_velocity']) > MAX_SPIN:
            raise ValueError('Initial speed or spin exceeds its envelope')
        if obj['role'] == 'static' and any(obj['velocity'] + obj['angular_velocity']):
            raise ValueError('Static geometry cannot have initial motion')
        obj['mass'] = _number(obj['mass'], 'relative mass', .1, 100)
        obj['friction'] = _vector(obj['friction'], 3, 'sliding/torsional/rolling friction', 0, 2)
        if shape == 'box':
            obj['size'] = _vector(obj['size'], 3, 'box size', .01, .99 if obj['role'] == 'static' else .6)
            points = corners(obj)
            low, high = points.min(axis=0), points.max(axis=0)
        else:
            obj['radius'] = _number(obj['radius'], 'radius', .008, .3)
            low, high = np.asarray(obj['position']) - obj['radius'], np.asarray(obj['position']) + obj['radius']
        if (low < 0).any() or (high > result['bounds']).any():
            raise ValueError('Initial object geometry exceeds scene bounds')
        if not isinstance(obj['provenance'], dict):
            raise ValueError('Spatial object provenance must be explicit')
    active = [o for o in result['objects'] if o['role'] == 'active']
    if len(active) != 1 or active[0]['id'] != result['controlled_id']:
        raise ValueError('Exactly one object must be the controlled actor')
    result['target'] = _vector(result['target'], 3, 'target', 0, 2)
    if (np.asarray(result['target']) > result['bounds']).any():
        raise ValueError('Target lies outside the scene bounds')
    task = result['task']
    if not isinstance(task, dict) or set(task) != {'position_tolerance', 'speed_tolerance', 'max_steps', 'acceleration'}:
        raise ValueError('Spatial task fields differ from contract')
    task['position_tolerance'] = _number(task['position_tolerance'], 'position tolerance', .01, .2)
    task['speed_tolerance'] = _number(task['speed_tolerance'], 'speed tolerance', .01, .5)
    task['acceleration'] = _number(task['acceleration'], 'virtual acceleration', .1, 10)
    if type(task['max_steps']) is not int or not 10 <= task['max_steps'] <= 900:
        raise ValueError('Spatial episodes require 10–900 steps')
    if not isinstance(result['assumptions'], list) or not 1 <= len(result['assumptions']) <= 100 or any(not isinstance(s, str) or len(s) > 2000 for s in result['assumptions']):
        raise ValueError('Spatial assumptions must be explicit bounded text')
    import json
    if len(json.dumps(result, allow_nan=False)) > 250000:
        raise ValueError('Spatial scene metadata exceeds its bound')
    return result


def model_xml(scene):
    root = ET.Element('mujoco', model='weft-spatial-rigid')
    ET.SubElement(root, 'compiler', angle='radian')
    ET.SubElement(root, 'option', timestep=str(scene['dt']/32), gravity=' '.join(map(str, scene['gravity'])),
                  integrator='implicitfast', cone='elliptic', solver='Newton', iterations='40', tolerance='1e-9')
    ET.SubElement(root, 'size', memory='16M')
    defaults = ET.SubElement(root, 'default')
    ET.SubElement(defaults, 'geom', condim='6', solref=f"{scene['contact']['time_constant']} {scene['contact']['damping_ratio']}",
                  solimp='.99 .99 .001', margin='0')
    world = ET.SubElement(root, 'worldbody')
    for obj in scene['objects']:
        body = ET.SubElement(world, 'body', name=obj['id'], pos=' '.join(map(str, obj['position'])),
                             quat=' '.join(map(str, obj['quaternion'])))
        if obj['role'] != 'static':
            ET.SubElement(body, 'freejoint', name=obj['id'] + '_joint')
        size = np.asarray(obj['size']) / 2 if obj['shape'] == 'box' else [obj['radius']]
        ET.SubElement(body, 'geom', name=obj['id'], type=obj['shape'], size=' '.join(map(str, size)),
                      mass=str(obj['mass']), friction=' '.join(map(str, obj['friction'])))
    return ET.tostring(root, encoding='unicode')


class SpatialRigidEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 30}

    def __init__(self, scene, render_mode=None):
        self.scene = validate_scene(scene)
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_string(model_xml(self.scene))
        self.data = mujoco.MjData(self.model)
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(-MAX_SPIN, MAX_SPIN, shape=(13*len(self.scene['objects']) + 4,), dtype=np.float32)
        self._ids = {obj['id']: self.model.body(obj['id']).id for obj in self.scene['objects']}
        self._minimum_extent = min(min(obj['size'])/2 if obj['shape'] == 'box' else obj['radius'] for obj in self.scene['objects'])
        self._radii = {obj['id']: float(np.linalg.norm(obj['size'])/2) if obj['shape'] == 'box' else obj['radius']
                       for obj in self.scene['objects'] if obj['role'] != 'static'}
        self._initialized, self._done = False, True

    def _objects(self):
        result = []
        for obj in self.scene['objects']:
            body = self._ids[obj['id']]
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
            result.append({k: deepcopy(v) for k, v in obj.items() if k != 'provenance'} | {
                'position': self.data.xpos[body].tolist(), 'quaternion': self.data.xquat[body].tolist(),
                'velocity': velocity[3:].tolist(), 'angular_velocity': velocity[:3].tolist()})
        return result

    def _observation(self, objects):
        controlled = next(o for o in objects if o['id'] == self.scene['controlled_id'])
        state = [value for o in objects for field in ('position', 'quaternion', 'velocity', 'angular_velocity') for value in o[field]]
        state += (np.asarray(self.scene['target']) - controlled['position']).tolist()
        state += [self.steps / self.scene['task']['max_steps']]
        observation = np.asarray(state, dtype=np.float32)
        if not self.observation_space.contains(observation):
            raise RuntimeError('Spatial observation exceeds declared finite bounds')
        return observation

    def select_actor(self, object_id):
        if not isinstance(object_id, str):
            raise ValueError('Controlled object ID must be a string')
        target = next((obj for obj in self.scene['objects'] if obj['id'] == object_id), None)
        if target is None:
            raise ValueError(f'Unknown spatial object: {object_id}')
        if target['role'] == 'static':
            raise ValueError(f'Static object {object_id!r} cannot be controlled')
        proposed = deepcopy(self.scene)
        for obj in proposed['objects']:
            if obj['role'] != 'static':
                obj['role'] = 'active' if obj['id'] == object_id else 'passive'
        proposed['controlled_id'] = object_id
        self.scene = validate_scene(proposed)
        if self._initialized:
            objects = self._objects()
            controlled = next(o for o in objects if o['id'] == object_id)
            distance = math.dist(controlled['position'], self.scene['target'])
            self.success = bool(
                distance <= self.scene['task']['position_tolerance']
                and np.linalg.norm(controlled['velocity']) <= self.scene['task']['speed_tolerance']
            )
            outside = any(
                (np.asarray(o['position']) < -.2).any()
                or (np.asarray(o['position']) > np.asarray(self.scene['bounds']) + .2).any()
                for o in objects if o['role'] != 'static'
            )
            self.terminated = bool(self.success or outside)
            self.truncated = bool(self.steps >= self.scene['task']['max_steps'] and not self.terminated)
            self._done = self.terminated or self.truncated
            self._force = [0., 0., 0.]
            self.data.xfrc_applied[:] = 0
            return self._observation(objects)
        return None

    def reset(self, *, seed=None, options=None):
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError('Seed must be a bounded nonnegative integer')
        if options is not None and not isinstance(options, dict):
            raise ValueError('Spatial reset options must be a mapping')
        options = options or {}
        if set(options) - {'jitter', 'position_jitter', 'velocity_jitter'}:
            raise ValueError('Spatial reset accepts only jitter, position_jitter, and velocity_jitter')
        use_jitter = options.get('jitter', False)
        if type(use_jitter) is not bool:
            raise ValueError('Spatial reset jitter must be a boolean')
        if not use_jitter and (options.get('position_jitter', 0.) != 0. or options.get('velocity_jitter', 0.) != 0.):
            raise ValueError('position_jitter and velocity_jitter require jitter=True')
        pos_jitter = _number(options.get('position_jitter', .03 if use_jitter else 0.), 'position_jitter', 0., .15)
        vel_jitter = _number(options.get('velocity_jitter', .15 if use_jitter else 0.), 'velocity_jitter', 0., 1.)
        super().reset(seed=seed)
        self.model.opt.timestep = self.scene['dt']/32
        mujoco.mj_resetData(self.model, self.data)
        for obj in self.scene['objects']:
            if obj['role'] == 'static':
                continue
            joint = self.model.joint(obj['id'] + '_joint').id
            address = self.model.jnt_dofadr[joint]
            self.data.qvel[address:address+3] = obj['velocity']
            self.data.qvel[address+3:address+6] = rotation(obj['quaternion']).T @ obj['angular_velocity']
        mujoco.mj_forward(self.model, self.data)
        for i, obj in enumerate(self.scene['objects']):
            for other in self.scene['objects'][i+1:]:
                if obj['role'] == other['role'] == 'static':
                    continue
                distance = mujoco.mj_geomDistance(self.model, self.data, self.model.geom(obj['id']).id,
                    self.model.geom(other['id']).id, .01, None)
                if distance < -1e-6:
                    raise ValueError('Initial spatial objects intersect')
        perturbations = {}
        if use_jitter:
            bounds = np.asarray(self.scene['bounds'], dtype=float)
            for obj in self.scene['objects']:
                if obj['role'] == 'static':
                    continue
                joint = self.model.joint(obj['id'] + '_joint').id
                qadr = self.model.jnt_qposadr[joint]
                vadr = self.model.jnt_dofadr[joint]
                nominal_pos = np.asarray(obj['position'], dtype=float)
                nominal_vel = np.asarray(obj['velocity'], dtype=float)
                chosen_pos = nominal_pos.copy()
                fallback = False
                if pos_jitter > 0.:
                    margin = min(float(self._radii[obj['id']]) + .005, float(np.min(bounds)) * .45)
                    low = np.minimum(nominal_pos, margin)
                    high = np.maximum(nominal_pos, bounds - margin)
                    geom_id = self.model.geom(obj['id']).id
                    placed = False
                    for _ in range(32):
                        delta = np.array([
                            float(self.np_random.uniform(-pos_jitter, pos_jitter)),
                            float(self.np_random.uniform(-pos_jitter, pos_jitter)),
                            float(self.np_random.uniform(0., .5 * pos_jitter)),
                        ])
                        candidate = np.clip(nominal_pos + delta, low, high)
                        self.data.qpos[qadr:qadr+3] = candidate
                        mujoco.mj_forward(self.model, self.data)
                        if all(
                            mujoco.mj_geomDistance(
                                self.model, self.data, geom_id, self.model.geom(other['id']).id, .01, None
                            ) >= 1e-4
                            for other in self.scene['objects'] if other['id'] != obj['id']
                        ):
                            chosen_pos = candidate
                            placed = True
                            break
                    if not placed:
                        self.data.qpos[qadr:qadr+3] = nominal_pos
                        mujoco.mj_forward(self.model, self.data)
                        fallback = True
                chosen_vel = nominal_vel.copy()
                if vel_jitter > 0.:
                    chosen_vel = nominal_vel + self.np_random.uniform(-vel_jitter, vel_jitter, 3)
                    speed = float(np.linalg.norm(chosen_vel))
                    if speed > .8 * MAX_SPEED:
                        chosen_vel *= (.8 * MAX_SPEED) / speed
                    self.data.qvel[vadr:vadr+3] = chosen_vel
                perturbations[obj['id']] = {
                    'position_delta': (chosen_pos - nominal_pos).tolist(),
                    'velocity_delta': (chosen_vel - nominal_vel).tolist(),
                    'fallback_unperturbed': fallback,
                }
            mujoco.mj_forward(self.model, self.data)
            for i, obj in enumerate(self.scene['objects']):
                for other in self.scene['objects'][i+1:]:
                    if obj['role'] == other['role'] == 'static':
                        continue
                    distance = mujoco.mj_geomDistance(self.model, self.data, self.model.geom(obj['id']).id,
                        self.model.geom(other['id']).id, .01, None)
                    if distance < -1e-6:
                        raise ValueError('Perturbed spatial objects intersect')
        self.steps, self.substeps = 0, 0
        self.success = self.terminated = self.truncated = False
        self._initialized, self._done = True, False
        self._force = [0., 0., 0.]
        info = {'reset_jitter': use_jitter, 'physical_validation': 'unverified'}
        if use_jitter:
            info['perturbation'] = perturbations
        return self._observation(self._objects()), info

    def step(self, action):
        if not self._initialized or self._done:
            raise RuntimeError('Reset before stepping and after an episode ends')
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError('Unknown spatial action')
        objects = self._objects()
        controlled = next(o for o in objects if o['id'] == self.scene['controlled_id'])
        before = math.dist(controlled['position'], self.scene['target'])
        acceleration = self.scene['task']['acceleration']
        if action == 7:
            requested = -8 * np.asarray(controlled['velocity'])
            requested *= min(1., acceleration / max(1e-12, float(np.linalg.norm(requested))))
        else:
            requested = DIRECTIONS[action] * acceleration
        self.data.xfrc_applied[:] = 0
        self._force = (requested * controlled['mass']).tolist()
        self.data.xfrc_applied[self._ids[controlled['id']], :3] = self._force
        speed = max((np.linalg.norm(o['velocity']) + np.linalg.norm(o['angular_velocity']) * self._radii[o['id']]
                     for o in objects if o['role'] != 'static'), default=0)
        speed += self.scene['dt'] * (np.linalg.norm(self.scene['gravity']) + acceleration)
        self.substeps = max(math.ceil(self.scene['dt']/.001), math.ceil(self.scene['dt'] * speed / (.4*self._minimum_extent)))
        if self.substeps > 256:
            self._done = True
            raise RuntimeError('Spatial integration demand exceeds 256 substeps')
        self.model.opt.timestep = self.scene['dt'] / self.substeps
        mujoco.mj_step(self.model, self.data, nstep=self.substeps)
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        objects = self._objects()
        if (not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
                or any(np.linalg.norm(o['velocity']) > MAX_SPEED+STATE_EPS or np.linalg.norm(o['angular_velocity']) > MAX_SPIN+STATE_EPS for o in objects)):
            self._done = True
            raise RuntimeError('Spatial speed, spin or finite-state envelope exceeded; no silent clamping')
        controlled = next(o for o in objects if o['id'] == self.scene['controlled_id'])
        distance = math.dist(controlled['position'], self.scene['target'])
        self.success = distance <= self.scene['task']['position_tolerance'] and np.linalg.norm(controlled['velocity']) <= self.scene['task']['speed_tolerance']
        outside = any((np.asarray(o['position']) < -.2).any() or (np.asarray(o['position']) > np.asarray(self.scene['bounds']) + .2).any()
                      for o in objects if o['role'] != 'static')
        self.terminated = bool(self.success or outside)
        self.truncated = self.steps >= self.scene['task']['max_steps'] and not self.terminated
        self._done = self.terminated or self.truncated
        reward = before - distance - .001 + float(self.success)
        return self._observation(objects), float(reward), self.terminated, self.truncated, {
            'success': bool(self.success), 'left_model_bounds': bool(outside), 'substeps': self.substeps,
            'virtual_force': self._force[:], 'physical_validation': 'unverified'}

    def snapshot(self):
        if not self._initialized:
            raise RuntimeError('Reset before requesting spatial state')
        if self.data.ncon > MAX_CONTACTS:
            raise RuntimeError('Spatial contact count exceeds recording bounds')
        contacts, pairs = [], set()
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            ids = sorted([self.model.geom(contact.geom1).name, self.model.geom(contact.geom2).name])
            pairs.add(tuple(ids))
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            contacts.append({'objects': ids, 'position': contact.pos.tolist(), 'distance': float(contact.dist),
                             'normal_force': float(force[0])})
        return {'profile': PROFILE, 'units': deepcopy(UNITS), 'bounds': self.scene['bounds'][:],
            'objects': self._objects(), 'controlled_id': self.scene['controlled_id'], 'target': self.scene['target'][:],
            'steps': self.steps, 'time_s': self.steps*self.scene['dt'], 'success': bool(self.success),
            'terminated': self.terminated, 'truncated': self.truncated, 'contacts': [list(pair) for pair in sorted(pairs)],
            'contact_details': contacts, 'contact_count': len(contacts), 'substeps': self.substeps,
            'virtual_force': self._force[:], 'physical_validation': 'unverified'}

    def render(self):
        return render_snapshot(self.snapshot(), self.scene)

    def close(self):
        self._initialized, self._done = False, True


def validate_snapshot(state, scene):
    if not isinstance(state, dict) or state.get('profile') != PROFILE or state.get('units') != UNITS or state.get('bounds') != scene['bounds']:
        raise ValueError('Spatial recording must preserve profile, units and bounds')
    expected = {o['id']: o for o in scene['objects']}
    if not isinstance(state.get('objects'), list) or len(state['objects']) != len(expected):
        raise ValueError('Spatial recording must retain all objects')
    seen = set()
    for obj in state['objects']:
        if not isinstance(obj, dict) or obj.get('id') not in expected or obj['id'] in seen:
            raise ValueError('Spatial recording object identity differs')
        seen.add(obj['id'])
        original = expected[obj['id']]
        immutable = set(original) - {'provenance', 'position', 'quaternion', 'velocity', 'angular_velocity'}
        if any(obj.get(key) != original[key] for key in immutable):
            raise ValueError('Spatial recording geometry or material changed')
        _vector(obj.get('position'), 3, 'recorded position', -5, 5)
        _vector(obj.get('velocity'), 3, 'recorded velocity', -MAX_SPEED-STATE_EPS, MAX_SPEED+STATE_EPS)
        _vector(obj.get('angular_velocity'), 3, 'recorded spin', -MAX_SPIN-STATE_EPS, MAX_SPIN+STATE_EPS)
        quaternion = _vector(obj.get('quaternion'), 4, 'recorded quaternion', -1-1e-8, 1+1e-8)
        if not math.isclose(np.linalg.norm(quaternion), 1, abs_tol=1e-8, rel_tol=0):
            raise ValueError('Recorded spatial quaternion must be normalized')
        if original['role'] == 'static' and (not np.allclose(obj['position'], original['position'], atol=1e-10, rtol=0)
                or abs(float(np.dot(quaternion, original['quaternion']))) < 1-1e-10
                or any(obj['velocity'] + obj['angular_velocity'])):
            raise ValueError('Recorded static geometry must remain fixed')
    if state.get('controlled_id') != scene['controlled_id'] or state.get('target') != scene['target'] or state.get('physical_validation') != 'unverified':
        raise ValueError('Spatial recording must preserve the task and evidence scope')
    if type(state.get('steps')) is not int or not 0 <= state['steps'] <= scene['task']['max_steps']:
        raise ValueError('Recorded spatial step exceeds its bound')
    if not math.isclose(_number(state.get('time_s'), 'recorded time', 0, 60), state['steps']*scene['dt'], rel_tol=0, abs_tol=1e-9):
        raise ValueError('Recorded spatial time differs from its steps')
    if type(state.get('substeps')) is not int or not 0 <= state['substeps'] <= 256:
        raise ValueError('Recorded integration demand exceeds its bound')
    for flag in ('success', 'terminated', 'truncated'):
        if type(state.get(flag)) is not bool:
            raise ValueError('Spatial termination flags must be explicit')
    _vector(state.get('virtual_force'), 3, 'recorded force', -1000, 1000)
    details = state.get('contact_details')
    if not isinstance(details, list) or type(state.get('contact_count')) is not int or not len(details) == state['contact_count'] <= MAX_CONTACTS:
        raise ValueError('Spatial contact details must be complete and bounded')
    pairs = set()
    for contact in details:
        if not isinstance(contact, dict) or set(contact) != {'objects', 'position', 'distance', 'normal_force'}:
            raise ValueError('Spatial contact fields differ from contract')
        ids = contact['objects']
        if not isinstance(ids, list) or len(ids) != 2 or len(set(ids)) != 2 or not set(ids) <= seen:
            raise ValueError('Spatial contacts must name distinct known objects')
        pairs.add(tuple(sorted(ids)))
        _vector(contact['position'], 3, 'contact position', -5, 5)
        _number(contact['distance'], 'contact distance', -1, 1)
        _number(contact['normal_force'], 'simulated normal force', 0, 1e9)
    if state.get('contacts') != [list(pair) for pair in sorted(pairs)]:
        raise ValueError('Spatial contact summaries differ from details')


def render_snapshot(state, scene, width=960):
    validate_snapshot(state, scene)
    width = max(480, min(1280, int(width)))
    width += width % 2
    panel_width, height = width / 3, round(width*.34)
    height += height % 2
    image = Image.new('RGB', (width, height+70), '#F5EADC')
    draw = ImageDraw.Draw(image)
    bounds = np.asarray(scene['bounds'])
    center = bounds/2
    views = [('Front: x / height', np.array([[1, 0, 0], [0, 0, -1]]), np.array([0, 1, 0])),
             ('Top: x / depth', np.array([[1, 0, 0], [0, -1, 0]]), np.array([0, 0, 1])),
             ('3D: same simulated state', np.array([[.866, -.5, 0], [.25, .433, -.866]]), np.array([.433, .75, .5]))]
    scale = min(panel_width*.86, height*.77 / max(bounds))
    for index, (label, matrix, depth) in enumerate(views):
        offset = np.array([(index+.5)*panel_width, height*.55])
        project = lambda points: (np.asarray(points) - center) @ matrix.T * scale + offset
        draw.text((index*panel_width+8, 8), label, fill='#214F50')
        draw.line((index*panel_width, 28, index*panel_width, height), fill='#D8C9BA')
        polygons, spheres = [], []
        for obj in state['objects']:
            if obj['shape'] == 'sphere':
                spheres.append((float(np.asarray(obj['position']) @ depth), obj))
                continue
            vertices = corners(obj)
            for face in FACES:
                points = vertices[list(face)]
                polygons.append((obj['role'] != 'static', float(points.mean(axis=0) @ depth), points, obj['color']))
        for moving, _, points, color in sorted(polygons, key=lambda item: item[:2]):
            draw.polygon([tuple(p) for p in project(points)], fill=color if moving else None,
                         outline='#214F50' if moving else '#AAA399', width=1)
        for _, obj in sorted(spheres, key=lambda item: item[0]):
            x, y = project(obj['position']);r = obj['radius']*scale
            draw.ellipse((x-r, y-r, x+r, y+r), fill=obj['color'] if obj['role'] != 'static' else None,
                         outline='#214F50' if obj['role'] != 'static' else '#AAA399', width=1)
        x, y = project(state['target']);r = scene['task']['position_tolerance']*scale
        draw.ellipse((x-r, y-r, x+r, y+r), outline='#965644', width=2)
        for contact in state['contact_details']:
            x, y = project(contact['position'])
            draw.ellipse((x-2, y-2, x+2, y+2), fill='#965644')
    draw.text((10, height+10), f'3D SIMULATION | step {state["steps"]} | {state["time_s"]:.3f}s | {state["contact_count"]} contacts', fill='#214F50')
    draw.text((10, height+34), 'Wireframe supports keep moving objects visible. Depth and materials remain hypotheses.', fill='#965644')
    return np.asarray(image)
