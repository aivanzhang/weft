"""Known analytic motion and counterexamples, separate from later replay scores."""

from copy import deepcopy
import math

import numpy as np
import pytest

from weft.initial_motion import propose_initial_motion
from weft.scene import validate_scene


def scene(damping=1., gravity=(0., 0.)):
    return validate_scene({'schema_version': 1, 'profile': 'planar-rigid-v1', 'width': 1., 'height': .7,
        'dt': 1/30, 'gravity': list(gravity), 'damping': damping, 'restitution': .5,
        'objects': [{'id': 'actor', 'label': 'Synthetic actor', 'role': 'active', 'shape': 'circle',
                     'radius': .02, 'position': [.2, .3], 'velocity': [.04, .01], 'mass': 1.,
                     'color': '#747474', 'provenance': {'source': 'analytic fixture'}}],
        'controlled_id': 'actor', 'target': [.8, .3], 'relationships': [], 'assumptions': ['Analytic control'],
        'task': {'position_tolerance': .06, 'speed_tolerance': .15, 'max_steps': 300, 'acceleration': 2.}})


def rows(points, *, origin=40., object_id='actor', fps=10):
    return [{'id': f'source:frame:{index}:object:{object_id}', 'object_id': object_id,
             'frame': index, 'time_s': origin + index / fps, 'x': point[0], 'y': point[1]}
            for index, point in enumerate(points)]


def propose(spec, observations):
    return propose_initial_motion(spec, observations, width=240, origin_time_s=40.)


def test_quantized_longer_window_estimates_speed_without_moving_initial_body():
    spec = scene()
    observed = rows([[.2 + round(index * .3) / 240, .3] for index in range(120)])
    before = deepcopy((spec, observed))
    candidate, audit = propose(spec, observed)
    assert abs(candidate['objects'][0]['velocity'][0] * 240 - 3) < .2
    assert abs(candidate['objects'][0]['velocity'][1]) < 1e-12
    assert audit['objects']['actor']['samples'] == 11
    assert audit['objects']['actor']['source_frames'] == list(range(11))
    assert not audit['accepted']
    assert audit['input_partition'] == 'generation_only'
    original = deepcopy(candidate)
    original['objects'][0]['velocity'] = spec['objects'][0]['velocity']
    original['objects'][0]['provenance'].pop('velocity_repair')
    assert original == spec
    assert (spec, observed) == before


@pytest.mark.parametrize('damping', [1., .9999999, .6, .01])
def test_known_gravity_and_damping_are_not_mistaken_for_initial_speed(damping):
    spec = scene(damping, (.04, .03))
    velocity = np.array([.03, -.02])
    points = []
    for index in range(31):
        t = index / 30
        rate = math.log(damping)
        if abs(rate) < 1e-6:
            point = np.array([.2, .3]) + velocity * t + .5 * np.array(spec['gravity']) * t**2
        else:
            point = np.array([.2, .3]) + velocity * (math.exp(rate * t) - 1) / rate
            point += np.array(spec['gravity']) * ((math.exp(rate * t) - 1) / rate - t) / rate
        points.append(point.tolist())
    candidate, audit = propose(spec, rows(points, fps=30))
    assert candidate is not None, audit
    assert candidate['objects'][0]['velocity'] == pytest.approx(velocity, abs=1e-8)
    assert audit['objects']['actor']['max_error_px'] < 1e-6


@pytest.mark.parametrize('kind', ['acceleration', 'impulse', 'outlier', 'shifted_initial_position'])
def test_inconsistent_motion_is_not_hidden_by_dropping_points_or_shifting_body(kind):
    spec = scene()
    points = [[.2 + .03 * index / 10, .3] for index in range(11)]
    for index, point in enumerate(points):
        if kind == 'acceleration':
            point[1] += .5 * .4 * (index / 10)**2
        elif kind == 'impulse':
            point[0] += max(0, index - 5) * .02
        elif kind == 'outlier' and index == 5:
            point[0] += .03
        elif kind == 'shifted_initial_position':
            point[0] += .02
    candidate, audit = propose(spec, rows(points))
    assert candidate is None
    assert audit['objects']['actor']['samples'] == 11
    assert audit['objects']['actor']['reason'] == 'motion_inconsistent_with_free_flight_or_noise_allowance'


def test_boundary_contact_refuses_free_flight_fit():
    spec = scene()
    spec['objects'][0]['position'][1] = .025
    candidate, audit = propose(spec, rows([[.2 + .01 * i, .025] for i in range(11)]))
    assert candidate is None
    assert audit['objects']['actor']['reason'] == 'possible_boundary_contact'


@pytest.mark.parametrize('missing', [False, True])
def test_object_contact_or_missing_neighbor_cannot_support_free_flight(missing):
    spec = scene()
    other = deepcopy(spec['objects'][0])
    other.update(id='other', role='static', position=[.27, .3], velocity=[0., 0.])
    spec['objects'].append(other)
    observed = rows([[.2 + .003 * i, .3] for i in range(11)])
    observed += rows([[.27, .3] for _ in range(11)], object_id='other')[1 if missing else 0:]
    candidate, audit = propose(spec, observed)
    assert candidate is None
    assert audit['objects']['other']['status'] == 'unchanged_static'
    assert audit['objects']['actor']['reason'] == ('missing_other_object_contact_evidence' if missing else 'possible_object_contact')


def test_initial_gap_is_unavailable_not_interpolated():
    observed = rows([[.2 + .001 * i, .3] for i in range(11)])
    candidate, audit = propose(scene(), [row for row in observed if row['frame'] not in (4, 5)])
    assert candidate is None
    assert audit['objects']['actor']['reason'] == 'insufficient_contiguous_initial_time_support'


def test_construction_observations_beyond_frozen_window_do_not_change_proposal():
    observed = rows([[.2 + .001 * i, .3] for i in range(120)])
    baseline = propose(scene(), observed)
    for row in observed[11:]:
        row['x'], row['y'] = .8, .6
    assert propose(scene(), observed) == baseline


@pytest.mark.parametrize('change', ['duplicate', 'time_mismatch', 'nan', 'too_many'])
def test_invalid_or_excessive_evidence_rejects_before_fitting(change):
    observed = rows([[.2 + .001 * i, .3] for i in range(11)])
    if change == 'duplicate':
        observed.append(observed[0])
    elif change == 'time_mismatch':
        observed[1]['time_s'] = 39.
    elif change == 'nan':
        observed[1]['x'] = float('nan')
    else:
        observed *= 1400
    with pytest.raises(ValueError):
        propose(scene(), observed)
