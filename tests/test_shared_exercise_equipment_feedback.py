"""Authored observation-contract checks only; no simulation or trained policy is run."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from types import SimpleNamespace

import numpy as np
import pytest

from weft import shared_exercise_equipment_feedback as actor
from weft import shared_exercise_feedback_residual as v5
from weft.mobile_human import ROOT_AXES
from weft.shared_exercise_task import SharedExerciseTaskEnv
from weft.spatial_pose import SPATIAL_JOINT_NAMES


def fixture(*, count=2, scale=1., origin=(0., 0., 0.), names=None, ids=None):
    names = tuple(names or (*SPATIAL_JOINT_NAMES, *ROOT_AXES))
    ids = ids or [f"object-{i}" for i in range(count)]
    units = ["prior_mass * prior_length^2 / second^2"] * 19 + ["prior_mass * prior_length / second^2"] * 3
    scene = {"profile": "shared-world-v2", "controlled_id": "human",
             "action_contract": {"names": list(names), "units": units, "physical_limits": [30.] * 19 + [1000.] * 3},
             "coordinate_frame": {"axes": "x_right_y_depth_z_up",
                                  "units": {"time": "second", "orientation": "quaternion_wxyz",
                                            "angular_velocity": "world_radians_per_second"},
                                  "scale_bridge": {"body_height_to_world_scale": scale, "origin_world": list(origin)}},
             "human_spec": {"profile": "exercise-v3", "mobility": {"base_height": 1.}}}
    task = {"profile": "shared-exercise-task-v1", "equipment_ids": list(ids), "contact_requirements": [],
            "control": {"root_assistance": "explicit_3_axis", "object_forces": "forbidden"}}
    env = SimpleNamespace(scene=scene, task=task, _scene=scene, _task=task, _static_support=False,
                          action_names=names, action_count=22, observation_size=156 + 17 * len(ids),
                          _equipment_ids=tuple(ids))
    env.action_contract = lambda: SharedExerciseTaskEnv.action_contract(env)
    env.observation_contract = lambda: SharedExerciseTaskEnv.observation_contract(env)

    def observe(*, body=None, positions=None, velocities=None, elapsed=.25):
        human = np.zeros(155) if body is None else body.copy()
        if body is None:
            human[154] = .125
        positions = positions if positions is not None else [[.5 + 2. * i, 0., 1.] for i in range(len(ids))]
        velocities = velocities if velocities is not None else [[0., 0., 0.] for _ in ids]
        objects = {name: SimpleNamespace(position=np.asarray(origin) + scale * np.asarray(position),
                                         quaternion=[1., 0., 0., 0.], velocity=scale * np.asarray(velocity),
                                         angular_velocity=[0., 0., 0.])
                   for name, position, velocity in zip(ids, positions, velocities)}
        env._accumulator = SimpleNamespace(scale=scale, latest=objects, initial=deepcopy(objects),
                                           contacts=[], elapsed=10. * elapsed, duration=10.)
        env.observation_space = SimpleNamespace(contains=lambda value: value.shape == (env.observation_size,)
                                                and np.isfinite(value).all() and np.all(abs(value) <= 1e6))
        return SharedExerciseTaskEnv._observation(env, human)

    return env, observe


def weights(group=None, first=.25, second=0.):
    value = np.zeros((8, 2))
    if group is not None:
        value[group] = [first, second]
    return value


def moving_body():
    value = np.zeros(155)
    value[:19] = value[44:63] = .125
    value[22:41] = value[66:85] = 2.
    value[154] = .125
    return value


@pytest.mark.parametrize("count,size", [(1, 173), (2, 190), (12, 360)])
def test_real_observation_builder_and_contract_include_final_elapsed_not_a_contact_row(count, size):
    env, observe = fixture(count=count)
    observation = observe(elapsed=.375)
    prepared = actor.prepare_equipment_feedback(env, weights())
    _, report = prepared.sample(np.zeros(22), observation)
    assert len(observation) == report["observation_size"] == size
    assert observation[-2] == 0. and report["elapsed_fraction"] == .375
    assert report["reference_phase"] == .125 and report["contact_requirement_count"] == 0
    assert report["nearest"]["id"] == "object-0"
    assert report["pairing"] == "caller_responsibility_not_authenticated"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_zero_coefficients_are_byte_exact_supplied_v5_action(dtype):
    env, observe = fixture()
    features = np.random.default_rng(42).normal(0., .8, (22, 3)).astype(dtype)
    env.calls = 0

    def read():
        env.calls += 1
        return features.copy()

    env.feature_matrix = read
    base, *_ = v5.action_sample(np.full((5, 2), .17), v5.joint_groups(env.action_names,
                               profile=v5.SPLIT_POLICY_PROFILE), env)
    original = base.tobytes()
    action, report = actor.prepare_equipment_feedback(env, weights()).sample(base, observe(body=moving_body()))
    assert env.calls == 1 and action.dtype == base.dtype
    assert action.tobytes() == base.tobytes() == original and report["zero_coefficients"]
    action[:] = 0.
    assert base.tobytes() == original and report["changed_channels"] == []


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_zero_residual_preserves_signed_zero_and_detaches_readonly_inputs(dtype):
    env, observe = fixture()
    base = np.zeros(22, dtype=dtype); base[::2] = -0.
    observation = observe().astype(dtype)
    base.flags.writeable = observation.flags.writeable = False
    before = (base.tobytes(), observation.tobytes())
    action, _ = actor.prepare_equipment_feedback(env, weights()).sample(base, observation)
    assert action.tobytes() == before[0] and not np.shares_memory(action, base)
    action[:] = .1
    assert before == (base.tobytes(), observation.tobytes())


def test_nearby_motion_can_change_joints_at_zero_tracking_error_but_distant_objects_do_not():
    env, observe = fixture()
    body = moving_body()
    env.feature_matrix = lambda: np.zeros((22, 3))
    base, *_ = v5.action_sample(np.full((5, 2), .25), v5.joint_groups(env.action_names,
                               profile=v5.SPLIT_POLICY_PROFILE), env)
    assert not np.any(base)
    prepared = actor.prepare_equipment_feedback(env, weights(1, -.25))
    near, report = prepared.sample(base, observe(body=body))
    expected = {3, 4, 5, 11, 12, 13}
    assert set(report["changed_channels"]) == expected and np.all(near[list(expected)] < 0.)
    assert report["features"][3] == pytest.approx([1. / 3., 1. / 3.])
    far, distant = prepared.sample(base, observe(body=body, positions=[[2., 0., 1.], [3., 0., 1.]]))
    assert far.tobytes() == base.tobytes() and distant["nearest"]["proximity"] == 0.


@pytest.mark.parametrize("group", range(8))
def test_each_tied_group_changes_only_its_documented_channels(group):
    env, observe = fixture()
    prepared = actor.prepare_equipment_feedback(env, weights(group, .2, .1))
    action, report = prepared.sample(np.zeros(22), observe(body=moving_body(),
        positions=[[-.25, -.25, .75], [3., 0., 1.]]))
    channels = np.r_[v5.joint_groups(env.action_names, profile=v5.SPLIT_POLICY_PROFILE), 5, 6, 7]
    expected = np.flatnonzero(channels == group).tolist()
    assert report["changed_channels"] == expected
    assert np.all(action[expected] > 0.) and np.all(action[channels != group] == 0.)


def test_joint_mapping_uses_existing_names_not_a_new_positional_partition():
    names = (*reversed(SPATIAL_JOINT_NAMES), *ROOT_AXES)
    env, observe = fixture(names=names)
    action, _ = actor.prepare_equipment_feedback(env, weights(2)).sample(np.zeros(22), observe(body=moving_body()))
    assert [names[i] for i in np.flatnonzero(action)] == ["right_elbow", "left_elbow"]


@pytest.mark.parametrize("pelvis_velocity,equipment_velocity,closing", [(1., 0., .5), (-1., 0., 0.), (1., 1., 0.)])
def test_relative_velocity_closing_sign_and_root_away_direction(pelvis_velocity, equipment_velocity, closing):
    env, observe = fixture()
    body = moving_body(); body[41] = pelvis_velocity
    action, report = actor.prepare_equipment_feedback(env, weights(5, 0., .25)).sample(np.zeros(22),
        observe(body=body, velocities=[[equipment_velocity, 0., 0.], [0., 0., 0.]]))
    nearest = report["nearest"]
    assert nearest["away_world_axes"] == [-1., 0., 0.]
    assert nearest["normalized_closing_speed"] == closing
    assert action[19] == pytest.approx(-.25 * (2. / 3.) * closing)
    assert not np.any(action[:19]) and not np.any(action[20:])


def test_nearest_tie_uses_task_order_and_zero_distance_reports_unknown_direction():
    env, observe = fixture(ids=["z-object", "a-object"])
    prepared = actor.prepare_equipment_feedback(env, np.full((8, 2), .25))
    _, tied = prepared.sample(np.zeros(22), observe(positions=[[.5, 0., 1.], [-.5, 0., 1.]]))
    assert tied["nearest"]["id"] == "z-object"
    _, switched = prepared.sample(np.zeros(22), observe(positions=[[2., 0., 1.], [.25, 0., 1.]]))
    assert switched["nearest"]["id"] == "a-object"
    action, coincident = prepared.sample(np.zeros(22), observe(body=moving_body(), positions=[[0., 0., 1.], [.5, 0., 1.]]))
    assert coincident["nearest"]["away_direction_known"] is False
    assert coincident["nearest"]["proximity"] == 1. and not np.any(action[19:])
    assert np.all(action[:19] == .25)


@pytest.mark.parametrize("scale,origin", [(.5, (0., 0., 0.)), (1., (1., 2., .5)), (2., (2., 4., 1.))])
def test_model_unit_features_are_invariant_to_declared_world_scale_and_origin(scale, origin):
    outputs = []
    for s, o in [(1., (0., 0., 0.)), (scale, origin)]:
        env, observe = fixture(scale=s, origin=o)
        body = moving_body()
        body[19:22] = s * np.asarray([.125, .25, 0.])
        body[41:44] = s * np.asarray([.5, .25, -.125])
        outputs.append(actor.prepare_equipment_feedback(env, np.full((8, 2), -.125)).sample(
            np.zeros(22), observe(body=body, velocities=[[.125, 0., 0.], [0., 0., 0.]])))
    assert outputs[0][0].tobytes() == outputs[1][0].tobytes()
    assert outputs[0][1]["features"] == outputs[1][1]["features"]
    assert outputs[0][1]["nearest"] == outputs[1][1]["nearest"]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sign", [-1., 1.])
def test_action_and_residual_bounds_include_inward_representable_rounding(dtype, sign):
    env, observe = fixture()
    body = moving_body(); body[22:41] = body[66:85] = 1e6
    base = np.full(22, -sign * .1, dtype=dtype); base[1] = sign * .95
    action, report = actor.prepare_equipment_feedback(env, np.full((8, 2), sign * .25)).sample(
        base, observe(body=body, positions=[[0., 0., 1.], [.5, 0., 1.]]))
    assert action.dtype == base.dtype and np.all(abs(action) <= 1.)
    assert np.all(abs(action.astype(float) - base.astype(float)) <= .25)
    assert report["requested_residual"][:19] == [sign * .25] * 19
    assert action[1] == sign and action[19:].tobytes() == base[19:].tobytes()
    if dtype is np.float32:
        rounded_outside = dtype(float(base[0]) + sign * .25)
        assert abs(float(rounded_outside) - float(base[0])) > .25
        assert action[0] == np.nextafter(rounded_outside, base[0])


@pytest.mark.parametrize("bad", [np.zeros((7, 2)), np.zeros((8, 3)), np.zeros((8, 2), dtype=bool),
    [[True, 0.]] * 8, [[.251, 0.]] * 8, [[float("nan"), 0.]] * 8, [[float("inf"), 0.]] * 8,
    np.full((8, 2), .251), np.full((8, 2), np.nan)])
def test_invalid_coefficient_shape_type_and_values_reject_before_preparation(bad):
    with pytest.raises(ValueError, match="coefficients"):
        actor.prepare_equipment_feedback(None, bad)


@pytest.mark.parametrize("which,change", [
    ("base", "length"), ("base", "rank"), ("base", "dtype"), ("base", "nan"), ("base", "bound"),
    ("observation", "length"), ("observation", "rank"), ("observation", "dtype"),
    ("observation", "nan"), ("observation", "bound"),
])
def test_bounded_materialized_arrays_are_required_for_every_sample(which, change):
    env, observe = fixture(); prepared = actor.prepare_equipment_feedback(env, weights())
    base, observation = np.zeros(22), observe()
    value = base if which == "base" else observation
    if change == "length":
        value = value[:-1]
    elif change == "rank":
        value = value.reshape(1, -1)
    elif change == "dtype":
        value = value.astype(int)
    elif change == "nan":
        value[-2] = np.nan
    else:
        value[-2] = 2. if which == "base" else 1_000_001.
    with pytest.raises(ValueError, match="finite bounded"):
        prepared.sample(value if which == "base" else base, value if which == "observation" else observation)


@pytest.mark.parametrize("index,value", [(154, -1.), (154, 1.01), (-1, -1.), (-1, 1.), (-1, 2.)])
def test_observable_invalid_phase_or_finished_elapsed_fraction_rejects(index, value):
    env, observe = fixture(); observation = observe(); observation[index] = value
    with pytest.raises(ValueError, match="unfinished elapsed"):
        actor.prepare_equipment_feedback(env, weights()).sample(np.zeros(22), observation)


@pytest.mark.parametrize("change", ["task_profile", "scene_profile", "contacts", "root_mode", "object_force", "duplicate_id",
    "too_many_ids", "unknown_joint", "root_order", "action_channels", "observation_tail", "observation_size",
    "equipment_field", "boolean_count", "scale", "origin", "height", "axes", "limit"])
def test_unsupported_or_inconsistent_metadata_rejects_without_native_admission(change):
    env, _ = fixture()
    if change == "task_profile": env.task["profile"] = "shared-support-task-v1"
    elif change == "scene_profile": env.scene["profile"] = "shared-world-v5"
    elif change == "contacts": env.task["contact_requirements"] = [{"object_id": "object-0"}]
    elif change == "root_mode": env.task["control"]["root_assistance"] = "disabled"
    elif change == "object_force": env.task["control"]["object_forces"] = "allowed"
    elif change == "duplicate_id": env.task["equipment_ids"][1] = "object-0"
    elif change == "too_many_ids": env.task["equipment_ids"] = [str(i) for i in range(13)]
    elif change == "unknown_joint": env.action_names = ("unknown", *env.action_names[1:])
    elif change == "root_order": env.action_names = (*env.action_names[:19], *reversed(ROOT_AXES))
    elif change == "action_channels":
        original = env.action_contract(); original["object_force_channels"] = [22]
        env.action_contract = lambda: original
    elif change in ("observation_tail", "equipment_field", "boolean_count"):
        original = env.observation_contract()
        if change == "observation_tail": original["fields"][-1]["name"] = "unknown"
        elif change == "equipment_field": original["fields"][1]["fields"][2] = "local_velocity:3"
        else: original["fields"][-1]["count"] = True
        env.observation_contract = lambda: original
    elif change == "observation_size": env.observation_size += 6
    elif change == "scale": env.scene["coordinate_frame"]["scale_bridge"]["body_height_to_world_scale"] = 0.
    elif change == "origin": env.scene["coordinate_frame"]["scale_bridge"]["origin_world"][0] = np.nan
    elif change == "height": env.scene["human_spec"]["mobility"]["base_height"] = np.inf
    elif change == "axes": env.scene["coordinate_frame"]["axes"] = "camera_axes"
    else: env.scene["action_contract"]["physical_limits"][0] = 0.
    with pytest.raises(ValueError):
        actor.prepare_equipment_feedback(env, weights())


def test_prepared_metadata_weights_and_reports_are_detached_and_sample_reads_no_environment(monkeypatch):
    env, observe = fixture(); original_weights = np.full((8, 2), .1)
    observation = observe(body=moving_body())
    prepared = actor.prepare_equipment_feedback(env, original_weights)
    first_action, first_report = prepared.sample(np.zeros(22), observation)
    original_weights[:] = -.25
    env.scene["coordinate_frame"]["scale_bridge"]["origin_world"][0] = 5.
    env.task["equipment_ids"][0] = "changed"
    for name in ("action_contract", "observation_contract", "feature_matrix", "observation", "snapshot"):
        setattr(env, name, lambda: pytest.fail("The pure sample cannot query live state or metadata"))
    monkeypatch.setattr(v5, "runtime_hashes", lambda *_: pytest.fail("The pure actor cannot hash policy dependencies"))
    monkeypatch.setattr(v5, "validate_shared_exercise_feedback_policy", lambda *_: pytest.fail("Native admission forbidden"))
    for array in (prepared._weights, prepared._channel_weights, prepared._pelvis_origin):
        with pytest.raises(ValueError):
            array.flags.writeable = True
    with pytest.raises(FrozenInstanceError):
        prepared._scale = 2.
    second_action, second_report = prepared.sample(np.zeros(22), observation)
    assert first_action.tobytes() == second_action.tobytes() and first_report == second_report
    first_report["features"][0][0] = 123.
    assert prepared.sample(np.zeros(22), observation)[1] == second_report


def test_full_supported_input_and_report_stay_bounded_and_input_conversion_is_not_invoked():
    env, observe = fixture(ids=["x" * 157 + f"{i:03}" for i in range(12)])
    prepared = actor.prepare_equipment_feedback(env, np.full((8, 2), .25))
    observation = observe(body=moving_body())
    _, report = prepared.sample(np.full(22, -.123456789), observation)
    assert len(json.dumps(report, allow_nan=False, separators=(",", ":")).encode()) <= actor.MAX_RESULT_BYTES

    class Trap:
        def __array__(self, *_args, **_kwargs):
            pytest.fail("Unbounded caller conversion must not run")

    with pytest.raises(ValueError, match="finite bounded"):
        prepared.sample(Trap(), observation)
    with pytest.raises(ValueError, match="finite bounded"):
        prepared.sample(np.zeros(22), Trap())
