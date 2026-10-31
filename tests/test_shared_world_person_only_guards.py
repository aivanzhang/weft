"""Unsupported coupled-policy routes must fail before a person-only simulation starts."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from test_shared_world_human_only import person_scene
from weft.io import canonical_hash, digest, write_json
from weft import shared_world_lab as lab
from weft import shared_world_learning as learning


@pytest.fixture(autouse=True)
def no_dynamics(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Unsupported generic route must not construct its runtime or advance dynamics")

    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    monkeypatch.setattr(learning, "SharedWorldEnv", forbidden)
    monkeypatch.setattr(lab, "_probe_shared_interventions", forbidden)


def policy(scene):
    return {"schema_version": 1, "profile": learning.POLICY_PROFILE, "algorithm": learning.ALGORITHM,
            "weights": [0.] * learning.WEIGHT_DIM, "scene_sha256": canonical_hash(scene),
            "action_names": scene["action_contract"]["names"][:]}


def test_generic_policy_validation_rejects_person_only_scene():
    scene = person_scene()
    document = policy(scene)
    with pytest.raises(ValueError, match="shared-exercise residual policy"):
        learning.validate_shared_policy(document, scene)
    assert learning.validate_shared_policy(document) == document


def test_generic_actor_rejects_person_only_before_reading_features():
    scene = person_scene()
    env = SimpleNamespace(scene=scene)
    with pytest.raises(ValueError, match="shared-exercise residual policy"):
        learning.policy_action(policy(scene), env)


def test_generic_lab_rejects_person_only_before_intervention_probe(tmp_path):
    source, out = tmp_path / "source", tmp_path / "lab"
    source.mkdir()
    write_json(source / "scene.json", person_scene())
    write_json(source / "manifest.json", {"status": "ready", "files": {"scene.json": digest(source / "scene.json")}})
    with pytest.raises(ValueError, match="shared-exercise-lab"):
        lab.prepare_shared_world_lab(source, out)
    assert not out.exists()


def test_existing_generic_trainer_rejects_person_only_before_output(tmp_path):
    out = tmp_path / "training"
    with pytest.raises(ValueError, match="requires a shared-world-v1 scene"):
        learning.train_shared_world_policy(person_scene(), out)
    assert not out.exists()


def test_legacy_coupled_validator_and_zero_weight_actor_remain_unchanged():
    scene = lab.authored_shared_world_scene()
    document = policy(scene)
    assert learning.validate_shared_policy(document, scene) == document
    env = SimpleNamespace(scene=scene, action_names=scene["action_contract"]["names"],
                          feature_matrix=lambda: np.zeros((22, 3)), group_indices=np.zeros(22, dtype=int))
    assert learning.policy_action(document, env).tobytes() == np.zeros(len(env.action_names)).tobytes()


@pytest.mark.parametrize("field,value", [("scene_sha256", "f" * 64), ("action_names", ["other"]), ("weights", [0.])])
def test_legacy_policy_identity_and_weight_guards_remain(field, value):
    scene = lab.authored_shared_world_scene()
    document = policy(scene)
    document[field] = value
    with pytest.raises(ValueError):
        learning.validate_shared_policy(document, scene)
