"""Analytic transforms exercise calibration math, not generated-avatar visual acceptance."""
from copy import deepcopy
import math
import unittest

import numpy as np

from avatar_assets.three_d import _local_matrix
from avatar_assets.weft_metadata import FRAMES, _alignment_context, matrix_quaternion


def matrix(translation=(0, 0, 0), rotation=(0, 0, 0, 1)):
    return np.asarray(_local_matrix({"translation": list(map(float, translation)), "rotation": list(map(float, rotation))})).reshape(4, 4, order="F")


def analytic_capture():
    q = [0, 0, math.sin(.2), math.cos(.2)]
    body = matrix(rotation=q)[:3, :3]
    frames = {}
    for index, name in enumerate(sorted(FRAMES)):
        frame = matrix([.3*index, -.2, 1.1], q)
        frames[name] = {"translation": frame[:3, 3].tolist(), "rotation_xyzw": q,
                        "matrix_column_major": frame.flatten(order="F").tolist()}
    names = [f"analytic_{index}" for index in range(19)]
    return {"schema_version": 1, "status": "weft_runtime_frame_capture", "scene_sha256": "a"*64,
            "target_body_rotation3x3": body.tolist(), "joint_names": names,
            "calibration": {"kind": "zero_angle_root_zero_not_lab_reset", "angles": dict.fromkeys(names, 0),
                            "root_translation": [0, 0, 0], "frames": frames},
            "parity": {"status": "passed", "max_abs_matrix_error": 1e-14}}


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.capture = analytic_capture()
        self.design = {"alignment": {"scale": .8,
            "source_to_target_body_xyzw": [0, math.sqrt(.5), math.sqrt(.5), 0],
            "source_anchor_node": "pelvis", "target_anchor_body": "torso",
            "rationale": "Analytic fixture only; no model or anatomy claim"}}
        self.names = {"root": 0, "pelvis": 1}
        self.world = [np.eye(4), matrix([0, .9, 0])]

    def context(self):
        return _alignment_context(self.capture, self.design, self.names, self.world)

    def test_nonidentity_native_frames_preserve_anchored_bind_shape(self):
        captured, frames, scale, orient, basis, alignment = self.context()
        self.assertTrue(captured)
        np.testing.assert_allclose((alignment @ self.world[1])[:3, 3], frames["torso"][:3, 3], atol=1e-12)
        for frame in frames.values():
            for world in self.world:
                offset = np.linalg.inv(frame) @ alignment @ world @ np.linalg.inv(basis)
                rebuilt = frame @ matrix(offset[:3, 3], matrix_quaternion(offset[:3, :3])) @ basis
                np.testing.assert_allclose(rebuilt, alignment @ world, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(basis[:3, :3], axis=0), [.8]*3)

    def test_facing_and_side_are_explicit_not_inferred_from_labels(self):
        self.capture["target_body_rotation3x3"] = np.eye(3).tolist()
        _, _, _, _, basis, _ = self.context()
        np.testing.assert_allclose(basis[:3, :3] @ [0, 1, 0], [0, 0, .8], atol=1e-12)
        np.testing.assert_allclose(basis[:3, :3] @ [0, 0, 1], [0, .8, 0], atol=1e-12)
        np.testing.assert_allclose(basis[:3, :3] @ [1, 0, 0], [-.8, 0, 0], atol=1e-12)

    def test_legacy_proposal_is_not_relabelled_runtime_capture(self):
        self.capture = {"status": "authored_proposal_not_weft_measurement", "scale": 1,
                        "orientation_xyzw": [math.sqrt(.5), 0, 0, math.sqrt(.5)],
                        "frames": self.capture["calibration"]["frames"]}
        captured, _, _, _, basis, alignment = self.context()
        self.assertFalse(captured)
        np.testing.assert_array_equal(basis, alignment)

    def test_bad_capture_cannot_produce_calibrated_metadata(self):
        mutations = [
            lambda c: c.update(status="unverified"),
            lambda c: c.update(scene_sha256="not-a-hash"),
            lambda c: c["calibration"].update(kind="lab-reset"),
            lambda c: c["calibration"]["angles"].update(analytic_0=.1),
            lambda c: c["calibration"].update(root_translation=[False, 0, 0]),
            lambda c: c["parity"].update(status="failed"),
            lambda c: c["parity"].update(max_abs_matrix_error=float("nan")),
            lambda c: c["parity"].update(max_abs_matrix_error=.01),
            lambda c: c["calibration"]["frames"].pop("torso"),
            lambda c: c["calibration"]["frames"]["torso"].update(translation=[101, 0, 0]),
            lambda c: c["calibration"]["frames"]["torso"].update(rotation_xyzw=[0, 0, 0, 2]),
            lambda c: c["calibration"]["frames"]["torso"]["matrix_column_major"].__setitem__(12, 77),
            lambda c: c.update(target_body_rotation3x3=[[2,0,0],[0,1,0],[0,0,1]]),
        ]
        original = deepcopy(self.capture)
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                self.capture = deepcopy(original)
                mutate(self.capture)
                with self.assertRaises(ValueError):
                    self.context()

    def test_unsafe_or_implicit_visual_design_rejected(self):
        original = deepcopy(self.design)
        for changes in ({"scale": True}, {"scale": float("inf")}, {"scale": 0},
                        {"source_to_target_body_xyzw": [0,0,0,0]}, {"source_anchor_node": "missing"},
                        {"target_anchor_body": "head"}, {"rationale": ""}, {"extra": 1}):
            with self.subTest(changes=changes):
                self.design = deepcopy(original)
                self.design["alignment"].update(changes)
                with self.assertRaises(ValueError):
                    self.context()


if __name__ == "__main__":
    unittest.main()
