"""An explicit numerical-rounding prior, never a replacement source calibration."""

from dataclasses import dataclass
import math
from types import MappingProxyType

import numpy as np

from weft.camera_pose import _array
from weft.io import canonical_hash


PROFILE = "bounded-polar-rounding-prior-v1"
CONTRACT = MappingProxyType({
    "profile": PROFILE,
    "method": "closest_proper_SO3_polar_factor",
    "assumption_status": "assumed",
    "assumption": "Finite numerical rounding explains the supplied near-rotation error; not verified calibration",
    "upstream_maximum_orthogonality_residual": 1e-5,
    "upstream_maximum_determinant_error": 1e-5,
    "downstream_maximum_orthogonality_residual": 1e-8,
    "downstream_maximum_determinant_error": 1e-8,
    "maximum_correction_frobenius_norm": 2e-5,
    "scope": "Dimensionless matrix prior only; no intrinsics, translation, scale, clock, body, ground or contact correction",
})


def _require(value, message):
    if not value:
        raise ValueError("Camera rotation prior: " + message)


def _metrics(rotation):
    return (float(np.max(np.abs(rotation @ rotation.T - np.eye(3)))),
            float(np.linalg.det(rotation)))


@dataclass(frozen=True)
class CameraRotationPrior:
    raw_rotation: tuple
    corrected_rotation: tuple
    raw_rotation_sha256: str
    corrected_rotation_sha256: str
    raw_orthogonality_residual: float
    raw_determinant: float
    corrected_orthogonality_residual: float
    corrected_determinant: float
    correction_frobenius_norm: float
    correction_max_abs_entry: float
    id: str

    def describe(self):
        return {"schema_version": 1, "profile": PROFILE, "id": self.id,
                "assumption_status": "assumed", "physical_validation": "unverified",
                "raw_rotation": [list(row) for row in self.raw_rotation],
                "corrected_rotation": [list(row) for row in self.corrected_rotation],
                "raw_rotation_sha256": self.raw_rotation_sha256,
                "corrected_rotation_sha256": self.corrected_rotation_sha256,
                "raw_orthogonality_residual": self.raw_orthogonality_residual,
                "raw_determinant": self.raw_determinant,
                "corrected_orthogonality_residual": self.corrected_orthogonality_residual,
                "corrected_determinant": self.corrected_determinant,
                "correction_frobenius_norm": self.correction_frobenius_norm,
                "correction_max_abs_entry": self.correction_max_abs_entry}


def prepare_camera_rotation_prior(rotation):
    """Return a bounded closest-rotation hypothesis while retaining exact supplied numbers."""
    _require(type(rotation) in (list, tuple) and len(rotation) == 3
             and all(type(row) in (list, tuple) and len(row) == 3 for row in rotation)
             and all(type(value) in (int, float) and -1.00001 <= value <= 1.00001
                     for row in rotation for value in row), "finite bounded 3x3 source rotation required")
    raw_values = tuple(tuple(row) for row in rotation)
    raw = _array(raw_values, (3, 3), "rotation prior", 1.00001)
    residual, determinant = _metrics(raw)
    _require(determinant > 0. and residual <= CONTRACT["upstream_maximum_orthogonality_residual"]
             and abs(determinant - 1.) <= CONTRACT["upstream_maximum_determinant_error"],
             "raw rotation exceeds the unchanged upstream proper-rotation envelope")
    try:
        left, singular, right = np.linalg.svd(raw, full_matrices=False)
    except np.linalg.LinAlgError as error:
        raise ValueError("Camera rotation prior: polar decomposition unavailable") from error
    _require(np.isfinite(singular).all() and float(np.min(singular)) > 0., "singular source rotation")
    corrected = left @ right
    corrected_residual, corrected_determinant = _metrics(corrected)
    delta = corrected - raw
    norm, maximum = float(np.linalg.norm(delta, ord="fro")), float(np.max(np.abs(delta)))
    # The upstream error bound implies a correction below 1.501e-5; 2e-5 leaves numerical slack.
    _require(math.isfinite(norm) and norm <= CONTRACT["maximum_correction_frobenius_norm"]
             and corrected_determinant > 0.
             and corrected_residual <= CONTRACT["downstream_maximum_orthogonality_residual"]
             and abs(corrected_determinant - 1.) <= CONTRACT["downstream_maximum_determinant_error"],
             "polar factor fails the bounded correction or unchanged downstream envelope")
    corrected_values = tuple(tuple(float(value) for value in row) for row in corrected)
    raw_sha, corrected_sha = canonical_hash(raw_values), canonical_hash(corrected_values)
    identity = canonical_hash({"contract": dict(CONTRACT), "raw_rotation_sha256": raw_sha,
                               "corrected_rotation_sha256": corrected_sha})
    return CameraRotationPrior(raw_values, corrected_values, raw_sha, corrected_sha, residual, determinant,
                               corrected_residual, corrected_determinant, norm, maximum, identity)
