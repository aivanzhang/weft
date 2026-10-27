import numpy as np

from weft.evaluate import MIN_RELATIVE_IMPROVEMENT, physics_checks, replay_metrics
from weft.io import canonical_hash


def propose(generation_rows, world):
    origin = generation_rows[0]["time_s"]
    t = np.array([r["time_s"] - origin for r in generation_rows])
    dx = np.array([r["x_px"] / world.width_px - world.x0 for r in generation_rows])
    direction = 1 if world.v0 >= 0 else -1
    velocity = float(np.dot(t, dx) / np.dot(t, t))
    coefficients = np.linalg.lstsq(np.column_stack((t, -0.5 * direction * t ** 2)), dx, rcond=None)[0]
    return [
        {"method": "least_squares_constant_velocity", "patch": {"v0": velocity, "drag": 0.0}},
        {"method": "least_squares_dissipative_motion",
         "patch": {"v0": float(coefficients[0]), "drag": float(coefficients[1])}},
    ]


def repair(world, generation_rows, validation_rows, extra_proposals=()):
    origin = generation_rows[0]["time_s"]
    current = world
    current_score = replay_metrics(current, validation_rows, origin)
    ledger = []
    candidates = propose(generation_rows, world) + list(extra_proposals)
    if len(candidates) > 3:
        raise ValueError("At most three repair proposals are permitted")
    for index, proposal in enumerate(candidates):
        record = {"attempt": index + 1, "proposal": proposal,
                  "base_hash": canonical_hash(current.to_dict()),
                  "generation_evidence_ids": [r["id"] for r in generation_rows],
                  "provider": "local_numeric" if index < 2 else "external_declared",
                  "cost_usd": 0.0 if index < 2 else None,
                  "before": current_score}
        try:
            candidate = current.patch(proposal["patch"])
            physical = physics_checks(candidate)
            score = replay_metrics(candidate, validation_rows, origin)
            improvement = (current_score["rmse_image_width"] - score["rmse_image_width"]) / max(current_score["rmse_image_width"], 1e-12)
            accepted = physical["passes"] and score["passes"] and improvement >= MIN_RELATIVE_IMPROVEMENT
            record.update(candidate_hash=canonical_hash(candidate.to_dict()), physics=physical,
                          after=score, relative_improvement=improvement, accepted=accepted,
                          reason="independent gates passed" if accepted else "insufficient improvement or failed gate")
            if accepted:
                current, current_score = candidate, score
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            record.update(accepted=False, reason=f"invalid proposal: {error}")
        ledger.append(record)
    return current, ledger
