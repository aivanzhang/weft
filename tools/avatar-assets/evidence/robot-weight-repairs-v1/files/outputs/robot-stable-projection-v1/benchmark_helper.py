"""Maximum declared helper dimensions; analytic weights, not avatar-quality evidence."""

import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from avatar_assets.safety import write_json
from avatar_assets.stable_role_projection import project_stable_roles


output = Path(__file__).resolve().parent
count = 100000
own = np.arange(count) % 32
joints = np.column_stack((own, (own+1) % 32, np.zeros(count), np.zeros(count))).astype(np.int32)
weights = np.tile([.8, .2, 0, 0], (count, 1))
names = ["Bone"+str(index) for index in range(64)]
roles = [{"state_id": "analytic", "role_id": "role-"+str(index), "bones": [names[index]], "stable": index % 2 == 0}
         for index in range(32)]
started = time.monotonic()
result, report = project_stable_roles(joints, weights, names, roles,
                                    bone_parents={name: None for name in names}, surface_kind="rigid_hard_surface")
elapsed = time.monotonic()-started
assert report["changed_vertices"] == 50000
assert sum(role["lost_fringe_count"] for role in report["roles"]) == 50000
assert report["core_memberships_unchanged"]
assert np.array_equal(result[own % 2 == 1], weights[own % 2 == 1])
receipt = {"schema_version": 1, "status": "passed", "vertices": count, "bones": 64, "roles": 32,
           "elapsed_helper_seconds": elapsed, "changed_vertices": report["changed_vertices"],
           "recorded_fringe_losses": sum(role["lost_fringe_count"] for role in report["roles"]),
           "core_memberships_unchanged": True, "unselected_weights_unchanged": True,
           "numpy_version": np.__version__, "quality_claim": False,
           "scope": "One maximum-dimension analytic fixture; not sustained-service latency or visual acceptance."}
write_json(output/"benchmark-helper.json", receipt)
print(json.dumps(receipt, sort_keys=True))
