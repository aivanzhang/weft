"""Advisory task checks outside sealed physics and recording implementations."""

from copy import deepcopy
from pathlib import Path
import time

from weft.moving_equipment_compatibility import assess_moving_equipment_compatibility
from weft.moving_equipment_task_lab import LoadingTaskLab, load_moving_equipment_task_lab
from weft.shared_world_packaging import read_package_file


def _report(admission):
    return assess_moving_equipment_compatibility(admission.scene_bytes, admission.task_bytes)


class LoadingCompatibleTaskLab(LoadingTaskLab):
    """Compute once in the existing loader worker; advisory failures never disable manual controls."""

    def _load(self):
        lab = super()._load()
        try:
            self._compatibility = _report(lab.session.admission)
        except Exception as error:
            self._compatibility = {"profile": "moving-equipment-compatibility-v1", "status": "unassessed",
                "reason": str(error)[:300], "blocks_manual_exploration": False}
        return lab

    def summary(self):
        result = super().summary()
        if result.get("status") == "ready":
            result["task_compatibility"] = deepcopy(self._compatibility)
        return result


def inspect_saved_compatibility(task_lab, *, max_seconds=30):
    """Admit the saved task once; reporting performs no inference, optimization or simulation step."""
    import math
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 30:
        raise ValueError("Task preflight requires a finite positive bound up to 30 seconds")
    deadline = time.monotonic() + max_seconds
    source = Path(task_lab)
    from weft.shared_world_action_replay import _json
    config_raw = read_package_file(source, "lab-config.json", max_bytes=64 << 10)
    admission = load_moving_equipment_task_lab(source / "run", _json(config_raw),
        max_seconds=max_seconds, deadline=deadline)
    report = _report(admission)
    if time.monotonic() >= deadline:
        raise TimeoutError("Task preflight exceeded its aggregate deadline")
    report["source_admission"] = {"manifest_sha256": admission.manifest_sha256,
        "parent_manifest_sha256": admission.parent_manifest_sha256}
    return report
