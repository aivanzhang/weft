"""Prepare compact task-authoring options in the existing source-loader worker."""

from copy import deepcopy

from weft.moving_equipment_compatibility_lab import LoadingCompatibleTaskLab
from weft.shared_world_exploration import LoadingExplorationLab


class _TaskOptions:
    def _load(self):
        from weft.portal_task_contract import task_options

        lab = super()._load()
        try:
            task = lab.session.task if lab.config.get("control_profile") == "moving_equipment_task" else None
            self._task_options = {"status": "ready", **task_options(lab.session.scene,
                parent_manifest_sha256=lab.config["manifest_sha256"], task=task)}
        except (ValueError, KeyError, TypeError) as error:
            # Optional authoring cannot disable an otherwise valid manual sandbox.
            self._task_options = {"status": "unsupported", "reason": str(error)[:500]}
        return lab

    def task_options(self):
        if self.loading:
            return {"status": "running", "message": "Checking the saved world before showing task settings"}
        self._ready()
        return deepcopy(self._task_options)


class LoadingTaskAuthoringLab(_TaskOptions, LoadingExplorationLab):
    pass


class LoadingTaskLearningLab(_TaskOptions, LoadingCompatibleTaskLab):
    pass
