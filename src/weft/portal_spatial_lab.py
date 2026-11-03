"""Background current-runtime loading for source-linked spatial portal imports."""

from copy import deepcopy
import time

from weft.portal_spatial_import import inspect_spatial_import
from weft.shared_world_exploration import LoadingExplorationLab
from weft.shared_world_packaging import read_package_file, read_package_manifest


class LoadingSpatialPortalLab(LoadingExplorationLab):
    def _load(self):
        from weft.lab import Lab
        deadline = time.monotonic() + 30
        config = read_package_file(self.root, "lab-config.json", max_bytes=64 << 10)
        raw, files = read_package_manifest(self.root / "run")
        admitted = inspect_spatial_import(self.root, config, raw, files, deadline=deadline)
        lab = Lab(self.root)
        try:
            lab.load_session()
            summary = lab.summary()
            if (summary.get("profile") != "spatial-rigid" or summary.get("scene") != admitted["scene"]
                    or summary.get("learned_policy_available") is not admitted["learned_policy_available"]
                    or getattr(lab.session, "policy_sha256", None) != files.get("policy.json")):
                raise ValueError("Current spatial loader differs from the admitted world or policy")
            if (read_package_file(self.root, "lab-config.json", max_bytes=64 << 10) != config
                    or read_package_file(self.root / "run", "manifest.json", max_bytes=1 << 20) != raw):
                raise ValueError("Spatial lab seals changed during current-runtime loading")
            if time.monotonic() >= deadline:
                raise TimeoutError("Spatial portal loading exceeded its 30-second bound")
            self._source_scope = deepcopy(admitted["source_scope"])
            self._runtime_compatibility = deepcopy(admitted["runtime_compatibility"])
            return lab
        except BaseException:
            lab.close()
            raise

    def summary(self):
        if not self._future.done():
            return {"status": "running", "profile": "spatial-rigid", "control_profile": "spatial_rigid",
                "recording": False, "events": [{"stage": "source_validation", "message": "Checking the saved 3D world and its original source"}]}
        try:
            result = self._ready().summary()
            result.update(control_profile="spatial_rigid", spatial_import_scope=deepcopy(self._source_scope),
                          spatial_import_runtime=deepcopy(self._runtime_compatibility))
            return result
        except Exception as error:
            return {"status": "failed", "profile": "spatial-rigid", "control_profile": "spatial_rigid",
                    "recording": False, "error": {"message": str(error)[:1000]}, "events": []}
