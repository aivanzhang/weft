"""Preloaded local appearances; no avatar setting enters the physical environment."""

from copy import deepcopy
import hashlib
import math
from pathlib import Path
from types import MappingProxyType

from weft.avatar_assets import load_imported_avatar


MAX_AVATARS = 4
# Eight assets at up to ~17 MiB of originals, render JSON and textures each.
MAX_CATALOG_BYTES = 160 * 1024 * 1024
DEFAULT_APPEARANCE = {"asset": None, "overlay": True, "show_muscles": True,
                      "camera": {"yaw": 0.5, "zoom": 1.0, "follow": False}}
VIEWER_FILES = ("avatar_retarget.js", "avatar_worker.js", "avatar_viewer.js")


def validate_appearance(value, available):
    if not isinstance(value, dict) or set(value) != set(DEFAULT_APPEARANCE):
        raise ValueError("Appearance requires asset, overlay, show_muscles and camera")
    asset = value["asset"]
    if asset is not None and (not isinstance(asset, str) or asset not in available):
        raise ValueError("Avatar is missing from this lab; choose the original body explicitly")
    if any(type(value[key]) is not bool for key in ("overlay", "show_muscles")):
        raise ValueError("Avatar overlays must be boolean")
    camera = value["camera"]
    if not isinstance(camera, dict) or set(camera) != {"yaw", "zoom", "follow"}:
        raise ValueError("Avatar camera requires yaw, zoom and follow")
    for name, low, high in (("yaw", -math.pi, math.pi), ("zoom", 1, 3)):
        number = camera[name]
        if type(number) not in (int, float) or not math.isfinite(number) or not low <= number <= high:
            raise ValueError("Avatar camera " + name + " is outside its supported range")
    if type(camera["follow"]) is not bool:
        raise ValueError("Avatar camera follow must be boolean")
    return deepcopy(value)


class AvatarCatalog:
    def __init__(self, directories=()):
        directories = tuple(directories)
        if len(directories) > MAX_AVATARS:
            raise ValueError("Load at most eight local avatars per lab")
        packages, entries, total = {}, [], 0
        names = set()
        for directory in directories:
            package = load_imported_avatar(directory)
            key = package.source_manifest_sha256
            identity = package.manifest["asset_id"]
            if key in packages or identity in names:
                raise ValueError("Avatar identities must be unique within a lab")
            total += sum(map(len, package.originals.values())) + len(package.render_json)
            total += sum(map(len, package.texture_files.values()))
            if total > MAX_CATALOG_BYTES:
                raise ValueError("Avatar catalog exceeds its 128 MiB byte bound")
            packages[key] = package
            names.add(identity)
            entries.append({"key": key, "asset_id": identity,
                            "display_name": package.manifest["display_name"],
                            "render_sha256": package.render_sha256,
                            "textures": len(package.render["textures"]),
                            "counts": deepcopy(package.render["counts"])})
        self.packages = MappingProxyType(packages)
        self.entries = tuple(entries)
        self.total_bytes = total
        self.viewer_files = {}
        if packages:
            # Freeze replay code before serving; later edits cannot change a saved trial.
            for name in VIEWER_FILES:
                path = Path(__file__).parent / "assets" / name
                with path.open("rb") as stream:
                    raw = stream.read(512 * 1024 + 1)
                if not 0 < len(raw) <= 512 * 1024:
                    raise ValueError("Avatar viewer code exceeds its byte bound")
                self.viewer_files[name] = raw

    def summary(self, settings):
        return {"profile": "weft-lab-appearance-v1", "choices": deepcopy(self.entries),
                "settings": deepcopy(settings), "total_bytes": self.total_bytes,
                "scope": "Appearance only; collisions, actions, rewards and task checks are unchanged"}

    def resource(self, key, name):
        package = self.packages.get(key)
        if package is None:
            raise ValueError("Avatar is not loaded in this lab")
        if name == "render.json":
            return package.render_json, "application/json", package.render_sha256
        if name.startswith("texture-") and name[8:].isdigit():
            index = int(name[8:])
            if index < len(package.render["textures"]):
                texture = package.render["textures"][index]
                raw = package.texture_files[texture["path"]]
                return raw, texture["mime_type"], hashlib.sha256(raw).hexdigest()
        raise ValueError("Avatar resource does not exist")
