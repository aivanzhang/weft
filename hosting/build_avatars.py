"""Pre-import the predefined avatars into the hosted asset set.

Avatar retargeting runs in the browser, but the normalizer that turns a delivery GLB into
a render payload is Python. Doing that per request would put geometry work on the single
container slot for an asset that never changes, so it happens once at build time and the
result ships as static files. Keys and the index are the lab's own catalog entries, and
the layout matches its /api/avatar/<key>/render.json contract, so the existing client
fetches hosted avatars with nothing but a different base path.
"""
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "website/avatars"
sys.path.insert(0, str(ROOT / "src"))


def main():
    from weft.avatar_lab import AvatarCatalog
    from weft.portal_avatar import prepare_builtin_avatars

    with tempfile.TemporaryDirectory(prefix="weft-avatars-") as temporary:
        # The importer refuses a symlinked parent, and macOS hands out /var/folders/...
        staged = prepare_builtin_avatars(Path(temporary).resolve() / "builtins", project=ROOT)
        catalog = AvatarCatalog([staged[name]["directory"] for name in sorted(staged)])
        builtin = {result["source_manifest_sha256"]: name for name, result in staged.items()}
        if OUT.exists():
            shutil.rmtree(OUT)
        OUT.mkdir(parents=True)
        entries = []
        for entry in catalog.entries:
            package = catalog.packages[entry["key"]]
            directory = OUT / entry["key"]
            directory.mkdir()
            (directory / "render.json").write_bytes(package.render_json)
            mime_types = []
            for index, texture in enumerate(package.render["textures"]):
                raw = package.texture_files[texture["path"]]
                # The client requests texture-<i> with no extension and trusts the content
                # type the response carries, so the mime type travels in the index.
                (directory / f"texture-{index}").write_bytes(raw)
                mime_types.append(texture["mime_type"])
            entries.append({**entry, "builtin": builtin[entry["key"]],
                            "render_bytes": len(package.render_json),
                            "texture_mime_types": mime_types})
        index = {"schema_version": 1, "profile": "weft-hosted-avatars-v1",
                 "scope": ("Predefined appearances only. Retargeting runs in the browser and "
                           "never drives physics; each rig is authored, not recovered."),
                 "avatars": entries}
        (OUT / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    files = sorted(path.relative_to(ROOT).as_posix() for path in OUT.rglob("*") if path.is_file())
    print(json.dumps({"avatars": [entry["builtin"] for entry in entries], "files": files,
                      "bytes": sum((ROOT / name).stat().st_size for name in files)}, indent=1))


if __name__ == "__main__":
    main()
