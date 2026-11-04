"""Pre-import the predefined avatars into the hosted asset set.

Avatar retargeting runs in the browser, but the normalizer that turns a delivery GLB into
a render payload is Python. Doing that per request would put geometry work on the single
container slot for an asset that never changes, so it happens once at build time and the
result ships as static files. Keys and the index are the lab's own catalog entries, and
the layout matches its /api/avatar/<key>/render.json contract, so the existing client
fetches hosted avatars with nothing but a different base path.
"""
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "website/avatars"
# AI-generated deliveries served beside the authored ones. Each carries its own
# rights record ("internal development" for the Studio panda); they ship because the
# owner decided they may, and nothing here asserts broader redistribution permission.
GENERATED = {"studio-panda-plush": "weft-studio-panda-plush-v1"}
sys.path.insert(0, str(ROOT / "src"))


def main():
    from weft.avatar_lab import AvatarCatalog
    from weft.portal_avatar import import_avatar_zip, prepare_builtin_avatars

    with tempfile.TemporaryDirectory(prefix="weft-avatars-") as raw_temporary:
        # The importer refuses a symlinked parent, and macOS hands out /var/folders/...
        temporary = Path(raw_temporary).resolve()
        staged = prepare_builtin_avatars(temporary / "builtins", project=ROOT)
        # The two authored deliveries need prepare_builtin_avatars' left/right correction
        # because they were modelled on the opposite lateral convention. A generated
        # delivery already resolves roles in Weft's convention, so it imports unflipped;
        # sending it through the correction fails the lateral check outright.
        for index, (name, directory) in enumerate(sorted(GENERATED.items())):
            source = ROOT / "tools/avatar-assets/outputs" / directory
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as pack:
                for member in ("model.glb", "mapping.json"):
                    pack.writestr(member, (source / member).read_bytes())
            receipt = import_avatar_zip(buffer.getvalue(), temporary / f"generated-{index}")
            staged[name] = receipt["asset"]
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
