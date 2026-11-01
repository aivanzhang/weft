"""Local authoring defaults; remote generation requires explicit Meshy commands."""

import argparse
import json
import sys
from pathlib import Path

from avatar_assets.safety import artifacts, contained, local_file, new_directory, read_json, require, sha256, write_json


def verify_pack(folder):
    root = Path(folder).resolve()
    manifest = read_json(root / "manifest.json")
    entries = manifest.get("artifacts", [])
    require(entries, "No artifact inventory")
    seen = set()
    for entry in entries:
        require(entry["path"] not in seen, "Duplicate artifact entry")
        seen.add(entry["path"])
        path = local_file(contained(root, entry["path"]), 512 * 1024 * 1024)
        require(path.stat().st_size == entry["bytes"] and sha256(path) == entry["sha256"], f"Artifact changed: {entry['path']}")
    require({x["path"] for x in artifacts(root)} == seen, "Unrecorded or missing artifacts")
    result = {"schema_version": 1, "artifact_hashes_verified": len(entries), "status": manifest.get("status", "inspected")}
    if (root / "model.glb").exists():
        if manifest.get("kind") == "weft_authored_model_and_mapping":
            result["rig_verification"] = "Hashes only; use the frozen Weft importer for this consumer bundle"
        else:
            from avatar_assets.three_d import validate_glb, validate_render_attributes
            result["rig"] = validate_glb(root / "model.glb", read_json(root / "rig-contract.json"))
            if manifest.get("scene_schema_version") == 2:
                result["render_attributes"] = validate_render_attributes(root / "model.glb")
    return result


def main():
    parser = argparse.ArgumentParser(description="Local avatar assets: editable 3D, 2D media, provenance and verification")
    sub = parser.add_subparsers(dest="command", required=True)
    from avatar_assets.meshy import add_commands
    add_commands(sub)
    from avatar_assets.pipeline import add_commands as pipeline_commands
    pipeline_commands(sub)
    inspect = sub.add_parser("inspect", help="Safely inspect a flat media ZIP without trusting its labels")
    inspect.add_argument("archive")
    inspect.add_argument("--out", required=True)
    inspect.add_argument("--sha256")
    build = sub.add_parser("build3d", help="Build an authored scene JSON into a real rigged GLB and Blender source")
    build.add_argument("scene")
    build.add_argument("--out", required=True)
    build.add_argument("--blender")
    build.add_argument("--resolution", type=int, default=384)
    build.add_argument("--samples", type=int, default=8)
    build.add_argument("--timeout", type=int, default=300)
    build.add_argument("--portrait-only", action="store_true", help="Skip pose contact renders during bounded look development; export every clip")
    export = sub.add_parser("export2d", help="Render full skeletal clips, a portrait and static fallbacks")
    export.add_argument("folder")
    export.add_argument("--out", required=True)
    export.add_argument("--preset", choices=["muse", "preview"], default="muse")
    export.add_argument("--blender")
    export.add_argument("--denoise-portrait", action="store_true", help="Filter the rendered portrait in an isolated CPU worker; retain the raw PNG")
    export.add_argument('--resume-frames', help='Reuse hash-bound completed generated frames after one wall-time failure; new output required')
    for name in ("size", "portrait", "seconds", "samples"):
        export.add_argument("--" + name, type=int)
    verify = sub.add_parser("verify", help="Recheck pack hashes and independently decode the rig")
    verify.add_argument("folder")
    gallery = sub.add_parser("gallery", help="Serve the read-only local browser library; never opens a browser")
    gallery.add_argument("--port", type=int, default=8765)
    gallery.add_argument("--catalog", help="Workspace-local gallery configuration JSON")
    reload = sub.add_parser("reload3d", help="Independently import and animate the GLB in native Blender")
    reload.add_argument("folder")
    reload.add_argument("--out", required=True)
    reload.add_argument("--blender")
    review = sub.add_parser("review", help="Preserve a hash-bound visual review, without modifying its pack")
    review.add_argument("folder")
    review.add_argument("review_json")
    review.add_argument("--out", required=True)
    skin = sub.add_parser("reskin", help="Produce a new scene with material-only changes")
    skin.add_argument("scene")
    skin.add_argument("materials_json")
    skin.add_argument("--out", required=True)
    skin.add_argument("--subject", help="New skin identity description")
    skin.add_argument("--prompt", help="Exact material-change request; preserved with the source prompt")
    weft = sub.add_parser("weft-metadata", help="Write explicit Weft node mappings against a disclosed neutral proposal")
    weft.add_argument("folder")
    weft.add_argument("design")
    weft.add_argument("--neutral", required=True)
    weft.add_argument("--out", required=True)
    three = sub.add_parser("verify3d", help="Check an existing GLB against an explicit control contract")
    three.add_argument("glb")
    three.add_argument("contract")
    unsupported = sub.add_parser("unsupported3d", help="Write an explicit unsupported conversion result")
    unsupported.add_argument("--subject", required=True)
    unsupported.add_argument("--reason", required=True)
    unsupported.add_argument("--out", required=True)
    plan = sub.add_parser("request", help="Capture a text request for agent-assisted authoring; does not fabricate a model")
    plan.add_argument("--subject", required=True)
    plan.add_argument("--style", default="stylized smooth 3D")
    plan.add_argument("--prompt", required=True)
    plan.add_argument("--states", default="idle,making_something,milestone_level_up,working")
    plan.add_argument("--reference", action="append", default=[])
    plan.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        if args.command == "generate":
            from avatar_assets.pipeline import main as generate_main
            result = generate_main(args)
        elif args.command == "meshy":
            from avatar_assets.meshy import main as meshy_main
            result = meshy_main(args)
        elif args.command == "inspect":
            from avatar_assets.archive import inspect_bounded
            result = inspect_bounded(args.archive, args.out, args.sha256)
        elif args.command == "build3d":
            from avatar_assets.build3d import build_3d
            result = build_3d(args.scene, args.out, args.blender, args.resolution, args.samples, args.timeout, args.portrait_only)
        elif args.command == "verify":
            result = verify_pack(args.folder)
        elif args.command == "gallery":
            from avatar_assets.gallery import serve_gallery
            serve_gallery(args.port, args.catalog)
            return
        elif args.command == "export2d":
            from avatar_assets.export2d import export_2d
            result = export_2d(args.folder, args.out, args.preset, args.blender, args.size,
                               args.portrait, args.seconds, args.samples, args.denoise_portrait, args.resume_frames)
        elif args.command == "reload3d":
            from avatar_assets.review import reload_3d
            result = reload_3d(args.folder, args.out, args.blender)
        elif args.command == "review":
            from avatar_assets.review import record_review
            result = record_review(args.folder, args.review_json, args.out)
        elif args.command == "reskin":
            from avatar_assets.scene import read_scene, reskin
            out = new_directory(args.out)
            result = reskin(read_scene(args.scene), read_json(args.materials_json))
            result["source_prompt"] = result["prompt"]
            result["prompt"] = args.prompt or ("Apply this material-only skin, preserving geometry and actions: " +
                                              json.dumps(read_json(args.materials_json), sort_keys=True))
            result["source_subject"] = result["subject"]
            result["subject"] = args.subject or f"{result['id']} material skin variant"
            write_json(out / "scene.json", result)
            write_json(out / "provenance.json", {"schema_version": 1, "source_scene_sha256": sha256(args.scene),
                       "material_changes_sha256": sha256(args.materials_json), "provider_cost_usd": 0})
        elif args.command == "weft-metadata":
            from avatar_assets.weft_metadata import create_metadata
            result = create_metadata(args.folder, args.design, args.neutral, args.out)
        elif args.command == "verify3d":
            from avatar_assets.three_d import validate_glb
            result = validate_glb(args.glb, read_json(args.contract))
        elif args.command == "unsupported3d":
            from avatar_assets.three_d import unsupported_3d
            result = unsupported_3d(args.subject, args.reason)
            out = new_directory(args.out)
            write_json(out / "unsupported.json", result)
        else:
            from avatar_assets.safety import slug
            states = [slug(s) for s in args.states.split(",")]
            require(1 <= len(states) <= 12 and len(set(states)) == len(states), "Use 1–12 unique state IDs")
            require(len(args.reference) <= 4, "At most four selected references")
            result = {"schema_version": 1, "status": "needs_agent_authoring", "subject": args.subject,
                      "style": args.style, "prompt": args.prompt, "states": states,
                      "references": [{"name": local_file(p).name, "sha256": sha256(local_file(p))} for p in args.reference],
                      "provider_calls": 0, "provider_cost_usd": 0,
                      "next_action": "Use skills/avatar-asset-workflow to author and review scene JSON, then build3d."}
            out = new_directory(args.out)
            write_json(out / "request.json", result)
        if args.command in ("build3d", "export2d", "inspect", "reload3d", "review", "weft-metadata"):
            result = {"status": result.get("status", "inspected"), "out": str(Path(args.out).resolve()),
                      "artifacts": len(result["artifacts"]), "provider_cost_usd": 0}
        print(json.dumps(result, sort_keys=True, indent=2))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
