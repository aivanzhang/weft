"""Run one sealed construction-only temporal equipment proposal; no simulation."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

from weft.equipment_temporal_silhouette import CONTRACT, propose_temporal_equipment_geometry, temporal_equipment_eligibility
from weft.io import canonical_hash, local_time
from weft.shared_world_packaging import read_package_file
from weft.silhouette_geometry import prepare_silhouette_evidence
from weft.source_equivalence import load_source_equivalence


MAX_INPUT_BYTES, MAX_OUTPUT_BYTES, MAX_FILES = 64 << 20, 32 << 20, 128


def raw(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def qualify(plan, out):
    started, deadline = time.monotonic(), time.monotonic() + 35.
    out = Path(out).absolute()
    if out.exists() or out.resolve() != out:
        raise ValueError("Use a new real output directory")
    out.mkdir()
    files, inputs, signatures, total = {"plan.json": raw(plan)}, {}, {}, 0

    def read(root, name, digest, destination):
        nonlocal total
        if time.monotonic() >= deadline or len(inputs) >= MAX_FILES:
            raise TimeoutError("Input time/file bound exceeded")
        path = root / name
        before = path.stat(follow_symlinks=False)
        data = read_package_file(root, name, max_bytes=min(16 << 20, max(1, MAX_INPUT_BYTES - total)), expected_sha256=digest)
        after = path.stat(follow_symlinks=False)
        signature = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if signature(before) != signature(after):
            raise ValueError("Source input changed while reading")
        total += len(data)
        inputs[str(path)] = {"sha256": sha256(data).hexdigest(), "bytes": len(data)}
        signatures[path] = signature(after)
        files[destination] = data
        return data

    try:
        expected = {"schema_version", "scene", "masks", "binaries", "objects", "construction_frames", "static_prior", "conditional_exterior"}
        if set(plan) != expected or plan["schema_version"] != 1:
            raise ValueError("Use the complete versioned study plan")
        roots, seals = {}, {}
        for key in ("scene", "masks", "binaries"):
            spec = plan[key]
            if set(spec) != {"path", "manifest_sha256"}:
                raise ValueError("Each source requires its exact manifest identity")
            roots[key] = Path(spec["path"]).absolute()
            if roots[key].resolve() != roots[key]:
                raise ValueError("Source paths cannot contain symlinks")
            manifest = read(roots[key], "manifest.json", spec["manifest_sha256"], f"inputs/{key}-manifest.json")
            seals[key] = json.loads(manifest)["files"]
        if (not isinstance(plan["objects"], list) or not 1 <= len(plan["objects"]) <= 2
                or len(set(plan["objects"])) != len(plan["objects"])):
            raise ValueError("One fixed proposal for each of one or two distinct objects is allowed")
        scene_raw = read(roots["scene"], "scene.json", seals["scene"]["scene.json"], "inputs/scene.json")
        scene = json.loads(scene_raw)
        documents = {digest: read(roots["scene"], name, digest, "inputs/documents/" + digest + ".json")
                     for digest, name in scene["source_binding"]["evidence_documents"].items()}
        segmentation = read(roots["masks"], "masks/original-evidence.json", seals["masks"]["masks/original-evidence.json"],
                            "inputs/segmentation.json")
        masks = read(roots["masks"], "masks/mask-evidence.json", seals["masks"]["masks/mask-evidence.json"], "inputs/masks.json")
        parsed = json.loads(masks)
        if (type(plan["construction_frames"]) is not list or not 2 <= len(plan["construction_frames"]) <= 16
                or any(type(frame) is not int for frame in plan["construction_frames"])
                or any(a >= b for a, b in zip(plan["construction_frames"], plan["construction_frames"][1:]))):
            raise ValueError("Freeze two to sixteen distinct increasing construction frames")
        document_ids = {row["observation_document_sha256"] for row in scene["source_binding"]["entity_bindings"]
                        if row["world_entity_id"] in plan["objects"]}
        if len(document_ids) != 1:
            raise ValueError("Selected equipment must share its exact observation document")
        document = json.loads(documents[next(iter(document_ids))])
        partitions = {(row["frame"], row["object_id"]): row["partition"] for row in document["observations"]}
        selected_slots = [row for row in parsed["observations"] if row["frame"] in plan["construction_frames"]]
        if any(partitions.get((row["frame"], row["object_id"])) != "construction" for row in selected_slots):
            raise ValueError("Withheld raster payloads cannot enter construction preparation")
        bindings = {row["world_entity_id"]: row for row in scene["source_binding"]["entity_bindings"]}
        preflights = {name: temporal_equipment_eligibility(parsed, bindings[name]["observation_entity_id"],
                                                        construction_frames=plan["construction_frames"])
                      for name in plan["objects"]}
        for name, preflight in preflights.items():
            files[f"objects/{name}-eligibility.json"] = raw(preflight)
        eligible = [name for name in plan["objects"] if preflights[name]["status"] == "requires_prepared_custody"]
        # Only the frozen construction selection requests raster payloads.
        selected = [row for row in selected_slots if row["mask_status"] == "decoded"] if eligible else []
        pngs, binaries = {}, {}
        for row in selected:
            name = row["raster_path"]
            binary = "binary/" + str(Path(name).with_suffix(".bin"))
            if binary in seals["binaries"]:
                binaries[name] = read(roots["binaries"], binary, seals["binaries"][binary], "inputs/" + binary)
            else:
                encoded = "masks/" + name
                pngs[name] = read(roots["masks"], encoded, seals["masks"][encoded], "inputs/png/" + name)
        equivalences = []
        for item in scene["source_binding"].get("source_equivalences", []) if eligible else []:
            mapping = load_source_equivalence(roots["scene"] / item["bundle_path"], deadline=deadline,
                                             max_input_bytes=MAX_INPUT_BYTES - total)
            if mapping.manifest_sha256 != item["bundle_manifest_sha256"]:
                raise ValueError("Source-equivalence identity differs from scene binding")
            total += mapping.total_input_bytes
            equivalences.append(mapping)
            files[f"inputs/source-mapping-{mapping.manifest_sha256}.json"] = raw({
                "path": str(roots["scene"] / item["bundle_path"]), "manifest_sha256": mapping.manifest_sha256,
                "verified_bytes": mapping.total_input_bytes, "scope": "Existing complete proof reverified; original media hashed, not decoded."})
        context = prepare_silhouette_evidence(scene, source_documents=documents, segmentation_evidence=segmentation,
            mask_document=masks, raster_documents=pngs, binary_raster_documents=binaries,
            source_equivalences=equivalences) if eligible else None
        results = []
        for name in plan["objects"]:
            if name not in eligible:
                results.append({"object_id": name, "status": "ineligible_source_track",
                                "rejection_reasons": sorted({reason for slot in preflights[name]["slots"] for reason in slot["reasons"]}),
                                "source_fit": "not_attempted", "collision_proxy": "unchanged"})
                continue
            result = propose_temporal_equipment_geometry(context, name, construction_frames=plan["construction_frames"],
                static_prior=plan["static_prior"], conditional_exterior=plan["conditional_exterior"],
                max_seconds=min(15., deadline - time.monotonic()))
            files[f"objects/{name}.json"] = raw(result)
            if result["geometry_config"] is not None:
                files[f"objects/{name}-geometry-config.json"] = raw(result["geometry_config"])
            results.append({"object_id": name, "status": result["status"],
                            "rejection_reasons": result["report"]["rejection_reasons"],
                            "source_fit": result["report"]["source_fit"],
                            "collision_proxy": result["report"]["collision_proxy"]})
        for path, signature in signatures.items():
            info = path.stat(follow_symlinks=False)
            if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns) != signature:
                raise ValueError("Source input changed before publication")
        summary = {"profile": "temporal-equipment-study-v1", "status": "completed", "created_at": local_time(),
                   "settings_sha256": canonical_hash(plan), "contract_sha256": canonical_hash(CONTRACT),
                   "objects": results, "input_files": inputs, "verified_input_bytes": total,
                   "cached_binary_masks": len(binaries), "cached_png_masks": len(pngs),
                   "prepared_source_context": bool(eligible),
                   "source_unchanged": True, "physics_steps": 0, "api_cost_usd": 0, "recurring_cost_usd": 0,
                   "source_fit_scope": "Conditional construction images only; all rejections and inherited foreground misses retained.",
                   "native_admission": "not_attempted", "contact": "unknown", "physical_validation": "unverified",
                   "seconds": time.monotonic() - started}
        files["result.json"] = raw(summary)
        if time.monotonic() >= deadline or len(files) >= MAX_FILES or sum(map(len, files.values())) > MAX_OUTPUT_BYTES:
            raise ValueError("Study result exceeded output/time bound")
        for name, data in files.items():
            path = out / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)
        with (out / "manifest.json").open("xb") as stream:
            stream.write(raw({"schema_version": 1, "status": "completed", "files": {name: sha256(data).hexdigest() for name, data in files.items()}}))
        return summary
    except Exception as error:
        with (out / "failure.json").open("xb") as stream:
            stream.write(raw({"status": "failed", "type": type(error).__name__, "message": str(error),
                              "seconds": time.monotonic() - started, "plan": plan, "input_files": inputs}))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan)
    if plan_path.stat().st_size > 16 << 10:
        raise ValueError("Study plan exceeds its byte bound")
    summary = qualify(json.loads(plan_path.read_bytes()), args.out)
    print(json.dumps({"status": summary["status"], "objects": summary["objects"]}, sort_keys=True))


if __name__ == "__main__":
    main()
