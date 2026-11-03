"""Source-pinned AI image prompts feed existing pose/tracking evidence, never physics truth."""

import base64
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import time

from weft.exercise_body_source import _json, _path
from weft.io import canonical_hash, digest, write_json
from weft.portal_pipeline import INSTRUCTIONS, SCHEMA, validate_plan
from weft.openai_access import MODEL, MAX_OUTPUT_TOKENS
from weft.shared_world_packaging import read_package_file, read_package_manifest, write_package_files

PROFILE = "exercise-perception-bridge-v1"
SCOPE = "AI-proposed equipment regions and inferred poses/masks; inventory, identity, depth, contact and physical parameters remain unverified."
PREFLIGHT_PROFILE = "exercise-perception-preflight-v1"
LEGACY_ANCHOR_PROFILE = "sealed-pose-tracking-anchor-v1"
VISIBLE_ANCHOR_PROFILE = "sealed-pose-visible-chain-tracking-anchor-v2"
# This published producer has identical preflight/body/anchor semantics; never execute its snapshot.
REVIEWED_PREFLIGHT_PRODUCERS = frozenset({'75434812f8b7727e79ba64b82d33bfb2e31fb752e18b8d99208865dfaea1479d',
    'd20a3c187762c5d1608bc2a793302b13f7e6a09992ea4a6a565952b0a016ed81'})


def _check_deadline(deadline):
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise TimeoutError("Perception preflight deadline expired")


def _preflight_metadata(value):
    if (not isinstance(value, dict) or type(value.get("frame_count")) is not int or not 12 <= value["frame_count"] <= 500
            or any(type(value.get(k)) is not int for k in ("width", "height"))
            or not 64 <= value["width"] <= 4096 or not 64 <= value["height"] <= 2160
            or value["width"]*value["height"] > 8_847_360
            or type(value.get("fps")) not in (int, float) or not math.isfinite(value["fps"]) or not 1 <= value["fps"] <= 120
            or value["frame_count"]/value["fps"] > 60):
        raise ValueError("Perception preflight requires complete original tracking-sized source metadata")
    times = value.get("decoder_timestamps_s")
    if (not isinstance(times, list) or len(times) != value["frame_count"]
            or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in times)
            or any(b <= a for a, b in zip(times, times[1:])) or times[-1]-times[0] > 60):
        raise ValueError("Perception preflight requires complete increasing decoder times within sixty seconds")
    return value


def _preflight_seed(seed, metadata, *, deadline, expected_manifest_sha256=None, anchor_profile=LEGACY_ANCHOR_PROFILE):
    from weft.shared_world_packaging import collect_shared_source_files

    _check_deadline(deadline)
    manifest, declared = read_package_manifest(seed)
    pin = sha256(manifest).hexdigest()
    if expected_manifest_sha256 is not None and pin != expected_manifest_sha256:
        raise ValueError("Preflight body seed manifest changed")
    seed_manifest = _json(manifest)
    if (len(declared) > 511 or "construction-seed.json" not in declared
            or seed_manifest.get("status") != "construction_only" or seed_manifest.get("simulation_ready") is not False
            or (Path(seed)/"failure.json").exists() or (Path(seed)/"failure.json").is_symlink()):
        raise ValueError("Preflight requires a bounded construction-only body seed")
    scene = _json(read_package_file(seed, "scene.json", max_bytes=16 << 20, expected_sha256=declared["scene.json"]))
    proof = collect_shared_source_files(scene, seed, expected_manifest_sha256=pin, deadline=deadline)
    total = len(manifest)
    for name, digest_value in declared.items():
        _check_deadline(deadline)
        if name in proof.files:
            raw = proof.files[name]
        else:
            raw = read_package_file(seed, name, max_bytes=min(64 << 20, max(1, (128 << 20)-total)), expected_sha256=digest_value)
        total += len(raw)
        if total > 128 << 20:
            raise ValueError("Preflight body seed exceeds its byte bound")
    if scene["source_binding"]["source_sha256"] != metadata["source_sha256"]:
        raise ValueError("Preflight body source differs from original video")
    human = next(row for row in scene["source_binding"]["entity_bindings"] if row["world_entity_id"] == "human")
    doc_pin = human["observation_document_sha256"]
    doc = _json(proof.files[scene["source_binding"]["evidence_documents"][doc_pin]])
    source = doc["source"]
    if (source["decoded_frames"] != metadata["frame_count"] or source["width_px"] != metadata["width"]
            or source["height_px"] != metadata["height"] or source.get("fps") != metadata["fps"]):
        raise ValueError("Preflight body frame inventory differs from decoded source")
    if "native_clock" in source:
        for row, actual in zip(source["native_clock"]["rows"], metadata["decoder_timestamps_s"], strict=True):
            if row["decoder_timestamp_s"] is not None and row["decoder_timestamp_s"] != actual:
                raise ValueError("Preflight native capture and source decoder clock differ")
    proposal = {**metadata, "boxes": []}
    anchor = body_tracking_anchor(seed, scene, proposal, **({"anchor_profile": anchor_profile} if anchor_profile != LEGACY_ANCHOR_PROFILE else {}))
    return pin, anchor


def _detail_metadata(original_raw, metadata):
    original = _json(original_raw)
    if (any(original[key] != metadata[other] for key, other in (
            ("source_sha256", "source_sha256"), ("width", "width"), ("height", "height"),
            ("decoded_frames", "frame_count"), ("fps", "fps")))
            or any(row.get("decoder_timestamp_s") is not None
                   and row["decoder_timestamp_s"] != metadata["decoder_timestamps_s"][row["frame"]]
                   for row in original["frames"])):
        raise ValueError("Body-detail history differs from the complete original source or decoder clock")


def prepare_exercise_perception_preflight(video, out, *, pose_model=None, body_source=None, max_seconds=210,
                                         anchor_profile=LEGACY_ANCHOR_PROFILE, retain_body_source=False,
                                         body_detail_recipe=None, body_detail_inputs=None):
    """Prepare source/body custody before either provider; never send or reserve API money."""
    from weft.segmentation_lab import inspect_video
    from weft.exercise_body_source import prepare_exercise_body_source
    from weft.shared_world_lab import prepare_shared_world_seed

    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= 210:
        raise ValueError("Perception preflight requires a 1–210 second bound")
    if anchor_profile not in (LEGACY_ANCHOR_PROFILE, VISIBLE_ANCHOR_PROFILE):
        raise ValueError("Unknown source tracking anchor profile")
    if body_source is not None and pose_model is not None:
        raise ValueError("Choose an existing body or a local pose model, not both")
    if type(retain_body_source) is not bool or retain_body_source and body_source is None:
        raise ValueError("Retained body custody requires an explicit existing sealed body")
    started = time.monotonic(); deadline = started+max_seconds
    detail, detail_files, detail_options = None, {}, {}
    if body_detail_recipe is not None or body_detail_inputs is not None:
        from weft import portal_body_detail_recipe as recipes
        if (body_source is not None or retain_body_source or body_detail_recipe is None
                or not isinstance(body_detail_inputs, tuple) or len(body_detail_inputs) != 2):
            raise ValueError("Body detail requires its original evidence/raw pair and a new local body")
        detail_files = {"implementation/"+name: Path(__file__).with_name(name).read_bytes()
                        for name in recipes.IMPLEMENTATIONS}
        detail = recipes.validate_body_detail_recipe(body_detail_recipe, *body_detail_inputs)
        detail_options = recipes.body_detail_options(detail)
        detail_options.pop("native_timing")
        detail_files.update(zip(recipes.INPUT_FILES, body_detail_inputs, strict=True))
        detail_files[recipes.RECIPE_FILE] = recipes._encode(detail)
        if pose_model is None:
            raise ValueError("Body detail requires the fixed local Full model")
        model = _path(pose_model)
        if (model.name != recipes.MODEL_FILENAME or not model.is_file() or model.stat().st_size != recipes.MODEL_BYTES
                or digest(model) != recipes.MODEL_SHA256):
            raise ValueError("Body detail requires the exact fixed local Full model")
        _check_deadline(deadline)
    video, out = _path(video), _path(out)
    supplied = [_path(v) for v in (body_source, pose_model) if v is not None]
    if out.exists() or any(out == p or out.is_relative_to(p) or p.is_relative_to(out) for p in [video, *supplied]):
        raise ValueError("Choose a new preflight directory outside every input")
    if body_source is None and (pose_model is None or not _path(pose_model).is_file()):
        raise ValueError("Local pose model or exact sealed body is required before AI analysis")
    metadata = _preflight_metadata(inspect_video(video, deadline=deadline, include_fps=True))
    if detail is not None:
        _detail_metadata(body_detail_inputs[0], metadata)
    _check_deadline(deadline); out.mkdir(parents=True)
    write_json(out/"metadata.json", metadata)
    if detail_files:
        write_package_files(out, detail_files)
    stage = "body"
    try:
        generated_body = body_source is None
        if generated_body:
            body_source = out/"body_source"
            prepare_exercise_body_source(video, body_source, pose_model=pose_model, native_timing=True,
                source_note="Local source-bound preflight before API analysis; no source rights claim",
                max_seconds=min(180, deadline-time.monotonic()), **detail_options)
        elif retain_body_source:
            from weft.perception_body_reuse import copy_sealed_body
            reuse = copy_sealed_body(body_source, out/"body_source",
                                     source_sha256=metadata["source_sha256"], deadline=deadline)
            body_source = out/"body_source"
            write_json(out/"body-reuse.json", reuse)
        stage = "seed"
        seed = prepare_shared_world_seed(body_source, out/"body_seed", deadline=deadline)
        stage = "anchor"
        seed_pin, anchor = _preflight_seed(seed, metadata, deadline=deadline, anchor_profile=anchor_profile)
        if detail is not None:
            _validate_detail_body(out/"body_source", detail, seed)
        write_json(out/"body-anchor.json", anchor)
        if digest(video) != metadata["source_sha256"]:
            raise ValueError("Original video changed during local preflight")
        result = {"profile": PREFLIGHT_PROFILE, "status": "local_ready", "source_sha256": metadata["source_sha256"],
            "metadata": metadata, "base_run": str(seed), "base_manifest_sha256": seed_pin,
            "body_anchor": anchor, "preflight_directory": str(out), "api_calls": 0,
            "inference_calls": int(generated_body), "simulation_ready": False, "seconds": time.monotonic()-started}
        write_json(out/"result.json", result)
        write_package_files(out, {"implementation/exercise_perception_pipeline.py": Path(__file__).read_bytes()})
        names = ["metadata.json", "body-anchor.json", "result.json", "implementation/exercise_perception_pipeline.py"]
        if anchor_profile == VISIBLE_ANCHOR_PROFILE:
            write_package_files(out, {"implementation/person_tracking_anchor.py": Path(__file__).with_name("person_tracking_anchor.py").read_bytes()})
            names.append("implementation/person_tracking_anchor.py")
        children = {"body_seed": seed_pin}
        if generated_body or retain_body_source:
            children["body_source"] = digest(out/"body_source/manifest.json")
        if retain_body_source:
            write_package_files(out, {"implementation/perception_body_reuse.py": Path(__file__).with_name("perception_body_reuse.py").read_bytes()})
            names += ["body-reuse.json", "implementation/perception_body_reuse.py"]
        if detail is not None:
            for name, expected in detail_files.items():
                if name.startswith("implementation/") and Path(__file__).with_name(Path(name).name).read_bytes() != expected:
                    raise ValueError("Body-detail implementation changed during preparation")
                if (out/name).read_bytes() != expected:
                    raise ValueError("Body-detail source or recipe changed during preparation")
            names += list(detail_files)
        _check_deadline(deadline)
        write_json(out/"manifest.json", {"schema_version": 1, "profile": PREFLIGHT_PROFILE, "status": "local_ready",
            "files": {name: digest(out/name) for name in names}, "children": children})
        return {**result, "manifest_sha256": digest(out/"manifest.json")}
    except Exception as error:
        write_json(out/"failure.json", {"stage": stage, "error_type": type(error).__name__, "message": str(error)[:500],
            "api_calls": 0, "retry": False, "scope": "Local source evidence retained; no provider call or reservation"})
        raise


def _validate_detail_body(body, recipe, seed):
    from weft.portal_body_detail_recipe import validate_body_detail_body
    body_manifest, members = read_package_manifest(body)
    _, seed_members = read_package_manifest(seed)
    original = "inputs/human-source/source-manifest.json"
    if (seed_members.get(original) != sha256(body_manifest).hexdigest()
            or read_package_file(seed, original, max_bytes=8 << 20,
                                 expected_sha256=seed_members.get(original)) != body_manifest):
        raise ValueError("Body-detail seed must retain the exact newly generated body manifest")
    marker, model = (_json(read_package_file(body, name, max_bytes=1 << 20,
        expected_sha256=members.get(name))) for name in ("body-source.json", "model.json"))
    validate_body_detail_body(recipe, marker, model)


def validate_perception_preflight(video, preflight, *, deadline):
    """Revalidate sealed inputs without inference, source decoding or rebuilding the seed."""
    _check_deadline(deadline)
    video, preflight = _path(video), _path(preflight)
    if (video.suffix.lower() != ".mp4" or not 1 <= video.stat().st_size <= 37_000_000
            or (preflight/"failure.json").exists() or (preflight/"failure.json").is_symlink()):
        raise ValueError("Preflight needs its unchanged original MP4 and no failure marker")
    raw, declared = read_package_manifest(preflight)
    manifest = _json(raw)
    names = {"metadata.json", "body-anchor.json", "result.json", "implementation/exercise_perception_pipeline.py"}
    reused_body = "body-reuse.json" in declared
    detailed = "body-detail/recipe.json" in declared
    if detailed:
        from weft import portal_body_detail_recipe as recipes
        names.update(recipes.ARTIFACTS)
        if reused_body or "body_source" not in manifest.get("children", {}):
            raise ValueError("Body detail must retain its newly generated body")
    if reused_body:
        names |= {"body-reuse.json", "implementation/perception_body_reuse.py"}
    stored_anchor = _json(read_package_file(preflight, "body-anchor.json", max_bytes=128 << 10,
                                           expected_sha256=declared.get("body-anchor.json")))
    anchor_profile = stored_anchor.get("profile")
    if anchor_profile == VISIBLE_ANCHOR_PROFILE:
        names.add("implementation/person_tracking_anchor.py")
    elif anchor_profile != LEGACY_ANCHOR_PROFILE:
        raise ValueError("Unknown sealed preflight anchor profile")
    if (set(declared) != names or manifest.get("profile") != PREFLIGHT_PROFILE or manifest.get("status") != "local_ready"
            or set(manifest.get("children", {})) not in ({"body_seed"}, {"body_seed", "body_source"})):
        raise ValueError("Preflight seal differs from the closed local preparation contract")
    files, detail_bytes = {}, 0
    for name, pin in declared.items():
        _check_deadline(deadline)
        limit = (min(16 << 20, (16 << 20)-detail_bytes) if detailed and name in recipes.INPUT_FILES
                 else recipes.MAX_RECIPE_BYTES if detailed and name == recipes.RECIPE_FILE else 1 << 20)
        files[name] = read_package_file(preflight, name, max_bytes=max(1, limit), expected_sha256=pin)
        if detailed and name in recipes.INPUT_FILES:
            detail_bytes += len(files[name])
            if detail_bytes > 16 << 20:
                raise ValueError("Body-detail raw inputs exceed their aggregate bound")
    producer = files["implementation/exercise_perception_pipeline.py"]
    if producer != Path(__file__).read_bytes() and (detailed or sha256(producer).hexdigest() not in REVIEWED_PREFLIGHT_PRODUCERS):
        raise ValueError("Preflight implementation differs from this reviewed runtime; explicit compatibility is required")
    if (anchor_profile == VISIBLE_ANCHOR_PROFILE
            and files["implementation/person_tracking_anchor.py"] != Path(__file__).with_name("person_tracking_anchor.py").read_bytes()):
        raise ValueError("Preflight person anchor implementation differs from its exact sealed runtime")
    metadata = _preflight_metadata(_json(files["metadata.json"]))
    if detailed:
        for name in recipes.IMPLEMENTATIONS:
            if files["implementation/"+name] != Path(__file__).with_name(name).read_bytes():
                raise ValueError("Body-detail recipe implementation differs from its reviewed snapshot")
        detail = recipes.validate_body_detail_recipe(_json(files[recipes.RECIPE_FILE]),
                                                     *(files[name] for name in recipes.INPUT_FILES))
        _detail_metadata(files[recipes.INPUT_FILES[0]], metadata)
        _validate_detail_body(preflight/"body_source", detail, preflight/"body_seed")
    if reused_body:
        from weft.perception_body_reuse import validate_reuse_receipt
        if files["implementation/perception_body_reuse.py"] != Path(__file__).with_name("perception_body_reuse.py").read_bytes():
            raise ValueError("Preflight body reuse differs from its reviewed implementation")
        validate_reuse_receipt(_json(files["body-reuse.json"]), source_sha256=metadata["source_sha256"],
                              body_manifest_sha256=manifest["children"].get("body_source"))
    if digest(video) != metadata.get("source_sha256"):
        raise ValueError("Preflight video differs from its original source")
    seed = preflight/"body_seed"
    pin, anchor = _preflight_seed(seed, metadata, deadline=deadline,
        expected_manifest_sha256=manifest["children"]["body_seed"], anchor_profile=anchor_profile)
    result = _json(files["result.json"])
    result_fields = {"profile", "status", "source_sha256", "metadata", "base_run", "base_manifest_sha256",
                     "body_anchor", "preflight_directory", "api_calls", "inference_calls", "simulation_ready", "seconds"}
    if (set(result) != result_fields or result.get("profile") != PREFLIGHT_PROFILE or result.get("status") != "local_ready"
            or type(result.get("api_calls")) is not int or result["api_calls"] != 0
            or type(result.get("seconds")) not in (int, float) or not math.isfinite(result["seconds"]) or not 0 <= result["seconds"] <= 210
            or result.get("simulation_ready") is not False or type(result.get("inference_calls")) is not int
            or result["inference_calls"] != int("body_source" in manifest["children"] and not reused_body)
            or result.get("metadata") != metadata or result.get("source_sha256") != metadata["source_sha256"]
            or result.get("base_run") != str(seed) or result.get("base_manifest_sha256") != pin
            or result.get("preflight_directory") != str(preflight) or result.get("body_anchor") != anchor
            or _json(files["body-anchor.json"]) != anchor):
        raise ValueError("Preflight result or body anchor differs from sealed source inputs")
    if "body_source" in manifest["children"]:
        if (preflight/"body_source/failure.json").exists() or (preflight/"body_source/failure.json").is_symlink():
            raise ValueError("Preflight generated body gained a failure marker")
        body_raw, body_files = read_package_manifest(preflight/"body_source")
        if sha256(body_raw).hexdigest() != manifest["children"]["body_source"] or len(body_files) > 511:
            raise ValueError("Preflight generated-body manifest changed")
        body_bytes = len(body_raw)
        for name, expected in body_files.items():
            _check_deadline(deadline)
            body_bytes += len(read_package_file(preflight/"body_source", name,
                max_bytes=min(64 << 20, max(1, (128 << 20)-body_bytes)), expected_sha256=expected))
            if body_bytes > 128 << 20:
                raise ValueError("Preflight generated body exceeds its original bound")
    _check_deadline(deadline)
    return {**result, "manifest_sha256": sha256(raw).hexdigest()}


def body_tracking_anchor(seed, scene, proposal, *, anchor_profile=LEGACY_ANCHOR_PROFILE):
    """A sealed pose supplies a tracking hint, never a rigid person or verified identity."""
    if anchor_profile not in (LEGACY_ANCHOR_PROFILE, VISIBLE_ANCHOR_PROFILE):
        raise ValueError("Unknown source tracking anchor profile")
    manifest, declared = read_package_manifest(seed)
    sealed_scene = _json(read_package_file(seed, "scene.json", max_bytes=16 << 20,
                                         expected_sha256=declared["scene.json"]))
    if scene != sealed_scene:
        raise ValueError("Body anchor scene differs from its sealed source binding")
    binding = scene["source_binding"]
    human = next(row for row in binding["entity_bindings"] if row["world_entity_id"] == "human")
    pin = human["observation_document_sha256"]
    name = binding["evidence_documents"][pin]
    if declared.get(name) != pin:
        raise ValueError("Body anchor observation is not sealed by the body seed")
    document = _json(read_package_file(seed, name, max_bytes=32 << 20, expected_sha256=pin))
    if (document["source"]["sha256"] != proposal["source_sha256"]
            or document["image"] != {"width_px": proposal["width"], "height_px": proposal["height"]}):
        raise ValueError("Body anchor differs from original source pixels")
    rows = [row for row in document["observations"] if row["frame"] == 0
            and row["object_id"] == human["observation_entity_id"]]
    if (len(rows) != 1 or rows[0]["partition"] != "construction"
            or rows[0]["visibility"] not in {"detected", "partial"} or not rows[0]["evidence_ids"]):
        raise ValueError("Body tracking anchor requires construction pose evidence at original frame zero")
    measurement = rows[0]["measurements"].get("landmarks", {})
    if tuple(measurement.get(k) for k in ("coordinate_system", "unit", "status")) != ("image", "pixels", "inferred"):
        raise ValueError("Body anchor requires inferred source-pixel landmarks")
    if anchor_profile == VISIBLE_ANCHOR_PROFILE:
        from weft.person_tracking_anchor import derive_visible_chain_anchor

        derived = derive_visible_chain_anchor(rows[0], width=proposal["width"], height=proposal["height"])
        if derived["status"] != "proposed":
            raise ValueError("Visible-chain body anchor abstained: " + ", ".join(derived["reasons"]))
        return {**derived, "source_sha256": proposal["source_sha256"], "body_seed_manifest_sha256": sha256(manifest).hexdigest(),
            "observation_document_sha256": pin, "minimum_confidence": .5,
            "prompt": [len(proposal["boxes"])+1, 0, *derived["roi_xywh"]]}
    points = {p["index"]: p for p in measurement["value"]}
    required = (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
    def confident(point):
        return all(type(point.get(k)) in (int, float) and math.isfinite(point[k]) and .5 <= point[k] <= 1
                   for k in ("visibility", "presence"))
    if any(i not in points or not confident(points[i]) for i in required):
        raise ValueError("Body anchor needs confident head, torso and limb endpoints at frame zero")
    selected = [p for p in points.values() if confident(p)]
    width, height = proposal["width"], proposal["height"]
    if any(len(p["position"]) != 2 or any(type(x) not in (int, float) or not math.isfinite(x) for x in p["position"])
           or not 0 <= p["position"][0] < width or not 0 <= p["position"][1] < height for p in selected):
        raise ValueError("Body anchor landmarks must lie inside original source pixels")
    xs, ys = [p["position"][0] for p in selected], [p["position"][1] for p in selected]
    padding = max(2, math.ceil(.05 * max(max(xs)-min(xs), max(ys)-min(ys))))
    left, top = max(0, math.floor(min(xs))-padding), max(0, math.floor(min(ys))-padding)
    right, bottom = min(width, math.ceil(max(xs))+padding), min(height, math.ceil(max(ys))+padding)
    if min(right-left, bottom-top) < 8:
        raise ValueError("Body tracking anchor is degenerate")
    return {"profile": "sealed-pose-tracking-anchor-v1", "source_sha256": proposal["source_sha256"],
        "body_seed_manifest_sha256": sha256(manifest).hexdigest(), "observation_document_sha256": pin,
        "observation_id": rows[0]["id"], "evidence_ids": rows[0]["evidence_ids"],
        "landmark_indices": sorted(p["index"] for p in selected), "minimum_confidence": .5,
        "padding_pixels": padding, "padding_rule": "max(2, ceil(5% longest confident-landmark span)), clipped to image",
        "prompt": [len(proposal["boxes"])+1, 0, left, top, right-left, bottom-top],
        "scope": "Inferred person tracking initialization only; ordinary source-role checks still decide correspondence. Never a rigid body box.",
        "physical_identity_verified": False}


def read_equipment_analysis(analysis):
    """Reconstruct the original bounded provider payload before trusting its region coordinates."""
    from weft.equipment_analysis_contract import LEGACY_PROFILE, SAMPLED_PROFILE, analysis_contract, request_profile
    analysis = _path(analysis)
    limits = {"frames.json": 32 << 10, "plan.json": 16 << 10, "request.json": 64 << 10,
              "response.json": 128 << 10, "receipt.json": 32 << 10}
    files = {name: read_package_file(analysis, name, max_bytes=limit) for name, limit in limits.items()}
    frames, plan, request, response, receipt = (_json(files[name]) for name in limits)
    profile = request_profile(request)
    contract = analysis_contract(profile)
    validate_plan(plan, analysis_profile=profile, frames=frames)
    if (plan["route"] != "exercise" or plan["camera"] != "fixed" or plan["people"] != 1
            or plan.get("exercise_inventory") != "equipment_or_obstacles" or not plan["objects"]):
        raise ValueError("Saved analysis must propose one fixed-camera person and explicit source-image equipment regions")
    if (not isinstance(frames.get("source_sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", frames["source_sha256"])
            or not isinstance(frames.get("frames"), list) or not 1 <= len(frames["frames"]) <= 6
            or frames["frames"][0].get("frame") != 0
            or any(type(frames.get(k)) is not int for k in ("width", "height", "decoded_frame_count"))
            or type(frames.get("fps")) not in (int, float) or not math.isfinite(frames["fps"]) or not 1 <= frames["fps"] <= 120
            or not 12 <= frames["decoded_frame_count"] <= 500
            or not 16 <= frames["width"] <= 4096 or not 16 <= frames["height"] <= 4096):
        raise ValueError("Equipment tracking needs original first-frame analysis within 500 decoded frames")
    payload = {k: v for k, v in request.items() if k not in {"images", "request_sha256", "scope"}}
    if profile == SAMPLED_PROFILE and (set(payload) != {'model', 'instructions', 'store', 'input', 'reasoning', 'max_output_tokens', 'text'}
            or payload.get('reasoning') != {'effort': 'none'}):
        raise ValueError('Sampled equipment analysis differs from its closed request envelope')
    if (request.get("images") != frames["frames"] or payload.get("store") is not False
            or payload.get("model") != MODEL or payload.get("instructions") != contract['instructions']
            or payload.get("max_output_tokens") != MAX_OUTPUT_TOKENS
            or payload.get("text") != {"format": {"type": "json_schema", "name": contract['format_name'], "strict": True, "schema": contract['schema']}}
            or not isinstance(payload.get("input"), list) or len(payload["input"]) != 1
            or payload["input"][0].get("role") != "user"):
        raise ValueError("Analysis payload does not bind its sampled source frames")
    content = payload["input"][0].get("content")
    if not isinstance(content, list) or len(content) != 1 or content[0].get("type") != "input_text":
        raise ValueError("Analysis requires its original single video-description prompt")
    description = _json(content[0]["text"].encode())
    expected = {"video": {k: frames[k] for k in ("width", "height", "fps", "decoded_frame_count")},
                "frames": [{k: row[k] for k in ("frame", "decoder_time_s", "nominal_time_s")} for row in frames["frames"]]}
    if description != expected:
        raise ValueError("Analysis prompt differs from source metadata and original frame identities")
    for row in frames["frames"]:
        name = row.get("artifact")
        if not isinstance(name, str) or not re.fullmatch(r"frame-\d{5}\.jpg", name) or name in files:
            raise ValueError("Analysis frame path is invalid or duplicated")
        if profile == SAMPLED_PROFILE and name != f"frame-{row['frame']:05d}.jpg":
            raise ValueError('Sampled equipment image must retain its original frame artifact identity')
        files[name] = read_package_file(analysis, name, max_bytes=300_000, expected_sha256=row["sha256"])
        content.append({"type": "input_image", "detail": "high", "image_url":
                        "data:image/jpeg;base64," + base64.b64encode(files[name]).decode()})
    request_hash = canonical_hash(payload)
    texts = [part["text"] for item in response.get("output", []) if item.get("type") == "message"
             for part in item.get("content", []) if part.get("type") == "output_text"]
    if (request_hash != request.get("request_sha256") or request_hash != receipt.get("request_sha256")
            or receipt.get("provider") != "openai" or receipt.get("authentication") != "server_project_api_key"
            or receipt.get("response_id") != response.get("id") or response.get("status") != "completed"
            or response.get("model") != payload.get("model") or receipt.get("model") != payload.get("model")
            or len(texts) != 1 or len(texts[0]) > 12_000 or _json(texts[0].encode()) != plan):
        raise ValueError("Equipment regions do not match the saved provider request/response/receipt")
    width, height = frames["width"], frames["height"]
    boxes = []
    for index, obj in enumerate(plan["objects"], 1):
        x0, y0, x1, y1 = obj["box_xyxy"]
        left, top, right, bottom = math.floor(x0 * width), math.floor(y0 * height), math.ceil(x1 * width), math.ceil(y1 * height)
        if min(right-left, bottom-top) < 8:
            raise ValueError("AI equipment region is too small for the tracking contract")
        boxes.append([index, obj['source_frame'] if profile == SAMPLED_PROFILE else 0, left, top, right-left, bottom-top])
    proposal = {"source_sha256": frames["source_sha256"], "width": width, "height": height,
            "frame_count": frames["decoded_frame_count"], "boxes": boxes, "objects": plan["objects"],
            "request_sha256": request_hash, "region_basis": ("AI first-image localization, not verified identity" if profile == LEGACY_PROFILE else
                "AI selected-sampled-image localization; source initialization frame retained, not verified identity or earlier motion"),
            "coordinate_conversion": "normalized endpoints to source pixels with outward rounding; no padding",
            "input_sha256": {name: sha256(raw).hexdigest() for name, raw in files.items()}}
    if profile == SAMPLED_PROFILE:
        proposal['analysis_profile'] = profile
        proposal['selected_samples'] = [{'object_id': index, 'source_frame': obj['source_frame'],
            'artifact': next(row['artifact'] for row in frames['frames'] if row['frame'] == obj['source_frame']),
            'sha256': next(row['sha256'] for row in frames['frames'] if row['frame'] == obj['source_frame'])}
            for index, obj in enumerate(plan['objects'], 1)]
    return proposal, files


def prepare_exercise_perception(video, analysis, out, *, pose_model=None, body_source=None,
                                saved_tracking=None, execute_tracking=False, max_meta_cost_usd=None,
                                project_root=None, max_seconds=420, preflight=None, progress=None):
    """Prepare offline by default; explicit execution reuses the canonical one-call Meta adapter."""
    from weft.segmentation_lab import inspect_video, track_video

    if type(execute_tracking) is not bool or (execute_tracking and saved_tracking is not None):
        raise ValueError("Choose one new tracking request or a saved exact-source tracking run")
    if preflight is not None and (pose_model is not None or body_source is not None):
        raise ValueError("Reusable preflight owns its original body/model; do not override it")
    if progress is not None and not callable(progress):
        raise ValueError("Perception progress requires a bounded stage callback")
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= 420:
        raise ValueError("Perception preparation requires a 1–420 second bound")
    started = time.monotonic(); deadline = started + max_seconds
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("Exercise perception exceeded its work deadline")
    def report(stage, message):
        check()
        if progress is not None:
            progress({"stage": stage, "message": message, "seconds": time.monotonic()-started})
    video, analysis, out = map(_path, (video, analysis, out))
    inputs = [video, analysis, *(_path(v) for v in (body_source, saved_tracking, pose_model, preflight) if v is not None)]
    if out.exists() or any(out == p or out.is_relative_to(p) or p.is_relative_to(out) for p in inputs):
        raise ValueError("Choose a new perception directory outside every input")
    report("validating_analysis", "Checking exact saved AI regions and source images")
    proposal, analysis_files = read_equipment_analysis(analysis)
    needs_body = execute_tracking or saved_tracking is not None or body_source is not None or preflight is not None
    if needs_body and len(proposal["boxes"]) > 3:
        raise ValueError("At most three AI equipment regions plus one sealed person anchor fit this route; no truncation")
    import cv2
    frame_rows = _json(analysis_files["frames.json"])
    sampled = frame_rows["frames"]
    if any(type(row.get("frame")) is not int for row in sampled) or [r["frame"] for r in sampled] != sorted(set(r["frame"] for r in sampled)):
        raise ValueError("Analysis frames must have unique ordered original indices")
    wanted = {row["frame"]: row for row in sampled}
    verified = set()
    def inspect_frame(index, frame, timestamp):
        check()
        if index not in wanted:
            return
        row = wanted[index]
        if row["decoder_time_s"] != timestamp or row["nominal_time_s"] != index / frame_rows["fps"]:
            raise ValueError("Analysis frame times differ from decoded source metadata")
        scale = min(1, 512 / max(frame.shape[:2]))
        thumb = cv2.resize(frame, (max(1, round(frame.shape[1]*scale)), max(1, round(frame.shape[0]*scale))))
        ok, encoded = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok or encoded.tobytes() != analysis_files[row["artifact"]]:
            raise ValueError("Analysis image differs from decoded source pixels")
        verified.add(index)
    prepared = None
    if preflight is not None:
        report("validating_preflight", "Reusing sealed local body evidence; no repeated pose inference")
        prepared = validate_perception_preflight(video, preflight, deadline=deadline)
    metadata = inspect_video(video, deadline=deadline, frame_inspector=inspect_frame, include_fps=True)
    if prepared is not None and any(metadata[k] != prepared["metadata"][k]
            for k in ("source_sha256", "width", "height", "frame_count", "fps", "decoder_timestamps_s")):
        raise ValueError("Analyzed source differs from the reusable preflight decoder metadata")
    if (any(metadata[k] != proposal[k] for k in ("source_sha256", "width", "height", "frame_count"))
            or metadata["fps"] != frame_rows["fps"] or verified != set(wanted)):
        raise ValueError("Tracking video differs from the exact analyzed source; implicit conversion is unsupported")
    estimate = max(.15, max(50, metadata["frame_count"]) * (len(proposal["boxes"])+int(needs_body)) / 5000)
    if execute_tracking and (type(max_meta_cost_usd) not in (int, float) or not math.isfinite(max_meta_cost_usd)
                             or max_meta_cost_usd < estimate):
        raise ValueError(f"Explicit tracking allowance must cover ${estimate:.6f}; shared ledger also applies")
    def check_tracking(document, boxes, *, equipment_only=False):
        initialized = sorted([[int(row["raw_model_id"]), row["initialization"]["frame"], *row["initialization"]["roi"]]
                              for row in document["objects"]])
        if (document["source"]["submitted_video_sha256"] != proposal["source_sha256"]
                or (initialized[:len(boxes)] != boxes if equipment_only else initialized != boxes)
                or equipment_only and len(initialized) != len(boxes)+1):
            raise ValueError("Saved tracking source or initialization differs from the proposed AI regions")
    tracking_pin = None
    saved = None
    if saved_tracking is not None:
        from weft.segmentation_evidence import adapt_tracking_run
        saved = adapt_tracking_run(saved_tracking)
        check_tracking(saved, proposal["boxes"], equipment_only=True)
        tracking_pin = digest(_path(saved_tracking) / "manifest.json")
    if (execute_tracking or saved_tracking is not None) and prepared is None and body_source is None and pose_model is None:
        raise ValueError("Executing perception requires a local pose model or exact sealed body source")
    check(); out.mkdir(parents=True)
    write_package_files(out, {"analysis/"+name: raw for name, raw in analysis_files.items()})
    plan = {"profile": PROFILE, "source_sha256": proposal["source_sha256"], "prompts": proposal,
            "scope": SCOPE, "maximum_new_api_calls": int(execute_tracking), "retries": 0,
            "source_frame_verification": {"frame_indices": sorted(verified), "all_sampled_pixels_match": True,
                "decoder_times_match": True, "metadata_fps": metadata["fps"],
                "scope": "Source metadata and sampled pixels only; native body timing remains independent and authoritative."},
            "tracking_reservation_usd": estimate, "maximum_seconds": max_seconds,
            "source_format": "original MP4 only; no implicit conversion or timing rewrite"}
    write_json(out / "plan.json", plan)
    if prepared is not None:
        preflight_path = _path(preflight)
        descriptor = {"profile": "exercise-perception-preflight-reference-v1", "preflight_directory": str(preflight_path),
            "preflight_manifest_sha256": prepared["manifest_sha256"], "base_run": prepared["base_run"],
            "base_manifest_sha256": prepared["base_manifest_sha256"],
            "metadata_sha256": digest(preflight_path/"metadata.json"), "body_anchor_sha256": digest(preflight_path/"body-anchor.json")}
        write_json(out/"preflight-descriptor.json", descriptor)
    stage = "prepare_tracking"
    try:
        if not needs_body:
            track_video(video, out / "tracking", proposal["boxes"], execute=False, project_root=project_root, deadline=deadline)
            result = {"status": "prepared_not_sent", "tracking": str(out / "tracking"), "api_calls": 0,
                      "body_anchor_status": "required_before_assembly_or_paid_tracking", "assembly_inputs_ready": False}
        else:
            stage = "body"
            report("preparing_body", "Checking source-bound body and person tracking anchor")
            from weft.shared_world_lab import prepare_shared_world_seed
            from weft.exercise_body_source import prepare_exercise_body_source

            if prepared is None and body_source is None:
                body_source = out / "body_source"
                prepare_exercise_body_source(video, body_source, pose_model=pose_model, native_timing=True,
                    source_note="Exact source from saved AI analysis; source rights remain uploader responsibility",
                    max_seconds=min(180, deadline-time.monotonic()))
            seed = (Path(prepared["base_run"]) if prepared is not None else
                    prepare_shared_world_seed(body_source, out / "body_seed", deadline=deadline))
            scene = _json(read_package_file(seed, "scene.json", max_bytes=16 << 20))
            if scene["source_binding"]["source_sha256"] != proposal["source_sha256"]:
                raise ValueError("Body source differs from the tracked original video")
            anchor_profile = prepared["body_anchor"]["profile"] if prepared is not None else LEGACY_ANCHOR_PROFILE
            anchor = body_tracking_anchor(seed, scene, proposal,
                **({"anchor_profile": anchor_profile} if anchor_profile != LEGACY_ANCHOR_PROFILE else {}))
            if prepared is not None:
                expected_anchor = {**prepared["body_anchor"], "prompt": [len(proposal["boxes"])+1, *prepared["body_anchor"]["prompt"][1:]]}
                if anchor != expected_anchor:
                    raise ValueError("Reusable preflight person ROI changed after AI analysis")
            boxes = [*proposal["boxes"], anchor["prompt"]]
            write_json(out / "body-anchor.json", anchor)
            if saved_tracking is not None:
                check_tracking(saved, boxes)
            check(); stage = "tracking"
            report("preparing_tracking", "Preparing exact original video and source-bound tracking regions")
            if saved_tracking is None:
                saved_tracking = out / "tracking"
                report("tracking" if execute_tracking else "preparing_tracking", "One bounded tracking request" if execute_tracking else "Preparing offline request; nothing sent")
                track_video(video, saved_tracking, boxes, execute=execute_tracking,
                            max_cost_usd=max_meta_cost_usd, project_root=project_root, deadline=deadline)
                tracking_pin = digest(saved_tracking / "manifest.json")
            if not execute_tracking and saved is None:
                result = {"status": "prepared_not_sent", "tracking": str(saved_tracking), "api_calls": 0,
                          "body_anchor_status": "sealed_pose_tracking_hypothesis", "assembly_inputs_ready": False}
            else:
                check(); stage = "tracking_evidence"
                report("exporting_evidence", "Preserving raw tracking slots, source clocks and unresolved identities")
                from weft.segmentation_observations import export_tracking_evidence
                export_tracking_evidence(saved_tracking, out / "equipment_evidence")
                check_tracking(_json(read_package_file(out / "equipment_evidence", "segmentation-evidence.json", max_bytes=64_000_000)), boxes)
                if digest(_path(saved_tracking) / "manifest.json") != tracking_pin:
                    raise ValueError("Tracking package changed during export")
                check()
                result = {"status": "evidence_ready", "base_run": str(seed), "evidence_run": str(out / "equipment_evidence"),
                          "source_mappings": [], "api_calls": int(execute_tracking), "assembly_inputs_ready": True,
                          "body_anchor_status": "sealed_pose_tracking_hypothesis",
                          "tracking_manifest_sha256": tracking_pin,
                          "tracking_origin": "new_request" if execute_tracking else "exact_saved_tracking"}
        if digest(video) != proposal["source_sha256"]:
            raise ValueError("Original video changed during perception")
        check()
        if prepared is not None:
            if (digest(_path(preflight)/"manifest.json") != prepared["manifest_sha256"]
                    or digest(Path(prepared["base_run"])/"manifest.json") != prepared["base_manifest_sha256"]):
                raise ValueError("Reusable preflight or body seed changed during perception")
            result["preflight_descriptor"] = "preflight-descriptor.json"
        result.update(profile=PROFILE, scope=SCOPE, source_sha256=proposal["source_sha256"],
                      seconds=time.monotonic()-started, physics_steps=0, simulation_ready=False)
        write_json(out / "result.json", result)
        write_package_files(out, {"implementation/exercise_perception_pipeline.py": Path(__file__).read_bytes()})
        names = ["plan.json", "result.json", "implementation/exercise_perception_pipeline.py",
                 *("analysis/"+name for name in analysis_files)]
        if 'analysis_profile' in proposal:
            helper = 'implementation/equipment_analysis_contract.py'
            write_package_files(out, {helper: Path(__file__).with_name('equipment_analysis_contract.py').read_bytes()})
            names.append(helper)
        if (out / "body-anchor.json").is_file():
            names.append("body-anchor.json")
        if prepared is not None:
            names.append("preflight-descriptor.json")
        children = [name for name in ("body_source", "body_seed", "tracking", "equipment_evidence")
                    if (out/name/"manifest.json").is_file()]
        write_json(out / "manifest.json", {"schema_version": 1, "profile": PROFILE, "status": result["status"],
            "seal_scope": "bridge metadata and independent child manifest pins; child packages retain their complete seals",
            "files": {name: digest(out/name) for name in names},
            "children": {name: digest(out/name/"manifest.json") for name in children}})
        return result
    except Exception as error:
        write_json(out / "failure.json", {"status": "failed", "stage": stage, "error_type": type(error).__name__,
            "message": str(error)[:500], "retry": False, "seconds": time.monotonic()-started,
            "scope": "Partial source evidence and any provider reservation remain preserved"})
        raise
