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


def body_tracking_anchor(seed, scene, proposal):
    """A sealed pose supplies a tracking hint, never a rigid person or verified identity."""
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
    analysis = _path(analysis)
    limits = {"frames.json": 32 << 10, "plan.json": 16 << 10, "request.json": 64 << 10,
              "response.json": 128 << 10, "receipt.json": 32 << 10}
    files = {name: read_package_file(analysis, name, max_bytes=limit) for name, limit in limits.items()}
    frames, plan, request, response, receipt = (_json(files[name]) for name in limits)
    validate_plan(plan)
    if (plan["route"] != "exercise" or plan["camera"] != "fixed" or plan["people"] != 1
            or plan.get("exercise_inventory") != "equipment_or_obstacles" or not plan["objects"]):
        raise ValueError("Saved analysis must propose one fixed-camera person and explicit first-frame equipment regions")
    if (not isinstance(frames.get("source_sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", frames["source_sha256"])
            or not isinstance(frames.get("frames"), list) or not 1 <= len(frames["frames"]) <= 6
            or frames["frames"][0].get("frame") != 0
            or any(type(frames.get(k)) is not int for k in ("width", "height", "decoded_frame_count"))
            or type(frames.get("fps")) not in (int, float) or not math.isfinite(frames["fps"]) or not 1 <= frames["fps"] <= 120
            or not 12 <= frames["decoded_frame_count"] <= 500
            or not 16 <= frames["width"] <= 4096 or not 16 <= frames["height"] <= 4096):
        raise ValueError("Equipment tracking needs original first-frame analysis within 500 decoded frames")
    payload = {k: v for k, v in request.items() if k not in {"images", "request_sha256", "scope"}}
    if (request.get("images") != frames["frames"] or payload.get("store") is not False
            or payload.get("model") != MODEL or payload.get("instructions") != INSTRUCTIONS
            or payload.get("max_output_tokens") != MAX_OUTPUT_TOKENS
            or payload.get("text") != {"format": {"type": "json_schema", "name": "weft_scene_plan", "strict": True, "schema": SCHEMA}}
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
        boxes.append([index, 0, left, top, right-left, bottom-top])
    return {"source_sha256": frames["source_sha256"], "width": width, "height": height,
            "frame_count": frames["decoded_frame_count"], "boxes": boxes, "objects": plan["objects"],
            "request_sha256": request_hash, "region_basis": "AI first-image localization, not verified identity",
            "coordinate_conversion": "normalized endpoints to source pixels with outward rounding; no padding",
            "input_sha256": {name: sha256(raw).hexdigest() for name, raw in files.items()}}, files


def prepare_exercise_perception(video, analysis, out, *, pose_model=None, body_source=None,
                                saved_tracking=None, execute_tracking=False, max_meta_cost_usd=None,
                                project_root=None, max_seconds=420):
    """Prepare offline by default; explicit execution reuses the canonical one-call Meta adapter."""
    from weft.segmentation_lab import inspect_video, track_video

    if type(execute_tracking) is not bool or (execute_tracking and saved_tracking is not None):
        raise ValueError("Choose one new tracking request or a saved exact-source tracking run")
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= 420:
        raise ValueError("Perception preparation requires a 1–420 second bound")
    started = time.monotonic(); deadline = started + max_seconds
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("Exercise perception exceeded its work deadline")
    video, analysis, out = map(_path, (video, analysis, out))
    inputs = [video, analysis, *(_path(v) for v in (body_source, saved_tracking, pose_model) if v is not None)]
    if out.exists() or any(out == p or out.is_relative_to(p) or p.is_relative_to(out) for p in inputs):
        raise ValueError("Choose a new perception directory outside every input")
    proposal, analysis_files = read_equipment_analysis(analysis)
    needs_body = execute_tracking or saved_tracking is not None or body_source is not None
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
    metadata = inspect_video(video, deadline=deadline, frame_inspector=inspect_frame, include_fps=True)
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
    if (execute_tracking or saved_tracking is not None) and body_source is None and pose_model is None:
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
    stage = "prepare_tracking"
    try:
        if not needs_body:
            track_video(video, out / "tracking", proposal["boxes"], execute=False, project_root=project_root, deadline=deadline)
            result = {"status": "prepared_not_sent", "tracking": str(out / "tracking"), "api_calls": 0,
                      "body_anchor_status": "required_before_assembly_or_paid_tracking", "assembly_inputs_ready": False}
        else:
            stage = "body"
            from weft.shared_world_lab import prepare_shared_world_seed
            from weft.exercise_body_source import prepare_exercise_body_source

            if body_source is None:
                body_source = out / "body_source"
                prepare_exercise_body_source(video, body_source, pose_model=pose_model, native_timing=True,
                    source_note="Exact source from saved AI analysis; source rights remain uploader responsibility",
                    max_seconds=min(180, deadline-time.monotonic()))
            seed = prepare_shared_world_seed(body_source, out / "body_seed", deadline=deadline)
            scene = _json(read_package_file(seed, "scene.json", max_bytes=16 << 20))
            if scene["source_binding"]["source_sha256"] != proposal["source_sha256"]:
                raise ValueError("Body source differs from the tracked original video")
            anchor = body_tracking_anchor(seed, scene, proposal)
            boxes = [*proposal["boxes"], anchor["prompt"]]
            write_json(out / "body-anchor.json", anchor)
            if saved_tracking is not None:
                check_tracking(saved, boxes)
            check(); stage = "tracking"
            if saved_tracking is None:
                saved_tracking = out / "tracking"
                track_video(video, saved_tracking, boxes, execute=execute_tracking,
                            max_cost_usd=max_meta_cost_usd, project_root=project_root, deadline=deadline)
                tracking_pin = digest(saved_tracking / "manifest.json")
            if not execute_tracking and saved is None:
                result = {"status": "prepared_not_sent", "tracking": str(saved_tracking), "api_calls": 0,
                          "body_anchor_status": "sealed_pose_tracking_hypothesis", "assembly_inputs_ready": False}
            else:
                check(); stage = "tracking_evidence"
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
        result.update(profile=PROFILE, scope=SCOPE, source_sha256=proposal["source_sha256"],
                      seconds=time.monotonic()-started, physics_steps=0, simulation_ready=False)
        write_json(out / "result.json", result)
        write_package_files(out, {"implementation/exercise_perception_pipeline.py": Path(__file__).read_bytes()})
        names = ["plan.json", "result.json", "implementation/exercise_perception_pipeline.py",
                 *("analysis/"+name for name in analysis_files)]
        if (out / "body-anchor.json").is_file():
            names.append("body-anchor.json")
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
