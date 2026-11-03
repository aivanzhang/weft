from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from weft import portal_body_detail_recipe as recipes
from weft.io import digest, write_json
from test_person_region_proposal import documents, encoded, competing
from test_exercise_perception_pipeline import local_preflight_source


PARENT = {"project_id": "1"*32, "job_sha256": "2"*64, "status_sha256": "3"*64,
          "preflight_failure_sha256": "4"*64}


def test_complete_union_fixed_recipe_and_detached_validation():
    e, r = documents(); competing(e, r); pair = encoded(e, r)
    recipe = recipes.build_body_detail_recipe(*pair, parent=PARENT)
    assert recipe["region_proposal"]["histories"][0]["hypotheses"] == 13
    assert recipe["region_proposal"]["region_xyxy"][2] > 576
    assert recipe["model"]["sha256"] == recipes.MODEL_SHA256
    assert recipes.body_detail_options(recipe) == {"pose_mode": "video", "pose_identity": "raw",
        "max_pose_hypotheses": 2, "pose_sampling": "all_frames", "native_timing": True,
        "person_region": recipe["region_proposal"]["region_xyxy"]}
    detached = recipes.validate_body_detail_recipe(recipe, *pair, parent=PARENT)
    detached["parent"]["status_sha256"] = "e"*64
    assert recipe["parent"] == PARENT


@pytest.mark.parametrize("change", ["region", "model", "sampling", "capacity", "parent", "extra", "raw", "evidence"])
def test_recipe_cannot_select_region_model_identity_or_altered_source(change):
    pair = encoded(*documents()); recipe = recipes.build_body_detail_recipe(*pair, parent=PARENT)
    if change == "region": recipe["region_proposal"]["region_xyxy"][0] += 1
    elif change == "model": recipe["model"]["sha256"] = "f"*64
    elif change == "sampling": recipe["inference"]["pose_sampling"] = "bounded"
    elif change == "capacity": recipe["inference"]["max_pose_hypotheses"] = 1
    elif change == "parent": recipe["parent"]["status_sha256"] = "f"*64
    elif change == "extra": recipe["client_path"] = "/tmp/model"
    elif change == "raw": pair = pair[0], pair[1]+b" "
    else: pair = pair[0]+b" ", pair[1]
    with pytest.raises(ValueError): recipes.validate_body_detail_recipe(recipe, *pair, parent=PARENT)


@pytest.mark.parametrize("parent", [{}, {**PARENT, "path": "/tmp"}, {**PARENT, "project_id": "A"*32}])
def test_parent_closed_pins(parent):
    with pytest.raises(ValueError): recipes.build_body_detail_recipe(*encoded(*documents()), parent=parent)


def test_raw_complete_bounds_and_no_native_imports():
    pair = encoded(*documents())
    with pytest.raises(ValueError): recipes.build_body_detail_recipe(pair[0], b" "*(16 << 20), parent=PARENT)
    result = subprocess.run([sys.executable, "-c", "import sys; import weft.portal_body_detail_recipe; "
        "assert not any(x in sys.modules for x in ('mujoco','cv2','mediapipe','weft.human','weft.exercise_vision'))"],
        capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr.decode()


@pytest.fixture
def detail_pipeline(tmp_path, monkeypatch):
    """Fake inference/seed only; exercise real recipe, option dispatch and sealed reader."""
    from weft import exercise_perception_pipeline as bridge
    import weft.exercise_body_source as bodies
    import weft.segmentation_lab as segmentation
    import weft.shared_world_lab as worlds
    video = tmp_path/"source.mp4"; video.write_bytes(b"analytical input, not a decoded video")
    e, r = documents(); e["source_sha256"] = digest(video)
    for document in (e, r):
        for row in document["frames"]: row["id"] = f"{digest(video)}:frame:{row['frame']}"
    pair = encoded(e, r)
    model = tmp_path/recipes.MODEL_FILENAME; model.write_bytes(b"fake local model, no inference")
    monkeypatch.setattr(recipes, "MODEL_SHA256", digest(model)); monkeypatch.setattr(recipes, "MODEL_BYTES", model.stat().st_size)
    recipe = recipes.build_body_detail_recipe(*pair, parent=PARENT)
    metadata = {"source_sha256": digest(video), "width": 640, "height": 480, "fps": 30,
                "frame_count": 12, "decoder_timestamps_s": [i/30 for i in range(12)]}
    monkeypatch.setattr(segmentation, "inspect_video", lambda *a, **k: deepcopy(metadata))
    calls = []
    def body(source, out, **options):
        calls.append(options); out.mkdir()
        pose_options = {k: options[k] for k in ("pose_mode", "pose_identity", "pose_sampling", "person_region")}
        write_json(out/"body-source.json", {"source_sha256": digest(source), "pose_options": pose_options,
            "construction": {"clock_profile": "source-native-clock-v1"}})
        write_json(out/"model.json", {"sha256": recipes.MODEL_SHA256, "bytes": recipes.MODEL_BYTES})
        write_json(out/"manifest.json", {"files": {name: digest(out/name) for name in ("body-source.json", "model.json")}})
        return out
    def seed(source, out, **options):
        path=out/"inputs/human-source/source-manifest.json"; path.parent.mkdir(parents=True)
        path.write_bytes((source/"manifest.json").read_bytes())
        write_json(out/"manifest.json", {"files": {"inputs/human-source/source-manifest.json": digest(path)}})
        return out
    monkeypatch.setattr(bodies, "prepare_exercise_body_source", body)
    monkeypatch.setattr(worlds, "prepare_shared_world_seed", seed)
    monkeypatch.setattr(bridge, "_preflight_seed", lambda seed, *a, **k:
        (digest(seed/"manifest.json"), {"profile": bridge.LEGACY_ANCHOR_PROFILE, "prompt": [1, 0, 0, 1, 1]}))
    return bridge, video, model, pair, recipe, metadata, calls


def prepare(detail_pipeline, out):
    bridge, video, model, pair, recipe, _, _ = detail_pipeline
    return bridge.prepare_exercise_perception_preflight(video, out, pose_model=model,
        body_detail_recipe=recipe, body_detail_inputs=pair)


def reseal(out):
    manifest = json.loads((out/"manifest.json").read_bytes())
    manifest["files"] = {name: digest(out/name) for name in manifest["files"]}
    manifest["children"] = {name: digest(out/name/"manifest.json") for name in manifest["children"]}
    write_json(out/"manifest.json", manifest)


def test_detail_dispatch_retains_raw_recipe_and_revalidates_without_inference(detail_pipeline, tmp_path):
    bridge, video, _, pair, recipe, _, calls = detail_pipeline
    out = tmp_path/"preflight"; result = prepare(detail_pipeline, out)
    assert len(calls) == 1
    assert all(calls[0][k] == v for k, v in recipes.body_detail_options(recipe).items())
    assert [(out/name).read_bytes() for name in recipes.INPUT_FILES] == list(pair)
    assert bridge.validate_perception_preflight(video, out, deadline=time.monotonic()+10) == result
    assert len(calls) == 1 and result["inference_calls"] == 1 and result["api_calls"] == 0


@pytest.mark.parametrize("change", ["recipe", "producer", "helper", "body", "model", "raw", "other_seed"])
def test_resealed_detail_tampering_rejected(detail_pipeline, tmp_path, change):
    bridge, video, _, _, _, _, _ = detail_pipeline
    out = tmp_path/"preflight"; prepare(detail_pipeline, out)
    if change == "recipe":
        value=json.loads((out/recipes.RECIPE_FILE).read_bytes()); value["inference"]["max_pose_hypotheses"]=1
        write_json(out/recipes.RECIPE_FILE,value)
    elif change == "producer":
        (out/"implementation/exercise_perception_pipeline.py").write_bytes(
            (Path(__file__).parent/"fixtures/perception/preflight-producer-d20a.py.txt").read_bytes())
    elif change == "helper": (out/"implementation/person_region_proposal.py").write_bytes(b"changed")
    elif change in {"body", "model"}:
        name = "body-source.json" if change == "body" else "model.json"
        value=json.loads((out/"body_source"/name).read_bytes())
        if change == "body": value["pose_options"]["pose_sampling"]="bounded"
        else: value["sha256"]="f"*64
        write_json(out/"body_source"/name,value)
        marker=json.loads((out/"body_source/manifest.json").read_bytes()); marker["files"][name]=digest(out/"body_source"/name)
        write_json(out/"body_source/manifest.json",marker)
        original=out/"body_seed/inputs/human-source/source-manifest.json"
        original.write_bytes((out/"body_source/manifest.json").read_bytes())
        write_json(out/"body_seed/manifest.json",{"files": {"inputs/human-source/source-manifest.json": digest(original)}})
    elif change == "other_seed":
        path=out/"body_seed/inputs/human-source/source-manifest.json"; write_json(path,{"files": {"other-valid-body": "a"*64}})
        write_json(out/"body_seed/manifest.json",{"files": {"inputs/human-source/source-manifest.json": digest(path)}})
    else: (out/recipes.INPUT_FILES[1]).write_bytes((out/recipes.INPUT_FILES[1]).read_bytes()+b" ")
    reseal(out)
    with pytest.raises(ValueError): bridge.validate_perception_preflight(video,out,deadline=time.monotonic()+10)


@pytest.mark.parametrize("change", ["model", "source", "clock", "body_override", "missing_pair"])
def test_detail_refusals_before_output_or_inference(detail_pipeline, tmp_path, change):
    bridge, video, model, pair, recipe, metadata, calls = detail_pipeline
    options={"pose_model":model,"body_detail_recipe":recipe,"body_detail_inputs":pair}
    if change == "model": model.write_bytes(b"wrong model")
    elif change == "source": metadata["source_sha256"]="f"*64
    elif change == "clock":
        e,r=map(json.loads,pair); e["frames"][0]["decoder_timestamp_s"]=.001; r["frames"]=deepcopy(e["frames"])
        pair=encoded(e,r); options.update(body_detail_inputs=pair,body_detail_recipe=recipes.build_body_detail_recipe(*pair,parent=PARENT))
    elif change == "body_override": options["body_source"]=tmp_path/"body"
    else: options["body_detail_inputs"]=None
    with pytest.raises(ValueError): bridge.prepare_exercise_perception_preflight(video,tmp_path/"out",**options)
    assert not calls and not (tmp_path/"out").exists()


def test_detail_failure_retains_original_pair_and_no_seal(detail_pipeline, tmp_path, monkeypatch):
    import weft.exercise_body_source as bodies
    def fail(*a, **k): raise ValueError("unchanged original quality gate")
    monkeypatch.setattr(bodies, "prepare_exercise_body_source", fail)
    out=tmp_path/"preflight"
    with pytest.raises(ValueError,match="quality gate"): prepare(detail_pipeline,out)
    assert (out/"failure.json").is_file() and not (out/"manifest.json").exists()
    assert [(out/name).read_bytes() for name in recipes.INPUT_FILES] == list(detail_pipeline[3])


def test_previous_published_preflight_keeps_full_sealed_source_validation(local_preflight_source, tmp_path, monkeypatch):
    from weft import exercise_perception_pipeline as bridge
    import weft.segmentation_lab as segmentation
    video, body = local_preflight_source; out = tmp_path/"preflight"
    bridge.prepare_exercise_perception_preflight(video, out, body_source=body)
    old = Path(__file__).parent/"fixtures/perception/preflight-producer-d20a.py.txt"
    assert digest(old) == "d20a3c187762c5d1608bc2a793302b13f7e6a09992ea4a6a565952b0a016ed81"
    destination = out/"implementation/exercise_perception_pipeline.py"
    destination.write_bytes(old.read_bytes()); reseal(out)
    monkeypatch.setattr(segmentation, "inspect_video", lambda *a, **k: pytest.fail("No validation decode"))
    assert bridge.validate_perception_preflight(video,out,deadline=time.monotonic()+10)["status"] == "local_ready"
    destination.write_bytes(old.read_bytes()+b"\n# changed producer\n"); reseal(out)
    with pytest.raises(ValueError, match="explicit compatibility"):
        bridge.validate_perception_preflight(video,out,deadline=time.monotonic()+10)
