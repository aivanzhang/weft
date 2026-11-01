"""Render and encode avatar media while retaining sources and measured declarations."""

import shutil
import sys
from pathlib import Path

from avatar_assets.build3d import ROOT
from avatar_assets.runner import run_bounded
from avatar_assets.scene import read_scene
from avatar_assets.safety import artifacts, new_directory, preserve_failure, read_json, require, sha256, slug, write_json

PRESETS = {
    "muse": {"size": 720, "portrait": 1600, "seconds": 8, "samples": 8},
    "preview": {"size": 320, "portrait": 640, "seconds": 4, "samples": 8},
}


def read_render_scene(path):
    data = read_json(path)
    if data.get('kind') != 'generated_render':
        return read_scene(path)
    require(set(data) == {'schema_version', 'kind', 'id', 'subject', 'style', 'states'}
            and data['schema_version'] == 1, 'Invalid generated render description')
    slug(data['id'])
    require(all(isinstance(data[k], str) and 0 < len(data[k]) <= 4000 for k in ('subject', 'style')), 'Invalid render metadata')
    require(isinstance(data['states'], list) and 1 <= len(data['states']) <= 8, 'Invalid render states')
    seen = set()
    for state in data['states']:
        require(set(state) == {'id', 'meaning', 'duration', 'loop'}, 'Invalid generated render state')
        name = slug(state['id'])
        require(name not in seen and isinstance(state['meaning'], str) and 0 < len(state['meaning']) <= 4000,
                'Duplicate state or invalid meaning')
        seen.add(name)
        require(type(state['duration']) in (int, float) and 0 < state['duration'] <= 10 and state['loop'] is False,
                'Generated media requires bounded duration and no unverified loop claim')
    return data


def export_2d(parent, output, preset="muse", blender=None, size=None, portrait=None, seconds=None, samples=None,
              denoise_portrait=False, resume_frames=None):
    existed = Path(output).exists()
    try:
        return _export_2d(parent, output, preset, blender, size, portrait, seconds, samples, denoise_portrait, resume_frames)
    except Exception as error:
        if not existed:
            preserve_failure(output, error)
        raise


def reuse_render_frames(previous, root, out, spec):
    from avatar_assets.media import image_info
    prior = Path(previous).resolve()
    manifest, job = read_json(prior/'manifest.json'), read_json(prior/'job.json')
    require(manifest.get('status') == 'failed' and job.get('failure') == 'wall-time budget exceeded'
            and job.get('process_exit_verified') and not (prior/'reused-frames.json').exists(),
            'Only one continuation of an exited, time-bounded render is supported')
    require(manifest.get('artifacts') == artifacts(prior), 'Previous render artifacts changed')
    receipt = read_json(prior/'resume-input-receipt.json')
    require(receipt['parent_manifest_sha256'] == sha256(root/'manifest.json')
            and receipt['source_blend_sha256'] == sha256(root/'source.blend')
            and receipt['scene_sha256'] == sha256(out/'sources/scene.json')
            and receipt['spec'] == spec, 'Render source or settings differ; cannot reuse frames')
    records, incomplete = [], []
    for state in read_render_scene(out/'sources/scene.json')['states']:
        for frame in range(spec['seconds']*24+1):
            name = f"frames/{state['id']}/{frame:04d}.png"
            file = prior/name
            if not file.exists():
                continue
            try:
                info = image_info(file)
            except (ValueError, OSError):
                incomplete.append(name)
                continue
            require(info['width'] == info['height'] == spec['size'], 'Previous frame dimensions differ')
            target = out/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, target)
            records.append(name)
    require(records, 'No completed frames to continue; do not repeat the unchanged failed render')
    info = image_info(prior/'portrait.png')
    require(info['width'] == info['height'] == spec['portrait'], 'Previous portrait dimensions differ')
    shutil.copyfile(prior/'portrait.png', out/'portrait.png')
    records.append('portrait.png')
    write_json(out/'reused-frames.json', {'paths': records, 'incomplete_originals_retained': incomplete, 'previous': str(prior),
               'previous_manifest_sha256': sha256(prior/'manifest.json'), 'input_receipt': receipt,
               'remaining_worker_wall_seconds': 600, 'aggregate_render_wall_limit_seconds': 1800})


def _export_2d(parent, output, preset, blender, size, portrait, seconds, samples, denoise_portrait=False, resume_frames=None):
    from avatar_assets.cli import verify_pack
    verify_pack(parent)
    root = Path(parent).resolve()
    parent_manifest = read_json(root / "manifest.json")
    require(parent_manifest["kind"] in ("authored_3d_avatar", 'generated_3d_avatar'), "Expected a verified authored/generated 3D pack")
    require(parent_manifest["rig_verification"]["motion"]["animation_verified"], "Rig has no verified animation")
    spec = dict(PRESETS[preset])
    for k, v in (("size", size), ("portrait", portrait), ("seconds", seconds), ("samples", samples)):
        if v is not None:
            spec[k] = v
    require(all(type(value) is int for value in spec.values()), 'Render size, duration and samples must be integers')
    require(128 <= spec["size"] <= 1024 and spec["size"] % 2 == 0, "Video size outside 128–1024 even pixels")
    require(128 <= spec["portrait"] <= 2048, "Portrait outside 128–2048 pixels")
    require(1 <= spec["seconds"] <= 20 and 1 <= spec["samples"] <= 64, "Duration/samples exceed limit")
    require(type(denoise_portrait) is bool, "denoise_portrait must be a boolean")
    generated = parent_manifest['kind'] == 'generated_3d_avatar'
    scene = ({'schema_version': 1, 'kind': 'generated_render',
              **{key: parent_manifest[key] for key in ('id', 'subject', 'style')},
              'states': [{'id': state['id'], 'meaning': state['meaning'], 'duration': state['duration_seconds'], 'loop': False}
                         for state in parent_manifest['states']]} if generated else read_scene(root/'sources/scene.json'))
    out = new_directory(output)
    (out / "sources").mkdir()
    if generated:
        write_json(out/'sources/scene.json', scene)
        read_render_scene(out/'sources/scene.json')
        prompt = read_json(root/'sources/finish-design.json')['prompt']
        (out/'sources/prompt.txt').write_text(prompt+'\n')
    else:
        for name in ("scene.json", "prompt.txt"):
            shutil.copyfile(root / "sources" / name, out / "sources" / name)
    shutil.copyfile(ROOT / "scripts/blender_render.py", out / "sources/blender_render.py")
    shutil.copyfile(ROOT / "scripts/blender_verify.py", out / "sources/blender_verify.py")
    write_json(out/'sources/output-spec.json', spec)
    write_json(out/'resume-input-receipt.json', {'schema_version': 1,
               'capture_scope': 'Before starting the owned native render',
               'parent_manifest_sha256': sha256(root/'manifest.json'), 'source_blend_sha256': sha256(root/'source.blend'),
               'scene_sha256': sha256(out/'sources/scene.json'), 'spec': spec,
               'maximum_additional_render_workers': 1, 'additional_wall_limit_seconds': 600,
               'aggregate_render_wall_limit_seconds': 1800, 'new_provider_calls': 0})
    if resume_frames:
        require(generated, 'Frame continuation currently supports generated packs only')
        reuse_render_frames(resume_frames, root, out, spec)
    exe = Path(blender) if blender else ROOT / ".tools/Blender.app/Contents/MacOS/Blender"
    cmd = [str(exe.resolve()), "--background", "--factory-startup", "--disable-autoexec", "--python-exit-code", "1",
           "--threads", "4", "--python", str(out / "sources/blender_render.py"), "--",
           str(root / "source.blend"), str(out / "sources/scene.json"), str(out),
           str(spec["size"]), str(spec["portrait"]), str(spec["seconds"]), str(spec["samples"]), "false"]
    job = run_bounded(cmd, out / "render.log", out, timeout=600 if resume_frames else 1200, memory_mb=2048)
    write_json(out / "job.json", job)
    if job["exit_code"] or job["failure"] or not (out / "render-complete.json").exists():
        failure = {"schema_version": 1, "status": "failed", "job": job, "artifacts": artifacts(out)}
        write_json(out / "manifest.json", failure)
        raise ValueError(f"Render failed; attempt preserved at {out}")
    portrait_processing = None
    if denoise_portrait:
        from scripts.denoise_portrait import run as denoise
        shutil.copyfile(ROOT / "scripts/denoise_portrait.py", out / "sources/denoise_portrait.py")
        raw = out / "portrait-raw.png"
        shutil.copyfile(out / "portrait.png", raw)
        portrait_processing = denoise(raw, out / "portrait-denoise/portrait.png", out / "portrait-denoise/report.json")
        require(portrait_processing["worker"]["worker_source_sha256"] == sha256(out / "sources/denoise_portrait.py"),
                "Portrait worker changed after its source snapshot")
        # The raw render remains an artifact; only the pack's display portrait uses the filtered pixels.
        shutil.copyfile(out / "portrait-denoise/portrait.png", out / "portrait.png")
    shutil.copyfile(ROOT / "avatar_assets/packaging.py", out / "sources/packaging.py")
    package_job = run_bounded([sys.executable, "-m", "avatar_assets.packaging", str(out), str(out),
                              str(out / "sources/scene.json"), str(out / "sources/output-spec.json")],
                             out / "packaging.log", out, timeout=180, memory_mb=768)
    write_json(out / "packaging-job.json", package_job)
    if package_job["exit_code"] or package_job["failure"] or not (out / "packaging-result.json").exists():
        write_json(out / "manifest.json", {"schema_version": 1, "status": "failed", "job": job,
                   "packaging_job": package_job, "artifacts": artifacts(out)})
        raise ValueError(f"Packaging failed; attempt preserved at {out}")
    packaged = read_json(out / "packaging-result.json")
    result = {"schema_version": 1, "kind": "rendered_2d_avatar", "status": "needs_visual_review",
              "id": scene["id"], "subject": scene["subject"], "style": scene["style"], "preset": preset,
              "output_spec": spec, "origin": "Native 3D skeletal rendering at 24fps; not stock video or repeated stills",
              "parent_3d_manifest_sha256": sha256(root / "manifest.json"), "parent_3d": str(root),
              "portrait": packaged["portrait"], "clips": packaged["clips"], "job": job, "packaging_job": package_job,
              "has_transparency": False, "reduced_motion": "Use still PNG fallback; do not autoplay video",
              "portrait_postprocessing": portrait_processing,
              "provider_calls": 0, "provider_cost_usd": 0, "rejected_attempts": [],
              "rights": parent_manifest["rights"], "artifacts": artifacts(out)}
    write_json(out / "manifest.json", result)
    return result
