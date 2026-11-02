"""Bounded image/video decoding and deterministic media packaging."""

import hashlib
import math
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

from avatar_assets.safety import Deadline, LIMITS, local_file, require, sha256

Image.MAX_IMAGE_PIXELS = LIMITS["pixels"]


def image_info(path):
    path = local_file(path)
    with Image.open(path) as im:
        require(im.width * im.height <= LIMITS["pixels"], "Image pixel limit exceeded")
        require(getattr(im, "n_frames", 1) == 1, "Expected a still image, not an animation")
        require(im.format in {"PNG", "WEBP", "JPEG"}, "Unsupported still-image format")
        im.load()
        extrema = im.convert("RGBA").getchannel("A").getextrema()
        return {"kind": "image", "format": im.format, "width": im.width, "height": im.height,
                "mode": im.mode, "frames": 1, "has_transparency": extrema[0] < 255,
                "alpha_extrema": list(extrema), "bytes": path.stat().st_size,
                "sha256": sha256(path), "metadata_keys": sorted(im.info)}


def thumbnail(im, size=160):
    out = im.convert("RGB")
    out.thumbnail((size, size))
    return out


def contact_sheet(images, labels, path, columns=4):
    require(0 < len(images) <= 48, "Contact sheet needs 1–48 images")
    width, height = 192, 216
    sheet = Image.new("RGB", (width * columns, height * math.ceil(len(images) / columns)), "#eeeeee")
    pen = ImageDraw.Draw(sheet)
    for i, (im, label) in enumerate(zip(images, labels)):
        tile = thumbnail(im, 184)
        x, y = i % columns * width, i // columns * height
        sheet.paste(tile, (x + (width - tile.width) // 2, y))
        pen.text((x + 5, y + 188), str(label)[:30], fill="black")
    sheet.save(path)


def video_frames(path):
    path = local_file(path)
    deadline = Deadline()
    # No playlists or network protocols: the demuxer sees only a bounded local handle.
    with path.open("rb") as handle, av.open(handle, mode="r", format="mp4") as container:
        require(len(container.streams.video) == 1, "Expected exactly one video stream")
        require(not container.streams.audio, "Silent avatar video required")
        require(len(container.streams) == 1, "Unexpected media streams")
        stream = container.streams.video[0]
        stream.thread_type = "NONE"
        stream.codec_context.thread_count = 1
        require(0 < stream.width * stream.height <= LIMITS["video_pixels"], "Video pixel limit exceeded")
        require(stream.codec_context.name == "h264", "Only H.264 input video is supported")
        if stream.frames:
            require(stream.frames <= LIMITS["frames"], "Video frame limit exceeded")
        if stream.duration is not None:
            require(float(stream.duration * stream.time_base) <= LIMITS["seconds"], "Video duration limit exceeded")
        previous = None
        for i, frame in enumerate(container.decode(stream)):
            deadline.check()
            require(i < LIMITS["frames"], "Decoded frame limit exceeded")
            require(frame.width * frame.height <= LIMITS["video_pixels"], "Decoded dimensions too large")
            stamp = float(frame.pts * frame.time_base) if frame.pts is not None else None
            require(stamp is not None and math.isfinite(stamp), "Missing frame timestamps")
            require(stamp >= 0 and (previous is None or stamp > previous), "Nonmonotonic video timestamps")
            require(stamp <= LIMITS["seconds"], "Decoded duration limit exceeded")
            previous = stamp
            yield frame.to_image(), stamp


def video_info(path, contact_path=None):
    path = local_file(path)
    times, deltas, signatures = [], [], set()
    first = previous = last = None
    sample = {}
    sample_stride = 1
    dimensions = None
    for i, (im, stamp) in enumerate(video_frames(path)):
        require(dimensions in (None, im.size), "Video dimensions changed")
        dimensions = im.size
        small = np.asarray(im.resize((64, 64)), dtype=np.float32) / 255
        signatures.add(hashlib.sha256(small.tobytes()).hexdigest())
        if first is None:
            first = small
        if previous is not None:
            deltas.append(float(np.abs(small - previous).mean()))
        previous, last = small, small
        times.append(stamp)
        # Adaptive spacing keeps short-clip interiors without retaining every frame.
        if i % sample_stride == 0:
            sample[i] = thumbnail(im)
            if len(sample) > 24:
                sample_stride *= 2
                sample = {j: thumb for j, thumb in sample.items() if j % sample_stride == 0}
        last_thumb = thumbnail(im)
    require(len(times) >= 2, "A video must contain at least two decoded frames")
    sample[len(times) - 1] = last_thumb
    intervals = np.diff(times)
    fps = 1 / float(np.median(intervals))
    duration = times[-1] - times[0] + float(np.median(intervals))
    require(0 < fps <= LIMITS["fps"] + 0.01, "Frame rate exceeds limit")
    require(duration <= LIMITS["seconds"] + 0.001, "Duration exceeds limit")
    require(max(abs(intervals - np.median(intervals))) < 0.002, "Variable frame rate is unsupported")
    seam = float(np.abs(last - first).mean())
    with path.open("rb") as h, av.open(h, format="mp4") as c:
        codec = c.streams.video[0].codec_context
        metadata = {k: str(v)[:2048] for k, v in list(c.metadata.items())[:32]}
        pixel_format = codec.format.name
    selected = sorted(sample)
    if len(selected) > 12:
        selected = [selected[i] for i in np.linspace(0, len(selected) - 1, 12).astype(int)]
    if contact_path:
        contact_sheet([sample[i] for i in selected], [f"frame {i} / {times[i]:.2f}s" for i in selected], contact_path)
    return {"kind": "video", "format": "MP4", "codec": "h264", "pixel_format": pixel_format,
            "width": dimensions[0], "height": dimensions[1], "frames": len(times),
            "fps": round(fps, 6), "duration_seconds": round(duration, 6), "audio": False,
            "has_transparency": False, "unique_thumbnails": len(signatures),
            "motion_detected": len(signatures) > 1 and max(deltas) > 0.0001,
            "adjacent_mae_mean": round(float(np.mean(deltas)), 6),
            "adjacent_mae_max": round(max(deltas), 6), "boundary_mae": round(seam, 6),
            "loop": "unverified", "metadata": metadata,
            "contact_sample_frames": selected,
            "bytes": path.stat().st_size, "sha256": sha256(path)}


def inspect(path, contact_path=None):
    if Path(path).suffix.lower() == ".mp4":
        return video_info(path, contact_path)
    return image_info(path)


def fit(im, width, height, background):
    require(im.width * height == im.height * width, "Aspect ratio differs; explicit crop/artwork revision required")
    out = im.convert("RGBA").resize((width, height), Image.Resampling.LANCZOS)
    if background != "transparent":
        canvas = Image.new("RGBA", out.size, background)
        canvas.alpha_composite(out)
        return canvas.convert("RGB")
    return out


def encode_video(frames, path, fps, width, height, background):
    require(type(fps) is int and 1 <= fps <= LIMITS["fps"], "Output frame rate exceeds limit")
    require(type(width) is int and type(height) is int and width > 0 and height > 0
            and width % 2 == height % 2 == 0 and width * height <= LIMITS["video_pixels"],
            "Output video dimensions invalid or too large")
    require(background != "transparent", "MP4/H.264 cannot preserve alpha; use opaque background or PNG cels")
    deadline = Deadline()
    with av.open(str(path), "w", format="mp4", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.codec_context.thread_count = 1
        stream.options = {"crf": "18", "preset": "medium", "threads": "1"}
        count = 0
        for count, im in enumerate(frames, 1):
            deadline.check()
            require(count <= LIMITS["frames"], "Output frame limit exceeded")
            frame = av.VideoFrame.from_image(fit(im, width, height, background))
            frame.pts, frame.time_base = count - 1, Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        require(count >= 2, "Animation requires multiple frames")
        for packet in stream.encode():
            container.mux(packet)


def tools_info():
    import platform
    import PIL
    return {"python": platform.python_version(), "pillow": PIL.__version__, "pyav": av.__version__,
            "numpy": np.__version__, "ffmpeg_libraries": {k: list(v) for k, v in av.library_versions.items()},
            "encoder": "libx264", "encoder_options": {"crf": 18, "preset": "medium", "threads": 1}}
