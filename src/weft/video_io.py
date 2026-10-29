"""Independent bounded container checks shared by video perception profiles."""

from pathlib import Path


def webm_video_frame_count(path):
    """Count bounded, unlaced video blocks independently of duration-derived metadata."""
    counts, types, laced = {}, {}, set()
    visited = 0
    with Path(path).open("rb") as stream:
        limit = Path(path).stat().st_size
        if limit > 100_000_000:
            raise ValueError("Container exceeds the 100 MB bound")

        def vint(identifier=False):
            first = stream.read(1)
            if not first:
                raise ValueError("Truncated EBML integer")
            first = first[0]
            width = next((n for n in range(1, 9) if first & (1 << (8 - n))), None)
            if width is None:
                raise ValueError("Invalid EBML integer")
            tail = stream.read(width - 1)
            if len(tail) != width - 1:
                raise ValueError("Truncated EBML integer")
            value = first if identifier else first & ((1 << (8 - width)) - 1)
            return int.from_bytes(bytes([value]) + tail, "big"), width

        def walk(end, depth=0, track=None):
            nonlocal visited
            if depth > 6:
                raise ValueError("Container nesting exceeds the audit bound")
            while stream.tell() < end:
                visited += 1
                if visited > 20_000:
                    raise ValueError("Container elements exceed the audit bound")
                identifier, _ = vint(True)
                size, width = vint()
                unknown = size == (1 << (7 * width)) - 1
                if unknown and identifier != 0x18538067:
                    raise ValueError("Only the segment may have unknown size in this audit")
                stop = end if unknown else stream.tell() + size
                if stop > end or stream.tell() > end:
                    raise ValueError("Truncated container element")
                if identifier in (0x18538067, 0x1654AE6B, 0x1F43B675, 0xA0):
                    walk(stop, depth + 1, track)
                elif identifier == 0xAE:
                    entry = {}
                    walk(stop, depth + 1, entry)
                    if "number" in entry:
                        types[entry["number"]] = entry.get("type")
                elif identifier in (0xD7, 0x83) and track is not None:
                    if not 1 <= size <= 8:
                        raise ValueError("Invalid track metadata")
                    track["number" if identifier == 0xD7 else "type"] = int.from_bytes(stream.read(size), "big")
                elif identifier in (0xA3, 0xA1):
                    number, _ = vint()
                    header = stream.read(3)
                    if len(header) != 3 or stream.tell() >= stop:
                        raise ValueError("Truncated video block")
                    if header[2] & 6:
                        laced.add(number)
                    counts[number] = counts.get(number, 0) + 1
                stream.seek(stop)

        walk(limit)
    video_tracks = [number for number, kind in types.items() if kind == 1]
    if len(video_tracks) != 1 or video_tracks[0] in laced:
        raise ValueError("Frame-count audit requires one unlaced video track")
    return counts.get(video_tracks[0], 0)
