"""Whitelist-only extraction; archive names and metadata remain untrusted data."""

import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath

from avatar_assets.media import inspect, tools_info
from avatar_assets.safety import LIMITS, artifacts, local_file, new_directory, require, sha256, write_json


def inspect_bounded(source, output, expected_sha256=None):
    from avatar_assets.runner import run_bounded
    from avatar_assets.safety import read_json
    destination = Path(output).resolve()
    require(not destination.exists(), "Output already exists; use a new folder")
    execution = new_directory(destination.with_name(destination.name + "-execution"))
    command = [sys.executable, "-c", "import sys; from avatar_assets.archive import inspect_archive; inspect_archive(*sys.argv[1:])",
               str(source), str(destination), expected_sha256 or ""]
    job = run_bounded(command, execution / "inspect.log", destination, timeout=180, memory_mb=1536)
    write_json(execution / "job.json", job)
    write_json(execution / "manifest.json", {"schema_version": 1, "kind": "inspection_execution",
               "status": "passed" if job["exit_code"] == 0 and not job["failure"] else "failed", "job": job, "artifacts": artifacts(execution)})
    require(job["exit_code"] == 0 and not job["failure"] and (destination / "manifest.json").exists(), f"Inspection failed; see {execution}")
    return read_json(destination / "manifest.json")


def inspect_archive(source, output, expected_sha256=None):
    source = local_file(source, LIMITS["archive_bytes"])
    digest = sha256(source)
    require(not expected_sha256 or digest == expected_sha256, "Archive SHA256 mismatch")
    with zipfile.ZipFile(source) as archive:
        entries = archive.infolist()
        require(0 < len(entries) <= LIMITS["archive_entries"], "Archive entry limit exceeded")
        seen, expanded = set(), 0
        for entry in entries:
            name = PurePosixPath(entry.filename)
            require(not name.is_absolute() and ".." not in name.parts and "\\" not in entry.filename,
                    "Unsafe archive path")
            require(len(name.parts) == 1 and not entry.is_dir(), "Only flat media archives are supported")
            require(name.suffix.lower() in {".webp", ".png", ".jpg", ".jpeg", ".mp4"}, "Archive contains nonmedia data")
            require(entry.filename.casefold() not in seen, "Duplicate archive filename")
            require(not (entry.flag_bits & 1), "Encrypted archive unsupported")
            mode = entry.external_attr >> 16
            require(stat.S_IFMT(mode) in (0, stat.S_IFREG), "Archive contains a link or special file")
            require(entry.file_size <= LIMITS["input_bytes"], "Archive member too large")
            require(entry.file_size / max(entry.compress_size, 1) <= LIMITS["compression_ratio"], "Archive compression ratio too high")
            expanded += entry.file_size
            require(expanded <= LIMITS["archive_expanded_bytes"], "Archive expanded byte limit exceeded")
            seen.add(entry.filename.casefold())
        out = new_directory(output)
        (out / "originals").mkdir()
        (out / "contact-sheets").mkdir()
        media = []
        for entry in entries:
            destination = out / "originals" / entry.filename
            copied = 0
            with archive.open(entry) as src, destination.open("xb") as dst:
                while chunk := src.read(1024 * 1024):
                    copied += len(chunk)
                    require(copied <= entry.file_size, "Archive expanded beyond declared size")
                    dst.write(chunk)
            require(copied == entry.file_size, "Truncated archive member")
            preview = out / "contact-sheets" / (destination.stem + ".png")
            media.append({"path": str(destination.relative_to(out)), "inspection": inspect(destination, preview)})
    require(sha256(source) == digest, "Original archive changed during inspection")
    report = {"schema_version": 1, "kind": "reference_inspection", "source_name": source.name,
              "source_sha256": digest, "source_bytes": source.stat().st_size, "media": media,
              "rights": "User-supplied reference; license absent; embedded provenance is not rights verification",
              "visual_review": "pending", "generation_cost_usd": 0, "tools": tools_info(),
              "limits": LIMITS, "artifacts": artifacts(out)}
    write_json(out / "manifest.json", report)
    return report
