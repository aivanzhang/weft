"""Read-only loopback gallery with explicit routes, bounded streams and no listings."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import socket
from socketserver import ThreadingMixIn
import stat
import threading
import time
from urllib.parse import urlsplit


WORKSPACE = Path(__file__).resolve().parent.parent
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_CATALOG_BYTES = 512 * 1024
MAX_ASSETS = 24
MAX_STATES = 24
MAX_REQUESTS = 8
SOCKET_TIMEOUT_SECONDS = 5
STREAM_TIMEOUT_SECONDS = 20
REQUEST_TIMEOUT_SECONDS = 25
CHUNK_BYTES = 64 * 1024
_SLUG = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png",
         ".webp": "image/webp", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
         ".mp4": "video/mp4", ".glb": "model/gltf-binary", ".blend": "application/octet-stream",
         ".json": "application/json; charset=utf-8"}
_STATIC = {"/": "browser/index.html", "/app.js": "browser/app.js",
           "/styles.css": "browser/styles.css", "/favicon.svg": "browser/favicon.svg",
           "/vendor/model-viewer.min.js": "browser/node_modules/@google/model-viewer/dist/model-viewer.min.js"}
_CSP = ("default-src 'none'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; media-src 'self' data: blob:; connect-src 'self' blob:; "
        "font-src 'self'; worker-src 'self' blob:; object-src 'none'; base-uri 'none'; "
        "frame-ancestors 'none'; form-action 'none'")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _relative(value):
    _require(isinstance(value, str) and 0 < len(value) <= 512
             and re.fullmatch(r"[A-Za-z0-9_./@-]+", value), "Expected a safe workspace-relative path")
    path = PurePosixPath(value)
    _require(not path.is_absolute() and all(part not in ("", ".", "..") for part in value.split("/"))
             and len(path.parts) <= 16, "Unsafe relative path")
    return path


def _open_file(root, relative, maximum=MAX_FILE_BYTES):
    parts = _relative(str(relative)).parts
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    leaf = None
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        leaf = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        info = os.fstat(leaf)
        _require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= maximum,
                 "Gallery resource is not a bounded regular file")
        handle = os.fdopen(leaf, "rb")
        leaf = None
        return handle, info
    finally:
        if leaf is not None:
            os.close(leaf)
        os.close(directory)


def _json_bytes(root, relative, maximum):
    with _open_file(root, relative, maximum)[0] as handle:
        raw = handle.read(maximum + 1)
    _require(len(raw) <= maximum, "Gallery JSON exceeded its byte bound")

    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("Non-finite JSON value")

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    except (RecursionError, UnicodeError) as exc:
        raise ValueError("Invalid gallery JSON") from exc
    _require(isinstance(value, dict), "Gallery JSON must be an object")
    return value, raw


def _text(value, maximum, label):
    _require(isinstance(value, str) and 0 < len(value) <= maximum, f"Invalid {label}")
    return value


def _slug(value):
    _require(isinstance(value, str) and _SLUG.fullmatch(value), "Invalid gallery ID")
    return value


def _duration(value):
    _require(type(value) in (int, float) and math.isfinite(value) and 0 < value <= 3600,
             "Invalid gallery animation duration")
    return value


def _fingerprint(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


@dataclass(frozen=True)
class _Resource:
    relative: str
    content_type: str
    fingerprint: tuple | None = None
    attachment: bool = False


class _Catalog:
    def __init__(self, root, catalog):
        self.root, self.routes = root, {}
        document, _ = _json_bytes(root, catalog, MAX_CATALOG_BYTES)
        entries = document.get("assets")
        _require(document.get("schema_version") == 1 and isinstance(entries, list)
                 and 0 < len(entries) <= MAX_ASSETS, "Expected bounded gallery schema_version 1")
        assets, seen = [], set()
        for entry in entries:
            _require(isinstance(entry, dict) and set(entry) <=
                     {"id", "title", "description", "kind", "pack", "media_pack", "mapping_pack"},
                     "Invalid gallery asset fields")
            identifier = _slug(entry.get("id"))
            _require(identifier not in seen, "Duplicate gallery asset")
            seen.add(identifier)
            kind = entry.get("kind")
            _require(kind in ("authored_3d", "reference_2d"), "Unsupported gallery asset kind")
            pack = self._pack(entry.get("pack"))
            base = {"id": identifier, "title": _text(entry.get("title"), 160, "asset title"),
                    "description": _text(entry.get("description"), 4000, "asset description"), "kind": kind}
            if kind == "authored_3d":
                base.update(self._authored(identifier, entry, pack))
            else:
                _require(not entry.get("media_pack") and not entry.get("mapping_pack"),
                         "Reference gallery assets cannot claim model or mapping packs")
                base.update(self._reference(identifier, pack))
            assets.append(base)
        self.data = {"schema_version": 1, "assets": assets}
        self.raw = json.dumps(self.data, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
        _require(len(self.raw) <= MAX_CATALOG_BYTES, "Expanded gallery catalog exceeds bound")
        self.etag = '"' + hashlib.sha256(self.raw).hexdigest() + '"'
        for route, relative in _STATIC.items():
            self.routes[route] = _Resource(relative, _MIME[Path(relative).suffix])

    def _pack(self, relative):
        relative = str(_relative(relative))
        data, raw = _json_bytes(self.root, relative + "/manifest.json", MAX_MANIFEST_BYTES)
        _require(data.get("schema_version") == 1, "Unsupported pack manifest version")
        entries = data.get("artifacts")
        _require(isinstance(entries, list) and 0 < len(entries) <= 8192, "Pack has no bounded artifact inventory")
        inventory = set()
        for entry in entries:
            _require(isinstance(entry, dict), "Invalid artifact entry")
            name = str(_relative(entry.get("path")))
            _require(name not in inventory, "Duplicate artifact path")
            inventory.add(name)
        return {"path": relative, "data": data, "inventory": inventory,
                "manifest_sha256": hashlib.sha256(raw).hexdigest()}

    def _resource(self, identifier, key, pack, name, suffixes, *, optional=False, attachment=False):
        _require(isinstance(key, str) and re.fullmatch(r"[a-z][a-z0-9_-]{0,95}", key), "Invalid route key")
        if optional and name is None:
            return None
        name = str(_relative(name))
        if optional and name not in pack["inventory"]:
            return None
        _require(name == "manifest.json" or name in pack["inventory"], "Gallery file is not inventoried")
        suffix = Path(name).suffix.lower()
        _require(suffix in suffixes, "Gallery file format differs from its role")
        relative = pack["path"] + "/" + name
        handle, info = _open_file(self.root, relative)
        handle.close()
        route = f"/asset/{identifier}/{key}"
        _require(route not in self.routes, "Duplicate gallery route")
        self.routes[route] = _Resource(relative, _MIME[suffix], _fingerprint(info), attachment)
        return route

    def _image(self, identifier, key, pack, name, optional=False):
        return self._resource(identifier, key, pack, name, {".png", ".webp", ".jpg", ".jpeg"}, optional=optional)

    def _authored(self, identifier, entry, pack):
        data = pack["data"]
        _require(data.get("kind") == "authored_3d_avatar", "Expected an authored 3D pack")
        model = self._resource(identifier, "model", pack, "model.glb", {".glb"})
        poster = self._image(identifier, "poster", pack, "portrait.png")
        downloads = {"model": model,
                     "manifest": self._resource(identifier, "manifest", pack, "manifest.json", {".json"}, attachment=True)}
        blend = self._resource(identifier, "blend", pack, "source.blend", {".blend"}, optional=True, attachment=True)
        if blend:
            downloads["blend"] = blend
        media, clips = None, {}
        if entry.get("media_pack"):
            media = self._pack(entry["media_pack"])
            _require(media["data"].get("kind") == "rendered_2d_avatar"
                     and media["data"].get("parent_3d_manifest_sha256") == pack["manifest_sha256"],
                     "Rendered media does not match the chosen 3D source manifest")
            media_clips = media["data"].get("clips")
            _require(isinstance(media_clips, list) and len(media_clips) <= MAX_STATES, "Invalid media clip list")
            for clip in media_clips:
                _require(isinstance(clip, dict), "Invalid media clip")
                name = _slug(clip.get("id"))
                _require(name not in clips, "Duplicate media state")
                clips[name] = clip
        portrait = (self._image(identifier, "portrait", media, "portrait.webp", optional=True) if media else None) or poster
        states, state_ids = [], set()
        source_states = data.get("states")
        _require(isinstance(source_states, list) and 0 < len(source_states) <= MAX_STATES, "Invalid authored state list")
        for state in source_states:
            _require(isinstance(state, dict), "Invalid authored state")
            name = _slug(state.get("id"))
            _require(name not in state_ids, "Duplicate authored state")
            state_ids.add(name)
            clip = clips.get(name)
            video, static, contact = None, None, None
            duration = _duration(state.get("duration_seconds"))
            if clip:
                video = self._resource(identifier, "video-"+name, media, clip.get("path"), {".mp4"})
                static = self._image(identifier, "still-"+name, media, clip.get("fallback"))
                contact = self._image(identifier, "contact-"+name, media,
                                      "contact-sheets/"+name+".png", optional=True)
                duration = _duration(clip.get("inspection", {}).get("duration_seconds"))
            else:
                static = self._image(identifier, "still-"+name, pack,
                                     "renders/"+name+"/0000.png", optional=True)
            if not contact:
                contact = self._image(identifier, "contact-"+name, pack, name+"-contact.png", optional=True)
            states.append({"id": name, "label": name.replace("_", " ").capitalize(),
                           "meaning": _text(state.get("meaning"), 4000, "state meaning"),
                           "duration_seconds": duration, "video": video,
                           "poster": static or poster, "contact": contact})
        _require(set(clips) <= state_ids, "Media contains an unknown authored state")
        rig = data.get("rig_verification", {})
        _require(isinstance(rig, dict) and isinstance(rig.get("motion", {}), dict), "Invalid rig facts")
        vertices, bones = rig.get("mesh_vertices"), rig.get("bones")
        animations = rig.get("motion", {}).get("animations")
        _require(type(vertices) is int and 0 < vertices <= 1_000_000
                 and isinstance(bones, list) and 0 < len(bones) <= 256
                 and isinstance(animations, list) and 0 < len(animations) <= MAX_STATES,
                 "Expected bounded numeric rig facts")
        facts = {"vertices": vertices, "bones": len(bones), "animations": len(animations)}
        notice = "Authored 3D; seamless loops are not certified."
        if media:
            notice += " Video timing may differ from 3D clips."
        if entry.get("mapping_pack"):
            mapping = self._pack(entry["mapping_pack"])
            _require(mapping["data"].get("kind") == "weft_authored_model_and_mapping"
                     and mapping["data"].get("source_manifest_sha256") == pack["manifest_sha256"],
                     "Mapping does not match the chosen 3D source manifest")
            downloads["mapping"] = self._resource(identifier, "mapping", mapping, "mapping.json", {".json"}, attachment=True)
            notice += " Weft alignment remains unverified."
        return {"poster": poster, "portrait": portrait, "model": model, "states": states, "downloads": downloads,
                "facts": facts, "notice": notice}

    def _reference(self, identifier, pack):
        _require(pack["data"].get("kind") == "reference_inspection", "Expected a reference inspection pack")
        media = pack["data"].get("media")
        _require(isinstance(media, list) and 0 < len(media) <= MAX_STATES+1
                 and all(isinstance(item, dict) and isinstance(item.get("inspection"), dict) for item in media),
                 "Invalid reference media list")
        images = [item for item in media if item.get("inspection", {}).get("kind") == "image"]
        _require(images, "Reference pack has no static portrait")
        poster = self._image(identifier, "poster", pack, images[0].get("path"))
        states = []
        for item in media:
            inspection = item.get("inspection", {})
            if inspection.get("kind") != "video":
                continue
            index = len(states)+1
            name = "reference-"+str(index)
            original = item.get("path")
            video = self._resource(identifier, "video-"+name, pack, original, {".mp4"})
            contact = self._image(identifier, "contact-"+name, pack,
                                  "contact-sheets/"+PurePosixPath(original).stem+".png", optional=True)
            states.append({"id": name, "label": f"Original clip {index}",
                           "meaning": "Filename label only: " + PurePosixPath(original).name +
                                      ". Motion meaning and seamless looping are unverified.",
                           "duration_seconds": _duration(inspection.get("duration_seconds")),
                           "video": video, "poster": poster, "contact": contact})
        return {"poster": poster, "model": None, "states": states,
                "downloads": {"manifest": self._resource(identifier, "manifest", pack, "manifest.json", {".json"}, attachment=True)},
                "facts": {"vertices": 0, "bones": 0, "animations": 0},
                "notice": "2D reference only: no 3D model or rig. Filename labels do not verify actions or seamless loops."}


def _byte_range(value, size):
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
    _require(match and any(match.groups()), "Unsupported byte range")
    start, end = match.groups()
    if not start:
        count = int(end)
        _require(count > 0, "Invalid suffix range")
        return max(0, size-count), size-1
    start = int(start)
    end = size-1 if not end else min(int(end), size-1)
    _require(0 <= start <= end and start < size, "Unsatisfiable byte range")
    return start, end


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "AvatarGallery/1"
    sys_version = ""

    def log_message(self, *_):
        pass

    def _headers(self, status, length, content_type=None, extra=None):
        self.send_response(status)
        self.send_header("Content-Length", str(length))
        self.send_header("Connection", "close")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Content-Security-Policy", _CSP)
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "private, no-cache")
        if content_type:
            self.send_header("Content-Type", content_type)
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.close_connection = True

    def _error(self, status, message, extra=None):
        raw = json.dumps({"error": message}, separators=(",", ":")).encode()
        self._headers(status, len(raw), "application/json; charset=utf-8", extra)
        if self.command != "HEAD":
            self.wfile.write(raw)

    def _allowed(self):
        hosts = self.headers.get_all("Host", [])
        port = self.server.server_address[1]
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if port == 80:
            allowed.update(("127.0.0.1", "localhost"))
        if len(hosts) != 1 or hosts[0] not in allowed:
            self._error(403, "Loopback Host required")
            return False
        origins = self.headers.get_all("Origin", [])
        if len(origins) > 1 or origins and origins[0] != "http://"+hosts[0]:
            self._error(403, "Cross-origin access is not allowed")
            return False
        if self.headers.get("Sec-Fetch-Site") not in (None, "same-origin", "none"):
            self._error(403, "Cross-origin access is not allowed")
            return False
        if sum(len(k)+len(v) for k, v in self.headers.items()) > 16*1024 or len(self.path) > 2048:
            self._error(431, "Request exceeds gallery bounds")
            return False
        return True

    def _get(self):
        if not self._allowed():
            return
        try:
            parsed = urlsplit(self.path)
        except ValueError:
            self._error(403, "Unsafe request path")
            return
        if (not self.path.startswith("/") or parsed.scheme or parsed.netloc or parsed.fragment
                or "%" in parsed.path or "\\" in parsed.path or ".." in parsed.path.split("/")):
            self._error(403, "Unsafe request path")
            return
        if parsed.path == "/api/catalog":
            catalog = self.server.catalog
            if self.headers.get("If-None-Match") in (catalog.etag, "*"):
                self._headers(304, 0, extra={"ETag": catalog.etag})
                return
            self._headers(200, len(catalog.raw), "application/json; charset=utf-8", {"ETag": catalog.etag})
            if self.command != "HEAD":
                self.wfile.write(catalog.raw)
            return
        resource = self.server.catalog.routes.get(parsed.path)
        if resource is None:
            self._error(404, "Unknown gallery resource")
            return
        try:
            handle, info = _open_file(self.server.catalog.root, resource.relative)
        except FileNotFoundError:
            self._error(404, "Gallery resource is unavailable")
            return
        except (OSError, ValueError):
            self._error(403, "Gallery resource is unsafe or outside bounds")
            return
        with handle:
            if resource.fingerprint is not None and _fingerprint(info) != resource.fingerprint:
                self._error(409, "Asset changed; restart the gallery to reload its catalog")
                return
            size = info.st_size
            etag = f'W/"{info.st_ino:x}-{info.st_size:x}-{info.st_mtime_ns:x}"'
            extra = {"ETag": etag, "Accept-Ranges": "bytes"}
            if self.headers.get("If-None-Match") in (etag, "*"):
                self._headers(304, 0, extra=extra)
                return
            start, end, status = 0, size-1, 200
            ranges = self.headers.get_all("Range", [])
            if len(ranges) > 1:
                self._error(416, "Only one byte range is supported", {"Content-Range": f"bytes */{size}"})
                return
            # A weak validator cannot satisfy If-Range; send the complete representation.
            if ranges and self.command == "GET" and "If-Range" not in self.headers:
                try:
                    start, end = _byte_range(ranges[0], size)
                except (ValueError, OverflowError):
                    self._error(416, "Unsatisfiable byte range", {"Content-Range": f"bytes */{size}"})
                    return
                status = 206
                extra["Content-Range"] = f"bytes {start}-{end}/{size}"
            if resource.attachment:
                extra["Content-Disposition"] = f'attachment; filename="{PurePosixPath(resource.relative).name}"'
            self._headers(status, end-start+1, resource.content_type, extra)
            if self.command == "HEAD":
                return
            handle.seek(start)
            remaining, deadline = end-start+1, time.monotonic()+STREAM_TIMEOUT_SECONDS
            while remaining:
                if time.monotonic() > deadline:
                    break
                chunk = handle.read(min(CHUNK_BYTES, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_GET(self):
        try:
            self._get()
        except (BrokenPipeError, ConnectionResetError, TimeoutError, socket.timeout):
            self.close_connection = True

    do_HEAD = do_GET

    def _method_not_allowed(self):
        if self._allowed():
            self._error(405, "Gallery is read-only", {"Allow": "GET, HEAD"})

    do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _method_not_allowed


class _Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    request_queue_size = 16
    allow_reuse_address = False

    def __init__(self, catalog, port):
        self.catalog = catalog
        self._slots = threading.BoundedSemaphore(MAX_REQUESTS)
        super().__init__(("127.0.0.1", port), _Handler)

    def get_request(self):
        request, address = super().get_request()
        request.settimeout(SOCKET_TIMEOUT_SECONDS)
        return request, address

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            try:
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        # An idle timeout alone lets a client hold its slot by trickling header bytes.
        def expire():
            try:
                request.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        timer = threading.Timer(REQUEST_TIMEOUT_SECONDS, expire)
        timer.daemon = True
        try:
            timer.start()
            super().process_request_thread(request, client_address)
        finally:
            timer.cancel()
            self._slots.release()


def create_server(port=8765, catalog_path=None):
    """Create, but do not run, a server bound only to 127.0.0.1."""
    _require(type(port) is int and 0 <= port <= 65535, "Invalid gallery port")
    root = WORKSPACE.resolve()
    path = Path(catalog_path) if catalog_path is not None else root/"browser/gallery.json"
    if not path.is_absolute():
        path = root/path
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("Gallery catalog must be inside the workspace") from exc
    return _Server(_Catalog(root, str(_relative(relative))), port)


def serve_gallery(port=8765, catalog=None):
    """Serve local previews until interrupted; never launch a browser or renderer."""
    server = create_server(port, catalog)
    print(f"Avatar gallery: http://127.0.0.1:{server.server_address[1]}/", flush=True)
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
