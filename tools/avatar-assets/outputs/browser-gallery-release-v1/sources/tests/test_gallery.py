"""Gallery security/stream tests use synthetic bytes, not generated asset evidence."""

import hashlib
import http.client
import json
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from avatar_assets import gallery


class _Fixture:
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)/"workspace"
        self.root.mkdir()
        self.video = bytes(range(256))*4
        self.model = b"glTF-synthetic-routing-fixture"
        self._files("browser", {"index.html": b"<!doctype html><title>Fixture</title>",
                               "app.js": b"export const fixture = true;", "styles.css": b"body { color: black; }",
                               "favicon.svg": b'<svg xmlns="http://www.w3.org/2000/svg"/>',
                               "node_modules/@google/model-viewer/dist/model-viewer.min.js": b"/* local vendor fixture */"})
        self._pack("outputs/model", {"model.glb": self.model, "portrait.png": b"synthetic-png",
                                    "source.blend": b"synthetic-blend", "idle-contact.png": b"contact",
                                    "renders/idle/0000.png": b"pose"},
                   {"kind": "authored_3d_avatar", "states": [{"id": "idle", "meaning": "A deliberate turn.", "duration_seconds": 4}],
                    "rig_verification": {"mesh_vertices": 3, "bones": ["base", "tip"],
                                         "motion": {"animations": [{"name": "idle"}]}}})
        parent_hash = hashlib.sha256((self.root/"outputs/model/manifest.json").read_bytes()).hexdigest()
        self._pack("outputs/media", {"animations/idle.mp4": self.video,
                                    "fallbacks/custom-idle.png": b"fallback",
                                    "contact-sheets/idle.png": b"contact"},
                   {"kind": "rendered_2d_avatar", "parent_3d_manifest_sha256": parent_hash,
                    "clips": [{"id": "idle", "path": "animations/idle.mp4", "fallback": "fallbacks/custom-idle.png",
                               "inspection": {"duration_seconds": 2}}]})
        self._pack("outputs/mapping", {"mapping.json": b'{"fixture":true}'},
                   {"kind": "weft_authored_model_and_mapping", "source_manifest_sha256": parent_hash})
        self._pack("outputs/reference", {"originals/reference.webp": b"reference-image",
                                        "originals/working.mp4": self.video,
                                        "contact-sheets/working.png": b"reference-contact"},
                   {"kind": "reference_inspection", "media": [
                       {"path": "originals/reference.webp", "inspection": {"kind": "image"}},
                       {"path": "originals/working.mp4", "inspection": {"kind": "video", "duration_seconds": 8}}]})
        self.config = {"schema_version": 1, "assets": [
            {"id": "robot", "title": "Robot", "description": "Synthetic HTTP fixture", "kind": "authored_3d",
             "pack": "outputs/model", "media_pack": "outputs/media", "mapping_pack": "outputs/mapping"},
            {"id": "reference", "title": "Reference", "description": "Synthetic reference fixture",
             "kind": "reference_2d", "pack": "outputs/reference"}]}
        self.write_catalog()

    def _files(self, folder, files):
        for name, raw in files.items():
            path = self.root/folder/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)

    def _pack(self, folder, files, fields):
        self._files(folder, files)
        data = {"schema_version": 1, **fields,
                "artifacts": [{"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                              for name, raw in files.items()]}
        (self.root/folder/"manifest.json").write_text(json.dumps(data))

    def write_catalog(self):
        (self.root/"browser/gallery.json").write_text(json.dumps(self.config))

    def close(self):
        self.temp.cleanup()


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.fixture = _Fixture()
        self.addCleanup(self.fixture.close)

    def load(self):
        return gallery._Catalog(self.fixture.root, "browser/gallery.json")

    def test_api_contract_and_manifest_selected_fallback(self):
        catalog = self.load()
        robot, reference = catalog.data["assets"]
        self.assertEqual(robot["facts"], {"vertices": 3, "bones": 2, "animations": 1})
        self.assertEqual(robot["states"][0]["duration_seconds"], 2)
        self.assertEqual(catalog.routes[robot["states"][0]["poster"]].relative,
                         "outputs/media/fallbacks/custom-idle.png")
        self.assertEqual(robot["downloads"]["mapping"], "/asset/robot/mapping")
        self.assertNotIn(str(self.fixture.root), catalog.raw.decode())
        self.assertIsNone(reference["model"])
        self.assertEqual(reference["facts"]["animations"], 0)
        self.assertIn("Filename label only", reference["states"][0]["meaning"])
        self.assertIn("no 3D model", reference["notice"])

    def test_unlisted_files_do_not_get_routes(self):
        (self.fixture.root/"outputs/model/private.txt").write_text("not public")
        catalog = self.load()
        self.assertFalse(any("private" in r.relative for r in catalog.routes.values()))
        self.assertNotIn("/outputs/model/model.glb", catalog.routes)

    def test_scoped_local_vendor_is_a_safe_whitelisted_file(self):
        catalog = self.load()
        vendor = catalog.routes["/vendor/model-viewer.min.js"]
        with gallery._open_file(self.fixture.root, vendor.relative)[0] as handle:
            self.assertEqual(handle.read(), b"/* local vendor fixture */")

    def test_path_escape_and_url_configuration_are_rejected(self):
        for unsafe in ("../outside", "/tmp/outside", "https://example.invalid/model", "outputs/../model"):
            with self.subTest(path=unsafe):
                self.fixture.config["assets"][0]["pack"] = unsafe
                self.fixture.write_catalog()
                with self.assertRaises(ValueError):
                    self.load()

    def test_symlink_pack_component_is_rejected(self):
        (self.fixture.root/"linked-pack").symlink_to(self.fixture.root/"outputs/model", target_is_directory=True)
        self.fixture.config["assets"][0]["pack"] = "linked-pack"
        self.fixture.write_catalog()
        with self.assertRaises((ValueError, OSError)):
            self.load()

    def test_symlink_leaf_is_rejected(self):
        path = self.fixture.root/"outputs/model/model.glb"
        path.unlink()
        path.symlink_to(self.fixture.root/"outputs/model/source.blend")
        with self.assertRaises((ValueError, OSError)):
            self.load()

    def test_wrong_media_parent_is_rejected(self):
        path = self.fixture.root/"outputs/media/manifest.json"
        data = json.loads(path.read_text())
        data["parent_3d_manifest_sha256"] = "0"*64
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.load()

    def test_wrong_mapping_parent_is_rejected(self):
        path = self.fixture.root/"outputs/mapping/manifest.json"
        data = json.loads(path.read_text())
        data["source_manifest_sha256"] = "0"*64
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "Mapping does not match"):
            self.load()

    def test_missing_or_nonnumeric_rig_facts_are_rejected(self):
        self.fixture.config["assets"][0].pop("media_pack")
        self.fixture.config["assets"][0].pop("mapping_pack")
        self.fixture.write_catalog()
        path = self.fixture.root/"outputs/model/manifest.json"
        data = json.loads(path.read_text())
        for invalid in (None, True, -1, "3", 1_000_001):
            with self.subTest(vertices=invalid):
                data["rig_verification"]["mesh_vertices"] = invalid
                path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, "numeric rig facts"):
                    self.load()

    def test_missing_video_and_oversized_resource_are_rejected(self):
        path = self.fixture.root/"outputs/media/manifest.json"
        data = json.loads(path.read_text())
        data["clips"][0].pop("path")
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            self.load()
        with patch.object(gallery, "MAX_FILE_BYTES", 4):
            # Exercise the explicit byte bound independently of the manifest error.
            with self.assertRaises(ValueError):
                gallery._open_file(self.fixture.root, "outputs/model/model.glb", maximum=4)

    def test_duplicate_ids_and_excess_asset_count_are_rejected(self):
        self.fixture.config["assets"].append(dict(self.fixture.config["assets"][0]))
        self.fixture.write_catalog()
        with self.assertRaisesRegex(ValueError, "Duplicate gallery"):
            self.load()
        self.fixture.config["assets"] *= gallery.MAX_ASSETS
        self.fixture.write_catalog()
        with self.assertRaisesRegex(ValueError, "bounded gallery"):
            self.load()

    def test_range_parser_boundaries(self):
        self.assertEqual(gallery._byte_range("bytes=0-3", 10), (0, 3))
        self.assertEqual(gallery._byte_range("bytes=4-", 10), (4, 9))
        self.assertEqual(gallery._byte_range("bytes=-3", 10), (7, 9))
        self.assertEqual(gallery._byte_range("bytes=0-100", 10), (0, 9))
        for invalid in ("bytes=10-", "bytes=8-3", "bytes=-0", "bytes=-", "bytes=0-1,3-4", "items=0-3"):
            with self.subTest(range=invalid), self.assertRaises(ValueError):
                gallery._byte_range(invalid, 10)

    def test_absolute_deadline_closes_stalled_request_and_releases_slot(self):
        server = object.__new__(gallery._Server)
        server._slots = threading.BoundedSemaphore(1)
        self.assertTrue(server._slots.acquire(blocking=False))
        expired = threading.Event()
        request = Mock()
        request.shutdown.side_effect = lambda _: expired.set()

        def stalled_request(*_):
            self.assertTrue(expired.wait(1), "Absolute request deadline did not fire")

        with patch.object(gallery, "REQUEST_TIMEOUT_SECONDS", .01), \
                patch.object(gallery.ThreadingMixIn, "process_request_thread", stalled_request):
            server.process_request_thread(request, ("127.0.0.1", 1))
        request.shutdown.assert_called_once_with(socket.SHUT_RDWR)
        self.assertTrue(server._slots.acquire(blocking=False))


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.fixture = _Fixture()
        self.addCleanup(self.fixture.close)
        self.workspace = patch.object(gallery, "WORKSPACE", self.fixture.root)
        self.workspace.start()
        self.addCleanup(self.workspace.stop)
        self.server = gallery.create_server(port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop)
        self.host = f"127.0.0.1:{self.server.server_address[1]}"

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.assertFalse(self.thread.is_alive())

    def request(self, path, method="GET", headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=3)
        try:
            connection.request(method, path, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_loopback_only_and_catalog_get_head_etag(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")
        status, headers, raw = self.request("/api/catalog")
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(raw)["assets"]), 2)
        status, head, body = self.request("/api/catalog", method="HEAD")
        self.assertEqual((status, body), (200, b""))
        self.assertEqual(int(head["Content-Length"]), len(raw))
        self.assertEqual(self.request("/api/catalog", headers={"If-None-Match": headers["ETag"]})[0], 304)
        self.assertNotIn("Access-Control-Allow-Origin", headers)
        self.assertIn("script-src 'self'", headers["Content-Security-Policy"])
        self.assertNotIn("'unsafe-eval'", headers["Content-Security-Policy"])
        self.assertIn("'wasm-unsafe-eval'", headers["Content-Security-Policy"])

    def test_static_whitelist_and_exact_model_download(self):
        for path in ("/", "/app.js", "/styles.css", "/favicon.svg", "/vendor/model-viewer.min.js"):
            with self.subTest(path=path):
                self.assertEqual(self.request(path)[0], 200)
        self.assertEqual(self.request("/asset/robot/model")[2], self.fixture.model)
        for path in ("/outputs/model/model.glb", "/asset/robot/source.blend", "/browser/", "/.git/config"):
            with self.subTest(path=path):
                self.assertEqual(self.request(path)[0], 404)

    def test_video_ranges_head_and_weak_if_range(self):
        path = "/asset/robot/video-idle"
        for value, expected in (("bytes=0-9", self.fixture.video[:10]),
                                ("bytes=1000-", self.fixture.video[1000:]),
                                ("bytes=-7", self.fixture.video[-7:])):
            with self.subTest(range=value):
                status, headers, raw = self.request(path, headers={"Range": value})
                self.assertEqual((status, raw), (206, expected))
                self.assertEqual(int(headers["Content-Length"]), len(expected))
                self.assertIn("/1024", headers["Content-Range"])
        status, headers, body = self.request(path, method="HEAD")
        self.assertEqual((status, body), (200, b""))
        self.assertEqual(int(headers["Content-Length"]), 1024)
        self.assertEqual(self.request(path, headers={"If-None-Match": headers["ETag"]})[0], 304)
        status, _, body = self.request(path, headers={"Range": "bytes=0-9", "If-Range": headers["ETag"]})
        self.assertEqual((status, body), (200, self.fixture.video))
        for invalid in ("bytes=1024-", "bytes=0-1,3-5", "bytes=-0"):
            self.assertEqual(self.request(path, headers={"Range": invalid})[0], 416)

    def test_host_origin_and_fetch_site_guards(self):
        for headers in ({"Host": "example.invalid"}, {"Host": "127.0.0.1:1"},
                        {"Origin": "https://example.invalid"}, {"Origin": "null"},
                        {"Origin": "http://localhost:"+str(self.server.server_address[1])},
                        {"Sec-Fetch-Site": "cross-site"}):
            with self.subTest(headers=headers):
                self.assertEqual(self.request("/api/catalog", headers=headers)[0], 403)
        self.assertEqual(self.request("/api/catalog", headers={"Origin": "http://"+self.host})[0], 200)

    def test_traversal_and_absolute_request_targets_are_rejected(self):
        for path in ("/../private", "/%2e%2e/private", "/asset/robot/%6dodel", "http://example.invalid/"):
            with self.subTest(path=path):
                self.assertEqual(self.request(path)[0], 403)
        # Python normalizes leading double slashes before dispatch; neither form is routable.
        self.assertIn(self.request("//example.invalid/")[0], (403, 404))

    def test_all_mutations_are_rejected(self):
        for method in ("POST", "PUT", "DELETE", "PATCH", "OPTIONS"):
            with self.subTest(method=method):
                self.assertEqual(self.request("/api/catalog", method=method)[0], 405)

    def test_symlink_swap_and_changed_asset_fail_closed(self):
        path = self.fixture.root/"outputs/model/model.glb"
        path.unlink()
        path.symlink_to(self.fixture.root/"outputs/model/source.blend")
        self.assertEqual(self.request("/asset/robot/model")[0], 403)
        poster = self.fixture.root/"outputs/model/portrait.png"
        poster.write_bytes(b"changed synthetic data")
        self.assertEqual(self.request("/asset/robot/poster")[0], 409)

    def test_duplicate_host_is_rejected(self):
        with socket.create_connection(self.server.server_address, timeout=3) as client:
            client.sendall(f"GET /api/catalog HTTP/1.1\r\nHost: {self.host}\r\nHost: example.invalid\r\n\r\n".encode())
            self.assertIn(b" 403 ", client.recv(1024).split(b"\r\n", 1)[0])


if __name__ == "__main__":
    unittest.main()
