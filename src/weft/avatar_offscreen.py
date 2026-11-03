"""Windowless avatar rendering with the same bounded pose math as the browser worker."""

import ctypes
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import selectors
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
from PIL import Image


MAX_PACKAGES, MAX_RENDER_BYTES = 8, 32 * 1024 * 1024
MAX_FRAMES, MAX_SECONDS = 18001, 120
_ACTIVE_THREADS = set()
_ACTIVE_LOCK = threading.Lock()


class AvatarRenderError(ValueError):
    pass


VERTEX = """#version 120
attribute vec3 aPosition; attribute vec3 aNormal; attribute vec2 aUv;
attribute vec4 aJoints; attribute vec4 aWeights;
uniform mat4 uView; uniform mat4 uNode; uniform bool uSkinned;
uniform sampler2D uPalette; uniform bool uCapsule; uniform float uRadius; uniform float uHalfLength;
varying vec3 vNormal; varying vec2 vUv;
mat4 bone(float index){float y=(index+.5)/64.;return mat4(texture2D(uPalette,vec2(.125,y)),texture2D(uPalette,vec2(.375,y)),texture2D(uPalette,vec2(.625,y)),texture2D(uPalette,vec2(.875,y)));}
void main(){
  mat4 skin=uSkinned?bone(aJoints.x)*aWeights.x+bone(aJoints.y)*aWeights.y+bone(aJoints.z)*aWeights.z+bone(aJoints.w)*aWeights.w:uNode;
  mat3 basis=mat3(skin); vec3 a=basis[0],b=basis[1],c=basis[2];float det=dot(a,cross(b,c));
  vNormal=abs(det)>1e-12?mat3(cross(b,c),cross(c,a),cross(a,b))*aNormal/det:aNormal;
  vec3 p=aPosition;if(uCapsule)p=vec3(p.xy*uRadius,p.z*uRadius+aWeights.x*uHalfLength);
  vUv=aUv;gl_Position=uView*skin*vec4(p,1.);
}"""
FRAGMENT = """#version 120
varying vec3 vNormal; varying vec2 vUv;
uniform sampler2D uTexture; uniform vec4 uColor; uniform bool uUnlit;
uniform bool uMask; uniform bool uDoubleSided; uniform float uCutoff;
float encode(float c){c=max(c,0.);return c<=.0031308?12.92*c:1.055*pow(c,1./2.4)-.055;}
void main(){
  vec4 base=texture2D(uTexture,vUv)*uColor;if(uMask&&base.a<uCutoff)discard;
  vec3 n=length(vNormal)>1e-12?normalize(vNormal):vec3(0.,0.,1.);if(uDoubleSided&&!gl_FrontFacing)n=-n;
  float light=uUnlit?1.:.35+.65*max(dot(n,normalize(vec3(.4,-.5,.8))),0.);
  vec3 color=base.rgb*light;gl_FragColor=vec4(encode(color.r),encode(color.g),encode(color.b),1.);
}"""
LINE_VERTEX = """#version 120
attribute vec3 aPosition;attribute vec3 aColor;uniform mat4 uView;varying vec3 vColor;
void main(){vColor=aColor;gl_Position=uView*vec4(aPosition,1.);gl_PointSize=3.;}"""
LINE_FRAGMENT = """#version 120
varying vec3 vColor;void main(){gl_FragColor=vec4(vColor,1.);}"""


def _encoded(value, maximum):
    try:
        raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError, RecursionError) as error:
        raise AvatarRenderError("Avatar render input must be bounded finite JSON") from error
    if len(raw) > maximum:
        raise AvatarRenderError("Avatar render input exceeds its byte bound")
    return raw


class _PoseBridge:
    def __init__(self, scene, assets, width, height):
        executable = shutil.which("node")
        if executable is None:
            raise AvatarRenderError("Avatar export requires the existing Node runtime; no automatic installation")
        self.process = None
        env = {"PATH": os.defpath, **{key: os.environ[key] for key in ("TMPDIR", "TEMP", "TMP") if key in os.environ}}
        bridge = Path(__file__).parent / "assets/avatar_offscreen_bridge.cjs"
        self.process = subprocess.Popen([executable, "--max-old-space-size=256", str(bridge)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=bridge.parent)
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)
        try:
            result = self.request({"type": "init", "scene": scene, "assets": assets, "width": width, "height": height},
                                  seconds=10, max_output=16 * 1024 * 1024)
            if result.get("type") != "ready" or not isinstance(result.get("world"), list):
                raise AvatarRenderError("Avatar pose bridge did not initialize")
            self.world = result["world"]
        except BaseException:
            self.close()
            raise

    def request(self, value, *, seconds=2, max_output=256 * 1024):
        if self.process is None or self.process.poll() is not None:
            raise AvatarRenderError("Avatar pose bridge is closed")
        request = _encoded(value, 36 * 1024 * 1024) + b"\n"
        deadline, offset, output = time.monotonic() + seconds, 0, bytearray()
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdin, selectors.EVENT_WRITE)
                selector.register(self.process.stdout, selectors.EVENT_READ)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise AvatarRenderError("Avatar pose bridge exceeded its response deadline")
                    events = selector.select(remaining)
                    if not events:
                        raise AvatarRenderError("Avatar pose bridge exceeded its response deadline")
                    for key, _ in events:
                        if key.fileobj is self.process.stdin:
                            offset += os.write(key.fd, request[offset:offset + 65536])
                            if offset == len(request):
                                selector.unregister(self.process.stdin)
                        else:
                            chunk = os.read(key.fd, min(65536, max_output + 1 - len(output)))
                            if not chunk:
                                raise AvatarRenderError("Avatar pose bridge stopped before completing a response")
                            output.extend(chunk)
                            if len(output) > max_output:
                                raise AvatarRenderError("Avatar pose bridge exceeded its output bound")
                            if b"\n" in output:
                                if offset != len(request) or not output.endswith(b"\n") or output.count(b"\n") != 1:
                                    raise AvatarRenderError("Avatar pose bridge returned an invalid response frame")
                                result = json.loads(output)
                                if not isinstance(result, dict) or result.get("type") == "error":
                                    raise AvatarRenderError("Avatar pose validation failed: " + str(result.get("message", "invalid response"))[:2000])
                                return result
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            self.close()
            if isinstance(error, AvatarRenderError):
                raise
            raise AvatarRenderError("Avatar pose bridge communication failed") from error

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            if process.stdin:
                process.stdin.close()
            try:
                process.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
            if process.stdout:
                process.stdout.close()


def _packages(packages):
    if not isinstance(packages, dict) or len(packages) > MAX_PACKAGES:
        raise AvatarRenderError("Offscreen rendering supports at most eight admitted avatar packages")
    output, total = {}, 0
    for key, package in packages.items():
        if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key):
            raise AvatarRenderError("Avatar package identities must be exact hashes")
        raw = getattr(package, "render_json", None)
        textures = getattr(package, "texture_files", None)
        if not isinstance(raw, bytes) or len(raw) > 8 * 1024 * 1024 or textures is None:
            raise AvatarRenderError("Offscreen avatars must come from the admitted asset loader")
        total += len(raw)
        if total > MAX_RENDER_BYTES:
            raise AvatarRenderError("Offscreen catalog exceeds its normalized-byte bound")
        try:
            document = json.loads(raw)
        except (ValueError, RecursionError) as error:
            raise AvatarRenderError("Malformed normalized avatar JSON") from error
        output[key] = {"document": document, "textures": textures}
    return output


def _linear(color):
    return [value / 12.92 if value <= .04045 else ((value + .055) / 1.055) ** 2.4 for value in color]


def _capsule():
    positions, normals, weights, indices = [], [], [], []
    rings = [(math.pi * i / 8, 1 if i <= 4 else -1) for i in range(9)]
    rings.insert(5, (math.pi / 2, -1))
    for theta, sign in rings:
        for j in range(13):
            phi = j * math.pi / 6
            p = [math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)]
            positions.extend(p); normals.extend(p); weights.extend([sign, 0, 0, 0])
    for i in range(len(rings) - 1):
        for j in range(12):
            a, b = i * 13 + j, (i + 1) * 13 + j
            indices.extend([a, b, a + 1, a + 1, b, b + 1])
    return {"positions": positions, "normals": normals, "weights": weights, "indices": indices}


def _segment_transform(segment):
    start, end = np.asarray(segment["a"], dtype=float), np.asarray(segment["b"], dtype=float)
    delta, result = end - start, np.eye(4)
    length = float(np.linalg.norm(delta))
    if length > 1e-12:
        z = delta / length
        reference = np.array([0., 1., 0.]) if abs(z[1]) < .9 else np.array([1., 0., 0.])
        x = np.cross(reference, z); x /= np.linalg.norm(x)
        result[:3, :3] = np.column_stack((x, np.cross(z, x), z))
    result[:3, 3] = (start + end) / 2
    return result.reshape(16, order="F"), length / 2


class AvatarOffscreen:
    """One thread-owned GPU context; bounded pose IPC and one resident avatar at a time."""

    def __init__(self, scene, packages_by_hash, width=1080, height=420):
        if type(width) is not int or type(height) is not int or not 3 <= width <= 2048 or not 1 <= height <= 2048:
            raise AvatarRenderError("Offscreen viewport must fit the 2048×2048 bound")
        self.width, self.height = width, height
        self._owner, self._started, self._frames = threading.get_ident(), time.monotonic(), 0
        self._closed, self._registered, self._context, self._bridge, self.gl = False, False, None, None, None
        self._buffers, self._textures, self._programs = set(), set(), set()
        self._resident, self._asset_meshes, self._asset_textures = None, [], []
        self._framebuffer, self._depth = None, None
        self._assets = _packages(packages_by_hash)
        self._last_pose = None
        try:
            self._bridge = _PoseBridge(scene, {key: value["document"] for key, value in self._assets.items()}, width, height)
            self._initialize_gpu()
            if time.monotonic() - self._started > 15:
                raise AvatarRenderError("Offscreen initialization exceeded its work deadline")
        except BaseException:
            self.close()
            raise

    def _initialize_gpu(self):
        if sys.platform != "darwin":
            raise AvatarRenderError("Native avatar video export currently requires the macOS CGL backend")
        with _ACTIVE_LOCK:
            if self._owner in _ACTIVE_THREADS:
                raise AvatarRenderError("Close the existing avatar renderer before opening another on this thread")
            _ACTIVE_THREADS.add(self._owner); self._registered = True
        from mujoco.cgl import GLContext
        from OpenGL import GL
        self.gl = gl = GL
        self._context = GLContext(self.width, self.height)
        self._context.make_current()
        if gl.glGetString(gl.GL_VERSION) is None:
            raise AvatarRenderError("Offscreen CGL is unavailable in this execution context")
        if int(gl.glGetIntegerv(gl.GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS)) < 1 or int(gl.glGetIntegerv(gl.GL_MAX_TEXTURE_SIZE)) < 2048:
            raise AvatarRenderError("Offscreen GPU lacks required palette/texture support")
        self.version = gl.glGetString(gl.GL_VERSION).decode("ascii", errors="replace")
        self._program, self._uniforms = self._program_for(VERTEX, FRAGMENT,
            ["aPosition", "aNormal", "aUv", "aJoints", "aWeights"],
            ["uView", "uNode", "uSkinned", "uPalette", "uTexture", "uColor", "uUnlit", "uMask", "uCutoff", "uDoubleSided", "uCapsule", "uRadius", "uHalfLength"])
        self._line_program, self._line_uniforms = self._program_for(LINE_VERTEX, LINE_FRAGMENT,
                                                                  ["aPosition", "aColor"], ["uView"])
        self._palette = self._new_texture()
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, 4, 64, 0, gl.GL_RGBA, gl.GL_FLOAT, np.zeros((64, 4, 4), np.float32))
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        self._white = self._new_texture()
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_SRGB8_ALPHA8, 1, 1, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, b"\xff" * 4)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        self._color = self._new_texture()
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, self.width, self.height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        self._framebuffer = int(gl.glGenFramebuffers(1)); gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._framebuffer)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._color, 0)
        self._depth = int(gl.glGenRenderbuffers(1)); gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._depth)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, self.width, self.height)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, self._depth)
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise AvatarRenderError("Offscreen avatar framebuffer is incomplete")
        self._world_meshes = [self._mesh(mesh) for mesh in self._bridge.world]
        self._capsule_mesh = self._mesh(_capsule())
        self._line_buffer = self._new_buffer()
        gl.glEnable(gl.GL_DEPTH_TEST); gl.glDepthFunc(gl.GL_LEQUAL); gl.glDisable(gl.GL_BLEND)
        gl.glDisable(gl.GL_FRAMEBUFFER_SRGB)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1); gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        self._check_gpu()

    def _program_for(self, vertex, fragment, attributes, uniforms):
        gl = self.gl
        from OpenGL.GL.shaders import compileShader
        shaders = []
        program = int(gl.glCreateProgram()); self._programs.add(program)
        try:
            for source, kind in ((vertex, gl.GL_VERTEX_SHADER), (fragment, gl.GL_FRAGMENT_SHADER)):
                shader = compileShader(source, kind); shaders.append(shader); gl.glAttachShader(program, shader)
            for index, name in enumerate(attributes):
                gl.glBindAttribLocation(program, index, name)
            gl.glLinkProgram(program)
            if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
                raise AvatarRenderError("Offscreen shader link failed: " + str(gl.glGetProgramInfoLog(program))[:1000])
            return program, {name: gl.glGetUniformLocation(program, name) for name in uniforms}
        finally:
            for shader in shaders:
                gl.glDeleteShader(shader)

    def _new_texture(self):
        value = int(self.gl.glGenTextures(1)); self._textures.add(value)
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, value)
        return value

    def _new_buffer(self):
        value = int(self.gl.glGenBuffers(1)); self._buffers.add(value)
        return value

    def _mesh(self, primitive):
        gl = self.gl
        positions = np.asarray(primitive["positions"], np.float32).reshape(-1, 3)
        count = len(positions)
        data = np.zeros((count, 16), np.float32)
        data[:, :3] = positions
        data[:, 3:6] = np.asarray(primitive["normals"], np.float32).reshape(-1, 3)
        for key, first, size in (("texcoords", 6, 2), ("joints", 8, 4), ("weights", 12, 4)):
            if primitive.get(key) is not None:
                data[:, first:first + size] = np.asarray(primitive[key], np.float32).reshape(-1, size)
        indices = np.asarray(primitive["indices"], np.uint32)
        vertex, index = self._new_buffer(), self._new_buffer()
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertex); gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, index); gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)
        return {"vertex": vertex, "index": index, "count": len(indices), "primitive": primitive}

    def _release_asset(self):
        if self.gl:
            for mesh in self._asset_meshes:
                for handle in (mesh["vertex"], mesh["index"]):
                    self.gl.glDeleteBuffers(1, [handle]); self._buffers.discard(handle)
            for handle in self._asset_textures:
                self.gl.glDeleteTextures([handle]); self._textures.discard(handle)
        self._asset_meshes, self._asset_textures, self._resident = [], [], None

    def _select(self, key):
        if key == self._resident:
            return
        self._release_asset()
        if key is None:
            return
        gl, asset = self.gl, self._assets[key]
        pixels = 0
        for info in asset["document"]["textures"]:
            raw = asset["textures"].get(info["path"])
            if not isinstance(raw, bytes) or len(raw) > 8 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != info["sha256"]:
                raise AvatarRenderError("Offscreen texture bytes differ from the admitted asset")
            with Image.open(BytesIO(raw)) as image:
                if image.size != (info["width"], info["height"]):
                    raise AvatarRenderError("Offscreen texture dimensions differ")
                pixels += image.width * image.height
                if pixels > 8 * 1024 * 1024:
                    raise AvatarRenderError("Offscreen textures exceed the decoded-pixel bound")
                rgba = image.convert("RGBA").tobytes()
            handle = self._new_texture(); self._asset_textures.append(handle)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_SRGB8_ALPHA8, info["width"], info["height"], 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, rgba)
            for parameter, name in ((gl.GL_TEXTURE_WRAP_S, "wrap_s"), (gl.GL_TEXTURE_WRAP_T, "wrap_t"),
                                    (gl.GL_TEXTURE_MAG_FILTER, "mag_filter"), (gl.GL_TEXTURE_MIN_FILTER, "min_filter")):
                gl.glTexParameteri(gl.GL_TEXTURE_2D, parameter, info[name])
            if info["min_filter"] >= 9984:
                gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        self._asset_meshes = [self._mesh(mesh) for mesh in asset["document"]["primitives"]]
        self._resident = key
        self._check_gpu()

    def _material(self, color, *, texture=None, unlit=False, mask=False, cutoff=.5, double_sided=False):
        gl, u = self.gl, self._uniforms
        gl.glUniform4fv(u["uColor"], 1, color)
        for name, value in (("uUnlit", unlit), ("uMask", mask), ("uDoubleSided", double_sided)):
            gl.glUniform1i(u[name], int(value))
        gl.glUniform1f(u["uCutoff"], cutoff)
        if double_sided:
            gl.glDisable(gl.GL_CULL_FACE)
        else:
            gl.glEnable(gl.GL_CULL_FACE); gl.glCullFace(gl.GL_BACK)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._white if texture is None else texture)

    def _draw(self, mesh, matrix, *, skinned=False, capsule=None):
        gl, u = self.gl, self._uniforms
        gl.glUniformMatrix4fv(u["uNode"], 1, False, np.asarray(matrix, np.float32))
        gl.glUniform1i(u["uSkinned"], int(skinned)); gl.glUniform1i(u["uCapsule"], int(capsule is not None))
        if capsule is not None:
            gl.glUniform1f(u["uRadius"], capsule[0]); gl.glUniform1f(u["uHalfLength"], capsule[1])
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, mesh["vertex"])
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, mesh["index"])
        for index, size, offset in ((0, 3, 0), (1, 3, 12), (2, 2, 24), (3, 4, 32), (4, 4, 48)):
            gl.glEnableVertexAttribArray(index)
            gl.glVertexAttribPointer(index, size, gl.GL_FLOAT, False, 64, ctypes.c_void_p(offset))
        gl.glDrawElements(gl.GL_TRIANGLES, mesh["count"], gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))

    def _overlay(self, frame, settings):
        rows = []
        if settings["overlay"]:
            for segment in frame["segments"]:
                rows.extend([point + [33/255, 79/255, 80/255] for point in (segment["a"], segment["b"])])
        if settings["show_muscles"]:
            for muscle in frame["muscles"]:
                a = muscle["activation"]
                color = [(150 + 55*a)/255, (92 - 30*a)/255, (72 - 20*a)/255]
                rows.extend([point + color for point in (muscle["a"], muscle["b"])])
        return np.asarray(rows, np.float32).reshape(-1, 6)

    def _check_gpu(self):
        error = self.gl.glGetError()
        if error != self.gl.GL_NO_ERROR:
            raise AvatarRenderError("Offscreen avatar GPU error: " + str(error))

    def render(self, snapshot, settings):
        if self._closed or threading.get_ident() != self._owner:
            raise AvatarRenderError("Avatar renderer is closed or belongs to a different thread")
        try:
            return self._render(snapshot, settings)
        except BaseException:
            self.close()
            raise

    def _render(self, snapshot, settings):
        if self._frames >= MAX_FRAMES or time.monotonic() - self._started > MAX_SECONDS:
            raise AvatarRenderError("Avatar renderer exceeded its frame or lifetime bound")
        started = time.monotonic()
        _encoded(snapshot, 512 * 1024)
        frame = self._bridge.request({"type": "frame", "snapshot": snapshot, "settings": settings})
        if frame.get("type") != "frame":
            raise AvatarRenderError("Avatar pose bridge returned the wrong response")
        self._select(settings["asset"])
        gl, u = self.gl, self._uniforms
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._framebuffer)
        gl.glActiveTexture(gl.GL_TEXTURE1); gl.glBindTexture(gl.GL_TEXTURE_2D, self._palette)
        if frame["palette"]:
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, 4, len(frame["palette"]), gl.GL_RGBA, gl.GL_FLOAT, np.asarray(frame["palette"], np.float32))
        gl.glDisable(gl.GL_SCISSOR_TEST); gl.glClearColor(251/255, 247/255, 241/255, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        nodes = dict(frame["nodes"])
        overlay = self._overlay(frame, settings)
        if len(overlay):
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._line_buffer)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, overlay.nbytes, overlay, gl.GL_DYNAMIC_DRAW)
        segments = [(_segment_transform(s), s["radius"]) for s in frame["segments"]] if settings["asset"] is None else []
        for view in frame["views"]:
            gl.glViewport(*view["viewport"]); gl.glEnable(gl.GL_SCISSOR_TEST); gl.glScissor(*view["viewport"])
            gl.glUseProgram(self._program); gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glUniformMatrix4fv(u["uView"], 1, False, np.asarray(view["matrix"], np.float32))
            gl.glUniform1i(u["uPalette"], 1); gl.glUniform1i(u["uTexture"], 0)
            for mesh, matrix in zip(self._world_meshes, frame["world_matrices"]):
                self._material(mesh["primitive"]["color"])
                self._draw(mesh, matrix)
            if self._resident is not None:
                document = self._assets[self._resident]["document"]
                for mesh in self._asset_meshes:
                    primitive = mesh["primitive"]; material = document["materials"][primitive["material"]]
                    texture = None if material["base_color_texture"] is None else self._asset_textures[material["base_color_texture"]]
                    self._material(material["base_color_factor"], texture=texture, unlit=material["unlit"],
                        mask=material["alpha_mode"] == "MASK", cutoff=material["alpha_cutoff"], double_sided=material["double_sided"])
                    self._draw(mesh, nodes[primitive["node"]], skinned=primitive["skin"] is not None)
            else:
                color = [198/255, 165/255, 138/255] if frame["controlled_human"] else [155/255, 179/255, 176/255]
                self._material(_linear(color) + [1])
                for (matrix, half), radius in segments:
                    self._draw(self._capsule_mesh, matrix, capsule=(radius, half))
            if len(overlay):
                gl.glDisable(gl.GL_DEPTH_TEST); gl.glUseProgram(self._line_program)
                gl.glUniformMatrix4fv(self._line_uniforms["uView"], 1, False, np.asarray(view["matrix"], np.float32))
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._line_buffer)
                for index, offset in ((0, 0), (1, 12)):
                    gl.glEnableVertexAttribArray(index); gl.glVertexAttribPointer(index, 3, gl.GL_FLOAT, False, 24, ctypes.c_void_p(offset))
                gl.glDrawArrays(gl.GL_LINES, 0, len(overlay))
                if settings["overlay"]:
                    gl.glPointSize(3); gl.glDrawArrays(gl.GL_POINTS, 0, len(overlay))
        gl.glDisable(gl.GL_SCISSOR_TEST)
        raw = gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        self._check_gpu()
        if time.monotonic() - started > 2:
            raise AvatarRenderError("Offscreen frame exceeded its two-second work deadline")
        self._frames += 1; self._last_pose = frame
        return np.frombuffer(raw, np.uint8).reshape(self.height, self.width, 3)[::-1].copy()

    def close(self):
        if self._closed:
            return
        if threading.get_ident() != self._owner:
            raise AvatarRenderError("Close the avatar renderer on its owning thread")
        self._closed = True
        try:
            if self.gl and self._context and threading.get_ident() == self._owner:
                gl = self.gl
                if self._buffers:
                    gl.glDeleteBuffers(len(self._buffers), list(self._buffers))
                if self._textures:
                    gl.glDeleteTextures(list(self._textures))
                for program in self._programs:
                    gl.glDeleteProgram(program)
                if self._framebuffer:
                    gl.glDeleteFramebuffers(1, [self._framebuffer])
                if self._depth:
                    gl.glDeleteRenderbuffers(1, [self._depth])
        finally:
            if self._context:
                self._context.free(); self._context = None
            if self._bridge:
                self._bridge.close(); self._bridge = None
            if self._registered:
                with _ACTIVE_LOCK:
                    _ACTIVE_THREADS.discard(self._owner)
                self._registered = False
            self._buffers.clear(); self._textures.clear(); self._programs.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
