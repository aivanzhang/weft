"""Bounded color-only OIDN post-process for locally rendered sRGB PNGs; never image generation.

The public CLI creates a new job directory and launches the isolated worker through
the existing RSS/wall/output monitor. Native code is restricted to bundled Blender OIDN.
"""

import argparse
import ctypes
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import struct
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
MAX_INPUT_BYTES = 32*1024*1024
MAX_EDGE = 2048
MAX_OUTPUT_BYTES = 64*1024*1024
WALL_SECONDS = 90
RSS_MB = 2048
THREADS = 4
FILTER_MEMORY_MB = 768
LIBRARY_NAMES = ("libOpenImageDenoise.dylib", "libOpenImageDenoise_core.dylib",
                 "libOpenImageDenoise_device_cpu.dylib", "libtbb.dylib")
LIBRARY_ROOT = ROOT/".tools/Blender.app/Contents/Resources/lib"
API_HEADER = "https://github.com/RenderKit/oidn/blob/v2.3.3/include/OpenImageDenoise/oidn.h"
API_DOCUMENTATION = "https://github.com/RenderKit/oidn/blob/v2.3.3/doc/api.md"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_png(path):
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), "Input must be a regular, non-symlink local PNG")
    require(0 < path.stat().st_size <= MAX_INPUT_BYTES, "PNG input exceeds 32 MiB")
    with path.open("rb") as stream:
        header = stream.read(33)
    require(len(header) == 33 and header[:8] == b"\x89PNG\r\n\x1a\n"
            and header[8:16] == b"\x00\x00\x00\rIHDR", "Input is not a PNG with a valid IHDR")
    width, height, bits, channels, compression, filtering, interlace = struct.unpack(">IIBBBBB", header[16:29])
    require(1 <= width == height <= MAX_EDGE, "PNG must be square and at most 2048 pixels per edge")
    require(bits == 8 and channels in (2, 6), "Only 8-bit RGB/RGBA PNG input is supported")
    require(compression == filtering == 0 and interlace in (0, 1), "Unsupported PNG encoding")
    return {"path":str(path.resolve()), "bytes":path.stat().st_size, "width":width,"height":height,
            "mode":"RGBA" if channels == 6 else "RGB", "sha256":sha256(path)}


def read_pixels(path):
    import numpy as np
    from PIL import Image, ImageFile

    info = inspect_png(path)
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_INPUT_BYTES+1)
    require(len(raw) <= MAX_INPUT_BYTES and hashlib.sha256(raw).hexdigest() == info["sha256"], "Input changed during bounded read")
    old_limit, old_truncated = Image.MAX_IMAGE_PIXELS, ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        Image.MAX_IMAGE_PIXELS = MAX_EDGE*MAX_EDGE
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                require(image.format == "PNG" and image.mode == info["mode"]
                        and image.size == (info['width'], info['height']), "PNG decoder differs from bounded IHDR")
                require(getattr(image,"n_frames",1) == 1, "Animated PNG input is unsupported")
                require(not image.info.get("icc_profile"), "Embedded ICC input is unsupported; export an sRGB PNG")
                gamma = image.info.get("gamma")
                require(gamma is None or abs(float(gamma)-1/2.2) < .001, "PNG gamma is not compatible with declared sRGB input")
                image.load()
                pixels = np.array(image, dtype=np.uint8, copy=True)
                dpi = image.info.get("dpi")
    finally:
        Image.MAX_IMAGE_PIXELS, ImageFile.LOAD_TRUNCATED_IMAGES = old_limit, old_truncated
    info["color_space"] = "8-bit display-referred sRGB, explicitly selected; no ICC transform"
    info["gamma_chunk"] = gamma
    if dpi and len(dpi) == 2 and all(math.isfinite(value) and 0 < value <= 10000 for value in dpi):
        info["dpi"] = list(dpi)
    info["alpha_sha256"] = hashlib.sha256(pixels[:,:,3].tobytes()).hexdigest() if pixels.shape[2] == 4 else None
    return pixels, info


def library_inventory():
    require(sys.platform == "darwin", "This worker supports the installed macOS Blender OIDN only")
    result = []
    for name in LIBRARY_NAMES:
        path = LIBRARY_ROOT/name
        require(not path.is_symlink() and path.is_file() and 0 < path.stat().st_size <= 64*1024*1024,
                "Required bundled OIDN library is unavailable: "+name)
        result.append({"path":str(path.resolve()),"bytes":path.stat().st_size,"sha256":sha256(path)})
    return result


class OIDN:
    def __init__(self):
        self.libraries = library_inventory()
        self.library = ctypes.CDLL(str(LIBRARY_ROOT/LIBRARY_NAMES[0]))
        ptr, string, integer, size = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_size_t
        signatures = {
            "oidnNewDevice":([integer],ptr), "oidnCommitDevice":([ptr],None),
            "oidnReleaseDevice":([ptr],None), "oidnSetDeviceInt":([ptr,string,integer],None),
            "oidnGetDeviceInt":([ptr,string],integer), "oidnGetDeviceBool":([ptr,string],ctypes.c_bool),
            "oidnGetDeviceError":([ptr,ctypes.POINTER(string)],integer),
            "oidnNewFilter":([ptr,string],ptr), "oidnReleaseFilter":([ptr],None),
            "oidnSetSharedFilterImage":([ptr,string,ptr,integer,size,size,size,size,size],None),
            "oidnSetFilterBool":([ptr,string,ctypes.c_bool],None),
            "oidnSetFilterInt":([ptr,string,integer],None), "oidnGetFilterInt":([ptr,string],integer),
            "oidnGetFilterBool":([ptr,string],ctypes.c_bool),
            "oidnCommitFilter":([ptr],None), "oidnExecuteFilter":([ptr],None),
        }
        for name,(args,result) in signatures.items():
            function = getattr(self.library,name)
            function.argtypes, function.restype = args,result

    def check(self, device, phase):
        message = ctypes.c_char_p()
        code = self.library.oidnGetDeviceError(device,ctypes.byref(message))
        if code:
            reason = message.value.decode("utf-8","replace") if message.value else "No native error text"
            raise RuntimeError(f"OIDN {phase} failed ({code}): {reason[:2048]}")

    def apply(self, rgb):
        import numpy as np

        require(rgb.dtype == np.float32 and rgb.ndim == 3 and rgb.shape[2] == 3
                and rgb.flags.c_contiguous, "OIDN color input must be contiguous float32 RGB")
        require(np.isfinite(rgb).all() and float(rgb.min()) >= 0 and float(rgb.max()) <= 1, "OIDN LDR input must be finite in [0,1]")
        lib, device, image_filter = self.library, None, None
        started = time.monotonic()
        try:
            device = lib.oidnNewDevice(1)  # OIDN_DEVICE_TYPE_CPU, never automatic GPU selection.
            self.check(device,"create CPU device")
            require(device, "OIDN CPU device was not created")
            lib.oidnSetDeviceInt(device,b"numThreads",THREADS)
            lib.oidnCommitDevice(device)
            self.check(device,"commit CPU device")
            version = [lib.oidnGetDeviceInt(device,key) for key in (b"versionMajor",b"versionMinor",b"versionPatch")]
            require(version[0] == 2, "Unsupported OIDN major API version")
            require(lib.oidnGetDeviceInt(device,b"type") == 1
                    and lib.oidnGetDeviceInt(device,b"numThreads") == THREADS
                    and lib.oidnGetDeviceBool(device,b"systemMemorySupported"), "CPU/thread/shared-memory contract differs")
            self.check(device,"read CPU properties")
            output = np.empty_like(rgb)
            image_filter = lib.oidnNewFilter(device,b"RT")
            self.check(device,"create RT filter")
            require(image_filter,"OIDN RT filter was not created")
            height,width = rgb.shape[:2]
            for name,array in ((b"color",rgb),(b"output",output)):
                lib.oidnSetSharedFilterImage(image_filter,name,array.ctypes.data,3,width,height,0,12,width*12)
            lib.oidnSetFilterBool(image_filter,b"hdr",False)
            lib.oidnSetFilterBool(image_filter,b"srgb",True)
            lib.oidnSetFilterInt(image_filter,b"quality",6)
            lib.oidnSetFilterInt(image_filter,b"maxMemoryMB",FILTER_MEMORY_MB)
            lib.oidnCommitFilter(image_filter)
            self.check(device,"commit filter")
            require(not lib.oidnGetFilterBool(image_filter,b"hdr") and lib.oidnGetFilterBool(image_filter,b"srgb")
                    and lib.oidnGetFilterInt(image_filter,b"quality") == 6, "OIDN filter settings differ")
            self.check(device,"verify filter settings")
            lib.oidnExecuteFilter(image_filter)
            self.check(device,"execute filter")
            require(np.isfinite(output).all(), "OIDN output contains nonfinite values")
            return output, {"name":"Open Image Denoise","version":".".join(map(str,version)),
                            "libraries":self.libraries,"filter":"RT","device":"CPU","threads":THREADS,
                            "hdr":False,"srgb":True,"quality":"high","quality_enum":6,
                            "maxMemoryMB_requested":FILTER_MEMORY_MB,"memory_request_is_hard_limit":False,
                            "auxiliary_images":[],"native_seconds":round(time.monotonic()-started,6),
                            "api_header":API_HEADER,"api_documentation":API_DOCUMENTATION}
        finally:
            if image_filter:
                lib.oidnReleaseFilter(image_filter)
            if device:
                lib.oidnReleaseDevice(device)


def compose_pixels(original, filtered):
    import numpy as np

    require(filtered.shape == (*original.shape[:2],3) and filtered.dtype == np.float32
            and np.isfinite(filtered).all(), "Filtered output shape/type/values differ")
    result = original.copy()
    result[:,:,:3] = np.rint(np.clip(filtered,0,1)*255).astype(np.uint8)
    # Fully transparent pixel colors are unobservable, but preserve them for exact alpha custody.
    if original.shape[2] == 4:
        result[original[:,:,3] == 0] = original[original[:,:,3] == 0]
    return result


def write_png(path, pixels, source_info):
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    info.add(b"sRGB",b"\x00")
    info.add_text("PostProcess","Open Image Denoise RT; color-only sRGB LDR; native render source preserved")
    with Path(path).open("xb") as stream:
        Image.fromarray(pixels).save(stream,format="PNG",pnginfo=info,compress_level=6,
                                     **({"dpi":source_info["dpi"]} if "dpi" in source_info else {}))
    require(Path(path).stat().st_size <= MAX_INPUT_BYTES, "Denoised PNG exceeds 32 MiB")


def worker(source, output, result_path):
    import numpy as np

    started = time.monotonic()
    source, output, result_path = map(Path,(source,output,result_path))
    require(output.suffix.lower() == ".png" and output.parent.resolve() == result_path.parent.resolve(), "Worker output/report must share a job folder")
    require(not output.exists() and not output.is_symlink() and not result_path.exists() and not result_path.is_symlink(), "Worker output/report already exists")
    original, source_info = read_pixels(source)
    rgb = np.ascontiguousarray(original[:,:,:3],dtype=np.float32)
    rgb *= np.float32(1/255)
    filtered, native = OIDN().apply(rgb)
    result = compose_pixels(original,filtered)
    require(sha256(source) == source_info["sha256"], "Raw input changed during denoising")
    write_png(output,result,source_info)
    output_info = inspect_png(output)
    reread, _ = read_pixels(output)
    require(np.array_equal(result,reread), "Written PNG pixels differ from processed pixels")
    alpha_unchanged = original.shape[2] != 4 or np.array_equal(original[:,:,3],reread[:,:,3])
    require(alpha_unchanged,"Alpha changed during denoising")
    change = result[:,:,:3].astype(np.float32)-original[:,:,:3]
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak *= 1024
    report = {"schema_version":1,"kind":"render_portrait_denoise","status":"needs_visual_review",
              "input":source_info,"output":output_info,"alpha_unchanged":alpha_unchanged,
              "raw_input_unchanged":True,"native":native,"wall_seconds":round(time.monotonic()-started,6),
              "peak_rss_bytes":int(peak),"rgb_mean_absolute_change_8bit":float(np.mean(np.abs(change))),
              "rgb_rmse_change_8bit":float(np.sqrt(np.mean(change*change))),
              "change_metrics_are_quality_scores":False,"worker_source_sha256":sha256(__file__),
              "provider_calls":0,"provider_cost_usd":0,"reference_images_used":[],
              "limitations":["Color-only display-referred denoising; no albedo/normal guides.",
                             "Fine hair or texture can be softened; compare this result with the unchanged raw render.",
                             "No geometry, rig, texture source or render-camera change; not evidence of Muse parity."]}
    with result_path.open("x") as stream:
        json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n")
    return report


def output_plan(output, report):
    output, report = Path(output), Path(report)
    require(output.suffix.lower() == ".png" and report.suffix.lower() == ".json", "Choose explicit PNG output and JSON report paths")
    require(output.parent.resolve() == report.parent.resolve(), "Output and report must share one new job directory")
    require(output.name not in {"denoise.log","worker-result.json","denoise-handle.json"}
            and report.name not in {"worker-result.json","denoise-handle.json"}, "Output/report name is reserved")
    require(not output.exists() and not output.is_symlink() and not report.exists() and not report.is_symlink(), "Output/report already exists")
    require(not output.parent.exists() and not output.parent.is_symlink(), "Use a new job directory; existing folders are preserved")
    return output.absolute(), report.absolute()


def run(source, output, report):
    if str(ROOT) not in sys.path:
        sys.path.insert(0,str(ROOT))
    from avatar_assets.runner import run_bounded
    from avatar_assets.safety import new_directory,write_json

    source_info = inspect_png(source)
    libraries = library_inventory()
    output, report = output_plan(output,report)
    job_dir = new_directory(output.parent)
    worker_result = job_dir/"worker-result.json"
    # No provider credentials or injected native-library/python paths reach the worker.
    command = ["/usr/bin/env","-i","PATH=/usr/bin:/bin:/usr/sbin:/sbin","LANG=en_US.UTF-8",
               "PYTHONDONTWRITEBYTECODE=1","OMP_NUM_THREADS=4","OPENBLAS_NUM_THREADS=1","VECLIB_MAXIMUM_THREADS=1",
               str(Path(sys.executable).absolute()),str(Path(__file__).resolve()),"--worker",
               "--input",source_info["path"],"--output",str(output),"--report",str(worker_result)]
    job = run_bounded(command,job_dir/"denoise.log",job_dir,timeout=WALL_SECONDS,memory_mb=RSS_MB,output_mb=64)
    valid = job["exit_code"] == 0 and not job["failure"] and worker_result.is_file() and output.is_file()
    result = {"schema_version":1,"status":"needs_visual_review" if valid else "failed",
              "job":job,"input":source_info,"libraries":libraries,"worker":None,
              "raw_input_unchanged":sha256(source_info["path"]) == source_info["sha256"],
              "provider_calls":0,"provider_cost_usd":0}
    if worker_result.is_file():
        require(worker_result.stat().st_size <= 1024*1024,"Worker report exceeds bound")
        result["worker"] = json.loads(worker_result.read_text())
    if not result["raw_input_unchanged"]:
        result["status"] = "failed"
    write_json(report,result)
    require(result["status"] != "failed",f"Denoise failed; evidence preserved at {report}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Denoise one locally rendered sRGB PNG in a bounded CPU worker; raw input is never replaced")
    parser.add_argument("--input",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--report",required=True)
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        resource.setrlimit(resource.RLIMIT_CPU,(THREADS*WALL_SECONDS,THREADS*WALL_SECONDS+5))
        resource.setrlimit(resource.RLIMIT_FSIZE,(MAX_OUTPUT_BYTES,MAX_OUTPUT_BYTES))
        result = worker(args.input,args.output,args.report)
    else:
        result = run(args.input,args.output,args.report)
    print(json.dumps({"status":result["status"],"output":str(Path(args.output).resolve()),
                      "report":str(Path(args.report).resolve())}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status":"failed","error":str(error)[:4096]}),file=sys.stderr)
        raise SystemExit(1)
