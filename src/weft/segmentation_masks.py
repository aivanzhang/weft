"""Bounded local decoding of saved Meta masks, separate from physical geometry."""

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import importlib.metadata
import json
import math
import multiprocessing
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image

from weft.benchmark import seal_suite
from weft.io import canonical_hash, digest, local_time, write_json
from weft.pipeline import _implementation_files
from weft.segmentation_evidence import _Bundle, _geometry
from weft.segmentation_scene import _normalize


CONTRACT = {
    'method': 'saved-sam-mask-raster-v1', 'codec': 'meta-sam-parser', 'codec_version': '0.1.0',
    'wheel_sha256': 'eca156907911c56cef3a350ffc81c34b7e70655e5fa3ccdddb77042c541a0152',
    'maximum_frames': 500, 'maximum_objects': 8, 'maximum_frame_mask_pixels': 8_847_360,
    'maximum_total_mask_pixels': 50_000_000, 'maximum_total_payload_characters': 32_000_000,
    'maximum_preview_dimension': 960, 'preview_jpeg_quality': 90,
    'maximum_export_bytes': 128_000_000, 'deadline_seconds': 60., 'cleanup_seconds': 2.,
    'mapping': 'Raster cell edges span the original half-open model box; centroid uses cell centers',
    'connectivity': 8, 'minimum_axis_eigenvalue_ratio': 1.1,
    'scope': 'Decoded model silhouettes, not measured boundaries, physical centers, identity or collision geometry',
}
COLORS = [(0, 215, 255), (255, 160, 40), (235, 100, 220), (90, 225, 110),
          (155, 130, 255), (255, 100, 100), (230, 220, 60), (140, 215, 210)]


def _codec_files():
    distribution = importlib.metadata.distribution(CONTRACT['codec'])
    if distribution.version != CONTRACT['codec_version']:
        raise ValueError('Mask decoding requires the pinned meta-sam-parser version')
    return {str(name): Path(distribution.locate_file(name)) for name in distribution.files
            if str(name).endswith('.py') or str(name).endswith('/licenses/LICENSE')}


def _plan(evidence):
    if evidence.get('profile') != 'segmentation-evidence-v1':
        raise ValueError('Decode only original saved segmentation evidence, not corrected coordinates')
    source, ids, _, times, rows, _, _ = _normalize(evidence)
    total_pixels = total_payload = 0
    for frame in range(len(times)):
        frame_pixels = 0
        for identifier in ids:
            row = rows[(frame, identifier)]
            raw = row['raw_model_output']
            if (not isinstance(raw, dict) or raw.get('frame') != frame
                    or raw.get('object_id') != row['raw_model_id']
                    or raw.get('status') != row['proposal_status']
                    or raw.get('source_dimensions') != [source['width_px'], source['height_px']]):
                raise ValueError('Raw mask identity, dimensions or status differs from its observation')
            geometry, confidences, mask = _geometry(raw, source['width_px'], source['height_px'])
            if (geometry != row['geometry'] or confidences != row['confidence']
                    or mask != row['mask_provenance']):
                raise ValueError('Saved mask geometry or provenance differs from the raw provider output')
            if mask is not None:
                frame_pixels += math.prod(mask['dimensions'])
                total_payload += len(raw['mask_encoded'])
        if frame_pixels > CONTRACT['maximum_frame_mask_pixels']:
            raise ValueError('Combined frame masks exceed the pixel bound')
        total_pixels += frame_pixels
    if total_pixels > CONTRACT['maximum_total_mask_pixels'] or total_payload > CONTRACT['maximum_total_payload_characters']:
        raise ValueError('Mask study exceeds its total decode bound')
    return source, ids, times, rows, {'mask_pixels': total_pixels, 'payload_characters': total_payload}


def decode_mask(raw):
    from meta_sam_parser import SegmentationMask, decode_mask_to_raster

    dimensions, encoded = raw['mask_dimensions'], raw['mask_encoded']
    if (not isinstance(dimensions, list) or len(dimensions) != 2
            or any(type(n) is not int or not 1 <= n <= 4096 for n in dimensions)
            or math.prod(dimensions) > CONTRACT['maximum_frame_mask_pixels']
            or not isinstance(encoded, str) or not 6 <= len(encoded) <= 250_000):
        raise ValueError('Mask dimensions or payload exceed bounds')
    raster = decode_mask_to_raster(SegmentationMask(encoding=raw['mask_encoding'], payload=encoded,
                                                   width=dimensions[1], height=dimensions[0]))
    if len(raster) != math.prod(dimensions):
        raise ValueError('Decoded mask length differs from its dimensions')
    array = np.frombuffer(raster, dtype=np.uint8).reshape(dimensions)
    if np.any(array > 1):
        raise ValueError('Decoded raster is not binary')
    return array


def mask_geometry(raster, box):
    if (not isinstance(raster, np.ndarray) or raster.ndim != 2 or raster.dtype != np.uint8
            or not raster.size or raster.size > CONTRACT['maximum_frame_mask_pixels'] or np.any(raster > 1)):
        raise ValueError('Geometry requires a bounded binary uint8 raster')
    if (not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box)
            or not 0 <= box[0] < box[2] <= 4096 or not 0 <= box[1] < box[3] <= 4096):
        raise ValueError('Geometry requires a bounded half-open source box')
    height, width = raster.shape
    sx, sy = (box[2]-box[0])/width, (box[3]-box[1])/height
    moments = cv2.moments(raster, binaryImage=True)
    count = int(moments['m00'])
    result = {'foreground_pixels': count, 'raster_pixels': raster.size, 'occupancy_fraction': count/raster.size,
              'mapped_area_px2': count*sx*sy, 'centroid_px': None, 'foreground_bounds_px': None,
              'components_8_connected': 0, 'principal_axis_radians': None,
              'principal_axis_scope': 'Unoriented projected silhouette axis modulo pi; not anatomical or rigid-body orientation'}
    if not count:
        return result
    cx, cy = moments['m10']/count, moments['m01']/count
    result['centroid_px'] = [box[0]+(cx+.5)*sx, box[1]+(cy+.5)*sy]
    x, y, w, h = cv2.boundingRect(raster)
    result['foreground_bounds_px'] = [box[0]+x*sx, box[1]+y*sy, box[0]+(x+w)*sx, box[1]+(y+h)*sy]
    result['components_8_connected'] = int(cv2.connectedComponents(raster, connectivity=8)[0])-1
    # Cell variance preserves a rectangle's shape when the raster is very coarse.
    covariance = np.array([[moments['mu20']/count*sx*sx+sx*sx/12, moments['mu11']/count*sx*sy],
                           [moments['mu11']/count*sx*sy, moments['mu02']/count*sy*sy+sy*sy/12]])
    values, vectors = np.linalg.eigh(covariance)
    if values[1] >= CONTRACT['minimum_axis_eigenvalue_ratio']*values[0]:
        result['principal_axis_radians'] = math.atan2(vectors[1, 1], vectors[0, 1]) % math.pi
    return result


def _write_raster(path, raster, color):
    image = Image.frombytes('P', (raster.shape[1], raster.shape[0]), raster.tobytes())
    image.putpalette([0, 0, 0, *color] + [0]*762)
    image.save(path, transparency=bytes([0, 120]), optimize=False)


def _decode_run(out):
    from meta_sam_parser import InvalidSegmentationMaskError

    started = time.monotonic()
    cv2.setNumThreads(1)
    evidence = json.loads((out/'original-evidence.json').read_text())
    source, ids, times, rows, demand = _plan(evidence)
    width, height = source['width_px'], source['height_px']
    scale = min(1., CONTRACT['maximum_preview_dimension']/max(width, height))
    preview_size = (round(width*scale), round(height*scale))
    (out/'frames').mkdir()
    (out/'masks').mkdir()
    capture = cv2.VideoCapture(str(out/'source.mp4'))
    frames, observations, statuses, export_bytes, decode_times = [], [], Counter(), 0, []
    try:
        for frame, stamp in enumerate(times):
            ok, image = capture.read()
            if not ok or image.shape[:2] != (height, width):
                raise ValueError('Source video frame count or dimensions differ from saved evidence')
            actual = capture.get(cv2.CAP_PROP_POS_MSEC)/1000
            if not math.isfinite(actual) or not math.isclose(actual, stamp, rel_tol=0, abs_tol=1e-6):
                raise ValueError('Source video clock differs from saved evidence')
            preview = cv2.resize(image, preview_size, interpolation=cv2.INTER_AREA) if scale < 1 else image
            ok, jpeg = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, CONTRACT['preview_jpeg_quality']])
            if not ok:
                raise ValueError('Could not encode source-frame preview')
            preview_path = f'frames/{frame:05d}.jpg'
            (out/preview_path).write_bytes(jpeg.tobytes())
            export_bytes += jpeg.nbytes
            indices = []
            for index, identifier in enumerate(ids):
                row = rows[(frame, identifier)]
                result = {key: deepcopy(row[key]) for key in ('id', 'frame', 'time_s', 'object_id', 'raw_model_id',
                          'phase', 'proposal_status', 'identity', 'confidence')}
                if 'original_source' in row:
                    result['original_source'] = deepcopy(row['original_source'])
                result.update(raw_observation_sha256=canonical_hash(row), mask_status='not_returned',
                              box=deepcopy(row['geometry']), raster_path=None, raster_sha256=None, silhouette=None,
                              encoded_sha256=None, color=list(COLORS[index]))
                if row['proposal_status'] == 'returned':
                    tick = time.monotonic()
                    result['encoded_sha256'] = row['mask_provenance']['encoded_sha256']
                    try:
                        raster = decode_mask(row['raw_model_output'])
                    except InvalidSegmentationMaskError:
                        result.update(mask_status='decode_failed', error='InvalidSegmentationMaskError')
                    else:
                        result.update(mask_status='decoded', silhouette=mask_geometry(raster, row['geometry']['box_xyxy_halfopen']),
                                      raster_sha256=sha256(raster.tobytes()).hexdigest(), raster_dimensions=list(raster.shape))
                        result['raster_path'] = f'masks/{frame:05d}-{index}.png'
                        _write_raster(out/result['raster_path'], raster, COLORS[index])
                        export_bytes += (out/result['raster_path']).stat().st_size
                    decode_times.append(time.monotonic()-tick)
                statuses[result['mask_status']] += 1
                indices.append(len(observations))
                observations.append(result)
            frames.append({'frame': frame, 'time_s': stamp, 'preview': preview_path, 'observation_indices': indices})
            if export_bytes > CONTRACT['maximum_export_bytes']:
                raise ValueError('Mask/preview exports exceeded the output bound; partial artifacts retained')
        if capture.read()[0]:
            raise ValueError('Source video contains unreported suffix frames')
        if digest(out/'source.mp4') != source['submitted_video_sha256']:
            raise ValueError('Source video changed during mask inspection')
    finally:
        capture.release()
    document = {'profile': 'decoded-segmentation-masks-v1', 'contract': deepcopy(CONTRACT), 'source': source,
                'original_evidence_sha256': canonical_hash(evidence), 'frames': frames, 'observations': observations,
                'physics_ready': False, 'identity_groups': deepcopy(evidence['identity_groups']),
                'preview': {'dimensions': list(preview_size), 'format': 'JPEG',
                            'scope': 'Sequentially decoded source frames; downscaled/lossy inspection previews, not grading pixels'}}
    write_json(out/'mask-evidence.json', document)
    return {'status': 'decoded_proposals' if not statuses['decode_failed'] else 'partial_decode',
            'frames': len(frames), 'observations': len(observations), 'counts': dict(statuses),
            'demand': demand, 'export_bytes': export_bytes, 'runtime_seconds': time.monotonic()-started,
            'decode_geometry_png_ms': {'samples': len(decode_times),
                'median': float(np.median(decode_times)*1000) if decode_times else None,
                'p95': float(np.percentile(decode_times, 95)*1000) if decode_times else None}}


def _worker(pipe, out):
    try:
        pipe.send(('done', _decode_run(Path(out))))
    except Exception as error:
        pipe.send(('error', type(error).__name__, str(error)[:300]))
    finally:
        pipe.close()


def _run_bounded(out):
    context = multiprocessing.get_context('spawn')
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(send, str(out)), daemon=True)
    try:
        process.start()
        send.close()
        if not receive.poll(CONTRACT['deadline_seconds']):
            raise TimeoutError('Mask decoding exceeded its worker deadline; partial exports retained')
        event = receive.recv()
        if event[0] != 'done':
            raise ValueError('Mask decoding failed: ' + ': '.join(event[1:]))
        return event[1]
    finally:
        receive.close()
        send.close()
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=CONTRACT['cleanup_seconds']/2)
            if process.is_alive():
                process.kill()
                process.join(timeout=CONTRACT['cleanup_seconds']/2)
            process.close()


def _viewer(out):
    document = json.loads((out/'mask-evidence.json').read_text())
    # Decode and geometry stay in the worker; the viewer loads only one frame's images.
    payload = {key: document[key] for key in ('frames', 'observations', 'source', 'preview')}
    page = (Path(__file__).parent/'assets/mask-viewer.html').read_text()
    (out/'index.html').write_text(page.replace('MASK_DATA', json.dumps(payload, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')))
    shutil.copyfile(Path(__file__).parent/'assets/mask-viewer.js', out/'mask-viewer.js')


def export_masks(run, video, out):
    bundle = _Bundle(run, ('report.json', 'segmentation-evidence.json'))
    evidence = bundle.read('segmentation-evidence.json')
    source, _, _, _, demand = _plan(evidence)
    video, out = Path(video).resolve(), Path(out).resolve()
    if out.exists() or out.is_relative_to(bundle.root):
        raise ValueError('Choose a new mask directory outside the immutable source evidence')
    if not video.is_file() or not 1 <= video.stat().st_size <= 37_000_000 or digest(video) != source['submitted_video_sha256']:
        raise ValueError('Use the exact bounded source video from the segmentation request')
    codec_files = _codec_files()
    implementation = _implementation_files()
    hashes = {name: digest(path) for name, path in implementation.items()}
    codec_hashes = {name: digest(path) for name, path in codec_files.items()}
    out.mkdir(parents=True)
    write_json(out/'original-evidence.json', evidence)
    shutil.copyfile(video, out/'source.mp4')
    if digest(out/'source.mp4') != source['submitted_video_sha256']:
        raise ValueError('Video changed while preparing mask inspection')
    write_json(out/'source-verification.json', bundle.provenance())
    report = {'created_at': local_time(), 'status': 'prepared', 'contract': deepcopy(CONTRACT), 'demand': demand,
              'source_run': str(bundle.root), 'source_verification': bundle.provenance(),
              'implementation_hashes': hashes, 'codec_hashes': codec_hashes, 'paid_requests': 0,
              'physics_ready': False, 'scope': CONTRACT['scope']}
    write_json(out/'pre-run.json', report)
    try:
        for category, files in (('implementation-source', implementation), ('dependency-source', codec_files)):
            for name, path in files.items():
                destination = out/category/name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
                expected = hashes[name] if category == 'implementation-source' else codec_hashes[name]
                if digest(destination) != expected:
                    raise ValueError('Implementation changed before mask decoding')
        report.update(_run_bounded(out))
        _viewer(out)
        if (_implementation_files() != implementation or _codec_files() != codec_files
                or any(digest(path) != hashes[name] for name, path in implementation.items())
                or any(digest(path) != codec_hashes[name] for name, path in codec_files.items())):
            raise ValueError('Implementation changed during mask decoding')
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, reason=str(error)[:300])
        raise
    finally:
        write_json(out/'report.json', report)
        seal_suite(out)
    return {key: value for key, value in report.items() if key not in ('implementation_hashes', 'codec_hashes', 'source_verification')}
