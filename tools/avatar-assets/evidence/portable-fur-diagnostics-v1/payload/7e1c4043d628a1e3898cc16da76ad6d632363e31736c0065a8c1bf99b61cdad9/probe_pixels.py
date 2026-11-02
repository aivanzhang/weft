"""Measure image.pixels only; no render, source-image save or color-policy decision."""
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    started = time.monotonic()
    output = HERE/'pixel-observations.json'
    require(not output.exists(), 'Never overwrite a color probe')
    config = json.loads((HERE/'samples.json').read_text())
    require(len(config['images']) == 2, 'Probe requires exactly two pinned images')
    results = []
    for entry in config['images']:
        path = HERE/entry['path']
        require(path.parent == HERE/'inputs' and sha(path) == entry['sha256'], 'Pinned texture changed')
        require(1 <= len(entry['samples']) <= 16, 'Sample count exceeds16 per image')
        image = bpy.data.images.load(str(path), check_existing=False)
        try:
            width, height = tuple(image.size)
            require([width, height] == entry['dimensions'] and width*height <= 16777216, 'Image dimensions changed')
            loaded_space = image.colorspace_settings.name
            image.colorspace_settings.name = 'sRGB'
            count = len(image.pixels)
            require(count % (width*height) == 0, 'Unexpected pixel array length')
            stride = count//(width*height)
            require(stride in (3, 4), 'Unsupported image.pixels stride')
            full = None
            if entry['full_foreach_get']:
                require(width*height <= 256, 'Full foreach_get is restricted to the tiny analytic texture')
                full = np.empty(count, dtype=np.float32)
                image.pixels.foreach_get(full)
            observed = []
            for sample in entry['samples']:
                require(time.monotonic()-started < 55, 'Probe inner deadline exceeded')
                x, y = sample['pixel_bottom_left_xy']
                require(type(x) is int and type(y) is int and 0 <= x < width and 0 <= y < height, 'Invalid pinned location')
                offset = stride*(y*width+x)
                values = [float(value) for value in image.pixels[offset:offset+stride]]
                require(len(values) == stride and all(math.isfinite(v) for v in values), 'Nonfinite or incomplete pixels')
                rgb = np.asarray(values[:3], dtype=np.float64)
                errors = {key: float(np.max(np.abs(rgb-np.asarray(sample[key], dtype=np.float64))))
                    for key in ('encoded_rgb', 'linear_rgb', 'opposite_row_encoded_rgb', 'opposite_row_linear_rgb')}
                observed.append({'id': sample['id'], 'pixel_top_left_xy': sample['pixel_top_left_xy'],
                    'queried_array_xy': [x, y], 'array_component_offset': offset,
                    'observed_components': values, 'maximum_absolute_error': errors,
                    'foreach_get_maximum_absolute_error': None if full is None else
                        float(np.max(np.abs(np.asarray(values)-full[offset:offset+stride])))})
            results.append({'id': entry['id'], 'classification': entry['classification'],
                'input_sha256': entry['sha256'], 'dimensions': [width, height],
                'loaded_color_space': loaded_space, 'queried_color_space': image.colorspace_settings.name,
                'image_source': image.source, 'image_is_float': bool(image.is_float), 'declared_channels': image.channels,
                'array_components_per_pixel': stride, 'pipeline_four_component_expectation_met': stride == 4,
                'full_foreach_get': bool(entry['full_foreach_get']),
                'maximum_python_full_buffer_bytes': 0 if full is None else full.nbytes,
                'samples': observed,
                'maximum_error_by_hypothesis': {key: max(r['maximum_absolute_error'][key] for r in observed)
                    for key in ('encoded_rgb', 'linear_rgb', 'opposite_row_encoded_rgb', 'opposite_row_linear_rgb')}})
            del full
        finally:
            image.buffers_free()
            bpy.data.images.remove(image)
            gc.collect()
        require(sha(path) == entry['sha256'], 'Probe texture bytes changed')
    report = {'schema_version': 1, 'status': 'pixel_measurements_complete_interpretation_pending',
        'blender_version': bpy.app.version_string, 'blender_build_hash': bpy.app.build_hash.decode(),
        'numpy_version': np.__version__, 'scene_view_transform': bpy.context.scene.view_settings.view_transform,
        'images': results, 'samples_config_sha256': sha(HERE/'samples.json'),
        'source_model_sha256': config['model_sha256'], 'renders': 0, 'source_images_saved': 0,
        'provider_calls': 0, 'quality_approved': False, 'elapsed_seconds': time.monotonic()-started,
        'scope': 'Known lossless PNG and sparse Pillow-decoded actual JPEG texels under explicit sRGB; row orientation and encoded/linear hypotheses are reported, not assumed.',
        'limitations': ['No full4K foreach_get allocation; indexed image.pixels samples only for the actual source.',
            'No GLB import, native groom, shading, renderer, full sampler throughput or aesthetic test.',
            'JPEG decoder rounding can differ; lossless known-byte PNG supplies the exact transfer/orientation control.']}
    encoded = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    require(len(encoded) <= 256*1024, 'Observation report exceeds256KiB')
    with output.open('xb') as handle:
        handle.write(encoded)
    print(json.dumps({'status': report['status'], 'images': len(results), 'samples': sum(len(r['samples']) for r in results),
        'elapsed_seconds': report['elapsed_seconds']}))


if __name__ == '__main__':
    main()
