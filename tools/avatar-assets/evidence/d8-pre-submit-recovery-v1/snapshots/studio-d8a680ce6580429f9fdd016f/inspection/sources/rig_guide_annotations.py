"""Coordinate overlays on native model renders; no inferred anatomy or image generation."""

import hashlib
import json
import math
from pathlib import Path
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def guide_projection(bounds, view, size=640, margin=64):
    require(view in ('front', 'side'), 'Unknown rig-guide view')
    require(type(size) is int and 128 <= size <= 640 and type(margin) is int and 32 <= margin <= 64,
            'Rig-guide canvas exceeds bounds')
    require(isinstance(bounds, dict) and set(bounds) == {'min', 'max'}, 'Invalid rig-guide bounds')
    low, high = bounds['min'], bounds['max']
    require(isinstance(low, list) and isinstance(high, list) and len(low) == len(high) == 3
            and all(type(v) in (int, float) and math.isfinite(v) for v in low+high), 'Nonfinite rig-guide bounds')
    spans = [b-a for a, b in zip(low, high)]
    require(all(1e-8 < span < 1e8 for span in spans), 'Degenerate or excessive rig-guide extent')
    horizontal = 0 if view == 'front' else 1
    center = [(a+b)/2 for a, b in zip(low, high)]
    scale = max(spans[horizontal], spans[2])*1.18
    return {'view': view, 'horizontal_axis': 'X' if horizontal == 0 else 'Y', 'vertical_axis': 'Z',
            'hidden_axis': 'Y' if horizontal == 0 else 'X', 'positive_horizontal': 'right', 'positive_vertical': 'up',
            'horizontal_index': horizontal, 'bounds_blender_z_up': bounds,
            'ortho_scale': scale, 'camera_target_blender_world': center,
            'projection_bounds_blender_world': {'horizontal_min': center[horizontal]-scale/2,
                                              'horizontal_max': center[horizontal]+scale/2,
                                              'vertical_min': center[2]-scale/2, 'vertical_max': center[2]+scale/2},
            'raw_image_pixels': [size, size], 'canvas_pixels': [size+2*margin, size+2*margin],
            'raw_image_offset_pixels': [margin, margin], 'normalized_ticks': [0, .25, .5, .75, 1],
            'pixel_convention': 'Continuous image-edge coordinates; origin top-left, X right and Y down',
            'normalization': '(world_position - mesh_bounds.min) / (mesh_bounds.max - mesh_bounds.min), per axis'}


def normalized_to_pixel(point, projection, canvas=False):
    require(isinstance(point, (list, tuple)) and len(point) == 3
            and all(type(v) in (int, float) and math.isfinite(v) for v in point), 'Invalid normalized point')
    bounds, box = projection['bounds_blender_z_up'], projection['projection_bounds_blender_world']
    world = [a+v*(b-a) for a, b, v in zip(bounds['min'], bounds['max'], point)]
    size_x, size_y = projection['raw_image_pixels']
    x = (world[projection['horizontal_index']]-box['horizontal_min'])/(box['horizontal_max']-box['horizontal_min'])*size_x
    y = (box['vertical_max']-world[2])/(box['vertical_max']-box['vertical_min'])*size_y
    offset = projection['raw_image_offset_pixels'] if canvas else [0, 0]
    return [x+offset[0], y+offset[1]]


def annotate(output):
    from PIL import Image, ImageDraw, ImageFont

    out = Path(output).resolve()
    path = out/'native-inspection.json'
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 1024*1024, 'Invalid native guide metadata')
    native = json.loads(path.read_text())
    guides = native['rig_guides']
    require(guides['source_sha256'] == digest(out/'original.glb'), 'Rig-guide source changed')
    require([v['view'] for v in guides['views']] == ['front', 'side'], 'Expected exact front/side guide pair')
    result = {'schema_version': 1, 'kind': 'analytic_coordinate_annotations', 'source_sha256': guides['source_sha256'],
              'native_inspection_sha256': digest(path), 'reference_images_edited': False,
              'raw_model_renders_preserved': True, 'views': []}
    font = ImageFont.load_default(size=16)
    heading_font = ImageFont.load_default(size=18)
    for view in guides['views']:
        name, projection = view['view'], view['projection']
        require(projection == guide_projection(guides['bounds_blender_z_up'], name), 'Unexpected guide projection')
        raw, target = out/f'guide-{name}-raw.png', out/f'guide-{name}.png'
        require(not raw.is_symlink() and raw.is_file() and raw.stat().st_size <= 8*1024*1024
                and digest(raw) == view['raw_sha256'], 'Raw guide render changed or exceeds bounds')
        require(not target.exists() and not target.is_symlink(), 'Rig-guide output already exists')
        with Image.open(raw) as source:
            require(source.format == 'PNG' and list(source.size) == projection['raw_image_pixels']
                    and getattr(source, 'n_frames', 1) == 1, 'Unexpected raw guide dimensions/format')
            source.load()
            picture = source.convert('RGBA')
        canvas = Image.new('RGBA', tuple(projection['canvas_pixels']), (248, 250, 252, 255))
        canvas.paste(picture, tuple(projection['raw_image_offset_pixels']))
        overlay = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        horizontal = projection['horizontal_axis']
        h_index = projection['horizontal_index']
        lower = [0.5, 0.5, 0.0]
        upper = [0.5, 0.5, 1.0]
        lower[h_index], upper[h_index] = 0.0, 1.0
        left, bottom = normalized_to_pixel(lower, projection, True)
        right, top = normalized_to_pixel(upper, projection, True)
        ticks = []
        for tick in projection['normalized_ticks']:
            p = [.5, .5, tick]
            p[h_index] = tick
            x, y = normalized_to_pixel(p, projection, True)
            color = (20, 65, 95, 175 if tick in (0, 1) else 95)
            draw.line([(x, top), (x, bottom)], fill=color, width=1)
            draw.line([(left, y), (right, y)], fill=color, width=1)
            label = f'{tick:.2f}'
            draw.text((x, bottom+7), label, font=font, fill=(14, 39, 59, 255), anchor='mt', stroke_width=2, stroke_fill='white')
            draw.text((left-8, y), label, font=font, fill=(14, 39, 59, 255), anchor='rm', stroke_width=2, stroke_fill='white')
            ticks.append({'value': tick, 'horizontal_canvas_x': x, 'vertical_canvas_y': y})
        canvas = Image.alpha_composite(canvas, overlay)
        draw = ImageDraw.Draw(canvas)
        draw.text((canvas.width/2, 17), f'{name.upper()}: +{horizontal} right / +Z up', font=heading_font,
                  fill=(14, 39, 59), anchor='mt')
        draw.text((canvas.width/2, 41), '0..1 labels are mesh bounds, not image edges', font=font,
                  fill=(14, 39, 59), anchor='mt')
        draw.text((canvas.width/2, canvas.height-39), f'Hidden axis: {projection["hidden_axis"]} — use the other view',
                  font=font, fill=(14, 39, 59), anchor='mt')
        draw.text((canvas.width/2, canvas.height-19), 'Neutral generated geometry; bone placement remains authored',
                  font=font, fill=(14, 39, 59), anchor='mt')
        canvas.convert('RGB').save(target)
        require(target.stat().st_size <= 8*1024*1024, 'Annotated guide exceeds image byte bound')
        result['views'].append({'view': name, 'path': target.name, 'sha256': digest(target),
                                'raw_path': raw.name, 'raw_sha256': digest(raw), 'ticks': ticks,
                                'canvas_pixels': projection['canvas_pixels']})
    receipt = out/'rig-guide-annotations.json'
    with receipt.open('x') as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
    return result


if __name__ == '__main__':
    result = annotate(sys.argv[1])
    print(json.dumps({'status': 'annotated', 'views': [v['path'] for v in result['views']]}))
