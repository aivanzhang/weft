"""Opt-in short-pile lighting detail; no source repaint or geometry displacement.

The pure baker uses original UV0, measured fur faces and the shared rest-space
groom field. It returns tangent normals, not recovered hairs or coat coverage.
"""

import hashlib
import math
from numbers import Real
from pathlib import Path
import time

import numpy as np

if __package__:
    from .groom_flow import PreparedFlow, evaluate_prepared_flow
else:
    from groom_flow import PreparedFlow, evaluate_prepared_flow


MAX_VERTICES = 100000
MAX_TRIANGLES = 100000
MAX_RASTER_WORK = 32000000
MAX_SECONDS = 90.0
CHUNK = 16384
PROFILE = 'short-pile-v1'
ADAPTIVE_UV_POLICY = 'material-adaptive-v1'
NORMAL_UV1_NAME = 'AuthoredUndercoatUV1'


def validate_undercoat(settings):
    fields = {'profile', 'resolution', 'seed', 'pitch_m', 'strength', 'roughness', 'min_faces', 'max_faces'}
    if (not isinstance(settings, dict) or set(settings) not in (fields, fields | {'uv_policy'}) or
            settings['profile'] != PROFILE or
            ('uv_policy' in settings and settings['uv_policy'] != ADAPTIVE_UV_POLICY)):
        raise ValueError('Undercoat requires the exact short-pile-v1 settings')
    if type(settings['resolution']) is not int or settings['resolution'] not in (1024, 2048):
        raise ValueError('Undercoat resolution must be1024 or2048')
    if type(settings['seed']) is not int or not 0 <= settings['seed'] <= 2**31-1:
        raise ValueError('Undercoat seed must be a bounded integer')
    for key, low, high in (('pitch_m', .002, .02), ('strength', 0, .5), ('roughness', .5, 1)):
        value = settings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError('Invalid undercoat '+key)
    if any(type(settings[key]) is not int for key in ('min_faces', 'max_faces')) or not (
            1 <= settings['min_faces'] <= settings['max_faces'] <= MAX_TRIANGLES):
        raise ValueError('Invalid undercoat face count bounds')
    return dict(settings)


def _array(value, shape, name, kinds='fiu'):
    # Inspect shape before conversion so nested or oversized input cannot allocate freely.
    if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype.kind not in kinds:
        raise ValueError('Invalid undercoat '+name+' array')
    if value.dtype.kind == 'f' and not np.isfinite(value).all():
        raise ValueError('Nonfinite undercoat '+name)
    return value


def _unit(vectors):
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(lengths <= 1e-8):
        raise ValueError('Undercoat direction or tangent is zero')
    return vectors/lengths


def _hash3(x, y, z, seed):
    value = x.astype(np.uint32)*np.uint32(0x8DA6B343)
    value += y.astype(np.uint32)*np.uint32(0xD8163841)
    value ^= z.astype(np.uint32)*np.uint32(0xCB1AB31F)
    value ^= np.uint32(seed)
    value ^= value >> np.uint32(16)
    value *= np.uint32(0x7FEB352D)
    value ^= value >> np.uint32(15)
    value *= np.uint32(0x846CA68B)
    value ^= value >> np.uint32(16)
    return (value >> np.uint32(8)).astype(np.float32)*(1/16777216)


def _noise_gradient(points, seed):
    """Analytic derivative of smooth bounded3D value noise, independent of UV seams."""
    cell = np.floor(points).astype(np.int32)
    fraction = points-cell
    smooth = fraction*fraction*(3-2*fraction)
    derivative = 6*fraction*(1-fraction)
    gradient = np.zeros_like(points, dtype=np.float32)
    for x in (0, 1):
        for y in (0, 1):
            for z in (0, 1):
                offset = (x, y, z)
                sample = _hash3(cell[:, 0]+x, cell[:, 1]+y, cell[:, 2]+z, seed)
                weights = [smooth[:, axis] if offset[axis] else 1-smooth[:, axis] for axis in range(3)]
                for axis in range(3):
                    others = [index for index in range(3) if index != axis]
                    gradient[:, axis] += sample*weights[others[0]]*weights[others[1]]*derivative[:, axis]*(1 if offset[axis] else -1)
    return gradient


def _raster(vertices, triangles, uvs, selected, size, started, remaining_work=None):
    owner = np.full((size, size), -1, dtype=np.int32)
    bary = np.zeros((size, size, 2), dtype=np.float32)
    work, overlap, subpixel = 0, 0, 0
    for order, index in enumerate(np.flatnonzero(selected)):
        if order % 128 == 0 and time.monotonic()-started > MAX_SECONDS:
            raise RuntimeError('Undercoat bake exceeded its90-second deadline')
        uv = uvs[index].astype(np.float64)
        edge_a, edge_b = uv[1]-uv[0], uv[2]-uv[0]
        determinant = edge_a[0]*edge_b[1]-edge_a[1]*edge_b[0]
        if abs(determinant) < 1e-12:
            raise ValueError('Selected undercoat face has degenerate UVs')
        low = np.maximum(np.ceil(uv.min(axis=0)*size-.5).astype(int), 0)
        high = np.minimum(np.floor(uv.max(axis=0)*size-.5).astype(int), size-1)
        if np.any(high < low):
            subpixel += 1
            continue
        width, height = high-low+1
        work += int(width*height)
        if work > (MAX_RASTER_WORK if remaining_work is None else remaining_work):
            raise ValueError('Undercoat UV raster work exceeds32000000 pixels')
        hit = False
        # Tall/overlapping triangles cannot create an unbounded temporary raster.
        for y0 in range(int(low[1]), int(high[1])+1, 32):
            y1 = min(y0+32, int(high[1])+1)
            x = (np.arange(low[0], high[0]+1)+.5)/size-uv[0, 0]
            y = (np.arange(y0, y1)+.5)/size-uv[0, 1]
            b1 = (x[None, :]*edge_b[1]-y[:, None]*edge_b[0])/determinant
            b2 = (y[:, None]*edge_a[0]-x[None, :]*edge_a[1])/determinant
            inside = (b1 >= -1e-8) & (b2 >= -1e-8) & (b1+b2 <= 1+1e-8)
            if not inside.any():
                continue
            hit = True
            target = owner[y0:y1, low[0]:high[0]+1]
            weights = bary[y0:y1, low[0]:high[0]+1]
            collision = inside & (target >= 0)
            if collision.any():
                old_ids, old_b = target[collision], weights[collision]
                old_p = vertices[triangles[old_ids, 0]]*(1-old_b.sum(axis=1))[:, None]
                old_p += vertices[triangles[old_ids, 1]]*old_b[:, :1]+vertices[triangles[old_ids, 2]]*old_b[:, 1:]
                new_p = vertices[triangles[index, 0]]*(1-b1[collision]-b2[collision])[:, None]
                new_p += vertices[triangles[index, 1]]*b1[collision, None]+vertices[triangles[index, 2]]*b2[collision, None]
                overlap += int(np.count_nonzero(np.max(np.abs(old_p-new_p), axis=1) > 1e-5))
            write = inside & (target < 0)
            target[write] = index
            weights[write, 0], weights[write, 1] = b1[write], b2[write]
        subpixel += not hit
    covered = int(np.count_nonzero(owner >= 0))
    if covered == 0 or overlap > max(8, covered*.01):
        raise ValueError(f'Undercoat UV overlap cannot preserve spatial detail: {overlap}/{covered} texels')
    return owner, bary, {'raster_candidate_pixels': work, 'covered_texels': covered,
                         'overlapping_different_surface_samples': overlap,
                         'overlap_limit_fraction': .01, 'subpixel_faces': int(subpixel)}


def bake_short_pile(vertices, triangles, triangle_uvs, triangle_normals, triangle_tangents,
                    triangle_signs, selected, prepared_flow, settings, *, _budget=None, _directions=None):
    """Return a bounded RGBA normal atlas and analytic evidence; input arrays stay unchanged."""
    settings = validate_undercoat(settings)
    if not isinstance(vertices, np.ndarray) or not 1 <= len(vertices) <= MAX_VERTICES:
        raise ValueError('Invalid or oversized undercoat vertices')
    if not isinstance(triangles, np.ndarray) or not 1 <= len(triangles) <= MAX_TRIANGLES:
        raise ValueError('Invalid or oversized undercoat triangles')
    nv, nt = len(vertices), len(triangles)
    vertices = _array(vertices, (nv, 3), 'vertices')
    triangles = _array(triangles, (nt, 3), 'triangles', 'iu')
    uv = _array(triangle_uvs, (nt, 3, 2), 'UV')
    normals = _array(triangle_normals, (nt, 3, 3), 'normals')
    tangents = _array(triangle_tangents, (nt, 3, 3), 'tangents')
    signs = _array(triangle_signs, (nt, 3), 'bitangent signs')
    selected = _array(selected, (nt,), 'selection', 'b')
    if (triangles.min() < 0 or triangles.max() >= nv or np.any(np.abs(vertices) > 10000) or
            np.any(uv < -1e-6) or np.any(uv > 1+1e-6) or
            np.any(np.abs(signs[selected]) != 1) or not isinstance(prepared_flow, PreparedFlow)):
        raise ValueError('Undercoat needs bounded rest coordinates, UV0 in0–1, valid indices/signs and shared flow')
    count = int(selected.sum())
    if not settings['min_faces'] <= count <= settings['max_faces']:
        raise ValueError('Undercoat selected face count is outside its declared bounds')
    normals, tangents = _unit(normals.astype(np.float32)), _unit(tangents.astype(np.float32))
    started = time.monotonic() if _budget is None else _budget['started']
    if _budget is not None and time.monotonic()-started > MAX_SECONDS:
        raise RuntimeError('Undercoat bake exceeded its90-second deadline')
    if _directions is None:
        directions = np.asarray([evaluate_prepared_flow(point, prepared_flow)[0] for point in vertices], dtype=np.float32)
    else:
        directions = _checked_directions(_directions, nv)
    size = settings['resolution']
    if _budget is None:
        owner, bary, report = _raster(vertices, triangles, uv, selected, size, started)
    else:
        owner, bary, report = _raster(vertices, triangles, uv, selected, size, started,
                                    remaining_work=_budget['remaining_raster'])
        _budget['remaining_raster'] -= report['raster_candidate_pixels']
    output = np.empty((size, size, 4), dtype=np.float32)
    output[:] = (.5, .5, 1, 1)
    locations = np.flatnonzero(owner.ravel() >= 0)
    flat, owners, fractions = output.reshape(-1, 4), owner.ravel(), bary.reshape(-1, 2)
    maximum = 0.0
    for start in range(0, len(locations), CHUNK):
        if time.monotonic()-started > MAX_SECONDS:
            raise RuntimeError('Undercoat bake exceeded its90-second deadline')
        ids = locations[start:start+CHUNK]
        faces, b = owners[ids], fractions[ids]
        weights = np.column_stack((1-b.sum(axis=1), b))
        indices = triangles[faces]
        point = np.sum(vertices[indices]*weights[:, :, None], axis=1)
        normal = _unit(np.sum(normals[faces]*weights[:, :, None], axis=1))
        tangent = np.sum(tangents[faces]*weights[:, :, None], axis=1)
        tangent = _unit(tangent-normal*np.sum(tangent*normal, axis=1)[:, None])
        sign = np.where(np.sum(signs[faces]*weights, axis=1) < 0, -1, 1)
        bitangent = np.cross(normal, tangent)*sign[:, None]
        flow = np.sum(directions[indices]*weights[:, :, None], axis=1)
        flow -= normal*np.sum(flow*normal, axis=1)[:, None]
        magnitude = np.linalg.norm(flow, axis=1)
        flow[magnitude <= 1e-8] = tangent[magnitude <= 1e-8]
        flow = _unit(flow)
        across = np.cross(normal, flow)
        gradient = _noise_gradient(point/settings['pitch_m'], settings['seed'])
        gradient += .32*_noise_gradient(point/(settings['pitch_m']*.47), settings['seed'] ^ 0x193A7)
        # Cross-flow tilt is stronger; no coarse color noise or painted fiber shadows.
        slope = across*np.sum(gradient*across, axis=1)[:, None]
        slope += .22*flow*np.sum(gradient*flow, axis=1)[:, None]
        slope *= settings['strength']
        lengths = np.linalg.norm(slope, axis=1)
        slope *= np.minimum(1, .45/np.maximum(lengths, 1e-8))[:, None]
        encoded = np.column_stack((-np.sum(slope*tangent, axis=1), -np.sum(slope*bitangent, axis=1), np.ones(len(ids))))
        encoded = _unit(encoded)
        maximum = max(maximum, float(np.degrees(np.arccos(np.clip(encoded[:, 2], -1, 1))).max(initial=0)))
        flat[ids, :3] = encoded*.5+.5
    report.update(profile=PROFILE, settings=settings, selected_triangles=count, excluded_triangles=nt-count,
                  total_triangles=nt, maximum_normal_tilt_degrees=maximum,
                  elapsed_seconds=round(time.monotonic()-started, 6),
                  normal_float32_sha256=hashlib.sha256(output.tobytes()).hexdigest(),
                  selected_faces_sha256=hashlib.sha256(np.flatnonzero(selected).astype('<i4').tobytes()).hexdigest(),
                  array_memory_bound_bytes=size*size*36+24*1024*1024,
                  palette_changed=False, original_uvs_changed=False, geometry_displaced=False,
                  assumption='Authored short-pile lighting detail, not recovered fibers or measured coat density')
    return output, report


def apply_undercoat(obj, mask, prepared_flow, settings, out, *, export_vertex_budget=None):
    """Clone selected face materials and bake one supported PNG normal texture."""
    import bpy
    settings = validate_undercoat(settings)
    if settings.get('uv_policy') == ADAPTIVE_UV_POLICY:
        return _apply_material_undercoat(obj, mask, prepared_flow, settings, out, export_vertex_budget)
    mesh = obj.data
    if len(mesh.vertices) > MAX_VERTICES or len(mesh.loops) > 300000 or len(mesh.materials) > 32:
        raise ValueError('Undercoat mesh or material count exceeds bounds')
    mask = _array(mask, (len(mesh.vertices),), 'vertex mask', 'b')
    if mesh.uv_layers.active is None:
        raise ValueError('Undercoat requires original UV0')
    mesh.calc_loop_triangles()
    if not 1 <= len(mesh.loop_triangles) <= MAX_TRIANGLES:
        raise ValueError('Undercoat triangle count exceeds bounds')
    mesh.calc_tangents(uvmap=mesh.uv_layers.active.name)
    vertices = np.asarray([tuple(v.co) for v in mesh.vertices], dtype=np.float32)
    triangles = np.asarray([tuple(t.vertices) for t in mesh.loop_triangles], dtype=np.int32)
    loops = np.asarray([tuple(t.loops) for t in mesh.loop_triangles], dtype=np.int32)
    uv = np.asarray([tuple(v.uv) for v in mesh.uv_layers.active.data], dtype=np.float32)[loops]
    normals = np.asarray([tuple(v.vector) for v in mesh.corner_normals], dtype=np.float32)[loops]
    tangents = np.asarray([tuple(v.tangent) for v in mesh.loops], dtype=np.float32)[loops]
    signs = np.asarray([v.bitangent_sign for v in mesh.loops], dtype=np.float32)[loops]
    selected = mask[triangles].all(axis=1)
    polygons = sorted({mesh.loop_triangles[i].polygon_index for i in np.flatnonzero(selected)})
    slots = sorted({mesh.polygons[i].material_index for i in polygons})
    # Validate every material before allocating new native objects or changing face ownership.
    for slot in slots:
        material = mesh.materials[slot]
        if not material or not material.use_nodes or not 1 <= len(material.node_tree.nodes) <= 256:
            raise ValueError('Undercoat needs bounded node-based materials')
        shaders = [node for node in material.node_tree.nodes if node.type == 'BSDF_PRINCIPLED']
        if len(shaders) != 1 or any(name not in shaders[0].inputs for name in ('Base Color', 'Normal', 'Roughness', 'Metallic')):
            raise ValueError('Undercoat requires one supported Principled shader')
    pixels, report = bake_short_pile(vertices, triangles, uv, normals, tangents, signs, selected, prepared_flow, settings)
    texture_dir = Path(out)/'textures'
    texture_dir.mkdir(exist_ok=True)
    path = texture_dir/'authored-short-pile-normal.png'
    if path.exists():
        raise ValueError('Undercoat texture output already exists')
    image = bpy.data.images.new('AuthoredShortPileNormal', width=settings['resolution'], height=settings['resolution'], alpha=False)
    image.colorspace_settings.name = 'Non-Color'
    image.pixels.foreach_set(pixels.ravel())
    image.filepath_raw, image.file_format = str(path), 'PNG'
    image.save()
    image.pack()
    del pixels
    clones, rows = {}, []
    for slot in slots:
        original = mesh.materials[slot]
        clone = original.copy()
        clone.name = original.name+'__short_pile_v1'
        shader = next(node for node in clone.node_tree.nodes if node.type == 'BSDF_PRINCIPLED')
        for name in ('Normal', 'Roughness', 'Metallic'):
            for link in list(shader.inputs[name].links):
                clone.node_tree.links.remove(link)
        shader.inputs['Roughness'].default_value = settings['roughness']
        shader.inputs['Metallic'].default_value = 0
        texture = clone.node_tree.nodes.new('ShaderNodeTexImage')
        texture.image, texture.extension, texture.interpolation = image, 'REPEAT', 'Linear'
        normal = clone.node_tree.nodes.new('ShaderNodeNormalMap')
        normal.space, normal.uv_map = 'TANGENT', mesh.uv_layers.active.name
        normal.inputs['Strength'].default_value = 1
        clone.node_tree.links.new(texture.outputs['Color'], normal.inputs['Color'])
        clone.node_tree.links.new(normal.outputs['Normal'], shader.inputs['Normal'])
        clones[slot] = len(mesh.materials)
        mesh.materials.append(clone)
        rows.append({'source_material': original.name, 'material': clone.name, 'basecolor_graph_preserved': True})
    for index in polygons:
        face = mesh.polygons[index]
        face.material_index = clones[face.material_index]
    report.update(texture_path=str(path.relative_to(out)), texture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  materials=rows, selected_polygons=len(polygons), excluded_polygons=len(mesh.polygons)-len(polygons),
                  original_normal_replaced_only_on_selected_faces=True, original_materials_preserved=True,
                  source_images_unchanged=True)
    return report


def _export_budget(value, native_vertices):
    fields = {'source_export_vertices', 'portable_export_vertices', 'reserve_vertices'}
    if not isinstance(value, dict) or set(value) != fields or any(type(v) is not int for v in value.values()):
        raise ValueError('Adaptive undercoat requires the exact aggregate export_vertex_budget')
    source, portable, reserve = (value[key] for key in
                                  ('source_export_vertices', 'portable_export_vertices', 'reserve_vertices'))
    if (not native_vertices <= source <= MAX_VERTICES or not 0 <= portable <= MAX_VERTICES or
            not 0 <= reserve <= 10000 or source+portable+reserve > MAX_VERTICES):
        raise ValueError('Adaptive undercoat exceeds the100000vertex aggregate export budget')
    return dict(value)


def _checked_directions(directions, count):
    directions = _array(directions, (count, 3), 'shared flow directions', 'f')
    if (directions.dtype != np.dtype(np.float32) or directions.flags.writeable or
            np.any(np.abs(directions) > 1.00001) or
            np.any(np.abs(np.linalg.norm(directions, axis=1)-1) > 1e-5)):
        raise ValueError('Shared undercoat directions must be read-only unit float32 vectors')
    return directions


def _prepare_directions(vertices, prepared_flow, started):
    # Material bakes share one call-local array; never cache source geometry or flow globally.
    directions = np.empty((len(vertices), 3), dtype=np.float32)
    for index, point in enumerate(vertices):
        if index % 256 == 0 and time.monotonic()-started > MAX_SECONDS:
            raise RuntimeError('Undercoat flow preparation exceeded its90-second deadline')
        directions[index] = evaluate_prepared_flow(point, prepared_flow)[0]
    if time.monotonic()-started > MAX_SECONDS:
        raise RuntimeError('Undercoat flow preparation exceeded its90-second deadline')
    directions.flags.writeable = False
    return _checked_directions(directions, len(vertices))


def plan_material_undercoat(vertices, triangles, uv0, selected, material_ids, settings, *, export_vertex_budget):
    """Plan normal-only UV changes using supplied whole-pack counts, never native count guesses."""
    if __package__:
        from .undercoat_uv_atlas import plan_normal_atlases
    else:
        from undercoat_uv_atlas import plan_normal_atlases
    settings = validate_undercoat(settings)
    if settings.get('uv_policy') != ADAPTIVE_UV_POLICY:
        raise ValueError('Material planning requires material-adaptive-v1')
    if not isinstance(vertices, np.ndarray) or vertices.ndim != 2 or not 1 <= len(vertices) <= MAX_VERTICES:
        raise ValueError('Invalid or oversized adaptive undercoat vertices')
    budget = _export_budget(export_vertex_budget, len(vertices))
    if not isinstance(triangles, np.ndarray) or triangles.ndim != 2 or not 1 <= len(triangles) <= MAX_TRIANGLES:
        raise ValueError('Invalid or oversized adaptive undercoat triangles')
    selected = _array(selected, (len(triangles),), 'selection', 'b')
    if not settings['min_faces'] <= int(selected.sum()) <= settings['max_faces']:
        raise ValueError('Undercoat selected face count is outside its declared bounds')
    plan, payloads = plan_normal_atlases(vertices, triangles, uv0, selected, material_ids,
        source_export_vertices=budget['source_export_vertices']+budget['portable_export_vertices'],
        resolution=settings['resolution'], atlas_resolution=1024, pitch_m=settings['pitch_m'],
        reserve_vertices=budget['reserve_vertices'])
    extra = 0
    for row in plan['regions']:
        region = material_ids == row['region_id']
        yes = np.bincount(triangles[selected & region].ravel(), minlength=len(vertices))
        no = np.bincount(triangles[~selected & region].ravel(), minlength=len(vertices))
        # One native vertex can already have several exported UV/normal seam copies.
        corner_bound = int(np.minimum(yes, no).sum())
        delta = corner_bound-row['material_split_extra_vertices']
        extra += delta
        row['material_split_corner_bound'] = corner_bound
        row['additional_material_seam_reserve_vertices'] = delta
    plan['additional_export_vertices'] += extra
    plan['projected_export_vertex_ceiling'] += extra
    if plan['projected_export_vertex_ceiling'] > MAX_VERTICES:
        raise ValueError('Adaptive material seams exceed the100000vertex aggregate export budget')
    plan.update(export_vertex_budget=budget, additional_material_seam_reserve_vertices=extra,
        source_export_vertices_scope='Pinned source GLB vertices plus planned whole-coat exported corners; not a measured source-only count',
        uv_policy=ADAPTIVE_UV_POLICY, selected_triangles=int(selected.sum()),
        excluded_triangles=int((~selected).sum()), max_simultaneous_float_atlases=1,
        aggregate_raster_limit=MAX_RASTER_WORK, aggregate_runtime_seconds=MAX_SECONDS,
        quality_approved=False, automatic_quality_policy_promoted=False)
    return plan, payloads


def _selected_triangular_polygons(loops, polygon_ids, polygon_loop_counts, selected, loop_count):
    nt = len(selected)
    selected = _array(selected, (nt,), 'selection', 'b')
    loops = _array(loops, (nt, 3), 'triangle loops', 'iu')
    polygon_ids = _array(polygon_ids, (nt,), 'polygon IDs', 'iu')
    if (not isinstance(polygon_loop_counts, np.ndarray) or polygon_loop_counts.ndim != 1 or
            not 1 <= len(polygon_loop_counts) <= MAX_TRIANGLES):
        raise ValueError('Invalid undercoat polygon loop counts')
    counts = _array(polygon_loop_counts, (len(polygon_loop_counts),), 'polygon loop counts', 'iu')
    if (type(loop_count) is not int or not 1 <= loop_count <= 300000 or
            loops.min() < 0 or loops.max() >= loop_count or polygon_ids.min() < 0 or
            polygon_ids.max() >= len(counts)):
        raise ValueError('Undercoat polygon or loop ownership is invalid')
    polygons, occurrences = np.unique(polygon_ids[selected], return_counts=True)
    owned = loops[selected].ravel()
    references = np.bincount(loops.ravel(), minlength=loop_count)
    if np.any(counts[polygons] != 3) or np.any(occurrences != 1) or np.any(references[owned] != 1):
        raise ValueError('Adaptive undercoat requires triangular selected polygons with unambiguous loop ownership')
    return polygons


def _material_shader(material):
    if not material or not material.use_nodes or not 1 <= len(material.node_tree.nodes) <= 252:
        raise ValueError('Adaptive undercoat needs bounded node-based materials')
    shaders = [node for node in material.node_tree.nodes if node.type == 'BSDF_PRINCIPLED']
    if len(shaders) != 1 or any(name not in shaders[0].inputs for name in ('Base Color', 'Normal', 'Roughness', 'Metallic')):
        raise ValueError('Adaptive undercoat requires one supported Principled shader')
    return shaders[0]


def _clone_material(original, image, uv_name, roughness):
    clone = original.copy()
    if clone is original or clone.node_tree is original.node_tree or clone.node_tree.nodes is original.node_tree.nodes:
        raise ValueError('Undercoat material copy did not isolate the source graph')
    clone.name = original.name+'__short_pile_adaptive_v1'
    shader = _material_shader(clone)
    for name in ('Normal', 'Roughness', 'Metallic'):
        for link in list(shader.inputs[name].links):
            clone.node_tree.links.remove(link)
    shader.inputs['Roughness'].default_value = roughness
    shader.inputs['Metallic'].default_value = 0
    uv = clone.node_tree.nodes.new('ShaderNodeUVMap')
    uv.uv_map = uv_name
    texture = clone.node_tree.nodes.new('ShaderNodeTexImage')
    texture.image, texture.extension, texture.interpolation = image, 'REPEAT', 'Linear'
    normal = clone.node_tree.nodes.new('ShaderNodeNormalMap')
    normal.space, normal.uv_map = 'TANGENT', uv_name
    normal.inputs['Strength'].default_value = 1
    clone.node_tree.links.new(uv.outputs['UV'], texture.inputs['Vector'])
    clone.node_tree.links.new(texture.outputs['Color'], normal.inputs['Color'])
    clone.node_tree.links.new(normal.outputs['Normal'], shader.inputs['Normal'])
    return clone


def _save_material_normal(bpy, pixels, path, size, label):
    temporary = bpy.data.images.new(label, width=size, height=size, alpha=False)
    try:
        temporary.colorspace_settings.name = 'Non-Color'
        temporary.pixels.foreach_set(pixels.ravel())
        temporary.filepath_raw, temporary.file_format = str(path), 'PNG'
        temporary.save()
    finally:
        bpy.data.images.remove(temporary)
    if not 0 < path.stat().st_size <= 32*1024*1024:
        raise ValueError('Undercoat normal PNG exceeds32MiB')
    # Retain packed PNG bytes, not four generated float buffers between regional bakes.
    image = bpy.data.images.load(str(path), check_existing=False)
    image.name = label
    image.colorspace_settings.name = 'Non-Color'
    image.pack()
    image.buffers_free()
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return image, digest.hexdigest()


def _apply_material_undercoat(obj, mask, prepared_flow, settings, out, export_vertex_budget):
    import bpy
    if __package__:
        from .undercoat_uv_atlas import atlas_tangents
    else:
        from undercoat_uv_atlas import atlas_tangents
    started = time.monotonic()
    mesh = obj.data
    if (not 1 <= len(mesh.vertices) <= MAX_VERTICES or not 1 <= len(mesh.loops) <= 300000 or
            not 1 <= len(mesh.materials) <= 32 or not isinstance(prepared_flow, PreparedFlow)):
        raise ValueError('Adaptive undercoat needs bounded mesh/materials and shared prepared flow')
    _export_budget(export_vertex_budget, len(mesh.vertices))
    mask = _array(mask, (len(mesh.vertices),), 'vertex mask', 'b')
    if len(mesh.uv_layers) != 1 or mesh.uv_layers.active_index != 0:
        raise ValueError('Adaptive undercoat requires only original UV0; preexisting UV1 is unsupported')
    uv_layer = mesh.uv_layers[0]
    if uv_layer.name == NORMAL_UV1_NAME or mesh.attributes.get(NORMAL_UV1_NAME) is not None:
        raise ValueError('Authored undercoat UV1 name conflicts with existing data')
    original_active, original_render = mesh.uv_layers.active_index, uv_layer.active_render
    mesh.calc_loop_triangles()
    if not 1 <= len(mesh.loop_triangles) <= MAX_TRIANGLES:
        raise ValueError('Undercoat triangle count exceeds bounds')
    vertices = np.asarray([tuple(v.co) for v in mesh.vertices], dtype=np.float32)
    triangles = np.asarray([tuple(t.vertices) for t in mesh.loop_triangles], dtype=np.int32)
    loops = np.asarray([tuple(t.loops) for t in mesh.loop_triangles], dtype=np.int32)
    polygon_ids = np.asarray([t.polygon_index for t in mesh.loop_triangles], dtype=np.int32)
    material_ids = np.asarray([mesh.polygons[i].material_index for i in polygon_ids], dtype=np.int32)
    selected = mask[triangles].all(axis=1)
    polygons = _selected_triangular_polygons(loops, polygon_ids,
        np.asarray([p.loop_total for p in mesh.polygons], dtype=np.int32), selected, len(mesh.loops))
    slots = np.unique(material_ids[selected])
    if np.any(slots < 0) or np.any(slots >= len(mesh.materials)) or len(mesh.materials)+len(slots) > 32:
        raise ValueError('Adaptive undercoat material ownership or clone count exceeds bounds')
    for slot in slots:
        _material_shader(mesh.materials[int(slot)])
    uv0_loops = np.asarray([tuple(value.uv) for value in uv_layer.data], dtype=np.float32)
    uv0 = uv0_loops[loops]
    normals = np.asarray([tuple(value.vector) for value in mesh.corner_normals], dtype=np.float32)[loops]
    normals = _array(normals, (len(triangles), 3, 3), 'normals')
    plan, payloads = plan_material_undercoat(vertices, triangles, uv0, selected, material_ids, settings,
                                           export_vertex_budget=export_vertex_budget)
    paths = {int(slot): Path(out)/'textures'/f'authored-short-pile-normal-material-{int(slot):02d}.png' for slot in slots}
    if any(path.exists() for path in paths.values()):
        raise ValueError('Adaptive undercoat texture output already exists')
    # Validate all requested tangent frames before adding a UV layer or material.
    mesh.calc_tangents(uvmap=uv_layer.name)
    tangents = np.asarray([tuple(value.tangent) for value in mesh.loops], dtype=np.float32)[loops]
    signs = np.asarray([value.bitangent_sign for value in mesh.loops], dtype=np.float32)[loops]
    tangents = _array(tangents, (len(triangles), 3, 3), 'tangents')
    signs = _array(signs, (len(triangles), 3), 'bitangent signs')
    uv1_loops, authored = uv0_loops.copy(), {}
    for slot, payload in payloads.items():
        ids = payload['face_ids']
        frames = atlas_tangents(vertices, triangles, ids, payload['uvs'], normals[ids])
        authored[slot] = frames
        uv1_loops[loops[ids].ravel()] = payload['uvs'].reshape(-1, 2)
    for slot in slots:
        ids = np.flatnonzero(selected & (material_ids == slot))
        unit_normal = _unit(normals[ids])
        if int(slot) not in authored:
            _unit(tangents[ids]-unit_normal*np.sum(tangents[ids]*unit_normal, axis=2)[:, :, None])
            if np.any(np.abs(signs[ids]) != 1):
                raise ValueError('Selected original UV0 has invalid tangent handedness')
    if time.monotonic()-started > MAX_SECONDS:
        raise RuntimeError('Undercoat bake exceeded its90-second deadline')
    flow_started = time.monotonic()
    directions = _prepare_directions(vertices, prepared_flow, started)
    flow_elapsed = time.monotonic()-flow_started
    if payloads:
        new_uv = mesh.uv_layers.new(name=NORMAL_UV1_NAME)
        if new_uv.name != NORMAL_UV1_NAME:
            raise ValueError('Native UV1 layer name changed unexpectedly')
        new_uv.data.foreach_set('uv', uv1_loops.ravel())
        new_uv.active_render = False
        mesh.uv_layers.active_index = original_active
        uv_layer.active_render = original_render
    Path(out, 'textures').mkdir(exist_ok=True)
    work = {'started': started, 'remaining_raster': MAX_RASTER_WORK-plan['raster_candidates']}
    rows, clones = [], {}
    for region in plan['regions']:
        slot = region['region_id']
        ids = np.flatnonzero(selected & (material_ids == slot))
        authored_uv = slot in payloads
        target_uv = NORMAL_UV1_NAME if authored_uv else uv_layer.name
        frames = authored.get(slot)
        bake_settings = {key: value for key, value in settings.items() if key != 'uv_policy'}
        bake_settings.update(resolution=1024 if authored_uv else settings['resolution'], min_faces=1, max_faces=len(ids))
        pixels, baked = bake_short_pile(vertices, triangles[ids],
            payloads[slot]['uvs'] if authored_uv else uv0[ids], normals[ids],
            frames[:, :, :3] if authored_uv else tangents[ids],
            frames[:, :, 3] if authored_uv else signs[ids], np.ones(len(ids), dtype=bool),
            prepared_flow, bake_settings, _budget=work, _directions=directions)
        image, image_hash = _save_material_normal(bpy, pixels, paths[slot], bake_settings['resolution'],
                                                  f'AuthoredShortPileNormal_{slot:02d}')
        del pixels
        if time.monotonic()-started > MAX_SECONDS:
            raise RuntimeError('Undercoat bake exceeded its90-second deadline')
        original = mesh.materials[slot]
        clone = _clone_material(original, image, target_uv, settings['roughness'])
        clones[slot] = len(mesh.materials)
        mesh.materials.append(clone)
        rows.append({'source_material': original.name, 'material': clone.name,
            'source_material_index': slot, 'selected_faces': len(ids), 'normal_uv_map': target_uv,
            'normal_texcoord': 1 if authored_uv else 0, 'basecolor_graph_preserved': True,
            'texture_path': str(paths[slot].relative_to(out)), 'texture_sha256': image_hash,
            'bake': baked, 'atlas': payloads[slot]['report'] if authored_uv else None,
            'authored_tangents_sha256': hashlib.sha256(frames.tobytes()).hexdigest() if authored_uv else None})
    for index in polygons:
        polygon = mesh.polygons[int(index)]
        polygon.material_index = clones[polygon.material_index]
    mesh.uv_layers.active_index = original_active
    uv_layer.active_render = original_render
    if payloads:
        mesh.uv_layers[1].active_render = False
    actual_uv0 = np.asarray([tuple(value.uv) for value in uv_layer.data], dtype=np.float32)
    if not np.array_equal(actual_uv0, uv0_loops):
        raise ValueError('Original undercoat UV0 changed unexpectedly')
    return {'schema_version': 1, 'profile': PROFILE, 'settings': settings, 'uv_policy': ADAPTIVE_UV_POLICY,
        'plan': plan, 'materials': rows, 'selected_triangles': int(selected.sum()),
        'excluded_triangles': int((~selected).sum()), 'selected_polygons': len(polygons),
        'excluded_polygons': len(mesh.polygons)-len(polygons),
        'selected_faces_sha256': hashlib.sha256(np.flatnonzero(selected).astype('<i4').tobytes()).hexdigest(),
        'original_uv0_sha256': hashlib.sha256(uv0_loops.tobytes()).hexdigest(),
        'authored_uv1_sha256': hashlib.sha256(uv1_loops.tobytes()).hexdigest() if payloads else None,
        'original_active_uv_index': original_active, 'original_uv0_active_render': original_render,
        'original_uvs_changed': False, 'original_materials_preserved': True, 'source_images_unchanged': True,
        'original_normal_replaced_only_on_selected_faces': True, 'palette_changed': False,
        'geometry_displaced': False, 'skin_or_animation_changed': False,
        'raster_candidate_pixels_including_plan': MAX_RASTER_WORK-work['remaining_raster'],
        'shared_flow': {'evaluated_vertices': len(vertices), 'material_bakes': len(rows),
                        'array_bytes': directions.nbytes, 'read_only': True,
                        'directions_float32_sha256': hashlib.sha256(directions.tobytes()).hexdigest(),
                        'elapsed_seconds': round(flow_elapsed, 6)},
        'elapsed_seconds': round(time.monotonic()-started, 6), 'quality_approved': False,
        'weft_appearance_compatible': False, 'max_simultaneous_float_atlases': 1,
        'assumption': 'Authored optional short-pile surface detail, not dense fur, recovered fibers or measured coat coverage'}
