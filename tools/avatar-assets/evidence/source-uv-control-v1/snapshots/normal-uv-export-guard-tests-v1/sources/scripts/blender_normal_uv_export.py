"""Read only live material normal routes before invoking the native glTF exporter."""

import hashlib
import json
from pathlib import Path
import time

if __package__:
    from .normal_uv_export import (EvidenceError, NormalUVExportError, MAX_OBJECTS, MAX_MATERIALS,
        MAX_ROWS, MAX_VERTICES, MAX_TRIANGLES, MAX_PACKED_BYTES, MAX_TOTAL_PACKED_BYTES,
        MAX_REPORT_BYTES, authored_normal_targets, evaluate_normal_uv_export, new_report, require, valid_name)
else:
    from normal_uv_export import (EvidenceError, NormalUVExportError, MAX_OBJECTS, MAX_MATERIALS,
        MAX_ROWS, MAX_VERTICES, MAX_TRIANGLES, MAX_PACKED_BYTES, MAX_TOTAL_PACKED_BYTES,
        MAX_REPORT_BYTES, authored_normal_targets, evaluate_normal_uv_export, new_report, require, valid_name)


def _pointer(value):
    method = getattr(value, 'as_pointer', None)
    return method() if callable(method) else id(value)


def _socket(node, name):
    socket = node.inputs.get(name)
    require(socket is not None, 'unsupported_graph', 'Live node is missing '+name)
    return socket


def _source(socket):
    require(len(socket.links) <= 1, 'unsupported_graph', 'Live socket has ambiguous links')
    if not socket.links:
        return None
    link = socket.links[0]
    require(getattr(link, 'is_valid', True) and not getattr(link, 'is_muted', False)
            and not getattr(link.from_node, 'mute', False),
            'unsupported_graph', 'Muted or invalid live link is unsupported')
    return link.from_node, link.from_socket.name


def _uv_index(name, names, render):
    if not name:
        require(len(render) == 1, 'uv0_incoherent', 'Default UV route has no unique render-active layer')
        return render[0]
    require(name in names, 'unsupported_graph', 'Live normal route names an absent UV layer')
    return names.index(name)


def _packed_png(image, cache, budget):
    key = _pointer(image)
    if key in cache:
        return cache[key]
    packed = getattr(image, 'packed_file', None)
    require(packed is not None and len(getattr(image, 'packed_files', [])) <= 1,
            'packed_png_required', 'Authored normal image must have one packed PNG')
    size = packed.size
    require(type(size) is int and 8 <= size <= MAX_PACKED_BYTES
            and budget[0]+size <= MAX_TOTAL_PACKED_BYTES,
            'resource_limit', 'Packed normal PNG bytes exceed bounds')
    # Packed bytes avoid pixel decoding and filesystem/path resolution.
    data = packed.data
    require(isinstance(data, bytes) and len(data) == size and data[:8] == b'\x89PNG\r\n\x1a\n',
            'packed_png_required', 'Packed authored normal bytes are not PNG')
    result = hashlib.sha256(data).hexdigest(), size
    budget[0] += size
    cache[key] = result
    return result


def _normal_route(material, names, render, targets, packed_cache, packed_budget):
    if not material.use_nodes:
        return None
    tree = material.node_tree
    require(tree is not None and 1 <= len(tree.nodes) <= 256 and len(tree.links) <= 1024,
            'resource_limit', 'Used material graph is missing or oversized')
    outputs = [n for n in tree.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output]
    require(len(outputs) == 1 and not getattr(outputs[0], 'mute', False),
            'unsupported_graph', 'Used material needs one live output')
    surface = _source(_socket(outputs[0], 'Surface'))
    if surface is None:
        return None
    shader, socket_name = surface
    require(shader.type == 'BSDF_PRINCIPLED' and socket_name == 'BSDF',
            'unsupported_graph', 'Only a directly connected Principled surface can be audited')
    live = _source(_socket(shader, 'Normal'))
    if live is None:
        return None
    normal, socket_name = live
    require(normal.type == 'NORMAL_MAP' and socket_name == 'Normal' and normal.space == 'TANGENT',
            'unsupported_graph', 'Live normal input must be a tangent-space NormalMap')
    ni = _uv_index(normal.uv_map, names, render)
    source = _source(_socket(normal, 'Color'))
    require(source is not None and source[0].type == 'TEX_IMAGE' and source[1] == 'Color',
            'unsupported_graph', 'NormalMap needs a directly connected image Color')
    texture = source[0]
    image = texture.image
    require(image is not None and valid_name(image.name) and getattr(image, 'source', None) in ('FILE', 'GENERATED')
            and texture.projection == 'FLAT', 'unsupported_graph', 'Normal texture needs a flat file-image route')
    vector = _source(_socket(texture, 'Vector'))
    if vector is None:
        ti = _uv_index('', names, render)
    else:
        require(vector[0].type == 'UVMAP' and vector[1] == 'UV'
                and not getattr(vector[0], 'from_instancer', False),
                'unsupported_graph', 'Normal texture vector must be a direct mesh UVMap')
        ti = _uv_index(vector[0].uv_map, names, render)
    needs_hash = ni != 0 and ni == ti and material.name in targets
    digest, size = _packed_png(image, packed_cache, packed_budget) if needs_hash else (None, 0)
    return {'normal_uv_index': ni, 'texture_uv_index': ti, 'image': image.name,
            'texture_sha256': digest, 'packed_png_bytes': size}


def assert_normal_uv_export(objects, undercoat_report=None, *, report_path=None):
    """Raise with a detached report before export; never mutate native data or read pixels."""
    started, rows, context = time.monotonic(), [], {}
    packed_cache, packed_budget = {}, [0]
    try:
        require(isinstance(objects, (list, tuple)) and 1 <= len(objects) <= MAX_OBJECTS,
                'resource_limit', 'Export requires 1..32 explicitly selected mesh objects')
        targets = authored_normal_targets(undercoat_report)
        object_ids, object_names, material_ids = set(), set(), {}
        vertices = triangles = 0
        for obj in objects:
            require(time.monotonic()-started <= 10, 'resource_limit', 'Normal UV inspection exceeded ten seconds')
            require(getattr(obj, 'type', None) == 'MESH' and getattr(obj, 'mode', 'OBJECT') == 'OBJECT'
                    and valid_name(obj.name), 'invalid_evidence', 'Export owner must be a named object-mode mesh')
            context = {'object': obj.name}
            require(_pointer(obj) not in object_ids and obj.name not in object_names,
                    'invalid_evidence', 'Duplicate export object')
            object_ids.add(_pointer(obj)); object_names.add(obj.name)
            mesh = obj.data
            vertices += len(mesh.vertices)
            require(0 < len(mesh.vertices) <= MAX_VERTICES and vertices <= MAX_VERTICES
                    and 0 < len(mesh.polygons) <= MAX_TRIANGLES and len(mesh.loops) <= 3*MAX_TRIANGLES
                    and 1 <= len(obj.material_slots) <= MAX_MATERIALS and len(mesh.uv_layers) <= 8,
                    'resource_limit', 'Mesh or material/UV inventory exceeds bounds')
            used = {}
            for i, polygon in enumerate(mesh.polygons):
                if i % 4096 == 0:
                    require(time.monotonic()-started <= 10, 'resource_limit', 'Normal UV inspection exceeded ten seconds')
                slot, corners = polygon.material_index, polygon.loop_total
                require(type(slot) is int and 0 <= slot < len(obj.material_slots)
                        and type(corners) is int and corners >= 3,
                        'invalid_evidence', 'Polygon has invalid material ownership or corner count')
                count = corners-2
                triangles += count
                require(triangles <= MAX_TRIANGLES, 'resource_limit', 'Aggregate triangles exceed bounds')
                row = used.setdefault(slot, [0, 0, True])
                row[0] += 1; row[1] += count; row[2] &= corners == 3
            names = [layer.name for layer in mesh.uv_layers]
            render = [index for index, layer in enumerate(mesh.uv_layers) if layer.active_render]
            for slot, (polygon_count, triangle_count, triangular) in sorted(used.items()):
                material = obj.material_slots[slot].material
                require(material is not None and valid_name(material.name), 'invalid_evidence', 'Used material is absent or unnamed')
                context = {'object': obj.name, 'material': material.name, 'slot': slot}
                pointer = _pointer(material)
                require(material_ids.setdefault(material.name, pointer) == pointer,
                        'invalid_evidence', 'Distinct used material datablocks have the same name')
                require(len(material_ids) <= MAX_MATERIALS and len(rows) < MAX_ROWS,
                        'resource_limit', 'Used material inventory exceeds bounds')
                normal = _normal_route(material, names, render, targets, packed_cache, packed_budget)
                require(time.monotonic()-started <= 10, 'resource_limit', 'Normal UV inspection exceeded ten seconds')
                rows.append({**context, 'polygons': polygon_count, 'triangles': triangle_count,
                    'triangular': triangular, 'uv_names': names, 'active_uv_index': mesh.uv_layers.active_index,
                    'render_uv_indices': render, 'normal': normal})
        report = evaluate_normal_uv_export(rows, undercoat_report)
        report['vertices'] = vertices
    except EvidenceError as exc:
        report = new_report()
        report['errors'] = [{'code': exc.code, 'message': str(exc), **context}]
        report['checked_material_rows'] = len(rows)
    report.update(elapsed_seconds=round(time.monotonic()-started, 6), packed_png_bytes_hashed=packed_budget[0])
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+'\n'
    require(len(encoded.encode('utf-8')) <= MAX_REPORT_BYTES, 'resource_limit', 'Normal UV report exceeds byte bound')
    report = json.loads(encoded)
    if report_path is not None:
        # Exclusive creation preserves an earlier success or failure receipt.
        with Path(report_path).open('x', encoding='utf-8') as handle:
            handle.write(encoded)
    if report['status'] != 'supported':
        raise NormalUVExportError(report)
    return report
