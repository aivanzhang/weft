"""Fail-closed normal-UV admission; this does not restore exported tangent frames."""

import json


POLICY = 'live-normal-uv-export-v1'
FRAME_POLICY = 'mikk-unshared-triangle-v-up-v2'
MAX_OBJECTS = 32
MAX_MATERIALS = 128
MAX_ROWS = 256
MAX_VERTICES = 100_000
MAX_TRIANGLES = 100_000
MAX_PACKED_BYTES = 32 * 1024**2
MAX_TOTAL_PACKED_BYTES = 128 * 1024**2
MAX_REPORT_BYTES = 512 * 1024


class NormalUVExportError(ValueError):
    def __init__(self, report):
        self.report = json.loads(json.dumps(report, allow_nan=False))
        first = self.report['errors'][0]
        super().__init__(f"Normal UV export: {first['code']}: {first['message']}")


class EvidenceError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def require(condition, code, message):
    if not condition:
        raise EvidenceError(code, message)


def valid_name(value):
    return (isinstance(value, str) and 0 < len(value.encode('utf-8')) <= 255
            and value.isprintable())


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def new_report():
    return {'schema_version': 1, 'policy': POLICY, 'status': 'unsupported',
            'materials': [], 'errors': [], 'authored_targets': [],
            'geometry_or_materials_changed': False, 'texture_pixels_read': False,
            'tangent_restoration_performed': False, 'quality_approved': False}


def authored_normal_targets(receipt):
    if receipt is None:
        return {}
    require(isinstance(receipt, dict), 'authored_receipt_invalid', 'Undercoat receipt must be an object')
    rows = receipt.get('materials', [])
    require(isinstance(rows, list) and len(rows) <= 32, 'authored_receipt_invalid',
            'Undercoat material receipt inventory is invalid or oversized')
    targets = {}
    for row in rows:
        require(isinstance(row, dict), 'authored_receipt_invalid', 'Invalid undercoat material receipt')
        uv = row.get('normal_texcoord')
        require(uv is None or _integer(uv, 0, 2), 'authored_receipt_invalid', 'Invalid receipt normal UV index')
        if uv in (None, 0):
            continue
        require(len(rows) <= 4 and type(receipt.get('schema_version')) is int and receipt['schema_version'] == 1
                and receipt.get('profile') == 'short-pile-v1'
                and receipt.get('uv_policy') == 'material-adaptive-v1'
                and receipt.get('frame_policy') == FRAME_POLICY
                and row.get('frame_policy') == FRAME_POLICY,
                'authored_receipt_invalid', 'Nonzero normal UV requires the current authored undercoat policy')
        name, uv_name, digest = row.get('material'), row.get('normal_uv_map'), row.get('texture_sha256')
        require(valid_name(name) and name not in targets and valid_name(uv_name)
                and _integer(row.get('selected_faces'), 1, 2048)
                and isinstance(digest, str) and len(digest) == 64
                and all(c in '0123456789abcdef' for c in digest),
                'authored_receipt_invalid', 'Authored material identity, count or PNG digest is invalid')
        authored_uv = receipt.get('authored_uv')
        require(isinstance(authored_uv, dict) and type(authored_uv.get('index')) is int and authored_uv['index'] == uv
                and authored_uv.get('name') == uv_name,
                'authored_receipt_invalid', 'Authored layer identity differs from the material receipt')
        targets[name] = {key: row[key] for key in ('material', 'normal_texcoord', 'normal_uv_map',
                                                 'selected_faces', 'texture_sha256', 'frame_policy')}
    return targets


def evaluate_normal_uv_export(rows, undercoat_report=None):
    """Evaluate detached live-route evidence; only the native adapter establishes graph facts."""
    report = new_report()
    context = {}
    try:
        require(isinstance(rows, list) and 1 <= len(rows) <= MAX_ROWS,
                'resource_limit', 'Used material rows must be a bounded nonempty list')
        targets = authored_normal_targets(undercoat_report)
        counts = {name: 0 for name in targets}
        seen, objects, materials, triangles = set(), set(), set(), 0
        fields = {'object', 'material', 'slot', 'polygons', 'triangles', 'triangular',
                  'uv_names', 'active_uv_index', 'render_uv_indices', 'normal'}
        for row in rows:
            require(isinstance(row, dict) and set(row) == fields, 'invalid_evidence', 'Invalid live material evidence fields')
            require(valid_name(row['object']) and valid_name(row['material']) and _integer(row['slot'], 0, 127),
                    'invalid_evidence', 'Invalid used object/material identity')
            context = {key: row[key] for key in ('object', 'material', 'slot')}
            key = (row['object'], row['slot'])
            require(key not in seen, 'invalid_evidence', 'Duplicate used material slot')
            seen.add(key); objects.add(row['object']); materials.add(row['material'])
            require(len(objects) <= MAX_OBJECTS and len(materials) <= MAX_MATERIALS,
                    'resource_limit', 'Object or used material count exceeds bounds')
            require(_integer(row['polygons'], 1, MAX_TRIANGLES)
                    and _integer(row['triangles'], row['polygons'], MAX_TRIANGLES)
                    and type(row['triangular']) is bool
                    and (not row['triangular'] or row['triangles'] == row['polygons']),
                    'invalid_evidence', 'Invalid polygon or triangle counts')
            triangles += row['triangles']
            require(triangles <= MAX_TRIANGLES, 'resource_limit', 'Aggregate triangles exceed bounds')
            names = row['uv_names']
            require(isinstance(names, list) and len(names) <= 8 and all(valid_name(v) for v in names)
                    and len(set(names)) == len(names), 'invalid_evidence', 'Invalid UV layer names')
            normal = row['normal']
            witness = {**context, 'polygons': row['polygons'], 'triangles': row['triangles'], 'normal': None}
            report['materials'].append(witness)
            if normal is None:
                witness['admission'] = 'no_live_normal_map'
                continue
            require(isinstance(normal, dict) and set(normal) == {
                'normal_uv_index', 'texture_uv_index', 'image', 'texture_sha256', 'packed_png_bytes'},
                'invalid_evidence', 'Invalid live normal route evidence')
            ni, ti = normal['normal_uv_index'], normal['texture_uv_index']
            require(_integer(ni, 0, len(names)-1) and _integer(ti, 0, len(names)-1)
                    and valid_name(normal['image']), 'invalid_evidence', 'Normal route has no valid UV layer or image')
            digest = normal['texture_sha256']
            require((digest is None or isinstance(digest, str) and len(digest) == 64
                    and all(c in '0123456789abcdef' for c in digest))
                    and _integer(normal['packed_png_bytes'], 0, MAX_PACKED_BYTES),
                    'invalid_evidence', 'Invalid packed normal image evidence')
            witness['normal'] = dict(normal)
            require(ni == ti, 'normal_uv_route_mismatch', 'NormalMap and image vector use different UV layers')
            require(type(row['active_uv_index']) is int and row['active_uv_index'] == 0
                    and row['render_uv_indices'] == [0] and type(row['render_uv_indices'][0]) is int,
                    'uv0_incoherent', 'Export requires UV0 to remain active and render-active')
            if ni == 0:
                require(row['material'] not in targets, 'authored_receipt_mismatch',
                        'Receipted authored material now uses UV0')
                witness['admission'] = 'coherent_uv0'
                continue
            require(ni in (1, 2) and row['material'] in targets,
                    'nonzero_normal_uv_without_receipt', 'Live nonzero normal UV is not an authored undercoat target')
            target = targets[row['material']]
            require(row['triangular'] and ni == target['normal_texcoord']
                    and names[ni] == target['normal_uv_map']
                    and normal['texture_sha256'] == target['texture_sha256']
                    and _integer(normal['packed_png_bytes'], 8, MAX_PACKED_BYTES),
                    'authored_receipt_mismatch', 'Live UV, triangle ownership or packed PNG differs from the authored receipt')
            counts[row['material']] += row['triangles']
            witness['admission'] = 'receipt_bound_authored_nonzero_uv'
        for name, target in targets.items():
            require(counts[name] == target['selected_faces'], 'authored_receipt_mismatch',
                    'Used authored material triangle count differs from its receipt: '+name)
        report.update(status='supported', authored_targets=list(targets.values()),
                      objects=len(objects), used_materials=len(materials), triangles=triangles)
    except EvidenceError as exc:
        report['errors'].append({'code': exc.code, 'message': str(exc), **context})
    # Never return caller-owned receipt or evidence containers.
    encoded = json.dumps(report, allow_nan=False, sort_keys=True)
    if len(encoded.encode('utf-8')) > MAX_REPORT_BYTES:
        report = new_report()
        report['errors'] = [{'code': 'resource_limit', 'message': 'Detached report exceeds its byte limit'}]
        return report
    return json.loads(encoded)
