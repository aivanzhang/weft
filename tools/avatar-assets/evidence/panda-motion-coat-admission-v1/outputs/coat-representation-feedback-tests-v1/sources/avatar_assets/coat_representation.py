"""Small manifest-bound representation facts; never infer visible coverage or approval."""
import json
import math
from pathlib import Path

from avatar_assets.safety import local_file, read_json, require, sha256

MAX_SUMMARY_BYTES = 16*1024


def _number(value, label, *, integer=False, maximum=10**9):
    require((type(value) is int if integer else type(value) in (int, float))
            and math.isfinite(value) and 0 <= value <= maximum, 'Invalid coat '+label)
    return value


def _text(value, label):
    require(isinstance(value, str) and len(value) <= 256, 'Invalid coat '+label)
    return value


def summarize_coat(pack, manifest, source_sha256):
    """Read only allowlisted, hash-bound recipe/report files; legacy omissions stay unknown."""
    pack = Path(pack)
    result = {'schema_version': 1, 'status': 'unavailable', 'source_sha256': source_sha256,
        'scope': 'Facts about this saved recipe only, not a current quality verdict or a prescribed repair.',
        'references': [], 'native_groom': {'status': 'unavailable'},
        'portable_geometry': {'status': 'unavailable'}, 'undercoat': {'status': 'unavailable'},
        'normal_detail': {'status': 'unavailable'}, 'export_capacity': {'status': 'unavailable'},
        'capabilities': [
            'Native fiber upper bounds and portable card counts are not one-to-one density or coverage measures; a card can depict multiple fibers.',
            'Normal detail changes lighting, not strands or silhouette. An absent undercoat does not mean absent normal maps or a flat surface.',
            'Export headroom is a conservative arithmetic allowance, not proof of UV eligibility, material compatibility or visual quality; capacity and UV admission require a probe.']}
    rows = manifest.get('artifacts')
    if rows is None:
        result['reason'] = 'Legacy manifest has no artifact bindings'
        return result
    require(isinstance(rows, list) and len(rows) <= 4096, 'Coat artifact inventory exceeds bounds')

    def bound(name):
        matches = [row for row in rows if row.get('path') == name]
        path = pack/name
        if not matches:
            require(not path.exists() and not path.is_symlink(), 'Coat artifact exists without manifest binding: '+name)
            return None
        require(len(matches) == 1, 'Duplicate coat artifact binding: '+name)
        local_file(path, 1024*1024)
        pin = {'path': name, 'bytes': path.stat().st_size, 'sha256': sha256(path)}
        require(matches == [pin], 'Coat artifact binding changed: '+name)
        data = read_json(path)
        require(isinstance(data, dict), 'Invalid coat artifact object: '+name)
        result['references'].append(pin)
        return data

    config = bound('sources/finish-design.json')
    report = bound('finish-report.json')
    for data in (config, report):
        if data is not None:
            require(data.get('source_sha256') == source_sha256, 'Coat source binding changed')
    if config is None or report is None:
        result['reason'] = 'Preserved recipe or finish report unavailable'
        return result
    if 'source_sha256' in manifest:
        require(manifest['source_sha256'] == source_sha256, 'Coat manifest source changed')

    native = report.get('groom')
    if native is not None:
        require(isinstance(native, dict), 'Invalid native groom report')
        item = {'status': 'available', 'missing_fields': []}
        for target, key in (('native_only', 'native_only'), ('export_supported', 'gltf_export_supported'),
                            ('render_fiber_upper_bound', 'render_fiber_upper_bound')):
            if key not in native:
                item['missing_fields'].append(target)
            elif target == 'render_fiber_upper_bound':
                item[target] = _number(native[key], key, integer=True)
            else:
                require(type(native[key]) is bool, 'Invalid native groom flag')
                item[target] = native[key]
        item['count_scope'] = 'Configured upper bound, not independently counted visible fibers'
        if item['missing_fields']:item['status'] = 'partial'
        result['native_groom'] = item

    portable = report.get('portable_fur')
    if portable is not None:
        require(isinstance(portable, list) and len(portable) <= 4, 'Coat geometry exceeds four bands')
        bands = []
        for index, band in enumerate(portable):
            require(isinstance(band, dict), 'Invalid portable coat band')
            item = {'index': index, 'status': 'unavailable'}
            if 'cards' in band:
                item.update(cards=_number(band['cards'], 'card count', integer=True, maximum=1000000), status='available')
            if 'geometry_profile' in band:item['geometry_profile'] = _text(band['geometry_profile'], 'geometry profile')
            bands.append(item)
        result['portable_geometry'] = {'status': 'available' if all(x['status']=='available' for x in bands) else 'partial',
            'bands': bands, 'count_scope': 'Reported geometry cards, not counted fibers or measured coverage'}
        if all('cards' in x for x in bands):result['portable_geometry']['total_cards'] = sum(x['cards'] for x in bands)

    configured = config.get('fur_undercoat')
    undercoat = {'status': 'available', 'configured': configured is not None,
                'reported': report.get('fur_undercoat') is not None,
                'scope': 'Optional lighting detail; absence does not establish a flat surface or absent source normals'}
    if configured is not None:
        require(isinstance(configured, dict), 'Invalid authored undercoat')
        settings = {}
        for key in ('profile', 'uv_policy', 'resolution', 'seed', 'pitch_m', 'strength', 'roughness', 'min_faces', 'max_faces'):
            if key in configured:
                settings[key] = _text(configured[key], key) if key in ('profile', 'uv_policy') else _number(configured[key], key)
        undercoat['authored_settings'] = settings
    measured = report.get('fur_undercoat')
    if measured is not None:
        require(isinstance(measured, dict), 'Invalid measured undercoat')
        undercoat['reported_measurements'] = {}
        for key in ('selected_triangles', 'selected_polygons', 'excluded_triangles', 'excluded_polygons'):
            if key in measured:undercoat['reported_measurements'][key] = _number(measured[key], key, integer=True)
        for key in ('profile', 'uv_policy'):
            if key in measured:undercoat['reported_measurements'][key] = _text(measured[key], key)
    result['undercoat'] = undercoat

    normal = {'status': 'unavailable', 'original_source_strength': {'status': 'unavailable'},
              'scope': 'Authored settings and measured finished-model routes; original source settings are not reconstructed'}
    for label, data in (('authored_strength', config), ('reported_strength', report)):
        if data.get('normal_strength') is not None:normal[label] = _number(data['normal_strength'], label)
    routes = bound('normal-uv-export.json')
    if routes is not None:
        if report.get('normal_uv_export') is not None:
            require(routes == report['normal_uv_export'], 'Coat normal routes differ from finish report')
    else:routes = report.get('normal_uv_export')
    if routes is not None:
        require(isinstance(routes, dict), 'Invalid coat normal routes')
        materials = routes.get('materials')
        require(isinstance(materials, list) and len(materials) <= 32, 'Coat normal routes exceed32materials')
        normal['routes'] = []
        for row in materials:
            require(isinstance(row, dict), 'Invalid coat material route')
            item = {key: _text(row[key], key) for key in ('object', 'material', 'admission') if key in row}
            route = row.get('normal')
            item['normal_map_reported'] = route is not None
            if route is not None:
                require(isinstance(route, dict), 'Invalid coat normal map route')
                for key in ('normal_uv_index', 'texture_uv_index'):
                    if route.get(key) is not None:item[key] = _number(route[key], key, integer=True, maximum=31)
            normal['routes'].append(item)
        normal['status'] = 'available'
    elif 'authored_strength' in normal or 'reported_strength' in normal:normal['status'] = 'partial'
    result['normal_detail'] = normal

    budget = bound('tuft-export-budget.json')
    embedded = report.get('portable_fur_export_budget')
    if budget is not None and embedded is not None:
        require(budget == embedded, 'Coat budget differs from finish report')
    if budget is None:budget = embedded
    if budget is not None:
        require(isinstance(budget, dict), 'Invalid coat export budget')
        keys = ('base_export_vertices', 'cards', 'exported_corners_per_card', 'limit', 'maximum_cards', 'projected_max_vertices', 'reserve_vertices')
        if all(key in budget for key in keys):
            capacity = {key: _number(budget[key], key, integer=True) for key in keys}
            base, cards, corners, limit, reserve = (capacity[key] for key in ('base_export_vertices', 'cards', 'exported_corners_per_card', 'limit', 'reserve_vertices'))
            require(corners > 0 and base+reserve <= limit and capacity['projected_max_vertices']==base+cards*corners+reserve
                    and capacity['maximum_cards']==(limit-base-reserve)//corners
                    and capacity['projected_max_vertices']<=limit, 'Inconsistent coat export capacity arithmetic')
            if 'total_cards' in result['portable_geometry']:
                require(result['portable_geometry']['total_cards']==cards, 'Coat card count differs from budget')
            capacity.update(status='available', incremental_split_headroom_vertices=limit-capacity['projected_max_vertices'],
                scope='Unused conservative allowance after base, card corners and reserved vertices; not measured export size or admission')
            result['export_capacity'] = capacity
    result['status'] = 'available' if all(result[key]['status']=='available' for key in
        ('native_groom', 'portable_geometry', 'undercoat', 'normal_detail', 'export_capacity')) else 'partial'
    require(len(json.dumps(result, allow_nan=False).encode()) <= MAX_SUMMARY_BYTES, 'Coat summary exceeds16KiB')
    return result
