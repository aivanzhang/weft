"""Bounded, append-only root archives; standalone imports also work in Blender."""

import ast
import hashlib
import io
import math
import os
from pathlib import Path
import re
import stat
import struct
import unicodedata
import zipfile


FOLDER = 'portable-root-bindings'
MAX_ROOTS = 5000
MAX_ARCHIVE_BYTES = 512*1024
MAX_FOLDER_BYTES = 1024*1024
MAX_FILES = 128
SPACE = 'blender_local_z_up_meters'
INSET_CONVENTION = 'surface_root_minus_bottom_edge_midpoint'
RECEIPT_FIELDS = {'schema_version', 'path', 'sha256', 'bytes', 'roots', 'space', 'root_inset_convention'}
EXPORT_POLICY = 'portable-fur-source-ids-v1'
EXPORT_ATTRIBUTE = '_AVATAR_FUR_VERTEX_ID'
MAX_NPY_HEADER_BYTES = 1024
MAX_DECODED_BYTES = MAX_ROOTS*64


def _require(value, message):
    if not value:
        raise ValueError(message)


def _count(value):
    _require(type(value) is int and 1 <= value <= MAX_ROOTS, 'Root count must be an integer in 1..5000')
    return value


def _name(value):
    _require(type(value) is str and len(value) <= 96 and '..' not in value
             and re.fullmatch(r'PortableCoat_[A-Za-z0-9_][A-Za-z0-9_.-]*', value),
             'Require a bounded generated PortableCoat_ name without traversal')
    return value


def _root(value):
    _require(isinstance(value, (str, Path)), 'Require a local output root')
    path = Path(value)
    _require(path.is_dir() and not path.is_symlink(), 'Root must be an existing regular directory')
    return path.resolve(strict=True)


def _inventory(root):
    folder = root/FOLDER
    _require(not folder.is_symlink(), 'Binding folder cannot be a symlink')
    if not folder.exists():
        return folder, {}
    _require(folder.is_dir(), 'Binding folder must be a directory')
    files, total = {}, 0
    for entry in folder.iterdir():
        _require(len(files) < MAX_FILES, 'Binding folder exceeds 128 files')
        _require(entry.suffix == '.npz', 'Unexpected file in binding folder')
        _name(entry.stem)
        info = entry.lstat()
        _require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= MAX_ARCHIVE_BYTES,
                 'Binding archive must be a regular file up to 512 KiB')
        total += info.st_size
        _require(total <= MAX_FOLDER_BYTES, 'Binding folder exceeds 1 MiB')
        files[entry.name] = info
    return folder, files


def _folder_fd(folder):
    return os.open(folder, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0))


def _read(folder, name, expected):
    directory = _folder_fd(folder)
    try:
        fd = os.open(name, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0), dir_fd=directory)
        with os.fdopen(fd, 'rb') as stream:
            before = os.fstat(stream.fileno())
            _require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= MAX_ARCHIVE_BYTES
                     and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                     (expected.st_dev, expected.st_ino, expected.st_size, expected.st_mtime_ns),
                     'Binding archive changed before read')
            data = stream.read(MAX_ARCHIVE_BYTES+1)
            after = os.fstat(stream.fileno())
            _require(len(data) == before.st_size and (before.st_size, before.st_mtime_ns) ==
                     (after.st_size, after.st_mtime_ns), 'Binding archive changed during read')
            return data
    finally:
        os.close(directory)


def _write(folder, name, data):
    directory = _folder_fd(folder)
    try:
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0),
                     0o600, dir_fd=directory)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
    finally:
        os.close(directory)
    info = (folder/name).lstat()
    _require(hashlib.sha256(_read(folder, name, info)).hexdigest() == hashlib.sha256(data).hexdigest(),
             'Copied binding archive hash differs')


def _array(value, rows, width, integer, label, np):
    shape = (rows,) if width == 1 else (rows, width)
    if type(value) is np.ndarray:
        _require(value.shape == shape and value.dtype.kind in ('iu' if integer else 'iuf'),
                 'Invalid '+label+' shape or numeric dtype')
    else:
        _require(type(value) is list and len(value) == rows, 'Invalid '+label+' list length')
        if width == 1:
            entries = value
        else:
            _require(all(type(row) is list and len(row) == width for row in value),
                     'Invalid '+label+' row shape')
            entries = (item for row in value for item in row)
        for item in entries:
            is_integer = type(item) is int or isinstance(item, np.integer) and not isinstance(item, np.bool_)
            is_number = is_integer or type(item) is float or isinstance(item, np.floating)
            _require(is_integer if integer else is_number, 'Invalid '+label+' scalar type')
            try:
                finite = math.isfinite(item)
            except OverflowError:
                finite = False
            _require(finite, 'Nonfinite '+label)
            if integer:
                _require(0 <= item <= 99999, 'Triangle index outside 0..99999')
    # Bound both Python containers and ndarray metadata before any numeric copy.
    if integer:
        _require(np.all(value >= 0) and np.all(value <= 99999) if type(value) is np.ndarray else True,
                 'Triangle index outside 0..99999')
    result = np.asarray(value, dtype='<i4' if integer else '<f8')
    _require(np.isfinite(result).all(), 'Nonfinite '+label)
    return result


def _node_name(value):
    _require(type(value) is str and value and not any(unicodedata.category(c) == 'Cc' for c in value),
             'Require a nonempty node name without control characters')
    try:
        size = len(value.encode('utf-8'))
    except UnicodeEncodeError:
        raise ValueError('Node name must be valid UTF-8') from None
    _require(size <= 255, 'Node name exceeds255UTF-8bytes')
    return value


def _export_correspondence(value, roots, name):
    fields = {'policy', 'attribute', 'source_node', 'coat_node', 'source_vertex_count',
              'source_triangle_count', 'coat_vertex_count'}
    _require(type(value) is dict and set(value) == fields, 'Invalid export correspondence fields')
    _require(value['policy'] == EXPORT_POLICY and value['attribute'] == EXPORT_ATTRIBUTE,
             'Unsupported export correspondence policy or attribute')
    _node_name(value['source_node']); _node_name(value['coat_node'])
    _require(value['coat_node'] == name, 'Coat node must equal the binding archive name')
    for key in ('source_vertex_count', 'source_triangle_count'):
        _require(type(value[key]) is int and 1 <= value[key] <= 100000, 'Invalid bounded '+key)
    _require(type(value['coat_vertex_count']) is int and value['coat_vertex_count'] == 6*roots,
             'Coat vertex count must equal six times the root count')
    return dict(value)


def _source_ids(value, triangle, roots, correspondence, np):
    ids = _array(value, roots, 3, True, 'source_vertex_ids', np)
    _require(np.all(ids < correspondence['source_vertex_count']), 'Source vertex ID exceeds declared source count')
    _require(np.all(ids[:, 0] != ids[:, 1]) and np.all(ids[:, 0] != ids[:, 2])
             and np.all(ids[:, 1] != ids[:, 2]), 'Source triangle repeats a corner vertex ID')
    _require(np.all(triangle < correspondence['source_triangle_count']),
             'Per-band triangle ordinal exceeds declared source triangle count')
    return ids


def write_bindings(output_root: Path, name: str, bindings: dict, expected_roots: int, *, correspondence=None) -> dict:
    """Store numeric roots only; caller retains responsibility for source-face identity."""
    roots, name = _count(expected_roots), _name(name)
    _require(type(bindings) is dict and set(bindings) == {'triangle_index', 'barycentric', 'root_inset'},
             'Binding arrays must have exactly triangle_index, barycentric and root_inset')
    exported = None
    if correspondence is not None:
        fields = {'source_node', 'coat_node', 'source_vertex_count', 'source_triangle_count', 'source_vertex_ids'}
        _require(type(correspondence) is dict and set(correspondence) == fields, 'Invalid source correspondence fields')
        exported = _export_correspondence({key: value for key, value in correspondence.items() if key != 'source_vertex_ids'}
                  | {'policy': EXPORT_POLICY, 'attribute': EXPORT_ATTRIBUTE, 'coat_vertex_count': 6*roots}, roots, name)
    import numpy as np
    triangle = _array(bindings['triangle_index'], roots, 1, True, 'triangle_index', np)
    barycentric = _array(bindings['barycentric'], roots, 3, False, 'barycentric', np)
    inset = _array(bindings['root_inset'], roots, 3, False, 'root_inset', np)
    _require(np.all(barycentric >= 0) and np.all(barycentric <= 1+1e-6)
             and np.all(abs(barycentric.sum(axis=1)-1) <= 1e-6),
             'Barycentric rows must be nonnegative and sum to 1 within 1e-6')
    _require(np.all(abs(inset) <= .0005+1e-12)
             and np.all(np.linalg.norm(inset, axis=1) <= .0005+1e-12), 'Root inset exceeds 0.5 mm')
    stream = io.BytesIO()
    if exported is None:
        np.savez(stream, triangle_index=triangle, barycentric=barycentric, root_inset=inset)
    else:
        ids = _source_ids(correspondence['source_vertex_ids'], triangle, roots, exported, np)
        np.savez(stream, triangle_index=triangle, barycentric=barycentric, root_inset=inset, source_vertex_ids=ids)
    data = stream.getvalue()
    _require(len(data) <= MAX_ARCHIVE_BYTES, 'Binding archive exceeds 512 KiB')
    root = _root(output_root)
    folder, existing = _inventory(root)
    filename = name+'.npz'
    _require(filename not in existing and not (folder/filename).is_symlink(), 'Binding archive already exists')
    _require(len(existing) < MAX_FILES and sum(x.st_size for x in existing.values())+len(data) <= MAX_FOLDER_BYTES,
             'New binding archive exceeds folder limits')
    if not folder.exists():
        folder.mkdir(mode=0o700)
    _write(folder, filename, data)
    return {'schema_version': 1 if exported is None else 2, 'path': FOLDER+'/'+filename,
            'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data), 'roots': roots,
            'space': SPACE, 'root_inset_convention': INSET_CONVENTION,
            **({'export_correspondence': exported} if exported is not None else {})}


def _receipt(value, roots):
    _require(type(value) is dict, 'Binding receipt has missing or unexpected fields')
    version = value.get('schema_version')
    _require(type(version) is int and version in (1, 2), 'Unsupported binding receipt schema or coordinate convention')
    _require(set(value) == RECEIPT_FIELDS | ({'export_correspondence'} if version == 2 else set()),
             'Binding receipt has missing or unexpected fields')
    _require(value['space'] == SPACE and value['root_inset_convention'] == INSET_CONVENTION,
             'Unsupported binding receipt schema or coordinate convention')
    _require(type(value['roots']) is int and value['roots'] == roots
             and type(value['bytes']) is int and 0 < value['bytes'] <= MAX_ARCHIVE_BYTES,
             'Binding receipt count or byte size is invalid')
    path = value['path']
    _require(type(path) is str and path.startswith(FOLDER+'/') and path.endswith('.npz'),
             'Invalid binding archive path')
    name = path[len(FOLDER)+1:-4]
    _name(name)
    _require(path == FOLDER+'/'+name+'.npz', 'Noncanonical binding archive path')
    _require(type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}', value['sha256']),
             'Invalid binding archive hash')
    if version == 2:
        _export_correspondence(value['export_correspondence'], roots, name)
    return name+'.npz'


def _npy_header(stream, size, shape, dtype):
    prefix = stream.read(8)
    _require(len(prefix) == 8 and prefix[:6] == b'\x93NUMPY' and prefix[6:] in (b'\x01\x00', b'\x02\x00'),
             'Unsupported NPY magic or version')
    width = 2 if prefix[6] == 1 else 4
    raw_length = stream.read(width)
    _require(len(raw_length) == width, 'Truncated NPY header length')
    length = int.from_bytes(raw_length, 'little')
    _require(0 < length <= MAX_NPY_HEADER_BYTES and 8+width+length <= size, 'NPY header size exceeds bounds')
    raw = stream.read(length)
    _require(len(raw) == length and raw.endswith(b'\n'), 'Truncated or malformed NPY header')
    try:
        expression = ast.parse(raw.decode('latin1').strip(), mode='eval').body
        _require(isinstance(expression, ast.Dict) and len(expression.keys) == 3
                 and all(isinstance(key, ast.Constant) and type(key.value) is str for key in expression.keys)
                 and {key.value for key in expression.keys} == {'descr', 'fortran_order', 'shape'},
                 'NPY header fields are not exact')
        header = ast.literal_eval(expression)
    except (SyntaxError, ValueError, TypeError, RecursionError, UnicodeError) as error:
        raise ValueError('Invalid NPY header: '+str(error)) from None
    _require(type(header['descr']) is str and header['descr'] == dtype, 'Unsupported NPY dtype')
    _require(type(header['fortran_order']) is bool, 'Invalid NPY layout')
    _require(type(header['shape']) is tuple and all(type(value) is int for value in header['shape'])
             and header['shape'] == shape, 'NPY shape differs from bounded receipt')
    decoded = math.prod(shape)*(4 if dtype == '<i4' else 8)
    offset = 8+width+length
    _require(offset % 16 == 0, 'NPY payload is not aligned to its format boundary')
    _require(offset+decoded == size, 'NPY payload length differs from its declared shape')
    return offset, decoded, 'F' if header['fortran_order'] else 'C'


def load_bindings(pack: Path, receipt: dict) -> dict:
    """Validate every stored NPY header before allocating any numeric array."""
    _require(type(receipt) is dict, 'Binding receipt has missing or unexpected fields')
    roots = _count(receipt.get('roots'))
    name = _receipt(receipt, roots)
    folder, files = _inventory(_root(pack))
    _require(name in files and files[name].st_size == receipt['bytes'], 'Binding archive is missing or its byte size changed')
    data = _read(folder, name, files[name])
    _require(hashlib.sha256(data).hexdigest() == receipt['sha256'], 'Binding archive hash changed')
    expected = {'triangle_index': ((roots,), '<i4'), 'barycentric': ((roots, 3), '<f8'),
                'root_inset': ((roots, 3), '<f8')}
    if receipt['schema_version'] == 2:
        expected['source_vertex_ids'] = ((roots, 3), '<i4')
    try:
        with zipfile.ZipFile(io.BytesIO(data), 'r') as archive:
            entries = archive.infolist()
            _require(len(entries) == len(expected) and {entry.filename for entry in entries} ==
                     {key+'.npy' for key in expected}, 'ZIP entries differ from the exact binding arrays')
            for entry in entries:
                _require(entry.compress_type == zipfile.ZIP_STORED and not entry.flag_bits & 1
                         and entry.compress_size == entry.file_size
                         and 0 < entry.file_size <= MAX_DECODED_BYTES+MAX_NPY_HEADER_BYTES+12
                         and not stat.S_ISLNK(entry.external_attr >> 16), 'Unsupported ZIP entry encoding or size')
            layouts, total = {}, 0
            for key, (shape, dtype) in expected.items():
                entry = archive.getinfo(key+'.npy')
                with archive.open(entry) as stream:
                    layouts[key] = _npy_header(stream, entry.file_size, shape, dtype)
                total += layouts[key][1]
            _require(total <= MAX_DECODED_BYTES, 'Binding decoded arrays exceed their aggregate bound')
            import numpy as np
            result = {}
            for key, (shape, dtype) in expected.items():
                raw = archive.read(key+'.npy')
                offset, _, order = layouts[key]
                result[key] = np.frombuffer(raw, dtype=dtype, offset=offset).reshape(shape, order=order).copy()
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError, EOFError, struct.error) as error:
        raise ValueError('Invalid binding archive: '+str(error)) from None
    triangle, bary, inset = result['triangle_index'], result['barycentric'], result['root_inset']
    _require(np.all(triangle >= 0) and np.all(triangle <= 99999), 'Triangle index outside0..99999')
    _require(np.isfinite(bary).all() and np.all(bary >= 0) and np.all(bary <= 1+1e-6)
             and np.all(abs(bary.sum(axis=1)-1) <= 1e-6), 'Invalid barycentric values')
    _require(np.isfinite(inset).all() and np.all(abs(inset) <= .0005+1e-12)
             and np.all(np.linalg.norm(inset, axis=1) <= .0005+1e-12), 'Root inset exceeds0.5mm or is nonfinite')
    if receipt['schema_version'] == 2:
        _source_ids(result['source_vertex_ids'], triangle, roots, receipt['export_correspondence'], np)
    return result


def copy_bindings(source_root: Path, dest_root: Path, parts: list, expected_roots: int) -> list[dict]:
    """Copy hash-bound bytes, not semantic evidence; the independent loader validates arrays."""
    total = _count(expected_roots)
    _require(type(parts) is list and 1 <= len(parts) <= MAX_FILES, 'Require 1..128 portable parts')
    source, dest = _root(source_root), _root(dest_root)
    _require(source != dest and not dest.is_relative_to(source), 'Destination cannot alter the source root')
    folder, files = _inventory(source)
    target, target_files = _inventory(dest)
    _require(not target_files, 'Destination binding folder must be new or empty')
    receipts, names, count = [], [], 0
    for part in parts:
        _require(type(part) is dict, 'Invalid portable part')
        roots = _count(part.get('cards'))
        summary = part.get('root_summary')
        _require(type(summary) is dict, 'Binding copy requires a root summary')
        receipt = summary.get('binding_archive')
        names.append(_receipt(receipt, roots))
        if receipt['schema_version'] == 1:
            _require(summary.get('root_placement') == 'skin-consistent-v1',
                     'Binding copy requires skin-consistent-v1 root placement')
        else:
            _require(part.get('geometry_profile') in ('legacy', 'combed-v1', 'tuft-v1'), 'Unsupported portable card geometry profile')
        receipts.append(dict(receipt) | ({'export_correspondence': dict(receipt['export_correspondence'])}
                                       if receipt['schema_version'] == 2 else {}))
        count += roots
        _require(count <= MAX_ROOTS, 'Aggregate root count exceeds 5000')
    _require(count == total and len(set(names)) == len(names), 'Root count mismatch or duplicate binding archive')
    _require(set(files) == set(names), 'Missing or extra binding files')
    payloads = []
    for name, receipt in zip(names, receipts):
        _require(files[name].st_size == receipt['bytes'], 'Binding archive byte size changed')
        data = _read(folder, name, files[name])
        _require(hashlib.sha256(data).hexdigest() == receipt['sha256'], 'Binding archive hash changed')
        payloads.append(data)
    # All source receipts and bytes pass before the destination is changed.
    if not target.exists():
        target.mkdir(mode=0o700)
    for name, data in zip(names, payloads):
        _write(target, name, data)
    return receipts
