"""Numeric archive creation and byte custody; no native or appearance claims."""

from copy import deepcopy
import hashlib
import importlib.util
import io
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

import numpy as np

from avatar_assets import fur_bindings as fb


def data(count=2):
    return {'triangle_index': [i % 100000 for i in range(count)],
            'barycentric': [[.25, .25, .5] for _ in range(count)],
            'root_inset': [[0, 0, .0002] for _ in range(count)]}


def part(receipt):
    return {'cards': receipt['roots'], 'root_summary': {
        'root_placement': 'skin-consistent-v1', 'binding_archive': deepcopy(receipt)}}


def correspondence(name='PortableCoat_0', count=2):
    return {'source_node': 'Furred Body 鎧', 'coat_node': name, 'source_vertex_count': 16,
            'source_triangle_count': 100000, 'source_vertex_ids': [[4, 9, 12] for _ in range(count)]}


def npy_bytes(header, payload=b'', *, version=1):
    text = repr(header).encode('latin1')
    prefix = b'\x93NUMPY'+bytes((version, 0))
    width = 2 if version == 1 else 4
    padding = (-(len(prefix)+width+len(text)+1)) % 16
    text += b' '*padding+b'\n'
    return prefix+len(text).to_bytes(width, 'little')+text+payload


class FurBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='fur-bindings-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root/'source'; self.source.mkdir()
        self.dest = self.root/'dest'; self.dest.mkdir()

    def write(self, name='PortableCoat_0', count=2, bindings=None):
        return fb.write_bindings(self.source, name, data(count) if bindings is None else bindings, count)

    def write_v2(self, name='PortableCoat_0', count=2, bindings=None, source=None):
        return fb.write_bindings(self.source, name, data(count) if bindings is None else bindings, count,
                                 correspondence=correspondence(name, count) if source is None else source)

    def replace_bytes(self, receipt, raw):
        (self.source/receipt['path']).write_bytes(raw)
        return {**receipt, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

    def members(self, receipt):
        with zipfile.ZipFile(self.source/receipt['path']) as archive:
            return [(entry.filename, archive.read(entry)) for entry in archive.infolist()]

    def replace_members(self, receipt, members, compression=zipfile.ZIP_STORED):
        stream = io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with zipfile.ZipFile(stream, 'w', compression=compression) as archive:
                for name, raw in members:
                    archive.writestr(name, raw)
        return self.replace_bytes(receipt, stream.getvalue())

    def reject_before_numeric(self, receipt):
        with patch.object(np, 'frombuffer', side_effect=AssertionError('Numeric allocation preceded admission')), \
                patch.object(np, 'load', side_effect=AssertionError('Reader must not use np.load')):
            with self.assertRaises(ValueError):
                fb.load_bindings(self.source, receipt)

    def reject_copy(self, parts, expected=2):
        with self.assertRaises((ValueError, OSError)):
            fb.copy_bindings(self.source, self.dest, parts, expected)
        self.assertFalse((self.dest/fb.FOLDER).exists())

    def test_writer_stores_exact_bounded_arrays_and_receipt(self):
        inputs = data(); inputs['triangle_index'] = [0, 99999]
        before = deepcopy(inputs)
        receipt = self.write(bindings=inputs)
        self.assertEqual(inputs, before)
        self.assertEqual(set(receipt), fb.RECEIPT_FIELDS)
        self.assertEqual(receipt['path'], 'portable-root-bindings/PortableCoat_0.npz')
        self.assertEqual(receipt['space'], 'blender_local_z_up_meters')
        self.assertEqual(receipt['root_inset_convention'], 'surface_root_minus_bottom_edge_midpoint')
        archive = self.source/receipt['path']; blob = archive.read_bytes()
        self.assertEqual(receipt['bytes'], len(blob))
        self.assertEqual(receipt['sha256'], hashlib.sha256(blob).hexdigest())
        with zipfile.ZipFile(archive) as z:
            self.assertEqual(set(z.namelist()), {'triangle_index.npy', 'barycentric.npy', 'root_inset.npy'})
            self.assertTrue(all(x.compress_type == zipfile.ZIP_STORED for x in z.infolist()))
        with np.load(archive, allow_pickle=False) as z:
            for key, shape, dtype in [('triangle_index', (2,), '<i4'), ('barycentric', (2,3), '<f8'),
                                      ('root_inset', (2,3), '<f8')]:
                self.assertEqual(z[key].shape, shape)
                self.assertEqual(z[key].dtype.str, dtype)
                np.testing.assert_array_equal(z[key], np.asarray(inputs[key], dtype=dtype))

    def test_noncontiguous_arrays_are_detached_without_mutation(self):
        bary = np.tile([.2, 9, .3, 9, .5, 9], (4, 1))[:, ::2]
        inset = np.zeros((4, 6))[:, ::2]
        inputs = {'triangle_index': np.arange(8, dtype=np.int64)[::2],
                  'barycentric': bary, 'root_inset': inset}
        originals = {k:v.copy() for k,v in inputs.items()}
        receipt = self.write(count=4, bindings=inputs)
        for key in inputs:
            np.testing.assert_array_equal(inputs[key], originals[key])
        bary[0, 0] = 99
        with np.load(self.source/receipt['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(z['barycentric'], originals['barycentric'])

    def test_maximum_5000_roots_fit_archive_bound(self):
        receipt = self.write(count=5000)
        self.assertLessEqual(receipt['bytes'], 512*1024)
        with np.load(self.source/receipt['path'], allow_pickle=False) as z:
            self.assertEqual(z['root_inset'].shape, (5000,3))

    def test_one_hot_and_inset_boundary_remain_exact(self):
        inputs={'triangle_index':[7,9], 'barycentric':[[1,0,0],[0,.5,.5000005]],
                'root_inset':[[.0005,0,0],[0,0,0]]}
        receipt=self.write(bindings=inputs)
        with np.load(self.source/receipt['path'],allow_pickle=False) as z:
            np.testing.assert_array_equal(z['barycentric'],inputs['barycentric'])
            np.testing.assert_array_equal(z['root_inset'],inputs['root_inset'])

    def test_shapes_and_container_types_reject_before_array_conversion(self):
        class ConversionTrap:
            def __array__(self, *args, **kwargs):
                raise AssertionError('Untrusted conversion was called')
        invalid = [ConversionTrap(), np.zeros((2,2), dtype=np.int32),
                   np.zeros(5001, dtype=np.int32), np.array([1,2], dtype=object),
                   (1,2), [1], [True,0], [1.0,2.0]]
        for value in invalid:
            with self.subTest(value=type(value).__name__):
                inputs = data(); inputs['triangle_index'] = value
                with patch.object(np, 'asarray', side_effect=AssertionError('Premature conversion')):
                    with self.assertRaises(ValueError):
                        self.write(bindings=inputs)
                self.assertFalse((self.source/fb.FOLDER).exists())

    def test_invalid_numeric_values_and_schema_leave_no_output(self):
        cases=[]
        for value in (-1, 100000, 10**1000):
            row=data(); row['triangle_index'][0]=value; cases.append(row)
        for value in ([True,0,0], [float('nan'),0,1], [float('inf'),0,1],
                      [-1e-12, .5, .500000000001], [.2,.2,.2], [1e308,1e308,0]):
            row=data(); row['barycentric'][0]=value; cases.append(row)
        for value in ([0,0,.00050001], [.0004,.0004,0], [0,0,float('nan')],
                      [0,False,0], [1e308,0,0]):
            row=data(); row['root_inset'][0]=value; cases.append(row)
        row=data(); row['root_inset']=np.zeros((2,3), dtype=bool); cases.append(row)
        row=data(); row['barycentric']=np.zeros((2,3), dtype=complex); cases.append(row)
        row=data(); row['extra']=[]; cases.append(row)
        row=data(); del row['root_inset']; cases.append(row)
        for index, inputs in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises(ValueError):
                    self.write(bindings=inputs)
                self.assertFalse((self.source/fb.FOLDER).exists())

    def test_counts_and_names_fail_before_writes(self):
        for count in (0, 5001, True, 2.0):
            with self.subTest(count=count), self.assertRaises(ValueError):
                fb.write_bindings(self.source, 'PortableCoat_0', data(), count)
        for name in ('', 'Coat_0', '../PortableCoat_0', 'PortableCoat_a/b',
                     'PortableCoat_..', 'PortableCoat_ä', 'PortableCoat_'+('a'*84)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.write(name=name)
        self.assertFalse((self.source/fb.FOLDER).exists())

    def test_no_overwrite_and_blender_suffix_name(self):
        receipt=self.write(name='PortableCoat_0.001')
        path=self.source/receipt['path']; before=path.read_bytes()
        with self.assertRaises(ValueError):
            self.write(name='PortableCoat_0.001')
        self.assertEqual(path.read_bytes(),before)

    def test_writer_rejects_folder_file_and_byte_limits(self):
        folder=self.source/fb.FOLDER; folder.mkdir()
        for i in range(128):
            (folder/f'PortableCoat_{i}.npz').write_bytes(b'analytic inventory byte')
        with self.assertRaises(ValueError):
            self.write(name='PortableCoat_over')
        other=self.root/'bytes'; other.mkdir(); huge=other/fb.FOLDER; huge.mkdir()
        for i in range(2):
            (huge/f'PortableCoat_{i}.npz').write_bytes(b'0'*(512*1024))
        with self.assertRaises(ValueError):
            fb.write_bindings(other,'PortableCoat_over',data(),2)
        self.assertEqual(len(list(huge.iterdir())),2)

    def test_writer_rejects_symlink_root_folder_and_archive(self):
        alias=self.root/'alias'; alias.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(ValueError):
            fb.write_bindings(alias,'PortableCoat_0',data(),2)
        target=self.root/'elsewhere'; target.mkdir()
        folder=self.source/fb.FOLDER; folder.symlink_to(target, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.write()
        self.assertEqual(list(target.iterdir()),[])
        folder.unlink(); folder.mkdir()
        (folder/'PortableCoat_0.npz').symlink_to(self.root/'absent')
        with self.assertRaises(ValueError):
            self.write()

    def test_copy_preserves_all_bytes_receipts_and_source_without_numpy_decode(self):
        receipts=[self.write(),self.write(name='PortableCoat_1',count=3)]
        before={r['path']:(self.source/r['path']).read_bytes() for r in receipts}
        with patch.object(np,'load',side_effect=AssertionError('Copy must not claim semantic decoding')):
            actual=fb.copy_bindings(self.source,self.dest,[part(r) for r in receipts],5)
        self.assertEqual(actual,receipts)
        for path,raw in before.items():
            self.assertEqual((self.dest/path).read_bytes(),raw)
            self.assertEqual((self.source/path).read_bytes(),raw)

    def test_copy_validates_every_archive_before_creating_destination(self):
        first=self.write(); second=self.write(name='PortableCoat_1')
        first_before=(self.source/first['path']).read_bytes()
        p=self.source/second['path']; raw=p.read_bytes()
        p.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
        corrupted=p.read_bytes()
        self.reject_copy([part(first),part(second)],4)
        self.assertEqual((self.source/first['path']).read_bytes(),first_before)
        self.assertEqual(p.read_bytes(),corrupted)

    def test_copy_missing_extra_and_oversized_files_reject(self):
        receipt=self.write(); path=self.source/receipt['path']; raw=path.read_bytes()
        path.unlink(); self.reject_copy([part(receipt)])
        path.write_bytes(raw)
        extra=self.source/fb.FOLDER/'PortableCoat_extra.npz'; extra.write_bytes(b'extra')
        self.reject_copy([part(receipt)]); extra.unlink()
        path.write_bytes(b'x'*(fb.MAX_ARCHIVE_BYTES+1))
        changed=deepcopy(receipt); changed['bytes']=path.stat().st_size
        changed['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        self.reject_copy([part(changed)])

    def test_copy_rejects_receipt_schema_coordinates_count_and_hash_changes(self):
        receipt=self.write()
        mutations=[('schema_version',True),('schema_version',2),('space','gltf_y_up'),
                   ('root_inset_convention','opposite'),('roots',True),('roots',1),
                   ('bytes',True),('bytes',receipt['bytes']+1),('sha256','0'*64),('sha256','A'*64)]
        for key,value in mutations:
            with self.subTest(key=key,value=value):
                modified=part(receipt); modified['root_summary']['binding_archive'][key]=value
                self.reject_copy([modified])
        for key in ('missing','extra'):
            modified=part(receipt)
            if key=='missing': del modified['root_summary']['binding_archive']['roots']
            else: modified['root_summary']['binding_archive']['extra']=0
            self.reject_copy([modified])

    def test_copy_rejects_wrong_placement_missing_receipt_duplicates_and_total(self):
        receipt=self.write()
        modified=part(receipt); modified['root_summary']['root_placement']='barycentric'
        self.reject_copy([modified])
        modified=part(receipt); del modified['root_summary']['binding_archive']
        self.reject_copy([modified])
        modified=part(receipt); modified['cards']=True
        self.reject_copy([modified])
        self.reject_copy([part(receipt),part(receipt)],4)
        self.reject_copy([part(receipt)],3)
        self.reject_copy([],2)

    def test_copy_rejects_global_root_limit_and_too_many_parts(self):
        one=self.write(count=3000); two=self.write(name='PortableCoat_1',count=3000)
        self.reject_copy([part(one),part(two)],5000)
        self.reject_copy([part(one)]*129,3000)

    def test_copy_aggregate_byte_limit_precedes_copy_even_with_matching_hashes(self):
        folder=self.source/fb.FOLDER; folder.mkdir()
        payload=b'analytic non-decoded custody bytes'.ljust(512*1024,b'x')
        receipts=[]
        for index in range(3):
            name=f'PortableCoat_{index}.npz'; (folder/name).write_bytes(payload)
            receipts.append({'schema_version':1,'path':fb.FOLDER+'/'+name,
                'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload),'roots':1,
                'space':fb.SPACE,'root_inset_convention':fb.INSET_CONVENTION})
        self.reject_copy([part(x) for x in receipts],3)

    def test_copy_rejects_traversal_and_symlink_escape(self):
        receipt=self.write()
        for path in ('../portable-root-bindings/PortableCoat_0.npz',
                     'portable-root-bindings/../PortableCoat_0.npz',
                     '/portable-root-bindings/PortableCoat_0.npz',
                     'portable-root-bindings/PortableCoat_a/PortableCoat_0.npz'):
            modified=part(receipt); modified['root_summary']['binding_archive']['path']=path
            self.reject_copy([modified])
        original=self.source/receipt['path']; external=self.root/'external.npz'
        original.rename(external); original.symlink_to(external)
        self.reject_copy([part(receipt)])

    def test_copy_rejects_existing_destination_and_source_overlap(self):
        receipt=self.write(); folder=self.dest/fb.FOLDER; folder.mkdir()
        saved=folder/'PortableCoat_saved.npz'; saved.write_bytes(b'untouched')
        with self.assertRaises(ValueError):
            fb.copy_bindings(self.source,self.dest,[part(receipt)],2)
        self.assertEqual(saved.read_bytes(),b'untouched')
        with self.assertRaises(ValueError):
            fb.copy_bindings(self.source,self.source,[part(receipt)],2)
        nested=self.source/'nested'; nested.mkdir()
        with self.assertRaises(ValueError):
            fb.copy_bindings(self.source,nested,[part(receipt)],2)

    def test_copy_rejects_symlink_destination_folder(self):
        receipt=self.write(); elsewhere=self.root/'elsewhere'; elsewhere.mkdir()
        (self.dest/fb.FOLDER).symlink_to(elsewhere,target_is_directory=True)
        with self.assertRaises(ValueError):
            fb.copy_bindings(self.source,self.dest,[part(receipt)],2)
        self.assertEqual(list(elsewhere.iterdir()),[])

    def test_standalone_snapshot_import_does_not_import_numpy_or_avatar_package(self):
        import builtins
        original=builtins.__import__
        def checked(name,*args,**kwargs):
            if name.startswith(('numpy','avatar_assets')):
                raise AssertionError('Standalone import has an eager package dependency')
            return original(name,*args,**kwargs)
        spec=importlib.util.spec_from_file_location('standalone_fur_bindings',Path(fb.__file__))
        module=importlib.util.module_from_spec(spec)
        with patch('builtins.__import__',side_effect=checked):
            spec.loader.exec_module(module)
        self.assertTrue(callable(module.write_bindings))
        self.assertTrue(callable(module.copy_bindings))
        self.assertTrue(callable(module.load_bindings))

    def test_v1_writer_exact_bytes_and_reader_remain_compatible(self):
        receipt = self.write()
        self.assertEqual(receipt['sha256'], '7ed97eb40d71eda21f57dfc1807503dbe0f80f439dbc2de93ca00ca32dbe9f51')
        self.assertEqual(receipt['bytes'], 892)
        result = fb.load_bindings(self.source, receipt)
        self.assertEqual(set(result), {'triangle_index', 'barycentric', 'root_inset'})
        for key, value in data().items():
            np.testing.assert_array_equal(result[key], value)
        self.assertNotIn('export_correspondence', receipt)

    def test_v2_exact_correspondence_roundtrip_and_returned_arrays_are_detached(self):
        inputs, source = data(), correspondence()
        before = deepcopy((inputs, source))
        receipt = self.write_v2(bindings=inputs, source=source)
        self.assertEqual((inputs, source), before)
        self.assertEqual(set(receipt), fb.RECEIPT_FIELDS | {'export_correspondence'})
        self.assertEqual(receipt['schema_version'], 2)
        self.assertEqual(receipt['export_correspondence'], {
            'policy': 'portable-fur-source-ids-v1', 'attribute': '_AVATAR_FUR_VERTEX_ID',
            'source_node': source['source_node'], 'coat_node': 'PortableCoat_0',
            'source_vertex_count': 16, 'source_triangle_count': 100000, 'coat_vertex_count': 12})
        original = (self.source/receipt['path']).read_bytes()
        result = fb.load_bindings(self.source, receipt)
        self.assertEqual(set(result), set(inputs)|{'source_vertex_ids'})
        for key in inputs:
            np.testing.assert_array_equal(result[key], inputs[key])
        np.testing.assert_array_equal(result['source_vertex_ids'], source['source_vertex_ids'])
        for key in result:
            self.assertEqual(result[key].dtype.str, '<i4' if key in ('triangle_index', 'source_vertex_ids') else '<f8')
        result['barycentric'][0, 0] = 99
        self.assertEqual((self.source/receipt['path']).read_bytes(), original)

    def test_v2_maximum_count_and_fortran_arrays_preserve_numeric_values(self):
        receipt = self.write_v2(count=5000)
        result = fb.load_bindings(self.source, receipt)
        self.assertEqual(result['source_vertex_ids'].shape, (5000, 3))
        self.assertLess(receipt['bytes'], fb.MAX_ARCHIVE_BYTES)
        inputs = data(); inputs['barycentric'] = np.asfortranarray(inputs['barycentric'])
        source = correspondence('PortableCoat_F'); source['source_vertex_ids'] = np.asfortranarray(source['source_vertex_ids'])
        receipt = self.write_v2(name='PortableCoat_F', bindings=inputs, source=source)
        result = fb.load_bindings(self.source, receipt)
        np.testing.assert_array_equal(result['barycentric'], inputs['barycentric'])
        np.testing.assert_array_equal(result['source_vertex_ids'], source['source_vertex_ids'])

    def test_v2_source_names_counts_ids_and_schema_reject_before_writes(self):
        invalid = []
        for key, value in (('source_node', ''), ('source_node', 'x\x7f'), ('source_node', 'x\x85'),
                           ('source_node', '\ud800'), ('source_node', 'é'*128), ('coat_node', 'PortableCoat_other'),
                           ('source_vertex_count', True), ('source_vertex_count', 100001),
                           ('source_triangle_count', 0), ('source_triangle_count', 1),
                           ('source_vertex_ids', [[4, 4, 12], [4, 9, 12]]),
                           ('source_vertex_ids', [[4, 9, 16], [4, 9, 12]]),
                           ('source_vertex_ids', [[-1, 9, 12], [4, 9, 12]]),
                           ('source_vertex_ids', [[4., 9, 12], [4, 9, 12]]),
                           ('source_vertex_ids', [[True, 9, 12], [4, 9, 12]]),
                           ('source_vertex_ids', np.zeros((5001, 3), dtype=np.int32))):
            row = correspondence(); row[key] = value; invalid.append(row)
        row = correspondence(); row['unexpected'] = 0; invalid.append(row)
        row = correspondence(); del row['source_vertex_ids']; invalid.append(row)
        for index, source in enumerate(invalid):
            with self.subTest(case=index), self.assertRaises(ValueError):
                self.write_v2(source=source)
            self.assertFalse((self.source/fb.FOLDER).exists())
        valid = correspondence(); valid['source_node'] = 'é'*127+'x'
        self.assertEqual(self.write_v2(source=valid)['export_correspondence']['source_node'], valid['source_node'])

    def test_v2_copy_supports_all_three_profiles_without_numeric_decoding(self):
        receipts, parts = [], []
        for index, profile in enumerate(('legacy', 'combed-v1', 'tuft-v1')):
            receipt = self.write_v2(name=f'PortableCoat_{index}'); receipts.append(receipt)
            row = part(receipt); row['geometry_profile'] = profile
            row['root_summary']['root_placement'] = 'area-stratified-v1' if profile == 'tuft-v1' else 'area-random-v1'
            parts.append(row)
        with patch.object(np, 'load', side_effect=AssertionError('No numeric copy decoding')), \
                patch.object(np, 'frombuffer', side_effect=AssertionError('No numeric copy decoding')):
            copied = fb.copy_bindings(self.source, self.dest, parts, 6)
        self.assertEqual(copied, receipts)
        for receipt in receipts:
            self.assertEqual((self.dest/receipt['path']).read_bytes(), (self.source/receipt['path']).read_bytes())
        copied[0]['export_correspondence']['source_node'] = 'Changed returned metadata'
        self.assertNotEqual(copied[0], receipts[0])
        self.assertEqual(parts[0]['root_summary']['binding_archive'], receipts[0])

    def test_v2_copy_rejects_wrong_profiles_or_correspondence_without_weakening_v1(self):
        receipt = self.write_v2()
        for profile in (None, 'unknown', True):
            row = part(receipt); row['geometry_profile'] = profile
            self.reject_copy([row])
        for key, value in (('policy', 'other'), ('attribute', '_WRONG'), ('coat_node', 'PortableCoat_other'),
                           ('coat_vertex_count', 13), ('source_vertex_count', False), ('source_triangle_count', 100001)):
            row = part(receipt); row['geometry_profile'] = 'legacy'
            row['root_summary']['binding_archive']['export_correspondence'][key] = value
            self.reject_copy([row])
        old = self.write(name='PortableCoat_old')
        row = part(old); row['geometry_profile'] = 'legacy'; row['root_summary']['root_placement'] = 'area-random-v1'
        self.reject_copy([row])

    def test_reader_validates_receipt_hash_paths_and_size_before_numeric_allocation(self):
        receipt = self.write_v2()
        cases = []
        for key, value in (('roots', 5001), ('bytes', fb.MAX_ARCHIVE_BYTES+1), ('bytes', receipt['bytes']+1),
                           ('sha256', '0'*64), ('schema_version', True), ('schema_version', 3),
                           ('path', '../PortableCoat_0.npz'), ('space', 'other')):
            cases.append({**receipt, key: value})
        for key, value in (('source_node', 'bad\nname'), ('source_vertex_count', 0), ('coat_vertex_count', 13)):
            row = deepcopy(receipt); row['export_correspondence'][key] = value; cases.append(row)
        for row in cases:
            with self.subTest(changes=row), patch.object(np, 'frombuffer', side_effect=AssertionError('Premature allocation')):
                with self.assertRaises(ValueError): fb.load_bindings(self.source, row)
        original = self.source/receipt['path']; external = self.root/'outside.npz'
        original.rename(external); original.symlink_to(external)
        self.reject_before_numeric(receipt)

    def test_reader_rejects_exact_entry_mismatches_compression_and_zip_size_bombs(self):
        receipt = self.write(); members = self.members(receipt)
        cases = [members[:-1], members+[('extra.npy', b'ignored')], members+[members[0]],
                 members[:-1]+[('../root_inset.npy', members[-1][1])]]
        for entries in cases:
            with self.subTest(names=[row[0] for row in entries]):
                self.reject_before_numeric(self.replace_members(receipt, entries))
        self.reject_before_numeric(self.replace_members(receipt, members, zipfile.ZIP_DEFLATED))
        for field in ('size', 'encrypted'):
            valid = self.replace_members(receipt, members)
            raw = bytearray((self.source/valid['path']).read_bytes()); central = raw.index(b'PK\x01\x02')
            if field == 'size': struct.pack_into('<I', raw, central+24, 2**31)
            else: raw[central+8] |= 1
            self.reject_before_numeric(self.replace_bytes(receipt, bytes(raw)))
        self.reject_before_numeric(self.replace_bytes(receipt, b'Not a ZIP archive'))

    def test_reader_checks_every_npy_header_before_allocating_the_first_array(self):
        receipt = self.write(); members = self.members(receipt)
        base = {'descr': '<f8', 'fortran_order': False, 'shape': (2, 3)}
        variants = []
        for key, value in (('shape', (10**18, 3)), ('shape', (-1, 3)), ('shape', (True, 3)),
                           ('shape', [2, 3]), ('shape', (2, 3, 1)), ('descr', '|O'), ('descr', '>f8'),
                           ('descr', '<f4'), ('descr', [('field', '<f8')]), ('fortran_order', 1)):
            header = {**base, key: value}; variants.append(npy_bytes(header, b'\0'*48))
        variants += [npy_bytes(base, b'\0'*47), npy_bytes(base, b'\0'*49),
                     b'\x93NUMPY\x01\x00'+struct.pack('<H', 65535),
                     b'\x93NUMPY\x03\x00'+b'\0'*120]
        for index, raw in enumerate(variants):
            changed = members[:-1]+[('root_inset.npy', raw)]
            with self.subTest(case=index):
                self.reject_before_numeric(self.replace_members(receipt, changed))

    def test_reader_rejects_malformed_duplicate_header_fields_and_crc(self):
        receipt = self.write(); members = self.members(receipt)
        header = b"{'descr':'<f8','shape':(2,3),'shape':(2,3),'fortran_order':False}"
        padding = (-(10+len(header)+1)) % 16
        header += b' '*padding+b'\n'
        raw = b'\x93NUMPY\x01\x00'+struct.pack('<H', len(header))+header+b'\0'*48
        self.reject_before_numeric(self.replace_members(receipt, members[:-1]+[('root_inset.npy', raw)]))
        valid = self.replace_members(receipt, members)
        raw = bytearray((self.source/valid['path']).read_bytes())
        first = raw.index(b'\x93NUMPY'); raw[first+len(members[0][1])-1] ^= 1
        self.reject_before_numeric(self.replace_bytes(receipt, bytes(raw)))

    def test_reader_validates_numeric_values_and_source_id_ownership_after_safe_load(self):
        receipt = self.write_v2(); members = self.members(receipt)
        mutations = [('triangle_index', np.array([-1, 1], dtype='<i4')),
                     ('barycentric', np.array([[np.nan, 0, 1], [.25, .25, .5]], dtype='<f8')),
                     ('barycentric', np.array([[.1, .1, .1], [.25, .25, .5]], dtype='<f8')),
                     ('root_inset', np.array([[.0004, .0004, 0], [0, 0, 0]], dtype='<f8')),
                     ('source_vertex_ids', np.array([[4, 4, 12], [4, 9, 12]], dtype='<i4')),
                     ('source_vertex_ids', np.array([[4, 9, 16], [4, 9, 12]], dtype='<i4'))]
        for key, array in mutations:
            payload = npy_bytes({'descr': array.dtype.str, 'fortran_order': False, 'shape': array.shape}, array.tobytes())
            changed = [(name, payload if name == key+'.npy' else raw) for name, raw in members]
            valid = self.replace_members(receipt, changed)
            with self.subTest(array=key), self.assertRaises(ValueError):
                fb.load_bindings(self.source, valid)
        valid = self.replace_members(receipt, members)
        valid = deepcopy(valid); valid['export_correspondence']['source_triangle_count'] = 1
        with self.assertRaisesRegex(ValueError, 'triangle ordinal'):
            fb.load_bindings(self.source, valid)


if __name__=='__main__':
    unittest.main()
