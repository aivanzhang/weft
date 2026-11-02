"""Numeric archive creation and byte custody; no native or appearance claims."""

from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
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


class FurBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='fur-bindings-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root/'source'; self.source.mkdir()
        self.dest = self.root/'dest'; self.dest.mkdir()

    def write(self, name='PortableCoat_0', count=2, bindings=None):
        return fb.write_bindings(self.source, name, data(count) if bindings is None else bindings, count)

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


if __name__=='__main__':
    unittest.main()
