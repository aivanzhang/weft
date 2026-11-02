"""Analytic full-skin custody and source-ID tests; no native or appearance proof."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from avatar_assets.fringe_export import _array
from avatar_assets.safety import sha256
from avatar_assets.source_skin_fidelity import ATTRIBUTE, restore_source_skin
from avatar_assets.three_d import _GLB
from tests.test_fringe_fidelity import authored_fixture
from tests.test_three_d import glb_bytes


def add_array(doc, binary, values, kind):
    value = np.ascontiguousarray(values)
    component = {'<f4':5126,'<u2':5123,'<u4':5125,'|u1':5121}[value.dtype.str]
    binary.extend(b'\0'*(-len(binary)%4)); offset = len(binary)
    binary.extend(value.tobytes())
    doc['bufferViews'].append({'buffer':0,'byteOffset':offset,'byteLength':value.nbytes})
    doc['accessors'].append({'bufferView':len(doc['bufferViews'])-1,'componentType':component,
                             'count':len(value),'type':kind})
    doc['buffers'][0]['byteLength'] = len(binary)
    return len(doc['accessors'])-1


def raw_array(doc, binary, aid):
    a = doc['accessors'][aid]; v = doc['bufferViews'][a['bufferView']]
    dtype = np.dtype({5126:'<f4',5123:'<u2',5125:'<u4'}[a['componentType']])
    width = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']]
    return np.ndarray((a['count'],width),dtype=dtype,buffer=binary,
                      offset=v.get('byteOffset',0)+a.get('byteOffset',0),
                      strides=(v.get('byteStride',width*dtype.itemsize),dtype.itemsize))


def skin_fixture(count=2):
    doc,binary,helper,_,_ = authored_fixture(count)
    attrs = [mesh['primitives'][0]['attributes'] for mesh in doc['meshes']]
    native_map = helper['triangles'].ravel()
    body = raw_array(doc,binary,attrs[0]['POSITION']).copy()[:,[0,2,1]]; body[:,1]*=-1
    positions = np.concatenate((body,helper['positions'][native_map])).astype('<f4')
    n = len(positions)
    joints = np.tile([0,1,2,3],(n,1)).astype('<u2')
    weights = np.tile(helper['weights'][0],(n,1)).astype('<f4')
    triangles = np.concatenate((np.array([[0,1,2]],dtype='<u4'),
                                np.arange(3,n,dtype='<u4').reshape(-1,3)))
    arrays = {'positions':positions,'joints':joints,'weights':weights,'triangles':triangles,
              'triangle_materials':np.zeros(len(triangles),dtype='<u2')}
    raw_array(doc,binary,attrs[0]['JOINTS_0'])[:] = [1,0,3,2]
    pruned = weights[:3].copy(); pruned[pruned <= .0001] = 0; pruned /= pruned.sum(axis=1)[:,None]
    raw_array(doc,binary,attrs[0]['WEIGHTS_0'])[:] = pruned
    attrs[0][ATTRIBUTE] = add_array(doc,binary,np.arange(3,dtype='<f4')[:,None],'SCALAR')
    attrs[1][ATTRIBUTE] = add_array(doc,binary,np.arange(3,n,dtype='<f4')[::-1,None],'SCALAR')
    owners = []
    for index,(name,start,vc,ts,tc,mat) in enumerate((('Body',0,3,0,1,'NormalMappedBody'),
            ('Fringe',3,n-3,1,len(triangles)-1,'SourceColoredFringe'))):
        doc['meshes'][index]['name'] = name+'Mesh'
        owners.append({'index':index,'name':name,'mesh_name':name+'Mesh','vertex_start':start,
            'vertex_count':vc,'triangle_start':ts,'triangle_count':tc,'materials':[mat],
            'ignored_non_skin_groups':['NativeGroomMask'] if index == 0 else []})
    report = {'schema_version':1,'policy':'source-skin-id-v1','attribute':ATTRIBUTE,
        'space':'applied_rest_Blender_XYZ_meters','vertex_count':n,'triangle_count':len(triangles),
        'bone_names':['B','A','D','C'],'rig':{'name':'Rig','data_name':'RigData','pose_position':'REST'},
        'owners':owners,'normal_snapshot_included':False}
    return doc,binary,arrays,report


class SourceSkinFidelityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source,self.target,self.snapshot,self.report_path = [self.root/name for name in
            ('source.glb','result.glb','source-skin.npz','source-skin.json')]
        self.reset()

    def reset(self):
        self.doc,self.binary,self.arrays,self.report = skin_fixture()

    def prepare(self):
        self.source.write_bytes(glb_bytes(self.doc,self.binary))
        np.savez_compressed(self.snapshot,**self.arrays)
        self.report.update(npz_file=self.snapshot.name,npz_sha256=sha256(self.snapshot),
            npz_bytes=self.snapshot.stat().st_size,arrays={key:{'dtype':value.dtype.str,
                'shape':list(value.shape),'sha256':hashlib.sha256(value.tobytes(order='C')).hexdigest()}
                for key,value in self.arrays.items()})
        self.report_path.write_text(json.dumps(self.report))
        return {'source':sha256(self.source),'snapshot':sha256(self.snapshot),'snapshot_report':sha256(self.report_path)}

    def execute(self, pins=None):
        return restore_source_skin(self.source,self.target,self.snapshot,self.report_path,
                                   experimental=True,expected_hashes=pins or self.prepare())

    def reject(self, pattern='.'):
        pins = self.prepare()
        with self.assertRaisesRegex(ValueError,pattern):self.execute(pins)
        self.assertFalse(self.target.exists())

    def assert_custody(self, result):
        before,after = _GLB(self.source),_GLB(self.target)
        expected = deepcopy(before.doc)
        allowed = np.zeros(len(before.binary),dtype=bool)
        for mi,mesh in enumerate(before.doc['meshes']):
            for pi,primitive in enumerate(mesh['primitives']):
                attrs = primitive['attributes']
                ids = _array(before,attrs[ATTRIBUTE],'SCALAR',{5126}).ravel().astype(int)
                expected['meshes'][mi]['primitives'][pi]['attributes'].pop(ATTRIBUTE)
                self.assertTrue(np.array_equal(_array(after,attrs['WEIGHTS_0'],'VEC4',{5126}),self.arrays['weights'][ids]))
                mapping = np.array([1,0,3,2],dtype='<u2')
                self.assertTrue(np.array_equal(_array(after,attrs['JOINTS_0'],'VEC4',{5123}),mapping[self.arrays['joints'][ids]]))
        for row in result['skin_writes']:
            for offset in range(row['byte_offset'],row['byte_offset']+row['count']*row['byte_stride'],row['byte_stride']):
                allowed[offset:offset+row['element_bytes']] = True
        self.assertEqual(after.doc,expected)
        old,new = np.frombuffer(before.binary,dtype='u1'),np.frombuffer(after.binary,dtype='u1')
        self.assertTrue(np.array_equal(old[~allowed],new[~allowed]))
        self.assertTrue(result['all_source_ids_represented']); self.assertFalse(result['motion_verified'])
        self.assertFalse(result['quality_approved']); self.assertFalse(result['body_normal_fidelity_verified'])

    def test_exact_complete_skin_and_all_other_bytes_preserved(self):
        pins = self.prepare(); original = self.source.read_bytes()
        result = self.execute(pins); self.assert_custody(result)
        self.assertEqual(self.source.read_bytes(),original)
        self.assertEqual(result['source_vertices'],27)
        self.assertEqual(result['exported_vertex_entries'],27)
        self.assertEqual(result['owners'],2)
        self.assertTrue(all(row['native_to_exported_joint_slots'] == [1,0,3,2] for row in result['primitives']))
        before,after = _GLB(self.source),_GLB(self.target)
        attrs = after.doc['meshes'][0]['primitives'][0]['attributes']
        self.assertGreater(_array(after,attrs['WEIGHTS_0'],'VEC4',{5126})[0,0],0)
        self.assertTrue(np.array_equal(_array(before,attrs['NORMAL'],'VEC3',{5126}),
                                       _array(after,attrs['NORMAL'],'VEC3',{5126})))

    def test_source_ids_resolve_coincident_points_and_remove_synthetic_root_gap(self):
        result = self.execute(); self.assert_custody(result)
        before,after = _GLB(self.source),_GLB(self.target)
        body = before.doc['meshes'][0]['primitives'][0]['attributes']
        fringe = before.doc['meshes'][1]['primitives'][0]['attributes']
        ids = _array(before,fringe[ATTRIBUTE],'SCALAR',{5126}).ravel().astype(int)
        root = int(np.flatnonzero(ids == 3)[0])
        self.assertTrue(np.array_equal(_array(after,body['POSITION'],'VEC3',{5126})[0],
                                       _array(after,fringe['POSITION'],'VEC3',{5126})[root]))
        translations = np.array([[0,0,0],[1,0,0],[0,0,0],[0,1,0]],float)
        def offset(model, attrs, index):
            joints = _array(model,attrs['JOINTS_0'],'VEC4',{5123})[index]
            weights = _array(model,attrs['WEIGHTS_0'],'VEC4',{5126})[index]
            return (translations[joints]*weights[:,None]).sum(axis=0)
        fringe_only_gap = np.linalg.norm(offset(before,body,0)-offset(after,fringe,root))
        full_gap = np.linalg.norm(offset(after,body,0)-offset(after,fringe,root))
        self.assertGreater(fringe_only_gap,5e-5); self.assertEqual(full_gap,0)

    def test_explicit_guard_hashes_new_target_and_deterministic_bytes(self):
        pins = self.prepare()
        with self.assertRaisesRegex(ValueError,'disabled'):
            restore_source_skin(self.source,self.target,self.snapshot,self.report_path,expected_hashes=pins)
        with self.assertRaisesRegex(ValueError,'hash'):
            self.execute({**pins,'source':'0'*64})
        with self.assertRaisesRegex(ValueError,'hash'):
            restore_source_skin(self.source,self.target,self.snapshot,self.report_path,experimental=True)
        result = self.execute(pins); output = self.target.read_bytes()
        with self.assertRaisesRegex(ValueError,'fresh'):self.execute(pins)
        self.assertEqual(self.target.read_bytes(),output)
        other = self.root/'repeat.glb'
        repeat = restore_source_skin(self.source,other,self.snapshot,self.report_path,experimental=True,expected_hashes=pins)
        self.assertEqual(result['result_sha256'],repeat['result_sha256'])

    def test_ids_are_exact_finite_owner_bound_and_not_nearest_matches(self):
        for mutation in ('fraction','nan','other_owner','missing','one_ulp_position'):
            self.reset(); attrs = self.doc['meshes'][0]['primitives'][0]['attributes']
            if mutation == 'missing':attrs.pop(ATTRIBUTE)
            elif mutation == 'one_ulp_position':
                value = raw_array(self.doc,self.binary,attrs['POSITION'])
                value[0,0] = np.nextafter(np.float32(0),np.float32(1))
            else:
                raw_array(self.doc,self.binary,attrs[ATTRIBUTE])[0,0] = {'fraction':.5,'nan':np.nan,'other_owner':3}[mutation]
            with self.subTest(mutation=mutation):self.reject()

    def test_owner_frames_rig_names_and_triangle_material_proof(self):
        for mutation in ('matrix','boolean_frame','animated','orphan','mesh_name','bone_name','winding','material','range_bool','ignored_bone'):
            self.reset()
            if mutation == 'matrix':self.doc['nodes'][0]['translation'] = [0,0,.001]
            elif mutation == 'boolean_frame':self.doc['nodes'][0]['scale'] = [True,1,1]
            elif mutation == 'animated':self.doc['animations'][0]['channels'][0]['target']['node'] = 0
            elif mutation == 'orphan':self.doc['scenes'][0]['nodes'].remove(0)
            elif mutation == 'mesh_name':self.doc['meshes'][0]['name'] = 'unknown'
            elif mutation == 'bone_name':self.doc['nodes'][1]['name'] = 'missing'
            elif mutation == 'winding':
                index = self.doc['meshes'][0]['primitives'][0]['indices']
                value = raw_array(self.doc,self.binary,index); value[:] = value[::-1].copy()
            elif mutation == 'material':self.doc['meshes'][0]['primitives'][0]['material'] = 1
            elif mutation == 'range_bool':self.report['owners'][0]['vertex_start'] = False
            else:self.report['owners'][0]['ignored_non_skin_groups'] = ['A']
            with self.subTest(mutation=mutation):self.reject()

    def test_distinct_same_named_rigs_cannot_be_confused(self):
        self.doc['skins'].append(deepcopy(self.doc['skins'][0]))
        cloned = []
        for node in self.doc['skins'][0]['joints']:
            cloned.append(len(self.doc['nodes'])); self.doc['nodes'].append(deepcopy(self.doc['nodes'][node]))
        self.doc['skins'][1]['joints'] = cloned; self.doc['nodes'][2]['skin'] = 1
        self.reject('same named rig nodes')

    def test_shared_skin_storage_requires_identical_proven_values(self):
        body = self.doc['meshes'][0]['primitives'][0]['attributes']
        fringe = self.doc['meshes'][1]['primitives'][0]['attributes']
        self.doc['accessors'][body['WEIGHTS_0']]['bufferView'] = self.doc['accessors'][fringe['WEIGHTS_0']]['bufferView']
        result = self.execute(); self.assert_custody(result)
        self.target = self.root/'conflict.glb'
        self.arrays['weights'][0] = [.1,.1,.1,.7]
        self.reject('conflicting encoded bytes')

    def test_interleaved_non_skin_bytes_are_retained(self):
        attrs = self.doc['meshes'][0]['primitives'][0]['attributes']
        dtype = np.dtype([('position','<f4',3),('joints','<u2',4),('weights','<f4',4)])
        rows = np.empty(3,dtype=dtype)
        for name,key in (('position','POSITION'),('joints','JOINTS_0'),('weights','WEIGHTS_0')):
            rows[name] = raw_array(self.doc,self.binary,attrs[key])
        self.binary.extend(b'\0'*(-len(self.binary)%4)); start = len(self.binary)
        self.binary.extend(rows.tobytes())
        self.doc['bufferViews'].append({'buffer':0,'byteOffset':start,'byteLength':rows.nbytes,'byteStride':36})
        for key,offset in (('POSITION',0),('JOINTS_0',12),('WEIGHTS_0',20)):
            self.doc['accessors'][attrs[key]].update(bufferView=len(self.doc['bufferViews'])-1,byteOffset=offset)
        self.doc['buffers'][0]['byteLength'] = len(self.binary)
        self.assert_custody(self.execute())

    def test_non_skin_aliases_sparse_extra_sets_and_orphans_reject(self):
        for mutation in ('orphan_accessor','view','image','normal','sparse','extra_skin','stale_bounds'):
            self.reset(); attrs = self.doc['meshes'][0]['primitives'][0]['attributes']
            accessor = self.doc['accessors'][attrs['WEIGHTS_0']]; view = self.doc['bufferViews'][accessor['bufferView']]
            if mutation == 'orphan_accessor':self.doc['accessors'].append(deepcopy(accessor))
            elif mutation == 'view':self.doc['bufferViews'].append(deepcopy(view))
            elif mutation == 'image':self.doc['images'].append({'bufferView':accessor['bufferView'],'mimeType':'image/png'})
            elif mutation == 'normal':attrs['NORMAL'] = attrs['WEIGHTS_0']
            elif mutation == 'sparse':accessor['sparse'] = {}
            elif mutation == 'extra_skin':attrs['WEIGHTS_1'] = attrs['WEIGHTS_0']
            else:accessor['max'] = [1,1,1,1]
            with self.subTest(mutation=mutation):self.reject()

    def test_uv_seam_duplicates_keep_source_id_and_triangle_multiplicity(self):
        primitive = self.doc['meshes'][0]['primitives'][0]; attrs = primitive['attributes']
        for key,aid in list(attrs.items()):
            a = self.doc['accessors'][aid]
            values = raw_array(self.doc,self.binary,aid).copy()
            values = np.concatenate((values,values[:1]))
            if key == 'TEXCOORD_0':values[-1] = [.4,.8]
            attrs[key] = add_array(self.doc,self.binary,values,a['type'])
        primitive['indices'] = add_array(self.doc,self.binary,np.array([0,1,2,3,1,2],dtype='<u2')[:,None],'SCALAR')
        self.arrays['triangles'] = np.insert(self.arrays['triangles'],1,[0,1,2],axis=0).astype('<u4')
        self.arrays['triangle_materials'] = np.insert(self.arrays['triangle_materials'],1,0).astype('<u2')
        self.report['triangle_count'] += 1
        self.report['owners'][0]['triangle_count'] = 2; self.report['owners'][1]['triangle_start'] = 2
        result = self.execute(); self.assert_custody(result)
        self.assertEqual(result['exported_vertex_entries'],result['source_vertices']+1)

    def test_snapshot_schema_weights_and_bounds_are_strict(self):
        for mutation in ('object','unknown','nan','weight_sum','duplicate_bones','loose','shape','size'):
            self.reset()
            if mutation == 'object':self.arrays['positions'] = self.arrays['positions'].astype(object)
            elif mutation == 'unknown':self.arrays['unknown'] = np.zeros(1,dtype='<f4')
            elif mutation == 'nan':self.arrays['positions'][0,0] = np.nan
            elif mutation == 'weight_sum':self.arrays['weights'][0,0] = .8
            elif mutation == 'duplicate_bones':self.arrays['joints'][0,0] = self.arrays['joints'][0,1]
            elif mutation == 'loose':self.arrays['triangles'][-1,-1] = 3
            elif mutation == 'shape':self.arrays['positions'] = np.zeros((100001,3),dtype='<f4')
            else:self.report['vertex_count'] = 100001
            with self.subTest(mutation=mutation):self.reject()

    def test_npz_decode_declared_budget_and_duplicate_entries_reject_before_load(self):
        pins = self.prepare()
        with zipfile.ZipFile(self.snapshot) as archive:entries = {i.filename:archive.read(i) for i in archive.infolist()}
        entries['positions.npy'] = b'\0'*(8*1024**2+1)
        with zipfile.ZipFile(self.snapshot,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for name,value in entries.items():archive.writestr(name,value)
        self.report.update(npz_sha256=sha256(self.snapshot),npz_bytes=self.snapshot.stat().st_size)
        self.report_path.write_text(json.dumps(self.report))
        pins.update(snapshot=sha256(self.snapshot),snapshot_report=sha256(self.report_path))
        with patch('avatar_assets.source_skin_fidelity.np.load',side_effect=AssertionError('Oversized archive decoded')):
            with self.assertRaisesRegex(ValueError,'decode exceeds'):self.execute(pins)
        self.assertFalse(self.target.exists())
        pins = self.prepare()
        with zipfile.ZipFile(self.snapshot,'a') as archive:
            with self.assertWarns(UserWarning):archive.writestr('positions.npy',b'duplicate')
        self.report.update(npz_sha256=sha256(self.snapshot),npz_bytes=self.snapshot.stat().st_size)
        self.report_path.write_text(json.dumps(self.report))
        pins.update(snapshot=sha256(self.snapshot),snapshot_report=sha256(self.report_path))
        with self.assertRaisesRegex(ValueError,'duplicate'):self.execute(pins)

    def test_oversized_or_symlink_input_is_not_read(self):
        pins = self.prepare()
        self.source = self.root/'too-large.glb'
        with self.source.open('wb') as handle:handle.truncate(64*1024**2+1)
        with self.assertRaisesRegex(ValueError,'too large'):self.execute(pins)
        self.source = self.root/'linked.glb'; self.source.symlink_to(self.root/'source.glb')
        with self.assertRaisesRegex(ValueError,'regular local'):self.execute(pins)

    def test_maximum_vertex_and_triangle_scale_is_bounded(self):
        n = 100000; angle = np.arange(n-1,dtype=float)*(2*np.pi/(n-1))
        positions = np.concatenate((np.zeros((1,3)),np.column_stack((np.cos(angle),np.sin(angle),np.zeros(n-1))))).astype('<f4')
        triangles = np.column_stack((np.zeros(n-2,dtype='<u4'),np.arange(1,n-1,dtype='<u4'),np.arange(2,n,dtype='<u4')))
        weights = np.tile(self.arrays['weights'][0],(n,1)); joints = np.tile([0,1,2,3],(n,1)).astype('<u2')
        self.arrays = {'positions':positions,'joints':joints,'weights':weights,'triangles':triangles,
                       'triangle_materials':np.zeros(len(triangles),dtype='<u2')}
        self.report.update(vertex_count=n,triangle_count=len(triangles))
        self.report['owners'] = [self.report['owners'][0]]
        self.report['owners'][0].update(vertex_count=n,triangle_count=len(triangles))
        self.doc['meshes'] = [self.doc['meshes'][0]]
        self.doc['nodes'][2].pop('mesh'); self.doc['nodes'][2].pop('skin')
        exported = positions[:,[0,2,1]].copy(); exported[:,2]*=-1
        pruned = weights.copy(); pruned[pruned <= .0001] = 0; pruned /= pruned.sum(axis=1)[:,None]
        values = {'POSITION':(exported,'VEC3'),'NORMAL':(np.tile([0.,1,0],(n,1)).astype('<f4'),'VEC3'),
            'TEXCOORD_0':(np.zeros((n,2),dtype='<f4'),'VEC2'),'TANGENT':(np.tile([1.,0,0,1],(n,1)).astype('<f4'),'VEC4'),
            'JOINTS_0':(np.array([1,0,3,2],dtype='<u2')[joints],'VEC4'),'WEIGHTS_0':(pruned,'VEC4'),
            ATTRIBUTE:(np.arange(n,dtype='<f4')[:,None],'SCALAR')}
        primitive = self.doc['meshes'][0]['primitives'][0]
        primitive['attributes'] = {key:add_array(self.doc,self.binary,value,kind) for key,(value,kind) in values.items()}
        primitive['indices'] = add_array(self.doc,self.binary,triangles.ravel()[:,None],'SCALAR')
        pins = self.prepare(); started = time.monotonic(); result = self.execute(pins)
        self.assertEqual(result['source_vertices'],100000); self.assertEqual(result['triangles'],99998)
        self.assertLess(time.monotonic()-started,30); self.assertLessEqual(result['sampled_peak_rss_bytes'],512*1024**2)
        if os.environ.get('SOURCE_SKIN_TEST_RECEIPT'):
            with Path(os.environ['SOURCE_SKIN_TEST_RECEIPT']).open('x') as handle:
                json.dump({'fixture':'analytic 100000 source/exported vertices, 99998 indexed triangles',
                    'native_rendered':False,'input_sizes':{key:path.stat().st_size for key,path in
                        [('source',self.source),('snapshot',self.snapshot),('snapshot_report',self.report_path)]},
                    'restore_report':result},handle,indent=2)


if __name__ == '__main__':unittest.main()
