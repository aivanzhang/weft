"""Analytic exact-correspondence restoration; no native or avatar-quality claims."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from avatar_assets.fringe_export import _array
from avatar_assets.fringe_fidelity import restore_authored_fringe
from avatar_assets.safety import sha256
from avatar_assets.three_d import _GLB, validate_render_attributes
from tests.test_fringe_export import fixture
from tests.test_three_d import glb_bytes


def authored_fixture(count=2):
    doc,binary = fixture(count*12,0)
    primitive = doc['meshes'][1]['primitives'][0]
    attrs = primitive['attributes']
    attrs.pop('TANGENT'); attrs.pop('COLOR_0')
    doc['nodes'][1]['name'] = 'A'
    for name in ('B','C','D'):
        doc['nodes'].append({'name':name})
        doc['scenes'][0]['nodes'].append(len(doc['nodes'])-1)
    doc['skins'][0]['joints'] = [1,3,4,5]
    identity = np.tile(np.eye(4,dtype='<f4').reshape(1,16),(4,1))
    binary.extend(b'\0'*(-len(binary)%4));offset=len(binary);binary.extend(identity.tobytes())
    doc['bufferViews'].append({'buffer':0,'byteOffset':offset,'byteLength':identity.nbytes})
    doc['accessors'].append({'bufferView':len(doc['bufferViews'])-1,'componentType':5126,'count':4,'type':'MAT4'})
    doc['skins'][0]['inverseBindMatrices'] = len(doc['accessors'])-1
    x = np.arange(count,dtype=np.float32)*.03
    points = np.tile(np.array([[0,0,0],[.012,0,0],[0,0,.008],[.012,0,.008]],dtype='<f4'),(count,1))
    points[:,0] += np.repeat(x,4)
    points[:,1] += np.repeat(np.sin(x)*.01,4)
    normal = np.array([[.2,.6,.77],[-.1,.65,.75]],dtype=np.float64)
    normal /= np.linalg.norm(normal,axis=1)[:,None]
    normals = np.tile(normal,(count*2,1)).astype('<f4')
    joints = np.tile([0,1,2,3],(count*4,1)).astype('<u2')
    weights = np.tile([.00007,.009,.00006,.99087],(count*4,1)).astype('<f4')
    uv = np.empty((count,4,2),dtype='<f4')
    uv[:,[0,2],0] = ((np.arange(count)+.2)/count)[:,None]
    uv[:,[1,3],0] = ((np.arange(count)+.8)/count)[:,None]
    uv[:,:2,1] = .985; uv[:,2:,1] = .015
    front = (np.arange(count)[:,None,None]*4+np.array([[0,1,3],[0,3,2]])).reshape(-1,3)
    triangles = np.concatenate((front,front[:,::-1])).astype('<i4')
    native_map = triangles.ravel()
    # Exported corner order differs, while actual oriented triangles remain identical.
    permutation = np.arange(count*12)[::-1]
    mapping = native_map[permutation]
    positions = points[:,[0,2,1]].copy();positions[:,2]*=-1
    output_uv = uv.reshape(-1,2).copy(); output_uv[:,1] = np.float32(1)-np.float32(np.float32(1)-output_uv[:,1])
    output_normals = normals[:,[0,2,1]].copy();output_normals[:,2]*=-1
    wrong_normals = output_normals[mapping].copy();wrong_normals[:,0]+=.01
    wrong_normals /= np.linalg.norm(wrong_normals,axis=1)[:,None]
    pruned = weights[mapping].copy();pruned[pruned <= .0001] = 0;pruned/=pruned.sum(axis=1)[:,None]
    native_to_exported = np.array([1,0,3,2],dtype='<u2')
    values = {'POSITION':positions[mapping],'NORMAL':wrong_normals,'TEXCOORD_0':output_uv[mapping],
              'JOINTS_0':native_to_exported[joints[mapping]],'WEIGHTS_0':pruned}
    for key,value in values.items():
        a=doc['accessors'][attrs[key]];v=doc['bufferViews'][a['bufferView']]
        binary[v['byteOffset']:v['byteOffset']+v['byteLength']] = value.tobytes()
    a=doc['accessors'][primitive['indices']];v=doc['bufferViews'][a['bufferView']]
    binary[v['byteOffset']:v['byteOffset']+v['byteLength']] = np.argsort(permutation).astype('<u2').tobytes()
    doc['buffers'][0]['byteLength'] = len(binary)
    image=doc['images'][0];v=doc['bufferViews'][image['bufferView']]
    report={'cards':count,'vertices':count*12,'geometry_profile':'edge-fringe-v1',
            'geometry':{'profile':'edge-fringe-source-surface-v1'},'bone_order':['B','A','D','C'],
            'geometry_arrays_npz':'helper.npz','material':{'material':'SourceColoredFringe',
                'sha256':hashlib.sha256(binary[v['byteOffset']:v['byteOffset']+v['byteLength']]).hexdigest()}}
    helper={'positions':points,'normals':normals,'triangles':triangles,'joints':joints,'weights':weights,
        'uv_gltf':uv.reshape(-1,2),'root_vertex_ids':np.arange(count*2,dtype='<i4').reshape(count,2),
        'root_face_ids':np.arange(count,dtype='<i8'),'root_corner_ids':np.tile([0,1],(count,1)).astype('<i8'),
        'root_source_uv':np.tile([[[-.1,.2],[1.1,.2]]],(count,1,1)).astype('<f8'),
        'material_ids':np.zeros(count,dtype='<i4'),'native_vertex_to_helper_vertex':native_map,
        'destination_for_source_fin':np.arange(count,dtype='<i4'),'source_fin_at_destination':np.arange(count,dtype='<i4')}
    return doc,binary,helper,report,mapping


class FringeFidelityTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name).resolve()
        self.source,self.target,self.npz,self.report_path=[self.root/name for name in ('source.glb','result.glb','helper.npz','report.json')]
        self.doc,self.binary,self.helper,self.report,self.mapping=authored_fixture()

    def prepare(self):
        self.source.write_bytes(glb_bytes(self.doc,self.binary))
        np.savez_compressed(self.npz,**self.helper)
        self.report_path.write_text(json.dumps(self.report))
        return {'source':sha256(self.source),'correspondence':sha256(self.npz),'native_report':sha256(self.report_path)}

    def execute(self, hashes=None, **kwargs):
        return restore_authored_fringe(self.source,self.target,self.npz,self.report_path,
                experimental=True,expected_hashes=hashes or self.prepare(),**kwargs)

    def assert_rejected(self, fragment=None):
        self.prepare()
        with self.assertRaisesRegex(ValueError,fragment or '.'):
            self.execute()
        self.assertFalse(self.target.exists())

    def test_exact_restoration_not_rounding_or_pruning_and_all_other_bytes_preserved(self):
        pins=self.prepare();original=self.source.read_bytes();result=self.execute(pins)
        after=_GLB(self.target);before=_GLB(self.source)
        self.assertEqual(after.doc,before.doc)
        attrs=after.doc['meshes'][1]['primitives'][0]['attributes']
        expected=self.helper['normals'][:,[0,2,1]].copy();expected[:,2]*=-1
        self.assertTrue(np.array_equal(_array(after,attrs['NORMAL'],'VEC3',{5126}),expected[self.mapping]))
        weights=_array(after,attrs['WEIGHTS_0'],'VEC4',{5126})
        self.assertTrue(np.array_equal(weights,self.helper['weights'][self.mapping]))
        self.assertTrue(np.all(weights[:,0] > 0))
        expected_joints=np.array([1,0,3,2],dtype='<u2')[self.helper['joints'][self.mapping]]
        self.assertTrue(np.array_equal(_array(after,attrs['JOINTS_0'],'VEC4',{5123}),expected_joints))
        cursor=0
        for row in sorted(result['changed_attributes'],key=lambda r:r['binary_offset']):
            a,b=row['binary_offset'],row['binary_offset']+row['bytes']
            self.assertEqual(after.binary[cursor:a],before.binary[cursor:a]);cursor=b
        self.assertEqual(after.binary[cursor:],before.binary[cursor:])
        self.assertEqual(self.source.read_bytes(),original)
        self.assertEqual(validate_render_attributes(self.target)['status'],'validated_render_attributes')
        self.assertEqual(result['native_to_exported_joint_slots'],[1,0,3,2])
        self.assertFalse(result['motion_verified']);self.assertFalse(result['quality_approved'])

    def test_default_guard_and_exact_hashes_and_new_target_only(self):
        pins=self.prepare()
        with self.assertRaisesRegex(ValueError,'disabled'):
            restore_authored_fringe(self.source,self.target,self.npz,self.report_path,expected_hashes=pins)
        with self.assertRaisesRegex(ValueError,'hash'):
            self.execute({**pins,'source':'0'*64})
        with self.assertRaisesRegex(ValueError,'hash'):
            restore_authored_fringe(self.source,self.target,self.npz,self.report_path,experimental=True)
        self.target.write_bytes(b'prior output')
        with self.assertRaisesRegex(ValueError,'fresh'):
            self.execute(pins)
        self.assertEqual(self.target.read_bytes(),b'prior output')

    def test_repeat_derivative_bytes_are_deterministic_and_atlas_binding_is_required(self):
        pins=self.prepare();first=self.execute(pins)
        other=self.root/'result-second.glb'
        second=restore_authored_fringe(self.source,other,self.npz,self.report_path,
                                     experimental=True,expected_hashes=pins)
        self.assertEqual(first['result_sha256'],second['result_sha256'])
        self.assertEqual(self.target.read_bytes(),other.read_bytes())
        self.report['material']['sha256']='0'*64
        self.target=self.root/'invalid-atlas.glb'
        self.assert_rejected('atlas hash')

    def test_no_nearest_match_or_ambiguous_helper_key(self):
        attrs=self.doc['meshes'][1]['primitives'][0]['attributes'];a=self.doc['accessors'][attrs['POSITION']]
        offset=self.doc['bufferViews'][a['bufferView']]['byteOffset']
        value=struct.unpack_from('<f',self.binary,offset)[0]
        struct.pack_into('<f',self.binary,offset,float(np.nextafter(np.float32(value),np.float32(np.inf))))
        self.assert_rejected('exact authored key')
        self.doc,self.binary,self.helper,self.report,self.mapping=authored_fixture()
        self.helper['positions'][1]=self.helper['positions'][0];self.helper['uv_gltf'][1]=self.helper['uv_gltf'][0]
        self.assert_rejected('Ambiguous')

    def test_winding_and_multiplicity_are_not_position_only_proof(self):
        a=self.doc['accessors'][self.doc['meshes'][1]['primitives'][0]['indices']]
        offset=self.doc['bufferViews'][a['bufferView']]['byteOffset']
        values=struct.unpack_from('<3H',self.binary,offset)
        struct.pack_into('<3H',self.binary,offset,*reversed(values))
        self.assert_rejected('winding')

    def test_transformed_animated_or_orphan_mesh_and_bad_bone_mapping_refused(self):
        for change in ('transform','boolean','animated','orphan','bone'):
            self.doc,self.binary,self.helper,self.report,self.mapping=authored_fixture()
            if change=='transform':self.doc['nodes'][2]['translation']=[0,.001,0]
            elif change=='boolean':self.doc['nodes'][2]['scale']=[True,1,1]
            elif change=='animated':self.doc['animations'][0]['channels'][0]['target']['node']=2
            elif change=='orphan':self.doc['scenes'][0]['nodes'].remove(2)
            else:self.report['bone_order'][0]='missing'
            with self.subTest(change=change):self.assert_rejected()

    def test_shared_storage_extra_skin_sets_and_unused_tangents_refused(self):
        for change in ('accessor','overlap','image','interleaved','extra_skin','tangent','orphan_accessor'):
            self.doc,self.binary,self.helper,self.report,self.mapping=authored_fixture()
            attrs=self.doc['meshes'][1]['primitives'][0]['attributes'];a=self.doc['accessors'][attrs['NORMAL']];v=self.doc['bufferViews'][a['bufferView']]
            if change=='accessor':self.doc['accessors'].append(deepcopy(a))
            elif change=='overlap':self.doc['bufferViews'].append(deepcopy(v))
            elif change=='image':self.doc['images'].append({'bufferView':a['bufferView'],'mimeType':'image/png'})
            elif change=='interleaved':v['byteStride']=16
            elif change=='extra_skin':attrs['JOINTS_1']=attrs['JOINTS_0']
            elif change=='tangent':attrs['TANGENT']=self.doc['meshes'][0]['primitives'][0]['attributes']['TANGENT']
            else:self.doc['meshes'][0]['primitives'][0]['attributes']['NORMAL']=attrs['NORMAL']
            with self.subTest(change=change):self.assert_rejected()

    def test_npz_object_unknown_nonfinite_oversized_and_duplicate_schema_refused(self):
        for change in ('object','unknown','nan','bad_shape'):
            self.doc,self.binary,self.helper,self.report,self.mapping=authored_fixture()
            if change=='object':self.helper['positions']=self.helper['positions'].astype(object)
            elif change=='unknown':self.helper['extra']=np.zeros(1)
            elif change=='nan':self.helper['normals'][0,0]=np.nan
            else:self.helper['positions']=np.zeros((100001,3),dtype='<f4')
            with self.subTest(change=change):self.assert_rejected()
        self.doc,self.binary,self.helper,self.report,self.mapping=authored_fixture();pins=self.prepare()
        with zipfile.ZipFile(self.npz,'a') as z:
            with self.assertWarns(UserWarning):z.writestr('positions.npy',b'duplicate')
        pins['correspondence']=sha256(self.npz)
        with self.assertRaisesRegex(ValueError,'duplicate'):self.execute(pins)
        self.prepare()
        with self.source.open('wb') as h:h.truncate(64*1024**2+1)
        with self.assertRaisesRegex(ValueError,'too large'):self.execute({'source':'0'*64,'correspondence':sha256(self.npz),'native_report':sha256(self.report_path)})

    def test_declared_decoded_zip_bytes_rejected_before_numpy_load(self):
        pins=self.prepare()
        with zipfile.ZipFile(self.npz) as archive:
            entries={row.filename:archive.read(row) for row in archive.infolist()}
        entries['positions.npy']=b'\0'*(8*1024**2+1)
        with zipfile.ZipFile(self.npz,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for name,value in entries.items():archive.writestr(name,value)
        pins['correspondence']=sha256(self.npz)
        with patch('avatar_assets.fringe_fidelity.np.load',side_effect=AssertionError('Decoded oversized archive')):
            with self.assertRaisesRegex(ValueError,'decoded bytes'):self.execute(pins)
        self.assertFalse(self.target.exists())

    def test_maximum_fin_fixture_is_bounded_and_reports_actual_resources(self):
        self.doc,self.binary,self.helper,self.report,self.mapping=authored_fixture(4800)
        started=time.monotonic();result=self.execute()
        self.assertEqual(result['exported_corners'],57600)
        self.assertLess(time.monotonic()-started,30)
        self.assertLessEqual(result['sampled_peak_rss_bytes'],512*1024**2)
        print(json.dumps({'analytic_max_fins':4800,'elapsed_seconds':result['elapsed_seconds'],
                          'sampled_peak_rss_bytes':result['sampled_peak_rss_bytes']}))


if __name__ == '__main__':
    unittest.main()
