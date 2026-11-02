"""Analytic native-adapter fixtures; no Blender or generated-avatar claims."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

import numpy as np


PATH = Path(__file__).resolve().parents[1]/'scripts/source_skin_snapshot.py'
SPEC = importlib.util.spec_from_file_location('source_skin_snapshot_fixture_module', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Collection(list):
    def foreach_get(self, field, target):
        target[:] = np.asarray([getattr(value, field) for value in self]).reshape(-1)


class AttributeData:
    def __init__(self, count):
        self.values = np.zeros(count, dtype=np.int32)

    def foreach_set(self, field, value):
        assert field == 'value'
        self.values[:] = value

    def foreach_get(self, field, target):
        assert field == 'value'
        target[:] = self.values


class Attributes(list):
    def __init__(self, count):
        super().__init__()
        self.count = count
        self.fail_new = False

    def new(self, *, name, type, domain):
        if self.fail_new:
            raise RuntimeError('injected native attribute allocation failure')
        attr = NS(name=name, data_type=type, domain=domain, data=AttributeData(self.count))
        self.append(attr)
        return attr


def rig(count=4):
    return NS(name='Rig', type='ARMATURE', mode='OBJECT', matrix_world=np.eye(4), matrix_basis=np.eye(4),
              data=NS(name='RigData', pose_position='REST', bones=[NS(name='Bone'+str(i)) for i in range(count)]))


def mesh_object(name, armature, *, members=None, offset=0):
    points=np.array([[offset,0,0],[offset+1,0,0],[offset,1,0]],dtype=np.float32)
    memberships = members or [[(0,1.)] for _ in range(3)]
    vertices=Collection(NS(index=i,co=point.copy(),groups=[NS(group=g,weight=w) for g,w in memberships[i]])
                        for i,point in enumerate(points))
    data=NS(name=name+'Mesh', vertices=vertices, polygons=Collection([NS(loop_total=3)]), loops=[None]*3,
            loop_triangles=Collection([NS(vertices=(0,1,2),material_index=0)]), attributes=Attributes(3))
    data.calc_loop_triangles=lambda:None
    modifier=NS(type='ARMATURE',object=armature,show_viewport=True,show_render=True,use_vertex_groups=True,
                use_bone_envelopes=False,use_deform_preserve_volume=False)
    return NS(name=name,type='MESH',mode='OBJECT',data=data,modifiers=[modifier],matrix_world=np.eye(4),
        matrix_basis=np.eye(4),material_slots=[NS(material=NS(name=name+'Material'))],
        vertex_groups=[NS(index=i,name=b.name) for i,b in enumerate(armature.data.bones)])


class SourceSkinSnapshotTests(unittest.TestCase):
    def assert_rejected_unchanged(self, objects, armature, message=None):
        before=[[a.name for a in obj.data.attributes] for obj in objects]
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, message or 'Source skin snapshot'):
                MODULE.snapshot_skin_sources(objects,armature,folder)
            self.assertEqual(list(Path(folder).iterdir()),[])
        self.assertEqual(before,[[a.name for a in obj.data.attributes] for obj in objects])

    def test_exact_tiny_weights_global_ids_and_content_hashes(self):
        armature=rig()
        members=[[(0,float(np.float32(.99985))),(1,float(np.float32(.00005))),
                  (2,float(np.float32(.0001)))],[(1,1.)],[(2,.75),(3,.25)]]
        body=mesh_object('Body',armature,members=members)
        fringe=mesh_object('Fringe',armature,offset=2)
        body.data.attributes.append(NS(name='custom_normal',data_type='FLOAT_VECTOR',domain='POINT'))
        original_points=np.array([v.co for v in body.data.vertices])
        original_weights=[[g.weight for g in v.groups] for v in body.data.vertices]
        with tempfile.TemporaryDirectory() as folder:
            report=MODULE.snapshot_skin_sources([body,fringe],armature,folder)
            path=Path(folder)
            saved=json.loads((path/'source-skin.json').read_text())
            self.assertEqual(report['report_sha256'],hashlib.sha256((path/'source-skin.json').read_bytes()).hexdigest())
            self.assertNotIn('report_sha256',saved)
            self.assertEqual(report['vertex_count'],6)
            self.assertEqual(report['triangle_count'],2)
            self.assertEqual([r['vertex_start'] for r in report['owners']],[0,3])
            self.assertEqual([r['triangle_start'] for r in report['owners']],[0,1])
            self.assertEqual(report['npz_sha256'],hashlib.sha256((path/'source-skin.npz').read_bytes()).hexdigest())
            with np.load(path/'source-skin.npz',allow_pickle=False) as arrays:
                self.assertEqual(set(arrays.files),{'positions','joints','weights','triangles','triangle_materials'})
                for key in arrays.files:
                    value=arrays[key];schema=report['arrays'][key]
                    self.assertEqual(value.dtype.str,schema['dtype'])
                    self.assertEqual(list(value.shape),schema['shape'])
                    self.assertEqual(hashlib.sha256(value.tobytes(order='C')).hexdigest(),schema['sha256'])
                np.testing.assert_array_equal(arrays['positions'][:3],original_points)
                np.testing.assert_array_equal(arrays['triangles'],[[0,1,2],[3,4,5]])
                np.testing.assert_array_equal(arrays['joints'][0],[0,2,1,0])
                np.testing.assert_array_equal(arrays['weights'][0],np.array([.99985,.0001,.00005,0],dtype=np.float32))
                self.assertEqual(np.count_nonzero(arrays['weights'][0]),3)
                self.assertFalse(report['normal_snapshot_included'])
            self.assertEqual(set(p.name for p in path.iterdir()),{'source-skin.npz','source-skin.json'})
        np.testing.assert_array_equal(body.data.attributes[-1].data.values,[0,1,2])
        np.testing.assert_array_equal(fringe.data.attributes[-1].data.values,[3,4,5])
        np.testing.assert_array_equal([v.co for v in body.data.vertices],original_points)
        self.assertEqual([[g.weight for g in v.groups] for v in body.data.vertices],original_weights)
        self.assertEqual(armature.data.pose_position,'REST')

    def test_reproducible_npz_and_json(self):
        hashes=[]
        for _ in range(2):
            armature=rig();body=mesh_object('Body',armature)
            with tempfile.TemporaryDirectory() as folder:
                report=MODULE.snapshot_skin_sources([body],armature,folder)
                hashes.append((report['npz_sha256'],report['report_sha256']))
        self.assertEqual(hashes[0],hashes[1])

    def test_normalization_is_checked_not_changed(self):
        armature=rig();body=mesh_object('Body',armature,members=[[(0,.5),(1,.5000005)],[(0,1.)],[(0,1.)]])
        with tempfile.TemporaryDirectory() as folder:
            report=MODULE.snapshot_skin_sources([body],armature,folder)
            with np.load(Path(folder)/report['npz_file'],allow_pickle=False) as arrays:
                self.assertEqual(float(arrays['weights'][0].sum()),float(np.float32(.5)+np.float32(.5000005)))
                self.assertGreater(report['maximum_source_weight_sum_error'],0)

    def test_invalid_weights_fail_before_attributes_or_files(self):
        cases=[[(0,float('nan'))],[(0,float('inf'))],[(0,True)],[(0,-.1)],[(0,1.1)],[(0,0.)],
               [(0,.8)],[(0,.5),(0,.5)],[(99,1.)],[(i,.2) for i in range(5)],[(0,1.),(1,1e-50)]]
        for case in cases:
            with self.subTest(case=case):
                armature=rig(5);body=mesh_object('Body',armature,members=[case,[(0,1.)],[(0,1.)]])
                self.assert_rejected_unchanged([body],armature)

    def test_non_skin_mask_groups_are_not_bone_influences(self):
        armature=rig();body=mesh_object('Body',armature)
        body.vertex_groups.append(NS(index=4,name='DensityMask'))
        for vertex in body.data.vertices:vertex.groups.append(NS(group=4,weight=1.))
        with tempfile.TemporaryDirectory() as folder:
            report=MODULE.snapshot_skin_sources([body],armature,folder)
            self.assertEqual(report['owners'][0]['ignored_non_skin_groups'],['DensityMask'])
            with np.load(Path(folder)/report['npz_file'],allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays['weights'],[[1,0,0,0]]*3)

    def test_identity_rest_and_modifier_contract(self):
        mutations=[lambda o,r:setattr(r.data,'pose_position','POSE'),
                   lambda o,r:r.matrix_world.__setitem__((0,3),.01),
                   lambda o,r:o.matrix_basis.__setitem__((1,1),2),
                   lambda o,r:setattr(o.modifiers[0],'object',rig()),
                   lambda o,r:setattr(o.modifiers[0],'use_bone_envelopes',True),
                   lambda o,r:setattr(o.modifiers[0],'use_deform_preserve_volume',True),
                   lambda o,r:setattr(o.modifiers[0],'show_render',False),
                   lambda o,r:o.modifiers.append(NS(type='SUBSURF'))]
        for index,mutation in enumerate(mutations):
            with self.subTest(index=index):
                armature=rig();body=mesh_object('Body',armature);mutation(body,armature)
                self.assert_rejected_unchanged([body],armature)

    def test_underscore_collision_and_shared_ownership_reject(self):
        for name in ('_AUTHOR_VERTEX_ID','_author_vertex_id','_PRIVATE'):
            with self.subTest(name=name):
                armature=rig();body=mesh_object('Body',armature)
                body.data.attributes.append(NS(name=name))
                self.assert_rejected_unchanged([body],armature,'underscore')
        for share in ('object','mesh','name'):
            with self.subTest(share=share):
                armature=rig();body=mesh_object('Body',armature);other=mesh_object('Other',armature)
                if share=='object':other=body
                elif share=='mesh':other.data=body.data
                else:other.name=body.name
                self.assert_rejected_unchanged([body,other],armature,'duplicate')

    def test_nonfinite_geometry_material_and_topology_fail_early(self):
        mutations=[lambda o:o.data.vertices[0].co.__setitem__(0,float('nan')),
                   lambda o:setattr(o.material_slots[0],'material',None),
                   lambda o:setattr(o.data.loop_triangles[0],'material_index',2),
                   lambda o:setattr(o.data.loop_triangles[0],'vertices',(0,0,2)),
                   lambda o:setattr(o.data.loop_triangles[0],'vertices',(0,1,3))]
        for index,mutation in enumerate(mutations):
            with self.subTest(index=index):
                armature=rig();body=mesh_object('Body',armature);mutation(body)
                self.assert_rejected_unchanged([body],armature)
        armature=rig();body=mesh_object('Body',armature)
        body.data.vertices.append(NS(index=3,co=np.array([1.,1.,1.]),groups=[NS(group=0,weight=1.)]))
        self.assert_rejected_unchanged([body],armature,'unreferenced')

    def test_invalid_and_oversized_owner_bone_counts(self):
        armature=rig(65);body=mesh_object('Body',armature)
        self.assert_rejected_unchanged([body],armature,'64')
        armature=rig();body=mesh_object('Body',armature)
        self.assert_rejected_unchanged([],armature)
        self.assert_rejected_unchanged([body]*33,armature)
        class OversizedVertices:
            def __len__(self):return 100001
            def __iter__(self):raise AssertionError('oversized data must not be traversed')
        body.data.vertices=OversizedVertices()
        self.assert_rejected_unchanged([body],armature,'oversized')

    def test_aggregate_bound_precedes_vertex_decode(self):
        class Vertices:
            def __len__(self):return 50001
            def foreach_get(self,*args):raise AssertionError('aggregate bound must fail before decode')
        armature=rig();objects=[mesh_object('Body',armature),mesh_object('Fringe',armature)]
        for obj in objects:obj.data.vertices=Vertices()
        self.assert_rejected_unchanged(objects,armature,'aggregate')

    def test_existing_output_is_preserved_without_attribute_mutation(self):
        for name in ('source-skin.npz','source-skin.json','source-skin.npz.tmp','source-skin.json.tmp'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as folder:
                armature=rig();body=mesh_object('Body',armature);path=Path(folder)/name;path.write_bytes(b'prior evidence')
                with self.assertRaisesRegex(ValueError,'already exists'):
                    MODULE.snapshot_skin_sources([body],armature,folder)
                self.assertEqual(path.read_bytes(),b'prior evidence')
                self.assertEqual(body.data.attributes,[])

    def test_partial_attribute_allocation_rolls_back_only_new_layers(self):
        armature=rig();body=mesh_object('Body',armature);fringe=mesh_object('Fringe',armature)
        normal=NS(name='custom_normal',data_type='FLOAT_VECTOR',domain='POINT');body.data.attributes.append(normal)
        fringe.data.attributes.fail_new=True
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError,'injected'):
                MODULE.snapshot_skin_sources([body,fringe],armature,folder)
            self.assertEqual(body.data.attributes,[normal]);self.assertEqual(fringe.data.attributes,[])
            self.assertFalse((Path(folder)/'source-skin.json').exists())
            self.assertTrue((Path(folder)/'source-skin.npz.tmp').exists())


if __name__=='__main__':unittest.main()
