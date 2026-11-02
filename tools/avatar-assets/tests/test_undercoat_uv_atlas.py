"""Analytic normal-UV planning fixtures; not actual fur appearance evidence."""
import unittest
from unittest.mock import patch

import numpy as np

from scripts import undercoat_uv_atlas as atlas


def fixture():
    square=np.array([[0,0,0],[.02,0,0],[.02,.02,0],[0,.02,0]],dtype=np.float32)
    points=np.concatenate((square,square+[.03,0,0],square+[.03,0,.02])).astype(np.float32)
    faces=np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],[8,9,10],[8,10,11]],dtype=np.int32)
    uv=np.tile(np.array([[[0,0],[1,0],[1,1]],[[0,0],[1,1],[0,1]]],dtype=np.float32),(3,1,1))
    return points,faces,uv,np.ones(6,dtype=bool),np.array([0,0,1,1,1,1],dtype=np.int16)


class NormalAtlasTests(unittest.TestCase):
    def test_only_measured_alias_region_receives_uv1_and_original_arrays_stay_exact(self):
        args=fixture();before=[value.copy() for value in args]
        report,payloads=atlas.plan_normal_atlases(*args,source_export_vertices=12,resolution=1024)
        self.assertEqual(report['atlas_regions'],[1])
        self.assertEqual(report['frame_policy'],atlas.NORMAL_UV1_FRAME_POLICY)
        self.assertEqual(payloads[1]['report']['frame_policy'],atlas.NORMAL_UV1_FRAME_POLICY)
        self.assertEqual(report['regions'][0]['method'],'keep_uv0')
        self.assertEqual(report['regions'][0]['conflicting_surface_samples'],0)
        self.assertGreater(report['regions'][1]['unique_conflicting_texels'],0)
        self.assertEqual(report['additional_export_vertices'],4)
        self.assertEqual(report['projected_export_vertex_ceiling'],4112)
        self.assertTrue(report['hd_only']);self.assertFalse(report['weft_appearance_compatible'])
        for actual,expected in zip(args,before):np.testing.assert_array_equal(actual,expected)
        changed=args[2].copy();changed[payloads[1]['face_ids']]=payloads[1]['uvs']
        measured=atlas.analyze_uv_regions(args[0],args[1],changed,args[3],args[4],resolution=1024)
        self.assertTrue(all(row['conflicting_surface_samples']==0 for row in measured['regions']))
        self.assertTrue(all(row['method']=='keep_uv0' for row in measured['regions']))

    def test_density_and_subpixel_gate_is_not_named_for_anatomy(self):
        points,faces,uv,selected,regions=fixture()
        selected[2:]=False;uv[:2]*=.0001
        report,_=atlas.plan_normal_atlases(points,faces,uv,selected,regions,
                                         source_export_vertices=12,resolution=1024)
        self.assertEqual(report['atlas_regions'],[0])
        self.assertIn('over_five_percent_subpixel_faces',report['regions'][0]['reasons'])

    def test_atlas_is_deterministic_padded_and_tangents_match_unchanged_normals(self):
        points,faces,*_=fixture();ids=np.arange(2,6,dtype=np.int32)
        uv,report=atlas.triangle_atlas(points,faces,ids)
        again,second=atlas.triangle_atlas(points,faces,ids)
        np.testing.assert_array_equal(uv,again)
        self.assertEqual(report,second)
        self.assertTrue(np.all(uv>0));self.assertTrue(np.all(uv<1))
        for i in range(4):
            low=uv[i].min(axis=0)*1024
            self.assertTrue(np.all(low>=[i%2*512+2,i//2*512+2]))
        normals=np.tile([0.,0.,1.],(4,3,1))
        tangents=atlas.atlas_tangents(points,faces,ids,uv,normals)
        np.testing.assert_allclose(np.linalg.norm(tangents[:,:,:3],axis=2),1,atol=1e-7)
        np.testing.assert_allclose(np.sum(tangents[:,:,:3]*normals,axis=2),0,atol=1e-7)
        self.assertTrue(np.all(np.abs(tangents[:,:,3])==1))
        mirrored=uv.copy();mirrored[:,:,0]=1-mirrored[:,:,0]
        inverse=atlas.atlas_tangents(points,faces,ids,mirrored,normals)
        np.testing.assert_array_equal(inverse[:,:,3],-tangents[:,:,3])

    def test_export_budget_rejects_before_packing_and_face_cap_is_enforced(self):
        with patch.object(atlas,'triangle_atlas') as pack,self.assertRaisesRegex(ValueError,'aggregate export budget'):
            atlas.plan_normal_atlases(*fixture(),source_export_vertices=99999,resolution=1024,reserve_vertices=0)
        pack.assert_not_called()
        with patch.object(atlas,'MAX_ATLAS_FACES',3),self.assertRaisesRegex(ValueError,'repair bound'):
            atlas.plan_normal_atlases(*fixture(),source_export_vertices=12,resolution=1024)

    def test_mikk_orientation_preserves_opposed_normals_with_known_world_vectors(self):
        points=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        faces=np.array([[0,1,2]]);ids=np.array([0])
        uv=np.array([[[0.,0.],[1.,0.],[0.,1.]]])
        normals=np.array([[[0.,0.,-1.],[0.,0.,1.],[0.,0.,-1.]]])
        before=[value.copy() for value in (points,faces,ids,uv,normals)]
        for mirrored in (False,True):
            chart=uv.copy()
            if mirrored:chart[:,:,0]=1-chart[:,:,0]
            result=atlas.atlas_tangents(points,faces,ids,chart,normals)
            direction=-1. if mirrored else 1.
            np.testing.assert_array_equal(result[:,:,:3],np.tile([direction,0.,0.],(1,3,1)))
            np.testing.assert_array_equal(result[:,:,3],np.full((1,3),direction))
            # Known X/Y/Z axes independently predict the sampled normal's world vector.
            color=np.array([.3,.4,np.sqrt(.75)])
            observed=result[:,:,:3]*color[0]+np.cross(normals,result[:,:,:3])*result[:,:,3,None]*color[1]+normals*color[2]
            expected=np.array([[[direction*.3,-.4,-np.sqrt(.75)],
                                [direction*.3,.4,np.sqrt(.75)],[direction*.3,-.4,-np.sqrt(.75)]]])
            np.testing.assert_allclose(observed,expected,atol=1e-7)
        for actual,original in zip((points,faces,ids,uv,normals),before):
            np.testing.assert_array_equal(actual,original)

    def test_invalid_inputs_fail_before_raster(self):
        for index,bad in [(0,np.full((12,3),np.nan)),(1,np.full((6,3),-1)),
                          (2,np.full((6,3,2),2.)),(3,np.ones(6,dtype=int)),
                          (4,np.full(6,-1)),(4,np.full(6,32)),
                          (0,np.broadcast_to(np.zeros((1,3)),(100001,3)))]:
            args=list(fixture());args[index]=bad
            with self.subTest(index=index),patch.object(atlas,'_coverage') as raster,self.assertRaises(ValueError):
                atlas.analyze_uv_regions(*args,resolution=1024)
            raster.assert_not_called()
        for kwargs in ({'resolution':True},{'resolution':4096},{'pitch_m':True},{'pitch_m':float('nan')}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):atlas.analyze_uv_regions(*fixture(),**kwargs)

    def test_runtime_raster_chart_and_tangent_failures_are_explicit(self):
        with patch.object(atlas,'MAX_SECONDS',-1),self.assertRaisesRegex(RuntimeError,'deadline'):
            atlas.analyze_uv_regions(*fixture(),resolution=1024)
        with patch.object(atlas,'MAX_PIXELS',8),self.assertRaisesRegex(ValueError,'raster work'):
            atlas.analyze_uv_regions(*fixture(),resolution=1024)
        points,faces,*_=fixture();ids=np.array([0],dtype=np.int32)
        bad=points.copy();bad[1:3]=bad[0]
        with self.assertRaisesRegex(ValueError,'degenerate geometric'):atlas.triangle_atlas(bad,faces,ids)
        with self.assertRaises(ValueError):atlas.triangle_atlas(points,faces,np.array([0,0]))
        uv,_=atlas.triangle_atlas(points,faces,ids)
        with self.assertRaisesRegex(ValueError,'tangent geometry is degenerate'):
            atlas.atlas_tangents(bad,faces,ids,uv,np.tile([0.,0.,1.],(1,3,1)))
        for bad_ids,normals in [(np.array([-1]),np.tile([0.,0.,1.],(1,3,1))),
                                (ids,np.zeros((1,3,3))),(ids,np.full((1,3,3),np.nan))]:
            with self.assertRaises(ValueError):atlas.atlas_tangents(points,faces,bad_ids,uv,normals)


if __name__=='__main__':unittest.main()
