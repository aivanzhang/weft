"""Analytic regional fin invariants, not rendered fur or avatar-quality approval."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import numpy as np

from avatar_assets.runner import resident_bytes
from scripts.fringe_flow import reshape_fringe


def fixture(count=4, angular=False):
    center = np.column_stack((np.linspace(.2,.8,count),np.full(count,.5),np.full(count,.5)))
    positions = np.tile(np.array([[-.006,0,0],[.006,0,0],[-.006,0,.008],[.006,0,.008]]),(count,1,1))
    normals = np.tile([0.,0,1],(count,4,1))
    if angular:
        angle = np.linspace(-.6,.6,count)
        for i,a in enumerate(angle):
            rotation = np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
            positions[i] = positions[i]@rotation.T; normals[i] = normals[i]@rotation.T
    positions += center[:,None]
    front = (np.arange(count)[:,None,None]*4+np.array([[0,1,3],[0,3,2]])).reshape(-1,3)
    weights = np.tile([[.1,.2,.3,.4],[.4,.3,.2,.1]],(count,2,1)).reshape(-1,4).astype('<f4')
    return {'positions':positions.reshape(-1,3).astype('<f4'),'normals':normals.reshape(-1,3).astype('<f4'),
        'uv':np.tile([[.1,.9],[.9,.9],[.1,.1],[.9,.1]],(count,1)).astype('<f4'),
        'triangles':np.concatenate((front,front[:,::-1])).astype('<i4'),
        'joints':np.tile([0,1,2,3],(count*4,1)).astype('<u2'),'weights':weights}


def region(name='flow', **changes):
    return {'name':name,'center':[.5,.5,.5],'radius':[2,2,2],'feather':.5,
            'direction':[0,1,0],'comb':.6,'length_scale':1,**changes}


def arguments(**changes):
    return {'low':[0,0,0],'span':[1,1,1],'base_direction':[0,1,0],'base_comb':.6,
            'groom_flow':{'regions':[region()]},'base_length_m':.0045,
            'source_export_vertices':20000,'source_export_triangles':60000,**changes}


class FringeFlowTests(unittest.TestCase):
    def test_exact_root_skin_normal_uv_topology_and_input_preservation(self):
        for angular in (False,True):
            source = fixture(9,angular); before = deepcopy(source)
            result = reshape_fringe(source,**arguments())
            for key in source:
                np.testing.assert_array_equal(source[key],before[key])
                self.assertFalse(np.shares_memory(source[key],result[key]))
                if key != 'positions':self.assertEqual(source[key].tobytes(),result[key].tobytes())
                self.assertEqual(result['report']['input_arrays'][key]['sha256'],hashlib.sha256(source[key].tobytes()).hexdigest())
            roots_before = source['positions'].reshape(-1,4,3)[:,:2]
            roots_after = result['positions'].reshape(-1,4,3)[:,:2]
            self.assertEqual(roots_before.tobytes(),roots_after.tobytes())
            self.assertFalse(np.array_equal(source['positions'],result['positions']))
            self.assertFalse(result['report']['quality_approved']); self.assertFalse(result['report']['motion_verified'])
            self.assertFalse(result['report']['selection_changed'])

    def test_distinct_midpoint_regions_change_actual_comb_and_length(self):
        source = fixture(2)
        regions = [region('first',center=[.2,.5,.5],radius=[.1]*3,comb=.25,length_scale=.7),
                   region('second',center=[.8,.5,.5],radius=[.1]*3,comb=.65,length_scale=1.4,direction=[0,-1,0])]
        result = reshape_fringe(source,**arguments(groom_flow={'regions':regions}))
        rows = result['positions'].reshape(2,4,3).astype(float)
        delta = rows[:,2]-rows[:,0]
        np.testing.assert_allclose(np.linalg.norm(delta,axis=1),[.00315,.0063],rtol=0,atol=1e-7)
        expected = np.array([[0,.25,.75],[0,-.65,.35]])
        expected /= np.linalg.norm(expected,axis=1)[:,None]
        np.testing.assert_allclose(delta/np.linalg.norm(delta,axis=1)[:,None],expected,rtol=0,atol=1e-5)
        self.assertLess(result['report']['outward_angle_degrees']['min'],19)
        self.assertGreater(result['report']['outward_angle_degrees']['max'],61)

    def test_roots_match_independent_joint_skin_motion_after_tip_change(self):
        source = fixture(5,True); result = reshape_fringe(source,**arguments())
        matrices = np.tile(np.eye(4),(4,1,1)); angles=[0,.3,-.8,1.2]
        for i,a in enumerate(angles):
            matrices[i,:3,:3]=[[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]]
            matrices[i,:3,3]=[i*.1,-i*.05,i*.03]
        def posed(data):
            p=np.column_stack((data['positions'],np.ones(len(data['positions']))))
            return np.sum(np.einsum('nkij,nj->nki',matrices[data['joints']],p)[...,:3]*data['weights'][...,None],axis=1)
        a,b=posed(source).reshape(-1,4,3),posed(result).reshape(-1,4,3)
        np.testing.assert_array_equal(a[:,:2],b[:,:2])
        self.assertGreater(np.max(abs(a[:,2:]-b[:,2:])),.001)

    def test_parallel_flow_does_not_invent_a_tangent_or_inward_direction(self):
        source=fixture(3)
        result=reshape_fringe(source,**arguments(groom_flow={'regions':[region(direction=[0,0,-1],comb=.95)]}))
        delta=result['positions'].reshape(-1,4,3)[:,2:]-source['positions'].reshape(-1,4,3)[:,:2]
        np.testing.assert_allclose(delta,np.tile([0,0,.0045],(3,2,1)),rtol=0,atol=1e-7)
        self.assertEqual(result['report']['parallel_field_roots'],3)

    def test_deterministic_arrays_and_field_binding(self):
        source=fixture(6,True); first=reshape_fringe(source,**arguments());second=reshape_fringe(source,**arguments())
        for key in source:self.assertEqual(first[key].tobytes(),second[key].tobytes())
        self.assertEqual(first['report']['prepared_field_sha256'],second['report']['prepared_field_sha256'])
        changed=reshape_fringe(source,**arguments(groom_flow={'regions':[region(comb=.4)]}))
        self.assertNotEqual(first['report']['prepared_field_sha256'],changed['report']['prepared_field_sha256'])
        self.assertNotEqual(first['report']['output_positions_sha256'],changed['report']['output_positions_sha256'])

    def test_inward_endpoint_response_is_rejected_without_clamping(self):
        source=fixture(1); a=np.radians(25)
        source['normals'][:]=np.array([[np.sin(a),0,np.cos(a)],[-np.sin(a),0,np.cos(a)]]*2,dtype='<f4')
        original=deepcopy(source)
        with self.assertRaisesRegex(ValueError,'inward at a root'):
            reshape_fringe(source,**arguments(groom_flow={'regions':[region(direction=[1,0,0],comb=.95)]}))
        for key in source:np.testing.assert_array_equal(source[key],original[key])

    def test_regional_length_bounds_are_rejected_not_silently_clipped(self):
        for length,scale in ((.03,2),(.001,.25)):
            with self.subTest(length=length),self.assertRaisesRegex(ValueError,'no silent clamp'):
                reshape_fringe(fixture(),**arguments(base_length_m=length,groom_flow={'regions':[region(length_scale=scale)]}))
        for value in (False,float('nan'),float('inf'),.0009,.041):
            with self.subTest(value=value),self.assertRaises(ValueError):
                reshape_fringe(fixture(),**arguments(base_length_m=value))

    def test_invalid_arrays_and_changed_skin_or_topology_are_refused(self):
        for mutation in ('nan','boolean','UV','bad_joint','weight_sum','tip_weight','tip_normal','winding','collapsed_edge','inward_input','injection'):
            source=fixture()
            if mutation=='nan':source['positions'][0,0]=np.nan
            elif mutation=='boolean':source['positions']=source['positions'].astype(bool)
            elif mutation=='UV':source['uv'][0,0]=1.01
            elif mutation=='bad_joint':source['joints'][0,0]=64
            elif mutation=='weight_sum':source['weights'][0,0]=.9
            elif mutation=='tip_weight':source['weights'][2]=source['weights'][1]
            elif mutation=='tip_normal':source['normals'][2]=[0,1,0]
            elif mutation=='winding':source['triangles'][0]=source['triangles'][0,::-1]
            elif mutation=='collapsed_edge':source['positions'][1]=source['positions'][0]
            elif mutation=='inward_input':source['positions'][2:,2]-=.1
            else:source['callback']=lambda:None
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):reshape_fringe(source,**arguments())

    def test_invalid_field_and_preallocation_resource_bounds(self):
        changes=[{'span':[1,0,1]},{'low':[0,False,0]},{'groom_flow':{'regions':[]}},
                 {'groom_flow':{'regions':[region()]*13}},{'groom_flow':{'regions':[region(comb=float('nan'))]}},
                 {'source_export_vertices':100000},{'source_export_triangles':100000}]
        for update in changes:
            with self.subTest(update=update),self.assertRaises(ValueError):reshape_fringe(fixture(),**arguments(**update))
        with patch('scripts.fringe_flow.prepare_flow',side_effect=AssertionError('Budget must fail first')):
            with self.assertRaisesRegex(ValueError,'100000'):reshape_fringe(fixture(),**arguments(source_export_vertices=100000))
        with self.assertRaisesRegex(ValueError,'4800'):reshape_fringe(fixture(4801),**arguments())

    def test_maximum_count_is_bounded_and_keeps_all_roots(self):
        source=fixture(4800,True);start=time.monotonic();result=reshape_fringe(source,**arguments())
        self.assertEqual(result['report']['fins'],4800);self.assertEqual(len(result['positions']),19200)
        self.assertEqual(len(result['triangles']),19200)
        self.assertLess(time.monotonic()-start,10)
        self.assertEqual(result['report']['export_budget']['projected_vertices'],81696)
        if os.environ.get('FRINGE_FLOW_TEST_RECEIPT'):
            receipt={'fixture':'Analytic4800 selected four-vertex fins; no actual avatar or renderer',
                     'report':result['report'],'rss_bytes':resident_bytes(os.getpid())}
            with Path(os.environ['FRINGE_FLOW_TEST_RECEIPT']).open('x') as handle:json.dump(receipt,handle,indent=2)


if __name__=='__main__':unittest.main()
