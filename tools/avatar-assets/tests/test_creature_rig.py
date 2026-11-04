"""Analytic skin fields, not finished dragon assets."""

from copy import deepcopy
import unittest

import numpy as np

from scripts.creature_rig_math import validate_rig, skin_weights


def design():
    return {'space':'normalized_bounds_blender_z_up','height_meters':1.2,
            'bones':[{'name':'Body','parent':None,'head':[.5,.5,0],'tail':[.5,.5,1]},
                     {'name':'Wing','parent':'Body','head':[.5,.5,.5],'tail':[1,.5,.5]}],
            'base_bones':['Body'],'layers':[{'bones':['Wing'],'min':[.6,0,0],'max':[1.1,1,1],
                                           'feather':.1,'min_vertices':1}]}


class CreatureRigTests(unittest.TestCase):
    def test_uv_seam_duplicates_receive_identical_normalized_weights(self):
        points=np.array([[.5,.5,.5],[.65,.5,.5],[.9,.5,.5],[.65,.5,.5]])
        weights, report=skin_weights(points,[0,0,0],[1,1,1],design())
        np.testing.assert_allclose(weights.sum(axis=1),1)
        np.testing.assert_array_equal(weights[1],weights[3])
        np.testing.assert_allclose(weights[0],[1,0])
        np.testing.assert_allclose(weights[2],[0,1])
        self.assertEqual(report['max_influences'],2)

    def test_bad_hierarchy_nonfinite_and_zero_bones_rejected(self):
        for edit in ('cycle','nan','zero','unknown'):
            cfg=design()
            if edit=='cycle': cfg['bones'][0]['parent']='Wing'
            if edit=='nan': cfg['bones'][1]['tail'][0]=float('nan')
            if edit=='zero': cfg['bones'][1]['tail']=cfg['bones'][1]['head'][:]
            if edit=='unknown': cfg['layers'][0]['bones']=['Missing']
            with self.subTest(edit=edit),self.assertRaises(ValueError): validate_rig(cfg)

    def test_empty_layer_and_oversized_points_fail_explicitly(self):
        for points in (np.array([[.5,.5,.5]]),np.zeros((100001,3))):
            with self.assertRaises(ValueError): skin_weights(points,[0,0,0],[1,1,1],design())

    def test_chain_distant_envelope_overlap_is_pruned(self):
        # A forearm segment resting right beside the hip: its envelope overlaps the
        # pelvis region spatially, but pelvis and forearm are far apart on the chain.
        config = {'space': 'normalized_bounds_blender_z_up', 'height_meters': 1.0,
                  'bones': [
                      {'name': 'root', 'parent': None, 'head': [.5, .5, .5], 'tail': [.5, .5, .6]},
                      {'name': 'chest', 'parent': 'root', 'head': [.5, .5, .6], 'tail': [.5, .5, .8]},
                      {'name': 'upperarm', 'parent': 'chest', 'head': [.6, .5, .8], 'tail': [.62, .5, .6]},
                      {'name': 'forearm', 'parent': 'upperarm', 'head': [.62, .5, .6], 'tail': [.62, .5, .45]}],
                  'base_bones': ['root', 'chest', 'upperarm', 'forearm'], 'layers': []}
        points = [[.52, .5, .5], [.62, .5, .5]]
        weights, proof = skin_weights(points, [0, 0, 0], [1, 1, 1], config)
        names = [b['name'] for b in config['bones']]
        hip = dict(zip(names, weights[0]))
        assert hip['forearm'] == 0, hip
        assert abs(sum(hip.values()) - 1) < 1e-6
        forearm_vertex = dict(zip(names, weights[1]))
        assert forearm_vertex['forearm'] > 0.5
        assert proof['chain_localized_vertices'] >= 1

    def test_many_bones_keep_at_most_four_influences(self):
        cfg=design();cfg['layers']=[]
        for i in range(6):
            cfg['bones'].append({'name':f'Tail{i}','parent':'Body','head':[.1+i*.1,.1,0],'tail':[.1+i*.1,.1,1]})
        cfg['base_bones']=[b['name'] for b in cfg['bones']]
        weights,_=skin_weights(np.array([[.3,.4,.5],[.7,.6,.5]]),[0,0,0],[1,1,1],cfg)
        self.assertLessEqual(int((weights>0).sum(axis=1).max()),4)
        np.testing.assert_allclose(weights.sum(axis=1),1)


if __name__=='__main__': unittest.main()
