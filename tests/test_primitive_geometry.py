from copy import deepcopy

import numpy as np
import pytest

from test_spatial_rigid import scene
from weft.primitive_geometry import conditional_box_depth, fit_initial_width, silhouette_overlap


def test_convex_overlap_uses_mask_cell_centers_and_counts_outside_predictions():
    mask=np.ones((3,3),dtype=np.uint8)
    result=silhouette_overlap(mask,[100,200,106,206],[[100,203],[103,200],[106,203],[103,206]])
    assert result['source_cells']==9 and result['projected_cells']==5
    assert result['intersection_cells']==5 and result['iou']==5/9
    outside=silhouette_overlap(mask,[100,200,106,206],[[108,202],[112,202],[112,206],[108,206]])
    assert outside['projected_cells']==4 and outside['intersection_cells']==0 and outside['union_cells']==13


def test_primitive_width_fit_recovers_known_projection_and_preserves_every_other_field():
    value=scene();value['objects'][0]['size'][0]=.1;before=deepcopy(value)
    projection={'scope':'construction','matrix_px':[[1000,0,0,0],[0,0,-1000,800]]}
    candidate,report=fit_initial_width(value,'block',projection,np.ones((50,80),np.uint8),[210,475,290,525])
    assert candidate['objects'][0]['size'][0]==pytest.approx(.08)
    assert report['selected']['overlap']['iou']==1
    assert report['depth_identified'] is False and len(report['alternatives'])==31
    old=deepcopy(value['objects'][0]);new=deepcopy(candidate['objects'][0])
    old.pop('provenance');new.pop('provenance');new['size'][0]=old['size'][0]
    assert old==new and value==before


def test_reported_volume_only_gives_a_depth_conditional_on_the_scale_hypothesis():
    result=conditional_box_depth(.03,.03,reported_volume_ml=70.3125,normalized_gravity=9.81/1.25)
    assert result['depth_scene_units']==pytest.approx(.04)
    assert result['conditional_meters_per_scene_width']==pytest.approx(1.25)
    assert result['metric_calibration_verified'] is False


@pytest.mark.parametrize('mask,box,polygon',[
    (np.zeros((3,3)),[0,0,3,3],[[0,0],[3,0],[3,3]]),
    (np.ones((3,3))*2,[0,0,3,3],[[0,0],[3,0],[3,3]]),
    (np.ones((3,3)),[0,0,0,3],[[0,0],[3,0],[3,3]]),
    (np.ones((3,3)),[0,0,3,3],[[0,0],[1,1],[2,2]]),
    (np.ones((3,3)),[0,0,3,3],[[-2000,0],[2000,0],[2000,2000]]),
])
def test_missing_invalid_or_unbounded_silhouettes_are_rejected(mask,box,polygon):
    with pytest.raises(ValueError):silhouette_overlap(mask,box,polygon)


@pytest.mark.parametrize('arguments',[
    {'reported_volume_ml':False},{'normalized_gravity':float('nan')},{'reported_volume_ml':10000},
])
def test_invalid_physical_scale_assumptions_are_rejected(arguments):
    values={'reported_volume_ml':33,'normalized_gravity':9.4}|arguments
    with pytest.raises(ValueError):conditional_box_depth(.03,.03,**values)


def test_evaluation_scope_and_static_geometry_cannot_drive_width_fitting():
    projection={'scope':'evaluation','matrix_px':[[1000,0,0,0],[0,0,-1000,800]]}
    with pytest.raises(ValueError):fit_initial_width(scene(),'block',projection,np.ones((50,80),np.uint8),[210,475,290,525])
    projection['scope']='construction'
    with pytest.raises(ValueError):fit_initial_width(scene(),'floor',projection,np.ones((50,80),np.uint8),[210,475,290,525])


def test_vanishing_source_extents_are_rejected_before_native_hull_or_grid_allocation():
    with pytest.raises(ValueError,match='sampling-grid bound'):
        silhouette_overlap(np.ones((3,3)),[0,0,1e-310,1e-310],[[0,0],[3,0],[3,3]])
