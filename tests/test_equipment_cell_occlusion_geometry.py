"""Analytic occlusion counterexamples; no source-video or physical claims."""

from copy import deepcopy
import json

import pytest

from weft import equipment_cell_occlusion_geometry as module
from weft.equipment_temporal_silhouette import propose_temporal_equipment_geometry
from weft.silhouette_geometry import prepare_silhouette_evidence
from weft.world_evidence import _freeze,_thaw
from test_equipment_temporal_silhouette import two_frames,PRIOR
from test_silhouette_geometry import OTHERS,repin


def fixture(*,full=False,holes=False):
    scene,kwargs=two_frames(first_occluded=True,second_occluded=True,holes=holes)
    seg,masks=json.loads(kwargs['segmentation_evidence']),json.loads(kwargs['mask_document'])
    for row in seg['observations']:
        if row['object_id']==OTHERS[0] and row['proposal_status']=='returned':
            b=[360,350,410,420] if full else [380,350,390,420]
            row['geometry'].update(box_xyxy_halfopen=b,box_xywh=[b[0],b[1],b[2]-b[0],b[3]-b[1]],
                center_px=[(b[0]+b[2])/2,(b[1]+b[3])/2])
    scene,kwargs=repin(scene,kwargs,seg,masks)
    context=prepare_silhouette_evidence(scene,**kwargs)
    previous=propose_temporal_equipment_geometry(context,'marker_box',construction_frames=[1,2],static_prior=PRIOR,conditional_exterior=True)
    assert previous['status']=='no_qualified_improvement'
    prior=previous['report']
    prior.update(initial_association_prior={'scope':'analytic assumed correspondence'},association_proof={'scope':'analytic fixture'},conditional_source_chain=[])
    return context,prior


def test_central_occluder_preserves_unknown_cells_but_clear_outer_edges_can_shrink():
    context,prior=fixture()
    result=module._fit(context,'marker_box',prior,max_seconds=2)
    assert result['status']=='conditional_proposal'
    assert all(result['report']['qualification'].values())
    assert all(f['removed_uncertain_cells']==f['new_foreground_misses']==0 for f in result['report']['frames'])
    assert all(f['actual_cell_uncertain_cells']<f['prior_whole_row_uncertain_cells'] for f in result['report']['frames'])
    assert result['report']['source_fit']['candidate_cell_conditional_exterior_cells']<result['report']['source_fit']['baseline_cell_conditional_exterior_cells']
    assert result['report']['contact']=='unknown' and not result['report']['physical_identity_verified']


def test_full_occlusion_cannot_produce_conditional_exterior_gain():
    context,prior=fixture(full=True)
    result=module._fit(context,'marker_box',prior,max_seconds=2)
    assert result['geometry_config'] is None
    assert not result['report']['qualification']['strict_conditional_exterior_reduction']
    assert all(f['removed_uncertain_cells']==0 for f in result['report']['frames'])


def test_missing_competitor_protects_every_view_cell():
    context,prior=fixture()
    rows=_thaw(context._rows)
    for frame in [1,2]:rows[frame,OTHERS[0]]['proposal_status']='missing'
    object.__setattr__(context,'_rows',_freeze(rows))
    result=module._fit(context,'marker_box',prior,max_seconds=2)
    assert result['geometry_config'] is None
    assert all(f['removed_uncertain_cells']==0 for f in result['report']['frames'])


def test_mask_holes_stay_filled_and_depth_inertia_unchanged():
    context,prior=fixture(holes=True)
    result=module._fit(context,'marker_box',prior,max_seconds=2)
    assert result['status']=='conditional_proposal'
    assert all(f['filled_hole_cells']>0 for f in result['report']['frames'])
    assert result['candidate_geometry']['inertia']==prior['candidate_geometry']['inertia']
    assert all(new['size'][1:]==old['size'][1:] and new['position'][1:]==old['position'][1:]
        for new,old in zip(result['candidate_geometry']['parts'],prior['candidate_geometry']['parts']))
    assert all(all(v==0 for v in band['lost_row_hull_cells']) for band in result['report']['bands'])


def test_opposite_occluders_across_views_remain_jointly_protected():
    context,prior=fixture()
    rows=_thaw(context._rows)
    for frame,b in [(1,[360,350,380,420]),(2,[390,350,410,420])]:
        rows[frame,OTHERS[0]]['box']['box_xyxy_halfopen']=b
    object.__setattr__(context,'_rows',_freeze(rows))
    result=module._fit(context,'marker_box',prior,max_seconds=2)
    assert all(all(v==0 for v in band['lost_uncertain_cells']) for band in result['report']['bands'])
    assert all(f['removed_uncertain_cells']==0 for f in result['report']['frames'])


def test_static_screen_failure_cannot_select_a_candidate():
    context,prior=fixture()
    rows=_thaw(context._rows)
    target=prior['source_reference']['object_id']
    rows[2,target]['box']['box_xyxy_halfopen'][0]+=4
    rows[2,target]['box']['box_xyxy_halfopen'][2]+=4
    object.__setattr__(context,'_rows',_freeze(rows))
    result=module._fit(context,'marker_box',prior,max_seconds=2)
    assert result['geometry_config'] is None
    assert not result['report']['qualification']['image_stationarity_screen']


def test_public_entry_rejects_missing_prior_or_unverified_archive():
    context,prior=fixture()
    with pytest.raises(ValueError):
        module.propose_cell_occlusion_geometry(context,'marker_box',retained_archive=b'not a proof',expected_archive_sha256='0'*64)


def test_empty_target_row_keeps_full_row_uncertainty():
    context,prior=fixture()
    rasters=_thaw(context._rasters)
    target=prior['source_reference']['object_id']
    for frame in [1,2]:
        path=context._rows[frame,target]['raster_path']
        shape,binary,pin=rasters[path]
        changed=bytearray(binary);width=shape[1]
        changed[30*width:31*width]=b'\0'*width
        rasters[path]=(shape,bytes(changed),pin)
    object.__setattr__(context,'_rasters',_freeze(rasters))
    result=module._fit(context,'marker_box',prior,max_seconds=2)
    assert all(f['removed_uncertain_cells']==0 for f in result['report']['frames'])
    assert all(all(value==0 for value in band['lost_uncertain_cells']) for band in result['report']['bands'])
