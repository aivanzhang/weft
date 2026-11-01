"""Exact radial brackets and outer enclosures; sampling supplements, not replaces, the proof."""

from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction
from hashlib import sha256
import importlib
import json
import math
import time

import pytest

from test_multiview_landmark_evidence import denied, local_only


@pytest.fixture
def api():
    return importlib.import_module('weft.source_surface_envelopes')


def fixture(count=4, channels=1, radial=.1):
    old = importlib.import_module('test_source_surface_envelopes')
    data = old.fixture(count, channels)
    data[1]['radial_k1'] = radial
    return data


def prepare(api, data, **changes):
    old = importlib.import_module('test_source_surface_envelopes')
    return api.prepare_source_surface_envelopes(**(old.encoded(data) | changes))


def inverse(api, radial=.1):
    return api._RadialInverse(api._inverse([[100., 0., 50.], [0., 100., 40.], [0., 0., 1.]]), radial, math.inf)


@pytest.mark.parametrize('radial,squared_radius', [
    (.1, Fraction(0)), (.1, Fraction(1, 100)), (.1, Fraction(16)),
    (.00047528394168603385, Fraction(3, 2)), (1., Fraction(16)),
    (1., Fraction(4)), (1e-200, Fraction(16)), (.5, Fraction(1, 10**20)),
])
def test_exact_polynomial_sign_brackets_certify_unique_inverse(api, radial, squared_radius):
    value = inverse(api, radial)
    low, high = value.scale(squared_radius)
    coefficient = Fraction(radial) * squared_radius
    assert 0 <= low <= high <= 1
    assert low + coefficient*low**3 <= 1 <= high + coefficient*high**3
    assert high-low <= Fraction(1, 2**48)
    assert value.bisections <= 48
    # Positive derivative proves uniqueness throughout the bracket, independently of point samples.
    assert 1 + 3*coefficient*low**2 > 0
    if squared_radius == 0:
        assert (low, high, value.bisections) == (1, 1, 0)


def test_inverse_scale_order_and_positive_sign_follow_radius_not_a_selected_point(api):
    value = inverse(api, .3)
    brackets = [value.scale(Fraction(s)) for s in (0, .1, 1, 4, 16)]
    for left, right in zip(brackets, brackets[1:]):
        assert left[0] >= right[0] and left[1] >= right[1] > 0
    assert value.radial == Fraction(.3)


@pytest.mark.parametrize('u,v', [(Fraction(1, 2), Fraction(1, 4)), (Fraction(-1, 2), Fraction(1, 4)),
    (Fraction(1, 2), Fraction(-1, 4)), (Fraction(-1, 2), Fraction(-1, 4)),
    (Fraction(0), Fraction(1, 2)), (Fraction(0), Fraction(0))])
def test_exact_forward_rational_points_are_enclosed_without_tolerance(api, u, v):
    data = fixture(radial=.5)
    data[0]['source'].update(width_px=128, height_px=128)
    data[0]['image'].update(width_px=128, height_px=128)
    data[1].update(width=128, height=128, K=[[64., 0., 64.], [0., 64., 64.], [0., 0., 1.]])
    radial_scale = 1+Fraction(1, 2)*(u*u+v*v)
    pixel = [float(64+64*u*radial_scale), float(64+64*v*radial_scale)]
    assert all(Fraction(value) == exact for value, exact in zip(pixel, (64+64*u*radial_scale, 64+64*v*radial_scale)))
    for item in data[3]['annotations']:
        item.update(pixel=pixel, uncertainty_px=0.)
    data[2]['surfaces'][0]['height_interval_m'] = [.5, .5]
    result = prepare(api, data)
    anchor = result.frame(0)['anchors'][0]
    assert anchor['status'] == 'conditional_surface_envelope'
    expected = (Fraction(3, 2)*u, -Fraction(3, 2)*v, Fraction(1, 2))
    for lo, point, hi in zip(anchor['world_envelope_m']['minimum'], expected, anchor['world_envelope_m']['maximum']):
        assert Fraction(lo) <= point <= Fraction(hi)
    lo, hi = anchor['radial_inverse']['inverse_scale_interval']
    assert Fraction(lo) <= 1/radial_scale <= Fraction(hi)
    assert anchor['radial_inverse']['radial_k1'] == .5
    assert 'position_world_m' not in anchor and 'midpoint' not in anchor
    assert result.describe()['contact'] == result.describe()['load'] == result.describe()['grasp'] == 'unknown'


def decimal_point(camera, pixel, height):
    """Independent high-precision radius Newton solve and Gaussian camera transform."""
    def d(value):
        return Decimal.from_float(value) if type(value) is float else Decimal(value)
    def solve(matrix, vector):
        rows = [[d(value) for value in row]+[vector[i]] for i, row in enumerate(matrix)]
        for column in range(3):
            pivot = max(range(column, 3), key=lambda i: abs(rows[i][column]))
            rows[column], rows[pivot] = rows[pivot], rows[column]
            scale = rows[column][column]
            rows[column] = [value/scale for value in rows[column]]
            for index in range(3):
                if index != column:
                    scale = rows[index][column]
                    rows[index] = [a-scale*b for a, b in zip(rows[index], rows[column])]
        return [row[-1] for row in rows]
    with localcontext() as context:
        context.prec = 100
        yd = (d(pixel[1])-d(camera['K'][1][2]))/d(camera['K'][1][1])
        xd = (d(pixel[0])-d(camera['K'][0][2])-d(camera['K'][0][1])*yd)/d(camera['K'][0][0])
        rd = (xd*xd+yd*yd).sqrt()
        radius, k = rd, d(camera['radial_k1'])
        for _ in range(80):
            radius -= (radius+k*radius**3-rd)/(1+3*k*radius**2)
        alpha = radius/rd if rd else Decimal(1)
        pose = camera['frames'][0]
        direction = solve(pose['R'], [alpha*xd, alpha*yd, Decimal(1)])
        origin = solve(pose['R'], [-d(value) for value in pose['t']])
        depth = (d(height)-origin[2])/direction[2]
        return [a+depth*b for a, b in zip(origin, direction)]


@pytest.mark.parametrize('radial', [.00047528394168603385, .7, 1.])
def test_full_pixel_box_including_interior_is_enclosed_with_skew_and_oblique_camera(api, radial):
    data = fixture(radial=radial)
    data[1]['K'][0][1] = 13.
    for pose in data[1]['frames']:
        pose.update(R=[[1., 0., 0.], [0., -.8, -.6], [0., .6, -.8]], t=[-1., 3., 4.])
    data[2]['surfaces'][0].update(kind='static_support', bounds_xy_m=[-10., -10., 10., 10.])
    result = prepare(api, data)
    anchor = result.frame(0)['anchors'][0]
    assert anchor['status'] == 'conditional_surface_envelope'
    lower, upper = ([Decimal.from_float(value) for value in anchor['world_envelope_m'][key]] for key in ('minimum', 'maximum'))
    for x in (59., 59.5, 60., 60.5, 61.):
        for y in (44., 44.5, 45., 45.5, 46.):
            for height in (.2, .4, .6):
                point = decimal_point(data[1], [x, y], height)
                assert all(lo <= value <= hi for lo, value, hi in zip(lower, point, upper))
    assert anchor['world_envelope_m']['certificate'] == 'exact_rational_monotone_radial_outer_box_plane_extrema_outward_binary64'


def test_interval_products_cover_axis_crossings_not_only_corner_inverse_samples(api):
    value = inverse(api, .8)
    box, reason = value.bounds((Fraction(40), Fraction(60)), (Fraction(30), Fraction(50)))
    assert reason is None and box['squared_radius'][0] == 0
    assert box['scales'][1] == 1
    for interval in box['intervals']:
        assert interval[0] < 0 < interval[1]
    # The box must include the axes' larger inverse scale; corner-only scaling would exclude it.
    lo, hi = value.scale(Fraction(1, 100))
    assert box['intervals'][0][1] >= Fraction(1, 10)*hi
    assert box['scales'][0] <= lo


def test_more_authored_pixel_or_height_uncertainty_cannot_shrink_the_outer_enclosure(api):
    data = fixture(radial=.5)
    small = prepare(api, data).frame(0)['anchors'][0]['world_envelope_m']
    data[2]['surfaces'][0]['height_interval_m'] = [0., .8]
    for item in data[3]['annotations']:
        item['uncertainty_px'] = 3.
    large = prepare(api, data).frame(0)['anchors'][0]['world_envelope_m']
    assert all(lo2 <= lo1 for lo1, lo2 in zip(small['minimum'], large['minimum']))
    assert all(hi2 >= hi1 for hi1, hi2 in zip(small['maximum'], large['maximum']))


@pytest.mark.parametrize('radial', [-.1, -1., -1e-200])
def test_negative_branch_stays_unavailable_without_silent_branch_selection(api, radial):
    result = prepare(api, fixture(radial=radial))
    assert result.frame(0)['anchors'][0]['reason'] == 'radial_distortion_has_no_certified_interval_bound'
    assert result.describe()['status'] == 'unavailable'
    assert result.frame_count == 4


def test_entire_normalized_radius_domain_is_checked_without_clipping(api):
    data = fixture(radial=1.)
    data[0]['source'].update(width_px=640, height_px=128)
    data[0]['image'].update(width_px=640, height_px=128)
    data[1].update(width=640, height=128, K=[[100., 0., 100.], [0., 100., 64.], [0., 0., 1.]])
    for item in data[3]['annotations']:
        item.update(pixel=[500., 64.], uncertainty_px=0.)
    exact = prepare(api, data)
    assert exact.frame(0)['anchors'][0]['status'] == 'conditional_surface_envelope'
    assert exact.frame(0)['anchors'][0]['radial_inverse']['distorted_radius_squared_interval'] == [16., 16.]
    for item in data[3]['annotations']:
        item.update(pixel=[math.nextafter(500., math.inf), 64.])
    assert prepare(api, data).frame(0)['anchors'][0]['reason'] == 'radial_pixel_interval_exceeds_inverse_domain'
    for item in data[3]['annotations']:
        item.update(pixel=[499., 64.], uncertainty_px=2.)
    assert prepare(api, data).frame(0)['anchors'][0]['reason'] == 'radial_pixel_interval_exceeds_inverse_domain'


@pytest.mark.parametrize('mode,reason', [('horizon', 'ray_interval_crosses_surface_horizon'),
    ('behind', 'height_interval_behind_camera_or_beyond_depth_bound'),
    ('support', 'envelope_exceeds_declared_surface_domain'), ('image', 'pixel_interval_outside_source_image'),
    ('measurement', 'missing_supplied_camera_measurement_receipt')])
def test_positive_radial_does_not_weaken_existing_geometric_or_evidence_guards(api, mode, reason):
    data = fixture()
    if mode == 'horizon':
        for pose in data[1]['frames']:
            pose.update(R=[[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]], t=[0., 2., 0.])
        for item in data[3]['annotations']:
            item['pixel'] = [50., 40.]
    elif mode == 'behind':
        data[2]['surfaces'][0]['height_interval_m'] = [2., 3.]
    elif mode == 'support':
        data[2]['surfaces'][0].update(kind='static_support', bounds_xy_m=[0., 0., .1, .1])
    elif mode == 'image':
        data[3]['annotations'][0]['pixel'] = [0., 40.]
    else:
        data[1]['quality'] = 'measured'
    result = prepare(api, data)
    assert result.frame(0)['anchors'][0]['reason'] == reason
    assert result.frame(0)['anchors'][0]['world_envelope_m'] is None


def test_excluded_occluded_unsampled_slots_never_enter_radial_geometry(api, monkeypatch):
    data = fixture()
    data[3]['annotations'][1]['status'] = 'occluded'
    data[0]['observations'][2]['partition'] = 'audit'
    data[0]['observations'][3].update(partition='unsampled', visibility='not_sampled')
    for item in data[3]['annotations'][1:]:
        item.update(pixel='do not read', uncertainty_px='do not read')
    old = api._enclose_radial; seen = []
    def wrapped(annotation, *args):
        seen.append(annotation['frame'])
        return old(annotation, *args)
    monkeypatch.setattr(api, '_enclose_radial', wrapped)
    result = prepare(api, data)
    assert seen == [0] and result.describe()['counts']['retained_anchor_slots'] == 4
    assert [result.frame(i)['anchors'][0]['reason'] for i in range(4)] == [None, 'annotation_occluded', 'excluded_evidence_partition', 'source_not_sampled']


def test_original_clock_raw_camera_and_cached_queries_stay_unchanged(api, monkeypatch):
    data = fixture()
    original = importlib.import_module('test_source_surface_envelopes').encoded(data)
    before = deepcopy(original)
    result = api.prepare_source_surface_envelopes(**original)
    assert original == before and result.additional_inputs[0] is original['camera_document']
    meta = result.describe()
    assert meta['bindings']['camera_sha256'] == sha256(original['camera_document']).hexdigest()
    assert meta['radial_inverse']['computed_boxes'] == 1 and meta['radial_inverse']['cache_hits'] == 3
    assert meta['radial_inverse']['bisections'] <= 96
    snapshot = result.frame(0)
    monkeypatch.setattr(api._RadialInverse, 'bounds', denied)
    monkeypatch.setattr(api, '_plane_box', denied)
    snapshot['anchors'][0]['radial_inverse']['radial_k1'] = 0.
    assert result.frame(0)['anchors'][0]['radial_inverse']['radial_k1'] == .1
    assert [result.frame(i)['anchors'][0]['source_time_s'] for i in range(4)] == [i/30 for i in range(4)]


def test_radial_support_cannot_retime_an_incompatible_original_decoder_clock(api):
    data = fixture()
    document = data[0]
    document['schema_version'] = 3
    document['source'].pop('fps')
    document['source']['frame_times_s'] = [0., .034, .068, .102]
    document['time'] = {'unit': 'seconds', 'origin': 'submitted_video_start', 'basis': 'decoder_presentation_timestamps', 'status': 'observed'}
    for values in (document['evidence'], document['observations'], data[3]['annotations']):
        for row in values:
            row['time_s'] = document['source']['frame_times_s'][row['frame']]
    result = prepare(api, data)
    assert result.frame(0)['anchors'][0]['status'] == 'conditional_surface_envelope'
    for index in range(1, 4):
        anchor = result.frame(index)['anchors'][0]
        assert anchor['reason'] == 'camera_and_source_clocks_differ'
        assert anchor['source_time_s'] == document['source']['frame_times_s'][index]


def test_cache_has_a_fixed_entry_limit_and_eviction_does_not_change_bounds(api, monkeypatch):
    monkeypatch.setattr(api, 'RADIAL_CACHE_ENTRIES', 3)
    value = inverse(api)
    initial, _ = value.bounds((Fraction(50), Fraction(51)), (Fraction(40), Fraction(41)))
    for index in range(1, 8):
        value.bounds((Fraction(50+index), Fraction(51+index)), (Fraction(40), Fraction(41)))
        assert len(value.cache) <= 3
    repeated, _ = value.bounds((Fraction(50), Fraction(51)), (Fraction(40), Fraction(41)))
    assert repeated == initial and value.computed_boxes == 9


def test_bisection_deadline_and_complete_cache_limit_cannot_return_partial_results(api, monkeypatch):
    value = inverse(api)
    checks = []
    def checked(deadline):
        checks.append(deadline)
        if len(checks) == 7:
            raise TimeoutError('inside exact radial bisection')
    with monkeypatch.context() as patch:
        patch.setattr(api, '_check', checked)
        with pytest.raises(TimeoutError, match='inside exact'):
            value.scale(Fraction(1, 100))
        assert value.bisections == 5
    monkeypatch.setattr(api, 'MAX_CACHE_BYTES', 1024)
    with pytest.raises(ValueError, match='cache exceeds'):
        prepare(api, fixture())


_ZERO = {
    'ordinary': ('cad1ca9b444324ca3c0b92825382458774a84f610442d2985680e3476158a0c2', [
        '9bedfe66ef69f589c2c5d07ada2c22d3e67a8a44ae141ff3f9dbca6b79e0d00e', '93b54daefb63fd16d2f56cacacd77ab906b38abc6d3d3c3e7b53471b33461b11',
        '7f0ee6e96361c2cb1a5c2d06e819f801bed7f2ba6d2ee9371d0ed9246ba34470', '7fecfde0ac8cb0878404344f59d61dad6f1249f42ddb42b3946cf21d941c5bbf']),
    'skew_oblique': ('8a71048c195f40119a9d130db61a998eecd3b99748dc04e50abb7829607a621a', [
        '72791c3c3ff698f328a781c2b24941b3caccf5f9b919d2670417ce7437421f52', '73de22c9c367296510718146168a10f6d3232d60b54d139b33b7f1d90dbbadd3',
        '689d7cd7ac721e81bad627e924e7dfeaffaf9b45778d077f964ed8a9fb8bf25d', '8969a377c85c4ac4fa1c300f74eff7ff1dc9632cbc4bbc6ea49bed60e0b94535']),
    'missing_camera': ('bec5bf9c717e924f185bb71f632a2d071fafc5ac70742636a953dca6494f8d26', [
        '496b09845b62931ade1a657efae4280413ffb8139323d9557a485954e730d204', '0f3746c8920bba603bcd9fa21c43e6a2a30d4166f0793bb112f392d7c1d9a978',
        '9f0c2a94154f23706c570ab783faa8a581e062413ca130ef0ee6623340e58d5f', 'f4f0804518badd3dee1f8058b4131e51efb4b117cffaccfb7193633934380f90']),
}


@pytest.mark.parametrize('case', _ZERO)
def test_zero_distortion_metadata_and_all_frames_remain_byte_identical(api, case):
    data = fixture(radial=0.)
    if case == 'skew_oblique':
        data[1]['K'][0][1] = 13.
        for pose in data[1]['frames']:
            pose.update(R=[[1., 0., 0.], [0., -.8, -.6], [0., .6, -.8]], t=[-1., 3., 4.])
        data[2]['surfaces'][0].update(kind='static_support', bounds_xy_m=[-10., -10., 10., 10.])
    result = prepare(api, data, **({'camera_document': None} if case == 'missing_camera' else {}))
    meta, frames = _ZERO[case]
    assert sha256(result._metadata).hexdigest() == meta
    assert [sha256(row).hexdigest() for row in result.iter_frame_bytes()] == frames


def test_representative_unique_radial_boxes_are_bounded_and_queries_are_cached(api, record_property):
    data = fixture(count=600, channels=2, radial=.00047528394168603385)
    for item in data[3]['annotations']:
        channel = 0 if item['channel_id'] == 'left_toe' else 1
        item['pixel'] = [55.+item['frame']*.01+channel*.003, 40.+item['frame']*.003]
    inputs = importlib.import_module('test_source_surface_envelopes').encoded(data)
    started = time.monotonic(); result = api.prepare_source_surface_envelopes(**inputs); prepared = time.monotonic()-started
    started = time.monotonic()
    for index in range(20):
        assert len(result.frame(index)['anchors']) == 2
    queried = time.monotonic()-started
    meta = result.describe()
    assert meta['counts']['retained_anchor_slots'] == meta['counts']['conditional_envelopes'] == 1200
    assert meta['radial_inverse']['computed_boxes'] == 1200
    assert meta['radial_inverse']['bisections'] <= 1200*96
    assert meta['cached_row_bytes']+len(result._metadata) <= 8<<20
    record_property('preparation_seconds', prepared)
    record_property('twenty_cached_queries_seconds', queried)
    record_property('unique_radial_boxes', 1200)
    record_property('exact_bisections', meta['radial_inverse']['bisections'])
    record_property('cached_bytes', meta['cached_row_bytes']+len(result._metadata))
    record_property('exact_input_bytes', sum(len(value) for key, value in inputs.items() if key.endswith('_document'))
                    + sum(map(len, inputs['observation_documents'].values())))
