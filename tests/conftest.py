import pytest

from weft.world import World


@pytest.fixture
def world():
    return World(x0=0.2, v0=0.6, drag=0.1, radius=0.06, support_slope=0.0,
                 support_intercept=0.4, width_px=640, height_px=360, source_frame=0).validate()
