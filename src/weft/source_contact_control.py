"""Pure authored-target adjustments; never assign an engine target or scorer field."""
from dataclasses import dataclass,field
import math
import numpy as np
from weft.human import reference_velocity_index

VARIABLES=('right_shoulder','right_shoulder_lateral','right_shoulder_twist','right_elbow','root_x','root_y','root_z')


@dataclass(frozen=True,slots=True)
class PreparedCorrection:
    times: np.ndarray
    positions: np.ndarray
    slopes: np.ndarray=field(init=False)

    def __post_init__(self):
        for name,value in zip(('times','positions','slopes'),_prepare_arrays(self.times,self.positions)):
            object.__setattr__(self,name,value)


def _readonly(value):
    # Bytes-backed arrays cannot have write access re-enabled by a caller.
    return np.frombuffer(value.tobytes(),dtype=float).reshape(value.shape)


def _prepare_arrays(times,corrections):
    times,corrections=np.asarray(times,dtype=float),np.asarray(corrections,dtype=float)
    if (times.ndim!=1 or not 3<=len(times)<=128 or corrections.shape!=(len(times),7)
            or not np.isfinite(times).all() or not np.isfinite(corrections).all()
            or times[0]<0 or times[-1]>60 or np.any(np.diff(times)<=0) or np.any(np.diff(times)>.25+1e-9)):
        raise ValueError('Correction needs3–128 finite ordered original references and seven coordinates')
    slopes=np.diff(corrections,axis=0)/np.diff(times)[:,None]
    if (np.any(corrections[0]!=0) or np.any(corrections[-1]!=0) or np.max(abs(corrections[:,:4]))>.15+1e-8
            or np.max(np.linalg.norm(corrections[:,4:],axis=1))>.08+1e-8
            or np.max(abs(slopes[:,:4]))>1.5+1e-8 or np.max(np.linalg.norm(slopes[:,4:],axis=1))>.3+1e-8):
        raise ValueError('Correction exceeds original angle/root/endpoint/rate bounds')
    return tuple(map(_readonly,(times,corrections,slopes)))


def prepare_correction(times,corrections):
    return PreparedCorrection(times,corrections)


def correction_at(engine,prepared):
    if not isinstance(prepared,PreparedCorrection):raise ValueError('Prepare the immutable correction once before control')
    times=prepared.times
    if not np.array_equal(times,engine._times) or not math.isfinite(engine.data.time) or not math.isfinite(engine.source_start):
        raise ValueError('Controller correction differs from the original finite source clock')
    stamp=float(np.clip(engine.source_start+engine.data.time,times[0],times[-1]))
    index=min(len(times)-2,max(0,int(np.searchsorted(times,stamp,side='right'))-1))
    position=prepared.positions[index]+prepared.slopes[index]*(stamp-times[index])
    velocity_index=reference_velocity_index(times,stamp,elapsed=engine.data.time,
        timestep=engine.model.opt.timestep,source_start=engine.source_start)
    velocity=prepared.slopes[velocity_index].copy() if velocity_index is not None else np.zeros(7)
    return position,velocity


def authored_target_action(engine,position,velocity):
    position,velocity=np.asarray(position,dtype=float),np.asarray(velocity,dtype=float)
    if position.shape!=(7,) or velocity.shape!=(7,) or not np.isfinite(position).all() or not np.isfinite(velocity).all():
        raise ValueError('Finite seven-variable authored correction required')
    if (np.max(abs(position[:4]))>.15+1e-8 or np.linalg.norm(position[4:])>.08+1e-8
            or np.max(abs(velocity[:4]))>1.5+1e-8 or np.linalg.norm(velocity[4:])>.3+1e-8):
        raise ValueError('Authored correction exceeds its source-qualified envelope')
    ids=[engine.human_action_names.index(name) for name in VARIABLES]
    units=np.array([1.]*4+[engine.scale]*3)
    action=engine.feature_matrix().sum(axis=1)
    action[ids]+=(engine._kp[ids]*position*units+engine._kd[ids]*velocity*units)/engine.human_limits[ids]
    return np.clip(action,-1.,1.)
