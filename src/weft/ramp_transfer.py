"""Initialize a repeated ramp trial from its prefix without refitting materials or gravity."""

from copy import deepcopy
import math

import numpy as np

from weft.material_fit import validate_inputs
from weft.spatial_rigid import rotation, validate_scene


def initialize_prefix(scene, observations, projection, *, actor_id, target_id, ramp_id, support_id,
                      initial_frame=8, prefix_end=18, steps_per_frame=2, maximum_error_px=12):
    result = validate_scene(scene)
    contract = validate_inputs(result, observations, projection, initial_frame, steps_per_frame)
    if (type(prefix_end) is not int or prefix_end-initial_frame < 4 or prefix_end-initial_frame > 30
            or contract['last_frame'] != prefix_end or not 0 < maximum_error_px <= 20):
        raise ValueError('Initializer accepts only its fixed complete prefix, never later evaluation rows')
    objects = {obj['id']: obj for obj in result['objects']}
    if len({actor_id,target_id,ramp_id,support_id}) != 4 or not {actor_id,target_id,ramp_id,support_id} <= objects.keys():
        raise ValueError('Initialization needs distinct actor, target, ramp and support identities')
    actor, target, ramp, support = (objects[key] for key in (actor_id,target_id,ramp_id,support_id))
    if any(obj['shape']!='box' for obj in (actor,target,ramp,support)):
        raise ValueError('Ramp-prefix initialization currently requires box geometry')
    if [obj['role'] for obj in (actor,target,ramp,support)] != ['active','passive','static','static']:
        raise ValueError('Ramp-prefix roles differ from the fixed scene')
    orientation = rotation(ramp['quaternion'])
    tangent, normal = orientation[:,0], orientation[:,2]
    if (abs(tangent[1]) > 1e-8 or tangent[0] <= .7 or not .7 < normal[2] <= 1
            or not np.allclose(rotation(actor['quaternion']),orientation,atol=1e-8)
            or not np.allclose(rotation(support['quaternion']),np.eye(3),atol=1e-8)
            or not np.allclose(rotation(target['quaternion']),np.eye(3),atol=1e-8)):
        raise ValueError('Prefix initializer requires the fixed x/z ramp and horizontal target support')
    matrix = contract['matrix'];planar_matrix = matrix[:,[0,2]]
    if np.linalg.cond(planar_matrix)>1e6:
        raise ValueError('Camera cannot invert x/z positions at the fixed lateral depth')
    by_object = {}
    for identifier in (actor_id,target_id):
        rows = sorted([row for row in observations if row['object_id']==identifier and initial_frame<=row['frame']<=prefix_end],key=lambda row:row['frame'])
        if len(rows)!=prefix_end-initial_frame+1 or any(row['center_px'] is None for row in rows):
            raise ValueError('Every opening pose must be observed; no missing-prefix interpolation')
        points = []
        for row in rows:
            x,z = np.linalg.solve(planar_matrix,np.asarray(row['center_px'])-matrix[:,1]*objects[identifier]['position'][1]-matrix[:,3])
            points.append([x,objects[identifier]['position'][1],z])
        by_object[identifier] = (rows,np.asarray(points))
    rows,points = by_object[actor_id]
    elapsed = np.array([row['time_s']-rows[0]['time_s'] for row in rows])
    origin = np.asarray(ramp['position']) + normal*(ramp['size'][2]/2+actor['size'][2]/2+1e-5)
    pair_friction = max(actor['friction'][0],ramp['friction'][0])
    gravity = np.asarray(result['gravity'])
    acceleration = float(gravity@tangent - pair_friction*abs(gravity@normal))
    if acceleration<=0:
        raise ValueError('Fixed materials and gravity do not predict a downhill slide')
    design = np.column_stack([np.ones(len(elapsed)),elapsed])
    distance = (points-origin)@tangent
    fitted,_,rank,_ = np.linalg.lstsq(design,distance-.5*acceleration*elapsed**2,rcond=None)
    if rank!=2 or not 0<=fitted[1]<=4:
        raise ValueError('Opening observations do not support a bounded downhill initial velocity')
    initial = origin+fitted[0]*tangent;initial[1]=actor['position'][1]
    actor['position'],actor['velocity'] = initial.tolist(),(fitted[1]*tangent).tolist()
    predicted = initial+np.outer(elapsed*fitted[1]+.5*acceleration*elapsed**2,tangent)
    actor_pixels = np.column_stack([predicted,np.ones(len(predicted))])@matrix.T
    actor_errors = np.linalg.norm(actor_pixels-np.asarray([row['center_px'] for row in rows]),axis=1)
    target_rows,target_points = by_object[target_id]
    target['position'] = [float(np.mean(target_points[:,0])),target['position'][1],
                          support['position'][2]+support['size'][2]/2+target['size'][2]/2+1e-5]
    target['velocity'] = [0.,0.,0.]
    target_pixel = matrix@np.r_[target['position'],1.]
    target_errors = np.linalg.norm(target_pixel-np.asarray([row['center_px'] for row in target_rows]),axis=1)
    errors = np.r_[actor_errors,target_errors]
    if max(errors)>maximum_error_px:
        raise ValueError('Prefix observations contradict the frozen camera/contact/motion assumptions')
    details = {'scope':'prefix_initialization_only','initial_frame':initial_frame,'prefix_end':prefix_end,
        'input_slots':len(observations),'available_actor_samples':len(rows),'available_target_samples':len(target_rows),
        'source_sha256':projection['source_sha256'],'source_ids':[row['source_id'] for row in observations],
        'fixed_downhill_acceleration':acceleration,'fixed_actor_ramp_friction':pair_friction,
        'initial_distance_along_ramp':float(fitted[0]),'initial_downhill_speed':float(fitted[1]),
        'prefix_max_error_px':float(max(errors)),'prefix_rmse_px':float(np.sqrt(np.mean(errors**2))),
        'later_frames_read':False,'fitted_parameters':['actor initial distance/speed','target initial horizontal position'],
        'fixed_parameters':['geometry','camera','gravity','all friction','all mass','contact solver','lateral offsets'],
        'physical_validation':'unverified'}
    for obj in (actor,target):
        obj['provenance']['repeat_initialization'] = deepcopy(details)
    result['assumptions'].append('Only source-prefix initial position and downhill speed change; camera, geometry, gravity and materials remain fixed from the first trial.')
    return validate_scene(result),details
