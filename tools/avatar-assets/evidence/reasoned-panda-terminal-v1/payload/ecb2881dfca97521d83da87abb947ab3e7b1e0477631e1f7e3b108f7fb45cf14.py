from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from avatar_assets.safety import read_json,write_json,sha256,artifacts
out=Path(__file__).resolve().parent;p=read_json(out/'prepared.json');native=read_json(out/'native-result.json')
baseline=Path(p['baseline_no_normal_map']['path']);image=out/'posed-smooth-normals.png'
assert sha256(baseline)==p['baseline_no_normal_map']['sha256']
arrays=[]
for path in (baseline,image):
 with Image.open(path) as im:
  assert im.size==(720,720);arrays.append(np.asarray(im.convert('RGB'),dtype=np.int16))
def compare(rect=None):
 a,b=arrays
 if rect is not None:
  x0,y0,x1,y1=rect;a=a[y0:y1,x0:x1];b=b[y0:y1,x0:x1]
 d=np.abs(a-b);m=d.max(axis=2)
 return {'pixels':int(m.size),'mean_absolute_rgb_difference_255':float(d.mean()),'maximum_channel_difference_255':int(m.max()),
         'pixels_over_8_levels':int((m>8).sum()),'fraction_within_2_levels':float((m<=2).mean())}
regions={'viewer_left_arm_facets':[207,260,316,366],'lower_cheek_flecks':[302,269,355,303],
         'viewer_left_chest_border':[282,348,338,556],'viewer_left_thigh':[211,425,286,604]}
metrics={'whole_image':compare(),'regions':{name:{'rectangle_xyxy':rect,**compare(rect)} for name,rect in regions.items()}}
assert native['posed_positions_triangles_uvs_materials_camera_lights_equal_to_baseline'] and native['source_files_unchanged']
assert native['body_state']['posed_body_corner_normals_sha256']!=native['original_body_state']['posed_body_corner_normals_sha256']
write_json(out/'result.json',{'schema_version':1,'status':'diagnostic_completed','accepted_delivery':False,
 'finding':'The large angular arm, cheek and chest artifacts remain with fresh smooth normals computed directly from the unchanged posed triangles.',
 'interpretation':'The original stored/skinned normal field is not the sole cause. Body geometry folds and unchanged texture/material patterns remain possible contributors; this static control does not uniquely separate them.',
 'changed_variable':'Only normals and smoothing overrides of an in-memory static posed-body copy; portable coats hidden and normal-map input disconnected in both controls',
 'pose':{'id':p['state'],'second':p['second'],'frame':native['frame']},'normal_method':native['normal_control'],
 'geometry_topology_uv_material_camera_light_preservation':True,'original_files_unchanged':True,
 'pixel_comparison':metrics,'baseline':p['baseline_no_normal_map'],
 'diagnostic_image':{'path':str(image),'sha256':sha256(image)},'source_model':p['model'],'source_scene':p['scene'],
 'normal_field_hashes':{'before':native['original_body_state']['posed_body_corner_normals_sha256'],
                        'after':native['body_state']['posed_body_corner_normals_sha256']},
 'preceding_normal_map_ablation':{'path':str(ROOT/'outputs/d8-body-normal-ablation-v1/manifest.json'),
                                'sha256':sha256(ROOT/'outputs/d8-body-normal-ablation-v1/manifest.json')},
 'limitations':['One static baked pose only; not a corrected exported animation or new deliverable.',
 'Normals accumulate over actual indexed triangles without vertex welding or cross-seam alias guessing.',
 'Basecolor and roughness/metallic textures remain unchanged; painted/baked patterns are not excluded.',
 'Known large posed body-edge stretch remains independent evidence; this control does not establish all visible pixels share one cause.'],
 'next_safe_action':'Use preserved body-only and smooth-normal controls with exact body-edge/weight witnesses in subsequent repair reasoning; no further blind coat reduction is demonstrated.'})
(out/'README.md').write_text('''# Static posed-normal diagnostic

One diagnostic render freezes the exported body at dance 2.0 s, with the portable coats hidden and the normal map disconnected as in the preceding control. The evaluated pose is copied into a temporary static mesh. Its 27,043 positions, 36,124 oriented triangles, UV bytes and material assignments are verified exact before and after. Source custom/sharp overrides are cleared, every face is smooth, and new vertex normals are computed as normalized sums of unit face normals weighted by face area and corner angle. Normal storage is checked against those calculated values.

**The large angular arm, cheek and chest facets remain.** The original stored/skinned normal field is not their sole cause. Actual body folds and unchanged basecolor/roughness/metallic texture patterns remain possible contributors. This does not establish that all residual marks have one cause; the separately retained +57.722 mm body-edge stretch is still concrete deformation evidence.

Camera, lights, pose and image settings match the body-without-normal-map control. `result.json` contains identical-region pixel comparisons and all source/custody links. Smoothing does not weld vertices or infer correspondence across coincident seams. No model or Blend is saved, no source bytes change, and no API call occurs.

The static copy temporarily has no active armature modifier because its coordinates are already evaluated. This is **not an exported-animation fix** and does not establish appearance acceptance. The one 720 px / 8-sample render completed within the unchanged local resource limits; no additional render is requested by this evidence.
''')
write_json(out/'manifest.json',{'schema_version':1,'kind':'static_posed_normal_diagnostic','status':'diagnostic_completed','artifacts':artifacts(out)})
print({'manifest_sha256':sha256(out/'manifest.json'),'metrics':metrics})
