from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from avatar_assets.safety import read_json,write_json,sha256,artifacts
out=Path(__file__).resolve().parent;p=read_json(out/'prepared.json');native=read_json(out/'native-result.json')
baseline=Path(p['baseline_body_only']['path']);image=out/'body-no-normal-map.png'
assert sha256(baseline)==p['baseline_body_only']['sha256']
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
assert native['body_normals_geometry_camera_lights_equal_to_baseline'] and native['source_files_unchanged']
write_json(out/'result.json',{'schema_version':1,'status':'diagnostic_completed','accepted_delivery':False,
 'finding':'The large angular arm, cheek and chest artifacts remain when the body normal-map input is disconnected and both portable coats are hidden.',
 'interpretation':'Normal-map shading and current short portable fur are not sufficient explanations. Body deformation/topology and stored mesh normals remain possible contributors; this test does not uniquely distinguish them.',
 'changed_variable':'Only the direct NormalMap output link into the body Principled Normal socket was removed in memory',
 'body_geometry_and_custom_normals_unchanged':True,'camera_lights_pose_unchanged':True,'basecolor_roughness_metallic_unchanged':True,
 'original_file_bytes_unchanged':True,'removed_links':native['removed_links'],'pixel_comparison':metrics,
 'pose':{'id':p['state'],'second':p['second'],'frame':native['frame']},
 'baseline':p['baseline_body_only'],'normal_map_disconnected_image':{'path':str(image),'sha256':sha256(image)},
 'source_model':p['model'],'source_scene':p['scene'],
 'preceding_coat_ablation':{'path':str(ROOT/'outputs/d8-coat-body-ablation-v2/manifest.json'),
                         'sha256':sha256(ROOT/'outputs/d8-coat-body-ablation-v2/manifest.json')},
 'scope':'One fixed-pose shading diagnostic; not a new model or quality/physics verdict',
 'next_safe_action':'Address the bound body-edge/influence evidence or separately prove a stored-normal defect; do not treat blanket fur reduction or normal-map removal as a demonstrated repair.'})
(out/'README.md').write_text('''# Body normal-map ablation

The saved exported model is posed at dance 2.0 s with both portable-coat nodes hidden. One 720 px / 8-sample render disconnects only the direct Normal Map → Principled Normal link. All other shader links and socket values, including basecolor and metallic/roughness routes, remain unchanged. Actual body positions and stored corner-normal hashes, camera, lights and source files match the preceding body-only control exactly.

**Large angular arm, cheek and chest artifacts remain.** The normal-map input is not their sole cause, just as the earlier visibility ablation ruled out the current short fur cards as their sole cause. This does not uniquely attribute every fleck: body deformation/topology and imported custom surface normals still require discrimination. The prior exact-pose body-edge witness remains a concrete deformation lead; this test neither changes thresholds nor weakens the dance.

`result.json` contains whole-image and identical-region pixel differences. Pixel changes quantify the removed shading contribution; they are not an appearance score. No GLB/Blend was saved or modified, no runtime code changed, and no provider call occurred. The native child exited after one diagnostic render.
''')
write_json(out/'manifest.json',{'schema_version':1,'kind':'body_normal_map_ablation','status':'diagnostic_completed','artifacts':artifacts(out)})
print({'manifest_sha256':sha256(out/'manifest.json'),'metrics':metrics})
