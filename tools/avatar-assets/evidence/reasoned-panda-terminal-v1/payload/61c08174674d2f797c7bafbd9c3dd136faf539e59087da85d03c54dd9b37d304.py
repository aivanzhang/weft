"""Pixel/material observations from a fixed-pose visibility ablation, not a quality score."""
from pathlib import Path
import hashlib,json,struct,sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from avatar_assets.safety import read_json,write_json,sha256,artifacts
out=Path(__file__).resolve().parent;prepared=read_json(out/'prepared.json');native=read_json(out/'native-result.json')
job=ROOT/'outputs/studio-d8a680ce6580429f9fdd016f/model-0'
paths={'with_coat':out/'with-coat.png','body_only':out/'body-only.png','original_16_samples':Path(prepared['baseline_image']['path'])}
images={}
for name,path in paths.items():
 with Image.open(path) as image:
  assert image.size==(720,720);images[name]=np.asarray(image.convert('RGB'),dtype=np.int16)

def differences(a,b,rect=None):
 if rect is not None:
  x0,y0,x1,y1=rect;a=a[y0:y1,x0:x1];b=b[y0:y1,x0:x1]
 d=np.abs(a-b);maximum=d.max(axis=2)
 return {'pixels':int(maximum.size),'mean_absolute_rgb_difference_255':float(d.mean()),
         'maximum_channel_difference_255':int(maximum.max()),'pixels_over_8_levels':int((maximum>8).sum()),
         'fraction_within_2_levels':float((maximum<=2).mean())}
rois={'viewer_left_arm_facets':[207,260,316,366],'lower_cheek_flecks':[302,269,355,303],
      'viewer_left_chest_border':[282,348,338,556],'viewer_left_thigh':[211,425,286,604]}
metrics={'coat_hidden_vs_visible':differences(images['with_coat'],images['body_only']),
 'eight_vs_original_sixteen_sample_control':differences(images['with_coat'],images['original_16_samples']),
 'regions':{name:{'rectangle_xyxy':rect,**differences(images['with_coat'],images['body_only'],rect)} for name,rect in rois.items()}}
model=Path(prepared['model']['path']);raw=model.read_bytes();length=struct.unpack_from('<I',raw,12)[0];doc=json.loads(raw[20:20+length])
primitives=[]
for mesh in doc['meshes']:
 for primitive in mesh['primitives']:
  primitives.append({'mesh':mesh.get('name'),'material':doc['materials'][primitive['material']],
   'attributes':{name:{key:doc['accessors'][index][key] for key in ('componentType','count','type')} for name,index in primitive['attributes'].items()}})
finish=read_json(job/'finish-report.json');attrs=read_json(job/'render-attributes.json')
state=next(s for s in finish['states'] if s['id']==prepared['state']);witnesses=[x for x in state['deformation_outliers'] if abs(x['second']-prepared['second'])<1e-8]
worst=max(witnesses,key=lambda x:x['excess_length'])
coat=[{'cards':b['cards'],'length_range_m':b['root_summary']['length_range_m'],'width_range_m':b['root_summary']['width_range_m'],
       'eligible_area_m2':b['root_summary']['eligible_area_m2'],'coat_node':b['root_summary']['binding_archive']['export_correspondence']['coat_node']} for b in finish['portable_fur']]
maximum_length=max(b['length_range_m'][1] for b in coat)
assert sha256(model)==prepared['model']['sha256'] and sha256(Path(prepared['scene']['path']))==prepared['scene']['sha256']
assert sha256(Path(prepared['finish_report']['path']))==prepared['finish_report']['sha256']
write_json(out/'result.json',{'schema_version':1,'status':'diagnostic_completed','accepted_delivery':False,
 'finding':'Large angular arm/cheek/chest body artifacts remain visible with both portable coat nodes hidden; current short cards are not their sole cause.',
 'interpretation':'The retained large body-edge stretch and arm-versus-torso influence discontinuity support a body-deformation contribution. This ablation alone does not separate every shading fleck from normal-map response.',
 'changed_variable':'Only hide_render of exact coat nodes from manifest-bound export correspondence',
 'pose':{'id':prepared['state'],'seconds':prepared['second'],'frame':native['frame']},
 'pixel_comparison':metrics,'body_positions_normals_camera_lights_unchanged':native['same_body_normals_positions_camera_lights'],
 'coat_geometry':coat,'actual_material_attribute_metadata':primitives,
 'normal_validation':{'path':str(job/'render-attributes.json'),'sha256':sha256(job/'render-attributes.json'),'status':attrs['status'],
                      'primitives':attrs['primitives'],'tangent_uv_derivation_verified':attrs['tangent_uv_derivation_verified']},
 'body_stretch_witness':{'path':str(job/'finish-report.json'),'sha256':sha256(job/'finish-report.json'),
   'field':'states[id=dance].deformation_outliers','scope':'Retained sampled body-edge witness, not exhaustive or distinct-physical-point coverage',
   'witness':worst,'excess_to_maximum_coat_length_ratio':worst['excess_length']/maximum_length},
 'source_bytes_unchanged':True,'source_model_sha256':prepared['model']['sha256'],'source_scene_sha256':prepared['scene']['sha256'],
 'limitations':['One2.0-second pose; EEVEE8samples for both controls, compared with original16sample review.',
 'No geometry, normal, material, skin, image or action edits; no accepted-asset promotion.',
 'Numeric root attachment and gross-spike gate passing do not establish smooth body deformation.',
 'Body-only comparison does not determine whether all residual marks originate in geometry, imported smooth normals, normal-map texture or lighting.'],
 'next_safe_action':'Use these body-only images and exact shoulder influence witness in repair feedback; do not treat further blanket fur reduction as the demonstrated fix.'})
(out/'README.md').write_text('''# Exported body versus portable coat

At dance 2.0 s, two 720 px / 8-sample EEVEE renders use the same saved exported scene, action, camera and lighting. Only the two manifest-identified portable-coat nodes change visibility. Body positions and corner-normal hashes remain identical.

**The large angular arm, cheek and chest artifacts remain in `body-only.png`.** The current short fur cards therefore are not the sole cause. The model's retained body diagnostic also records a 28.297 mm edge stretched to 86.019 mm at this pose (+57.722 mm, 3.040×): adjacent points have sharply different arm-versus-spine ownership. The gross-spike threshold did not reject that ratio; this is not proof of acceptable visual deformation. The longest recorded coat length is 5.022 mm.

`result.json` preserves exact material/attribute routes, the original body-edge witness, image-region differences and source hashes. The body still uses its original UV0 normal map (strength0.3) and metallic/roughness texture. Coat materials have no normal map, metallic0 and roughness0.9. Finite/unit/orthogonal frame validation does not establish smooth skin deformation or correct normal-map appearance.

This isolates coat visibility, not every body shading cause. It is one-pose diagnostic evidence, not a new accepted model, a complete body-skin verdict, or a reason to weaken the dance. No source asset was modified and no provider call occurred. The first sandboxed attempt in `d8-coat-body-ablation-v1` crashed during Blender startup before rendering; the separately authorized environment change is recorded in `prepared.json`. Both attempts remain intact.
''')
write_json(out/'manifest.json',{'schema_version':1,'kind':'exported_coat_visibility_ablation','status':'diagnostic_completed','artifacts':artifacts(out)})
print(json.dumps({'manifest_sha256':sha256(out/'manifest.json'),'pixel_metrics':metrics,'maximum_coat_m':maximum_length,
 'edge_excess_m':worst['excess_length'],'source_unchanged':True}))
