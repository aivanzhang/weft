"""Check completed staged stills and their measured native-resource receipts."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from avatar_assets.safety import require,sha256,write_json
prep=ROOT/'outputs/panda-staged-fringe-preparation-v1';root=ROOT/'outputs/panda-staged-fringe-v1'
plan=json.loads((root/'prepared-native-scene.json').read_text());config=json.loads((prep/'prepared.json').read_text())
require(len(plan['tasks'])==13 and len(list(prep.glob('job-*-receipt.json')))==14,'Incomplete native stages')
rows=[];jobs=[];pose_pixels=[];pairs=[]
for row in config['snapshots']:require(sha256(prep/row['path'])==row['sha256'],'Snapshot changed')
for row in config['inputs']:require(sha256(ROOT/row['path'])==row['sha256'],'Input changed')
require(sha256(root/plan['scene'])==plan['scene_sha256'],'Editable source changed')
for path in sorted(prep.glob('job-*-receipt.json')):
 receipt=json.loads(path.read_text());job=receipt['job'];jobs.append(job)
 require(job['exit_code']==0 and job['process_exit_verified'] and not job['failure']
         and receipt['frozen_inputs_unchanged'],'Native job did not finish cleanly')
 require(job['rss_samples']>0 and job['sampled_peak_rss_bytes']<=2048*1024**2,'Unproven or exceeded memory bound')
for index,task in enumerate(plan['tasks']):
 folder=root/'stills'/str(index).zfill(3);path=folder/'image.png';receipt=json.loads((folder/'render-report.json').read_text())
 require(receipt['image_sha256']==sha256(path) and receipt['plan_sha256']==sha256(root/'prepared-native-scene.json')
         and receipt['scene_sha256']==plan['scene_sha256'] and receipt['task_index']==index
         and receipt['max_transform_error']<=2e-6,'Staged image custody or transforms differ')
 with Image.open(path) as image:
  require(image.format=='PNG' and image.size==(720,720) and image.mode=='RGB','Wrong image format')
  image.load();pixels=np.array(image)
 require(receipt['resolution']==task['resolution'] and receipt['samples']==task['samples']
         and receipt['denoise']==task['denoise'],'Render controls changed')
 row={'task_index':index,'target':task['path'],'path':str(path.relative_to(ROOT)),'sha256':sha256(path),
      'pixel_sha256':hashlib.sha256(pixels.tobytes()).hexdigest(),'frame':task['frame'],
      'max_transform_error':receipt['max_transform_error'],'samples':receipt['samples'],
      'denoise':receipt['denoise'],'format':'RGB PNG','dimensions':[720,720]}
 rows.append(row)
 if task['action'] is not None:pose_pixels.append(row['pixel_sha256'])
 old=ROOT/'outputs/panda-portable-fringe-finish-v2'/task['path']
 if old.is_file():
  with Image.open(old)as image:image.load();previous=np.array(image)
  require(previous.shape==pixels.shape,'Prior image shape differs')
  difference=np.abs(previous.astype(np.int16)-pixels.astype(np.int16));maximum=difference.max(axis=2)
  pairs.append({'old_path':str(old.relative_to(ROOT)),'old_sha256':sha256(old),'new_sha256':sha256(path),
   'identical_pixels':bool(not difference.any()),'mean_channel_error_8bit':float(difference.mean()),
   'max_channel_error_8bit':int(difference.max()),'pixels_changed':int((maximum>0).sum()),
   'pixels_error_gt_10':int((maximum>10).sum()),'pixel_max_error_p99':float(np.quantile(maximum,.99))})
require(len(pose_pixels)==8 and len(set(pose_pixels))>=6,'Motion samples are missing or repeated stills')
result={'schema_version':1,'status':'staged_native_stills_complete_not_avatar_acceptance','rendered_stills':len(rows),
 'sampled_action_stills':len(pose_pixels),'distinct_action_pixel_hashes':len(set(pose_pixels)),
 'aggregate_native_seconds':sum(j['elapsed_seconds']for j in jobs),
 'max_sampled_child_rss_bytes':max(j['sampled_peak_rss_bytes']for j in jobs),
 'all_owned_children_exited':True,'new_generation_calls':0,'quality_approved':False,'loop_verified':False,
 'motion_scope':'Eight authored dance_flight sample stills, not a full video or physical motion validation',
 'source_scene_sha256':plan['scene_sha256'],'prepared_plan_sha256':sha256(root/'prepared-native-scene.json'),
 'render_settings_unchanged':True,'images':rows,'prior_native_comparisons':pairs,
 'appearance_scope':'Pixel differences reported without an invented acceptance tolerance; visual review is separate.'}
write_json(HERE/'verification.json',result)
print(json.dumps({k:result[k]for k in ('status','rendered_stills','distinct_action_pixel_hashes','aggregate_native_seconds','max_sampled_child_rss_bytes')}))
