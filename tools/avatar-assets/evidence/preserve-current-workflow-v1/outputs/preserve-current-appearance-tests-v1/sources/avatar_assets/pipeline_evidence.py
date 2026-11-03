"""Full-size visual evidence from native poses, exported skin motion and encoded media."""

from pathlib import Path
import json
import math
import re
import shutil
import sys

import av
from PIL import Image, ImageDraw

from avatar_assets.safety import artifacts, contained, local_file, new_directory, read_json, require, sha256, slug, write_json

TILE = 720
LABEL = 28
VIEWS = ('front', 'side', 'back', 'three_quarter')


def select_indices(count):
    require(type(count) is int and 2 <= count <= 480, 'Unbounded or static evidence sequence')
    return sorted({round(i*(count-1)/3) for i in range(4)})


def decoded_poses(path, expected, indices=None):
    """Decode once with strict frame bounds; never substitute source PNGs for delivered video."""
    path = local_file(path, 64*1024**2)
    count = expected['frames']
    require(type(count) is int and 2 <= count <= 480 and expected['width']*expected['height'] <= 1024**2,
            'Video evidence exceeds decode bounds')
    selected = select_indices(count) if indices is None else indices
    require(isinstance(selected,list) and 1<=len(selected)<=4 and len(set(selected))==len(selected)
            and all(type(i) is int and 0<=i<count for i in selected), 'Invalid bounded video evidence indices')
    wanted, frames = set(selected), {}
    with av.open(str(path)) as container:
        require(len(container.streams.video) == 1 and not container.streams.audio, 'Expected one silent video stream')
        stream = container.streams.video[0]
        require((stream.width, stream.height) == (expected['width'], expected['height']), 'Video dimensions changed')
        decoded = 0
        for index, frame in enumerate(container.decode(stream)):
            require(index < count and frame.width*frame.height <= 1024**2, 'Video frame bound changed')
            if index in wanted:
                frames[index] = frame.to_image().convert('RGB')
            decoded += 1
    require(decoded == count and set(frames) == wanted, 'Video frame count changed')
    return [(index, frames[index]) for index in sorted(frames)]


def _sampling_inputs(pack, source):
    """Require complete finalized custody for pipeline jobs; standalone packs keep explicit fallback."""
    match=re.fullmatch(r'model-([0-2])',pack.name)
    if match is None:return None
    attempt=match.group(1);parent=pack.parent
    def read(path,maximum=1024**2):
        return read_json(source(local_file(path,maximum)))
    model=read(pack/'manifest.json',4*1024**2)
    model_rows={row['path']:row for row in model.get('artifacts',[])}
    def bound(folder,row,maximum):
        require(isinstance(row,dict) and set(row)=={'path','bytes','sha256'} and isinstance(row['path'],str)
                and not Path(row['path']).is_absolute() and '..' not in Path(row['path']).parts,
                'Motion sampling artifact path is invalid')
        path=contained(folder,row['path']);local_file(path,maximum)
        require(path.stat().st_size==row['bytes'] and sha256(source(path))==row['sha256'],
                'Motion sampling finalized artifact changed')
        return path
    for name,maximum in (('model.glb',64*1024**2),('sources/finish-design.json',1024**2)):
        require(name in model_rows,'Motion sampling model/config custody missing')
        bound(pack,model_rows[name],maximum)
    config=read(pack/'sources/finish-design.json')
    require(isinstance(model.get('source_sha256'),str) and re.fullmatch('[0-9a-f]{64}',model['source_sha256'])
            and model['source_sha256']==config.get('source_sha256') and model.get('status')=='needs_visual_review',
            'Motion sampling model/source differs')
    audit_folder=parent/f'action-audit-{attempt}';manifest=read(audit_folder/'manifest.json')
    rows=manifest.get('artifacts');require(isinstance(rows,list) and 1<=len(rows)<=32,'Motion audit inventory exceeds bound')
    require(len({row.get('path') for row in rows})==len(rows),'Motion audit inventory has duplicate paths')
    total=0
    for row in rows:
        total+=row.get('bytes',0);require(total<=16*1024**2,'Motion audit receipts exceed16MiB')
        bound(audit_folder,row,4*1024**2)
    require({'report.json','intent.json','job.json'}<={row['path'] for row in rows},'Motion audit custody is incomplete')
    audit=read(audit_folder/'report.json',4*1024**2);intent=read(audit_folder/'intent.json');job=read(audit_folder/'job.json')
    model_hash=model_rows['model.glb']['sha256']
    require(audit.get('passed') is True and audit.get('source_unchanged') is True
            and audit.get('model_sha256')==manifest.get('model_sha256')==intent.get('source_sha256')==model_hash
            and audit.get('intent_sha256')==sha256(audit_folder/'intent.json')
            and job.get('process_exit_verified') is True and not job.get('manual_stop_required')
            and job.get('exit_code')==0,'Motion sampling audit/model or process custody differs')
    return {'config':config,'audit':audit,'intent':intent,'source_hashes':{
        'model_manifest':sha256(pack/'manifest.json'),'model_glb':model_hash,
        'authored_config':model_rows['sources/finish-design.json']['sha256'],
        'action_audit_manifest':sha256(audit_folder/'manifest.json'),
        'action_audit_report':sha256(audit_folder/'report.json'),'action_intent':sha256(audit_folder/'intent.json')}}


def motion_samples(state_id,clip,context):
    """Select observable control extrema without interpreting action prose or approving meaning."""
    expected=clip['inspection'];count=expected['frames'];fallback=select_indices(count)
    fps=expected.get('fps');duration=expected.get('duration_seconds')
    require(type(fps) in (int,float) and math.isfinite(fps) and fps==24
            and type(duration) in (int,float) and math.isfinite(duration) and duration>0,
            'Motion sampling requires bounded24fps timing')
    result={'state_id':state_id,'policy':'bound-control-extrema-v1','semantic_approval':False,
        'foot_clearance':'unverified; role-root rise is not foot-ground clearance or airtime',
        'indices':fallback,'selection':'uniform_fallback','reason':'No finalized pipeline motion context',
        'omitted_candidates':[],'selection_limit':4,
        'candidate_scope':'One strongest authored key per relevant control track; equal peaks use earliest key. Not all motion extrema are shown.'}
    if context is None:
        result['frames']=[{'index':i,'video_second':i/fps,'rationale':'uniform_fallback'} for i in fallback]
        return result
    config,audit,intent=context['config'],context['audit'],context['intent']
    def unique(rows,label):
        found=[row for row in rows if row.get('id')==state_id]
        require(len(found)==1,'Motion sampling '+label+' state differs');return found[0]
    measured=unique(audit['states'],'audit');planned=unique(intent['states'],'intent')
    authored=config.get('authored_actions') or []
    require(isinstance(authored,list) and len(authored)<=8,'Motion sampling authored action bound')
    action=unique(authored,'authored') if authored else None
    if action is None or action.get('interpolation')!='LINEAR':
        result.update(source_hashes=context['source_hashes'],
            reason='No bounded linear authored controls; uniform samples do not infer control extrema or action meaning',
            frames=[{'index':i,'video_second':i/fps,'rationale':'uniform_fallback'} for i in fallback])
        return result
    keys=action['keys']
    require(isinstance(keys,list) and 2<=len(keys)<=64,'Motion sampling authored key bound')
    times=[row['second'] for row in keys]
    require(all(type(t) in (int,float) and math.isfinite(t) and t>=0 for t in times)
            and all(a<b for a,b in zip(times,times[1:])),'Motion sampling authored times are invalid')
    start,end=times[0],times[-1];span=end-start
    require(0<span<=20 and abs(clip.get('source_clip_seconds',span)-span)<1e-5
            and abs(duration-count/fps)<1/fps,'Motion sampling source/video timing differs')
    def vector(row,field,bone):
        value=row.get(field,{}).get(bone,[0,0,0])
        require(isinstance(value,list) and len(value)==3 and all(type(v) in (int,float) and math.isfinite(v) for v in value),
                'Invalid authored motion control vector')
        return value
    candidates=[];observations=[]
    requirements=planned.get('requirements',[])
    require(isinstance(requirements,list) and len(requirements)<=8,'Motion sampling requirement bound')
    role_details=measured['role_details'];metrics=measured['evidence']['roles']
    require(len(role_details)<=8 and set(role_details)==set(metrics),'Motion sampling audited roles differ')
    for role,detail in role_details.items():
        kinds={row['kind'] for row in requirements if row.get('role')==role}
        root=detail['root_bone'];values=[vector(row,'translations_m',root) for row in keys]
        shifts=[[v-axis for v,axis in zip(value,values[0])] for value in values]
        vertical=[shift[2] for shift in shifts];norms=[math.sqrt(sum(v*v for v in shift)) for shift in shifts]
        recorded=metrics[role]
        for field in ('max_vertical_rise_m','max_travel_m','scale_m'):
            require(type(recorded[field]) in (int,float) and math.isfinite(recorded[field]) and recorded[field]>=0,
                    'Invalid measured root motion summary')
        observations.append({'role':role,'root_bone':root,'requirement_kinds':sorted(kinds),
            'authored_root_vertical_rise_m':max(0,max(vertical)),'authored_root_translation_excursion_m':max(norms),
            'measured_role_root_vertical_rise_m':recorded['max_vertical_rise_m'],
            'measured_role_root_travel_m':recorded['max_travel_m'],
            'measurement_space':'exported glTF world meters; summary, not a measured peak timestamp',
            'authored_space':'Blender Z-up meter controls; local offsets follow animated parents'})
        translation_kinds=kinds & {'travel','takeoff','landing','rhythm'}
        if translation_kinds and max(norms)>1e-8:
            # Positive measured rise keeps vertical cues ahead of an out-of-phase horizontal maximum.
            vertical_cue=max(vertical)>1e-8 and recorded['max_vertical_rise_m']>1e-8
            scores=vertical if vertical_cue else norms;peak=max(range(len(scores)),key=lambda i:scores[i])
            candidates.append({'priority':0 if vertical_cue else 1,'strength':scores[peak]/max(recorded['scale_m'],1e-8),
                'role':role,'bone':root,'control':'authored_vertical_translation' if vertical_cue else 'authored_translation_excursion',
                'key_index':peak,'second':times[peak],'magnitude':scores[peak],'units':'meters',
                'recovery_scores':norms})
            if vertical_cue:
                alternate=max(range(len(norms)),key=lambda i:norms[i])
                if alternate!=peak:candidates.append({'priority':1,'strength':norms[alternate]/max(recorded['scale_m'],1e-8),
                    'role':role,'bone':root,'control':'authored_translation_excursion','key_index':alternate,
                    'second':times[alternate],'magnitude':norms[alternate],'units':'meters','recovery_scores':norms})
        if kinds & {'articulation','rhythm'}:
            require(len(detail['bones'])<=64,'Motion sampling role bone bound')
            for bone in detail['bones']:
                rotations=[vector(row,'rotations',bone) for row in keys]
                scores=[math.sqrt(sum((v-b)**2 for v,b in zip(row,rotations[0]))) for row in rotations]
                if max(scores)>1e-6:
                    peak=max(range(len(scores)),key=lambda i:scores[i])
                    candidates.append({'priority':2,'strength':scores[peak]/180,'role':role,'bone':bone,
                        'control':'authored_euler_control_excursion','key_index':peak,'second':times[peak],
                        'magnitude':scores[peak],'units':'degrees; authored Euler components, not measured angular distance',
                        'recovery_scores':scores})
    require(len(candidates)<=528,'Motion candidate bound exceeded')
    result.update(source_hashes=context['source_hashes'],observations=observations,
        source_start_second=start,source_duration_seconds=span,video_duration_seconds=duration,
        timing_map='video_second=(source_second-source_start_second)*video_duration/source_duration')
    if not candidates:
        result['reason']='No nonzero relevant authored control extremum; uniform samples are not action proof'
        result['frames']=[{'index':i,'video_second':i/fps,'source_second':start+i/fps*span/duration,
                           'rationale':'uniform_fallback'} for i in fallback];return result
    candidates.sort(key=lambda row:(row['priority'],-row['strength'],row['second'],row['role'],row['bone']))
    chosen=candidates[0];peak=chosen['key_index']
    later=list(range(peak+1,len(times)))
    recovery=min(later,key=lambda i:(chosen['recovery_scores'][i],times[i])) if later else len(times)-1
    def index(second):return max(0,min(count-1,round((second-start)/span*duration*fps)))
    labels={0:['baseline'],count-1:['delivered_final_frame']}
    for second,label in ((chosen['second'],'authored_extremum'),(times[recovery],'authored_recovery_nearest_baseline')):
        labels.setdefault(index(second),[]).append(label)
    for i in fallback:
        if len(labels)>=min(4,count):break
        labels.setdefault(i,['uniform_fill'])
    selected=sorted(labels)
    def public(row):return {k:v for k,v in row.items() if k not in ('recovery_scores','strength','priority','key_index')}
    result.update(indices=selected,selection='baseline_extremum_recovery_end',reason='Audited-role-bound authored control extrema; not semantic approval',
        primary_extremum=public(chosen),omitted_candidates=[public(row) for row in candidates[1:17]],
        omitted_candidate_count=len(candidates)-1,omitted_candidates_truncated=len(candidates)>17,
        frames=[{'index':i,'video_second':i/fps,'source_second':start+i/fps*span/duration,
                 'rationale':labels[i]} for i in selected])
    return result


def _read_image(path):
    path = local_file(path, 8*1024**2)
    with Image.open(path) as image:
        require(image.width*image.height <= 16_777_216 and getattr(image, 'n_frames', 1) == 1,
                'Visual evidence exceeds image bounds')
        return image.convert('RGB')


def _tile(sheet, image, x, y, label):
    # Smaller media stays visibly smaller; padding is not additional resolution.
    image = image.copy()
    image.thumbnail((TILE, TILE))
    sheet.paste(image, (x+(TILE-image.width)//2, y+LABEL+(TILE-image.height)//2))
    ImageDraw.Draw(sheet).text((x+8, y+6), label, fill='black')


def _appearance_baseline(request_path,out,source):
    from avatar_assets.appearance_policy import verify_policy
    from avatar_assets.repair_composition import digest
    request_path=local_file(request_path,1024*1024);job=request_path.parent;root=job.parent.parent
    request=read_json(source(request_path));policy=verify_policy(root,job,request);pin=policy['baseline']
    source(job/'inputs/appearance-preference.json')
    parent=root/pin['parent_output'];review_path=source(parent/f"review-{pin['attempt']}.json")
    require(sha256(review_path)==pin['review_sha256'],'Appearance baseline review changed')
    receipt=read_json(review_path);folder=parent/f"final-evidence-{pin['attempt']}"
    manifest_path=source(folder/'manifest.json');manifest=read_json(manifest_path)
    require(manifest.get('model_manifest_sha256')==pin['model_manifest_sha256'],
            'Appearance baseline evidence belongs to another model')
    rows={r['path']:r for r in manifest['artifacts']};selected=[]
    for name in ('native-views','exported-views'):
        candidates=[r for r in manifest['images'] if r.get('id')=='evidence-'+name]
        require(len(candidates)==1,'Appearance baseline multiview missing')
        image=candidates[0];path=contained(folder,image['path']);local_file(path,8*1024*1024)
        require(rows.get(image['path'])=={'path':image['path'],'bytes':path.stat().st_size,'sha256':sha256(path)}
                and image['sha256']==sha256(path)
                and any(r.get('id')==image['id'] and r.get('sha256')==image['sha256'] for r in receipt['evidence']),
                'Appearance baseline image differs from reviewed evidence')
        selected.append((image,source(path)))
    decoded=[_read_image(path) for _,path in selected]
    width=sum(im.width for im in decoded);height=max(im.height for im in decoded)+LABEL
    require(width*height<=8_000_000,'Appearance baseline comparison exceeds8MP')
    sheet=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(sheet);x=0
    for label,image in zip(('Selected baseline: native','Selected baseline: exported'),decoded):
        draw.text((x+8,5),label,fill='black');sheet.paste(image,(x,LABEL));x+=image.width
    target=out/'preserved-appearance-baseline.png';sheet.save(target)
    return target,{'appearance_policy':policy,'appearance_policy_sha256':digest(policy),
        'baseline_evidence_manifest_sha256':sha256(manifest_path),
        'baseline_images':[{'id':row['id'],'sha256':row['sha256']} for row,_ in selected],
        'scope':'Unchanged baseline pixels joined for comparison, not current motion evidence or human approval. Existing native/exported coat differences remain.'}


def build_evidence(pack, exported, media, output, reference=None, *, appearance_request=None):
    pack, exported, media = map(lambda p: Path(p).resolve(), (pack, exported, media))
    out = new_directory(output)
    finish = read_json(pack/'finish-report.json')
    encoded = read_json(media/'manifest.json')
    require(encoded.get('parent_3d_manifest_sha256') == sha256(pack/'manifest.json'),
            'Delivered media belongs to a different model')
    states = finish['states']
    require(1 <= len(states) <= 8 and {s['id'] for s in states} == {s['id'] for s in encoded['clips']},
            'Native and delivered states differ')
    images, coverage, sources = [], [], {}

    def source(path):
        path = local_file(path, 64*1024**2)
        sources[str(path)] = sha256(path)
        return path

    def record(path, kind, state_ids=()):
        with Image.open(path) as im:
            require(im.width*im.height <= 16_777_216, 'Evidence montage exceeds image bounds')
            width, height = im.size
        require(path.stat().st_size <= 8*1024**2, 'Evidence image exceeds agent attachment limit')
        identifier = 'evidence-'+path.stem
        images.append({'id': identifier, 'path': str(path.relative_to(out)), 'kind': kind,
                       'state_ids': list(state_ids), 'width': width, 'height': height, 'sha256': sha256(path)})
        return identifier, width, height

    for representation, root in (('native', pack/'views'), ('exported', exported)):
        sheet = Image.new('RGB', (TILE*2, (TILE+LABEL)*2), 'white')
        for index, view in enumerate(VIEWS):
            name = 'three-quarter' if representation == 'exported' and view == 'three_quarter' else view
            path = source(root/(name+'.png'))
            image = _read_image(path)
            require(min(image.size) >= TILE, 'Multiview review requires actual720px renders')
            _tile(sheet, image, index%2*TILE, index//2*(TILE+LABEL), representation+' / '+view)
        path = out/(representation+'-views.png')
        sheet.save(path)
        identifier, _, _ = record(path, 'multiview')
        coverage.extend({'id': identifier+'-'+view, 'kind': 'view', 'representation': representation,
                         'view': view, 'width': TILE, 'height': TILE} for view in VIEWS)

    sampling_context=_sampling_inputs(pack,source)
    motion_sampling=[]
    batch_size = 1 if len(states) <= 4 else 2
    pose_rows = 2 if batch_size == 1 else 1
    rows_per_state = 2*pose_rows+1
    for first in range(0, len(states), batch_size):
        batch = states[first:first+batch_size]
        sheet = Image.new('RGB', (TILE*4, len(batch)*rows_per_state*(TILE+LABEL)), 'white')
        for row, state in enumerate(batch):
            identifier = slug(state['id'])
            for lane, (representation, root) in enumerate((('native', pack), ('exported', exported))):
                poses = sorted((root/'renders'/identifier).glob('*.png'))
                require(2 <= len(poses) <= 8, 'Missing bounded native/exported motion poses')
                require(len(poses) == len(state['seconds']), 'Pose files differ from actual sample times')
                indices = list(range(len(poses))) if batch_size == 1 else select_indices(len(poses))
                for tile, index in enumerate(indices):
                    image = _read_image(source(poses[index]))
                    require(min(image.size) >= TILE, 'State review requires actual720px poses')
                    second = state['seconds'][index]
                    _tile(sheet, image, tile%4*TILE, (row*rows_per_state+lane*pose_rows+tile//4)*(TILE+LABEL),
                          f'{identifier} / {representation} / {second:.3f}s')
            clip = next(c for c in encoded['clips'] if c['id'] == identifier)
            video = source(contained(media, clip['path']))
            require(sha256(video) == clip['inspection']['sha256'], 'Delivered video changed')
            selection=motion_samples(identifier,clip,sampling_context);motion_sampling.append(selection)
            for column, (index, image) in enumerate(decoded_poses(video, clip['inspection'],selection['indices'])):
                _tile(sheet, image, column*TILE, (row*rows_per_state+2*pose_rows)*(TILE+LABEL),
                      f'{identifier} / delivered video / {index/24:.3f}s / {image.width}px')
        path = out/f'motion-{first//batch_size:02d}.png'
        sheet.save(path)
        identifier, width, height = record(path, 'native_exported_and_delivered_motion', [s['id'] for s in batch])
        images[-1]['motion_sampling']=[row for row in motion_sampling if row['state_id'] in {s['id'] for s in batch}]
        for representation in ('native', 'exported'):
            coverage.append({'id': identifier+'-'+representation, 'kind': 'state', 'representation': representation,
                             'state_ids': [s['id'] for s in batch], 'width': width, 'height': height,
                             'pose_width': TILE, 'pose_height': TILE})

    portrait = source(media/'portrait.webp')
    target = out/'delivered-portrait.webp'
    shutil.copyfile(portrait, target)
    identifier, width, height = record(target, 'delivered_portrait')
    coverage.append({'id': identifier, 'kind': 'portrait', 'representation': 'native', 'width': width, 'height': height})
    if reference:
        path = source(reference)
        target = out/('selected-reference'+path.suffix.lower())
        shutil.copyfile(path, target)
        record(target, 'user_reference')
    if len(states) == 1:
        # Isolated poses retain review detail when a vision service scales large montages.
        state = states[0]
        seconds = state['seconds']
        outliers = state.get('deformation_outliers', [])
        selected = (min(range(len(seconds)), key=lambda i: abs(seconds[i]-max(
            outliers, key=lambda row: row['excess_length'])['second'])) if outliers else len(seconds)//2)
        for representation, root in (('native', pack), ('exported', exported)):
            poses = sorted((root/'renders'/slug(state['id'])).glob('*.png'))
            path = source(poses[selected])
            target = out/('isolated-'+representation+'-'+state['id']+'.png')
            shutil.copyfile(path, target)
            record(target, 'isolated_motion_pose', [state['id']])
            images[-1].update(representation=representation, second=seconds[selected], sample_index=selected,
                             selection='greatest_recorded_edge_stretch' if outliers else 'middle_sequence_sample')
    appearance=None
    if appearance_request is not None:
        require(len(images)<8,'No attachment capacity for required appearance baseline')
        target,appearance=_appearance_baseline(appearance_request,out,source)
        record(target,'appearance_baseline')
    require(len(images) <= 8 and sum((out/i['path']).stat().st_size for i in images) <= 32*1024**2,
            'Final evidence exceeds the bounded agent attachment allowance')
    report = {'schema_version': 1, 'kind': 'full_resolution_review_evidence', 'status': 'prepared_for_review',
              'sources': sources, 'model_manifest_sha256': sha256(pack/'manifest.json'),
              'exported_manifest_sha256': sha256(exported/'manifest.json'),
              'media_manifest_sha256': sha256(media/'manifest.json'), 'images': images, 'coverage': coverage,
              'motion_sampling':motion_sampling,
              **({'appearance_preservation':appearance} if appearance is not None else {}),
              'scope': 'Up to eight actual poses per representation for one to four states; four selected poses for five to eight states. One-state jobs also include two unchanged individual pose images. Up to four decoded delivered video samples per state, selected from bound control extrema or explicit uniform fallback. Not all-frame visual, physics or loop certification.',
              'quality_equivalence_verified': False, 'artifacts': artifacts(out)}
    write_json(out/'manifest.json', report)
    return report


if __name__ == '__main__':
    args = sys.argv[1:]
    flags=[a for a in args if a.startswith('--appearance-request=')]
    require(len(flags)<=1,'Duplicate appearance request')
    args=[a for a in args if not a.startswith('--appearance-request=')]
    require(4 <= len(args) <= 5, 'Use model-pack exported-pack media-pack new-output [reference]')
    build_evidence(*args,appearance_request=flags[0].split('=',1)[1] if flags else None)
