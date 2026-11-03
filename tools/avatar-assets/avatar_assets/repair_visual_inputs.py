"""Select authenticated review images; never follow freeform repair paths."""
from pathlib import Path

from avatar_assets.safety import local_file, read_json, require, sha256
from avatar_assets.agent_runner import _image_inputs, MAX_IMAGE_BYTES, MAX_IMAGE_TOTAL_BYTES


def _file(folder, name, maximum=1024**2):
    relative=Path(name)
    require(isinstance(name,str) and not relative.is_absolute() and '..' not in relative.parts,
            'Repair image custody: escaped path')
    path=folder/relative
    require(path.resolve()==path and path.is_relative_to(folder),'Repair image custody: linked or escaped path')
    return local_file(path,maximum)


def author_base_images(job,reference):
    job=Path(job)
    images=[_file(job,'inspection/'+name,MAX_IMAGE_BYTES) for name in
            ('guide-front.png','guide-side.png','three-quarter.png')]
    if reference:
        require(isinstance(reference,dict) and isinstance(reference.get('path'),str)
                and Path(reference['path']).parts[0]=='inputs','Author reference must be a prepared input')
        path=_file(job,reference['path'],MAX_IMAGE_BYTES)
        require(sha256(path)==reference['sha256'] and path.stat().st_size==reference['bytes'],
                'Author reference input changed')
        if sha256(path) not in {sha256(image) for image in images}:images.append(path)
    return images


def author_images(base_images,repair_images):
    require(len(base_images) in (3,4),'Author images need three guides and optional reference')
    capacity=min(5,8-len(base_images));base=_image_inputs(base_images)
    seen={row['sha256'] for row in base};selected=[]
    for value in repair_images:
        path=local_file(value,MAX_IMAGE_BYTES);checksum=sha256(path)
        if checksum not in seen:selected.append(path);seen.add(checksum)
        if len(selected)==capacity:break
    records=_image_inputs([*base_images,*selected])
    return [row['path'] for row in records]


def _pins(rows, *, maximum=1024):
    require(isinstance(rows,list) and len(rows)<=maximum,'Repair image custody: inventory bound')
    result={row['path']:row for row in rows}
    require(len(result)==len(rows),'Repair image custody: duplicate inventory path')
    return result


def _bound(folder,pins,name,maximum=1024**2):
    path=_file(folder,name,maximum);row=pins.get(name)
    require(row=={'path':name,'bytes':path.stat().st_size,'sha256':sha256(path)},
            'Repair image custody: changed artifact '+name)
    return path


def _lineage_inventory(root,child,request,target):
    from avatar_assets.pipeline_continuation import _read_context, MAX_FILES
    for _ in range(8):
        receipt,_=_read_context(child,request)
        parent=root/request['continuation']['parent']
        require(parent.resolve()==parent and parent.parent==root/'outputs',
                'Repair image custody: escaped ancestor')
        pins=_pins(receipt['inventory'],maximum=MAX_FILES)
        parent_request=read_json(_bound(parent,pins,'request.json'))
        require(sha256(parent/'request.json')==receipt['parent_request_sha256'],
                'Repair image custody: ancestor request changed')
        if str(parent.relative_to(root))==target:return parent,pins,parent_request
        require('continuation' in parent_request,'Repair image custody: review is outside bound ancestry')
        for name in ('continuation/receipt.json','continuation/context.json'):
            _bound(parent,pins,name,4*1024**2)
        child,request=parent,parent_request
    raise ValueError('Repair image custody: ancestor depth exceeds8')


def _review_evidence(folder,source_sha,attempt,review,evidence,pins=None):
    require(type(attempt) is int and 0<=attempt<=2,'Repair image custody: attempt bound')
    def load(name,maximum=1024**2):
        return read_json(_bound(folder,pins,name,maximum) if pins is not None else _file(folder,name,maximum))
    receipt=load(f'review-{attempt}.json');model=load(f'model-{attempt}/manifest.json',4*1024**2)
    model_sha=sha256(folder/f'model-{attempt}/manifest.json')
    require(receipt['review']==review and receipt['evidence']==evidence and review.get('verdict')=='repair'
            and receipt['model_manifest_sha256']==model_sha and model.get('source_sha256')==source_sha,
            'Repair image custody: review/source/model mismatch')
    paid=load(f'agents/reviewer-{attempt}/receipt.json')
    answer_path=(f'agents/reviewer-{attempt}/answer.json')
    answer=load(answer_path);paid_request=load(f'agents/reviewer-{attempt}/request.json')
    from avatar_assets.pipeline_prompts import decode_typed_review
    canonical=decode_typed_review(answer) if answer.get('schema_version')==2 else answer
    require(paid.get('status')=='succeeded' and paid.get('answer_sha256')==sha256(folder/answer_path)
            and canonical==review,'Repair image custody: paid answer mismatch')
    return _image_evidence(folder,attempt,model_sha,evidence,paid_request,pins)


def _image_evidence(folder,attempt,model_sha,evidence,paid_request,pins):
    def load(name,maximum=1024**2):
        return read_json(_bound(folder,pins,name,maximum) if pins is not None else _file(folder,name,maximum))
    proof=load(f'final-evidence-{attempt}/manifest.json',4*1024**2)
    require(proof['model_manifest_sha256']==model_sha,'Repair image custody: image model mismatch')
    expected=[{**row,'path':str(folder/f'final-evidence-{attempt}'/row['path'])} for row in proof['images']]
    require(evidence==expected and len(evidence)<=8,'Repair image custody: critic evidence changed')
    image_pins=_pins(proof['artifacts']);records=[]
    for image in evidence:
        path=Path(image['path']);name=str(path.relative_to(folder/f'final-evidence-{attempt}'))
        path=_bound(folder/f'final-evidence-{attempt}',image_pins,name,MAX_IMAGE_BYTES)
        relative=str(path.relative_to(folder))
        if pins is not None:_bound(folder,pins,relative,MAX_IMAGE_BYTES)
        require(sha256(path)==image['sha256'],'Repair image custody: image hash mismatch')
        records.append({**image,'bytes':path.stat().st_size})
    require(paid_request.get('images')==[{'bytes':r['bytes'],'sha256':r['sha256'],'suffix':Path(r['path']).suffix}
                                        for r in records],'Repair image custody: paid images differ')
    return records


def _selected(review,records,base_images):
    cited=set()
    for row in review.get('checks',[]):
        if row.get('passed') is False:cited.update(row.get('evidence_ids',[]))
    for row in review.get('state_reviews',[]):
        if row.get('meaning_passed') is False or row.get('motion_passed') is False:cited.update(row.get('evidence_ids',[]))
        for requirement in row.get('requirement_reviews',[]):
            if requirement.get('passed') is False:cited.update(requirement.get('evidence_ids',[]))
    def rank(row):
        name=row['id']
        priority=(0 if name=='evidence-exported-views' else 1 if name.startswith('evidence-isolated-exported-')
                  else 2 if name=='evidence-native-views' else 3 if name.startswith('evidence-isolated-native-')
                  else 4 if name.startswith('evidence-motion-') else 5)
        return priority,name
    candidates=sorted((row for row in records if row['id'] in cited),key=rank)
    return _select_images(candidates,base_images)


def _select_images(candidates,base_images, *, rejected=False):
    require(len(base_images) in (3,4),'Repair image inputs require three source guides and optional reference')
    capacity=min(5,8-len(base_images))
    base=_image_inputs(base_images)
    chosen=[];seen={row['sha256'] for row in base}
    for row in candidates:
        if row['sha256'] not in seen:chosen.append(row);seen.add(row['sha256'])
        if len(chosen)==capacity:break
    if not chosen:return {'status':'unavailable','reason':'No critic-failure-cited image evidence; no freeform path was loaded.','images':[]}
    inputs=_image_inputs([*base_images,*[Path(row['path']) for row in chosen]])
    require(sum(row['bytes'] for row in inputs)<=MAX_IMAGE_TOTAL_BYTES,'Repair images exceed total byte bound')
    motion=sum(row['id'].startswith(('evidence-motion-','evidence-isolated-')) for row in chosen)
    result={'status':'available','reason':'Bound critic failure images; static-only evidence does not establish motion.' if not motion
            else 'Bound critic failure images, including sampled pose or motion evidence.',
            'images':chosen,'motion_images':motion,'total_agent_images':len(inputs),
            'total_agent_image_bytes':sum(row['bytes'] for row in inputs)}
    if rejected:
        result['reason']='Authenticated images from a rejected review; independently reassess them. Invalid votes are not selection criteria or approval.'
        result['review_status']='invalid_untrusted'
    return result


def _from_rejected(root,child,request,summary,base_images):
    require(isinstance(summary,dict) and summary.get('status')=='invalid_untrusted',
            'Repair image custody: rejected review status differs')
    parent,pins,parent_request=_lineage_inventory(root,child,request,summary['parent_output'])
    source_sha=request['reuse_model']['sha256']
    require(parent_request['reuse_model']['sha256']==summary['source_sha256']==source_sha,
            'Repair image custody: rejected source differs')
    attempt=summary['attempt']
    require(type(attempt) is int and 0<=attempt<=2 and summary['role']==f'reviewer-{attempt}',
            'Repair image custody: rejected attempt differs')
    def load(name,maximum=1024**2):return read_json(_bound(parent,pins,name,maximum))
    role=summary['role'];answer_name=f'agents/{role}/answer.json'
    for name,key in ((answer_name,'answer_sha256'),(f'agents/{role}/receipt.json','receipt_sha256'),
                     (f'model-{attempt}/manifest.json','model_manifest_sha256')):
        require(sha256(_bound(parent,pins,name,4*1024**2))==summary[key],
                'Repair image custody: rejected reference changed')
    answer=load(answer_name);paid=load(f'agents/{role}/receipt.json');terminal=load('unsupported.json')
    model=load(f'model-{attempt}/manifest.json',4*1024**2)
    require(paid.get('status')=='succeeded' and paid.get('answer_sha256')==summary['answer_sha256']
            and answer==summary['raw_answer'] and model.get('source_sha256')==source_sha
            and terminal.get('stage')=='reviewer' and terminal.get('failure_kind')=='review_contract'
            and terminal.get('attempt')==attempt and terminal.get('reason')==summary['reason'],
            'Repair image custody: rejected answer/source/terminal differs')
    proof=load(f'final-evidence-{attempt}/manifest.json',4*1024**2)
    evidence=[{**row,'path':str(parent/f'final-evidence-{attempt}'/row['path'])} for row in proof['images']]
    records=_image_evidence(parent,attempt,summary['model_manifest_sha256'],evidence,
                            load(f'agents/{role}/request.json'),pins)
    require(summary['evidence']==[{'id':r['id'],'path':str(Path(r['path']).relative_to(parent)),
                                  'sha256':r['sha256'],'bytes':r['bytes']} for r in records],
            'Repair image custody: rejected evidence differs')
    def rank(row):
        name=row['id']
        return (0 if name=='evidence-exported-views' else 1 if name=='evidence-native-views'
                else 2 if name.startswith('evidence-motion-') else 3 if name.startswith('evidence-isolated-exported-')
                else 4 if name.startswith('evidence-isolated-native-') else 5),name
    # A malformed verdict cannot choose its own evidence or suppress the motion view.
    return _select_images(sorted(records,key=rank),base_images,rejected=True)


def from_continuation(root,child,request,context,base_images):
    """Context must already pass pipeline_continuation.verify; recheck every selected file."""
    from avatar_assets.pipeline_continuation import _read_context
    _,saved_context=_read_context(Path(child),request)
    require(context==saved_context,'Repair image custody: continuation context differs')
    rejected=(context or {}).get('findings',{}).get('rejected_review')
    if rejected is not None:
        return _from_rejected(Path(root),Path(child),request,rejected,base_images)
    history=(context or {}).get('findings',{}).get('historical_visual_reviews',[])
    require(isinstance(history,list) and len(history)<=8,'Repair visual history bound')
    if not history:return {'status':'unavailable','reason':'No authenticated prior visual review.','images':[]}
    summary=history[-1]
    parent,pins,parent_request=_lineage_inventory(Path(root),Path(child),request,summary['parent_output'])
    source_sha=request['reuse_model']['sha256']
    require(parent_request['reuse_model']['sha256']==source_sha,'Repair image custody: ancestor source mismatch')
    attempt=summary['attempt'];role=f'reviewer-{attempt}'
    for name,checksum in ((f'review-{attempt}.json',summary['review_reference']['sha256']),
                          (f'agents/{role}/answer.json',summary['answer_sha256']),
                          (f'agents/{role}/receipt.json',summary['receipt_sha256']),
                          (f'model-{attempt}/manifest.json',summary['model_manifest_sha256'])):
        require(sha256(_bound(parent,pins,name,4*1024**2))==checksum,'Repair image custody: history reference changed')
    receipt=read_json(parent/f'review-{attempt}.json')
    records=_review_evidence(parent,source_sha,attempt,receipt['review'],receipt['evidence'],pins)
    require(summary['evidence']==[{'id':r['id'],'path':str(Path(r['path']).relative_to(parent)),
                                  'sha256':r['sha256'],'bytes':r['bytes']} for r in records],
            'Repair image custody: historical images differ')
    return _selected(receipt['review'],records,base_images)


def from_current_review(job,source_sha,attempt,review,evidence,base_images):
    """Call after validate_review/_bind_review and the saved review receipt, never before."""
    return _selected(review,_review_evidence(Path(job),source_sha,attempt,review,evidence),base_images)
