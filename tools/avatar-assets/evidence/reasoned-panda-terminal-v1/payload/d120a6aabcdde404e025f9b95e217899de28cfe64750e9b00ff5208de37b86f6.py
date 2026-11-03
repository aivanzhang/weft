"""Select existing critic-cited images; never generate images or follow freeform repair paths."""
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


def _pins(rows):
    require(isinstance(rows,list) and len(rows)<=1024,'Repair image custody: inventory bound')
    result={row['path']:row for row in rows}
    require(len(result)==len(rows),'Repair image custody: duplicate inventory path')
    return result


def _bound(folder,pins,name,maximum=1024**2):
    path=_file(folder,name,maximum);row=pins.get(name)
    require(row=={'path':name,'bytes':path.stat().st_size,'sha256':sha256(path)},
            'Repair image custody: changed artifact '+name)
    return path


def _lineage_inventory(root,child,request,target):
    from avatar_assets.pipeline_continuation import _read_context
    for _ in range(8):
        receipt,_=_read_context(child,request)
        parent=root/request['continuation']['parent']
        require(parent.resolve()==parent and parent.parent==root/'outputs',
                'Repair image custody: escaped ancestor')
        pins=_pins(receipt['inventory'])
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
    require(paid.get('status')=='succeeded' and paid.get('answer_sha256')==sha256(folder/answer_path)
            and answer==review,'Repair image custody: paid answer mismatch')
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
    chosen=[];seen=set()
    for row in candidates:
        if row['sha256'] not in seen:chosen.append(row);seen.add(row['sha256'])
        if len(chosen)==5:break
    if not chosen:return {'status':'unavailable','reason':'No critic-failure-cited image evidence; no freeform path was loaded.','images':[]}
    require(len(base_images)==3,'Repair image inputs require the three source guides')
    inputs=_image_inputs([*base_images,*[Path(row['path']) for row in chosen]])
    require(sum(row['bytes'] for row in inputs)<=MAX_IMAGE_TOTAL_BYTES,'Repair images exceed total byte bound')
    motion=sum(row['id'].startswith(('evidence-motion-','evidence-isolated-')) for row in chosen)
    return {'status':'available','reason':'Bound critic failure images; static-only evidence does not establish motion.' if not motion
            else 'Bound critic failure images, including sampled pose or motion evidence.',
            'images':chosen,'motion_images':motion,'total_agent_images':len(inputs),
            'total_agent_image_bytes':sum(row['bytes'] for row in inputs)}


def from_continuation(root,child,request,context,base_images):
    """Context must already pass pipeline_continuation.verify; recheck every selected file."""
    from avatar_assets.pipeline_continuation import _read_context
    _,saved_context=_read_context(Path(child),request)
    require(context==saved_context,'Repair image custody: continuation context differs')
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
