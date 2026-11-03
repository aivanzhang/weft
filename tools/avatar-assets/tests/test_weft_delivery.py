"""Bounded derivative packing; native decimation remains a separately recorded local execution."""
from copy import deepcopy
from hashlib import sha256
import io,json,struct,time
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from avatar_assets import weft_delivery as m


def settings():
    return {'asset_id':'explicit-avatar','display_name':'Explicit avatar','rights':{'holder':'Fixture','license':'test','source':'analytic','declaration':'test only'},'provenance':{'scope':'analytic'},'body_roles':{r:m.ROOT_NAME if r=='mobile_root' else r for r in m.ROLES},'scale':2,'orientation_xyzw':[2**-.5,0,0,2**-.5]}


def glb(path,color=True):
    binary=bytearray();d={'asset':{'version':'2.0'},'bufferViews':[],'accessors':[],'materials':[{'normalTexture':{'index':0},'pbrMetallicRoughness':{'baseColorTexture':{'index':0},'metallicFactor':.4,'roughnessFactor':.3}}]}
    def view(b):
        binary.extend(b'\0'*(-len(binary)%4));o=len(binary);binary.extend(b);d['bufferViews'].append({'buffer':0,'byteOffset':o,'byteLength':len(b)});return len(d['bufferViews'])-1
    def a(v,kind,component=5126):
        v=np.asarray(v,dtype='<f4' if component==5126 else '<u2');row={'bufferView':view(v.tobytes()),'componentType':component,'count':len(v),'type':kind}
        if kind=='VEC3':row.update(min=v.min(axis=0).tolist(),max=v.max(axis=0).tolist())
        d['accessors'].append(row);return len(d['accessors'])-1
    attrs={'POSITION':a([[0,0,0],[1,0,0],[0,1,0]],'VEC3'),'NORMAL':a([[0,0,1]]*3,'VEC3'),'TEXCOORD_0':a([[0,0],[1,0],[0,1]],'VEC2'),'JOINTS_0':a([[0,0,0,0]]*3,'VEC4',5123),'WEIGHTS_0':a([[1,0,0,0]]*3,'VEC4')}
    if color:attrs['COLOR_0']=a([[1,.5,0,1]]*3,'VEC4')
    indices=a([[0],[1],[2]],'SCALAR',5123);joints=sorted(m.ROLES-{'mobile_root'});ibm=a([np.eye(4).reshape(16).tolist() for _ in joints],'MAT4')
    image=io.BytesIO();Image.new('RGBA',(2,2),(255,255,255,255)).save(image,format='PNG');d['images']=[{'bufferView':view(image.getvalue()),'mimeType':'image/png'}];d['textures']=[{'source':0}]
    d['nodes']=[{'name':name} for name in joints]+[{'name':'mesh','mesh':0,'skin':0},{'name':'OriginalRoot','children':list(range(len(joints)+1))}]
    d.update(skins=[{'joints':list(range(len(joints))),'inverseBindMatrices':ibm}],scenes=[{'nodes':[len(d['nodes'])-1]}],scene=0,meshes=[{'name':'skin','primitives':[{'attributes':attrs,'indices':indices,'material':0}]}],buffers=[{'byteLength':len(binary)}])
    binary.extend(b'\0'*(-len(binary)%4));j=json.dumps(d,separators=(',',':')).encode();j+=b' '*(-len(j)%4);path.write_bytes(struct.pack('<4sII',b'glTF',2,28+len(j)+len(binary))+struct.pack('<II',len(j),0x4e4f534a)+j+struct.pack('<II',len(binary),0x004e4942)+binary)
    return path


@pytest.mark.parametrize('color',[True,False])
def test_packer_retains_skin_and_explicit_roles_without_consumer_cap_changes(tmp_path,color):
    p=glb(tmp_path/'native.glb',color);before=p.read_bytes();s=settings();original=deepcopy(s)
    mapping,report=m.pack_delivery(p,s,tmp_path/'model.glb',deadline=time.monotonic()+5)
    checked=m._GLB(tmp_path/'model.glb');attrs=checked.doc['meshes'][0]['primitives'][0]['attributes']
    assert set(attrs)=={'POSITION','NORMAL','TEXCOORD_0','JOINTS_0','WEIGHTS_0'}
    assert np.array_equal(checked.accessor(attrs['WEIGHTS_0']),[[1,0,0,0]]*3)
    assert report['vertices']==3 and report['triangles']==1
    assert len(mapping['retargeting']['bindings'])==10 and mapping['retargeting']['scale']==2
    assert checked.doc['nodes'][-1]['name']==m.ROOT_NAME and checked.doc['scenes'][0]['nodes']==[len(checked.doc['nodes'])-1]
    material=checked.doc['materials'][0];assert 'normalTexture' not in material
    assert material['pbrMetallicRoughness']['metallicFactor']==0 and material['pbrMetallicRoughness']['roughnessFactor']==1
    assert 'animations' not in checked.doc and report['animations_copied'] is False
    assert p.read_bytes()==before and s==original
    assert bool(report['primitives'][0]['color_bake']) is color


def test_color_bake_preserves_linear_tint_alpha_and_bounds():
    uv=np.array([[[0,0],[1,0],[0,1]]],float);c=np.array([[[.5,0,1,.5]]*3]);im=np.full((2,2,4),255,np.uint8)
    mapped,raw,receipt=m.bake_vertex_colors(uv,c,im,{},deadline=time.monotonic()+5)
    image=np.asarray(Image.open(io.BytesIO(raw)));assert image.shape==(16,16,4)
    assert image[4,4].tolist()==[188,0,255,128]
    assert np.all((mapped>0)&(mapped<1)) and receipt['tile_pixels']==16


def test_deadline_no_published_output(tmp_path):
    p=glb(tmp_path/'native.glb')
    with pytest.raises(TimeoutError):m.pack_delivery(p,settings(),tmp_path/'no.glb',deadline=time.monotonic()-1)
    assert not (tmp_path/'no.glb').exists()


@pytest.mark.parametrize('change',['role','duplicate','scale','orientation','extra'])
def test_closed_explicit_settings(change):
    s=settings()
    if change=='role':s['body_roles'].pop('torso')
    if change=='duplicate':s['body_roles']['torso']=m.ROOT_NAME
    if change=='scale':s['scale']=10**10000
    if change=='orientation':s['orientation_xyzw']=[0,0,0,0]
    if change=='extra':s['autoguess']=True
    with pytest.raises(ValueError):m.validate_settings(s)


def test_wrong_source_pin_never_starts_blender(tmp_path,monkeypatch):
    p=glb(tmp_path/'native.glb');s=tmp_path/'settings.json';s.write_text(json.dumps(settings()))
    monkeypatch.setattr(m,'run_bounded',lambda *a,**k:pytest.fail('No native process'))
    with pytest.raises(ValueError,match='hash differs'):m.prepare_weft_delivery(p,s,tmp_path/'out',expected_source_sha256='0'*64)


def test_disconnected_cards_are_retained_whole_and_within_budget():
    import importlib.util
    spec=importlib.util.spec_from_file_location('weft_delivery_mesh',m.ROOT/'scripts/weft_delivery_mesh.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    polygons=[[4*i,4*i+1,4*i+2,4*i+3] for i in range(100)]
    plan=module.component_plan(400,polygons,30)
    assert plan['method']=='whole_component_thinning' and plan['triangles']==30
    assert plan['retained_components']==15 and len(plan['keep_vertices'])==60
    kept=set(plan['keep_vertices'])
    assert all(len(kept.intersection(p)) in (0,4) for p in polygons)
    assert module.component_plan(400,polygons,30)==plan
    assert module.component_plan(4,polygons[:1],4)['method']=='connected_collapse'
    with pytest.raises(ValueError):module.component_plan(400,[[0,1,400]],30)


def test_user_command_preparation_writes_exact_two_member_zip(tmp_path,monkeypatch):
    from zipfile import ZipFile
    (tmp_path/'original').mkdir();source=glb(tmp_path/'original/model.glb');raw=source.read_bytes();s=tmp_path/'settings.json';s.write_text(json.dumps(settings()))
    parsed=m._GLB(source);joints=[parsed.doc['nodes'][i]['name'] for i in parsed.doc['skins'][0]['joints']]
    def native(command,log,out,**kwargs):
        assert (out/'weft_delivery_mesh.py').is_file()
        (out/'decimated.glb').write_bytes(raw)
        (out/'native-report.json').write_text(json.dumps({'meshes':[{'name':'mesh'}],'joint_names':joints}))
        return {'exit_code':0,'failure':None}
    monkeypatch.setattr(m,'run_bounded',native)
    out=tmp_path/'derivative';report=m.prepare_weft_delivery(source,s,out,expected_source_sha256=sha256(raw).hexdigest())
    with ZipFile(out/'weft-avatar.zip') as archive:
        assert archive.namelist()==['model.glb','mapping.json']
        assert archive.read('model.glb')==(out/'model.glb').read_bytes()
        assert archive.read('mapping.json')==(out/'mapping.json').read_bytes()
    assert report['zip_sha256']==sha256((out/'weft-avatar.zip').read_bytes()).hexdigest()
    assert source.read_bytes()==raw
