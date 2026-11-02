"""Read-only verification of this compact evidence inventory, not avatar acceptance."""
import hashlib
import json
from pathlib import Path


def verify(root):
    root=Path(root).resolve()
    manifest=root/'manifest.json'
    if not manifest.is_file() or manifest.is_symlink() or manifest.stat().st_size>512*1024:
        raise ValueError('Invalid compact manifest')
    data=json.loads(manifest.read_text())
    rows=data.get('artifacts')
    if data.get('schema_version')!=1 or not isinstance(rows,list) or not 1<=len(rows)<=256:
        raise ValueError('Invalid bounded inventory')
    expected=set();total=manifest.stat().st_size
    for row in rows:
        relative=Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or row['path'] in expected:
            raise ValueError('Invalid or duplicate evidence path')
        path=root/relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError('Evidence path escaped its bundle')
        if path.stat().st_size!=row['bytes']:
            raise ValueError('Evidence size changed: '+row['path'])
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=row['sha256']:raise ValueError('Evidence hash changed: '+row['path'])
        total+=row['bytes'];expected.add(row['path'])
    actual=set()
    for path in root.rglob('*'):
        if path.is_symlink():raise ValueError('Symlinks are not compact evidence')
        if path.is_file() and path!=manifest:actual.add(str(path.relative_to(root)))
        if len(actual)>256:raise ValueError('Evidence inventory grew beyond its bound')
    if actual!=expected or total>10*1024*1024:raise ValueError('Inventory or10MiB bound changed')
    return {'status':'verified_exact_compact_inventory','files':len(rows)+1,'total_bytes':total,
        'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'complete_avatar_pack':False,'standalone_recreation':False,'provider_calls':0}


if __name__=='__main__':
    print(json.dumps(verify(Path(__file__).parent),sort_keys=True))
