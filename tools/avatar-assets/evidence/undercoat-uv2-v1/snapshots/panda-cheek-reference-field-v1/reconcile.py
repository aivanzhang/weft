"""Small JSON-only receipt reconciliation; no source mesh read or new fitting."""
import hashlib
import json
import math
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]


def read(path):
    assert path.stat().st_size<=1024*1024
    return json.loads(path.read_text())


def stats(a):
    a=sorted(a);n=len(a)
    def percentile(q):
        at=(n-1)*q;lo=int(at);hi=min(n-1,lo+1)
        return a[lo]+(a[hi]-a[lo])*(at-lo)
    return {'count':n,'median':percentile(.5)if n else None,'p95':percentile(.95)if n else None,
            'max':max(a)if n else None,'rms':math.sqrt(sum(x*x for x in a)/n)if n else None}


records=read(OUT/'face-diagnostics.json')
domain=read(ROOT/'outputs/panda-cheek-targeted-selection-v2/candidate.json')
fit=read(ROOT/'outputs/panda-cheek-curved-candidate-v1/fit-report.json')
field=read(OUT/'proposed-reference-field.json')['field']
rows={r['source_triangle']:r for r in records};excluded=set(fit['excluded_reference_corner_ids'])
custom=[];geometric=[];custom_geo=[];supported_outliers=[]
for ordinal,reference in enumerate(domain['healthy_remaining_original_body_reference_triangles']):
    r=rows[reference['source_face'][1]];support=[i for i in range(3)if ordinal*3+i not in excluded]
    custom.extend(r['custom_vs_failed_fit_corner_degrees'][i]for i in support)
    geometric.extend(r['geometric_vs_failed_fit_corner_degrees'][i]for i in support)
    custom_geo.extend(r['custom_geometric_corner_degrees'][i]for i in support)
    if any(r['custom_vs_failed_fit_corner_degrees'][i]>45 for i in support):
        supported_outliers.append({'source_primitive':0,'source_triangle':r['source_triangle'],
            'partition_original_face_id':r['partition_original_face_id'],
            'maximum_dihedral_degrees':r['maximum_original_body_edge_dihedral_degrees'],
            'maximum_supported_custom_fit_degrees':max(r['custom_vs_failed_fit_corner_degrees'][i]for i in support),
            'maximum_custom_geometric_degrees':max(r['custom_geometric_corner_degrees']),
            'contact_interface_edge_distance':r['contact_interface_edge_distance']})
assert len(custom)==304
delta=abs(stats(custom)['p95']-fit['supported_normal_error_degrees']['p95'])
assert delta<1e-7


def same(a,b):return max(abs(x-y)for x,y in zip(a,b))<=2e-7


def normalize(n):
    length=math.sqrt(sum(x*x for x in n));assert length>1e-12
    return [x/length for x in n]


def fixed_normal(point):
    local=[sum((point[k]-fit['origin_blender_m'][k])*fit['basis_blender_columns'][k][j]for k in range(3))/fit['normalization_scale_m']for j in range(3)]
    x,y=local[:2];c=fit['quadratic_coefficients'];n=[-(c[1]+2*c[3]*x+c[4]*y),-(c[2]+c[4]*x+2*c[5]*y),1]
    return normalize([sum(n[j]*fit['basis_blender_columns'][k][j]for j in range(3))for k in range(3)])


boundary=[]
for i,point in enumerate(domain['oriented_boundary_blender_m']):
    clean=[r['source_triangle']for r in field if any(same(point,p)for p in r['rest_positions_blender_m'])]
    old=[n for r in domain['healthy_remaining_original_body_reference_triangles']
         for p,n in zip(r['rest_positions_blender_m'],r['corner_normals_blender'])if same(point,p)]
    assert old
    mean=normalize([sum(n[k]for n in old)for k in range(3)]);predicted=fixed_normal(point)
    angle=math.degrees(math.acos(max(-1,min(1,sum(a*b for a,b in zip(mean,predicted))))))
    boundary.append({'boundary_ordinal':i,'body_welded_vertex':domain['oriented_boundary_body_welded_ids'][i],
        'position_blender_m':point,'cleaner_field_exact_reference_faces':clean,
        'original_boundary_custom_mean_vs_fixed_fit_degrees':angle})
value={'schema_version':1,'scope':'Reaggregation of already-written small JSON only; no new fit/mesh/texture operation',
    'supported_old_mask_custom_vs_fit_degrees':stats(custom),
    'supported_old_mask_geometric_vs_fit_degrees':stats(geometric),
    'supported_old_mask_custom_geometric_degrees':stats(custom_geo),
    'reproduced_failed_p95_delta_degrees':delta,'supported_outlier_face_count':len(supported_outliers),
    'supported_outlier_faces':supported_outliers,
    'supported_outliers_with_geometric_crease':sum(r['maximum_dihedral_degrees']>30 for r in supported_outliers),
    'supported_outliers_with_custom_disagreement':sum(r['maximum_custom_geometric_degrees']>15 for r in supported_outliers),
    'boundary_vertices':len(boundary),'boundary_vertices_with_cleaner_exact_support':sum(bool(r['cleaner_field_exact_reference_faces'])for r in boundary),
    'original_boundary_custom_mean_vs_fixed_fit_degrees':stats([r['original_boundary_custom_mean_vs_fixed_fit_degrees']for r in boundary]),
    'boundary_rows':boundary,'quality_approved':False}
with(OUT/'support-reconciliation.json').open('x')as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
print(json.dumps({k:v for k,v in value.items()if k not in['boundary_rows','supported_outlier_faces']},sort_keys=True))
