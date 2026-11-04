"""Deterministic whole-component thinning preserves surviving skin and corner attributes."""
from hashlib import sha256


def component_plan(vertex_count, polygons, triangle_budget, vertex_budget=None):
    if type(vertex_count) is not int or not 1 <= vertex_count <= 100000:
        raise ValueError('Bounded mesh vertices required')
    if type(triangle_budget) is not int or not 4 <= triangle_budget <= 40000:
        raise ValueError('Bounded triangle budget required')
    if not isinstance(polygons, list) or not 1 <= len(polygons) <= 100000:
        raise ValueError('Bounded polygons required')
    parents=list(range(vertex_count))
    def root(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]];i=parents[i]
        return i
    for polygon in polygons:
        if not isinstance(polygon, (list,tuple)) or not 3 <= len(polygon) <= 64 or any(type(i) is not int or not 0 <= i < vertex_count for i in polygon):
            raise ValueError('Bounded polygon indices required')
        a=root(polygon[0])
        for i in polygon[1:]:parents[root(i)]=a
    groups={}
    for polygon in polygons:
        key=root(polygon[0]);group=groups.setdefault(key,{'vertices':set(),'triangles':0})
        group['vertices'].update(polygon);group['triangles']+=len(polygon)-2
    components=sorted(groups.values(),key=lambda c:min(c['vertices']))
    if len(components)<16 or max(c['triangles'] for c in components)>64:
        return {'method':'connected_collapse','components':len(components),'keep_vertices':None}
    # A fixed permutation spreads surviving cards without changing any selected card's attributes.
    if vertex_budget is not None and (type(vertex_budget) is not int or not 4 <= vertex_budget <= 40000):
        raise ValueError('Bounded vertex budget required')
    ordered=sorted(enumerate(components),key=lambda x:sha256(str(x[0]).encode()).digest())
    selected=[];used=0;used_vertices=0
    for index,c in ordered:
        if used+c['triangles']<=triangle_budget and (vertex_budget is None
                or used_vertices+len(c['vertices'])<=vertex_budget):
            selected.append(index);used+=c['triangles'];used_vertices+=len(c['vertices'])
    if not selected:raise ValueError('No whole component fits the triangle budget')
    keep=sorted(v for i in selected for v in components[i]['vertices'])
    return {'method':'whole_component_thinning','components':len(components),'retained_components':len(selected),'triangles':used,'keep_vertices':keep}
