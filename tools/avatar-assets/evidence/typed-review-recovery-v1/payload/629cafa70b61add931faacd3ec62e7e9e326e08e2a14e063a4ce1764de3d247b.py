"""Complete analytic reviewer fixtures; no claims about rendered assets."""
from copy import deepcopy


def review_fixture(verdict='repair', summary='Analytic portable flecks need adjustment', evidence_ids=()):
    return {'verdict':verdict,'summary':summary,
        'checks':[{'name':name,'passed':verdict=='accept' or name!='appearance',
                   'reason':'Analytic test finding, not a visual inspection','evidence_ids':list(evidence_ids)}
                  for name in ('identity','appearance','framing','deformation')],
        'state_reviews':[],'required_repairs':[] if verdict=='accept' else ['Resolve analytic appearance finding'],
        'limitations':['Analytic orchestration fixture; no media was rendered.']}


def wire_review(canonical):
    result=deepcopy(canonical)
    result['schema_version']=2
    result['checks']={row['name']:{key:value for key,value in row.items() if key!='name'} for row in result['checks']}
    return result
