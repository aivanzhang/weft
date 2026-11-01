"""Missing image support must cross exact configuration, worker and report boundaries."""

from copy import deepcopy
import json

import pytest

from weft import source_surface_fit_inputs as inputs
from weft import whole_body_reference_pipeline as pipeline
from weft import whole_body_reference_report as report
from weft import whole_body_reference_worker as worker
from test_surface_fit_workflow import bundle, raw, snapshot
from test_whole_body_reference_pipeline import workflow, verify_seal


@pytest.mark.parametrize('policy', [None, 'preserve', 'qualified_surfaces'])
def test_exact_config_freezes_policy_without_changing_default_options(tmp_path, policy):
    path, config, body = bundle(tmp_path / 'config')
    if policy is not None:
        config['missing_image_policy'] = policy
        path.write_bytes(raw(config))
    files, descriptor = snapshot(path, config['source_sha256'])
    options = inputs.prepare_surface_fit_objective(files | {'inputs/human.json': body}, descriptor)
    assert (path.read_bytes() == files[inputs.CONFIG_MEMBER])
    if policy is None:
        assert 'missing_image_policy' not in descriptor and 'missing_image_policy' not in options
    else:
        assert descriptor['missing_image_policy'] == options['missing_image_policy'] == policy
    forged = descriptor | {'missing_image_policy': 'preserve' if policy == 'qualified_surfaces' else 'qualified_surfaces'}
    with pytest.raises(ValueError, match='descriptor'):
        inputs.validate_surface_fit_snapshot(files, forged)


@pytest.mark.parametrize('policy', [True, None, 'infer', {}])
def test_invalid_policy_never_becomes_implicit_permission(tmp_path, policy):
    path, config, _ = bundle(tmp_path / 'config')
    _, descriptor = snapshot(path, config['source_sha256'])
    with pytest.raises(ValueError, match='missing-image policy'):
        inputs.parse_surface_fit_config(raw(config | {'missing_image_policy': policy}))
    with pytest.raises(ValueError, match='missing-image policy'):
        inputs.surface_fit_member_names(descriptor | {'missing_image_policy': policy})


def test_public_package_archives_and_checks_the_explicit_worker_policy(workflow, tmp_path, monkeypatch):
    source, out, source_snapshot, _ = workflow
    path, config, body = bundle(tmp_path / 'config')
    config['missing_image_policy'] = 'qualified_surfaces'
    path.write_bytes(raw(config))
    source_snapshot['identity']['source_sha256'] = config['source_sha256']
    source_snapshot['files']['inputs/human.json'] = body

    def runtime(*, surface=False):
        names = (*pipeline.RUNTIME_MODULES, *(inputs.RUNTIME_MODULES if surface else ()))
        return ({f'runtime/weft/{name}.py': b'# frozen test boundary\n' for name in names},
                dict.fromkeys(('python', 'numpy', 'mujoco', 'gymnasium'), 'fixture'))

    monkeypatch.setattr(pipeline, '_runtime', runtime)
    result = pipeline.prepare_whole_body_reference_proposal(source, out, proposal_objective='surface_interval_v1',
                                                           surface_config=path, fit_seconds=.5)
    assert result['status'] == 'completed'
    plan = worker._plan((out / 'proposal/plan.json').read_bytes())
    assert plan['surface_constraints']['missing_image_policy'] == 'qualified_surfaces'
    assert (out / 'proposal/inputs/surface-config.json').read_bytes() == path.read_bytes()
    verify_seal(out)
    verify_seal(out / 'proposal')
    for invalid in [True, None, 'infer']:
        forged = deepcopy(plan)
        forged['surface_constraints']['missing_image_policy'] = invalid
        with pytest.raises(ValueError, match='missing-image policy'):
            worker._plan(raw(forged))


def policy_documents():
    source = {'reference': [{'joints': {'knee': 0.}}, {'joints': {'knee': 0.}}]}
    candidate = deepcopy(source)
    candidate['reference'][1]['joints']['knee'] = .1
    candidate['whole_body_reference_refinement'] = {'missing_image_policy': 'qualified_surfaces'}
    value = {'profile': report.SURFACE_METHOD, 'status': 'unadmitted_construction_proposal',
             'objective': {'missing_image_policy': 'qualified_surfaces'}, 'references': [
                 {'points': [{'weight': 0.}], 'surface_only_eligible': False, 'surface_only_attempted': False,
                  'status': 'frozen_first_reference'},
                 {'points': [{'weight': 0.}], 'surface_only_eligible': True, 'surface_only_attempted': True,
                  'status': 'changed_surface_only_proposal'}],
             'surface_intervals': {'references': [
                 {'before': {'objective_m2': .4, 'counts': {'evaluated_channels': 1}}},
                 {'before': {'objective_m2': .2, 'counts': {'evaluated_channels': 1}}}],
                 'trial_journal': [{'kind': 'qp', 'reference_index': 1}]}}
    return value, candidate, source, {'missing_image_policy': 'qualified_surfaces'}


def inspect(documents):
    return report._surface_row_policy(*documents, lambda: None)


def test_report_checks_source_only_origin_against_saved_qp_and_interval():
    documents = policy_documents()
    result = inspect(documents)
    assert result['surface_only_rows'] == {'eligible': 1, 'attempted': 1, 'changed': 1}
    documents[0]['references'][1]['status'] = 'changed_inferred_proposal'
    with pytest.raises(ValueError, match='qualified surface-only origin'):
        inspect(documents)


@pytest.mark.parametrize('mutation', [
    lambda r: r['references'][0].update(surface_only_eligible=True),
    lambda r: r['references'][1]['points'][0].update(weight=.8),
    lambda r: r['surface_intervals']['references'][1]['before'].update(objective_m2=0.),
    lambda r: r['surface_intervals']['references'][1]['before']['counts'].update(evaluated_channels=0),
])
def test_report_rejects_fabricated_missing_image_eligibility(mutation):
    documents = policy_documents()
    mutation(documents[0])
    with pytest.raises(ValueError, match='eligibility'):
        inspect(documents)


def test_report_preserves_interrupted_qp_without_claiming_completed_evidence():
    documents = policy_documents()
    documents[0]['surface_intervals']['trial_journal'] = []
    with pytest.raises(ValueError, match='saved QP journal'):
        inspect(documents)
    documents[0]['status'] = 'partial_budget_exhausted'
    documents[0]['references'][1]['status'] = 'unchanged_no_accepted_surface_step'
    documents[1]['reference'][1]['joints']['knee'] = 0.
    assert inspect(documents)['surface_only_rows'] == {'eligible': 1, 'attempted': 1, 'changed': 0}


@pytest.mark.parametrize('policy', [None, 'preserve'])
def test_report_legacy_policy_cannot_silently_claim_surface_only_changes(policy):
    documents = list(policy_documents())
    documents[3] = {} if policy is None else {'missing_image_policy': policy}
    with pytest.raises(ValueError, match='not requested'):
        inspect(documents)
    documents[0]['objective'] = {}
    documents[1]['whole_body_reference_refinement'] = {}
    documents[1]['reference'][1]['joints']['knee'] = 0.
    for index, row in enumerate(documents[0]['references']):
        row.pop('surface_only_eligible'); row.pop('surface_only_attempted')
        row['status'] = 'frozen_first_reference' if index == 0 else 'unchanged_missing_source_targets'
    assert inspect(documents) == {}


def test_report_rejects_policy_changed_after_request():
    documents = policy_documents()
    documents[0]['objective']['missing_image_policy'] = 'preserve'
    with pytest.raises(ValueError, match='exact requested config'):
        inspect(documents)


def test_preserve_rejects_a_correction_relabelled_as_image_supported():
    documents = list(policy_documents())
    documents[3] = {}
    documents[0]['objective'] = {}
    documents[1]['whole_body_reference_refinement'] = {}
    for row in documents[0]['references']:
        row.pop('surface_only_eligible'); row.pop('surface_only_attempted')
    documents[0]['references'][1]['status'] = 'changed_inferred_proposal'
    with pytest.raises(ValueError, match='preserve'):
        inspect(documents)


@pytest.mark.parametrize('eligible', [True, False])
def test_surface_unchanged_status_must_match_exact_row_eligibility(eligible):
    documents = policy_documents()
    row = documents[0]['references'][1]
    documents[1]['reference'][1]['joints']['knee'] = 0.
    if eligible:
        row['status'] = 'unchanged_missing_source_targets'
    else:
        row.update(surface_only_eligible=False, surface_only_attempted=False,
                   status='unchanged_no_accepted_surface_step')
        documents[0]['surface_intervals']['references'][1]['before']['objective_m2'] = 0.
        documents[0]['surface_intervals']['trial_journal'] = []
    with pytest.raises(ValueError, match='unchanged label'):
        inspect(documents)
