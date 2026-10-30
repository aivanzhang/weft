"""Opt-in full arm fitting with proved box redundancy screening and unchanged nonlinear gates."""

from copy import deepcopy
import time

import numpy as np

from weft import constrained_arm_alignment as base
from weft import restored_arm_alignment as restored
from weft.box_screened_linear_feasibility import CONTRACT as SCREEN_CONTRACT, find_box_screened_feasible_point
from weft.io import canonical_hash


CONTRACT = {**deepcopy(restored.CONTRACT), 'method': 'box-screened-curvature-restored-arm-alignment-v1',
    'restoration_initializer': deepcopy(SCREEN_CONTRACT),
    'maximum_screening_report_bytes': 4 * 1024 * 1024, 'maximum_screening_record_bytes': 512 * 1024,
    'qualification': 'All original restoration rows and exact box are checked; screened rows remain in the minimum-norm QP and nonlinear audit',
    'geometry_scope': 'Fixed nominal authored box proxies only; handle/depth and physical recovery are unverified; current compound admission remains separate'}


class _ScreenedInitializer:
    def __init__(self, deadline):
        self.deadline, self.receipts, self.receipt_bytes, self.calls = deadline, [], 0, 0

    def __call__(self, matrix, lower, lo, hi, *, seed, settings):
        base._check_deadline(self.deadline, 'screened_initializer_preparation')
        if self.receipt_bytes + CONTRACT['maximum_screening_record_bytes'] > CONTRACT['maximum_screening_report_bytes']:
            raise base._BudgetExpired('screening_receipt_budget')
        inputs = {'matrix': np.asarray(matrix).tolist(), 'lower': np.asarray(lower).tolist(),
            'bounds_lower': np.asarray(lo).tolist(), 'bounds_upper': np.asarray(hi).tolist(), 'seed': np.asarray(seed).tolist()}
        receipt = {'call_index': self.calls, 'input_sha256': canonical_hash(inputs),
            'bounds_lower': inputs['bounds_lower'], 'bounds_upper': inputs['bounds_upper'], 'seed': inputs['seed']}
        self.calls += 1
        try:
            result = find_box_screened_feasible_point(matrix, lower, lo, hi, seed=seed, settings=settings)
            receipt['result'] = result
        except ValueError as error:
            receipt.update(status='invalid_bounded_problem', error=str(error)[:256]); raise
        finally:
            raw = restored._bytes(receipt)
            oversized = len(raw) > CONTRACT['maximum_screening_record_bytes']
            if oversized:
                receipt = {'call_index': self.calls - 1, 'input_sha256': receipt['input_sha256'],
                    'status': 'screening_receipt_bound_exceeded', 'complete_result_retained': False, 'result_bytes': len(raw)}
                raw = restored._bytes(receipt)
            self.receipts.append(receipt); self.receipt_bytes += len(raw)
            if oversized: raise base._BudgetExpired('screening_record_budget')
        report = result['report']; phase = report['phase_one']; phase_report = {} if phase is None else phase['report']
        # The unchanged restoration flow consumes its old report shape; complete new receipts remain additive.
        return {'status': result['status'], 'point': result['point'], 'report': {
            'iterations': phase_report.get('iterations', 0), 'seconds': report['seconds'],
            'explicit_phase_slack': phase_report.get('explicit_phase_slack'), 'last_point': report['last_point'],
            'original_primal': report['original_primal'] or {'passes': False, 'maximum_violation': None}}}


def propose_box_screened_arm_alignment(human_document: bytes, observation_document: bytes, *,
        human_entity_ref, camera_hypothesis, equipment_boxes, human_xml_document: bytes):
    """Start from original references; numerical improvement never admits the authored geometry."""
    started = time.monotonic(); deadline = started + base.CONTRACT['maximum_fit_seconds']
    initializer = _ScreenedInitializer(deadline)
    solver = restored._SequentialSolver(deadline, feasibility_finder=initializer)
    result = base._run_alignment(human_document, observation_document, human_entity_ref=human_entity_ref,
        camera_hypothesis=camera_hypothesis, equipment_boxes=equipment_boxes, human_xml_document=human_xml_document,
        projection_solver=solver.project, curvature_restorer=solver.restore)
    report = result['report']
    report.update(profile=CONTRACT['method'], base_contract=report['contract'], contract=deepcopy(CONTRACT),
        restorations=solver.restorations, saved_diagnostic_problems=solver.diagnostics,
        screening_receipts=initializer.receipts,
        numerical_summary={'direction_qp_calls': solver.direction_calls, 'restoration_qp_calls': solver.restoration_calls,
            'phase_one_calls': solver.phase_calls, 'restoration_attempts': len(solver.restorations),
            'restored_steps_accepted': sum(r['status'] == 'accepted_restored_step' for r in solver.restorations),
            'restoration_report_bytes': solver.restoration_bytes, 'screened_initializer_calls': initializer.calls,
            'screening_report_bytes': initializer.receipt_bytes,
            'screened_rows_total': sum(len(r.get('result', {}).get('report', {}).get('omitted_rows', [])) for r in initializer.receipts),
            'global_infeasibility_claim': False})
    deadline = started + base.CONTRACT['maximum_helper_seconds']
    try:
        base._check_deadline(deadline, 'screened_output_serialization')
        if len(restored._bytes(result)) > base.MAX_OUTPUT_BYTES: raise ValueError('Screened proposal exceeds its complete output bound')
        base._check_deadline(deadline, 'screened_output_bound_check')
    except base._BudgetExpired as error:
        result['status'] = report['status'] = 'partial_budget_exhausted'
        report['deadline_stage'] = str(error); report['gates']['complete_bounded_solve'] = False
        report['failures'].append('cooperative_deadline:' + str(error))
        if len(restored._bytes(result)) > base.MAX_OUTPUT_BYTES: raise ValueError('Screened partial receipt exceeds its output bound')
    return result
