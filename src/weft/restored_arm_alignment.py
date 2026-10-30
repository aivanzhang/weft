"""Opt-in sequential arm fitting with feasible QPs and bounded curvature restoration."""

from copy import deepcopy
import json
import time

import numpy as np

from weft import constrained_arm_alignment as base
from weft.bounded_linear_feasibility import CONTRACT as PHASE_CONTRACT, find_bounded_feasible_point, original_primal_check
from weft.constrained_qp import CONTRACT as QP_CONTRACT, solve_feasible_qp
from weft.io import canonical_hash


CONTRACT = {**deepcopy(base.CONTRACT), 'method': 'active-set-curvature-restored-arm-alignment-v1',
    'direction_solver': deepcopy(QP_CONTRACT), 'restoration_initializer': deepcopy(PHASE_CONTRACT),
    'maximum_restorations_per_update': 2, 'restoration_direction_scale': 1 / 32,
    'restoration_trigger': 'All six backtracks completed without acceptance; smallest trial remains nonlinear-invalid',
    'restoration_objective': 'Minimum Euclidean norm of eight-angle correction; exact identity Hessian and zero gradient',
    'restoration_linear_pair_tolerance_spending': 0., 'restoration_minimum_cost_decrease': 1e-12,
    'maximum_restoration_report_bytes': 2 * 1024 * 1024, 'maximum_restoration_record_bytes': 4096,
    'maximum_saved_diagnostic_problems': 4, 'maximum_saved_problem_bytes': 64 * 1024,
    'composition': 'One accepted pose update at a time; every affected interpolation sample uses the latest accepted neighbors',
    'final_gates': 'Exactly the original full-source image, endpoint, motion, completeness and nominal-box clearance gates',
    'global_infeasibility_claim': False}


def _bytes(value):
    return json.dumps(value, allow_nan=False, separators=(',', ':')).encode()


def _limits(deadline):
    remaining = deadline - time.monotonic()
    if remaining < .001: raise base._BudgetExpired('restored_solver_deadline')
    return {'maximum_seconds': min(.5, remaining)}


def _qp_summary(result):
    report = result['report']
    keys = ('iterations', 'seconds', 'primal_maximum_violation', 'stationarity_relative',
            'complementarity_relative', 'dual_maximum_negative_multiplier', 'hessian_condition',
            'active_constraints', 'last_feasible_point')
    return {'status': result['status'], **{key: report[key] for key in keys if key in report}}


class _SequentialSolver:
    def __init__(self, deadline):
        self.deadline = deadline
        self.restorations, self.diagnostics = [], []
        self.restoration_bytes = 0
        self.direction_calls = self.restoration_calls = self.phase_calls = 0

    def _save_problem(self, kind, hessian, gradient, matrix, lower):
        problem = {'kind': kind, 'hessian': np.asarray(hessian).tolist(), 'gradient': np.asarray(gradient).tolist(),
                   'matrix': np.asarray(matrix).tolist(), 'lower': np.asarray(lower).tolist()}
        identity = canonical_hash(problem)
        if len(self.diagnostics) < CONTRACT['maximum_saved_diagnostic_problems'] and len(_bytes(problem)) <= CONTRACT['maximum_saved_problem_bytes']:
            self.diagnostics.append({'sha256': identity, 'problem': problem})
        return identity

    def _solve(self, hessian, gradient, matrix, lower, initial, deadline, *, restoration=False):
        settings = _limits(deadline)
        if restoration: self.restoration_calls += 1
        else: self.direction_calls += 1
        try:
            result = solve_feasible_qp(hessian, gradient, matrix, lower, initial=initial, settings=settings)
        except ValueError as error:
            return None, {'status': 'invalid_bounded_problem', 'reason': str(error)[:256]}
        summary = _qp_summary(result)
        if result['status'] != 'optimal': return None, summary
        step = np.asarray(result['step']); a, b = np.asarray(matrix), np.asarray(lower)
        residual = b - a @ step
        if not np.all(np.isfinite(residual)) or np.max(residual, initial=0.) > QP_CONTRACT['primal_tolerance']:
            summary['status'] = 'independent_original_primal_failed'; return None, summary
        summary['independent_original_primal'] = max(0., float(np.max(residual, initial=0.)))
        return step, summary

    def project(self, hessian, gradient, matrix, lower, *, cycles, tolerance):
        step, report = self._solve(hessian, gradient, matrix, lower, np.zeros(8), self.deadline)
        if step is None:
            report['problem_sha256'] = self._save_problem('direction', hessian, gradient, matrix, lower)
        return (np.zeros(8) if step is None else step), {
            'converged': step is not None, 'method': QP_CONTRACT['method'], **report,
            'maximum_violation': report.get('primal_maximum_violation'), 'global_infeasibility_claim': False}

    def restore(self, *, index, sweep, sigma, baseline, direction, lower, upper, baseline_cost,
                evaluate, affected_count, pair_count, deadline):
        if self.restoration_bytes + CONTRACT['maximum_restoration_record_bytes'] > CONTRACT['maximum_restoration_report_bytes']:
            raise base._BudgetExpired('restoration_report_budget')
        record = {'reference_index': index, 'sweep': sweep, 'collision_sigma_model_units': sigma,
            'baseline_cost': baseline_cost, 'rounds': [], 'status': 'started', 'nonlinear_rejections': 0}
        delta = np.asarray(direction) * CONTRACT['restoration_direction_scale']
        output = {'trial': None, 'states': None, 'nonlinear_rejections': 0}
        try:
            for iteration in range(CONTRACT['maximum_restorations_per_update']):
                base._check_deadline(deadline, 'curvature_restoration_derivatives')
                _, _, _, _, matrix, bounds, _ = evaluate(baseline + delta, True)
                stride = pair_count + 2 * len(base.POINT_NAMES)
                if len(bounds) != affected_count * stride or np.asarray(matrix).shape != (len(bounds), 8):
                    record['status'] = 'incomplete_restoration_constraints'; break
                matrix, bounds = np.asarray(matrix), np.asarray(bounds)
                # Restoration targets the true pair boundary; roundoff remains only in the final nonlinear gate.
                pair_rows = np.array([start + j for start in range(0, len(bounds), stride) for j in range(pair_count)])
                bounds[pair_rows] += base.CONTRACT['numerical_tolerance']
                lo, hi = lower - delta, upper - delta
                matrix = np.vstack((matrix, np.eye(8), -np.eye(8))); bounds = np.r_[bounds, lo, -hi]
                round_report = {'iteration': iteration, 'trial_delta_rad': delta.tolist(), 'constraint_rows': len(bounds),
                    'pair_rows': len(pair_rows), 'linear_pair_tolerance_spending': 0.}
                record['rounds'].append(round_report)
                settings = _limits(deadline); self.phase_calls += 1
                try:
                    phase = find_bounded_feasible_point(matrix, bounds, lo, hi, seed=-delta, settings=settings)
                except ValueError as error:
                    record['status'] = 'invalid_phase_one_problem'; round_report['reason'] = str(error)[:256]; break
                phase_report = phase['report']
                round_report['phase_one'] = {'status': phase['status'], 'iterations': phase_report['iterations'],
                    'seconds': phase_report['seconds'], 'explicit_phase_slack': phase_report['explicit_phase_slack'],
                    'last_point': phase_report['last_point'], 'maximum_original_violation': phase_report['original_primal']['maximum_violation']}
                if phase['point'] is None:
                    record['status'] = 'no_verified_restoration_initializer'
                    round_report['problem_sha256'] = self._save_problem('restoration', np.eye(8), np.zeros(8), matrix, bounds)
                    break
                check = original_primal_check(matrix, bounds, lo, hi, np.asarray(phase['point']))
                if not check['passes']: raise ValueError('Restoration initializer failed its independent original-unit check')
                correction, solved = self._solve(np.eye(8), np.zeros(8), matrix, bounds, phase['point'], deadline, restoration=True)
                round_report['minimum_norm_qp'] = solved
                if correction is None:
                    record['status'] = 'minimum_norm_restoration_unresolved'
                    round_report['problem_sha256'] = self._save_problem('restoration', np.eye(8), np.zeros(8), matrix, bounds)
                    break
                round_report['correction_rad'] = correction.tolist(); delta = delta + correction
                base._check_deadline(deadline, 'restoration_fresh_nonlinear_check')
                trial = baseline + delta
                cost, _, _, valid, _, _, states = evaluate(trial)
                bounded = bool(np.all(delta >= lower - QP_CONTRACT['primal_tolerance']) and
                               np.all(delta <= upper + QP_CONTRACT['primal_tolerance']))
                cost_decreased = bool(cost < baseline_cost - CONTRACT['restoration_minimum_cost_decrease'])
                round_report.update(cost=cost, nonlinear_valid=bool(valid), update_bounds_valid=bounded, cost_decreased=cost_decreased)
                if valid and bounded and cost_decreased:
                    record['status'] = 'accepted_restored_step'
                    output.update(trial=trial, states=states); break
                if not valid: record['nonlinear_rejections'] += 1
                record['status'] = 'restoration_nonlinear_or_cost_rejected'
            output['nonlinear_rejections'] = record['nonlinear_rejections']
            return output
        except base._BudgetExpired:
            record['status'] = 'partial_budget_exhausted'; raise
        finally:
            raw = _bytes(record)
            if len(raw) > CONTRACT['maximum_restoration_record_bytes']:
                raise ValueError('Restoration receipt exceeds its reserved record bound')
            self.restorations.append(record); self.restoration_bytes += len(raw)


def propose_restored_arm_alignment(human_document: bytes, observation_document: bytes, *,
                                  human_entity_ref, camera_hypothesis, equipment_boxes, human_xml_document: bytes):
    """Fit original source references sequentially; the candidate remains unadmitted under unchanged final gates."""
    started = time.monotonic(); solver = _SequentialSolver(started + base.CONTRACT['maximum_fit_seconds'])
    result = base._run_alignment(human_document, observation_document, human_entity_ref=human_entity_ref,
        camera_hypothesis=camera_hypothesis, equipment_boxes=equipment_boxes, human_xml_document=human_xml_document,
        projection_solver=solver.project, curvature_restorer=solver.restore)
    report = result['report']
    report.update(profile=CONTRACT['method'], base_contract=report['contract'], contract=deepcopy(CONTRACT),
        restorations=solver.restorations, saved_diagnostic_problems=solver.diagnostics,
        numerical_summary={'direction_qp_calls': solver.direction_calls, 'restoration_qp_calls': solver.restoration_calls,
            'phase_one_calls': solver.phase_calls, 'restoration_attempts': len(solver.restorations),
            'restored_steps_accepted': sum(r['status'] == 'accepted_restored_step' for r in solver.restorations),
            'restoration_report_bytes': solver.restoration_bytes, 'global_infeasibility_claim': False})
    deadline = started + base.CONTRACT['maximum_helper_seconds']
    try:
        base._check_deadline(deadline, 'restored_output_serialization')
        if len(_bytes(result)) > base.MAX_OUTPUT_BYTES: raise ValueError('Restored proposal exceeds its complete output bound')
        base._check_deadline(deadline, 'restored_output_bound_check')
    except base._BudgetExpired as error:
        result['status'] = report['status'] = 'partial_budget_exhausted'
        report['deadline_stage'] = str(error); report['gates']['complete_bounded_solve'] = False
        report['failures'].append('cooperative_deadline:' + str(error))
        if len(_bytes(result)) > base.MAX_OUTPUT_BYTES: raise ValueError('Restored partial receipt exceeds its output bound')
    return result
