"""One nonlinear check of a sealed Phase-I frame-240 correction; no fitting or installation."""

import argparse
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np

import check_arm_curvature_restoration as restoration
import check_arm_phase_one as phase
from weft import constrained_arm_alignment as arm
from weft.io import canonical_hash, local_time
from weft.shared_world_packaging import read_package_file, read_package_manifest, write_package_files


FRAME = 240
MAX_SECONDS = 10
MAX_INPUT_BYTES = 32 * 1024 * 1024
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_RESULT_BYTES = 2 * 1024 * 1024


def load_chain(phase_one, *, deadline):
    """Verify all four input seals and archive only the exact required documents."""
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    packages, saved, byte_count = {}, {}, 0
    def package(label, source, wanted, expected_manifest=None):
        nonlocal byte_count
        source = Path(source).resolve(); manifest_raw, hashes = read_package_manifest(source)
        if len(hashes) > 128 or expected_manifest is not None and sha(manifest_raw) != expected_manifest:
            raise ValueError('Input study manifest identity/count differs: ' + label)
        byte_count += len(manifest_raw)
        selected = {}
        for name, expected in hashes.items():
            arm._check_deadline(deadline, 'nonlinear_parent_readback')
            raw = read_package_file(source, name, expected_sha256=expected, max_bytes=MAX_OUTPUT_BYTES)
            byte_count += len(raw)
            if byte_count > MAX_INPUT_BYTES: raise ValueError('Four-study input verification exceeds its byte envelope')
            if name in wanted: selected[name] = raw; saved[f'{label}/{name}'] = raw
        if set(selected) != set(wanted): raise ValueError('Input study lacks required documents: ' + label)
        saved[label + '/manifest.json'] = manifest_raw
        packages[label] = {'path': str(source), 'manifest_sha256': sha(manifest_raw), 'verified_files': len(hashes)}
        return selected
    solved = package('phase-one', phase_one, {'pre-phase-one.json', 'result.json'})
    solved_plan = restoration.local._json(solved['pre-phase-one.json'])
    if solved_plan['frame'] != FRAME or solved_plan['phase_one_contract'] != phase.CONTRACT:
        raise ValueError('Expected the frozen frame-240 Phase-I profile')
    prior = solved_plan['input_studies']['initializer']
    first = package('initializer', prior['path'], {'pre-initializer.json', 'results.json'}, prior['manifest_sha256'])
    first_plan = restoration.local._json(first['pre-initializer.json'])
    second = package('restoration', first_plan['parent'], {'pre-restoration.json', f'frames/frame-{FRAME:05d}.json'}, first_plan['parent_manifest_sha256'])
    prior_restoration = solved_plan['input_studies']['restoration']
    if (Path(prior_restoration['path']).resolve() != Path(first_plan['parent']).resolve() or
            prior_restoration['manifest_sha256'] != first_plan['parent_manifest_sha256']):
        raise ValueError('Phase-I restoration ancestor differs')
    for name, raw in saved.items():
        if name.startswith(('initializer/', 'restoration/')) and sha(raw) != solved_plan['input_sha256'][name]:
            raise ValueError('Phase-I saved matrix parent binding differs: ' + name)
    second_plan = restoration.local._json(second['pre-restoration.json'])
    required = {'pre-capture.json', 'comparisons.json', 'inputs/parent/proposal.json', 'inputs/parent/pre-fit.json',
        'inputs/parent/inputs/human.json', 'inputs/parent/inputs/construction-evidence.json', 'inputs/parent/inputs/human.xml', f'subproblems/frame-{FRAME:05d}.json'}
    third = package('active-set', second_plan['parent'], required, second_plan['parent_manifest_sha256'])
    initialized = restoration.local._json(solved['result.json'])
    prior_initializer = restoration.local._json(first['results.json'])[str(FRAME)]
    corrected = restoration.local._json(second[f'frames/frame-{FRAME:05d}.json'])
    original_comparisons = restoration.local._json(third['comparisons.json'])['comparisons']
    original = [r for r in original_comparisons if r['frame'] == FRAME]
    if (corrected['frame'] != FRAME or len(original) != 1 or initialized['status'] != 'verified_minimum_norm_linear_restoration' or
            original[0]['active_set']['status'] != 'optimal' or corrected['direction_scale'] != 1 / 32 or
            corrected['saved_direction_rad'] != original[0]['active_set']['step'] or len(corrected['corrections']) != 1):
        raise ValueError('Sealed direction, correction or frame identity differs')
    correction = np.asarray(initialized['qualified_linear_correction']); direction = np.asarray(corrected['saved_direction_rad'])
    problem = corrected['corrections'][0]['problem']; trial = np.asarray(corrected['corrections'][0]['trial_delta_rad'])
    if (correction.shape != (8,) or direction.shape != (8,) or not np.all(np.isfinite(correction)) or
            not np.array_equal(direction / 32, trial) or canonical_hash(problem) != initialized['problem_sha256'] or
            not np.array_equal(correction, initialized['active_set']['step']) or initialized['active_set']['status'] != 'optimal' or
            initialized['problem_sha256'] != prior_initializer['problem_sha256'] or initialized['problem_sha256'] != solved_plan['problem_sha256']):
        raise ValueError('Sealed matrix/correction origin differs')
    lo, hi = phase.exact_box_bounds(problem)
    check = phase.original_primal_check(np.asarray(problem['matrix']), np.asarray(problem['lower']), lo, hi, correction)
    primal = check['maximum_violation']
    if not check['passes']: raise ValueError('Saved correction is not independently linearly feasible')
    sealed_matrix_raw = third[f'subproblems/frame-{FRAME:05d}.json']
    sealed_matrix = restoration.local._json(sealed_matrix_raw)
    if sha(sealed_matrix_raw) != original[0]['matrix_sha256'] or sealed_matrix['frame'] != FRAME:
        raise ValueError('Saved direction variable-order origin differs')
    previous = restoration.local._json(third['inputs/parent/proposal.json']); candidate = previous['human_candidate']
    captured = restoration.local._json(third['pre-capture.json']); prefit = restoration.local._json(third['inputs/parent/pre-fit.json'])
    if (canonical_hash(candidate) != captured['candidate_model_sha256'] or captured['arm_contract'] != arm.CONTRACT or
            any(captured[k] != prefit[k] for k in ('human_entity_ref', 'camera_hypothesis', 'equipment_boxes'))):
        raise ValueError('Frozen candidate, camera or equipment binding differs')
    for name, raw in third.items():
        if name.startswith('inputs/parent/') and sha(raw) != captured['input_sha256'][name.removeprefix('inputs/parent/')]:
            raise ValueError('Frozen source document changed: ' + name)
    return {'packages': packages, 'saved': saved, 'active_inputs': third, 'candidate': candidate, 'prefit': prefit,
        'direction': direction, 'correction': correction, 'delta': trial + correction,
        'sealed_matrix': sealed_matrix, 'linear_primal_violation': primal, 'frame_reference_index': corrected['reference_index'], 'verified_bytes': byte_count}


def run(phase_one, out):
    started = time.monotonic(); out = Path(out).resolve()
    if out.exists(): raise ValueError('Choose a new nonlinear audit directory')
    chain = load_chain(phase_one, deadline=started + MAX_SECONDS)
    for package in chain['packages'].values():
        source = Path(package['path'])
        if out.is_relative_to(source) or source.is_relative_to(out): raise ValueError('Audit output overlaps an immutable input study')
    project = Path(__file__).resolve().parents[2]; package = project / 'src/weft'
    paths = {name: package / name for name in tuple(dict.fromkeys(restoration.local.MODULES + ('bounded_linear_feasibility.py',)))}
    paths.update({'study.py': Path(__file__), 'phase-one-study.py': Path(phase.__file__),
        'previous-initializer.py': Path(phase.previous.__file__), 'restoration-inspection.py': Path(restoration.__file__),
        'local-context.py': Path(restoration.local.__file__), 'custody.py': Path(restoration.local.custody.__file__),
        **{'project/' + name: project / name for name in ('requirements-lab.lock', 'pyproject.toml')}})
    implementation = {name: path.read_bytes() for name, path in paths.items()}; sha = lambda raw: hashlib.sha256(raw).hexdigest()
    plan = {'schema_version': 1, 'created_at': local_time(), 'profile': 'sealed-phase-one-frame240-nonlinear-audit-v1',
        'frame': FRAME, 'input_studies': chain['packages'], 'verified_input_bytes': chain['verified_bytes'],
        'input_sha256': {name: sha(raw) for name, raw in chain['saved'].items()},
        'candidate_model_sha256': canonical_hash(chain['candidate']), 'human_entity_ref': chain['prefit']['human_entity_ref'],
        'camera_hypothesis': chain['prefit']['camera_hypothesis'], 'equipment_boxes': chain['prefit']['equipment_boxes'],
        'sealed_direction': chain['direction'].tolist(), 'direction_scale': 1 / 32,
        'sealed_correction': chain['correction'].tolist(), 'audited_delta_rad': chain['delta'].tolist(),
        'independent_linear_primal_violation': chain['linear_primal_violation'],
        'phase_one_contract': deepcopy(phase.CONTRACT), 'arm_contract': deepcopy(arm.CONTRACT), 'restoration_inspection_contract': deepcopy(restoration.CONTRACT),
        'maximum_seconds': MAX_SECONDS, 'maximum_output_bytes': MAX_OUTPUT_BYTES,
        'new_qp_captures': 0, 'new_solver_calls': 0, 'additional_corrections': 0, 'full_source_fits': 0,
        'physics_steps': 0, 'paid_calls': 0, 'video_decodes': 0, 'candidate_installed': False,
        'scope': 'Single independent frame on original nominal boxes; other frames, composed motion and current-compound full-source admission are outside this audit',
        'implementation_sha256': {name: sha(raw) for name, raw in implementation.items()},
        'runtime_versions': {name: version(name) for name in ('numpy', 'mujoco', 'gymnasium')}}
    custody = restoration.local.custody
    initial = {'pre-audit.json': custody._bytes(plan), **{'inputs/' + name: raw for name, raw in chain['saved'].items()},
        **{'implementation/' + name: raw for name, raw in implementation.items()}}
    if sum(map(len, initial.values())) + MAX_RESULT_BYTES + custody.METADATA_RESERVE_BYTES > MAX_OUTPUT_BYTES:
        raise ValueError('Complete nonlinear audit archive and reserves exceed sixteen MiB')
    out.mkdir(parents=True, mode=0o700); old_handler = signal.getsignal(signal.SIGALRM)
    def alarm(*_): raise TimeoutError('Single-frame nonlinear audit exceeded ten seconds')
    signal.signal(signal.SIGALRM, alarm); signal.setitimer(signal.ITIMER_REAL, max(.001, MAX_SECONDS - (time.monotonic() - started)))
    report = {'status': 'failed', 'promoted': False, 'physical_validation': 'unverified'}
    try:
        write_package_files(out, initial)
        raw, prefit = chain['active_inputs'], chain['prefit']
        context = restoration.local.capture_local_problems(raw['inputs/parent/inputs/human.json'], raw['inputs/parent/inputs/construction-evidence.json'], chain['candidate'],
            human_entity_ref=prefit['human_entity_ref'], camera_hypothesis=prefit['camera_hypothesis'], equipment_boxes=prefit['equipment_boxes'],
            human_xml_document=raw['inputs/parent/inputs/human.xml'], frames=(FRAME,), deadline=started + MAX_SECONDS, prepare_only=True)
        index = context['indices'][0]
        if (index != chain['frame_reference_index'] or tuple(context['geometry'].arm_names) != tuple(chain['sealed_matrix']['variables']) or
                not np.array_equal(context['changes'][index], chain['sealed_matrix']['current_angle_correction_rad'])):
            raise ValueError('Prepared reference index, variable order or correction origin differs')
        baseline_cost, _, _, valid, _, _, _ = context['evaluate'](index, context['changes'][index])
        if not valid: raise ValueError('Frozen baseline no longer satisfies its nonlinear guards')
        audit = restoration.inspect_local(context, index, chain['delta'], baseline_cost)
        if audit['pair_comparisons'] != 180 or len(audit['pair_checks']) != 180 or audit['affected_unique_samples'] != 5:
            raise ValueError('The fixed frame-240 audit requires all 180 pair checks across five samples')
        audit.update(frame=FRAME, reference_index=index, all_references=len(context['rows']), all_source_slots=len(context['source_rows']),
            candidate_installed=False, other_frames_status='not_modified_or_audited', composed_motion_admission='not_executed')
        reference = deepcopy(chain['candidate']['reference'][index])
        for j, name in enumerate(context['geometry'].arm_names): reference['joints'][name] += float(chain['delta'][j])
        audit['proposed_reference_unadmitted'] = reference
        result_raw = custody._bytes(audit)
        if len(result_raw) > MAX_RESULT_BYTES: raise ValueError('Complete pair audit exceeds its output reserve')
        write_package_files(out, {'nonlinear-audit.json': result_raw})
        if canonical_hash(chain['candidate']) != plan['candidate_model_sha256']: raise ValueError('Nonlinear audit modified its input candidate')
        for name, raw in chain['saved'].items():
            label, relative = name.split('/', 1)
            read_package_file(Path(chain['packages'][label]['path']), relative, expected_sha256=sha(raw), max_bytes=MAX_OUTPUT_BYTES)
        for name, raw in implementation.items():
            if paths[name].read_bytes() != raw: raise ValueError('Pinned implementation/configuration changed: ' + name)
        report.update(status='local_step_passes_original_nonlinear_checks' if audit['local_step_acceptable'] else 'local_step_rejected',
            frame=FRAME, seconds=time.monotonic() - started, input_and_implementation_unchanged=True,
            pair_comparisons=audit['pair_comparisons'], affected_unique_samples=audit['affected_unique_samples'],
            maximum_violation_model_units=audit['maximum_violation_model_units'], cost=audit['cost'], baseline_cost=audit['baseline_cost'],
            maximum_angle_correction_rad=audit['maximum_angle_correction_rad'], maximum_correction_rate_rad_s=audit['maximum_correction_rate_rad_s'],
            maximum_endpoint_depth_change_model_units=audit['maximum_endpoint_depth_change_model_units'],
            local_step_acceptable=audit['local_step_acceptable'], all_references=audit['all_references'], all_source_slots=audit['all_source_slots'],
            new_solver_calls=0, new_qp_captures=0, full_source_fits=0, physics_steps=0, candidate_installed=False,
            full_source_admission='not_executed', other_frames_status='not_modified_or_audited')
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) + len(custody._bytes(report)) + custody.METADATA_RESERVE_BYTES > MAX_OUTPUT_BYTES:
            raise ValueError('Nonlinear audit publication exceeds its output envelope')
        custody._publish_manifest(out, report)
        sealed, outputs = read_package_manifest(out)
        for name, expected in outputs.items(): read_package_file(out, name, expected_sha256=expected, max_bytes=MAX_OUTPUT_BYTES)
        return report | {'manifest_sha256': sha(sealed), 'readback_verified_files': len(outputs)}
    except Exception as error:
        signal.setitimer(signal.ITIMER_REAL, 0); custody._failure_receipt(out, error, report, plan['input_studies']['phase-one']['manifest_sha256']); raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old_handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase-one', required=True); parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.phase_one, args.out), indent=2, allow_nan=False))
