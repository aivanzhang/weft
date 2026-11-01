"""Cold archived-runtime checks; no engine, source processing or surface admission."""

import ast
import builtins
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from test_shared_exercise_reference_inputs import packages, passing_documents
from test_whole_body_reference_interval_report import interval_documents
from test_whole_body_reference_report import save_study


SOURCE = Path(__file__).resolve().parents[1] / 'src/weft'
HELPER = 'weft.source_surface_fit_inputs'


def archived_runtime(path):
    tree = ast.parse((SOURCE / 'shared_exercise_reference_trial.py').read_text())
    names = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'RUNTIME_FILES' for target in node.targets))
    assert 'whole_body_reference_report.py' in names and 'source_surface_fit_inputs.py' not in names
    total = 0
    for name in names:
        raw = (SOURCE / name).read_bytes(); total += len(raw)
        assert total <= 16 << 20
        target = path / 'weft' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    return path


_COLD = r'''
import builtins,json,pathlib,sys
runtime,proposal,source,task,proposal_pin,source_pin,mode=sys.argv[1:]
sys.path.insert(0,runtime)
original=builtins.__import__
def guarded(name,*args,**kwargs):
    assert name.split('.')[0] not in {'numpy','mujoco','cv2','mediapipe','torch','pymunk','socket','subprocess'}, name
    assert not name.startswith('weft.source_surface'), 'Optional surface dependency imported: '+name
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from weft import whole_body_reference_report as report
from weft.shared_exercise_reference_inputs import load_reference_inputs,PROPOSAL_GATES
assert pathlib.Path(report.__file__).resolve().is_relative_to(pathlib.Path(runtime).resolve())
assert not pathlib.Path(runtime,'weft/source_surface_fit_inputs.py').exists()
assert len(PROPOSAL_GATES)==11
if mode=='legacy':
    loaded=load_reference_inputs(proposal,source,task,expected_proposal_manifest_sha256=proposal_pin,
                                 expected_scene_manifest_sha256=source_pin)
    assert loaded.original_human_bytes and loaded.candidate_human_bytes
    assert loaded.receipt()['identities']
else:
    try:load_reference_inputs(proposal,source,task,expected_proposal_manifest_sha256=proposal_pin,
                              expected_scene_manifest_sha256=source_pin)
    except ValueError:pass
    else:raise AssertionError('Surface proposal gained legacy reference-trial admission')
assert not any(name.startswith('weft.source_surface') for name in sys.modules)
print(json.dumps({'mode':mode,'optional_helper_imported':False,'protected_gate_count':len(PROPOSAL_GATES)}))
'''


@pytest.mark.parametrize('floor_first', [False, True], ids=['image-first', 'floor-first'])
def test_legacy_report_and_reference_inputs_load_from_cold_archive_without_surface_helper(tmp_path, floor_first):
    runtime = archived_runtime(tmp_path / 'runtime')
    values = packages(tmp_path, documents=passing_documents(floor_first=floor_first))
    result = subprocess.run([sys.executable, '-I', '-c', _COLD, str(runtime), *(str(v) for v in values), 'legacy'],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'mode': 'legacy', 'optional_helper_imported': False, 'protected_gate_count': 11}


def test_cold_legacy_reference_inputs_still_reject_surface_proposal(tmp_path):
    runtime = archived_runtime(tmp_path / 'runtime')
    values = packages(tmp_path, documents=interval_documents())
    result = subprocess.run([sys.executable, '-I', '-c', _COLD, str(runtime), *(str(v) for v in values), 'surface'],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['optional_helper_imported'] is False


def test_actual_surface_report_imports_and_validates_optional_helper_only_when_selected(tmp_path, monkeypatch):
    from weft import source_surface_fit_inputs as helper

    calls, imports = [], []
    validate, importing = helper.validate_surface_fit_snapshot, builtins.__import__
    def checked(*args, **kwargs):
        calls.append(kwargs)
        return validate(*args, **kwargs)
    def guarded(name, *args, **kwargs):
        assert name.split('.')[0] not in {'numpy', 'mujoco', 'cv2', 'mediapipe', 'torch', 'pymunk', 'socket', 'subprocess'}, name
        assert name not in {'weft.source_surface_envelopes', 'weft.source_surface_interval_construction',
                            'weft.whole_body_reference_refinement'}, name
        if name == HELPER: imports.append(name)
        return importing(name, *args, **kwargs)
    monkeypatch.setattr(helper, 'validate_surface_fit_snapshot', checked)
    monkeypatch.setattr(builtins, '__import__', guarded)
    spec = importlib.util.spec_from_file_location('_cold_reference_report_dependency_test', SOURCE / 'whole_body_reference_report.py')
    report = importlib.util.module_from_spec(spec); spec.loader.exec_module(report)
    assert imports == [] and calls == []
    documents = interval_documents(); source = tmp_path / 'surface'
    pin = save_study(source, documents)
    result = report.prepare_whole_body_reference_report(source, tmp_path / 'report', expected_manifest_sha256=pin)
    assert result['status'] == 'report_generated'
    assert imports == [HELPER] and calls == [{'source_sha256': documents['inputs/human.json']['source_sha256']}]
    documents['plan.json']['surface_constraints']['weight'] = 17.
    invalid = tmp_path / 'invalid'; pin = save_study(invalid, documents)
    with pytest.raises(ValueError, match='config differs from its frozen descriptor'):
        report.prepare_whole_body_reference_report(invalid, tmp_path / 'rejected', expected_manifest_sha256=pin)
    assert imports == [HELPER, HELPER] and len(calls) == 2
    assert not (tmp_path / 'rejected').exists()
