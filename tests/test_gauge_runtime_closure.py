"""Optional axial reconstruction stays in every native and saved-receipt closure."""

import json
from pathlib import Path

import pytest

from test_joint_twist_gauge import construct
from test_reconstructed_camera_coupling import inputs, local_native_only, package, source_files
from weft import camera_shared_candidate as candidate
from weft.camera_shared_policy_receipt import policy_runtime_requirements
from weft.camera_shared_receipt import BODY_RUNTIME_FILES, RUNTIME_REQUIRED, body_runtime_files, runtime_required
from weft.io import digest, write_json
from weft.pipeline import verify_artifacts
from weft.shared_exercise_lab import load_shared_exercise_lab, prepare_shared_exercise_lab, runtime_files


@pytest.mark.parametrize('version', [1, 2, 3])
def test_optional_dependency_reaches_lab_manual_and_policy_receipts(version):
    body = {'camera_reference': {'body_refinement': {'joint_pixel_fit': {'uncertainty': {'schema_version': version}}}}}
    scene = {'profile': 'shared-world-v6', 'human_spec': body,
             'camera_coupling': {'profile': 'camera-shared-coupling-v2', 'body_reconstruction': {}}}
    extra = ('limb_twist_gauge.py',) if version == 3 else ()
    assert body_runtime_files(body) == (*BODY_RUNTIME_FILES, *extra)
    assert runtime_required(scene) == RUNTIME_REQUIRED | set(BODY_RUNTIME_FILES) | set(extra)
    manual = runtime_files(scene)
    assert ('limb_twist_gauge.py' in manual) == (version == 3)
    for required in policy_runtime_requirements(scene):
        assert ('limb_twist_gauge.py' in required) == (version == 3)
    scene['camera_coupling'] = {'profile': 'camera-shared-coupling-v1'}
    assert runtime_required(scene) == RUNTIME_REQUIRED
    assert 'limb_twist_gauge.py' not in runtime_files(scene)


@pytest.mark.parametrize('body', [None, {}, {'camera_reference': None},
    {'camera_reference': {'body_refinement': {'joint_pixel_fit': None}}}])
def test_legacy_metadata_does_not_acquire_an_optional_dependency(body):
    assert body_runtime_files(body) == BODY_RUNTIME_FILES


def test_contract_marker_cannot_omit_gauge_pin():
    body = {'camera_reference': {'body_refinement': {'joint_pixel_fit': {
        'contract': {'method': 'time-coupled-joint-pixels-v3'}}}}}
    assert 'limb_twist_gauge.py' in body_runtime_files(body)


def test_full_gauged_body_is_packaged_and_reopens_with_exact_runtime(inputs, tmp_path):
    body, audit = construct(inputs[3])
    source = package(tmp_path / 'source', source_files(inputs[3], body, audit))
    before = verify_artifacts(source)
    output = tmp_path / 'candidate'
    result = candidate.prepare_camera_shared_candidate(source, output)
    assert result['status'] == 'constructed', result
    runtime = Path(candidate.__file__).parent
    for name in ('limb_twist_gauge.py', 'camera_shared_receipt.py'):
        assert (output / 'candidate-implementation' / name).read_bytes() == (runtime / name).read_bytes()
    lab = tmp_path / 'lab'
    prepare_shared_exercise_lab(output, lab, root_assistance=True)
    loaded = load_shared_exercise_lab(lab / 'run')
    assert loaded['scene']['human_spec'] == body
    pins = loaded['contract']['runtime_sha256']
    assert pins['limb_twist_gauge.py'] == digest(runtime / 'limb_twist_gauge.py')
    assert runtime_required(loaded['scene']) <= pins.keys()
    assert verify_artifacts(source) == before
    assert loaded['task']['control']['root_assistance'] == 'explicit_3_axis'
    assert loaded['task']['scope']['physical_validation'] == 'unverified'
    # Dropping a dependency remains invalid even if a caller reseals the contract.
    contract_path = lab / 'run' / 'exercise-lab.json'
    contract = json.loads(contract_path.read_bytes())
    del contract['runtime_sha256']['limb_twist_gauge.py']
    write_json(contract_path, contract)
    manifest_path = lab / 'run' / 'manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    manifest['files']['exercise-lab.json'] = digest(contract_path)
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match='runtime'):
        load_shared_exercise_lab(lab / 'run')
