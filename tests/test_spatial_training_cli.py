"""Training limits are explicit without starting the learner in CLI checks."""

import sys
from types import SimpleNamespace

import pytest

from weft import cli


@pytest.mark.parametrize('options,seconds', [([], 90), (['--max-seconds', '12.5'], 12.5)])
def test_training_limit_reaches_existing_learner(monkeypatch, options, seconds):
    calls = []
    monkeypatch.setitem(sys.modules, 'weft.spatial_learning', SimpleNamespace(
        train_spatial_policy=lambda *args, **kwargs: calls.append((args, kwargs)) or {'directory': 'new/policy'}))
    monkeypatch.setattr(sys, 'argv', ['weft', 'spatial-train', '--run', 'saved/scene', '--out', 'new/policy', *options])
    cli.main()
    assert calls == [(('saved/scene', 'new/policy'), {
        'seed': 1729, 'eval_seed': 938401, 'generations': 8, 'population': 16,
        'eval_episodes': 12, 'max_seconds': seconds})]


def test_rejected_training_source_returns_no_success_output(monkeypatch, capsys):
    def reject(*args, **kwargs):
        raise ValueError('Source scene is not sealed')

    monkeypatch.setitem(sys.modules, 'weft.spatial_learning', SimpleNamespace(train_spatial_policy=reject))
    monkeypatch.setattr(sys, 'argv', ['weft', 'spatial-train', '--run', 'saved/scene', '--out', 'new/policy'])
    with pytest.raises(SystemExit) as error:
        cli.main()
    captured = capsys.readouterr()
    assert error.value.code == 1 and captured.out == ''
    assert 'Source scene is not sealed' in captured.err
