import assert from 'node:assert/strict';
import test from 'node:test';
import { sandboxCommand } from './run.mjs';

test('sandbox uses its Python environment and canonical project state, independent of caller cwd', () => {
  const command = sandboxCommand([], '/example/weft');
  assert.equal(command.executable, '/example/weft/.venv-lab/bin/python');
  assert.equal(command.cwd, '/example/weft');
  assert.deepEqual(command.args, ['-m', 'weft', 'portal', '--root', '.local/portal', '--port', '8765']);
});

test('custom project paths stay single subprocess arguments without shell interpretation', () => {
  const command = sandboxCommand(['--root', 'saved worlds/$(not-a-command)', '--port=8766', '--check']);
  assert.equal(command.args[4], 'saved worlds/$(not-a-command)');
  assert.equal(command.args[6], '8766');
  assert.equal(command.check, true);
});

test('invalid or incomplete options fail before launching Python', () => {
  for (const args of [['--port', '-1'], ['--port=65536'], ['--port=1.2'], ['--root'], ['--root', '--check'], ['--host=0.0.0.0']]) {
    assert.throws(() => sandboxCommand(args));
  }
  assert.equal(sandboxCommand(['--port', '0']).args[6], '0');
});
