import { spawn } from 'node:child_process';
import { access } from 'node:fs/promises';
import { constants } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repository = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

export function sandboxCommand(args, root = repository) {
  let projectRoot = '.local/portal';
  let port = '8765';
  let check = false;
  let help = false;
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (argument === '--check') { check = true; continue; }
    if (argument === '--help' || argument === '-h') { help = true; continue; }
    const match = /^(--root|--port)(?:=(.*))?$/.exec(argument);
    if (!match) throw new Error(`Unknown sandbox option: ${argument}`);
    const value = match[2] ?? args[++index];
    if (!value || value.startsWith('--')) throw new Error(`${match[1]} requires a value`);
    if (match[1] === '--root') projectRoot = value;
    else port = value;
  }
  if (!/^\d+$/.test(port) || Number(port) > 65535) {
    throw new Error('--port must be an integer from 0 to 65535');
  }
  return {
    executable: path.join(root, '.venv-lab', 'bin', 'python'),
    args: ['-m', 'weft', 'portal', '--root', projectRoot, '--port', port],
    cwd: root,
    check,
    help,
  };
}

async function main() {
  const command = sandboxCommand(process.argv.slice(2));
  if (command.help) {
    console.log('Usage: npm run sandbox -- [--root PATH] [--port PORT] [--check]');
    console.log('Defaults: .local/portal, 127.0.0.1:8765. --check verifies the launcher without starting a server.');
    return;
  }
  try {
    await access(command.executable, constants.X_OK);
  } catch {
    throw new Error('Sandbox Python is unavailable. Follow sandbox/README.md to set up .venv-lab.');
  }
  if (command.check) {
    console.log(JSON.stringify(command, null, 2));
    return;
  }
  // Keep relative credentials, budget ledgers and saved projects at their canonical repository paths.
  const child = spawn(command.executable, command.args, { cwd: command.cwd, stdio: 'inherit' });
  const stop = () => child.kill('SIGINT');
  process.once('SIGINT', stop);
  process.once('SIGTERM', stop);
  child.once('error', (error) => { console.error(error.message); process.exitCode = 1; });
  child.once('exit', (code, signal) => {
    process.removeListener('SIGINT', stop);
    process.removeListener('SIGTERM', stop);
    process.exitCode = code ?? (signal ? 1 : 0);
  });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => { console.error(error.message); process.exitCode = 1; });
}
