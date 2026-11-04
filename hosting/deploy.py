"""Serialize Weft releases without starting local services."""
import fcntl
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parent


def avatar_interpreter():
    """The avatar normalizer needs numpy, which the system interpreter usually lacks."""
    for candidate in (root.parent / '.venv-lab/bin/python', root.parent / '.venv/bin/python',
                      Path(sys.executable)):
        if candidate.is_file() and subprocess.run(
                [candidate, '-c', 'import numpy'], capture_output=True).returncode == 0:
            return candidate
    raise SystemExit('No interpreter with numpy found; create .venv-lab before deploying.')


with (Path(tempfile.gettempdir()) / 'weft-cloudflare-deployment.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    subprocess.run([avatar_interpreter(), 'build_avatars.py'], cwd=root, check=True)
    subprocess.run(['node', 'build-assets.mjs'], cwd=root, check=True)
    subprocess.run(['node', 'node_modules/wrangler/bin/wrangler.js', 'deploy', *sys.argv[1:]], cwd=root, check=True)
