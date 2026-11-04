"""Serialize Weft releases without starting local services."""
import fcntl
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parent
with (Path(tempfile.gettempdir()) / 'weft-cloudflare-deployment.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    subprocess.run([sys.executable, 'build_avatars.py'], cwd=root, check=True)
    subprocess.run(['node', 'build-assets.mjs'], cwd=root, check=True)
    subprocess.run(['node', 'node_modules/wrangler/bin/wrangler.js', 'deploy', *sys.argv[1:]], cwd=root, check=True)
