"""Run only Weft's fixed pipeline, never submitted code or user-selected commands."""
import json
import os
from pathlib import Path
import resource
import sys


def main():
    directory = Path(sys.argv[1]).resolve()
    resource.setrlimit(resource.RLIMIT_CPU, (150, 150))
    resource.setrlimit(resource.RLIMIT_AS, (1536 * 1024**2, 1536 * 1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (128 * 1024**2, 128 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    import cv2
    from weft.gallery import result_record
    from weft.pipeline import run
    cv2.setNumThreads(1)
    source = directory / 'upload.mp4'
    cap = cv2.VideoCapture(str(source))
    fps, count, width, height = (cap.get(k) for k in (cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not (1 <= fps <= 60 and 10 <= count <= 450 and count / fps <= 15 and 0 < width <= 1920 and 0 < height <= 1080):
        raise ValueError('unsupported_input')
    run(source, directory / 'run', episodes=15000, eval_episodes=300, provider='local')
    result = result_record(directory / 'run')
    (directory / 'result.json').write_text(json.dumps(result, allow_nan=False))
    import zipfile
    # The download is a fixed artifact allowlist, not arbitrary files or application credentials.
    with zipfile.ZipFile(directory / 'environment.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in (directory / 'run').rglob('*'):
            if path.is_file(): archive.write(path, str(path.relative_to(directory / 'run')))
        for name in ('__init__.py', 'world.py', 'env.py', 'brand.py', 'assets/palette.json'):
            archive.write(Path(__file__).resolve().parents[1] / 'src/weft' / name, 'weft/' + name)
        archive.writestr('requirements.txt', 'numpy>=2.2,<3\ngymnasium>=1.1,<2\n')
        archive.writestr('example.py', 'import json\nfrom weft.world import World\nfrom weft.env import RollingEnv\nenv = RollingEnv(World.from_dict(json.load(open("world.json"))))\nobservation, info = env.reset(seed=1729)\nprint(env.step(1))\n')
        archive.writestr('README.txt', 'Weft rolling-body environment\n\nInstall requirements.txt, then run python example.py from this folder. The fixed Weft runtime is included. Load world.json with weft.world.World.from_dict, then pass it to weft.env.RollingEnv.\nUse reset(seed=...) and step(action): 0=left, 1=coast, 2=right.\nTask controls and the parking goal are authored extensions, not recovered actions.\nSource: https://github.com/kintsugi-software/weft\nmanifest.json verifies the full run files; example.py and the weft/ runtime are added for convenience.\n')


if __name__ == '__main__':
    main()
