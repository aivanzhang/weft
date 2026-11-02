"""Call the public finisher; the parent owns its sole native process."""
import json
import os
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
WIRE = sys.stdout
sys.stdout = sys.stderr


def send(value):
    WIRE.write(json.dumps(value, allow_nan=False) + '\n')
    WIRE.flush()


def native(command, log, output, **limits):
    send({'kind': 'native', 'command': [os.fspath(x) for x in command],
          'log': str(log), 'output': str(output), 'cwd': os.getcwd(), 'requested_limits': limits})
    raw = sys.stdin.buffer.readline(131073)
    if not raw.endswith(b'\n') or len(raw) > 131072:
        raise ValueError('Missing or oversized parent native receipt')
    response = json.loads(raw)
    if set(response) != {'kind', 'job'} or response['kind'] != 'native_result':
        raise ValueError('Unexpected parent response')
    return response['job']


def main():
    from avatar_assets import finish_generated
    prepared = json.loads((HERE / 'prepared.json').read_text())
    # Only native ownership/limits change; public config, build and validators do not.
    finish_generated.run_bounded = native
    try:
        result = finish_generated.finish(HERE / 'finish-config.json', ROOT / prepared['output'])
        send({'kind': 'complete', 'result': result})
    except BaseException as error:
        traceback.print_exc()
        send({'kind': 'error', 'type': type(error).__name__, 'message': str(error)[:4000]})
        raise


if __name__ == '__main__':
    main()
