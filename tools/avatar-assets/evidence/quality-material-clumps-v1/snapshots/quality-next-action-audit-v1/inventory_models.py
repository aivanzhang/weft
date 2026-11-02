"""Complete the named model inventory without copying the oversized fox recipe."""
import json
import resource
import time

from inventory import MODELS, OUT, ROOT, VIEWED, census, record

started = time.monotonic()
result = {'schema_version': 1, 'status': 'running', 'models': {},
          'prior_failure': 'inventory.json: fox recipe exceeds the small-snapshot cap; retained by hash only',
          'scope': 'Known models only; no recursive listing, texture decode or geometric validation'}
try:
    for label, (relative, expected) in MODELS.items():
        item = record(relative)
        if expected and item['sha256'] != expected:
            raise ValueError('Named model changed: ' + label)
        result['models'][label] = {**item, **census(relative), 'expected_sha256_matched': True if expected else None}
    result['viewed_images'] = [record(path) for path in VIEWED]
    result['large_recipe_bound_reference'] = record('examples/scenes/fox-hd-v3.json')
    result['status'] = 'recorded'
except Exception as error:
    result['status'] = 'failed'
    result['failure'] = {'type': type(error).__name__, 'message': str(error)}
    raise
finally:
    result['elapsed_seconds'] = time.monotonic() - started
    result['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['memory_scope'] = 'macOS ru_maxrss, this read-only Python process only'
    with (OUT / 'model-inventory.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
