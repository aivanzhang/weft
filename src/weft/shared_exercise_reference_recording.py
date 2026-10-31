"""Replay a pinned source-fit recipe, without admitting a new body or physical claim.

Scene/task validation belongs to the existing recording boundary. This helper
checks compact execution recipes and current code bytes; it never builds a model.
"""

from functools import partial
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat

from weft import shared_exercise_reference_controller as controller


MAX_DOCUMENT_BYTES = controller.MAX_RECIPE_BYTES
MAX_IMPLEMENTATION_BYTES = 1 << 20
IMPLEMENTATION_FILES = (
    "shared_exercise_reference_controller.py", "shared_exercise_reference_recording.py",
    "human.py", "shared_world.py",
)
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _require(condition, message):
    if not condition:
        raise ValueError("Recorded reference controller: " + message)


def _read_implementation(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        _require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= MAX_IMPLEMENTATION_BYTES,
                 "implementation must be a bounded regular source file")
        raw = stream.read(before.st_size + 1)
        after = os.fstat(stream.fileno())
    _require(len(raw) == before.st_size and
             (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
             (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
             "implementation changed while reading")
    return raw


def _implementation_hashes():
    package = Path(__file__).absolute().parent
    _require(Path(controller.__file__).absolute() == package / IMPLEMENTATION_FILES[0],
             "provider must come from the current package")
    hashes = {name: sha256(_read_implementation(package / name)).hexdigest() for name in IMPLEMENTATION_FILES}
    _require(all(hashes[name] == expected for name, expected in controller.RUNTIME_SHA256.items()),
             "unsupported human feedback or velocity runtime")
    return hashes


def _trusted_velocity_index():
    # Import only after the reviewed runtime bytes and recording pins match.
    from weft import human

    _require(Path(human.__file__).absolute() == Path(__file__).absolute().parent / "human.py",
             "velocity selector must come from the current package")
    return human.reference_velocity_index


def _document(raw):
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_DOCUMENT_BYTES,
             "recipe requires at most 1 MiB of exact UTF-8 bytes")
    text = raw.decode("utf-8")

    def unique(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate recipe field")
            result[key] = value
        return result

    value = json.loads(text, object_pairs_hook=unique)
    json.dumps(value, allow_nan=False)
    _require(type(value) is dict, "recipe must be a JSON object")
    return text, value


def _prepare(raw, scene, task, *, expected=None, action_contract=None):
    text, recipe = _document(raw)
    hashes = _implementation_hashes()
    _require(expected is None or hashes == expected, "recorded implementation pins differ from this runtime")
    action = controller.reference_action_contract(scene, task) if action_contract is None else action_contract
    provider = controller.prepare_reference_recipe(recipe, scene=scene, task=task, action_contract=action,
        runtime_sha256={name: hashes[name] for name in controller.RUNTIME_SHA256},
        reference_velocity_index=_trusted_velocity_index())
    return {"document_utf8": text, "sha256": sha256(raw).hexdigest(), "implementation_sha256": hashes}, provider


def _envelope(envelope):
    _require(type(envelope) is dict and set(envelope) == {"document_utf8", "sha256", "implementation_sha256"},
             "envelope fields differ from the contract")
    text, digest, hashes = (envelope[key] for key in ("document_utf8", "sha256", "implementation_sha256"))
    _require(type(text) is str and 0 < len(text) <= MAX_DOCUMENT_BYTES
             and type(digest) is str and _SHA.fullmatch(digest), "invalid document or SHA256")
    _require(type(hashes) is dict and set(hashes) == set(IMPLEMENTATION_FILES)
             and all(type(value) is str and _SHA.fullmatch(value) for value in hashes.values()),
             "implementation pins require exactly four SHA256 identities")
    raw = text.encode("utf-8")
    _require(len(raw) <= MAX_DOCUMENT_BYTES and sha256(raw).hexdigest() == digest,
             "recipe hash or byte bound differs")
    return raw, hashes


def reference_controller_envelope(document_bytes, scene, task):
    """Pin a compact execution recipe against an already validated scene/task."""
    try:
        return _prepare(document_bytes, scene, task)[0]
    except (OSError, UnicodeError, TypeError, KeyError, AttributeError, RecursionError, OverflowError) as error:
        raise ValueError("Recorded reference controller: invalid recipe or runtime") from error


def validate_reference_controller_envelope(envelope, scene, task):
    """Validate saved bytes and code pins, not the original proposal's admission."""
    try:
        raw, hashes = _envelope(envelope)
        return _prepare(raw, scene, task, expected=hashes)[0]
    except (OSError, UnicodeError, TypeError, KeyError, AttributeError, RecursionError, OverflowError) as error:
        raise ValueError("Recorded reference controller: invalid envelope or runtime") from error


def prepare_recorded_reference_controller(envelope, env):
    """Bind once; each action uses only the provider's bounded public feedback."""
    try:
        raw, hashes = _envelope(envelope)
        _, provider = _prepare(raw, env.scene, env.task, expected=hashes, action_contract=env.action_contract())
        return partial(provider.action, env)
    except (OSError, UnicodeError, TypeError, KeyError, AttributeError, RecursionError, OverflowError) as error:
        raise ValueError("Recorded reference controller: invalid envelope or environment") from error
