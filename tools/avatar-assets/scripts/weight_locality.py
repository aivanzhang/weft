"""Anatomical locality for authored skin influences; deterministic, judged by the gates.

The author paints regional weights well but a handful of vertices keep stray
influences from far-away bones (a hip vertex carrying forearm weight), and the
fallback transfer can pair them with the fallback bone — exactly the cross-body
blends the deformation gate rejects. This enforces one invariant before the gate
runs: a bone only influences vertices near its own segment, relative to the
vertex's nearest influence, with a tighter tube for distal bones that travel
farthest. Vertices left empty rebind rigidly to their nearest influence bone.
The deformation gate still judges the result; nothing here relaxes it.
"""

POLICY = {
    "profile": "authored-weight-locality-v1",
    "relative_factor": 2.5,
    "absolute_floor": 0.12,
    "distal_markers": ("hand", "foot", "_lower"),
    "distal_factor": 1.6,
    "distal_floor": 0.05,
}


def segment_distance(point, head, tail):
    direction = [tail[i] - head[i] for i in range(3)]
    length2 = sum(d * d for d in direction)
    offset = [point[i] - head[i] for i in range(3)]
    if length2 < 1e-12:
        return sum(o * o for o in offset) ** 0.5
    t = sum(offset[i] * direction[i] for i in range(3)) / length2
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return sum((offset[i] - direction[i] * t) ** 2 for i in range(3)) ** 0.5


def prune_influences(point, weights, segments, diameter):
    """Return (kept weights dict, pruned count); rebinds to the nearest bone when emptied.

    weights: {bone: weight}; segments: {bone: (head, tail)} in the same space as point;
    diameter: model bounding size setting the absolute floors.
    """
    if not weights:
        return weights, 0
    distances = {name: segment_distance(point, *segments[name]) for name in weights}
    nearest = min(distances.values())
    kept = {}
    for name, weight in weights.items():
        distal = any(marker in name for marker in POLICY["distal_markers"])
        factor = POLICY["distal_factor"] if distal else POLICY["relative_factor"]
        floor = diameter * (POLICY["distal_floor"] if distal else POLICY["absolute_floor"])
        if distances[name] <= max(factor * nearest, floor):
            kept[name] = weight
    pruned = len(weights) - len(kept)
    if not kept:
        name = min(segments, key=lambda n: segment_distance(point, *segments[n]))
        kept = {name: 1.0}
    return kept, pruned
