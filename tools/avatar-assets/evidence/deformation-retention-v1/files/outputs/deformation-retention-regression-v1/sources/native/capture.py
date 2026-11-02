"""Exact bounded capture function from the preceding immutable native diagnostic."""
import bpy
import numpy as np
from blender_verify import require
MAX_VERTICES, MAX_EDGES, MAX_TRIANGLES = 100000, 300000, 100000

def capture(obj):
    graph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(graph)
    mesh = evaluated.to_mesh()
    try:
        require(0 < len(mesh.vertices) <= MAX_VERTICES and 0 < len(mesh.edges) <= MAX_EDGES
                and 0 < len(mesh.polygons) <= MAX_TRIANGLES and len(mesh.loops) == len(mesh.polygons)*3,
                "Imported evaluated geometry exceeds diagnostic bounds")
        points = np.empty((len(mesh.vertices), 3), dtype=np.float64)
        mesh.vertices.foreach_get("co", points.ravel())
        matrix = np.asarray(evaluated.matrix_world, dtype=np.float64)
        points = points@matrix[:3, :3].T+matrix[:3, 3]
        edges = np.empty((len(mesh.edges), 2), dtype=np.int32)
        mesh.edges.foreach_get("vertices", edges.ravel())
        loops = np.empty(len(mesh.loops), dtype=np.int32)
        mesh.loops.foreach_get("vertex_index", loops)
        starts, totals = np.empty(len(mesh.polygons), dtype=np.int32), np.empty(len(mesh.polygons), dtype=np.int32)
        mesh.polygons.foreach_get("loop_start", starts)
        mesh.polygons.foreach_get("loop_total", totals)
        require(np.all(totals == 3) and np.isfinite(points).all() and np.max(np.abs(points)) <= 100,
                "Nonfinite/unbounded positions or nontriangle topology")
        require(edges.min() >= 0 and edges.max() < len(points) and loops.min() >= 0 and loops.max() < len(points),
                "Invalid imported topology indices")
        return points, edges, loops, starts, totals
    finally:
        evaluated.to_mesh_clear()
