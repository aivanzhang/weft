"""Exercise the output-local numeric gate without importing or launching Blender."""

import ast
import math
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/"sources/native"))
from deformation_diagnostics import diagnose_stretched_edges


def require(value, message):
    if not value:
        raise ValueError(message)


tree = ast.parse((HERE/"native_inspection.py").read_text())
node = next(value for value in tree.body if isinstance(value, ast.FunctionDef) and value.name == "measure_edges")
scope = {"np": np, "math": math, "require": require, "diagnose_stretched_edges": diagnose_stretched_edges,
         "RATIO_LIMIT": 8., "EXCESS_HEIGHT_RATIO": .03}
exec(compile(ast.Module(body=[node], type_ignores=[]), "actual-native-edge-function", "exec"), scope)
measure = scope["measure_edges"]


class NativeEdgeContractTests(unittest.TestCase):
    def test_severe_fifth_outlier_cannot_hide_behind_four_longer_nonsevere_edges(self):
        rest, posed = [], []
        for index in range(5):
            a = [index*2., 0., 0.]
            rest.extend([a, [a[0]+(.1 if index < 4 else .001), 0., 0.]])
            posed.extend([a, [a[0]+(.7 if index < 4 else .051), 0., 0.]])
        rest.append([0., 0., 1.])
        posed.append([0., 0., 1.])
        edges = np.arange(10, dtype=np.int32).reshape(5, 2)
        report = measure(np.asarray(rest), np.asarray(posed), edges, 1.)
        self.assertTrue(all(row["stretch_ratio"] < 8 for row in report["deformation_outliers"]))
        self.assertEqual(report["severe_edge_count"], 1)
        self.assertEqual(report["severe_edges"][0]["edge_index"], 4)
        self.assertFalse(report["passed_no_sampled_severe_spike"])

    def test_unchanged_and_rigidly_translated_meshes_pass_edge_only_measurement(self):
        rest = np.array([[0., 0., 0.], [1., 0., 0.], [0., 0., 1.]])
        edges = np.array([[0, 1], [1, 2], [2, 0]])
        for posed in (rest.copy(), rest+np.array([.1, .2, .3])):
            report = measure(rest, posed, edges, 1.)
            self.assertEqual(report["severe_edge_count"], 0)
            self.assertTrue(report["passed_no_sampled_severe_spike"])

    def test_both_strict_thresholds_are_required(self):
        rest = np.array([[0., 0., 0.], [1., 0., 0.], [0., 0., 1.]])
        posed = rest.copy()
        posed[1, 0] = 8.
        self.assertEqual(measure(rest, posed, np.array([[0, 1]]), 1.)["severe_edge_count"], 0)
        posed[1, 0] = 8.1
        self.assertEqual(measure(rest, posed, np.array([[0, 1]]), 1.)["severe_edge_count"], 1)
        rest[1, 0], posed[1, 0] = .001, .029
        self.assertEqual(measure(rest, posed, np.array([[0, 1]]), 1.)["severe_edge_count"], 0)

    def test_zero_edges_and_nonfinite_geometry_are_not_fabricated_spikes(self):
        rest = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]])
        posed = rest.copy()
        posed[1, 0] = 1.
        result = measure(rest, posed, np.array([[0, 1]]), 1.)
        self.assertEqual(result["ignored_tiny_rest_edges"], 1)
        self.assertEqual(result["severe_edge_count"], 0)
        posed[1, 0] = np.nan
        with self.assertRaises(ValueError):
            measure(rest, posed, np.array([[0, 1]]), 1.)


if __name__ == "__main__":
    unittest.main()
