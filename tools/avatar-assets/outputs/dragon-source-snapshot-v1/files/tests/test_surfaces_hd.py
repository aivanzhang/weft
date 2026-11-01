"""Native HD procedural texture checks; not evidence of reference reconstruction."""

import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np

from scripts import blender_surfaces_hd as hd


def settings(pattern="fur", resolution=1024, **changes):
    return {"pattern": pattern, "resolution": resolution, "strength": .8, "seed": 17, **changes}


def measure_surface(pattern, resolution):
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.monotonic()
    base, normal = hd.surface_pixels_hd([.71, .4, .16, 1], settings(pattern, resolution))
    seconds = time.monotonic()-started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        baseline *= 1024
        peak *= 1024
    # Detail left after a 512px reduction is a screen, not a visual-quality score.
    channel = normal[:, :, 0]
    stride = resolution//512
    reduced = channel.reshape(512, stride, 512, stride).mean(axis=(1, 3))
    restored = reduced.repeat(stride, axis=0).repeat(stride, axis=1)
    residual = float(np.mean(np.abs(channel-restored)))
    return {"pattern": pattern, "resolution": resolution, "elapsed_seconds": round(seconds, 6),
            "peak_rss_bytes": peak, "incremental_peak_rss_bytes": peak-baseline,
            "returned_array_bytes": base.nbytes+normal.nbytes, "normal_detail_residual_from_512": residual}


class HDSurfaceTests(unittest.TestCase):
    def test_invalid_settings_fail_before_array_allocation(self):
        invalid = [settings(resolution=512), settings(resolution=4096), settings(resolution=1024.0),
                   settings(resolution=True), settings(pattern="expressions"), settings(pattern=[]),
                   settings(strength=-.01), settings(strength=1.1), settings(strength=float("nan")),
                   settings(strength=True), settings(seed=-1), settings(seed=2**31), settings(seed=True),
                   settings(seed="17"), {**settings(), "script": "not executable"}, {"pattern": "fur"}]
        for value in invalid:
            with self.subTest(settings=value), patch.object(hd.np, "empty", side_effect=AssertionError("allocated")):
                with self.assertRaises(ValueError):
                    hd.surface_pixels_hd([.7, .4, .1, 1], value)

    def test_invalid_color_is_rejected(self):
        for value in ([1, 1, 1], [1, 1, 1, 2], [float("inf"), 0, 0, 1], [True, 0, 0, 1],
                      ["1", 0, 0, 1], [-.1, 0, 0, 1], {"r": 1}):
            with self.subTest(color=value), self.assertRaises(ValueError):
                hd.surface_pixels_hd(value, settings())

    def test_all_patterns_have_finite_srgb_and_unit_tangent_normals(self):
        for pattern in sorted(hd.PATTERNS):
            with self.subTest(pattern=pattern):
                base, normal = hd.surface_pixels_hd([.71, .4, .16, .75], settings(pattern))
                self.assertEqual(base.shape, (1024, 1024, 4))
                self.assertEqual(base.dtype, np.float32)
                self.assertEqual(normal.dtype, np.float32)
                self.assertTrue(np.isfinite(base).all() and np.isfinite(normal).all())
                self.assertTrue((base >= 0).all() and (base <= 1).all())
                self.assertTrue((normal >= 0).all() and (normal <= 1).all())
                np.testing.assert_array_equal(base[:, :, 3], .75)
                np.testing.assert_array_equal(normal[:, :, 3], 1)
                np.testing.assert_allclose(np.linalg.norm(normal[:, :, :3]*2-1, axis=2), 1, atol=3e-7)
                self.assertGreater(float(np.std(base[:, :, 0])), .0002)
                self.assertGreater(float(np.std(normal[:, :, 0])), .0002)

    def test_reproducibility_and_seed_changes(self):
        for pattern in sorted(hd.PATTERNS):
            with self.subTest(pattern=pattern):
                first = hd.surface_pixels_hd([.65, .33, .17, 1], settings(pattern))
                second = hd.surface_pixels_hd([.65, .33, .17, 1], settings(pattern))
                np.testing.assert_array_equal(first[0], second[0])
                np.testing.assert_array_equal(first[1], second[1])
                del second
                changed = hd.surface_pixels_hd([.65, .33, .17, 1], settings(pattern, seed=18))
                self.assertFalse(np.array_equal(first[0], changed[0]))
                self.assertFalse(np.array_equal(first[1], changed[1]))

    def test_zero_strength_is_exact_flat_input_color(self):
        color = [.71, .4, .16, .75]
        for pattern in sorted(hd.PATTERNS):
            with self.subTest(pattern=pattern):
                base, normal = hd.surface_pixels_hd(color, settings(pattern, strength=0))
                for channel in range(4):
                    np.testing.assert_array_equal(base[:, :, channel], np.float32(color[channel]))
                    np.testing.assert_array_equal(normal[:, :, channel], [.5, .5, 1, 1][channel])

    def test_tile_size_does_not_create_seams_or_change_pixels(self):
        for pattern in ("fur", "scales", "iris"):
            with self.subTest(pattern=pattern):
                reference = hd.surface_pixels_hd([.7, .4, .15, 1], settings(pattern))
                with patch.object(hd, "ROW_BATCH", 53):
                    tiled = hd.surface_pixels_hd([.7, .4, .15, 1], settings(pattern))
                np.testing.assert_array_equal(reference[0], tiled[0])
                np.testing.assert_array_equal(reference[1], tiled[1])

    def test_fur_has_short_directional_detail_without_long_repeating_bands(self):
        base, _ = hd.surface_pixels_hd([.7, .4, .15, 1], settings("fur", strength=1))
        value = base[:, :, 0]
        across = float(np.mean(np.abs(np.diff(value, axis=1))))
        along = float(np.mean(np.abs(np.diff(value, axis=0))))
        self.assertGreater(across/along, 1.4)
        self.assertLess(float(np.std(value.mean(axis=0)))/float(np.std(value)), .2)

    def test_iris_has_centered_pupil_and_outer_limbal_depth(self):
        base, normal = hd.surface_pixels_hd([.7, .4, .15, 1], settings("iris", strength=1))
        axis = (np.arange(1024, dtype=np.float32)+.5)/1024-.5
        radius = np.sqrt(axis[:, None]**2+axis[None, :]**2)
        value = base[:, :, 0]
        pupil = float(value[radius < .1].mean())
        colored_ring = float(value[(radius > .2) & (radius < .37)].mean())
        limbal = float(value[(radius > .46) & (radius < .48)].mean())
        self.assertLess(pupil, colored_ring*.3)
        self.assertLess(limbal, colored_ring*.6)
        self.assertLess(float(np.max(np.abs(normal[radius < .1, :3]-[.5, .5, 1]))), 1e-6)

    def test_cooperative_deadline_is_enforced(self):
        with patch.object(hd.time, "monotonic", side_effect=[0, hd.MAX_SECONDS+1]):
            with self.assertRaisesRegex(RuntimeError, "20-second deadline"):
                hd.surface_pixels_hd([.7, .4, .15, 1], settings())

    def test_native_1024_2048_detail_and_fresh_process_resource_bounds(self):
        root = Path(__file__).resolve().parents[1]
        command = "from tests.test_surfaces_hd import measure_surface; import json,sys; print(json.dumps(measure_surface(sys.argv[1],int(sys.argv[2]))))"
        measurements = []
        for pattern in sorted(hd.PATTERNS):
            for resolution in (1024, 2048):
                with self.subTest(pattern=pattern, resolution=resolution):
                    completed = subprocess.run([sys.executable, "-c", command, pattern, str(resolution)],
                                               cwd=root, capture_output=True, text=True, timeout=30, check=True)
                    measured = json.loads(completed.stdout)
                    measurements.append(measured)
                    self.assertLess(measured["peak_rss_bytes"], 250_000_000)
                    self.assertLess(measured["elapsed_seconds"], hd.MAX_SECONDS)
                    self.assertEqual(measured["returned_array_bytes"], resolution*resolution*4*4*2)
                    self.assertGreater(measured["normal_detail_residual_from_512"], .00025)
        print("HD_SURFACE_MEASUREMENTS="+json.dumps(measurements, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
