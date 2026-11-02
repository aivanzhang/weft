"""Pure exported-preview engine admission; actual renderer control is separate."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class Render:
    def __init__(self, behavior='accept'):
        self.behavior, self.selected, self.assignments = behavior, 'CYCLES', []

    @property
    def engine(self):
        return self.selected

    @engine.setter
    def engine(self, value):
        self.assignments.append(value)
        if self.behavior == 'unavailable':
            raise TypeError('Unsupported engine')
        if self.behavior == 'accept':
            self.selected = value


class Scene:
    def __init__(self, render=None, controls=None):
        self.render = render or Render()
        self.eevee = controls if controls is not None else SimpleNamespace(taa_render_samples=64)

    @property
    def cycles(self):
        raise AssertionError('Exported preview must not touch native Cycles controls')


class SampleControl:
    def __init__(self, behavior):
        self.behavior, self.value = behavior, 64

    @property
    def taa_render_samples(self):
        return self.value

    @taa_render_samples.setter
    def taa_render_samples(self, value):
        if self.behavior == 'read_only':
            raise AttributeError('Read-only setting')
        self.value = value+1


class ExportPreviewEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1]/'scripts/blender_provider_preview.py'
        node = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'configure_exported_preview_engine')
        scope = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
        cls.configure = staticmethod(scope[node.name])

    def test_actual_engine_and_requested_samples_are_recorded_without_cycles_access(self):
        for samples in (1, 16, 64):
            scene = Scene()
            report = self.configure(scene, samples)
            with self.subTest(samples=samples):
                self.assertEqual(scene.render.engine, 'BLENDER_EEVEE_NEXT')
                self.assertEqual(scene.render.assignments, ['BLENDER_EEVEE_NEXT'])
                self.assertEqual(scene.eevee.taa_render_samples, samples)
                self.assertEqual(report['engine'], scene.render.engine)
                self.assertEqual(report['samples_requested'], samples)
                self.assertEqual(report['samples_actual'], scene.eevee.taa_render_samples)
                self.assertEqual(report['sample_control'], 'scene.eevee.taa_render_samples')
                self.assertEqual(report['policy'], 'gltf-single-sided-culling-eevee-next-v1')
                self.assertIs(report['engine_fallback'], False)

    def test_invalid_work_bounds_fail_before_setting_an_engine(self):
        for samples in (True, False, 0, 65, -1, 4., float('nan'), None, '16'):
            scene = Scene()
            with self.subTest(samples=samples), self.assertRaisesRegex(ValueError, 'integer in 1..64'):
                self.configure(scene, samples)
            self.assertEqual(scene.render.assignments, [])

    def test_missing_engine_raises_without_a_fallback_assignment(self):
        scene = Scene(render=Render('unavailable'))
        with self.assertRaisesRegex(ValueError, 'no engine fallback'):
            self.configure(scene, 4)
        self.assertEqual(scene.render.assignments, ['BLENDER_EEVEE_NEXT'])

    def test_ignored_engine_selection_is_not_reported_as_success(self):
        scene = Scene(render=Render('ignore'))
        with self.assertRaisesRegex(ValueError, 'no engine fallback'):
            self.configure(scene, 4)
        self.assertEqual(scene.eevee.taa_render_samples, 64)
        self.assertEqual(scene.render.assignments, ['BLENDER_EEVEE_NEXT'])

    def test_missing_sample_control_is_explicitly_unsupported(self):
        for scene in (SimpleNamespace(render=Render()), Scene(controls=SimpleNamespace())):
            with self.subTest(scene=scene), self.assertRaisesRegex(ValueError, 'bounded EEVEE Next'):
                self.configure(scene, 4)

    def test_read_only_or_clamped_sample_setting_is_not_silently_accepted(self):
        for behavior in ('read_only', 'clamp'):
            scene = Scene(controls=SampleControl(behavior))
            with self.subTest(behavior=behavior), self.assertRaisesRegex(ValueError, 'bounded EEVEE Next'):
                self.configure(scene, 4)
            self.assertEqual(scene.render.assignments, ['BLENDER_EEVEE_NEXT'])


if __name__ == '__main__':
    unittest.main()
