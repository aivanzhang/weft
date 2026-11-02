"""Analytic task-custody checks; no native render or completed-avatar claims."""

from copy import deepcopy
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.native_render_tasks import render_or_record, require, validate_task


def fixture(root):
    matrix = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
    scene = NS(name='Scene', frame_current=13, frame_subframe=.25,
        render=NS(engine='CYCLES', resolution_percentage=100, film_transparent=False,
                  image_settings=NS(file_format='PNG', color_mode='RGB'),
                  filepath=str(root/'renders/greet/0000.png'), resolution_x=720, resolution_y=720,
                  fps=24, fps_base=1.),
        cycles=NS(device='CPU', samples=16, use_denoising=True),
        view_settings=NS(view_transform='AgX'),
        camera=NS(name='Camera', matrix_world=deepcopy(matrix),
                  data=NS(type='ORTHO', ortho_scale=1.8, clip_start=.01, clip_end=100.)))
    rig = NS(name='Rig', matrix_world=deepcopy(matrix), data=NS(pose_position='POSE'),
             pose=NS(bones=[NS(name='Body', matrix=deepcopy(matrix))]))
    return scene, rig, NS(name='greet'), [(NS(name='Rig', bl_rna=NS(identifier='Object')), NS(identifier='OBRig'))]


class NativeRenderTaskTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bpy = NS(context=NS(view_layer=NS(update=Mock())), ops=NS(render=NS(render=Mock())))
        patcher = patch.dict(sys.modules, {'bpy': self.bpy}); patcher.start(); self.addCleanup(patcher.stop)

    def capture(self):
        scene, rig, action, bindings = fixture(self.root)
        tasks = []
        result = render_or_record(scene, rig, action, bindings, tasks, self.root, hash_output=True)
        self.assertIsNone(result)
        return scene, rig, action, bindings, tasks

    def test_preparation_captures_subframe_action_and_quality_without_rendering(self):
        scene, rig, _, _, tasks = self.capture()
        task = tasks[0]
        self.assertEqual(task['frame'], 13.25)
        self.assertEqual(task['bindings'], [{'kind': 'Object', 'name': 'Rig', 'slot': 'OBRig'}])
        self.assertEqual((task['resolution'], task['samples'], task['denoise']), ([720, 720], 16, True))
        self.assertEqual(task['path'], 'renders/greet/0000.png')
        scene.camera.matrix_world[0][3] = 50
        rig.pose.bones[0].matrix[1][3] = 50
        self.assertEqual(task['camera']['matrix_world'][0][3], 0)
        self.assertEqual(task['bone_matrices']['Body'][7], 0)
        self.bpy.ops.render.render.assert_not_called()
        self.assertFalse((self.root/'renders').exists())

    def test_capture_rejects_duplicate_and_escaped_outputs_without_rendering(self):
        scene, rig, action, bindings, tasks = self.capture()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            render_or_record(scene, rig, action, bindings, tasks, self.root)
        scene.render.filepath = str(self.root.parent/'escaped.png')
        with self.assertRaises(ValueError):
            render_or_record(scene, rig, action, bindings, [], self.root)
        self.bpy.ops.render.render.assert_not_called()

    def test_capture_refuses_unsupported_render_semantics_and_task_growth(self):
        scene, rig, action, bindings = fixture(self.root)
        for group, field, value in ((scene.render, 'film_transparent', True),
                                    (scene.render, 'resolution_percentage', 50),
                                    (scene.cycles, 'device', 'GPU')):
            previous = getattr(group, field); setattr(group, field, value)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'render state'):
                render_or_record(scene, rig, action, bindings, [], self.root)
            setattr(group, field, previous)
        with self.assertRaisesRegex(ValueError, '80tasks'):
            render_or_record(scene, rig, action, bindings, [{}]*80, self.root)
        self.bpy.ops.render.render.assert_not_called()

    def test_plan_admission_rejects_path_matrix_binding_and_resource_corruption(self):
        _, _, _, _, tasks = self.capture()
        for kind in ('path', 'matrix', 'bindings', 'frame', 'samples', 'resolution', 'bone_proof'):
            task = deepcopy(tasks[0])
            if kind == 'path': task['path'] = '../image.png'
            if kind == 'matrix': task['camera']['matrix_world'][0][0] = float('nan')
            if kind == 'bindings': task['bindings'] = []
            if kind == 'frame': task['frame'] = float('inf')
            if kind == 'samples': task['samples'] = 65
            if kind == 'resolution': task['resolution'] = [4096, 4096]
            if kind == 'bone_proof': task['bone_matrices']['Body'] = [1.]*15
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_task(task)

    def test_existing_immediate_render_path_keeps_the_render_operation(self):
        scene, rig, action, bindings = fixture(self.root)
        render_or_record(scene, rig, action, bindings, None, self.root)
        self.bpy.ops.render.render.assert_called_once_with(write_still=True)


class NativeStageAdmissionTests(unittest.TestCase):
    def function(self, name, filename, namespace):
        path = Path(__file__).resolve().parents[1]/'scripts'/filename
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        return namespace[name]

    def test_staging_rejects_imported_actions_before_loading_source(self):
        header = Mock(side_effect=RuntimeError('source boundary'))
        finish = self.function('finish', 'blender_finish_generated.py',
                               {'Path': Path, 'json': json, 'source_header': header})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'config.json'
            for cfg in ({}, {'authored_rig': {}}, {'authored_actions': [{}]}):
                path.write_text(json.dumps(cfg))
                with self.subTest(config=cfg), self.assertRaisesRegex(ValueError, 'bone-only'):
                    finish(path, temporary, prepare_only=True)
            header.assert_not_called()

    def test_worker_rejects_existing_or_dangling_outputs_before_scene_load(self):
        open_scene = Mock(side_effect=RuntimeError('native load boundary'))
        digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        render = self.function('render', 'blender_render_prepared.py',
            {'Path': Path, 'json': json, 'time': NS(monotonic=lambda: 0),
             'hash_file': digest, 'require': require, 'validate_task': validate_task,
             'bpy': NS(ops=NS(wm=NS(open_mainfile=open_scene)))})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root/'prepared-source.blend'
            source.write_bytes(b'ANALYTIC NOT A BLEND FILE')
            scene, rig, action, bindings = fixture(root)
            fake = NS(context=NS(view_layer=NS(update=Mock())))
            with patch.dict(sys.modules, {'bpy': fake}):
                tasks = []; render_or_record(scene, rig, action, bindings, tasks, root)
            plan = root/'plan.json'
            plan.write_text(json.dumps({'schema_version': 1, 'status': 'prepared_not_rendered',
                'scene': source.name, 'scene_sha256': digest(source), 'tasks': tasks}))
            for name in ('image.png', 'before-render.json', 'render-report.json'):
                for symlink in (False, True):
                    out = root/(name+str(symlink)); out.mkdir()
                    target = out/name
                    if symlink: target.symlink_to(out/'missing')
                    else: target.write_text('preserve')
                    with self.subTest(name=name, symlink=symlink), self.assertRaisesRegex(ValueError, 'new regular files'):
                        render(plan, digest(plan), 0, out)
            open_scene.assert_not_called()


if __name__ == '__main__':
    unittest.main()
