"""Analytic coordinate fixtures; native guide images remain separate evidence."""

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts.rig_guide_annotations import annotate, digest, guide_projection, normalized_to_pixel


class RigGuideMathTests(unittest.TestCase):
    def setUp(self):
        self.bounds = {'min': [-2, -10, 3], 'max': [4, 14, 9]}

    def test_front_x_and_side_y_increase_right_z_increases_up(self):
        for view, index in (('front', 0), ('side', 1)):
            projection = guide_projection(self.bounds, view)
            middle = normalized_to_pixel([.5, .5, .5], projection)
            for coordinate in middle:
                self.assertAlmostEqual(coordinate, 320)
            point = [.5, .5, .5]
            point[index] = 1
            self.assertGreater(normalized_to_pixel(point, projection)[0], middle[0])
            self.assertEqual(normalized_to_pixel(point, projection)[1], middle[1])
            self.assertLess(normalized_to_pixel([.5, .5, 1], projection)[1], middle[1])
            hidden = 1-index
            point = [.5, .5, .5]
            point[hidden] = -1
            self.assertEqual(normalized_to_pixel(point, projection), middle)

    def test_nonuniform_bounds_are_not_mistaken_for_image_edges(self):
        front = guide_projection(self.bounds, 'front')
        side = guide_projection(self.bounds, 'side')
        self.assertAlmostEqual(front['ortho_scale'], 6*1.18)
        self.assertAlmostEqual(side['ortho_scale'], 24*1.18)
        front_low = normalized_to_pixel([0, .5, 0], front)
        side_low = normalized_to_pixel([.5, 0, 0], side)
        self.assertAlmostEqual(front_low[0], 640*(.18/2)/1.18)
        self.assertAlmostEqual(side_low[0], front_low[0])
        self.assertGreater(front_low[1], side_low[1])
        for low in (front_low, side_low):
            self.assertGreater(low[0], 0)
            self.assertLess(low[1], 640)

    def test_uniform_calibration_preserves_normalized_projection(self):
        transformed = {'min': [v*7.5+20 for v in self.bounds['min']],
                       'max': [v*7.5+20 for v in self.bounds['max']]}
        for view in ('front', 'side'):
            a, b = guide_projection(self.bounds, view), guide_projection(transformed, view)
            for point in ([0, .2, .3], [.75, 1, .1], [1, 0, 1]):
                for x, y in zip(normalized_to_pixel(point, a), normalized_to_pixel(point, b)):
                    self.assertAlmostEqual(x, y)
                raw = normalized_to_pixel(point, a)
                canvas = normalized_to_pixel(point, a, canvas=True)
                self.assertEqual(canvas, [v+64 for v in raw])

    def test_invalid_bounds_and_output_sizes_fail_before_drawing(self):
        cases = [({'min':[0,0,0], 'max':[0,1,1]}, 'front', 640, 64),
                 ({'min':[0,0,0], 'max':[1,float('nan'),1]}, 'side', 640, 64),
                 (self.bounds, 'back', 640, 64), (self.bounds, 'front', 1024, 64),
                 (self.bounds, 'front', 640, 100)]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                guide_projection(*args)

    def test_annotation_preserves_raw_renders_and_records_projection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'original.glb').write_bytes(b'analytic-source-identity-not-a-model')
            views, before = [], {}
            for view in ('front', 'side'):
                path = root/f'guide-{view}-raw.png'
                Image.new('RGB', (640, 640), (225, 225, 225)).save(path)
                before[view] = digest(path)
                views.append({'view':view, 'raw_sha256':before[view],
                              'projection':guide_projection(self.bounds, view)})
            native = {'rig_guides':{'source_sha256':digest(root/'original.glb'),
                                    'bounds_blender_z_up':self.bounds, 'views':views}}
            (root/'native-inspection.json').write_text(json.dumps(native))
            result = annotate(root)
            self.assertTrue(result['raw_model_renders_preserved'])
            self.assertFalse(result['reference_images_edited'])
            self.assertEqual(result['kind'], 'analytic_coordinate_annotations')
            for record in result['views']:
                self.assertEqual(digest(root/record['raw_path']),before[record['view']])
                self.assertAlmostEqual(record['ticks'][2]['horizontal_canvas_x'],384)
                self.assertAlmostEqual(record['ticks'][2]['vertical_canvas_y'],384)
                with Image.open(root/record['path']) as image:
                    self.assertEqual(image.size,(768,768))
                    self.assertEqual(image.mode,'RGB')
                    self.assertNotEqual(image.getpixel((384,384)),(225,225,225))
                    self.assertEqual(image.getpixel((90,90)),(225,225,225))
            with self.assertRaisesRegex(ValueError,'already exists'):
                annotate(root)

    def test_changed_native_render_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'original.glb').write_bytes(b'analytic-source')
            views=[]
            for view in ('front','side'):
                path=root/f'guide-{view}-raw.png'
                Image.new('RGB',(640,640)).save(path)
                views.append({'view':view,'raw_sha256':digest(path),
                              'projection':guide_projection(self.bounds,view)})
            (root/'native-inspection.json').write_text(json.dumps({'rig_guides':{
                'source_sha256':digest(root/'original.glb'),'bounds_blender_z_up':self.bounds,'views':views}}))
            Image.new('RGB',(640,640),'red').save(root/'guide-front-raw.png')
            with self.assertRaisesRegex(ValueError,'render changed'):
                annotate(root)


if __name__ == '__main__':
    unittest.main()
