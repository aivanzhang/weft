"""Analytic validation fixtures only; the separate portrait probe uses native OIDN."""

import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock
import zlib

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import denoise_portrait as denoise


class PortraitDenoiseValidation(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input = self.root/"input.png"
        self.original = np.array([[[250,30,10,0],[20,80,140,96]],[[1,2,3,255],[40,50,60,200]]],dtype=np.uint8)
        Image.fromarray(self.original).save(self.input)

    def tearDown(self):
        self.temporary.cleanup()

    def test_square_rgb_and_rgba_decode_preserves_raw_bytes(self):
        before = self.input.read_bytes()
        pixels,info = denoise.read_pixels(self.input)
        self.assertTrue(np.array_equal(pixels,self.original))
        self.assertEqual(info["sha256"],hashlib.sha256(before).hexdigest())
        self.assertEqual(self.input.read_bytes(),before)
        rgb = self.root/"rgb.png"
        Image.fromarray(self.original[:,:,:3]).save(rgb)
        self.assertEqual(denoise.read_pixels(rgb)[0].shape,(2,2,3))

    def test_oversized_ihdr_is_rejected_without_decoder_allocation(self):
        payload = struct.pack(">IIBBBBB",2049,2049,8,6,0,0,0)
        self.input.write_bytes(b"\x89PNG\r\n\x1a\n"+struct.pack(">I",13)+b"IHDR"+payload
                               +struct.pack(">I",zlib.crc32(b"IHDR"+payload)))
        with mock.patch.object(Image,"open") as decoder:
            with self.assertRaisesRegex(ValueError,"2048"):
                denoise.inspect_png(self.input)
            decoder.assert_not_called()

    def test_sparse_oversized_file_is_rejected_before_header_decode(self):
        with self.input.open("r+b") as stream:
            stream.truncate(denoise.MAX_INPUT_BYTES+1)
        with self.assertRaisesRegex(ValueError,"32 MiB"):
            denoise.inspect_png(self.input)

    def test_non_square_and_non_rgb_png_are_explicitly_unsupported(self):
        for image,reason in ((Image.new("RGBA",(3,2)),"square"),(Image.new("L",(2,2)),"RGB/RGBA")):
            image.save(self.input)
            with self.assertRaisesRegex(ValueError,reason):
                denoise.inspect_png(self.input)

    def test_symlink_and_invalid_magic_are_rejected(self):
        link = self.root/"link.png"
        link.symlink_to(self.input)
        with self.assertRaisesRegex(ValueError,"non-symlink"):
            denoise.inspect_png(link)
        self.input.write_bytes(b"not a png"*10)
        with self.assertRaisesRegex(ValueError,"IHDR"):
            denoise.inspect_png(self.input)

    def test_animated_png_is_rejected(self):
        Image.fromarray(self.original).save(self.input,save_all=True,
            append_images=[Image.fromarray(np.flip(self.original,axis=0))],duration=100,loop=0)
        with self.assertRaisesRegex(ValueError,"Animated PNG"):
            denoise.read_pixels(self.input)

    def test_linear_gamma_and_embedded_icc_are_not_silently_treated_as_srgb(self):
        metadata = PngInfo()
        metadata.add(b"gAMA",struct.pack(">I",100000))
        Image.fromarray(self.original).save(self.input,pnginfo=metadata)
        with self.assertRaisesRegex(ValueError,"gamma"):
            denoise.read_pixels(self.input)
        Image.fromarray(self.original).save(self.input,icc_profile=b"analytic unsupported ICC fixture")
        with self.assertRaisesRegex(ValueError,"ICC"):
            denoise.read_pixels(self.input)

    def test_alpha_and_hidden_rgb_survive_changed_filtered_colors(self):
        filtered = np.full((2,2,3),.5,dtype=np.float32)
        output = denoise.compose_pixels(self.original,filtered)
        self.assertTrue(np.array_equal(output[:,:,3],self.original[:,:,3]))
        self.assertTrue(np.array_equal(output[0,0],self.original[0,0]))
        self.assertTrue(np.all(output[1,0,:3] == 128))
        self.assertTrue(np.array_equal(self.original[1,0],[1,2,3,255]))

    def test_nonfinite_and_wrong_shape_filter_output_are_rejected(self):
        for output in (np.full((2,2,3),np.nan,dtype=np.float32),np.zeros((2,2,4),dtype=np.float32)):
            with self.assertRaisesRegex(ValueError,"shape/type/values"):
                denoise.compose_pixels(self.original,output)

    def test_worker_roundtrip_preserves_alpha_and_source_with_analytic_filter_stub(self):
        class AnalyticFilter:
            def apply(self,rgb):
                return np.ascontiguousarray(rgb*.8+.1,dtype=np.float32),{"name":"analytic validation stub, not OIDN"}
        output,report = self.root/"output.png",self.root/"worker.json"
        before = self.input.read_bytes()
        with mock.patch.object(denoise,"OIDN",return_value=AnalyticFilter()):
            result = denoise.worker(self.input,output,report)
        pixels,info = denoise.read_pixels(output)
        self.assertEqual(self.input.read_bytes(),before)
        self.assertTrue(np.array_equal(pixels[:,:,3],self.original[:,:,3]))
        self.assertTrue(result["alpha_unchanged"] and result["raw_input_unchanged"])
        self.assertEqual(result["status"],"needs_visual_review")
        self.assertEqual(result["output"]["sha256"],info["sha256"])
        self.assertEqual(json.loads(report.read_text())["reference_images_used"],[])

    def test_existing_outputs_are_never_overwritten_or_sent_to_native_code(self):
        output = self.root/"existing.png"
        output.write_bytes(b"preserve previous output")
        with mock.patch.object(denoise,"OIDN") as native:
            with self.assertRaisesRegex(ValueError,"already exists"):
                denoise.worker(self.input,output,self.root/"report.json")
            native.assert_not_called()
        self.assertEqual(output.read_bytes(),b"preserve previous output")

    def test_output_plan_requires_explicit_new_isolated_folder(self):
        output,report = self.root/"job/portrait.png",self.root/"job/report.json"
        self.assertEqual(denoise.output_plan(output,report),(output,report))
        with self.assertRaisesRegex(ValueError,"share one"):
            denoise.output_plan(output,self.root/"other/report.json")
        output.parent.mkdir()
        with self.assertRaisesRegex(ValueError,"new job directory"):
            denoise.output_plan(output,report)
        with self.assertRaisesRegex(ValueError,"reserved"):
            denoise.output_plan(self.root/"new/portrait.png",self.root/"new/worker-result.json")


if __name__ == "__main__":
    unittest.main()
