"""Analytic full-inventory custody regression; no provider or generated assets."""
import json
from pathlib import Path
import unittest
import test_repair_visual_inputs as fixture
from avatar_assets.pipeline_continuation import MAX_FILES,_inventory
from avatar_assets.safety import read_json,write_json,sha256
from avatar_assets import repair_visual_inputs as proposed

class RepairVisualInventoryTests(unittest.TestCase):
    def test_lineage_beyond_1024_preserves_real_bound_images(self):
        f=fixture.RepairVisualInputsTests();f.setUp();self.addCleanup(f.doCleanups)
        original=f.continuation()
        padding=f.parent/'retained';padding.mkdir()
        for n in range(1025): (padding/f'{n}.json').write_text('{}')
        receipt=read_json(f.child/'continuation/receipt.json');receipt['inventory']=_inventory(f.parent)
        self.assertGreater(len(receipt['inventory']),1024)
        (f.child/'continuation/receipt.json').write_text(json.dumps(receipt))
        f.request['continuation']['sha256']=sha256(f.child/'continuation/receipt.json')
        actual=proposed.from_continuation(f.root,f.child,f.request,f.context,f.guides)
        self.assertEqual(actual,original)

    def test_parent_boundary_and_duplicate_checks(self):
        rows=[{'path':str(n)} for n in range(MAX_FILES)]
        self.assertEqual(len(proposed._pins(rows,maximum=MAX_FILES)),MAX_FILES)
        with self.assertRaisesRegex(ValueError,'inventory bound'):proposed._pins(rows+[{'path':'extra'}],maximum=MAX_FILES)
        with self.assertRaisesRegex(ValueError,'duplicate'):proposed._pins([rows[0],rows[0]],maximum=MAX_FILES)

    def test_final_evidence_default_bound_unchanged(self):
        rows=[{'path':str(n)} for n in range(1024)]
        self.assertEqual(len(proposed._pins(rows)),1024)
        with self.assertRaisesRegex(ValueError,'inventory bound'):proposed._pins(rows+[{'path':'extra'}])

    def test_selected_image_tamper_still_rejected_in_proposal(self):
        f=fixture.RepairVisualInputsTests();f.setUp();self.addCleanup(f.doCleanups)
        Path(f.evidence[0]['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed artifact'):
            proposed.from_continuation(f.root,f.child,f.request,f.context,f.guides)

if __name__=='__main__':unittest.main(verbosity=2)
