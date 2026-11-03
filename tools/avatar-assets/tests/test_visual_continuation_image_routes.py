"""Analytic image selection custody; no asset generation, native run or provider call."""
from pathlib import Path
import json
import shutil
import unittest
from PIL import Image
from avatar_assets import pipeline_continuation as cont
from avatar_assets.safety import read_json,sha256
import test_visual_repair_continuation as visual_fixtures


class VisualContinuationImageRoutesTests(unittest.TestCase):
    setUp=visual_fixtures.VisualContinuationTests.setUp
    prepare=visual_fixtures.VisualContinuationTests.prepare
    make_terminal=visual_fixtures.VisualContinuationTests.make_terminal
    visual=visual_fixtures.VisualContinuationTests.visual
    seal=staticmethod(visual_fixtures.VisualContinuationTests.seal)

    def selecting(self,remove_contacts=True):
        p=self.visual()
        for i,name in enumerate(('guide-front.png','guide-side.png','three-quarter.png')):
            Image.new('RGB',(2,2),(i*30,90,120)).save(p/'inspection'/name)
        self.seal(p/'inspection',**read_json(p/'inspection/manifest.json'))
        terminal=read_json(p/'unsupported.json')
        terminal['findings']['repair_images']=[str(p/'final-evidence-0/front.png')]
        (p/'unsupported.json').write_text(json.dumps(terminal))
        if remove_contacts:shutil.rmtree(p/'evidence-0')
        return p

    def paths(self,p):
        inventory=cont._inventory(p);pins={r['path']:r for r in inventory}
        return cont._visual_repair_paths(p,read_json(p/'request.json'),0,read_json(p/'review-0.json'),pins,inventory,
            sha256(p/'model-0/finish-report.json'))

    def test_selected_paid_images_continue_without_contact_pack(self):
        p=self.selecting();before=cont._inventory(p)
        captured=cont.collect(self.root,p)
        self.assertEqual(captured['context']['findings']['repair_images'],[str(p/'final-evidence-0/front.png')])
        self.assertFalse((p/'evidence-0').exists())
        self.assertEqual(cont._inventory(p),before);self.network.assert_not_called()

    def test_selected_image_tamper_rejected(self):
        p=self.selecting();(p/'final-evidence-0/front.png').write_bytes(b'Changed image')
        with self.assertRaisesRegex(ValueError,'custody changed|image changed'):cont.collect(self.root,p)
        self.network.assert_not_called()

    def test_existing_contacts_cannot_override_nonempty_selection(self):
        p=self.selecting(remove_contacts=False)
        terminal=read_json(p/'unsupported.json');terminal['findings']['repair_images']=[str(p/'evidence-0/front.png')]
        (p/'unsupported.json').write_text(json.dumps(terminal))
        with self.assertRaisesRegex(ValueError,'terminal findings'):cont.collect(self.root,p)

    def test_changed_source_pinned_selection_semantics_rejected(self):
        p=self.selecting();path=p/'sources/repair_visual_inputs.py'
        path.write_text(path.read_text().replace('capacity=min(5,8-len(base_images))','capacity=min(4,8-len(base_images))'))
        with self.assertRaisesRegex(ValueError,'selector semantics changed'):self.paths(p)

    def test_changed_recorded_caller_branch_rejected(self):
        p=self.selecting();path=p/'sources/pipeline.py'
        path.write_text(path.read_text().replace("repair_images = [Path(row['path']) for row in visual_inputs['images']]",
                                              "repair_images = [Path(row['path']) for row in visual_inputs['images'][:1]]"))
        with self.assertRaisesRegex(ValueError,'caller selection semantics changed'):self.paths(p)

    def test_current_selector_missing_guide_does_not_fall_back(self):
        p=self.selecting(remove_contacts=False);(p/'inspection/guide-front.png').unlink()
        self.seal(p/'inspection',**read_json(p/'inspection/manifest.json'))
        with self.assertRaisesRegex(ValueError,'Not a regular'):self.paths(p)

    def test_legacy_no_selector_uses_only_bound_contacts(self):
        p=self.visual()
        # A source-only analytic pre-selector profile; it is never executed.
        (p/'sources/pipeline.py').write_text('def legacy_pipeline():\n    return None\n')
        self.assertEqual(self.paths(p),[str(p/'evidence-0/front.png')])
        contacts=read_json(p/'evidence-0/evidence.json');contacts['finish_report_sha256']='f'*64
        (p/'evidence-0/evidence.json').write_text(json.dumps(contacts))
        with self.assertRaisesRegex(ValueError,'another model'):self.paths(p)


if __name__=='__main__':unittest.main()
