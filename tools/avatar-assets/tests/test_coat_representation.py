"""Analytic custody/bounds fixtures; not generated assets or appearance approval."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from avatar_assets.coat_representation import summarize_coat, MAX_SUMMARY_BYTES
from avatar_assets import pipeline, pipeline_continuation
from avatar_assets.safety import artifacts, read_json, sha256, write_json

SOURCE = 'a'*64


class CoatRepresentationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name); self.pack = self.parent/'model-0'
        (self.pack/'sources').mkdir(parents=True)
        self.config = {'source_sha256': SOURCE, 'normal_strength': .2}
        self.budget = {'base_export_vertices': 50004, 'cards': 3825, 'exported_corners_per_card': 12,
            'limit': 100000, 'maximum_cards': 3825, 'projected_max_vertices': 100000, 'reserve_vertices': 4096}
        self.routes = {'materials': [{'object': 'Body', 'material': 'Paint', 'admission': 'coherent_uv0',
            'normal': {'normal_uv_index': 0, 'texture_uv_index': 0}}]}
        self.report = {'source_sha256': SOURCE, 'normal_strength': .2,
            'groom': {'native_only': True, 'gltf_export_supported': False, 'render_fiber_upper_bound': 78000},
            'portable_fur': [{'cards': 3825, 'geometry_profile': 'tuft-v1', 'root_summary': {'roots': 3825,
                'eligible_area_m2': 2, 'length_range_m': [.002,.004], 'width_range_m': [.003,.006]}}],
            'normal_uv_export': self.routes, 'portable_fur_export_budget': self.budget}
        self.seal()

    def seal(self):
        for name, value in [('sources/finish-design.json', self.config), ('finish-report.json', self.report),
                            ('normal-uv-export.json', self.routes), ('tuft-export-budget.json', self.budget)]:
            (self.pack/name).write_text(json.dumps(value))
        self.manifest = {'source_sha256': SOURCE, 'artifacts': artifacts(self.pack)}
        (self.pack/'manifest.json').write_text(json.dumps(self.manifest))
        (self.parent/'finish-0.json').write_bytes((self.pack/'sources/finish-design.json').read_bytes())

    def facts(self):
        return summarize_coat(self.pack,self.manifest,SOURCE)

    def test_distinct_representations_and_zero_conservative_headroom(self):
        facts=self.facts()
        self.assertEqual(facts['native_groom']['render_fiber_upper_bound'],78000)
        self.assertTrue(facts['native_groom']['native_only']); self.assertFalse(facts['native_groom']['export_supported'])
        self.assertEqual(facts['portable_geometry']['total_cards'],3825)
        self.assertEqual(facts['export_capacity']['incremental_split_headroom_vertices'],0)
        self.assertIn('not one-to-one',facts['capabilities'][0])
        self.assertLess(len(json.dumps(facts).encode()),MAX_SUMMARY_BYTES)

    def test_absent_undercoat_is_not_absent_normal_map_or_flat_surface(self):
        facts=self.facts()
        self.assertFalse(facts['undercoat']['configured']); self.assertFalse(facts['undercoat']['reported'])
        self.assertEqual(facts['normal_detail']['authored_strength'],.2)
        self.assertTrue(facts['normal_detail']['routes'][0]['normal_map_reported'])
        self.assertEqual(facts['normal_detail']['original_source_strength'],{'status':'unavailable'})
        self.assertIn('does not mean absent normal',facts['capabilities'][1])

    def test_configured_undercoat_and_reported_counts_are_separate(self):
        self.config['fur_undercoat']={'profile':'short-pile-normal-v1','strength':.3,'min_faces':1,'max_faces':4}
        self.report['fur_undercoat']={'profile':'short-pile-normal-v1','selected_triangles':3,'geometry_displaced':False}
        self.seal(); facts=self.facts()['undercoat']
        self.assertTrue(facts['configured']);self.assertTrue(facts['reported'])
        self.assertEqual(facts['reported_measurements']['selected_triangles'],3)
        self.assertNotIn('geometry_displaced',facts['reported_measurements'])

    def test_positive_capacity_is_not_uv_admission(self):
        self.budget.update(cards=3800,projected_max_vertices=99700)
        self.report['portable_fur'][0]['cards']=3800
        self.seal(); facts=self.facts()
        self.assertEqual(facts['export_capacity']['incremental_split_headroom_vertices'],300)
        self.assertIn('require a probe',facts['capabilities'][2])
        self.assertNotIn('approved',facts['export_capacity'])

    def test_every_consumed_file_is_hash_bound(self):
        for name in ('sources/finish-design.json','finish-report.json','normal-uv-export.json','tuft-export-budget.json'):
            with self.subTest(name=name):
                original=(self.pack/name).read_bytes();(self.pack/name).write_bytes(original+b' ')
                with self.assertRaisesRegex(ValueError,'binding changed'):self.facts()
                (self.pack/name).write_bytes(original)

    def test_unbound_present_file_and_duplicate_reference_rejected(self):
        changed=deepcopy(self.manifest);changed['artifacts']=[r for r in changed['artifacts'] if r['path']!='finish-report.json']
        with self.assertRaisesRegex(ValueError,'without manifest binding'):summarize_coat(self.pack,changed,SOURCE)
        self.manifest['artifacts'].append(self.manifest['artifacts'][0])
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.facts()

    def test_wrong_source_and_report_disagreement_rejected(self):
        self.report['source_sha256']='b'*64;self.seal()
        with self.assertRaisesRegex(ValueError,'source binding'):self.facts()
        self.report['source_sha256']=SOURCE;self.report['normal_uv_export']={'materials':[]};self.seal()
        with self.assertRaisesRegex(ValueError,'routes differ'):self.facts()

    def test_budget_arithmetic_and_card_counts_are_not_trusted_blindly(self):
        self.budget['projected_max_vertices']=99999;self.seal()
        with self.assertRaisesRegex(ValueError,'arithmetic'):self.facts()
        self.budget['projected_max_vertices']=100000;self.report['portable_fur'][0]['cards']=1;self.seal()
        with self.assertRaisesRegex(ValueError,'count differs'):self.facts()

    def test_legacy_without_reports_and_missing_fields_remain_unavailable(self):
        self.assertEqual(summarize_coat(self.pack,{},SOURCE)['status'],'unavailable')
        self.report={'source_sha256':SOURCE};self.routes={'materials':[]}
        self.seal();facts=self.facts()
        self.assertEqual(facts['native_groom']['status'],'unavailable')
        self.assertEqual(facts['portable_geometry']['status'],'unavailable')
        self.assertEqual(facts['status'],'partial')
        self.assertNotIn('total_cards',facts['portable_geometry'])

    def test_lists_and_numbers_are_bounded(self):
        self.report['portable_fur']=[{}]*5;self.seal()
        with self.assertRaisesRegex(ValueError,'four bands'):self.facts()
        self.report['portable_fur']=[];self.routes['materials']=[{}]*33;self.seal()
        with self.assertRaisesRegex(ValueError,'32materials'):self.facts()
        self.routes['materials']=[];self.report['groom']['render_fiber_upper_bound']=True;self.seal()
        with self.assertRaisesRegex(ValueError,'render_fiber'):self.facts()

    def test_oversized_and_symlinked_inputs_rejected_before_read(self):
        path=self.pack/'finish-report.json';path.write_bytes(b' '*(1024*1024+1))
        with self.assertRaisesRegex(ValueError,'Input too large'):self.facts()
        path.unlink();path.symlink_to(self.pack/'sources/finish-design.json')
        with self.assertRaisesRegex(ValueError,'Not a regular'):self.facts()

    def test_both_summary_consumers_preserve_sources_and_legacy_diagnostics(self):
        before={row['path']:row['sha256'] for row in artifacts(self.pack)}
        guard=patch.object(pipeline.Pipeline,'_diagnostics',side_effect=AssertionError('Summary must not alter or invoke diagnostics'))
        guarded=guard.start();self.addCleanup(guard.stop)
        pinned={row['path']:row for row in pipeline_continuation._inventory(self.parent)}
        historical=pipeline_continuation._portable_geometry_facts(self.parent,self.pack,self.manifest,pinned,self.config,0)
        receipt={'model_manifest_sha256':sha256(self.pack/'manifest.json'),'review':{'summary':'Analytic repair'},'evidence':[]}
        path=self.parent/'review-0.json';write_json(path,receipt)
        current=pipeline._current_visual_memory(SOURCE,self.pack,path,receipt)
        self.assertEqual(historical['coat_representation'],current['coat_representation'])
        guarded.assert_not_called()
        self.assertEqual({row['path']:row['sha256'] for row in artifacts(self.pack)},before)
        self.assertLess(len(json.dumps(historical).encode()),32*1024)
        self.assertLess(len(json.dumps(current).encode()),64*1024)
        current['coat_representation']['undercoat']['configured']=True
        self.assertFalse(historical['coat_representation']['undercoat']['configured'])


if __name__=='__main__':unittest.main()
