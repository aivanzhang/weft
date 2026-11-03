"""Independent job grouping fixtures; no provider, worker or rendered asset."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from avatar_assets.studio_jobs import StudioJobs
from avatar_assets.studio import _scene_wire
from test_studio_jobs import FixtureExecutor


def cards():
    return [dict(id='otter',name='River otter',prompt='One river otter',style='Soft toy',custom_actions=['Roll over'],preset='preview'),
            dict(id='heron',name='Heron',prompt='One blue heron',style='Painted wood',custom_actions=['Spread its wings'],preset='draft')]


class StudioSceneJobsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()/'tools/avatar-assets';self.root.mkdir(parents=True)
        self.executor=FixtureExecutor();self.jobs=StudioJobs(self.root,executor=self.executor)

    def test_pair_creates_distinct_immutable_jobs_without_dispatch(self):
        scene=self.jobs.create_scene('Different animals',cards(),idempotency_key='pair-one')
        self.assertFalse(scene['automatic_decomposition']);self.assertFalse(scene['combined_scene_verified'])
        self.assertEqual(scene['status'],'saved');self.assertEqual(self.executor.calls,[])
        ids=[a['job']['id'] for a in scene['assets']];self.assertEqual(len(set(ids)),2)
        for asset,card in zip(scene['assets'],cards()):
            job=asset['job'];self.assertEqual(job['status'],'created')
            self.assertEqual(job['request']['custom_actions'],card['custom_actions'])
            self.assertEqual(job['scene']['asset_id'],card['id'])
            self.assertFalse(self.jobs.resolve_job(job['id']).exists())
            raw=json.loads((self.jobs.jobs_dir/job['id']/'request.json').read_text())
            self.assertIsNone(raw['reuse_model']);self.assertNotIn('scene',raw)
        self.assertEqual([a['job']['id'] for a in self.jobs.create_scene('Different animals',cards(),idempotency_key='pair-one')['assets']],ids)
        self.assertEqual(self.jobs.list()['total'],2)

    def test_all_cards_validate_before_any_job_is_saved(self):
        for mutate in [lambda a:a[1].update(custom_actions=['x'*601]),lambda a:a[1].update(id=a[0]['id']),
                       lambda a:a[1].update(name=a[0]['name']),lambda a:a[1].update(reuse_model='/tmp/fused.glb'),
                       lambda a:a[1].update(prompt=''),lambda a:a.append(deepcopy(a[0]))]:
            value=cards();mutate(value)
            with self.assertRaises(ValueError):self.jobs.create_scene('Invalid',value,idempotency_key='invalid')
            self.assertEqual(self.jobs.list()['total'],0)

    def test_partial_save_repeats_exact_child_keys_without_dispatch(self):
        original=self.jobs.create;calls=[]
        def interrupted(*args,**kwargs):
            calls.append(kwargs['idempotency_key'])
            if len(calls)==2:raise OSError('Analytic save interruption')
            return original(*args,**kwargs)
        with patch.object(self.jobs,'create',side_effect=interrupted):
            with self.assertRaises(OSError):self.jobs.create_scene('Pair',cards(),idempotency_key='interrupted')
        self.assertEqual(self.jobs.list()['total'],1)
        existing=self.jobs.list()['jobs'][0]['id']
        with self.jobs._db() as db:scene_id=db.execute('SELECT id FROM scenes').fetchone()[0]
        self.assertEqual(self.jobs.get_scene(scene_id)['status'],'partially_saved')
        scene=self.jobs.create_scene('Pair',cards(),idempotency_key='interrupted')
        self.assertEqual(scene['assets'][0]['job']['id'],existing)
        self.assertEqual(self.jobs.list()['total'],2);self.assertEqual(self.executor.calls,[])
        with self.assertRaises(ValueError):self.jobs.create_scene('Changed',cards(),idempotency_key='interrupted')

    def test_references_are_distinct_bound_and_unsafe_second_rejected_before_save(self):
        folder=self.root/'references';folder.mkdir()
        first=folder/'first.png';second=folder/'second.png';first.write_bytes(b'analytic first');second.write_bytes(b'analytic second')
        value=cards();value[0]['reference']=first;value[1]['reference']=second
        scene=self.jobs.create_scene('References',value,idempotency_key='references')
        for asset,source in zip(scene['assets'],[first,second]):
            raw=json.loads((self.jobs.jobs_dir/asset['job']['id']/'request.json').read_text())
            self.assertEqual(raw['reference']['path'],str(source.resolve()))
        link=folder/'link.png';link.symlink_to(second);value[1]['reference']=link
        with self.assertRaises(ValueError):self.jobs.create_scene('Bad reference',value,idempotency_key='link')
        self.assertEqual(self.jobs.list()['total'],2)

    def test_row_progress_never_promotes_other_subject_or_exposes_paths(self):
        scene=self.jobs.create_scene('Separate progress',cards(),idempotency_key='progress')
        first=scene['assets'][0]['job']['id']
        with self.jobs._db(True) as db:
            row=self.jobs._row(db,first);data=json.loads(row['data']);data['error']='Analytic failure'
            self.jobs._update(db,row,data,'unsupported')
        current=_scene_wire(self.jobs.get_scene(scene['id']))
        self.assertEqual([a['job']['status'] for a in current['assets']],['unsupported','created'])
        self.assertNotEqual(current['status'],'complete');self.assertEqual(self.executor.calls,[])
        self.assertNotIn(str(self.root),json.dumps(current))

    def test_reserved_slots_survive_interrupt_and_finish_after_restart(self):
        original=self.jobs.create;count=0
        def interrupted(*args,**kwargs):
            nonlocal count
            count+=1
            if count==2:raise OSError('Analytic interruption')
            return original(*args,**kwargs)
        with patch('avatar_assets.studio_jobs.MAX_JOBS',3):
            with patch.object(self.jobs,'create',side_effect=interrupted):
                with self.assertRaises(OSError):self.jobs.create_scene('Reserved',cards(),idempotency_key='reserved')
            scene_id=self.jobs.list_scenes()['scenes'][0]['id']
            self.jobs.create('One unrelated asset')
            with self.assertRaises(ValueError):self.jobs.create('Would consume reserved slot')
            restarted=StudioJobs(self.root,executor=self.executor)
            resumed=restarted.finish_scene_save(scene_id)
            self.assertEqual(resumed['status'],'saved');self.assertEqual(restarted.list()['total'],3)
            self.assertEqual(self.executor.calls,[])

    def test_scene_before_first_child_remains_discoverable_and_finishable(self):
        with patch.object(self.jobs,'create',side_effect=OSError('Analytic failure before first child')):
            with self.assertRaises(OSError):self.jobs.create_scene('Discovered',cards(),idempotency_key='discovered')
        listing=self.jobs.list_scenes()['scenes'];self.assertEqual(len(listing),1)
        scene=self.jobs.get_scene(listing[0]['id']);self.assertTrue(all(a['job'] is None for a in scene['assets']))
        resumed=self.jobs.finish_scene_save(scene['id']);self.assertEqual(resumed['status'],'saved')
        self.assertEqual(self.executor.calls,[])

    def test_child_reference_race_is_rejected_before_creation_or_binding(self):
        folder=self.root/'references';folder.mkdir();ref=folder/'one.png';ref.write_bytes(b'old analytic')
        value=cards();value[0]['reference']=ref;original=self.jobs._selected;calls=0
        def changing(path,reference):
            nonlocal calls
            if path is not None:
                calls+=1
                if calls==3:ref.write_bytes(b'new larger analytic reference')
            return original(path,reference)
        with patch.object(self.jobs,'_selected',side_effect=changing):
            with self.assertRaisesRegex(ValueError,'Scene child inputs changed'):
                self.jobs.create_scene('Race',value,idempotency_key='reference-race')
        self.assertEqual(self.jobs.list()['total'],0)
        scene_id=self.jobs.list_scenes()['scenes'][0]['id']
        self.assertEqual(self.jobs.get_scene(scene_id)['status'],'partially_saved')
        with self.assertRaises(ValueError):self.jobs.finish_scene_save(scene_id)
        self.assertEqual(self.executor.calls,[])

    def test_both_children_ready_are_not_claimed_as_verified_separate_geometry(self):
        scene=self.jobs.create_scene('No inferred geometry',cards(),idempotency_key='not-separated')
        original=self.jobs.get
        with patch.object(self.jobs,'get',side_effect=lambda identifier:{**original(identifier),'status':'complete'}):
            result=self.jobs.get_scene(scene['id'])
        self.assertEqual(result['status'],'children_complete')
        self.assertIs(result['separation_verified'],False);self.assertIs(result['combined_scene_verified'],False)

    def test_scene_row_cannot_be_redirected_to_an_unrelated_job(self):
        scene=self.jobs.create_scene('Bound pair',cards(),idempotency_key='bound')
        unrelated=self.jobs.create('Unrelated asset')
        with self.jobs._db(True) as db:
            db.execute('UPDATE scene_assets SET job=? WHERE scene=? AND asset=?',(unrelated['id'],scene['id'],'otter'))
        with self.assertRaisesRegex(ValueError,'Scene child request changed'):self.jobs.get_scene(scene['id'])
