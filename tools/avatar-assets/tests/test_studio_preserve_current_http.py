"""Analytic HTTP routing only; no provider, native worker or actual saved job."""
import unittest
import test_studio as fixtures


class PreserveCurrentHTTPTests(unittest.TestCase):
    setUp = fixtures.StudioHTTPTests.setUp
    stop = fixtures.StudioHTTPTests.stop
    request = fixtures.StudioHTTPTests.request
    post = fixtures.StudioHTTPTests.post

    def test_exact_explicit_choice_routes_without_changing_ordinary_retry(self):
        def retry(identifier, **kwargs):
            self.manager.calls.append(('retry', identifier, kwargs))
            return self.manager.get(identifier)
        self.manager.retry = retry
        self.assertEqual(self.post(self.job_path+'/retry', {'appearance_policy':'preserve_current'})[0], 202)
        self.assertEqual(self.post(self.job_path+'/retry')[0], 202)
        self.assertEqual(self.manager.calls, [
            ('retry', self.manager.seed_id, {'appearance_policy':'preserve_current'}),
            ('retry', self.manager.seed_id, {})])

    def test_no_path_budget_unrecognized_mode_or_missing_token_can_reach_manager(self):
        self.manager.retry = lambda *args, **kwargs: self.fail('Invalid request reached manager')
        for body in [{'appearance_policy':None}, {'appearance_policy':True},
                     {'appearance_policy':'approved'}, {'appearance_policy':{'path':'/tmp/model'}},
                     {'appearance_policy':'preserve_current','max_calls':99},
                     {'appearance_policy':'preserve_current','baseline':'/tmp/model'},
                     {'appearance_policy':'preserve_current','api_key':'analytic-not-secret'}]:
            with self.subTest(body=body):
                self.assertEqual(self.post(self.job_path+'/retry', body)[0], 400)
        self.assertEqual(self.post(self.job_path+'/retry', {'appearance_policy':'preserve_current'},
            headers={'X-Avatar-Studio-Token':None})[0], 403)


class PreserveCurrentWireTests(unittest.TestCase):
    def test_only_safe_explicit_mode_reaches_public_request(self):
        for mode in ['preserve_current', None, {'path':'/private/baseline'}, 'approved']:
            value = fixtures.studio._wire({'id':'a'*24, 'request':{'prompt':'fixture',
                'appearance_policy':mode, 'appearance_receipt':{'path':'private.json'}}})
            self.assertNotIn('appearance_receipt', value['request'])
            self.assertEqual(value['request'].get('appearance_policy'),
                             'preserve_current' if mode == 'preserve_current' else None)
