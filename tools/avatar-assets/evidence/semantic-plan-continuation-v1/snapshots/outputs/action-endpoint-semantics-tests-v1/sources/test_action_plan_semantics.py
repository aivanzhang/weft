"""Planner semantic regressions; copied rejected state is evidence, not an accepted asset."""
from copy import deepcopy
import json
import unittest

from avatar_assets.action_intent import PlanIntentError, required_kinds, validate_state_intents

# Immutable source plan SHA7c4b9c1a078513dcc448dda68884c8ca3a916ddf182259eb9d6d948405c28f0c.
SAVED_STATE = json.loads('{"id": "dance", "meaning": "A cheerful four-second toy dance with readable alternating side-to-side beats, small whole-body hops, coordinated arm swings, and head tilts, ending back in the starting neutral pose.", "requirements": [{"description": "The main body shifts side to side in three clear alternating beats over the clip, creating a readable dance pattern rather than standing nearly still.", "id": "body_rhythm", "kind": "rhythm", "role": "body"}, {"description": "The body performs small visible hops with brief upward travel and return, with the feet allowed to lift during the hops.", "id": "body_hops", "kind": "travel", "role": "body"}, {"description": "Both arms swing in gentle coordinated motion relative to the torso so the arm movement is visible and not just carried by overall body travel.", "id": "arms_swing", "kind": "articulation", "role": "arms"}, {"description": "The head tilts in coordination with the dance and moves relative to the torso, remaining visibly distinct from whole-body travel alone.", "id": "head_tilts", "kind": "articulation", "role": "head"}, {"description": "The torso remains readable and supportive while the appendages move, without requiring simulated balance or external support interaction.", "id": "torso_steady", "kind": "stability", "role": "torso"}, {"description": "By the end of the clip, the character returns to the neutral starting pose established at the beginning.", "id": "return_neutral", "kind": "landing", "role": "body"}], "roles": [{"description": "the full panda character body for overall dance travel and return", "id": "body"}, {"description": "both arms as the swinging appendages", "id": "arms"}, {"description": "the head for coordinated tilts", "id": "head"}, {"description": "the torso as the stable central mass during appendage motion", "id": "torso"}], "user_request": "Perform a cheerful four-second dance with three clear alternating side-to-side beats, small whole-body hops, gentle coordinated arm swings and head tilts. The arms and head must move relative to the torso, not just travel with the entire character. Return to the neutral starting pose by the end. Feet may lift for the hops. No props or interactions. This is an authored toy dance, not simulated balance or physical contact."}')


class PlanSemanticsTests(unittest.TestCase):
    def test_exact_rejected_dance_plan_cannot_add_a_stationary_torso(self):
        before=deepcopy(SAVED_STATE)
        with self.assertRaises(PlanIntentError) as caught:
            validate_state_intents([SAVED_STATE],[SAVED_STATE['user_request']])
        self.assertEqual(caught.exception.code,'unrequested_stability')
        self.assertEqual(SAVED_STATE,before)

    def test_return_to_neutral_cannot_be_relabelled_landing(self):
        state=deepcopy(SAVED_STATE)
        state['requirements']=[r for r in state['requirements'] if r['kind']!='stability']
        state['roles']=[r for r in state['roles'] if r['id']!='torso']
        with self.assertRaises(PlanIntentError) as caught:
            validate_state_intents([state],[state['user_request']])
        self.assertEqual(caught.exception.code,'unrequested_landing')
        state['requirements'][-1]['kind']='return_pose'
        self.assertEqual(validate_state_intents([state],[state['user_request']]),[state])
        state['requirements'].pop()
        with self.assertRaises(PlanIntentError) as caught:
            validate_state_intents([state],[state['user_request']])
        self.assertEqual(caught.exception.code,'missing_return_pose')

    def test_affirmative_return_grounding_preserves_negations_and_actual_landings(self):
        for text in ('Return to the neutral starting pose by the end.',
                     'Return to its original pose.', 'Go back to the start pose.', 'Return to neutral.'):
            with self.subTest(text=text):self.assertIn('return_pose',required_kinds(text))
        for text in ('Do not return to the starting pose.', 'Wave without returning to neutral.',
                     'Return the borrowed book to the shelf.'):
            with self.subTest(text=text):self.assertNotIn('return_pose',required_kinds(text))
        self.assertEqual(required_kinds('Land softly after flying, then return to the starting pose.'),
                         {'landing','travel','return_pose'})

    def test_return_semantic_error_does_not_reclassify_malformed_contracts(self):
        malformed=deepcopy(SAVED_STATE);malformed['id']='Bad ID'
        with self.assertRaises(ValueError) as caught:
            validate_state_intents([malformed],[malformed['user_request']])
        self.assertNotIsInstance(caught.exception,PlanIntentError)


if __name__=='__main__':unittest.main()
