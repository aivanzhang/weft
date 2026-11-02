"""Analytic intent/measurement fixtures, not generated assets or semantic acceptance."""

from copy import deepcopy
import unittest

from avatar_assets.action_intent import (
    evaluate_action_evidence, required_kinds, validate_action_bindings,
    validate_custom_actions, validate_state_intents,
)


def state(identifier="dance", text="Dance with rhythmic whole-body hops.", kinds=("rhythm",)):
    return {"id": identifier, "meaning": text, "user_request": text,
            "roles": [{"id": "body", "description": "The requested character's whole body"}],
            "requirements": [{"id": kind, "kind": kind, "role": "body", "description": "Visible " + kind}
                             for kind in kinds]}


def measured(**overrides):
    values = {"scale_m": 1., "weighted_vertices": 1000, "moving_vertices": 500,
              "max_root_relative_displacement_m": .04, "max_world_displacement_m": .1, "max_travel_m": .1,
              "max_vertical_rise_m": .1, "descent_from_peak_m": .1,
              "rhythmic_reversals": 4, "active_control_count": 3}
    values.update(overrides)
    return {"state_id": "dance", "evidence_id": "analytic-skin-measurement", "roles": {"body": values}}


class ActionIntentTests(unittest.TestCase):
    def test_explicit_stationary_requests_need_stability_not_articulation(self):
        exact = "Wave the right hand side to side in a clear greeting while keeping the torso and feet steady."
        item = {"id": "greeting", "meaning": exact, "user_request": exact,
                "roles": [{"id": "hand", "description": "Right hand"}, {"id": "support", "description": "Torso and feet"}],
                "requirements": [{"id": "wave", "kind": "articulation", "role": "hand", "description": "Wave the right hand visibly."},
                                 {"id": "support", "kind": "articulation", "role": "support", "description": "Keep the torso and both feet steady throughout the greeting."}]}
        with self.assertRaisesRegex(ValueError, "Stationary requirement must use stability"):
            validate_state_intents([item], [exact])
        item["requirements"][1]["kind"] = "stability"
        self.assertEqual(validate_state_intents([item], [exact])[0]["user_request"], exact)
        self.assertEqual(required_kinds(exact), {"articulation", "stability"})
        self.assertEqual(required_kinds("Keep a steady rhythm while dancing."), {"rhythm"})
        self.assertNotIn("stability", required_kinds("Walk upright across the room."))
        self.assertEqual(required_kinds("Wave without moving the feet."), {"articulation", "stability"})
        item["requirements"].append({"id": "support-move", "kind": "travel", "role": "support", "description": "Move across the room."})
        with self.assertRaisesRegex(ValueError, "whole-clip role"):
            validate_state_intents([item], [exact])

    def test_stability_uses_fixed_world_displacement_not_positive_movement(self):
        requirement = state(kinds=("stability",))["requirements"]
        stationary = measured(max_world_displacement_m=0, max_root_relative_displacement_m=0,
                              max_travel_m=0, moving_vertices=0, active_control_count=0)
        self.assertTrue(evaluate_action_evidence(requirement, stationary, 1.)["passed"])
        self.assertTrue(evaluate_action_evidence(requirement, measured(max_world_displacement_m=.005), 1.)["passed"])
        self.assertFalse(evaluate_action_evidence(requirement, measured(max_world_displacement_m=.005001), 1.)["passed"])
        self.assertFalse(evaluate_action_evidence(requirement, measured(scale_m=.2, max_world_displacement_m=.00101), 1.)["passed"])
        self.assertFalse(evaluate_action_evidence(requirement, measured(scale_m=10, max_world_displacement_m=.006), 1.)["passed"])
        inherited = measured(max_root_relative_displacement_m=0, max_world_displacement_m=.2, active_control_count=0)
        self.assertFalse(evaluate_action_evidence(requirement, inherited, 1.)["passed"])
        absent = measured(); absent["roles"]["body"].pop("max_world_displacement_m")
        self.assertEqual(evaluate_action_evidence(requirement, absent, 1.)["requirements"][0]["status"], "missing")
        for invalid in (True, float("nan"), float("inf"), -1, 101):
            with self.assertRaises(ValueError):
                evaluate_action_evidence(requirement, measured(max_world_displacement_m=invalid), 1.)
        with self.assertRaises(ValueError):
            evaluate_action_evidence([{**requirement[0], "tolerance_m": 1}], stationary, 1.)

    def test_return_pose_has_bounded_endpoint_only_metric_and_required_coverage(self):
        requirements=state(kinds=('return_pose',))['requirements']
        self.assertTrue(evaluate_action_evidence(requirements,measured(max_final_world_displacement_m=.005),1.)['passed'])
        for changes in ({'max_final_world_displacement_m':.005001},
                        {'scale_m':.2,'max_final_world_displacement_m':.00101},
                        {'scale_m':10.,'max_final_world_displacement_m':.006},
                        {'weighted_vertices':7,'moving_vertices':7,'max_final_world_displacement_m':0}):
            self.assertFalse(evaluate_action_evidence(requirements,measured(**changes),1.)['passed'])
        missing=evaluate_action_evidence(requirements,measured(),1.)
        self.assertEqual(missing['requirements'][0]['status'],'missing')
        for value in (True,-1,float('nan'),float('inf'),101):
            with self.assertRaises(ValueError):
                evaluate_action_evidence(requirements,measured(max_final_world_displacement_m=value),1.)

    def test_exact_custom_actions_survive_order_unicode_and_whitespace(self):
        actions = ["  Dance twice — then bow. ", "Fly left while flapping both wings."]
        self.assertEqual(validate_custom_actions(actions), actions)
        states = [state(text=actions[0]), state("fly", actions[1], ("travel", "articulation"))]
        copy = validate_state_intents(states, actions, ["dance", "fly"])
        copy[0]["user_request"] = "Different"
        self.assertEqual(states[0]["user_request"], actions[0])
        for mutate in (lambda s: s.reverse(), lambda s: s[0].update(user_request=actions[0].strip()),
                       lambda s: s[1].update(id="standing_flap")):
            bad = deepcopy(states); mutate(bad)
            with self.assertRaises(ValueError):
                validate_state_intents(bad, actions, ["dance", "fly"])

    def test_flying_is_not_relabelled_as_stationary_flapping(self):
        intent = state("fly", "Fly across the scene while flapping.", ("articulation",))
        with self.assertRaisesRegex(ValueError, "omitted a required action capability"):
            validate_state_intents([intent], [intent["user_request"]])
        intent["requirements"].append({"id": "travel", "kind": "travel", "role": "body", "description": "Move through the scene"})
        self.assertEqual(len(validate_state_intents([intent], [intent["user_request"]])), 1)
        self.assertEqual(required_kinds("Wave without flying."), {"articulation"})

    def test_bad_actions_and_unsupported_requirements_fail_before_execution(self):
        for actions in (None, "dance", [True], [" "], ["x"*601], ["dance"]*9):
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                validate_custom_actions(actions)
        item = state("paint", "Paint a new picture using a brush.", ("interaction",))
        with self.assertRaisesRegex(ValueError, "unsupported interaction"):
            validate_state_intents([item], [item["user_request"]])
        self.assertEqual(validate_state_intents([item], [item["user_request"]], ready=False), [item])
        with self.assertRaisesRegex(ValueError, "matching counts"):
            validate_state_intents([], ["dance"], ["one", "two"])

    def test_roles_bind_actual_descendants_not_a_convenient_other_root(self):
        intent = state()
        parents = {"Root": None, "Torso": "Root", "Head": "Torso", "Part": "Root", "Wing": "Part"}
        bindings = [{"state_id": "dance", "roles": [{"id": "body", "root_bone": "Root", "bones": ["Root", "Torso"]}]}]
        self.assertEqual(validate_action_bindings(bindings, [intent], parents), bindings)
        bad = deepcopy(bindings); bad[0]["roles"][0].update(root_bone="Part", bones=["Torso"])
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_action_bindings(bad, [intent], parents)
        for bones in (["Invented"], ["Wing", "Wing"], [], ["Wing"]*65):
            bad[0]["roles"][0].update(root_bone="Root", bones=bones)
            with self.assertRaises(ValueError):
                validate_action_bindings(bad, [intent], parents)
        with self.assertRaisesRegex(ValueError, "Cyclic"):
            validate_action_bindings(bindings, [intent], {"Root": "Torso", "Torso": "Root"})

    def test_root_travel_never_proves_wing_articulation(self):
        requirements = state(kinds=("travel", "articulation"))["requirements"]
        audit = measured(max_travel_m=.8, max_root_relative_displacement_m=0., moving_vertices=0)
        result = evaluate_action_evidence(requirements, audit, 1.)
        self.assertEqual([row["status"] for row in result["requirements"]], ["passed", "failed"])
        self.assertFalse(result["passed"])
        self.assertFalse(result["semantic_approval"])
        # A few stray vertices do not turn an otherwise unweighted wing into a success.
        audit = measured(weighted_vertices=1000, moving_vertices=2, max_root_relative_displacement_m=.2)
        self.assertFalse(evaluate_action_evidence(requirements, audit, 1.)["passed"])
        self.assertTrue(evaluate_action_evidence(requirements, measured(), 1.)["passed"])

    def test_small_single_control_nod_is_not_a_dance(self):
        requirements = state()["requirements"]
        for changes in ({"active_control_count": 1}, {"rhythmic_reversals": 1},
                        {"max_travel_m": .001, "max_root_relative_displacement_m": .001}):
            with self.subTest(changes=changes):
                self.assertFalse(evaluate_action_evidence(requirements, measured(**changes), 1.)["passed"])
        self.assertTrue(evaluate_action_evidence(requirements, measured(), 1.)["passed"])

    def test_flight_needs_rise_and_descent_when_requested(self):
        requirements = state(kinds=("takeoff", "landing"))["requirements"]
        result = evaluate_action_evidence(requirements, measured(max_vertical_rise_m=0, descent_from_peak_m=0), 1.)
        self.assertFalse(result["passed"])
        self.assertTrue(evaluate_action_evidence(requirements, measured(), 1.)["passed"])
        result = evaluate_action_evidence(requirements, {"state_id": "dance", "evidence_id": "missing", "roles": {}}, 1.)
        self.assertTrue(all(row["status"] == "missing" for row in result["requirements"]))

    def test_measurements_are_bounded_finite_and_not_author_thresholds(self):
        requirements = state()["requirements"]
        for changes in ({"scale_m": 0}, {"max_travel_m": float("nan")}, {"moving_vertices": True},
                        {"weighted_vertices": 100001}, {"moving_vertices": 1001},
                        {"max_travel_m": 101}, {"override_pass": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                evaluate_action_evidence(requirements, measured(**changes), 1.)
        with self.assertRaises(ValueError):
            evaluate_action_evidence(requirements*9, measured(), 1.)
        for bad in (None, [{"role": []}], [{"role": "body", "kind": "travel"}]):
            with self.assertRaises(ValueError):
                evaluate_action_evidence(bad, measured(), 1.)
        unsupported = evaluate_action_evidence(state(kinds=("expression",))["requirements"], measured(), 1.)
        self.assertEqual(unsupported["requirements"][0]["status"], "unsupported")


if __name__ == "__main__":
    unittest.main()
