"""Guard against outcome-only approval, DEV leakage and unsafe TIMEOUT cuts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import review_stage4_quality as quality
from src.imitation.audit import EvidenceError, load_json

RULES = load_json(quality.ROOT / "experiments/stage4/g1_quality_rules_v1.json")


def trajectory(n=150, moving=True, actor=None, passage=None):
    ds, ts = [], []
    for i in range(n):
        features = [1.] * 259
        features[8] = .5
        ds.append({"step_index":i,"canonical_action":[0., .2 if moving else -.1],
                   "decision":{"diagnostics":{"behavior_state":"YIELD"}},
                   "agent_input":{"core_observation":{"features":features},
                     "ego_motion":{"lane_heading_error_rad":0.,"velocity_x_mps":2. if moving else 0.,"velocity_y_mps":0.},
                     "task_context":{"current_lane_width":3.5,"route_passage":passage,
                                     "lane_context":{"lanes":[{"side":"CURRENT","route_compatible":True}]}},
                     "traffic_context":{"overflow_count":0,"motion_paths_overflow_count":0,
                                        "actors":[actor] if actor else [],"validity_mask":[True] if actor else []}}})
        ts.append({"post_vehicle":{"speed_km_h":7.2 if moving else 0.,"position_m":[(i+1)*.2 if moving else 0.,0.]},
                   "pre_vehicle":{"position_m":[i*.2 if moving else 0.,0.]},
                   "step_info":{"route_completion":(i+1)*.001 if moving else .3}})
    return ds,ts


def front_actor(x=10., vx=0., y=0.):
    return {"relative_position_x":x,"relative_position_y":y,"relative_velocity_x":vx,"relative_velocity_y":0.,
            "relative_heading":0.,"length":4.6,"width":1.85}


class TestQualitySelection(unittest.TestCase):
    def test_success_requires_all_windows_not_just_outcome(self):
        d,t=trajectory()
        t[70]["step_info"]["crash_vehicle"]=True
        r=quality.annotate(d,t,"SUCCESS",RULES)
        self.assertFalse(r["whole_success_approved"])
        self.assertEqual(r["eligible_steps"],0)
        self.assertIn("SAFETY_EVENT",r["windows"][1]["reasons"])

    def test_clean_progress_success_includes_short_final_window(self):
        d,t=trajectory(120)
        r=quality.annotate(d,t,"SUCCESS",RULES)
        self.assertTrue(r["whole_success_approved"])
        self.assertEqual(r["eligible_steps"],120)

    def test_timeout_tail_excluded_even_when_progressing(self):
        d,t=trajectory(350)
        r=quality.annotate(d,t,"TIMEOUT",RULES)
        self.assertEqual(r["eligible_steps"],250)
        self.assertEqual([(s["start"],s["end"],s["decision"]) for s in r["intervals"]],[(0,250,"KEEP"),(250,350,"EXCLUDE")])

    def test_timeout_bad_window_has_both_boundary_guards(self):
        d,t=trajectory(500)
        d[175]["agent_input"]["ego_motion"]["lane_heading_error_rad"]=.5
        r=quality.annotate(d,t,"TIMEOUT",RULES)
        keeps=[(s["start"],s["end"]) for s in r["intervals"] if s["decision"]=="KEEP"]
        self.assertEqual(keeps,[(0,100),(250,400)])
        self.assertEqual(sum(s["end"]-s["start"] for s in r["intervals"]),500)

    def test_short_timeout_segment_not_retained(self):
        d,t=trajectory(130)
        self.assertEqual(quality.annotate(d,t,"TIMEOUT",RULES)["eligible_steps"],0)

    def test_safety_failure_never_contributes_even_good_prefix(self):
        d,t=trajectory(350)
        self.assertEqual(quality.annotate(d,t,"CRASH_VEHICLE",RULES)["eligible_steps"],0)

    def test_yield_diagnostic_alone_does_not_justify_wait(self):
        d,t=trajectory(150,moving=False,passage={"exit_distance_m":None})
        r=quality.annotate(d,t,"TIMEOUT",RULES)
        self.assertTrue(all(w["label"]=="UNCERTAIN" for w in r["windows"]))
        self.assertEqual(r["eligible_steps"],0)

    def test_front_actor_and_hold_support_wait(self):
        d,t=trajectory(150,moving=False,actor=front_actor())
        r=quality.annotate(d,t,"TIMEOUT",RULES)
        self.assertTrue(all(w["label"]=="JUSTIFIED_WAIT" for w in r["windows"]))
        self.assertEqual(r["eligible_steps"],50)

    def test_rear_actor_does_not_justify_front_wait(self):
        d,_=trajectory(1,moving=False,actor=front_actor(x=-6.,vx=1.))
        self.assertEqual(quality.public_evidence(d[0],RULES)["wait_support"],[])

    def test_wait_counter_continues_across_window_boundaries(self):
        d,t=trajectory(300,moving=False,actor=front_actor())
        r=quality.annotate(d,t,"TIMEOUT",RULES)
        self.assertEqual(r["windows"][3]["label"],"JUSTIFIED_WAIT")
        self.assertEqual(r["windows"][4]["label"],"UNCERTAIN")
        self.assertEqual(r["windows"][4]["evidence"]["supported_stop_steps"],250)

    def test_clear_stationary_context_flagged_as_suspected_stall(self):
        d,t=trajectory(150,moving=False)
        self.assertEqual(quality.annotate(d,t,"TIMEOUT",RULES)["windows"][0]["label"],"DEADLOCK_OR_LOOP")

    def test_missing_lidar_is_uncertain_even_if_progressing(self):
        d,t=trajectory(150)
        d[75]["agent_input"]["core_observation"]["features"][19:]=[0.]*240
        self.assertEqual(quality.annotate(d,t,"SUCCESS",RULES)["windows"][1]["label"],"UNCERTAIN")

    def test_traffic_overflow_prevents_success_approval(self):
        d,t=trajectory()
        d[75]["agent_input"]["traffic_context"]["overflow_count"]=1
        self.assertFalse(quality.annotate(d,t,"SUCCESS",RULES)["whole_success_approved"])

    def test_route_regression_not_confused_with_travel(self):
        d,t=trajectory()
        t[75]["step_info"]["route_completion"]-=.01
        self.assertIn("ROUTE_REGRESSION_OR_PROJECTION_JUMP",quality.annotate(d,t,"SUCCESS",RULES)["windows"][1]["reasons"])

    def test_braking_distinguished_from_accelerating_at_low_ttc(self):
        d,t=trajectory(150,actor=front_actor(x=8.,vx=-10.))
        self.assertIn("CLOSING_RISK_WITHOUT_BRAKING",quality.annotate(d,t,"SUCCESS",RULES)["windows"][0]["reasons"])
        for row in d: row["canonical_action"][1]=-.5
        self.assertTrue(quality.annotate(d,t,"SUCCESS",RULES)["whole_success_approved"])

    def test_intervals_use_past_context_only(self):
        d,t=trajectory(500)
        d[175]["agent_input"]["ego_motion"]["lane_heading_error_rad"]=.5
        kept=[s for s in quality.annotate(d,t,"TIMEOUT",RULES)["intervals"] if s["decision"]=="KEEP"]
        self.assertEqual(kept[1]["context_before_start"],241)

    def test_mismatched_trajectory_fails(self):
        d,t=trajectory()
        with self.assertRaises(EvidenceError): quality.annotate(d,t[:-1],"SUCCESS",RULES)


class TestQualityEvidence(unittest.TestCase):
    def test_tampered_frozen_rules_rejected_before_source_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            frozen=Path(tmp)/"frozen"
            frozen.mkdir()
            (frozen/"rules.json").write_bytes(b"changed")
            (frozen/"freeze.json").write_text(json.dumps({"files":{"rules.json":"0"*64}}))
            with patch.object(quality,"partition_table",return_value=({},"partition")),patch.object(quality,"verify_seal") as verify:
                with self.assertRaises(EvidenceError):
                    quality.review(quality.ROOT,["never-open"],quality.ROOT/"experiments/stage4/g1_quality_rules_v1.json",Path(tmp)/"attempt",frozen=frozen)
                verify.assert_not_called()

    def test_freeze_rejects_dev_in_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile=Path(tmp)/"profile"
            profile.mkdir()
            (profile/"review.json").write_text(json.dumps({"scope":"FIT_CALIBRATION_ONLY","episodes":[{"partition":"FIT","outcome":"SUCCESS"},{"partition":"DEV","outcome":"TIMEOUT"}]}))
            with patch.object(quality,"verify_seal"):
                with self.assertRaises(EvidenceError): quality.freeze(profile,Path(tmp)/"frozen")
            self.assertFalse((Path(tmp)/"frozen").exists())

    def test_fit_profile_skips_dev_before_opening_its_trace(self):
        d,t=trajectory(50)
        verified={"steps":50,"trace_sha256":"hash","transitions_sha256":"hash","map_sha256":"hash","terminal_reason":"SUCCESS"}
        plans=[{"resolved_cases":[{"split":"TRAIN","geometry_sha256":side,"case_id":side}]} for side in ("FIT","DEV")]
        fake={"pilot_summary.json":{"failed_episodes":0,"completed_episodes":2,"expected_episodes":2,"episodes":[verified,verified]},
              "request.json":{},"resolved_plans.json":plans,"agent_provenance.json":{}}
        real_load=quality.load_json
        def load(path):
            return fake[path.name] if path.name in fake else real_load(path)
        sides={side:{"partition":side,"geometry_id":side} for side in ("FIT","DEV")}
        with tempfile.TemporaryDirectory() as tmp, patch.object(quality,"load_json",side_effect=load), \
             patch.object(quality,"partition_table",return_value=(sides,"hash")), \
             patch.object(quality,"source_hashes",return_value={}),patch.object(quality,"sha256",return_value="hash"), \
             patch.object(quality,"verify_seal",return_value={"files":{}}), \
             patch.object(quality,"verify_episode",return_value=verified) as check, \
             patch.object(quality,"read_jsonl",side_effect=[d,t]) as reader:
            report=quality.review(quality.ROOT,["synthetic"],quality.ROOT/"experiments/stage4/g1_quality_rules_v1.json",Path(tmp)/"attempt",fit_only=True)
            self.assertEqual([e["partition"] for e in report["episodes"]],["FIT"])
            self.assertEqual(check.call_count,1)
            self.assertEqual(reader.call_count,2)
            self.assertEqual(check.call_args.args[2]["resolved_cases"][0]["case_id"],"FIT")

    def test_dev_application_requires_freeze_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"attempt"
            with patch.object(quality,"partition_table",return_value=({},"partition")):
                with self.assertRaises(EvidenceError):
                    quality.review(quality.ROOT,[],quality.ROOT/"experiments/stage4/g1_quality_rules_v1.json",output)
            self.assertEqual(load_json(output/"run_state.json")["status"],"FAILED")
            self.assertEqual(load_json(output/"run_integrity.json")["status"],"FAILED")

    def test_existing_output_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            (output/"original").write_bytes(b"original")
            with self.assertRaises(FileExistsError):
                quality.review(quality.ROOT,[],quality.ROOT/"experiments/stage4/g1_quality_rules_v1.json",output,fit_only=True)
            self.assertEqual((output/"original").read_bytes(),b"original")

    def test_freeze_requires_fit_timeout_and_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile=Path(tmp)/"profile"
            profile.mkdir()
            (profile/"review.json").write_text(json.dumps({"scope":"FIT_CALIBRATION_ONLY","episodes":[{"partition":"FIT","outcome":"SUCCESS"}]}))
            with patch.object(quality,"verify_seal"):
                with self.assertRaises(EvidenceError): quality.freeze(profile,Path(tmp)/"frozen")
            self.assertFalse((Path(tmp)/"frozen").exists())

    def test_invalid_numeric_rules_fail_before_reading_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"rules.json"
            rules=copy.deepcopy(RULES)
            rules["window_steps"]=0
            p.write_text(json.dumps(rules))
            with self.assertRaises(EvidenceError): quality.review(quality.ROOT,[],p,Path(tmp)/"attempt",fit_only=True)


if __name__=="__main__":
    unittest.main()
