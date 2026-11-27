from collections import Counter

from cloud.kaggle.final_sac_evaluation import CONTRACT, SPLITS, density, jobs_for
from cloud.kaggle.final_sac_report import aggregate, pairs
from src.platform.episode import classify_episode_outcome


def test_locked_case_counts_and_traffic_pairing():
    expected_density = {"Easy": .08, "Medium": .08, "Hard": .15, "Extreme": .25}
    for split, (_, total, seeds) in SPLITS.items():
        jobs = jobs_for(split)
        assert len(jobs) == total * 2
        assert {case.environment_seed for case, _ in jobs} == seeds
        assert Counter((case.tier, condition) for case, condition in jobs) == {
            (tier, condition): total // 4 for tier in expected_density for condition in ("ON", "OFF")
        }
        for (on, on_condition), (off, off_condition) in zip(jobs[::2], jobs[1::2], strict=True):
            assert on == off
            assert on_condition == "ON" and density(on, on_condition) == expected_density[on.tier]
            assert off_condition == "OFF" and density(off, off_condition) == 0.


def test_pair_counts_and_switch_direction():
    rows = [{"case_id": str(index), "traffic": condition, "clean_success": clean}
            for index, result in enumerate([(True, True), (False, True), (False, True), (True, False), (False, False)])
            for condition, clean in zip(("ON", "OFF"), result, strict=True)]
    summary = pairs(rows, 5)
    assert summary == {"expected_pairs": 5, "matched_pairs": 5, "both_clean": 1, "off_only_clean": 2,
                       "on_only_clean": 1, "neither_clean": 1, "matched_off_minus_on_pp": 20., "off_minus_on_pp": 20.}
    assert pairs(rows[:-1], 5)["off_minus_on_pp"] is None


def test_incomplete_cohort_has_no_final_rates_and_safety_has_precedence():
    assert aggregate([], 60)["clean_success_rate"] is None
    assert aggregate([], 60)["mean_time_to_clean_success_s"] is None
    outcome = classify_episode_outcome({"arrive_dest": True, "crash_vehicle": True}, True, True)
    assert not outcome.clean_success and outcome.primary_reason.value == "CRASH_VEHICLE"
    assert not CONTRACT["canonical_run"] and CONTRACT["training_replicates"] == 1
