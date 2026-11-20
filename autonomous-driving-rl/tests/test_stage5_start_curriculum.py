from pathlib import Path

import numpy as np
import pytest

from src.platform.task_context_producer import REFERENCE_TASK_PRODUCER
from src.training.stage5.env_adapter import Stage5TrainingEnv
from src.training.stage5.start_curriculum import (
    CONTROL_FINGERPRINT,
    CURRICULUM_FINGERPRINT,
    CURRICULUM_TASK_ID,
    curriculum_stage,
)
from src.training.stage5.train_sampler import SamplerStrategy, TrainCaseSampler


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("episode", "stage", "distance"),
    [
        (0, "near_60m", 60.0),
        (2, "near_60m", 60.0),
        (3, "mid_120m", 120.0),
        (5, "mid_120m", 120.0),
        (6, "full_route", None),
        (100, "full_route", None),
    ],
)
def test_p2f_curriculum_schedule_is_fixed(episode, stage, distance):
    selected = curriculum_stage(episode)
    assert selected.stage_id == stage
    assert selected.requested_remaining_route_distance_m == distance


def test_p2f_task_fingerprints_are_distinct_and_stable():
    assert CONTROL_FINGERPRINT != CURRICULUM_FINGERPRINT
    assert len(CONTROL_FINGERPRINT) == 64
    assert len(CURRICULUM_FINGERPRINT) == 64


def test_reset_current_route_baseline_prevents_initial_progress_reward():
    sampler = TrainCaseSampler(
        training_root_seed=101,
        strategy=SamplerStrategy.FIXED,
        tier_filter="Easy",
        sequence_filter="SCS",
        fixed_geometry_seed=0,
        allowed_geometry_ids=["geom_easy_SCS_seed0"],
        project_root=ROOT,
    )
    case = sampler.get_case(0, tier="Easy", sequence="SCS", environment_seed=0)
    env = Stage5TrainingEnv(
        sampler=sampler,
        project_root=ROOT,
        task_context_producer_id=REFERENCE_TASK_PRODUCER,
        training_start_curriculum=True,
    )
    try:
        observation, reset_info = env.reset(options={"case": case})
        assert observation.shape == (409,)
        assert reset_info["training_task_id"] == CURRICULUM_TASK_ID
        assert reset_info["curriculum_stage_id"] == "near_60m"
        assert reset_info["effective_remaining_route_distance_m"] == pytest.approx(60.0, abs=0.10)
        assert reset_info["valid_task_waypoints"] >= 19
        assert env._last_route_completion == reset_info["initial_route_completion"]

        _, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        assert not terminated
        assert not truncated
        assert info["delta_route"] == pytest.approx(
            info["route_completion"] - reset_info["initial_route_completion"], abs=1e-12
        )
        assert info["reward_breakdown"]["progress_reward"] == pytest.approx(info["delta_route"], abs=1e-12)
        assert reward == pytest.approx(info["reward_breakdown"]["total_reward"], abs=1e-12)
    finally:
        env.close()


def test_curriculum_eval_override_is_restricted_to_registered_short_stages():
    sampler = TrainCaseSampler(
        training_root_seed=101, strategy=SamplerStrategy.FIXED, tier_filter="Easy",
        sequence_filter="SCS", fixed_geometry_seed=0,
        allowed_geometry_ids=["geom_easy_SCS_seed0"], project_root=ROOT,
    )
    env = Stage5TrainingEnv(
        sampler=sampler, project_root=ROOT,
        task_context_producer_id=REFERENCE_TASK_PRODUCER,
        training_start_curriculum=True,
    )
    try:
        case = sampler.get_case(0, tier="Easy", sequence="SCS", environment_seed=0)
        _, info = env.reset(options={"case": case, "curriculum_episode_index": 3})
        assert info["curriculum_stage_id"] == "mid_120m"
        assert info["effective_remaining_route_distance_m"] == pytest.approx(120.0, abs=0.10)
        with pytest.raises(ValueError, match="preregistered 60m or 120m"):
            env.reset(options={"case": case, "curriculum_episode_index": 6})
    finally:
        env.close()
