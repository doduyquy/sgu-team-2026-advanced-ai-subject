from types import SimpleNamespace

import pytest

from src.launcher.cli import main
from src.launcher.steering import SteeringFilter, steering_sample
from src.platform.action_adapter import CanonicalActionV1


def task(headings, mask=None):
    return SimpleNamespace(waypoints=[SimpleNamespace(relative_heading=h) for h in headings],
                           validity_mask=[True for _ in headings] if mask is None else mask)


def test_steering_ema_preserves_throttle_and_resets_between_episodes():
    initial = CanonicalActionV1(-1.0, 0.2)
    next_action = CanonicalActionV1(1.0, -0.3)
    off, fixed, adaptive = (SteeringFilter(p) for p in ("off", "fixed", "adaptive"))
    road = task([0.0] * 4)
    for profile in (off, fixed, adaptive):
        assert profile.apply(initial, 40, road)[0] == initial
    assert off.apply(next_action, 40, road)[0] == next_action
    output, _, _ = fixed.apply(next_action, 40, road)
    assert output.steering == pytest.approx(0.10)
    assert output.throttle_brake == next_action.throttle_brake
    output, _, _ = adaptive.apply(next_action, 40, road)
    assert output.steering == pytest.approx(-0.30)
    assert SteeringFilter("adaptive").apply(next_action, 40, road)[0] == next_action


def test_adaptive_profile_follows_curves_and_excludes_invalid_preview():
    action = CanonicalActionV1(1.0, 0.0)
    for road in (task([0.0, 0.2, 0.4, 0.6]), task([0.0, 0.0, 0.0, 0.0], [True, False, False, False])):
        output, alpha, _ = SteeringFilter("adaptive", previous=0.0).apply(action, 40, road)
        assert alpha == pytest.approx(0.85)
        assert output.steering == pytest.approx(0.85)


def test_steering_telemetry_measures_world_velocity_change_and_skips_first_jerk():
    vehicle = SimpleNamespace(steering=0.5, speed_km_h=36.0, lane=None)
    action = CanonicalActionV1(0.5, 0.0)
    series = steering_sample(vehicle, (0.0, (10, 0)), (0.0, (10, 0.2)), 0.1, action, action, 1.0, 0, None)
    assert series["yaw_rate_rad_s"] == 0.0
    assert series["lateral_accel_m_s2"] == pytest.approx(2.0)
    assert series["lateral_jerk_m_s3"] is None


def test_cli_does_not_allow_steering_profile_for_official_benchmark():
    assert main(["run", "--mode", "TEST", "--agent", "stage5_sac_terminal_preview",
                 "--steering-profile", "adaptive"]) == 1
