from types import SimpleNamespace

import numpy as np
import pytest

from cloud.kaggle.traffic_context_experiment import assemble, core_radius, dynamic_mask


def test_dynamic_admits_closing_crossing_and_rear_threats_but_not_distant_static_traffic():
    positions = np.array([[5., 0.], [31.4, 0.], [32., 0.], [14., -13.], [-28., 0.], [52., 0.]])
    velocities = np.array([[0., 0.], [0., 0.], [-9., 0.], [-3., 4.], [8., 0.], [-20., 0.]])
    radius, mask = dynamic_mask(positions, velocities, 0.)
    assert radius == 6.
    np.testing.assert_array_equal(mask, [True, False, True, True, True, False])
    unavailable, fallback = dynamic_mask(positions, velocities, 0., reliable=False)
    assert unavailable == 50.
    np.testing.assert_array_equal(fallback, [True, True, True, True, True, False])
    assert 6 < core_radius(8.) < core_radius(14.) <= 50


@pytest.mark.parametrize("profile,expected", [("A", 50.), ("B", 20.), ("C", 15.), ("D", 10.), ("E", 5.)])
def test_fixed_profile_calls_same_producer_and_preserves_raw_observation(monkeypatch, profile, expected):
    from src.platform import agent_input

    ego = SimpleNamespace(navigation=object())
    road = object()
    others = [object(), object()]
    sim = SimpleNamespace(agent=ego, engine=SimpleNamespace(traffic_manager=SimpleNamespace(traffic_vehicles=others)),
                          current_map=SimpleNamespace(road_network=road))
    base = SimpleNamespace(_env=sim, task_context_producer_id="test")
    raw = np.arange(259)
    monkeypatch.setattr(agent_input, "build_agent_input", lambda **kwargs: kwargs)
    produced = assemble(profile, base, raw, step_idx=0)
    assert produced["raw_obs"] is raw
    assert produced["road_network"] is road
    assert produced["traffic_vehicles"] == others
    assert produced["traffic_radius_m"] == expected
    assert base._experimental_radius == expected
