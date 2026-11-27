from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from cloud.kaggle.risk_guidance import risk_action


def observation(speed_mps=10.0, actor=None, forward_lidar=50.0):
    result = np.zeros(409, dtype=np.float32)
    result[3] = (speed_mps * 3.6 + 1) / 81
    result[19:259] = 1.0
    result[19] = forward_lidar / 50
    route = result[325:385].reshape(20, 3)
    route[:, 0] = np.arange(1, 21, dtype=np.float32) * 2.5 / 50
    result[385:405] = 1
    if actor:
        result[259:266] = np.asarray(actor) / [50, 50, 30, 30, np.pi, 5, 5]
        result[315] = 1
    return result


def test_adjacent_actor_does_not_cap_speed_or_change_steering():
    action = np.array([.15, .7], dtype=np.float32)
    obs = observation(actor=(14, 3.5, -6, 0, 0, 4.5, 1.9))
    guided, evidence = risk_action(obs, action)
    np.testing.assert_array_equal(guided, action)
    assert evidence["reason"] == "none"


def test_short_gap_and_closing_rate_trigger_brake():
    action = np.array([.15, .7], dtype=np.float32)
    obs = observation(actor=(12, .4, -8, 0, 0, 4.5, 1.9))
    guided, evidence = risk_action(obs, action)
    assert guided[0] == action[0]
    assert guided[1] < 0
    assert evidence["reason"] == "closing_risk"
    np.testing.assert_allclose([evidence["gap_m"], evidence["closing_mps"], evidence["ttc_s"]],
                               [7.5, 8., .9375], atol=1e-5)


def test_crossing_prediction_requires_conflict_at_the_same_time():
    action = np.array([0., .55], dtype=np.float32)
    crossing = observation(actor=(12., -7.5, -10., 6., np.pi / 2, 4.5, 1.9))
    guided, evidence = risk_action(crossing, action)
    assert guided[1] < 0 and evidence["reason"] == "crossing_risk"
    distant_crossing = observation(actor=(30., -7.5, -10., 6., np.pi / 2, 4.5, 1.9))
    guided, evidence = risk_action(distant_crossing, action)
    np.testing.assert_array_equal(guided, action)
    assert evidence["reason"] == "none"


def test_crawl_requires_valid_route_and_clear_forward_lidar():
    action = np.array([.15, -.2], dtype=np.float32)
    obs = observation(speed_mps=.1)
    guided, evidence = risk_action(obs, action)
    assert guided[1] > 0 and evidence["reason"] == "clear_crawl"
    blocked, evidence = risk_action(observation(speed_mps=.1, forward_lidar=5), action)
    np.testing.assert_array_equal(blocked, action)
    assert evidence["reason"] == "none"
    obs[385:405] = 0
    unresolved, _ = risk_action(obs, action)
    np.testing.assert_array_equal(unresolved, action)


@pytest.mark.parametrize("retention", (False, True))
def test_replay_keeps_all_three_frozen_v2_pools_and_both_new_pools(retention):
    torch = pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.type_aliases import ReplayBufferSamples

    from cloud.kaggle.local_risk import MIX, RETENTION_MIX, RiskReplay

    class Buffer:
        def __init__(self, identity):
            self.identity = identity

        def size(self):
            return 256

        def sample(self, n, env):
            return ReplayBufferSamples(*(torch.full((n, 1), self.identity) for _ in range(5)))

    replay = RiskReplay.__new__(RiskReplay)
    expected = RETENTION_MIX if retention else MIX
    replay.mixture = dict(sorted(expected.items()))
    replay.base = Buffer(0)
    replay.frozen = SimpleNamespace(external={"Easy": Buffer(1), "Medium": Buffer(2)}, online=Buffer(3))
    replay.rings = {"risk": Buffer(4), "online": Buffer(5)}
    replay.draws = Counter()
    sample = replay.sample(256)
    assert len(sample.observations) == 256
    assert sample.observations.flatten().tolist() == [
        i for i, name in enumerate(MIX) for _ in range(expected[name])]
    assert replay.draws == replay.mixture
