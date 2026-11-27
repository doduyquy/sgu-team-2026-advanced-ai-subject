import numpy as np
import pytest

from cloud.kaggle.observation_sensitivity import pack_traffic
from cloud.kaggle.smooth_traffic import blend, variants


def observation():
    values = np.zeros(409, dtype=np.float32)
    values[3] = 1 / 81
    values[19:259] = 1
    values[325:385:3] = np.arange(1, 21) / 20
    values[385:405] = 1
    return values


class CountSensitiveActor:
    def predict(self, observations, deterministic):
        assert deterministic
        assert np.allclose(observations[:, 315:323].sum(axis=1) / 8, observations[:, 323])
        return np.stack((np.full(len(observations), .8), 1 - 8 * observations[:, 323]), axis=1), None


def test_range_edge_is_continuous_and_steering_is_unchanged():
    actor = CountSensitiveActor()
    raw = np.array([.8, 0.], dtype=np.float32)
    edge = pack_traffic(observation(), [[50, 0, 0, 0, 0, 4.5, 1.8]])
    action = blend(actor, edge, raw)
    assert action[0] == raw[0]
    assert action[1] == pytest.approx(1.)
    assert edge[315] == 1


@pytest.mark.parametrize("actor", [[9, 0, 0, 0, 0, 4, 2], [32, 0, -12, 0, 0, 4, 2], [29, -8, 0, 8, 1.57, 4, 2]])
def test_uncertain_or_conflicting_actor_keeps_original_observation(actor):
    packed = pack_traffic(observation(), [actor])
    (weight, vector), = variants(packed)
    assert weight == 1 and vector is packed


def test_overflow_preserves_original_observation():
    obs = pack_traffic(observation(), [[30 + i, 0, 0, 0, 0, 4, 2] for i in range(9)])
    (weight, vector), = variants(obs)
    assert weight == 1 and vector is obs
