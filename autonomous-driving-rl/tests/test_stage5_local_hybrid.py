from collections import Counter

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from stable_baselines3.common.type_aliases import ReplayBufferSamples

from cloud.kaggle.local_hybrid import ThreeWayReplay


class ReplayStub:
    def __init__(self, value, rows=512):
        self.value = value
        self.rows = rows

    def size(self):
        return self.rows

    def sample(self, n, env):
        return ReplayBufferSamples(*(torch.full((n, 1), self.value) for _ in range(5)))


@pytest.mark.parametrize("online_rows,base_count,online_count", [(255, 192, 0), (256, 128, 64)])
def test_hybrid_batch_reserves_exact_online_and_donor_tier_quotas(online_rows, base_count, online_count):
    replay = ThreeWayReplay.__new__(ThreeWayReplay)
    replay.base = ReplayStub(0)
    replay.external = {"Easy": ReplayStub(1), "Medium": ReplayStub(2)}
    replay.online = ReplayStub(3, online_rows)
    replay.draws = Counter()
    sample = replay.sample(256)

    assert Counter(sample.observations.flatten().tolist()) == {key: count for key, count in enumerate(
        (base_count, 22, 42, online_count)) if count}
    assert replay.draws == {"base": base_count, "external_easy": 22,
                            "external_medium": 42, **({"online": online_count} if online_count else {})}
