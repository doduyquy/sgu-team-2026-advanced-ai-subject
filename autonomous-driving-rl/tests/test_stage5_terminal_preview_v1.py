import numpy as np
import pytest

from src.agents.stage5_terminal_preview_v1 import Stage5TerminalPreviewAdapterV1


def observation(valid_count=17, route_end=True):
    obs = np.zeros(409, dtype=np.float32)
    points = obs[325:385].reshape(20, 3)
    for i in range(valid_count):
        points[i] = (2.5 * (i + 1), 0.0, 0.0)
        obs[385 + i] = 1.0
    obs[405:408] = (0.7, 1.0, 0.0)
    obs[408] = float(route_end)
    return obs


def test_preview_extends_only_masked_suffix_and_preserves_public_features():
    source = observation(17, True)
    result = Stage5TerminalPreviewAdapterV1().apply(source)
    assert result.activated
    assert result.real_waypoint_count == 17
    assert result.synthetic_waypoint_count == 3
    np.testing.assert_array_equal(result.observation[:325], source[:325])
    np.testing.assert_array_equal(result.observation[405:], source[405:])
    np.testing.assert_array_equal(result.observation[385:402], source[385:402])
    np.testing.assert_allclose(result.observation[325:385].reshape(20, 3)[17:, 0], [45.0, 47.5, 50.0])
    np.testing.assert_array_equal(result.observation[402:405], np.ones(3, dtype=np.float32))
    np.testing.assert_array_equal(source[402:405], np.zeros(3, dtype=np.float32))


def test_adapter_is_identity_without_public_route_end_flag():
    source = observation(17, False)
    result = Stage5TerminalPreviewAdapterV1().apply(source)
    assert not result.activated
    assert result.synthetic_waypoint_count == 0
    np.testing.assert_array_equal(result.observation, source)


def test_adapter_is_identity_when_no_missing_tail():
    source = observation(20, True)
    result = Stage5TerminalPreviewAdapterV1().apply(source)
    assert not result.activated
    np.testing.assert_array_equal(result.observation, source)


def test_noncontiguous_mask_fails_closed():
    source = observation(17, True)
    source[385 + 10] = 0.0
    with pytest.raises(ValueError, match="contiguous valid prefix"):
        Stage5TerminalPreviewAdapterV1().apply(source)


def test_no_valid_waypoints_does_not_invent_preview():
    source = observation(0, True)
    result = Stage5TerminalPreviewAdapterV1().apply(source)
    assert not result.activated
    assert result.real_waypoint_count == 0
    assert result.synthetic_waypoint_count == 0
    np.testing.assert_array_equal(result.observation, source)


@pytest.mark.parametrize("bad", [np.zeros((1, 409), dtype=np.float32), np.full(409, np.nan, dtype=np.float32)])
def test_invalid_input_fails_closed(bad):
    with pytest.raises(ValueError):
        Stage5TerminalPreviewAdapterV1().apply(bad)
