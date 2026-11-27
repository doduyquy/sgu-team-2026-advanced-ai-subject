import numpy as np
import pytest

from cloud.kaggle.observation_sensitivity import DIVISORS, action_row, boundary_pair, mask_rays, pack_traffic


def traffic(*distances):
    return [[x, 0, -2, 0, 0, 4.5, 1.8] for x in distances]


def test_repack_promotes_overflow_and_encodes_empty_slots():
    original = np.zeros(409, dtype=np.float32)
    original[19:259] = 1
    rows = traffic(19, 8, 12, 2, 3, 4, 5, 6, 50, 51)
    packed = pack_traffic(original, rows)
    np.testing.assert_allclose(packed[259:315].reshape(8, 7)[:, 0]*DIVISORS[0], [2, 3, 4, 5, 6, 8, 12, 19])
    assert packed[323] == 1 and packed[324] == pytest.approx(.1)
    fewer = pack_traffic(packed, rows[1:8])
    assert fewer[322] == 0 and fewer[323] == .875 and fewer[324] == 0
    np.testing.assert_array_equal(fewer[308:315], 0)
    np.testing.assert_array_equal(packed[19:259], original[19:259])
    assert original[315:325].sum() == 0


def test_ray_filter_preserves_other_objects_and_input_and_reveals_background():
    original = np.ones(409, dtype=np.float32)
    original[19:23] = [.22, .3, .4, .5]
    hits = {"0": [["target", .22, "car"], ["pending", .48, "car"], ["wall", .61, "wall"]],
            "1": [["target", .3, "car"]], "2": [["other", .4, "car"], ["target", .45, "car"]]}
    changed, count = mask_rays(original, hits, {"target", "pending"})
    assert count == 2
    np.testing.assert_allclose(changed[19:23], [.61, 1., .4, .5])
    np.testing.assert_allclose(original[19:23], [.22, .3, .4, .5])


def test_border_probe_conserves_width_and_sign_flips_have_deadband():
    original = np.zeros(409, dtype=np.float32)
    original[:2] = [3/18, 7/18]
    changed = boundary_pair(original, "left", 1.5, 18)
    assert changed[0] * 18 == pytest.approx(1.5)
    assert changed[:2].sum() == pytest.approx(original[:2].sum())
    np.testing.assert_array_equal(changed[2:], original[2:])
    assert action_row([-.01, -.5], [.5, .5])["steering_reversal"] == 0
    assert action_row([-.01, -.5], [.5, .5])["brake_to_gas"] == 1
