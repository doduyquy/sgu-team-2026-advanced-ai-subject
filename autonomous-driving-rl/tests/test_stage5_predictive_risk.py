"""Synthetic cases for the public-observation collision and release checks."""
import numpy as np
import pytest

from cloud.kaggle.predictive_risk import PredictiveController, Scene


def scene(*, speed=7., actor=None, edges=(5.25, 5.25), curve=0., lidar=None, overflow=False):
    obs = np.zeros(409, dtype=np.float32)
    obs[:2] = np.asarray(edges) / 18
    obs[3] = (speed * 3.6 + 1) / 81
    obs[19:259] = 1
    if lidar is not None:
        ray, distance = lidar
        obs[19 + ray] = distance / 50.
    if actor is not None:
        obs[259:266] = np.asarray(actor) / (50., 50., 30., 30., np.pi, 5., 5.)
        obs[315] = 1
    s = np.arange(1, 21) * 2.5
    points = obs[325:385].reshape(20, 3)
    if curve:
        points[:, 0] = np.sin(s * curve) / curve / 50.
        points[:, 1] = (1 - np.cos(s * curve)) / curve / 50.
    else:
        points[:, 0] = s / 50.
    obs[385:405] = 1
    obs[405] = .7
    obs[324] = float(overflow)
    return obs


@pytest.mark.parametrize("label, actor, expected", [
    ("distant separating", (37, 0, 4, 0, 0, 4.5, 1.9), False),
    ("same speed", (16, 0, 0, 0, 0, 4.5, 1.9), False),
    ("slowing lead", (12, 0, -5, 0, 0, 4.5, 1.9), True),
    ("stopped in path", (14, 0, -7, 0, 0, 4.5, 1.9), True),
    ("adjacent", (7, 3.5, -6, 0, 0, 4.5, 1.9), False),
    ("passing from behind in adjacent lane", (-8, -3.5, 5, 0, 0, 4.5, 1.9), False),
    ("crossing in time", (11, -8, -7, 5, np.pi/2, 4.5, 1.9), True),
])
def test_collision_geometry(label, actor, expected):
    prediction = Scene(scene(actor=actor)).predict((0., 0.))
    assert (prediction.status == "conflict") is expected, label
    if expected:
        assert prediction.earliest_s and prediction.min_clearance_m < 0


def test_curved_reference_does_not_report_false_vehicle_conflict():
    value = Scene(scene(speed=2., curve=.026, actor=(15., -4., 0., 0., 0., 4.5, 1.9))).predict((-.12, .15))
    assert value.status != "conflict"


def test_near_lateral_marking_vs_footprint_exit():
    # A lateral return and a nearby left edge do not hit a straight footprint.
    obs = scene(speed=.4, edges=(1.1, 5.9), lidar=(60, 1.1))
    assert Scene(obs).predict((0., .1)).status == "clear"
    edge = Scene(scene(speed=2., edges=(1.2, 5.8))).predict((-.65, .35))
    assert edge.status == "local_road_exit"


def test_untracked_front_lidar_return_and_overflow():
    assert Scene(scene(lidar=(0, 8))).predict((0., 0.)).status == "conflict"
    assert Scene(scene(speed=.2, overflow=True)).predict((0., .3)).status == "uncertain"
    obs = scene(speed=.2)
    obs[385:405] = 0
    assert Scene(obs).predict((0., .3)).status == "uncertain"


def test_restart_requires_stable_evidence_and_abstains_on_overflow():
    raw = np.array([0., -.25], dtype=np.float32)
    policy = PredictiveController()
    obs = scene(speed=.2)
    for _ in range(4):
        action, info = policy.act(obs, raw)
        assert info["reason"] == "actor"
        np.testing.assert_array_equal(action, raw)
    action, info = policy.act(obs, raw)
    assert action[1] > 0 and info["reason"] == "predictive_release"
    action, info = policy.act(scene(speed=.2, overflow=True), raw)
    np.testing.assert_array_equal(action, raw)
    assert info["status"] == "uncertain"


def test_lead_ttc_requires_closing_and_is_distinct_from_footprint_overlap():
    closing = Scene(scene(actor=(12, 0, -5, 0, 0, 4.5, 1.9))).predict((0., 0.))
    assert closing.lead_gap_m == pytest.approx(7.5)
    assert closing.lead_ttc_s == pytest.approx(1.5)
    assert closing.earliest_s is not None
    separating = Scene(scene(actor=(37, 0, 4, 0, 0, 4.5, 1.9))).predict((0., 0.))
    assert separating.lead_gap_m is not None
    assert separating.lead_ttc_s is None


def test_guarded_restart_waits_for_genuine_stop_and_preserves_steering():
    raw = np.array([.08, -.25], dtype=np.float32)
    policy = PredictiveController(guarded=True)
    for _ in range(50):
        np.testing.assert_array_equal(policy.act(scene(speed=.7), raw)[0], raw)
    stopped = scene(speed=.1)
    for _ in range(43):
        np.testing.assert_array_equal(policy.act(stopped, raw)[0], raw)
    action, info = policy.act(stopped, raw)
    assert info["reason"] == "predictive_release"
    assert action[0] == raw[0] and action[1] > 0
