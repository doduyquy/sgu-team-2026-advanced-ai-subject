"""Longitudinal behavior guidance from the existing public traffic observation."""
from __future__ import annotations

import numpy as np

PROFILE = "traffic_follow_v1"


def traffic_follow(observations, actions):
    """Limit traffic-exposed throttle during collection; leave steering unchanged."""
    actors = observations[:, 259:315].reshape(-1, 8, 7)
    valid = observations[:, 315:323] > .5
    x, y = actors[:, :, 0] * 50., actors[:, :, 1] * 50.
    vx = actors[:, :, 2] * 30.
    length, width = actors[:, :, 5] * 5., actors[:, :, 6] * 5.
    speed = np.maximum(0., (observations[:, 3] * 81. - 1.) / 3.6)
    gap = x - (length + 4.5) / 2.
    leads = valid & (x > 0.) & (x < 32.) & (np.abs(y) < (width + 1.9) / 2. + .3)
    target = np.minimum(8., np.min(np.where(leads, np.maximum(0., speed[:, None] + vx + (gap - 8.) / 1.5),
                                           np.inf), axis=1))
    limit = np.clip((target - speed) / 3., -.9, .65)
    guided = actions.copy()
    exposed = valid.any(axis=1)
    guided[exposed, 1] = np.minimum(guided[exposed, 1], limit[exposed])
    return guided
