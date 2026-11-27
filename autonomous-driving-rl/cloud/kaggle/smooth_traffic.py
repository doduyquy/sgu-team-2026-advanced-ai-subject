"""Opt-in diagnostic blend of schema-valid TrafficContext counterfactuals."""
from __future__ import annotations

import math

import numpy as np

from cloud.kaggle.observation_sensitivity import DIVISORS, geometry_features, pack_traffic
from cloud.kaggle.risk_guidance import _project


def smoothstep(value):
    x = np.clip(value, 0., 1.)
    return x * x * (3. - 2. * x)


def ablation_weights(obs, actors):
    if obs[324] > 0 or obs[408] > .5 or not np.all(np.isfinite(obs)):
        return np.zeros(len(actors))
    _, path, _ = geometry_features(obs)
    if path is None:
        return np.zeros(len(actors))
    xy, velocity = actors[:, :2], actors[:, 2:4]
    distance = np.linalg.norm(xy, axis=1)
    if not len(distance):
        return distance
    ego_speed = max(0., (float(obs[3]) * 81 - 1) / 3.6)
    station, lateral, _ = _project(xy, path)
    clearance = distance - 4 * np.linalg.norm(velocity, axis=1) - 7.
    # A large relative-motion envelope reduces suppression even when observed
    # displacement points away. Route relevance restores weight for potential
    # conflicts along the near-term swept path.
    envelope = smoothstep((clearance - 18.) / 16.)
    along_reachable = smoothstep((station - (ego_speed * 4 + 8.)) / 10.)
    outside_corridor = smoothstep((lateral - 2.5) / 4)
    route_distant = np.maximum(along_reachable, outside_corridor)
    # Vehicles behind the ego still have to clear the motion envelope.
    route_distant = np.where(station < -12., 1., route_distant)
    return smoothstep((distance - 27.) / 23.) * envelope * route_distant


def variants(observation):
    """Integrate nested valid subsets using a shared uniform relevance threshold."""
    actors = observation[259:315].reshape(8, 7)[observation[315:323] > .5] * DIVISORS
    if not len(actors):
        return [(1., observation)]
    weights = ablation_weights(observation, actors)
    levels = sorted({0., 1., *map(float, weights)})
    output = []
    for lower, upper in zip(levels, levels[1:]):
        keep = weights <= lower
        vector = observation if keep.all() else pack_traffic(observation, actors[keep])
        output.append((upper - lower, vector))
    return output


def blend(actor, observation, raw):
    choices = variants(observation)
    if len(choices) == 1 and choices[0][1] is observation:
        return raw
    weights, vectors = zip(*choices)
    predicted, _ = actor.predict(np.stack(vectors), deterministic=True)
    result = raw.copy()
    result[1] = math.fsum(weight * action[1] for weight, action in zip(weights, predicted))
    return result
