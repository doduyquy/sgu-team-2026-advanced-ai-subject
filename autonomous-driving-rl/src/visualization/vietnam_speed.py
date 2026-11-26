"""High-speed race-only controls; planners still choose their own local paths."""
from functools import lru_cache
import math
import numpy as np

from src.visualization.vietnam_track import centerline


@lru_cache(maxsize=1)
def _speed_envelope():
    points=centerline()
    distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    tangent=np.gradient(points,axis=0)
    heading=np.unwrap(np.arctan2(tangent[:,1],tangent[:,0]))
    curvature=np.abs(np.gradient(heading,distance))
    speeds=np.minimum(80./3.6,np.sqrt(1.5/np.maximum(curvature,1e-5)))
    return points,distance,np.maximum(8./3.6,speeds)


def curve_speed_limit(vehicle):
    """Shared public road-geometry limit with 150 m braking anticipation."""
    points,distance,speeds=_speed_envelope()
    index=int(np.argmin(np.sum((points-np.asarray(vehicle.position))**2,axis=1)))
    end=min(len(points),int(np.searchsorted(distance,distance[index]+150))+1)
    ahead=distance[index:end]-distance[index]
    allowable=np.sqrt(speeds[index:end]**2+2*2.8*ahead)
    return min(80.,float(np.min(allowable))*3.6)


def decision_action(decision, requested, vehicle, enabled=False):
    """Continuous pursuit of the selected search path, not a replacement planner.

    The Stage-2 discrete steering bins are too coarse at road speeds. Preserve
    the planner's chosen target, throttle, failures and obstacle decisions;
    only convert that public target to proportional steering in the VN demo.
    MPPI/player actions have no target_x/target_y and pass through unchanged.
    """
    if not enabled:
        return requested
    diagnostics=decision.diagnostics
    if 'target_x' not in diagnostics or 'target_y' not in diagnostics:
        return requested
    from src.platform import CanonicalActionV1
    x,y=float(diagnostics['target_x']),float(diagnostics['target_y'])
    angle=math.atan2(2.46894*2*y,max(x*x+y*y,1.))
    steer=max(-.6,min(.6,angle/math.radians(40)))
    return CanonicalActionV1(steer,requested.throttle_brake)


def vehicle_speed_limit(env, vehicle):
    if vars(env).get('race_speed_limit_kmh',20.)>20.:
        return curve_speed_limit(vehicle)
    return 20.
