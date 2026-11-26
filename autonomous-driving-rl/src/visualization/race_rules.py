"""Shared rules for interactive races; canonical benchmark policy stays intact."""

from dataclasses import dataclass
from src.platform.episode import SAFETY_CRITICAL_FLAGS, classify_episode_outcome


RULESET_ID = 'race_strict_v1'
RACE_SPEED_LIMIT_KMH = 20.
RACE_PLANNER_CRUISE_KMH = 15.
SOLID_LINE_FLAGS = ('on_yellow_continuous_line', 'on_white_continuous_line')
RACE_FLAGS = (*SAFETY_CRITICAL_FLAGS, *SOLID_LINE_FLAGS, 'on_broken_line',
              'arrive_dest', 'max_step')


def rules_manifest(practice=False, speed_limit_kmh=RACE_SPEED_LIMIT_KMH, cruise_kmh=RACE_PLANNER_CRUISE_KMH):
    return {'id': RULESET_ID, 'practice': practice, 'same_rules_for_all_controllers': True,
            'speed_limit_kmh': speed_limit_kmh,
            'planner_cruise_kmh': cruise_kmh,
            'solid_line': 'eliminated', 'broken_line': 'allowed',
            'collision_or_off_road': 'eliminated', 'recovery_allowed': practice,
            'ranked': not practice, 'winner': 'clean_finish_then_simulation_time',
            'control_limits': None if practice else {'steering':[-.6,.6],'throttle_brake':[-.8,.6]},
            'contact_sensor': 'MetaDrive vehicle footprint, not decorative mesh'}


@dataclass(frozen=True)
class RaceOutcome:
    reason: str
    is_done: bool
    clean_success: bool


def race_planner_options(vehicle_max_speed_kmh, cruise_kmh=RACE_PLANNER_CRUISE_KMH):
    # Pinned StateObservation uses (km/h + 1)/(vehicle max km/h + 1).
    return {'target_normalized_speed': (cruise_kmh+1.)/(float(vehicle_max_speed_kmh)+1.)}


def apply_race_planner_options(policy, options):
    """Override only this race instance; leave factory JSON and registry intact."""
    if options:
        if set(options) == {'mppi_target_speed_m_s'}:
            from dataclasses import replace
            policy.config = replace(policy.config,target_speed_m_s=options['mppi_target_speed_m_s'])
            return
        if not set(options).issubset({'target_normalized_speed','control_lookahead_m','vn_dynamic_lookahead'}) or not isinstance(policy.config,dict):
            raise ValueError('Unsupported race planner options')
        policy.config = {**policy.config, **options}


def prepare_race_policy(policy, inputs):
    config=getattr(policy,'config',None)
    if isinstance(config,dict) and config.get('vn_dynamic_lookahead'):
        speed=max(0.,float(inputs.core_observation.ego_state[3])*81.-1.)
        config['control_lookahead_m']=max(3.,min(9.,speed*.13))


def limit_race_action(action, practice=False, speed_kmh=0., speed_limit_kmh=RACE_SPEED_LIMIT_KMH):
    """Use the discrete9 envelope for continuous AI and the player as well."""
    from src.platform import CanonicalActionV1
    steering = action.steering if practice else max(-.6,min(.6,action.steering))
    throttle = action.throttle_brake if practice else max(-.8,min(.6,action.throttle_brake))
    # One km/h approach band avoids pulsing full throttle into the limiter.
    if throttle > 0:
        throttle *= max(0.,min(1.,speed_limit_kmh-float(speed_kmh)))
    if speed_kmh > speed_limit_kmh:
        throttle = min(throttle,-min(.8,(float(speed_kmh)-speed_limit_kmh)*.2))
    if steering == action.steering and throttle == action.throttle_brake:
        return action
    return CanonicalActionV1(steering,throttle)


def cap_race_velocity(vehicle, speed_limit_kmh=RACE_SPEED_LIMIT_KMH):
    """Cap planar speed after a race tick, retaining direction and vertical motion."""
    if vehicle.speed_km_h > speed_limit_kmh:
        vehicle.set_velocity(vehicle.velocity,value=speed_limit_kmh/3.6)


def classify_race_outcome(flags, terminated=False, truncated=False, practice=False):
    """Crash > off-road > solid line > finish > timeout, for human and AI."""
    checked = {key: value for key, value in flags.items()
               if not practice or key not in SAFETY_CRITICAL_FLAGS}
    base = classify_episode_outcome(checked, terminated=terminated, truncated=truncated)
    reason = base.primary_reason.value
    if not practice and any(flags.get(key, False) for key in SOLID_LINE_FLAGS):
        if not any(flags.get(key, False) for key in SAFETY_CRITICAL_FLAGS):
            reason = 'SOLID_LINE'
        return RaceOutcome(reason, True, False)
    return RaceOutcome(reason, base.is_done, base.clean_success and not practice)


def ordered_racers(racers):
    """Clean finishers first; eliminated cars cannot lead by route progress."""
    def key(racer):
        row = racer.summary
        if row and row.get('clean_success'):
            return (0, row['simulation_time_s'], racer.index)
        if row is None:
            return (1, -(racer.route or 0.), racer.index)
        return (2, -(racer.route or 0.), racer.index)
    return sorted(racers, key=key)


def race_positions(racers):
    racers = list(racers)
    finishers = [r for r in racers if r.summary and r.summary.get('clean_success')]
    active = [r for r in racers if r.summary is None]
    result = {}
    for racer in racers:
        if racer in finishers:
            rank = 1 + sum(r.summary['simulation_time_s'] < racer.summary['simulation_time_s'] for r in finishers)
        elif racer.summary is None:
            rank = len(finishers) + 1 + sum((r.route or 0.) > (racer.route or 0.) for r in active)
        else:
            rank = None
        result[racer.vehicle_id] = int(rank) if rank is not None else None
    return result
