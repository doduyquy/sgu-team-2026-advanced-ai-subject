"""Read-only compatibility audit of local models and the main project's public API.

Writes a new evidence directory in this workspace. Optional simulator trials use
TRAIN seed 170 and IDM control to compare extractors; they are NOT model-driving
or benchmark results. The PDD96 launcher smoke uses the existing rule policy
on one TRAIN sandbox case. The speed pilot compares a frozen PDD checkpoint
with Stage 1 in matched TRAIN traffic. Explicit junction-forecast modes train
offline from original PDD only. No main-project writes occur.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import csv
from datetime import datetime, timezone
from functools import partial
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def public_feature_digest(snapshot):
    from agent4_pdd_common_v2 import encode_public

    return hashlib.sha256(struct.pack('<96f', *encode_public(snapshot))).hexdigest()


def lane_frame_decoder_binding(bundle):
    """Accept the original trainer or explicitly pinned unchanged legacy decoder.

    Frame05 retains its original trainer receipt. New training modes elsewhere
    in that file must not invalidate it or silently change its decoding path.
    """
    import inspect
    import train_pdd_baseline as trainer

    current = digest(ROOT / 'train_pdd_baseline.py')
    if current == bundle['trainer_sha256']:
        return {'mode': 'WHOLE_TRAINER_SHA256', 'trainer_sha256': current}
    expected = {
        'plan_reference': '138934c4cf0368f38213ea64cbea71552252b23c630cf89e293150660dc7c43d',
        'transform_plan_lane_frame': '28a07e57e2c62170e61ab55ad9181dc72ff43875e62e7bd5e6d25a8b0b5a5ddf',
        'plan_model_reference': 'c7a56e5fa75070c521c2f2ae1c300d5366827c0a4ea90819c589eb6ce37eab78',
        'predict_c0_plan': '9422c513a97be68929d1f74ec739c5a16486ee5542c3e3e29589a85f45d226ce',
    }
    measured = {name: hashlib.sha256((inspect.getsource(getattr(trainer, name)).replace('\r\n', '\n').strip()
                                      + '\n').encode('utf-8')).hexdigest() for name in expected}
    if (bundle['trainer_sha256'] != 'dc1c9d604f8b9fbcad7f954d561f7af89bde953176164c1fc4f6aae45d8bc97c'
            or bundle['schema'] != 'PDD_C0_PLAN_LANE_FRAME_V1' or measured != expected):
        raise ValueError('Incompatible lane-frame decoder source binding')
    return {'mode': 'PINNED_LEGACY_FRAME05_FUNCTIONS', 'function_sha256': measured,
            'original_trainer_sha256': bundle['trainer_sha256'], 'current_trainer_sha256': current}


def verify_frozen_function_sources(source, bindings):
    """Compare live behavior with a pinned source copy without executing that copy."""
    import ast
    import inspect
    import textwrap

    lines, nodes = source.splitlines(), {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            nodes[node.name] = node
        elif isinstance(node, ast.ClassDef):
            nodes.update({f'{node.name}.{entry.name}': entry for entry in node.body if isinstance(entry, ast.FunctionDef)})
    hashes = {}
    for name, function in bindings.items():
        if name not in nodes:
            raise ValueError(f'Frozen behavior is missing: {name}')
        node = nodes[name]
        frozen = textwrap.dedent('\n'.join(lines[node.lineno-1:node.end_lineno])).strip() + '\n'
        live = textwrap.dedent(inspect.getsource(function)).replace('\r\n', '\n').strip() + '\n'
        if frozen != live:
            raise ValueError(f'Frozen controller behavior changed: {name}')
        hashes[name] = hashlib.sha256(live.encode('utf-8')).hexdigest()
    return hashes


def frozen_pace_lineage():
    """Hold the two-case profile fixed before the six-case TRAIN transfer run."""
    previous = ROOT / 'agent4_integration_audit/run_20261120T154047_708296Z'
    report_path = previous / 'report.json'
    source_path = previous / 'Medium_SCXCS_14_1/launcher/agent4_pdd_curve06_guarded_pace/sources/policy.py'
    expected_report = '9e3a4974abe549effc6f679aae270df319f1ac45d10c21dd65670e3e90fd1644'
    expected_source = '0758b0645602e42847c725bfe3d0236f9740ef9f267073cf8a024aad24b2ff8e'
    if digest(report_path) != expected_report or digest(source_path) != expected_source:
        raise ValueError('Expected the frozen two-case guarded-pace report and source')
    report = json.loads(report_path.read_bytes())
    if (report['controller_support_parameters'] != PACE_SUPPORT_PARAMETERS or report['pdd_clean'] != 2
            or report['complete_pairs'] != 2 or report['checkpoint_sha256'] != PDDCurvePaceDiagnostic.checkpoint_hash
            or report['rear_axle_offset_m'] != PDDCurvePaceDiagnostic.rear_axle_offset_m):
        raise ValueError('The transfer profile differs from the two-case diagnostic')
    bindings = {name: globals()[name] for name in ('track_pdd_plan', 'lane_footprint_margin', 'pace_bonus_reasons')}
    bindings.update({f'PDDPlanPilot.{name}': getattr(PDDPlanPilot, name) for name in (
        '__init__', 'reset', 'plan_features', 'plan_geometry_diagnostics', 'predict_plan', 'track_plan', 'control_plan', 'act')})
    hashes = verify_frozen_function_sources(source_path.read_text(encoding='utf-8-sig'), bindings)
    return {'scope': 'FROZEN_SIX_ADDITIONAL_TRAIN_CASES', 'report_sha256': expected_report,
            'source_sha256': expected_source, 'behavior_function_sha256': hashes,
            'parameters': dict(PACE_SUPPORT_PARAMETERS), 'checkpoint_sha256': report['checkpoint_sha256'],
            'development_clean_cases': 2, 'parameter_tuning_during_transfer': False}


def frozen_flow_lineage():
    """Verify the successful five-case flow40 behavior before Extreme transfer."""
    import ast

    previous = ROOT / 'agent4_integration_audit/run_20261121T015940_628993Z'
    report_path = previous / 'report.json'
    source_path = previous / 'Medium_SCXCS_14_1/launcher/agent4_pdd_curve06_flow40/sources/policy.py'
    expected_report = '2d66d0e9bbdef215f14e01f7202db6472919b7f921ca9029f533cf941ef8b022'
    expected_source = 'd08921156b3ca6d4e7857677198c41074748b5c6c70d16156be77681ad75b5cc'
    if digest(report_path) != expected_report or digest(source_path) != expected_source:
        raise ValueError('Expected the frozen five-case flow40 report and source')
    report = json.loads(report_path.read_bytes())
    if (report['policy'] != PDDCurveFlow40Diagnostic.agent_id or report['pdd_clean'] != 5
            or report['complete_pairs'] != 5 or report['planned_pairs'] != 5
            or report['checkpoint_sha256'] != PDDCurveFlow40Diagnostic.checkpoint_hash
            or report['rear_axle_offset_m'] != PDDCurveFlow40Diagnostic.rear_axle_offset_m
            or report['controller_support_parameters'] != PACE_SUPPORT_PARAMETERS
            or report['navigation40_parameters'] != NAV40_PARAMETERS
            or report['flow40_parameters'] != FLOW40_PARAMETERS):
        raise ValueError('The transfer profile differs from the successful five-case flow40 diagnostic')
    source = source_path.read_text(encoding='utf-8-sig')
    # Risk field availability is part of the behavior, including measured None.
    risk_fields = next(ast.literal_eval(node.value) for node in ast.parse(source).body
                       if isinstance(node, ast.Assign) and any(
                           isinstance(target, ast.Name) and target.id == 'FLOW_RISK_FIELDS' for target in node.targets))
    if risk_fields != FLOW_RISK_FIELDS:
        raise ValueError('Frozen flow40 risk fields changed')
    bindings = {name: globals()[name] for name in (
        'track_pdd_plan', 'lane_footprint_margin', 'pace_bonus_reasons', 'cruise40_reasons', 'flow_progress_reasons')}
    for cls, names in (
        (PDDSpeedPilot, ('__init__', 'reset', '_observe')),
        (PDDPlanPilot, ('__init__', 'reset', 'plan_features', 'plan_geometry_diagnostics',
                        'predict_plan', 'track_plan', 'control_plan', 'act')),
        (PDDCurveNavigation40Diagnostic, ('__init__', '_observe', 'control_plan')),
        (PDDCurveFlow40Diagnostic, ('_observe', 'control_plan')),
    ):
        bindings.update({f'{cls.__name__}.{name}': getattr(cls, name) for name in names})
    hashes = verify_frozen_function_sources(source, bindings)
    return {'scope': 'FROZEN_THREE_EXTREME_TRAIN_CASES', 'report_sha256': expected_report,
            'source_sha256': expected_source, 'behavior_function_sha256': hashes,
            'checkpoint_sha256': report['checkpoint_sha256'], 'development_clean_cases': 5,
            'parameter_tuning_during_transfer': False}


GAP_REVIEW_PARAMETERS = {'minimum_launch_acceleration_mps2': .8, 'reaction_delay_s': .4,
                         'minimum_clear_window_s': .5, 'stop_threshold_mps': .3}


def junction_arrival_clock(distance, speed, cap, acceleration):
    """One coherent accelerate-then-cruise clock, for offline sensitivity review."""
    import numpy as np

    distance = np.asarray(distance, dtype=float)
    if (not np.isfinite(distance).all() or not all(math.isfinite(value) for value in (speed, cap, acceleration))
            or speed < 0 or cap <= 0 or acceleration <= 0):
        raise ValueError('Invalid junction clock inputs')
    distance, initial = np.maximum(0., distance), min(speed, cap)
    ramp_distance = (cap*cap-initial*initial)/(2*acceleration)
    ramp_time = (np.sqrt(initial*initial+2*acceleration*distance)-initial)/acceleration
    return np.where(distance <= ramp_distance, ramp_time,
                    (cap-initial)/acceleration+(distance-ramp_distance)/cap)


def junction_occupancy_window(arcs, envelope, speed, cap, acceleration, *, lower_cap=None,
                              lower_acceleration=None, delay=0., gap_s=3.):
    """Bound entry/exit with the same clock; never infer acceleration feasibility."""
    import numpy as np

    if lower_cap is None:
        lower_cap = cap
    if lower_acceleration is None:
        lower_acceleration = acceleration
    envelope = np.asarray(envelope, dtype=float)
    if (not np.isfinite(envelope).all() or np.any(envelope < 0) or not math.isfinite(delay)
            or not math.isfinite(gap_s) or delay < 0 or gap_s < 0
            or not 0 < lower_cap <= cap or not 0 < lower_acceleration <= acceleration):
        raise ValueError('Invalid junction occupancy bounds')
    arcs = np.asarray(arcs, dtype=float)
    entry = junction_arrival_clock(arcs-envelope, speed, cap, acceleration)-gap_s/2
    exit_time = junction_arrival_clock(arcs+envelope, speed, lower_cap, lower_acceleration)+delay+gap_s/2
    return entry, exit_time


def review_junction_gap(controller, corridor, tracks, speed, config, plan):
    """Shadow only: preserve queue/geometry checks and expose timing assumptions."""
    import numpy as np

    cap, acceleration = config.junction_speed_kmh/3.6, config.junction_launch_acceleration_mps2
    lower_acceleration = min(acceleration, GAP_REVIEW_PARAMETERS['minimum_launch_acceleration_mps2'])
    valid_plan = (isinstance(plan, list) and len(plan) == 6 and all(
        isinstance(point, list) and len(point) == 3 and all(math.isfinite(value) for value in point) for point in plan))
    model_cap = min(cap, min(point[2] for point in plan)) if valid_plan else 0.
    plan_clock_available = model_cap > 0
    # Missing/zero forecast cannot establish a model-supported traversal clock.
    lower_cap = model_cap if plan_clock_available else cap
    exit_known = controller.conflict_exit is not None and (
        controller.conflict_exit+config.junction_exit_clearance_m <= corridor.arc[-1])
    end = controller.conflict_exit if controller.conflict_exit is not None else corridor.arc[-1]
    blocked, nominal_conflicts, bounded_conflicts = False, set(), set()
    times = np.arange(0, config.junction_prediction_s+1e-6, config.prediction_dt_s)
    for track in tracks:
        actor = track.actor
        arcs, distances, tangents = corridor.project([(actor.relative_position_x, actor.relative_position_y)])
        s, distance, tangent = float(arcs[0]), float(distances[0]), float(tangents[0])
        delta = actor.relative_heading-tangent
        aligned = abs(math.atan2(math.sin(delta), math.cos(delta))) < .5
        actor_speed = float(np.linalg.norm(track.velocity))
        footprint = actor.length/2+config.vehicle_length_m/2+config.minimum_following_gap_m
        if aligned and distance < config.vehicle_width_m/2+actor.width/2+config.traffic_margin_m:
            along_speed = float(track.velocity @ np.array((math.cos(tangent), math.sin(tangent))))
            if (s > max(0, controller.yield_entry) and s-footprint < end+config.junction_exit_clearance_m
                    and along_speed < config.junction_queue_speed_mps):
                blocked = True
            if along_speed > config.junction_queue_speed_mps and abs(track.yaw_rate) < .1:
                continue
        models = list(track.predictions(times, nearby_turn_horizon_s=2.))
        if actor_speed < .5 and not aligned:
            launch = .5*config.stationary_actor_acceleration_mps2*np.minimum(times, 2)**2
            launch += np.maximum(0, times-2)*2*config.stationary_actor_acceleration_mps2
            direction = np.array((math.cos(actor.relative_heading), math.sin(actor.relative_heading)))
            origin = np.array((actor.relative_position_x, actor.relative_position_y))
            models.append((origin+launch[:, None]*direction, np.full(len(times), actor.relative_heading),
                           np.ones(len(times), dtype=bool)))
        for predicted, actor_headings, valid in models:
            arcs, distances, headings = corridor.project(predicted)
            angle = actor_headings-headings
            half_width = (actor.width*np.abs(np.cos(angle))+actor.length*np.abs(np.sin(angle)))/2
            half_length = (actor.length*np.abs(np.cos(angle))+actor.width*np.abs(np.sin(angle)))/2
            geometry = (valid & (arcs >= max(0, controller.yield_entry-config.vehicle_length_m/2))
                        & (arcs <= min(corridor.arc[-1], end+config.vehicle_length_m/2))
                        & (distances <= config.vehicle_width_m/2+half_width+config.traffic_margin_m+track.uncertainty_m))
            envelope = half_length+config.vehicle_length_m/2
            nominal = junction_occupancy_window(arcs, envelope, speed, cap, acceleration, gap_s=config.junction_gap_s)
            bounded = junction_occupancy_window(arcs, envelope, speed, cap, acceleration,
                lower_cap=lower_cap, lower_acceleration=lower_acceleration,
                delay=GAP_REVIEW_PARAMETERS['reaction_delay_s'], gap_s=config.junction_gap_s)
            for window, conflicts in ((nominal, nominal_conflicts), (bounded, bounded_conflicts)):
                if np.any(geometry & (times >= window[0]) & (times <= window[1])):
                    conflicts.add(track.track_id)
    horizon_covered = bool(exit_known and plan_clock_available and float(junction_arrival_clock(
        end+config.vehicle_length_m/2, speed, lower_cap, lower_acceleration))
        +GAP_REVIEW_PARAMETERS['reaction_delay_s'] <= config.junction_prediction_s)
    return {'queue_blocked': bool(blocked), 'exit_known': bool(exit_known),
            'nominal_conflicts': sorted(nominal_conflicts), 'bounded_conflicts': sorted(bounded_conflicts),
            'plan_clock_available': plan_clock_available, 'model_speed_cap_mps': model_cap,
            'prediction_horizon_covered': horizon_covered}


def compare_trace_prefixes(first, second):
    """Locate repeat divergence without conflating environment and policy repeatability."""
    with gzip.open(first, 'rt', encoding='utf-8') as left, gzip.open(second, 'rt', encoding='utf-8') as right:
        count = 0
        while True:
            a, b = left.readline(), right.readline()
            if not a or not b:
                return {'identical_prefix_steps': count, 'status': 'IDENTICAL' if a == b else 'LENGTH_DIVERGED'}
            a, b = json.loads(a), json.loads(b)
            sections = [key for key in set(a['agent_input']) | set(b['agent_input'])
                        if a['agent_input'].get(key) != b['agent_input'].get(key)]
            action_changed = a['canonical_action'] != b['canonical_action']
            if sections or action_changed:
                return {'identical_prefix_steps': count, 'first_divergent_step': a['step_index'],
                        'status': 'INPUT_DIVERGED' if sections else 'ACTION_DIVERGED_WITH_IDENTICAL_INPUT_PREFIX',
                        'changed_input_sections': sorted(sections), 'action_changed': action_changed}
            count += 1


def recorded_stop_metrics(trace):
    stopped = longest = streak = 0
    with gzip.open(trace, 'rt', encoding='utf-8') as stream:
        for line in stream:
            motion = json.loads(line)['agent_input']['ego_motion']
            slow = motion is not None and math.hypot(motion['velocity_x_mps'], motion['velocity_y_mps']) < .3
            stopped += int(slow)
            streak = streak+1 if slow else 0
            longest = max(longest, streak)
    return {'stopped_s': stopped/10, 'longest_stop_s': longest/10}


def compare_stop_observations(current, historical, historical_stops):
    """A shorter failed episode cannot establish less waiting than a full run."""
    raw_stop = round(current['pdd_stopped_s']-historical_stops['stopped_s'], 3)
    raw_longest = round(current['pdd_longest_stop_s']-historical_stops['longest_stop_s'], 3)
    both_clean = bool(current['pdd_clean'] and historical['pdd_clean'])
    same_duration = math.isclose(current['pdd_episode_time_s'], historical['pdd_episode_time_s'],
                                 abs_tol=.100001, rel_tol=0.)
    comparable = both_clean or same_duration
    scope = ('BOTH_CLEAN' if both_clean else 'EQUAL_OBSERVATION_DURATION_FAILED_RUNS'
             if same_duration else 'UNEQUAL_DURATION_WITH_FAILURE')
    return {'raw_stopped_change_vs_historical_flow40_s': raw_stop,
            'raw_longest_stop_change_vs_historical_flow40_s': raw_longest,
            'stopped_change_vs_historical_flow40_s': raw_stop if comparable else None,
            'longest_stop_change_vs_historical_flow40_s': raw_longest if comparable else None,
            'stop_observation_comparison_scope': scope,
            'stop_reduction_with_both_clean_arrivals': bool(both_clean and raw_stop < 0)}


VISIBILITY_APPROACH_PARAMETERS = {'speed_cap_kmh': 2., 'minimum_wait_s': 5.,
    'prediction_s': 4., 'sample_dt_s': .2, 'stop_margin_m': .1, 'footprint_margin_m': .35,
    'geometry_contract': 'OBSERVED_ZERO_WITH_OPTIONAL_REAR_PREFIX_V2'}


def visibility_approach(snapshot, hint, reference, was_active=False):
    """Allow upstream observation approach only; never grants junction admission."""
    import numpy as np

    parameters = VISIBILITY_APPROACH_PARAMETERS
    reasons, risk = [], hint.get('traffic_risk') or {}
    command = reference['original_command']
    result = {'active': False, 'command': command, 'original_command': command, 'block_reasons': reasons}
    if hint.get('junction_phase') not in ('YIELD', 'BLOCKED') or hint.get('junction_reason') not in ('exit_blocked', 'exit_not_visible'):
        reasons.append('NOT_VISIBILITY_APPROACH_PHASE')
    if reference.get('branch_locked') is not False or reference.get('front_entered') is not False:
        reasons.append('ALREADY_COMMITTED_OR_ENTERED')
    if not was_active and (reference.get('wait_s') is None or reference['wait_s'] < parameters['minimum_wait_s']):
        reasons.append('WAIT_THRESHOLD')
    motion, task = snapshot.get('ego_motion'), snapshot['task_context']
    lane_context = task.get('lane_context')
    lanes = [] if not lane_context else [lane for lane in lane_context['lanes'] if lane['side'] == 'CURRENT']
    if motion is None or len(lanes) != 1 or lanes[0].get('route_compatible') is not True:
        reasons.append('MISSING_CURRENT_ROUTE')
    required = ('emergency_brake', 'conflict_distance_m', 'conflict_time_s', 'lidar_obstacle_gap_m',
                'traffic_overflow_count', 'actor_motion_path_overflow_count', 'ghost_track_count', 'tracked_actor_count')
    if any(name not in risk for name in required):
        reasons.append('MISSING_TRAFFIC_ASSESSMENT')
    elif (risk['emergency_brake'] is not False or risk['conflict_distance_m'] is not None
            or risk['conflict_time_s'] is not None or risk['lidar_obstacle_gap_m'] is not None):
        reasons.append('IMMEDIATE_HAZARD')
    if (risk.get('traffic_overflow_count') != 0 or risk.get('actor_motion_path_overflow_count') != 0
            or risk.get('ghost_track_count') != 0
            or not isinstance(risk.get('tracked_actor_count'), int)
            or risk['tracked_actor_count'] < snapshot['traffic_context']['active_count']):
        reasons.append('INCOMPLETE_TRAFFIC')
    if hint.get('lane_change_phase') != 'KEEP_LANE' or (lane_context or {}).get('required_side') is not None:
        reasons.append('LANE_MANEUVER_OR_REQUIRED_SIDE')
    limits = reference.get('speed_limits_kmh') or {}
    if hint.get('speed_limit_reason') != 'junction_phase' or 'junction_phase' not in limits:
        reasons.append('OTHER_LIMITER')
    valid_waypoints = [point['lookahead_distance_m'] for point, valid in zip(task['waypoints'], task['validity_mask']) if valid]
    visible = max(valid_waypoints, default=0.)
    conflict_exit = reference.get('conflict_exit_m')
    if conflict_exit is not None and conflict_exit+reference['exit_clearance_m'] <= visible:
        reasons.append('EXIT_ALREADY_VISIBLE')
    yield_distance = reference.get('yield_distance_m')
    heading_error = 0. if motion is None else motion['lane_heading_error_rad']
    half_length = (reference['vehicle_length_m']*abs(math.cos(heading_error))
                   +reference['vehicle_width_m']*abs(math.sin(heading_error)))/2
    protected = None if yield_distance is None else yield_distance-half_length-reference['entry_buffer_m']
    if protected is None or protected <= parameters['stop_margin_m']+.05:
        reasons.append('PROTECTED_STOP_REACHED')
    if reasons:
        return result
    lane = lanes[0]
    points = np.asarray(lane['points'], dtype=float)
    if (points.ndim != 2 or points.shape[1] != 4 or len(points) < 2 or not np.isfinite(points).all()
            or np.any(np.diff(points[:, 0]) <= 0)):
        reasons.append('INVALID_APPROACH_GEOMETRY')
        return result
    # Public lane geometry can start behind ego. Require the actual observed
    # zero sample, as the existing lane-pose encoder does; never extrapolate it.
    zero_points = points[np.abs(points[:, 0]) <= 1e-6]
    if len(zero_points) != 1:
        reasons.append('MISSING_OR_AMBIGUOUS_CURRENT_ZERO_POSE')
        return result
    zero = zero_points[0]
    if (abs(zero[2]) > .2 or abs(motion['lane_heading_error_rad']) > .08
            or abs(zero[3]) > .08
            or lane['width_m']/2-abs(zero[2])-reference['vehicle_width_m']/2 < .35):
        reasons.append('LANE_POSE_OR_MARGIN')
        return result
    speed = math.hypot(motion['velocity_x_mps'], motion['velocity_y_mps'])
    other_caps = [float(value) for name, value in limits.items() if name != 'junction_phase']
    if not other_caps or not all(math.isfinite(value) for value in other_caps):
        reasons.append('MISSING_OTHER_CAPS')
        return result
    usable = max(0., protected-parameters['stop_margin_m']-reference['response_time_s']*speed)
    cap = min(parameters['speed_cap_kmh'], min(other_caps),
              3.6*math.sqrt(2*reference['deceleration_mps2']*usable))
    if cap <= 0:
        reasons.append('OTHER_CAP_OR_STOPPING_DISTANCE')
        return result
    preview = min(protected-parameters['stop_margin_m'], cap/3.6*parameters['prediction_s']
                  +reference['response_time_s']*speed+speed*speed/(2*reference['deceleration_mps2'])+.5)
    if preview <= 0 or preview > points[-1, 0] or preview > lane['remaining_m']:
        reasons.append('APPROACH_GEOMETRY_TOO_SHORT')
        return result
    samples = np.linspace(0., preview, max(2, math.ceil(preview/.25)+1))
    centers = np.column_stack([np.interp(samples, points[:, 0], points[:, index]) for index in (1, 2)])
    headings = np.interp(samples, points[:, 0], np.unwrap(points[:, 3]))
    if np.max(np.abs(headings-headings[0])) > .1:
        reasons.append('APPROACH_BEND')
        return result
    times = np.arange(0., parameters['prediction_s']+1e-6, parameters['sample_dt_s'])
    traffic = snapshot['traffic_context']
    actors = traffic['actors']
    ego_velocity = np.array([motion['velocity_x_mps'], motion['velocity_y_mps']])
    for slot, (actor, valid) in enumerate(zip(actors, traffic['validity_mask'])):
        if not valid:
            continue
        position = np.array([actor['relative_position_x'], actor['relative_position_y']])
        velocity = ego_velocity+np.array([actor['relative_velocity_x'], actor['relative_velocity_y']])
        actor_speed = float(np.linalg.norm(velocity))
        travel = times*actor_speed
        if actor_speed < .5:
            travel = np.maximum(travel, .5*np.minimum(times, 2)**2+2*np.maximum(0, times-2))
        models = [(position+times[:, None]*velocity, np.full(len(times), actor['relative_heading']), np.ones(len(times), dtype=bool))]
        if actor_speed < .5:
            direction = np.array([math.cos(actor['relative_heading']), math.sin(actor['relative_heading'])])
            models.append((position+travel[:, None]*direction, np.full(len(times), actor['relative_heading']), np.ones(len(times), dtype=bool)))
        for path in traffic.get('motion_paths', []):
            if path['actor_slot'] != slot:
                continue
            path_points = np.asarray(path['points'], dtype=float)
            models.append((np.column_stack([np.interp(travel, path_points[:, 0], path_points[:, index]) for index in (1, 2)]),
                np.interp(travel, path_points[:, 0], np.unwrap(path_points[:, 3])), travel <= path_points[-1, 0]))
        if actor_speed > .3:
            direction = math.atan2(velocity[1], velocity[0])
            for yaw_rate in (-.6, .6):
                turning = position+actor_speed/yaw_rate*np.column_stack((
                    np.sin(direction+yaw_rate*times)-math.sin(direction),
                    -np.cos(direction+yaw_rate*times)+math.cos(direction)))
                models.append((turning, actor['relative_heading']+yaw_rate*times, np.ones(len(times), dtype=bool)))
        for positions, actor_headings, prediction_valid in models:
            delta = positions[:, None, :]-centers[None, :, :]
            angle = actor_headings[:, None]-headings[None, :]
            cosine, sine = np.abs(np.cos(angle)), np.abs(np.sin(angle))
            margin = parameters['footprint_margin_m']+.15*times[:, None]+parameters['sample_dt_s']*(actor_speed+2)/2
            ego_l, ego_w = reference['vehicle_length_m']/2, reference['vehicle_width_m']/2
            actor_l, actor_w = actor['length']/2, actor['width']/2
            egox = np.abs(delta[:, :, 0]*np.cos(headings)+delta[:, :, 1]*np.sin(headings))
            egoy = np.abs(-delta[:, :, 0]*np.sin(headings)+delta[:, :, 1]*np.cos(headings))
            actorx = np.abs(delta[:, :, 0]*np.cos(actor_headings[:, None])+delta[:, :, 1]*np.sin(actor_headings[:, None]))
            actory = np.abs(-delta[:, :, 0]*np.sin(actor_headings[:, None])+delta[:, :, 1]*np.cos(actor_headings[:, None]))
            overlap = (prediction_valid[:, None] & (egox <= ego_l+actor_l*cosine+actor_w*sine+margin)
                & (egoy <= ego_w+actor_l*sine+actor_w*cosine+margin)
                & (actorx <= actor_l+ego_l*cosine+ego_w*sine+margin)
                & (actory <= actor_w+ego_l*sine+ego_w*cosine+margin))
            if np.any(overlap):
                reasons.append('ACTOR_SWEPT_APPROACH')
                result['conflict_actor_slot'] = slot
                return result
    proposed = max(-1., min(reference['max_throttle'], reference['speed_gain']*(cap-speed*3.6)))
    if proposed <= command+1e-8:
        reasons.append('REFERENCE_ALREADY_PROGRESSING')
        return result
    result.update(active=True, command=proposed, speed_cap_kmh=cap, protected_stop_distance_m=protected,
                  checked_preview_m=preview)
    return result


def observed_lane_curvature(start, end):
    """Signed mean curvature from two observed centre poses (x, y, left yaw).

    Uses the actual XY chord, not OpenDRIVE road-s or a nominal lookahead.
    Exact for a planar circular arc; otherwise a chord-based local estimate.
    """
    if not all(math.isfinite(float(value)) for point in (start, end) for value in point):
        raise ValueError('Nonfinite observed lane geometry')
    chord = math.hypot(end[0] - start[0], end[1] - start[1])
    angle = math.atan2(math.sin(end[2] - start[2]), math.cos(end[2] - start[2]))
    if chord < .1 or abs(angle) > math.pi / 2:
        return None, chord
    return 2 * math.sin(angle / 2) / chord, chord


def public_lane_curvature(snapshot, *, pose_valid=True):
    """Read CURRENT's real 0/2.5/5m samples; never extrapolate across lane ends."""
    context = snapshot['task_context']['lane_context']
    lanes = [] if context is None else [lane for lane in context['lanes'] if lane['side'] == 'CURRENT']
    if len(lanes) > 1:
        raise ValueError('Ambiguous CURRENT public lane')
    points = [] if not lanes else lanes[0]['points']
    samples = []
    for distance in (2.5, 5.):
        if not pose_valid:
            samples.append({'distance_m': distance, 'valid': False, 'curvature_left_inv_m': None,
                            'chord_m': None, 'flags': ['SOURCE_POSE_MASKED']})
            continue
        start = [point[1:] for point in points if abs(float(point[0])) < 1e-6]
        end = [point[1:] for point in points if abs(float(point[0]) - distance) < 1e-6]
        if len(start) > 1 or len(end) > 1:
            raise ValueError('Duplicate public lane sample')
        value, chord = (None, None) if not start or not end else observed_lane_curvature(start[0], end[0])
        samples.append({'distance_m': distance, 'valid': value is not None,
                        'curvature_left_inv_m': value, 'chord_m': chord,
                        'flags': [] if value is not None else ['PUBLIC_SAMPLE_UNAVAILABLE_OR_DEGENERATE']})
    return samples


def pdd_lane_curvature(road_map, review):
    """Query current recorded lane only; mask mismatches and road/section ends."""
    match = review['map_match']
    waypoint = None if match is None or not review['valid'] else road_map.get_waypoint_xodr(
        int(match['road_id']), int(match['lane_id']), float(match['s_m']))
    matches = waypoint is not None and (
        waypoint.road_id == match['road_id'] and waypoint.lane_id == match['lane_id']
        and waypoint.section_id == match['section_id'] and abs(waypoint.s - match['s_m']) <= .02
        and abs(waypoint.lane_width - match['width_m']) <= .05)
    samples = []
    for distance in (2.5, 5.):
        value, chord = None, None
        flags = ['SOURCE_POSE_MASKED'] if not review['valid'] else ['MAP_REQUERY_MISMATCH']
        if matches:
            candidates = [candidate for candidate in waypoint.next(distance)
                          if (candidate.road_id, candidate.lane_id, candidate.section_id) ==
                          (waypoint.road_id, waypoint.lane_id, waypoint.section_id)]
            flags = ['NO_UNIQUE_SAME_ROAD_LANE_SECTION_SAMPLE']
            if len(candidates) == 1:
                def pose(candidate):
                    transform = candidate.transform
                    return (transform.location.x, -transform.location.y, -math.radians(transform.rotation.yaw))
                value, chord = observed_lane_curvature(pose(waypoint), pose(candidates[0]))
                flags = [] if value is not None else ['DEGENERATE_OR_LARGE_HEADING_CHANGE']
        samples.append({'distance_m': distance, 'valid': value is not None,
                        'curvature_left_inv_m': value, 'chord_m': chord, 'flags': flags})
    return samples


def run_pdd_lane_curvature_audit(out):
    """Offline data/observation audit; no fitting, controller change or simulator."""
    from importlib.metadata import version
    from agent4_pdd_common_v2 import encode_public

    sources = {
        'runs/agent4_pdd_development_c0_02/audit.json': '3bd2be48eb74b6b869665eef4278625514d16030c43a11c86c7cd9ff4b22b5b9',
        'runs/agent4_pdd_development_c0_02/c0_screened_candidates.jsonl': '42eaf94fdb980c64e9d6047267b5383348778c820946cc4e26b606dbd676781e',
        'runs/agent4_pdd_lane_map_review_01/audit.json': '8cfdd7c518daf6b530dbc7b3138feb32cd04583fbeebc58177d2ef1e505004ac',
        'runs/agent4_pdd_lane_map_review_01/lane_pose_review.jsonl': 'b911a47f93ed2f8a95bff54ed1012c0f2c43b8a86b84bdd930b1f2e6d05b5997',
        'runs/agent4_pdd_town12_map_01/Town12.xodr': '26fa53d4902ac3397d35f948969f0c77c1bf023755ede2c67edf00bef817151f',
        'agent4_integration_audit/run_20261120T135335_843752Z/report.json': '6f50b3c040f6aca8cbd8dd5e105c25137f76e8a957607cd985fdf1789a7fac6b',
        'agent4_integration_audit/run_20261120T135335_843752Z/Medium_SCXCS_14_1/report.json': '558ce0bc0ddc9a64253bd8adaad79f71ddc90c41d3dd58288fdc42756dfce1a4',
        'agent4_integration_audit/run_20261120T135335_843752Z/Medium_SCXCCS_6_2/report.json': 'c4276bba5daead150aab6b0bbff2902b6453a7d8bac26c02093f056dbc78277e',
        'agent4_pdd_common_v2.py': 'cda499ff1c47fd23fef5a6b90eb26d4ebf29aab7badd80007137f4523b5eacc9',
    }
    for name, expected in sources.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Frozen curvature audit source differs: {name}')
    if version('carla') != '0.9.15':
        raise ValueError('Expected installed CARLA Python API 0.9.15')
    import carla

    def identity(row):
        expected = 'TRAIN' if 1 <= int(row['driver'].split('_')[1]) <= 21 else 'DEV'
        if row['partition'] != expected or not 1 <= int(row['driver'].split('_')[1]) <= 25:
            raise ValueError('Only frozen TRAIN/DEV identities are allowed')
        return row['driver'], row['scenario'], row['telemetry_step']

    data = {}
    with (ROOT / 'runs/agent4_pdd_development_c0_02/c0_screened_candidates.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            key = identity(row)
            if key in data:
                raise ValueError(f'Duplicate C0 identity: {key}')
            data[key] = row
    print('Parsing Town12 locally; no simulator or training...', flush=True)
    road_map = carla.Map('Town12', (ROOT / 'runs/agent4_pdd_town12_map_01/Town12.xodr').read_text(encoding='utf-8-sig'))

    def new_group():
        return {'rows': 0, 'valid': Counter(), 'flags': Counter(), 'values': {2.5: [], 5.: []},
                'speed_groups': {}, 'joint_groups': Counter()}

    def accumulate(group, samples, features, future_speed=None):
        group['rows'] += 1
        speed = float(features[0]) if features[3] else None
        for sample in samples:
            distance = sample['distance_m']
            group['flags'].update(sample['flags'])
            if sample['valid']:
                group['valid'][distance] += 1
                group['values'][distance].append(sample['curvature_left_inv_m'])
        # Diagnostic counts only. No actor is not a claim that the road is clear.
        traffic = ('overflow' if features[89] > 0 else
                   'actors_present' if features[88] > 0 else 'no_visible_actors')
        speed_group = group['speed_groups'].setdefault(traffic, {'n': 0, 'speed_sum_mps': 0.,
                      'speed_ge_2': 0, 'speed_ge_5': 0, 'speed_ge_8': 0,
                      'human_accelerate_1s': 0, 'human_slowdown_1s': 0})
        if speed is not None:
            speed_group['n'] += 1
            speed_group['speed_sum_mps'] += speed
            for threshold in (2, 5, 8):
                speed_group[f'speed_ge_{threshold}'] += int(speed >= threshold)
            if future_speed is not None:
                speed_group['human_accelerate_1s'] += int(future_speed - speed >= .5)
                speed_group['human_slowdown_1s'] += int(speed - future_speed >= .5)
        five = samples[1]
        curve_bin = ('missing' if not five['valid'] else
                     'abs_k_ge_0.04' if abs(five['curvature_left_inv_m']) >= .04 else
                     'abs_k_0.02_to_0.04' if abs(five['curvature_left_inv_m']) >= .02 else 'abs_k_lt_0.02')
        speed_bin = 'missing' if speed is None else 'v_lt_2' if speed < 2 else 'v_2_to_5' if speed < 5 else 'v_ge_5'
        group['joint_groups'][f'{curve_bin}/{speed_bin}/{traffic}'] += 1

    def summarize(group):
        distributions = {}
        for distance, values in group['values'].items():
            ordered = sorted(values)
            distributions[str(distance)] = {'n': len(values), 'min': min(values) if values else None,
                'max': max(values) if values else None,
                'p10': ordered[int((len(ordered)-1)*.1)] if values else None,
                'median': ordered[len(ordered)//2] if values else None,
                'p90': ordered[int((len(ordered)-1)*.9)] if values else None,
                'abs_ge_0.04': sum(abs(value) >= .04 for value in values),
                'abs_ge_1_over_17': sum(abs(value) >= 1/17 - 1e-5 for value in values)}
        speeds = {}
        for name, stats in group['speed_groups'].items():
            speeds[name] = {key: value for key, value in stats.items() if key != 'speed_sum_mps'}
            speeds[name]['mean_speed_mps'] = stats['speed_sum_mps'] / stats['n'] if stats['n'] else None
        return {'rows': group['rows'], 'valid_by_distance': dict(group['valid']),
                'sample_flag_counts': dict(group['flags']), 'curvature_distributions': distributions,
                'speed_groups': speeds, 'joint_curve_speed_traffic_counts': dict(group['joint_groups'])}

    groups = {split: new_group() for split in ('TRAIN', 'DEV')}
    seen = set()
    artifact = out / 'lane_curvature_review.jsonl'
    with artifact.open('x', encoding='utf-8') as destination:
        with (ROOT / 'runs/agent4_pdd_lane_map_review_01/lane_pose_review.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                review = json.loads(line)
                key = identity(review)
                if key not in data or key in seen or data[key]['partition'] != review['partition']:
                    raise ValueError(f'Curvature/C0 identity mismatch: {key}')
                seen.add(key)
                samples = pdd_lane_curvature(road_map, review)
                accumulate(groups[review['partition']], samples, data[key]['features'],
                           data[key]['human_trajectory'][3][2])
                destination.write(json.dumps({'driver': key[0], 'scenario': key[1], 'telemetry_step': key[2],
                    'partition': review['partition'], 'source_pose_valid': review['valid'],
                    'samples': samples}, allow_nan=False, separators=(',', ':')) + '\n')
                if len(seen) % 5000 == 0:
                    print(f'Reviewed {len(seen)}/{len(data)} PDD rows', flush=True)
    if seen != set(data) or {split: group['rows'] for split, group in groups.items()} != {'TRAIN': 24381, 'DEV': 3893}:
        raise ValueError('Incomplete frozen TRAIN/DEV coverage')
    del data, road_map

    runtime = ROOT / 'agent4_integration_audit/run_20261120T135335_843752Z'
    root_report = json.loads((runtime / 'report.json').read_bytes())
    public = []
    for pair in root_report['pairs']:
        case = pair['case_id']
        folder = runtime / '_'.join(case.split('/')[1:])
        case_path = folder / 'report.json'
        case_report = json.loads(case_path.read_bytes())
        agent = case_report['agents'][root_report['policy']]
        trace = folder / 'launcher' / root_report['policy'] / 'agent_decisions.jsonl.gz'
        if (agent['episode']['split'] != 'TRAIN' or digest(trace) != agent['trace_sha256']
                or case_report['checkpoint_sha256'] != root_report['checkpoint_sha256']):
            raise ValueError(f'Public TRAIN trace integrity mismatch: {case}')
        group, changed, witnesses = new_group(), 0, []
        with gzip.open(trace, 'rt', encoding='utf-8') as stream:
            for index, line in enumerate(stream):
                row = json.loads(line)
                if row['case_id'] != case or row['step_index'] != index:
                    raise ValueError(f'Public case/step mismatch: {case}/{index}')
                diag = row['decision']['diagnostics']
                samples = public_lane_curvature(row['agent_input'], pose_valid=bool(diag['lane_pose_valid']))
                features = encode_public(row['agent_input'])
                accumulate(group, samples, features)
                changed += int(diag['pdd_command'] is not None
                               and diag['pdd_command'] < diag['stage1_command'] - 1e-6)
                if index in (100, 250, 300, 600, 800, 1000, 1060, 1100):
                    witnesses.append({'step_index': index, 'speed_mps': features[0], 'samples': samples,
                                      'pdd_plan': diag['pdd_plan'], 'pdd_command': diag['pdd_command'],
                                      'stage1_command': diag['stage1_command'],
                                      'ego_motion': row['agent_input']['ego_motion'],
                                      'lane_context': row['agent_input']['task_context']['lane_context']})
        if group['rows'] != int(agent['episode']['episode_steps']):
            raise ValueError(f'Incomplete public trace: {case}')
        public.append({'case_id': case, 'summary': summarize(group), 'model_limits_longitudinal_steps': changed,
                       'trace_sha256': digest(trace), 'case_report_sha256': digest(case_path), 'witnesses': witnesses})
    report = {'schema': 'PDD_CURRENT_LANE_CURVATURE_AUDIT_V1', 'status': 'REVIEW_REQUIRED',
              'training_ready': False, 'deployment_eligible': False, 'features_added_to_model': False,
              'human_labels_changed': False, 'simulator_launched': False,
              'carla_api_version': version('carla'), 'source_sha256': sources,
              'implementation_sha256': digest(Path(__file__)), 'artifact_sha256': digest(artifact),
              'contract': {'lookahead_m': [2.5, 5.], 'positive_direction': 'left',
                  'formula': '2*sin(wrap(heading_end-heading_start)/2)/actual_xy_chord',
                  'pdd': 'Waypoint.next; same road/lane/section only; original pose mask retained',
                  'public': 'CURRENT.points real samples only; original pose mask retained; no interpolation or endpoint extrapolation',
                  'human_speed_change_threshold_mps': .5,
                  'human_speed_change_horizon_s': 1.,
                  'limitations': ['mean local estimate, not exact pointwise curvature on varying curves',
                      'OpenDRIVE sampling distance and public lane distance require coverage review',
                      'curvature support counts do not establish full observation distribution equivalence',
                      'no_visible_actors is not an unobstructed-road or optimal-speed label',
                      'C0 excludes junction or non-None traffic-light states; broader behavior remains open']},
              'pdd': {split: summarize(group) for split, group in groups.items()}, 'public_train': public}
    with (out / 'report.json').open('x', encoding='utf-8') as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2, allow_nan=False)
    print(f'CURVATURE_REVIEW_REQUIRED. No checkpoint changed. Read {out / "report.json"}', flush=True)


class PDD96LauncherFixture:
    """Exercise the PDD data contract while the existing rule policy drives."""

    def __init__(self, data_audit):
        from agent4_pdd_common_v2 import FEATURE_NAMES, SCHEMA
        from src.agents.rule_based_agent import Stage1RuleBasedAgent
        from src.platform.agent import AgentDescriptor

        payload = data_audit.read_bytes()
        audit = json.loads(payload)
        encoder_sha256 = digest(ROOT / 'agent4_pdd_common_v2.py')
        if (audit['schema'] != SCHEMA or audit['feature_names'] != list(FEATURE_NAMES)
                or audit['horizons_s_provisional'] != [.25, .5, .75, 1., 1.25, 1.5]
                or audit['implementation_sha256']['agent4_pdd_common_v2.py'] != encoder_sha256
                or not audit['c0_geometry_screen']['exported']):
            raise ValueError('Incompatible PDD pilot data audit')
        self._data_audit_sha256 = hashlib.sha256(payload).hexdigest()
        self._encoder_sha256 = encoder_sha256
        self._rule = Stage1RuleBasedAgent()
        self._next_step = None
        self.descriptor = AgentDescriptor(agent_id='agent4_pdd96_fixture', agent_version='0.1.0',
                                          stateful_within_episode=True, method_family='fixture')

    def reset(self, public_context, agent_seed=None):
        if (public_context.control_frequency_hz != 10 or public_context.input_profile_id != 'STATE_DECISION_V1'
                or public_context.action_adapter_id != 'continuous_box2_v1'):
            raise ValueError('Unexpected PDD fixture runtime contract')
        self._rule.reset(public_context, agent_seed)
        self._next_step = 0

    def act(self, agent_input):
        from src.platform.agent import AgentDecision

        if agent_input.step_index != self._next_step:
            raise ValueError('PDD fixture expects consecutive public steps after reset')
        feature_sha256 = public_feature_digest(agent_input)
        decision = self._rule.act(agent_input)
        self._next_step += 1
        return AgentDecision([float(value) for value in decision.action_payload],
                             {'pdd96_feature_sha256': feature_sha256})

    def export_config(self):
        return {'controller': 'stage1_rule_based', 'measurement_scope': 'PDD96_LAUNCHER_FIXTURE',
                'data_audit_sha256': self._data_audit_sha256, 'encoder_sha256': self._encoder_sha256,
                'controller_config': self._rule.export_config()}

    def close(self):
        self._rule.close()
        self._next_step = None


def launch_train_case(project, out, registry, agent_id, tier, sequence, geometry_seed, run_id, environment_seed=0):
    from src.launcher.executor import ExperimentExecutor
    from src.launcher.models import LauncherMode, LaunchRequestV1
    from src.launcher.resolver import resolve_experiment_plan
    from src.platform import WandbMode

    request = LaunchRequestV1(mode=LauncherMode.SANDBOX, agent_id=agent_id, tier=tier, sequence=sequence,
                              geometry_generation_seed=geometry_seed, environment_seed=environment_seed,
                              render_mode='OFF', wandb_mode=WandbMode.DISABLED)
    plan = resolve_experiment_plan(request, registry=registry, project_root=project)
    if len(plan.resolved_cases) != 1 or plan.resolved_cases[0].split != 'TRAIN' or plan.canonical_run:
        raise ValueError('Expected one TRAIN sandbox case')
    result = ExperimentExecutor(plan, registry=registry, project_root=project,
                                runs_root=out / 'launcher', custom_run_id=run_id).execute()
    if result.status != 'COMPLETE' or result.completed_episodes != 1:
        raise RuntimeError(f'{agent_id}: launcher did not complete an episode. See {result.run_dir}')
    run_dir = Path(result.run_dir)
    with (run_dir / 'episodes.csv').open(encoding='utf-8', newline='') as stream:
        episodes = list(csv.DictReader(stream))
    if len(episodes) != 1 or episodes[0]['has_technical_failure'].lower() != 'false':
        raise RuntimeError(f'{agent_id}: technical failure. See {run_dir}')
    return result, run_dir, episodes[0]


def run_pdd_runtime_smoke(project, out):
    from src.launcher.models import AgentRegistrationV1
    from src.launcher.registry import build_canonical_agent_registry
    from src.logging.agent_experiment import write_json

    data_audit = ROOT / 'runs/agent4_pdd_g2_c0_screen_01/audit.json'
    registry = build_canonical_agent_registry()
    registration = AgentRegistrationV1(
        agent_id='agent4_pdd96_fixture', agent_version='0.1.0', method_family='fixture',
        implementation_ref=f'{PDD96LauncherFixture.__module__}:{PDD96LauncherFixture.__qualname__}',
        stateful_within_episode=True,
        description='PDD96 public encoder with existing rule-based fixture control.',
    )
    registry.register(registration, partial(PDD96LauncherFixture, data_audit))
    result, run_dir, episode = launch_train_case(project, out, registry, registration.agent_id,
                                                 'Easy', 'SCS', 1, 'episode')
    trace = run_dir / 'agent_decisions.jsonl.gz'
    count = 0
    max_error = 0.0
    with gzip.open(trace, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            decision, observation = row['decision'], row['agent_input']
            if (row['case_id'] != episode['case_id'] or row['step_index'] != count or observation['step_index'] != count
                    or decision['diagnostics']['pdd96_feature_sha256'] != public_feature_digest(observation)):
                raise ValueError(f'PDD fixture encoder trace mismatch at step {count}')
            raw, applied = decision['action_payload'], row['canonical_action']
            if len(raw) != 2 or len(applied) != 2 or any(not math.isfinite(x) or abs(x) > 1 for x in raw + applied):
                raise ValueError(f'Invalid PDD fixture action at step {count}')
            max_error = max(max_error, *(abs(x - y) for x, y in zip(raw, applied)))
            count += 1
    if not count or count != int(episode['episode_steps']) or max_error > 1e-6:
        raise ValueError('PDD fixture action trace failed parity')
    provenance = json.loads((run_dir / 'agent_provenance.json').read_text(encoding='utf-8'))['configuration']
    report = {'status': 'PASS', 'scope': 'PDD96_LAUNCHER_FIXTURE', 'training_ready': False,
              'controller': 'stage1_rule_based', 'trace_steps': count, 'max_action_error': max_error,
              'data_audit_sha256': provenance['data_audit_sha256'], 'encoder_sha256': provenance['encoder_sha256'],
              'trace': str(trace), 'trace_sha256': digest(trace), 'execution': result.to_dict()}
    write_json(out / 'report.json', report, exclusive=True)
    print(f'PDD96 fixture PASS ({count} steps). Read {out / "report.json"}', flush=True)


class PDDSpeedPilot:
    agent_id = 'agent4_pdd_speed_pilot02'
    traffic_only = False
    forward_only = False
    checkpoint_file = 'runs/agent4_pdd_speed_pilot_02/model_bundle.joblib'
    reference_reports = None
    plan_control = False
    checkpoint_hash = '3cf936a9f72238d66f8afe5fb2df439bc4ee79b5ab8faa62a5e31b30a820e359'

    def __init__(self):
        import joblib
        from agent4_pdd_common_v2 import FEATURE_NAMES, SCHEMA
        from src.agents.rule_based_agent import Stage1RuleBasedAgent
        from src.platform.agent import AgentDescriptor

        path = ROOT / self.checkpoint_file
        if digest(path) != self.checkpoint_hash:
            raise ValueError(f'Checkpoint checksum mismatch: {path}')
        bundle = joblib.load(path)
        names = [*FEATURE_NAMES[:90], 'speed_change_past_0p5s_mps', 'speed_change_past_0p5s_valid']
        if (bundle['schema'] != 'PDD_C0_SPEED_HISTORY_V1' or bundle['source_schema'] != SCHEMA
                or bundle['features'] != names or bundle['offline_status'] != 'DEV_IMPROVED'
                or bundle['encoder_sha256'] != digest(ROOT / 'agent4_pdd_common_v2.py')
                or bundle['history_seconds'] != 0.5 or bundle['horizons_s'] != [.25, .5, .75, 1., 1.25, 1.5]):
            raise ValueError('Incompatible speed pilot schema, history or horizons')
        self.model = bundle['model'].set_params(n_jobs=1)
        self.rule = Stage1RuleBasedAgent()
        self.history = deque(maxlen=5)
        self.next_step = None
        self.descriptor = AgentDescriptor(agent_id=self.agent_id, agent_version='0.1.0',
                                          stateful_within_episode=True, method_family='HYBRID')

    def export_config(self):
        return {'checkpoint_sha256': self.checkpoint_hash, 'schema': 'PDD_C0_SPEED_HISTORY_V1',
                'tracking_horizon_s': 1.0, 'history_ticks': 5, 'adapter_sha256': digest(__file__),
                'apply_pdd_only_with_public_actor': self.traffic_only,
                'require_forward_corridor_actor': self.forward_only,
                'controller': 'min(PDD speed tracking command, Stage 1 longitudinal command)',
                'stage1': self.rule.export_config()}

    def reset(self, public_context, agent_seed=None):
        if (public_context.control_frequency_hz != 10 or not math.isclose(public_context.control_dt_s, .1)
                or public_context.input_profile_id != 'STATE_DECISION_V1'
                or public_context.action_adapter_id != 'continuous_box2_v1'):
            raise ValueError('Unexpected speed pilot runtime contract')
        self.rule.reset(public_context, agent_seed)
        self.history.clear()
        self.next_step = 0

    def _observe(self, agent_input):
        from agent4_pdd_common_v2 import encode_public

        if agent_input.step_index != self.next_step:
            raise ValueError('Nonconsecutive speed pilot steps')
        snapshot = agent_input.to_dict()
        features = encode_public(snapshot)[:90]
        speed = features[0] if features[3] else None
        past = self.history[0] if len(self.history) == 5 else None
        self.history.append(speed)
        self.next_step += 1
        reference = self.rule.act(agent_input)
        return snapshot, features, speed, past, reference

    def act(self, agent_input):
        from src.platform.agent import AgentDecision

        snapshot, features, speed, past, reference = self._observe(agent_input)
        steer, rule_command = map(float, reference.action_payload)
        command = rule_command
        target = None
        requested = None
        enabled = not self.traffic_only or agent_input.traffic_context.active_count > 0
        if self.forward_only:
            enabled = has_forward_corridor_actor(snapshot)
        if speed is not None and enabled:
            features.extend((speed - past, 1.0) if past is not None else (0.0, 0.0))
            residuals = self.model.predict([features])[0]
            if len(residuals) != 6 or not all(map(math.isfinite, residuals)):
                raise ValueError('Invalid PDD prediction')
            target = max(0., speed + float(residuals[3]))
            requested = max(-1., min(self.rule.max_throttle, self.rule.k_speed * 3.6 * (target - speed)))
            command = min(requested, rule_command)
        rule_diag = reference.diagnostics or {}
        return AgentDecision([steer, command], {'pdd_target_1s_mps': target, 'pdd_command': requested,
                             'junction_phase': rule_diag.get('junction_phase'),
                             'junction_reason': rule_diag.get('junction_reason'),
                             'junction_exit_blocked': rule_diag.get('junction_exit_blocked'),
                             'junction_passage_id': rule_diag.get('junction_passage_id'),
                             'speed_limit_reason': rule_diag.get('speed_limit_reason'),
                             'stage1_behavior': rule_diag.get('behavior_state'),
                             'stage1_lane_change': rule_diag.get('lane_change'),
                             'stage1_lateral_offset_right_m': rule_diag.get('lateral_offset_right_m'),
                             'stage1_lane_clearance_m': rule_diag.get('lane_clearance_estimate_m'),
                             'stage1_road_clearance_m': rule_diag.get('road_clearance_estimate_m'),
                             'stage1_emergency_brake': rule_diag.get('emergency_brake'),
                             'history_valid': speed is not None and past is not None,
                             'stage1_command': rule_command, 'stage1_steer': steer})

    def close(self):
        self.rule.close()
        self.history.clear()
        self.next_step = None


class PDDTrafficOnlyPilot(PDDSpeedPilot):
    agent_id = 'agent4_pdd_speed_pilot02_traffic_only'
    traffic_only = True


def has_forward_corridor_actor(snapshot):
    """Match the offline forward-corridor group using only the current public frame."""
    lane_context = snapshot['task_context']['lane_context']
    if lane_context is None:
        return False
    current = next((lane for lane in lane_context['lanes'] if lane['side'] == 'CURRENT'), None)
    if current is None:
        return False
    traffic = snapshot['traffic_context']
    return any(actor['relative_position_x'] > 0 and abs(actor['relative_position_y']) <= current['width_m'] / 2
               for actor in traffic['actors'][:traffic['active_count']])


class PDDForwardPilot(PDDTrafficOnlyPilot):
    agent_id = 'agent4_pdd_speed_pilot02_forward'
    forward_only = True


class PDDExpandedForward(PDDForwardPilot):
    agent_id = 'agent4_pdd_speed_expanded01_forward'
    checkpoint_file = 'runs/agent4_pdd_speed_expanded_01/model_bundle.joblib'
    checkpoint_hash = '799098c4e6f4624430647c577ce8cf96ef19d98622f795f2c8f55b4990bef2dd'
    # Reuse the exact-case Stage 1 and pilot 02 references after source checks.
    reference_reports = {
        ('Medium', 'SCXCS', 14, 1): ('run_20261119T033358_046922Z', '7580d53388ea20c3f057e1b1e108d1819463cff97bf8e8b41c6066237a2901b1'),
        ('Medium', 'SCTCS', 11, 0): ('run_20261119T033358_046922Z', '2e45279066fa49586c56601658fc8a1221a21f4c07743314014ddbbbb41770fd'),
        ('Hard', 'SCXOCS', 15, 0): ('run_20261119T033358_046922Z', 'd0eefb160291502fdc3a4eb007460831f599d833a799fa0fdaf0a878fbf7b16c'),
        ('Medium', 'SCXCCS', 6, 2): ('run_20261119T034237_537097Z', '907f11392de2de37ef54f44787d44e0adf95b3172890c607af2f55e85d024f4e'),
        ('Hard', 'XTOCS', 8, 2): ('run_20261119T034237_537097Z', 'cbb504429194bd2abb430e01871a4b09a7340a8e37e4b2fb644f703e93d966ae'),
        ('Extreme', 'CrXROSTR', 3, 2): ('run_20261119T034237_537097Z', '21a85294be8b3edffb95d21388df30a4ed9ef927f5dd174df46c960b015903d0'),
        ('Extreme', 'CrXROSTR', 3, 0): ('run_20261119T035104_050648Z', '7b846319231c9b657f7dbb766c3704d4bfef8343c99c50200aaeb127a31e844e'),
        ('Extreme', 'CrXROSTR', 3, 1): ('run_20261119T035104_050648Z', '9026621b710c79f7340c5ba8a3aff9d560e43f14f1fddc038cb76a2733b0fd8c'),
    }


REVIEWED_TRAFFIC_PAIRS = {
    ('Medium', 'SCXCS', 14, 1): 'e72524b314c218ce0f317ed3a988b93051fa70aab4c2800aa079b458f2807cc0',
    ('Medium', 'SCTCS', 11, 0): 'ef06f462e5f1cae6312e4fa8e98a8bba2ab9aa7b628c57669106f9b6675f6548',
    ('Hard', 'SCXOCS', 15, 0): '07b32b4b45c8d9c4cfe6b7f6c60f7fa0a2daec8cc7aa6113da0e0ad10bb747db',
}

# Selected from TRAIN before running the frozen forward-corridor policy on them.
FORWARD_TRANSFER_CASES = (
    ('Medium', 'SCXCCS', 6, 2), ('Hard', 'XTOCS', 8, 2), ('Extreme', 'CrXROSTR', 3, 2),
)

EXTREME_EXIT_CASES = (('Extreme', 'CrXROSTR', 3, 0), ('Extreme', 'CrXROSTR', 3, 1))

PACE_SUPPORT_PARAMETERS = {
    'bonus_kmh': 2., 'bonus_target_cap_kmh': 32., 'minimum_speed_mps': 2.,
    'straight_preview_m': 30., 'maximum_preview_heading_change_rad': .1,
    'maximum_lane_offset_m': .2, 'maximum_heading_error_rad': .03,
    'maximum_tracker_steering': .05,
    'maximum_plan_lateral_m': .3, 'guard_enter_margin_m': .35,
    'guard_release_margin_m': .5, 'guard_release_ticks': 5,
    'footprint_forecast_s': 1.,
}

NAV40_PARAMETERS = {'target_kmh': 40., 'straight_preview_m': 40., 'maximum_model_brake': .2}
FLOW40_PARAMETERS = {'minimum_ttc_s': 4., 'standstill_gap_m': 3., 'following_time_s': 1.8,
                     'progress_increment_kmh': 3.}
FLOW_RISK_FIELDS = ('emergency_brake', 'junction_entry_conflict', 'conflict_distance_m', 'conflict_time_s',
                    'lidar_obstacle_gap_m', 'traffic_overflow_count', 'actor_motion_path_overflow_count',
                    'tracked_actor_count', 'ghost_track_count', 'min_ttc_s', 'front_gap_m')


def flow_progress_reasons(snapshot, plan, speed, raw_command, stage1_command, hint, guard_active, navigation_active):
    """Use explicit reference risk assessment, rather than any-actor presence, for progress."""
    if plan is None or speed is None:
        return ['MODEL_PLAN_UNAVAILABLE']
    reasons, risk = [], hint.get('traffic_risk')
    if not isinstance(risk, dict) or any(name not in risk for name in FLOW_RISK_FIELDS):
        return ['REFERENCE_RISK_UNAVAILABLE']
    if stage1_command <= 0:
        reasons.append('REFERENCE_NOT_ACCELERATING')
    if hint.get('junction_phase') in ('BLOCKED', 'DEGRADED_STOP') or hint.get('lane_change_phase') == 'BLOCKED_CHANGE':
        reasons.append('REFERENCE_BLOCKED')
    if hint.get('junction_reason') in ('exit_blocked', 'crossing_occupancy', 'committed_route_input_mismatch'):
        reasons.append('REFERENCE_PASSAGE_HAZARD')
    if hint.get('speed_limit_reason') not in ('cruise', 'curve', 'visibility', 'junction', 'junction_phase', 'lane_change'):
        reasons.append('REFERENCE_LIMIT_NOT_ELIGIBLE')
    if hint.get('target_speed_kmh') is None:
        reasons.append('REFERENCE_TARGET_UNAVAILABLE')
    if raw_command < -NAV40_PARAMETERS['maximum_model_brake']:
        reasons.append('STRONG_MODEL_BRAKE')
    if guard_active and not navigation_active:
        reasons.append('LANE_MARGIN_GUARD')
    if risk['emergency_brake'] is not False or risk['junction_entry_conflict'] is not False:
        reasons.append('REFERENCE_EMERGENCY_OR_JUNCTION_CONFLICT')
    if risk['conflict_distance_m'] is not None or risk['conflict_time_s'] is not None or risk['lidar_obstacle_gap_m'] is not None:
        reasons.append('REFERENCE_CONFLICT_OR_OBSTACLE')
    if (risk['traffic_overflow_count'] != 0 or risk['actor_motion_path_overflow_count'] != 0
            or risk['ghost_track_count'] != 0):
        reasons.append('REFERENCE_OVERFLOW_OR_GHOST')
    if risk['tracked_actor_count'] is None or risk['tracked_actor_count'] < snapshot['traffic_context']['active_count']:
        reasons.append('REFERENCE_ACTOR_COVERAGE_INCOMPLETE')
    if risk['min_ttc_s'] is not None and risk['min_ttc_s'] < FLOW40_PARAMETERS['minimum_ttc_s']:
        reasons.append('REFERENCE_TTC_TOO_SMALL')
    required_gap = FLOW40_PARAMETERS['standstill_gap_m']+FLOW40_PARAMETERS['following_time_s']*speed
    if risk['front_gap_m'] is not None and risk['front_gap_m'] < required_gap:
        reasons.append('REFERENCE_HEADWAY_TOO_SMALL')
    return reasons


def cruise40_reasons(snapshot, features, names, plan, speed, stage1_command, guard_active,
                     tracker_steering, raw_model_command, navigation_active):
    reasons = pace_bonus_reasons(snapshot, features, names, plan, speed, stage1_command, guard_active, tracker_steering)
    # This explicit cruise controller replaces the small bonus, including weak
    # model deceleration on an observed clear straight. Strong brakes remain.
    reasons = [reason for reason in reasons if reason not in (
        'SPEED_OUTSIDE_BONUS_RANGE', 'MODEL_TARGET_AT_OR_ABOVE_BONUS_CAP', 'MODEL_PREDICTS_SLOWDOWN')]
    if speed is None or speed < PACE_SUPPORT_PARAMETERS['minimum_speed_mps'] or speed*3.6 >= NAV40_PARAMETERS['target_kmh']:
        reasons.append('SPEED_OUTSIDE_CRUISE40_RANGE')
    if raw_model_command < -NAV40_PARAMETERS['maximum_model_brake']:
        reasons.append('STRONG_MODEL_BRAKE')
    if navigation_active:
        reasons.append('NAVIGATION_MANEUVER')
    context = snapshot['task_context']['lane_context']
    lanes = [] if context is None else [lane for lane in context['lanes'] if lane['side'] == 'CURRENT']
    distance = NAV40_PARAMETERS['straight_preview_m']
    if len(lanes) != 1 or lanes[0]['remaining_m'] < distance:
        reasons.append('CRUISE40_PREVIEW_INCOMPLETE')
    else:
        points = [point for point in lanes[0]['points'] if 0 <= point[0] <= distance]
        zero = [point for point in points if point[0] == 0]
        if not zero or not points or max(point[0] for point in points) < distance:
            reasons.append('CRUISE40_PREVIEW_INCOMPLETE')
        elif any(abs(math.atan2(math.sin(point[3]-zero[0][3]), math.cos(point[3]-zero[0][3])))
                 > PACE_SUPPORT_PARAMETERS['maximum_preview_heading_change_rad'] for point in points):
            reasons.append('BEND_WITHIN_CRUISE40_PREVIEW')
    return reasons


def lane_footprint_margin(snapshot, features, names, config):
    """Current/constant-velocity 1s rectangle clearance estimate, not a safety certificate."""
    motion = snapshot['ego_motion']
    context = snapshot['task_context']['lane_context']
    current = [] if context is None else [lane for lane in context['lanes'] if lane['side'] == 'CURRENT']
    if motion is None or len(current) != 1 or not features[names.index('lane_pose_valid')]:
        return None
    offset = features[names.index('lane_lateral_left_m')]
    heading = features[names.index('lane_heading_error_rad')]
    lane_left_speed = (-math.sin(heading)*motion['velocity_x_mps']
                       + math.cos(heading)*motion['velocity_y_mps'])
    predicted_offset = offset + PACE_SUPPORT_PARAMETERS['footprint_forecast_s']*lane_left_speed
    footprint_half_width = (config.ego_width_m*abs(math.cos(heading))
                            + config.ego_length_m*abs(math.sin(heading)))/2
    return float(current[0]['width_m']/2-max(abs(offset), abs(predicted_offset))-footprint_half_width)


def pace_bonus_reasons(snapshot, features, names, plan, speed, stage1_command, guard_active, tracker_steering=0.):
    """Conservative eligibility for an explicitly declared controller speed bonus."""
    if plan is None or speed is None:
        return ['MODEL_PLAN_UNAVAILABLE']
    parameters, task = PACE_SUPPORT_PARAMETERS, snapshot['task_context']
    reasons = []
    if guard_active:
        reasons.append('LANE_MARGIN_GUARD')
    if abs(tracker_steering) > parameters['maximum_tracker_steering']:
        reasons.append('STEERING_NOT_SETTLED')
    if speed < parameters['minimum_speed_mps'] or speed*3.6 >= parameters['bonus_target_cap_kmh']:
        reasons.append('SPEED_OUTSIDE_BONUS_RANGE')
    if stage1_command <= 0:
        reasons.append('STAGE1_NOT_ACCELERATING')
    traffic = snapshot['traffic_context']
    if traffic['active_count'] != 0 or features[names.index('log1p_overflow_count')] > 0:
        reasons.append('OBSERVED_TRAFFIC_OR_OVERFLOW')
    if (not features[names.index('lane_pose_valid')]
            or abs(features[names.index('lane_lateral_left_m')]) > parameters['maximum_lane_offset_m']
            or abs(features[names.index('lane_heading_error_rad')]) > parameters['maximum_heading_error_rad']):
        reasons.append('LANE_POSE_NOT_SETTLED')
    if (task.get('decision_point') is not None or task.get('route_passage') is not None
            or task.get('route_end_within_lookahead', True)):
        reasons.append('DECISION_PASSAGE_OR_ROUTE_END')
    context = task['lane_context']
    lanes = [] if context is None else [lane for lane in context['lanes'] if lane['side'] == 'CURRENT']
    if (len(lanes) != 1 or context.get('required_side') is not None
            or lanes[0].get('route_compatible') is not True
            or lanes[0]['remaining_m'] < parameters['straight_preview_m']):
        reasons.append('CURRENT_LANE_PREVIEW_UNAVAILABLE')
    else:
        points = [point for point in lanes[0]['points'] if 0 <= point[0] <= parameters['straight_preview_m']]
        zero = [point for point in points if point[0] == 0]
        if not zero or not points or max(point[0] for point in points) < parameters['straight_preview_m']:
            reasons.append('STRAIGHT_PREVIEW_INCOMPLETE')
        elif any(abs(math.atan2(math.sin(point[3]-zero[0][3]), math.cos(point[3]-zero[0][3])))
                 > parameters['maximum_preview_heading_change_rad'] for point in points):
            reasons.append('BEND_WITHIN_PREVIEW')
    if any(point[2] < speed for point in plan):
        reasons.append('MODEL_PREDICTS_SLOWDOWN')
    if any(point[0] <= 0 or abs(point[1]) > parameters['maximum_plan_lateral_m'] for point in plan):
        reasons.append('MODEL_PLAN_NOT_STRAIGHT_FORWARD')
    if plan[3][2]*3.6 >= parameters['bonus_target_cap_kmh']:
        reasons.append('MODEL_TARGET_AT_OR_ABOVE_BONUS_CAP')
    return reasons


def track_pdd_plan(plan, speed, config, previous_steering, dt, speed_gain, max_throttle, *, rear_axle_offset_m=0.):
    """Pure pursuit of a learned ego-frame plan with the main vehicle calibration."""
    if not math.isfinite(rear_axle_offset_m) or rear_axle_offset_m < 0:
        raise ValueError('Invalid chassis-to-rear-axle distance')
    # Public/model coordinates originate at the chassis. For rear-axle pure
    # pursuit the same target has x_rear=x_chassis+offset, y_rear=y_chassis.
    # Retain the original forward-target test and all longitudinal behavior.
    forward = [(point[0]+rear_axle_offset_m, point[1], point[2]) for point in plan if point[0] > 0]
    if not forward:
        return 0., -.5
    lookahead = max(config.lookahead_min_m, min(config.lookahead_max_m, config.lookahead_time_s * speed))
    x, y, _ = min(forward, key=lambda point: abs(math.hypot(point[0], point[1]) - lookahead))
    curvature = 2 * y / max(x*x + y*y, 1.)
    steering = math.atan(config.wheelbase_m * curvature) / math.radians(config.max_wheel_angle_deg)
    steering = max(-config.max_steering, min(config.max_steering, steering))
    limit = config.steering_rate_per_s * dt
    steering = max(previous_steering - limit, min(previous_steering + limit, steering))
    command = max(-1., min(max_throttle, speed_gain * 3.6 * (plan[3][2] - speed)))
    return float(steering), float(command)


class PDDPlanPilot(PDDSpeedPilot):
    agent_id = 'agent4_pdd_plan01_tracker'
    plan_control = True
    navigation_conditioned = False
    lane_conditioned = False
    lane_frame_conditioned = False
    curvature_conditioned = False
    rear_axle_offset_m = 0.
    pace_support_enabled = False
    compare_previous_plan_run = True
    frozen_transfer = False
    navigation40_enabled = False
    flow40_enabled = False
    flow_transfer = False
    visibility_approach_enabled = False
    expected_offline_status = 'DEV_IMPROVED'
    diagnostic_only = False
    diagnostic_plan_kind = None
    diagnostic_selected_model = None
    diagnostic_scope = None
    checkpoint_file = 'runs/agent4_pdd_plan_01/model_bundle.joblib'
    checkpoint_hash = '676b3346632b58c2095bdceb4feb0a9164c224c4f6cfd7a66226a907e2a45fa1'
    reference_reports = {case: PDDExpandedForward.reference_reports[case]
                         for case in (('Medium', 'SCXCS', 14, 1), ('Medium', 'SCXCCS', 6, 2))}

    def __init__(self):
        import joblib
        from agent4_pdd_common_v2 import FEATURE_NAMES, SCHEMA
        from src.agents.rule_based_agent import Stage1RuleBasedAgent
        from src.control_stabilizer.controller import StabilizerConfig
        from src.platform.agent import AgentDescriptor
        from train_pdd_baseline import (
            CURVE_COORDINATE_FRAME, CURVE_HELPER_HASHES, PLAN_NAV_FEATURES, PLAN_LANE_FEATURES,
            PLAN_CURVE_FEATURES, function_source_sha256,
        )

        if digest(ROOT / self.checkpoint_file) != self.checkpoint_hash:
            raise ValueError('Plan checkpoint checksum mismatch')
        bundle = joblib.load(ROOT / self.checkpoint_file)
        self.names = [*FEATURE_NAMES[:90], 'speed_change_past_0p5s_mps', 'speed_change_past_0p5s_valid']
        self.schema = ('PDD_C0_PLAN_LANE_CURVE_V1' if self.curvature_conditioned else
                       'PDD_C0_PLAN_LANE_FRAME_V1' if self.lane_frame_conditioned else
                       'PDD_C0_PLAN_LANE_V1' if self.lane_conditioned else
                       'PDD_C0_PLAN_NAV_V1' if self.navigation_conditioned else 'PDD_C0_PLAN_HISTORY_V1')
        if self.navigation_conditioned:
            self.names.extend(PLAN_NAV_FEATURES)
        if self.lane_conditioned:
            self.names.extend(PLAN_LANE_FEATURES)
        if self.curvature_conditioned:
            self.names.extend(PLAN_CURVE_FEATURES)
        self.horizons = [.25, .5, .75, 1., 1.25, 1.5]
        if (bundle['schema'] != self.schema or bundle['source_schema'] != SCHEMA
                or bundle['features'] != self.names or bundle['horizons_s'] != self.horizons
                or bundle['output_fields'] != ['x_forward_m', 'y_left_m', 'speed_mps']
                or bundle['encoder_sha256'] != digest(ROOT / 'agent4_pdd_common_v2.py')
                or bundle['history_seconds'] != .5 or bundle['offline_status'] != self.expected_offline_status
                or bundle['route_conditioned'] != self.navigation_conditioned
                or bundle.get('lane_conditioned', False) != self.lane_conditioned
                or bundle.get('lane_frame_conditioned', False) != self.lane_frame_conditioned
                or bundle.get('curvature_conditioned', False) != self.curvature_conditioned
                or bundle['selected_model'] not in ('single_plan', 'mixture_plan', 'factorized_plan')):
            raise ValueError('Incompatible learned plan checkpoint')
        self.lane_pose_contract = bundle.get('lane_pose_contract')
        if self.lane_conditioned and (not self.navigation_conditioned or self.lane_pose_contract is None
                                     or self.lane_pose_contract['helper_sha256'] != digest(ROOT / 'build_agent4_pdd_pilot.py')):
            raise ValueError('Incompatible lane observation helper contract')
        self.plan_model = bundle['plan_model']
        self.decoder_binding = None
        self.curvature_contract = bundle.get('curvature_contract')
        if self.lane_frame_conditioned:
            indices = [self.names.index(name) for name in PLAN_LANE_FEATURES]
            expected_frame = CURVE_COORDINATE_FRAME if self.curvature_conditioned else 'CURRENT_LANE_TANGENT_V1'
            if (not self.lane_conditioned or self.plan_model.get('coordinate_frame') != expected_frame
                    or self.plan_model.get('lane_indices') != indices):
                raise ValueError('Incompatible lane-frame plan decoder')
            if self.curvature_conditioned:
                contract = self.curvature_contract
                measured_helpers = {name: function_source_sha256(globals()[name]) for name in CURVE_HELPER_HASHES}
                if (contract is None or contract['public_helper_sha256'] != measured_helpers
                        or measured_helpers != CURVE_HELPER_HASHES
                        or contract['sample_distance_m'] != 2.5 or contract['positive_direction'] != 'left'
                        or contract['audit_sha256'] != 'c80d3fc783d92b07432b55763dd3bbd1e949878dbcedd1f0c740b3b5c40417e5'
                        or contract['artifact_sha256'] != '1fcb6c99b58b64456c12f5175fd95378265f82f372be6adea4173e84049601e8'
                        or contract['map_sha256'] != self.lane_pose_contract['map_sha256']
                        or self.plan_model.get('curvature_indices') != [self.names.index(name) for name in PLAN_CURVE_FEATURES]
                        or self.plan_model.get('curvature_train_range') != [contract['validity']['TRAIN']['min'],
                                                                           contract['validity']['TRAIN']['max']]
                        or bundle['trainer_sha256'] != digest(ROOT / 'train_pdd_baseline.py')):
                    raise ValueError('Incompatible curvature observation or decoder source contract')
            self.decoder_binding = lane_frame_decoder_binding(bundle)
        elif self.plan_model.get('coordinate_frame') is not None:
            raise ValueError('Unexpected coordinate transform for body-frame checkpoint')
        self.offline_status = bundle['offline_status']
        if self.diagnostic_only and (bundle['selected_model'] != self.diagnostic_selected_model
                                     or self.plan_model['kind'] != self.diagnostic_plan_kind
                                     or self.diagnostic_scope is None):
            raise ValueError('Diagnostic checkpoint does not match its frozen model contract')
        if self.plan_model['kind'] == 'single':
            self.plan_model['estimator'].regressor_.set_params(n_jobs=1)
        elif self.plan_model['kind'] == 'factorized':
            for key in ('base_estimator', 'traffic_estimator'):
                self.plan_model[key].regressor_.set_params(n_jobs=1)
        elif self.plan_model['kind'] == 'mixture':
            self.plan_model['classifier'].set_params(n_jobs=1)
            for expert in self.plan_model['experts']:
                expert.regressor_.set_params(n_jobs=1)
        else:
            raise ValueError('Unknown learned plan model kind')
        self.rule = Stage1RuleBasedAgent()
        self.config = StabilizerConfig()
        self.tracker_geometry = {'origin': 'CHASSIS', 'rear_axle_offset_m': self.rear_axle_offset_m}
        if self.rear_axle_offset_m:
            # Static actuator calibration only, never a simulator/vehicle handle.
            import inspect
            from metadrive.component.vehicle.vehicle_type import DefaultVehicle

            front, rear = DefaultVehicle.FRONT_WHEELBASE, DefaultVehicle.REAR_WHEELBASE
            if (not math.isclose(rear, self.rear_axle_offset_m, abs_tol=1e-9)
                    or not math.isclose(front+rear, self.config.wheelbase_m, abs_tol=1e-9)):
                raise ValueError('Installed DefaultVehicle differs from the rear-axle tracker calibration')
            self.tracker_geometry.update(origin='REAR_AXLE', vehicle_model='DefaultVehicle',
                front_axle_offset_m=front, vehicle_type_source_sha256=digest(inspect.getfile(DefaultVehicle)),
                target_reference='unchanged learned chassis-position targets, expressed relative to current rear axle')
        self.history = deque(maxlen=5)
        self.next_step = None
        self.previous_steering = 0.
        self.lane_guard_active = False
        self.lane_guard_release_ticks = 0
        self.descriptor = AgentDescriptor(agent_id=self.agent_id, agent_version='0.1.0',
                                          stateful_within_episode=True, method_family='HYBRID')
        # Import prediction dependencies and initialize estimators before timed act calls.
        self.predict_plan([0.] * len(self.names))
        if self.lane_frame_conditioned:
            warmup = [0.] * len(self.names)
            warmup[self.names.index('lane_pose_valid')] = 1.
            self.predict_plan(warmup)
            if self.curvature_conditioned:
                warmup[self.names.index(PLAN_CURVE_FEATURES[1])] = 1.
                for curvature in (-.01, .01):
                    warmup[self.names.index(PLAN_CURVE_FEATURES[0])] = curvature
                    self.predict_plan(warmup)
        if self.plan_model['kind'] == 'factorized':
            warmup = [0.] * len(self.names)
            warmup[self.plan_model['traffic_count_index']] = .125
            self.predict_plan(warmup)

    def reset(self, public_context, agent_seed=None):
        super().reset(public_context, agent_seed)
        self.previous_steering = 0.
        self.lane_guard_active = False
        self.lane_guard_release_ticks = 0

    def export_config(self):
        from dataclasses import asdict

        return {'checkpoint_sha256': self.checkpoint_hash, 'schema': self.schema,
                'adapter_sha256': digest(__file__), 'history_ticks': 5,
                'prediction_source_sha256': digest(ROOT / 'train_pdd_baseline.py'),
                'tracker': 'pure pursuit of model forward/left plan; model speed at 1 second',
                'tracker_geometry': self.tracker_geometry,
                'pace_support': dict(PACE_SUPPORT_PARAMETERS) if self.pace_support_enabled else None,
                'frozen_transfer_lineage': getattr(self, 'transfer_lineage', None),
                'vehicle_and_tracker': asdict(self.config),
                'longitudinal_constraint': ('minimum of explicitly assisted tracking command and Stage 1 command'
                                            if self.pace_support_enabled else
                                            'minimum of model tracking command and Stage 1 command'),
                'stage1': self.rule.export_config(), 'route_conditioned': self.navigation_conditioned,
                'plan_kind': self.plan_model['kind'], 'offline_status': self.offline_status,
                'diagnostic_only': self.diagnostic_only, 'lane_conditioned': self.lane_conditioned,
                'lane_frame_conditioned': self.lane_frame_conditioned,
                'curvature_conditioned': self.curvature_conditioned, 'curvature_contract': self.curvature_contract,
                'decoder_source_binding': self.decoder_binding,
                'lane_pose_contract': self.lane_pose_contract}

    def plan_features(self, snapshot, features, speed, past):
        values = features + list((speed-past, 1.) if speed is not None and past is not None else (0., 0.))
        distances = None
        if self.navigation_conditioned:
            from train_pdd_baseline import public_plan_navigation

            navigation, distances = public_plan_navigation(snapshot)
            values.extend(navigation)
        if self.lane_conditioned:
            from train_pdd_baseline import public_plan_lane_pose

            values.extend(public_plan_lane_pose(snapshot))
        if self.curvature_conditioned:
            from train_pdd_baseline import public_plan_curvature

            values.extend(public_plan_curvature(snapshot, values[-1]))
        return values, distances

    def plan_geometry_diagnostics(self, features):
        """Named masks: the last feature is curvature validity in the 103D schema."""
        diagnostics = {'lane_pose_valid': bool(features[self.names.index('lane_pose_valid')])
                       if self.lane_conditioned else None}
        if self.curvature_conditioned:
            curvature = float(features[self.names.index('lane_curvature_2p5m_left_inv_m')])
            valid = bool(features[self.names.index('lane_curvature_2p5m_valid')])
            lower, upper = self.plan_model['curvature_train_range']
            diagnostics.update(lane_curvature_left_inv_m=curvature, curvature_valid=valid,
                               curvature_outside_train_range=valid and not lower <= curvature <= upper)
        return diagnostics

    def predict_plan(self, features):
        import numpy as np
        from train_pdd_baseline import plan_reference, predict_c0_plan, predict_plan_candidate

        values = np.asarray([features], dtype=np.float64)
        reference = plan_reference(values, self.names, np.array(self.horizons))
        predictor = predict_plan_candidate if self.curvature_conditioned else predict_c0_plan
        return predictor(self.plan_model, values, reference)[0].tolist()

    def track_plan(self, plan, speed, previous_steering):
        return track_pdd_plan(plan, speed, self.config, previous_steering, .1,
                              self.rule.k_speed, self.rule.max_throttle,
                              rear_axle_offset_m=self.rear_axle_offset_m)

    def control_plan(self, snapshot, features, plan, speed, previous_steering, stage1_steer, stage1_command):
        steering, requested = (0., -.5) if plan is None else self.track_plan(plan, speed, previous_steering)
        if not self.pace_support_enabled:
            return steering, requested, {}
        raw_steering, raw_command = steering, requested
        margin = lane_footprint_margin(snapshot, features, self.names, self.config)
        if margin is None or margin < PACE_SUPPORT_PARAMETERS['guard_enter_margin_m']:
            self.lane_guard_active = True
            self.lane_guard_release_ticks = 0
        elif self.lane_guard_active:
            self.lane_guard_release_ticks = self.lane_guard_release_ticks + 1 if margin >= PACE_SUPPORT_PARAMETERS['guard_release_margin_m'] else 0
            if self.lane_guard_release_ticks >= PACE_SUPPORT_PARAMETERS['guard_release_ticks']:
                self.lane_guard_active = False
                self.lane_guard_release_ticks = 0
        reasons = pace_bonus_reasons(snapshot, features, self.names, plan, speed,
                                     stage1_command, self.lane_guard_active, raw_steering)
        original_target = None if plan is None else float(plan[3][2])
        target = original_target
        if not reasons:
            target = min(original_target + PACE_SUPPORT_PARAMETERS['bonus_kmh']/3.6,
                         PACE_SUPPORT_PARAMETERS['bonus_target_cap_kmh']/3.6)
            requested = max(-1., min(self.rule.max_throttle, self.rule.k_speed*3.6*(target-speed)))
        if self.lane_guard_active and plan is not None:
            limit = self.config.steering_rate_per_s*.1
            steering = max(-self.config.max_steering, min(self.config.max_steering, float(stage1_steer)))
            steering = max(previous_steering-limit, min(previous_steering+limit, steering))
        return float(steering), float(requested), {
            'raw_plan_steer': raw_steering, 'raw_plan_command': raw_command,
            'steering_guard_active': self.lane_guard_active and plan is not None,
            'steering_source': ('INVALID_PLAN_STOP' if plan is None else
                                'STAGE1_LANE_MARGIN_GUARD' if self.lane_guard_active else 'PDD_PLAN_TRACKER'),
            'lane_footprint_forecast_margin_m': margin,
            'pace_bonus_active': not reasons, 'pace_bonus_block_reasons': reasons,
            'model_target_speed_mps': original_target, 'controller_target_speed_mps': target,
        }

    def act(self, agent_input):
        from src.platform.agent import AgentDecision

        snapshot, features, speed, past, reference = self._observe(agent_input)
        features, distances = self.plan_features(snapshot, features, speed, past)
        plan = self.predict_plan(features) if speed is not None else None
        steering, requested, support = self.control_plan(snapshot, features, plan, speed, self.previous_steering,
                                                        float(reference.action_payload[0]), float(reference.action_payload[1]))
        self.previous_steering = steering
        command = min(requested, float(reference.action_payload[1]))
        return AgentDecision([steering, command], {
            'pdd_plan': plan, 'pdd_feature_sha256': hashlib.sha256(struct.pack(f'<{len(features)}f', *features)).hexdigest(),
            'navigation_source_lookahead_m': distances,
            **self.plan_geometry_diagnostics(features),
            **support,
            'model_steer': steering, 'pdd_command': requested,
            'stage1_steer': float(reference.action_payload[0]),
            'stage1_command': float(reference.action_payload[1]),
            'history_valid': speed is not None and past is not None,
            'stage1_behavior': reference.diagnostics.get('behavior_state'),
            'junction_phase': reference.diagnostics.get('junction_phase'),
            'junction_reason': reference.diagnostics.get('junction_reason'),
        })


class PDDNavPlanPilot(PDDPlanPilot):
    agent_id = 'agent4_pdd_plan_nav02_tracker'
    navigation_conditioned = True
    checkpoint_file = 'runs/agent4_pdd_plan_nav_02/model_bundle.joblib'
    checkpoint_hash = '90ccfb2eba0724cdf44a4009312c6525b51e1b84a6ddc29f704e67f49c771c27'


class PDDLanePlanPilot(PDDPlanPilot):
    agent_id = 'agent4_pdd_plan_lane04_tracker'
    navigation_conditioned = True
    lane_conditioned = True
    checkpoint_file = 'runs/agent4_pdd_plan_lane_04/model_bundle.joblib'
    checkpoint_hash = '8e278ae355b98d7df69011a4126139b24a1eeb86c1de07e3f2a96bb8b86f60cb'


class PDDLaneFramePlanDiagnostic(PDDPlanPilot):
    """Frozen failed-DEV candidate; TRAIN closed-loop diagnosis only."""

    agent_id = 'agent4_pdd_plan_lane_frame05_tracker'
    navigation_conditioned = True
    lane_conditioned = True
    lane_frame_conditioned = True
    expected_offline_status = 'NEEDS_FIX'
    diagnostic_only = True
    diagnostic_plan_kind = 'single'
    diagnostic_selected_model = 'single_plan'
    diagnostic_scope = 'PDD_LANE_FRAME_NEEDS_FIX_TRAIN_DIAGNOSTIC'
    checkpoint_file = 'runs/agent4_pdd_plan_lane_frame_05/model_bundle.joblib'
    checkpoint_hash = '190e2574cd68f68d352b1887cb0740240964bcd77db19fcfc2c614677f615512'


class PDDCurvePlanDiagnostic(PDDPlanPilot):
    """Frozen curve06 checkpoint: failed DEV gates, TRAIN diagnosis only."""

    agent_id = 'agent4_pdd_plan_curve06_tracker'
    navigation_conditioned = True
    lane_conditioned = True
    lane_frame_conditioned = True
    curvature_conditioned = True
    expected_offline_status = 'NEEDS_FIX'
    diagnostic_only = True
    diagnostic_plan_kind = 'single'
    diagnostic_selected_model = 'single_plan'
    diagnostic_scope = 'PDD_LANE_CURVE_NEEDS_FIX_TRAIN_DIAGNOSTIC'
    checkpoint_file = 'runs/agent4_pdd_plan_curve_06/model_bundle.joblib'
    checkpoint_hash = '35d4af9af0a2920e186d13fa03876df657792e902fea3e610bc2b2143512daa9'


class PDDCurveRearAxleDiagnostic(PDDCurvePlanDiagnostic):
    """Same curve06 weights and speed law; diagnose the rear-axle tracker origin."""

    agent_id = 'agent4_pdd_curve06_rearaxle_tracker'
    rear_axle_offset_m = 1.4166
    diagnostic_scope = 'PDD_CURVE06_REAR_AXLE_NEEDS_FIX_TRAIN_DIAGNOSTIC'


class PDDCurvePaceDiagnostic(PDDCurveRearAxleDiagnostic):
    """Curve06 hybrid with separately measured lane-margin guard and clear-road speed bonus."""

    agent_id = 'agent4_pdd_curve06_guarded_pace'
    pace_support_enabled = True
    diagnostic_scope = 'PDD_CURVE06_GUARDED_PACE_NEEDS_FIX_TRAIN_DIAGNOSTIC'


class PDDCurvePaceTransferDiagnostic(PDDCurvePaceDiagnostic):
    """Same frozen weights/controller on the six remaining reviewed TRAIN cases."""

    agent_id = 'agent4_pdd_curve06_pace_transfer'
    diagnostic_scope = 'PDD_CURVE06_FROZEN_PACE_TRANSFER_TRAIN_DIAGNOSTIC'
    compare_previous_plan_run = False
    frozen_transfer = True
    reference_reports = {case: reference for case, reference in PDDExpandedForward.reference_reports.items()
                         if case not in PDDPlanPilot.reference_reports}

    def __init__(self):
        self.transfer_lineage = frozen_pace_lineage()
        super().__init__()


class PDDCurveNavigation40Diagnostic(PDDCurvePaceDiagnostic):
    """New hybrid profile: execute reference maneuvers and request 40 on clear straights."""

    agent_id = 'agent4_pdd_curve06_navigation40'
    diagnostic_scope = 'PDD_CURVE06_NAVIGATION40_TRAIN_DIAGNOSTIC'
    compare_previous_plan_run = False
    navigation40_enabled = True
    reference_reports = {case: value for case, value in PDDExpandedForward.reference_reports.items()
                         if case[0] in ('Medium', 'Hard')}

    def __init__(self):
        super().__init__()
        self.reference_navigation = {}

    def _observe(self, agent_input):
        observed = super()._observe(agent_input)
        diagnostics = observed[-1].diagnostics
        lane_change = diagnostics.get('lane_change') or {}
        self.reference_navigation = {
            'behavior': diagnostics.get('behavior_state'), 'junction_phase': diagnostics.get('junction_phase'),
            'junction_reason': diagnostics.get('junction_reason'), 'lane_change_phase': lane_change.get('phase'),
            'lane_change_reason': lane_change.get('reason'), 'speed_limit_reason': diagnostics.get('speed_limit_reason'),
            'target_speed_kmh': diagnostics.get('target_speed_kmh'),
        }
        return observed

    def export_config(self):
        return {**super().export_config(), 'navigation40': dict(NAV40_PARAMETERS),
                'authority': 'Stage1 maneuver steering; empty committed maneuver may use Stage1 progress command; clear-straight cruise40; all commands capped by Stage1'}

    def control_plan(self, snapshot, features, plan, speed, previous_steering, stage1_steer, stage1_command):
        steering, requested, diagnostics = super().control_plan(
            snapshot, features, plan, speed, previous_steering, stage1_steer, stage1_command)
        hint = self.reference_navigation
        lane_phase, junction = hint.get('lane_change_phase'), hint.get('junction_phase')
        lane_active = lane_phase in ('PREPARE_CHANGE', 'CHANGE_LANE', 'ABORT_CHANGE', 'BLOCKED_CHANGE')
        junction_active = junction in ('APPROACH_JUNCTION', 'COMMIT_JUNCTION', 'TRAVERSE_JUNCTION',
                                       'EXIT_JUNCTION', 'YIELD', 'BLOCKED', 'DEGRADED_STOP')
        navigation = plan is not None and (lane_active or junction_active)
        diagnostics['base_lane_guard_armed'] = diagnostics['steering_guard_active']
        diagnostics['navigation_steering_active'] = navigation
        diagnostics['stage1_navigation_hint'] = dict(hint)
        if navigation:
            limit = self.config.steering_rate_per_s*.1
            steering = max(-self.config.max_steering, min(self.config.max_steering, stage1_steer))
            steering = max(previous_steering-limit, min(previous_steering+limit, steering))
            diagnostics.update(steering_source='STAGE1_MANEUVER', steering_guard_active=False,
                               pace_bonus_active=False)
        clear_maneuver = (navigation and (lane_active or junction in ('COMMIT_JUNCTION', 'TRAVERSE_JUNCTION', 'EXIT_JUNCTION'))
                          and snapshot['traffic_context']['active_count'] == 0
                          and features[self.names.index('log1p_overflow_count')] == 0
                          and stage1_command > 0
                          and diagnostics['raw_plan_command'] >= -NAV40_PARAMETERS['maximum_model_brake'])
        if clear_maneuver:
            requested = max(requested, stage1_command)
            if hint.get('target_speed_kmh') is not None:
                diagnostics['controller_target_speed_mps'] = min(hint['target_speed_kmh'], NAV40_PARAMETERS['target_kmh'])/3.6
        reasons = cruise40_reasons(snapshot, features, self.names, plan, speed, stage1_command,
            self.lane_guard_active, diagnostics['raw_plan_steer'], diagnostics['raw_plan_command'], navigation)
        cruise = not reasons
        if cruise:
            requested = max(-1., min(self.rule.max_throttle, self.rule.k_speed*(NAV40_PARAMETERS['target_kmh']-speed*3.6)))
            diagnostics.update(controller_target_speed_mps=NAV40_PARAMETERS['target_kmh']/3.6, pace_bonus_active=False)
        cap_command = None if speed is None else max(-1., min(self.rule.max_throttle,
            self.rule.k_speed*(NAV40_PARAMETERS['target_kmh']-speed*3.6)))
        if cap_command is not None:
            if requested > cap_command and diagnostics['controller_target_speed_mps'] is not None:
                diagnostics['controller_target_speed_mps'] = min(diagnostics['controller_target_speed_mps'], NAV40_PARAMETERS['target_kmh']/3.6)
            requested = min(requested, cap_command)
        diagnostics.update(cruise40_active=cruise, cruise40_block_reasons=reasons,
                           navigation_speed_handoff=clear_maneuver, speed40_cap_command=cap_command)
        return float(steering), float(requested), diagnostics


class PDDCurveFlow40Diagnostic(PDDCurveNavigation40Diagnostic):
    """Retain nav40 and allow bounded progress around explicitly assessed benign traffic."""

    agent_id = 'agent4_pdd_curve06_flow40'
    diagnostic_scope = 'PDD_CURVE06_FLOW40_TRAIN_DIAGNOSTIC'
    compare_previous_plan_run = True
    flow40_enabled = True

    def _observe(self, agent_input):
        observed = super()._observe(agent_input)
        reference = observed[-1].diagnostics
        # Preserve missing keys: an unavailable measurement is not a measured None.
        self.reference_navigation['traffic_risk'] = {name: reference[name] for name in FLOW_RISK_FIELDS if name in reference}
        return observed

    def export_config(self):
        return {**super().export_config(), 'flow40': dict(FLOW40_PARAMETERS),
                'progress_assessment': 'Reference positive command, known risk flags, no conflicts/overflow/ghosts, TTC/headway; not an absence-of-all-vehicles test'}

    def control_plan(self, snapshot, features, plan, speed, previous_steering, stage1_steer, stage1_command):
        steering, requested, diagnostics = super().control_plan(
            snapshot, features, plan, speed, previous_steering, stage1_steer, stage1_command)
        base_command = requested
        reasons = flow_progress_reasons(snapshot, plan, speed, diagnostics['raw_plan_command'], stage1_command,
            self.reference_navigation, self.lane_guard_active, diagnostics['navigation_steering_active'])
        target, active = None, False
        if not reasons:
            hint = self.reference_navigation
            maneuver = (hint.get('lane_change_phase') in ('PREPARE_CHANGE', 'CHANGE_LANE', 'ABORT_CHANGE')
                        or hint.get('junction_phase') in ('COMMIT_JUNCTION', 'TRAVERSE_JUNCTION', 'EXIT_JUNCTION'))
            target_kmh = hint['target_speed_kmh'] if maneuver else (
                diagnostics['model_target_speed_mps']*3.6+FLOW40_PARAMETERS['progress_increment_kmh'])
            target = min(target_kmh, hint['target_speed_kmh'], NAV40_PARAMETERS['target_kmh'])/3.6
            proposal = min(stage1_command, max(-1., min(self.rule.max_throttle, self.rule.k_speed*3.6*(target-speed))))
            if proposal > requested+1e-8:
                requested, active = proposal, True
                diagnostics.update(controller_target_speed_mps=target, pace_bonus_active=False, cruise40_active=False)
            else:
                reasons.append('BASE_CONTROL_NOT_SLOWER')
        diagnostics.update(flow_progress_active=active, flow_progress_block_reasons=reasons,
                           flow_base_command=base_command, flow_target_speed_mps=target)
        return steering, float(requested), diagnostics


class PDDCurveFlowTransferDiagnostic(PDDCurveFlow40Diagnostic):
    """Unchanged successful flow40 profile on the three remaining Extreme cases."""

    agent_id = 'agent4_pdd_curve06_flow40_extreme_transfer'
    diagnostic_scope = 'PDD_CURVE06_FROZEN_FLOW40_EXTREME_TRAIN_DIAGNOSTIC'
    flow_transfer = True
    reference_reports = {case: reference for case, reference in PDDExpandedForward.reference_reports.items()
                         if case[0] == 'Extreme'}

    def __init__(self):
        self.transfer_lineage = frozen_flow_lineage()
        super().__init__()


class PDDVisibilityApproachDiagnostic(PDDCurveFlow40Diagnostic):
    """Separate experiment: revise only the upstream junction approach cap."""

    agent_id = 'agent4_pdd_curve06_visibility_approach'
    diagnostic_scope = 'PDD_VISIBILITY_APPROACH_TRAIN_DIAGNOSTIC'
    visibility_approach_enabled = True
    compare_previous_plan_run = False
    reference_reports = PDDExpandedForward.reference_reports

    def __init__(self):
        self.parent_flow_lineage = frozen_flow_lineage()
        super().__init__()
        self.approach_active = False

    def reset(self, public_context, agent_seed=None):
        super().reset(public_context, agent_seed)
        self.approach_active = False

    def _observe(self, agent_input):
        from src.platform.agent import AgentDecision

        observed = super()._observe(agent_input)
        original = observed[-1]
        ref = original.diagnostics
        self.reference_navigation['approach_reference'] = {
            'original_command': float(original.action_payload[1]), 'speed_limits_kmh': ref.get('speed_limits_kmh'),
            'branch_locked': ref.get('junction_branch_locked'), 'front_entered': ref.get('junction_front_entered'),
            'wait_s': ref.get('junction_wait_s'), 'yield_distance_m': ref.get('junction_yield_distance_m'),
            'conflict_exit_m': ref.get('junction_conflict_exit_distance_m'),
            'vehicle_length_m': self.rule.vehicle_length_m, 'vehicle_width_m': self.rule.vehicle_width_m,
            'entry_buffer_m': self.rule.junction_entry_buffer_m, 'exit_clearance_m': self.rule.junction_exit_clearance_m,
            'response_time_s': self.rule.response_time_s, 'deceleration_mps2': self.rule.comfortable_deceleration_mps2,
            'max_throttle': self.rule.max_throttle, 'speed_gain': self.rule.k_speed,
        }
        choice = visibility_approach(observed[0], self.reference_navigation,
            self.reference_navigation['approach_reference'], self.approach_active)
        return (*observed[:-1], AgentDecision([float(original.action_payload[0]), choice['command']], dict(ref)))

    def control_plan(self, snapshot, features, plan, speed, previous_steering, stage1_steer, stage1_command):
        hint = self.reference_navigation
        choice = visibility_approach(snapshot, hint, hint['approach_reference'], self.approach_active)
        if not math.isclose(stage1_command, choice['command'], abs_tol=1e-8, rel_tol=0.):
            raise ValueError('Visibility approach reference replay mismatch')
        steering, requested, diagnostics = super().control_plan(
            snapshot, features, plan, speed, previous_steering, stage1_steer, stage1_command)
        original_final = min(requested, choice['original_command'])
        if choice['active'] and plan is not None and diagnostics['raw_plan_command'] >= -NAV40_PARAMETERS['maximum_model_brake']:
            requested = max(requested, choice['command'])
            diagnostics.update(controller_target_speed_mps=choice['speed_cap_kmh']/3.6,
                               pace_bonus_active=False, cruise40_active=False)
        self.approach_active = choice['active']
        diagnostics.update(visibility_approach=choice,
            visibility_progress_applied=choice['active'] and min(requested, stage1_command) > original_final+1e-8,
            original_stage1_command=choice['original_command'])
        return steering, requested, diagnostics

    def export_config(self):
        return {**super().export_config(), 'visibility_approach': dict(VISIBILITY_APPROACH_PARAMETERS),
            'parent_flow40_lineage': self.parent_flow_lineage,
            'longitudinal_constraint': 'min(PDD assisted command, revised upstream reference); original reference command logged separately',
            'authority': 'Only the upstream junction_phase cap may be relaxed; other reference caps, protected stop, traffic sweep and 2 km/h approach cap retained; no junction admission override'}


class PDDFactorizedPlanDiagnostic(PDDPlanPilot):
    """Exercise one frozen NEEDS_FIX candidate on development cases only."""

    agent_id = 'agent4_pdd_plan_factorized03_tracker'
    navigation_conditioned = True
    expected_offline_status = 'NEEDS_FIX'
    diagnostic_only = True
    diagnostic_plan_kind = 'factorized'
    diagnostic_selected_model = 'factorized_plan'
    diagnostic_scope = 'PDD_FACTORIZED_NEEDS_FIX_TRAIN_DIAGNOSTIC'
    checkpoint_file = 'runs/agent4_pdd_plan_factorized_03/model_bundle.joblib'
    checkpoint_hash = 'dc322ce7639f53855ad58ea7de3239d35d8fc4e16361a7e701d4134bb62d8692'


def run_pdd_speed_traffic(project, out, policy=PDDSpeedPilot, case=None):
    from src.launcher.models import AgentRegistrationV1
    from src.launcher.registry import build_canonical_agent_registry
    from src.logging.agent_experiment import write_json

    registry = build_canonical_agent_registry()
    registry.register(AgentRegistrationV1(
        agent_id=policy.agent_id, agent_version='0.1.0', method_family='HYBRID', purpose='CANDIDATE',
        implementation_ref=f'{policy.__module__}:{policy.__qualname__}',
        requires_checkpoint=True, stateful_within_episode=True), policy)
    reports = {}
    agents = ('stage1_rule_based', policy.agent_id)
    prior_path = None
    prior_agent = PDDSpeedPilot.agent_id
    if case is None and policy.traffic_only:
        prior_path = ROOT / 'agent4_integration_audit/run_20261119T031046_283869Z/report.json'
        if digest(prior_path) != 'c7b28fb89cf9a043e8d8d15a24e9685bb049318889c6ade8e033ab12f30be761':
            raise ValueError('Expected the reviewed same-case comparison for pilot 02')
        reports.update(json.loads(prior_path.read_bytes())['agents'])
        agents = (policy.agent_id,)
    tier, sequence, geometry_seed, env_seed = case or ('Medium', 'SCXCS', 14, 0)
    if policy.reference_reports is not None:
        run, expected_hash = policy.reference_reports[case]
        prior_path = ROOT / 'agent4_integration_audit' / run / f'{tier}_{sequence}_{geometry_seed}_{env_seed}' / 'report.json'
        if digest(prior_path) != expected_hash:
            raise ValueError(f'Reference checksum mismatch: {prior_path}')
        references = json.loads(prior_path.read_bytes())['agents']
        prior_agent = PDDForwardPilot.agent_id
        for agent in ('stage1_rule_based', prior_agent):
            previous = references[agent]
            if previous['episode']['case_id'] != f'sandbox/{tier}/{sequence}/{geometry_seed}/{env_seed}':
                raise ValueError('Reference case mismatch')
            if digest(Path(previous['run_dir']) / 'agent_decisions.jsonl.gz') != previous['trace_sha256']:
                raise ValueError('Reference trace checksum mismatch')
        source_dir = Path(references['stage1_rule_based']['run_dir']) / 'sources/agents'
        for source in source_dir.glob('*.py'):
            if digest(source) != digest(project / 'src/agents' / source.name):
                raise ValueError(f'Main controller changed since reference: {source.name}')
        reports.update(references)
        agents = (policy.agent_id,)
    elif policy.forward_only and case in REVIEWED_TRAFFIC_PAIRS:
        prior_path = (ROOT / 'agent4_integration_audit/run_20261119T032825_424908Z'
                      / f'{tier}_{sequence}_{geometry_seed}_{env_seed}' / 'report.json')
        if digest(prior_path) != REVIEWED_TRAFFIC_PAIRS[case]:
            raise ValueError('Expected the verified reference pair for this case')
        prior_agent = PDDTrafficOnlyPilot.agent_id
        reports.update(json.loads(prior_path.read_bytes())['agents'])
        agents = (policy.agent_id,)
    for agent in agents:
        _, run_dir, episode = launch_train_case(project, out, registry, agent, tier, sequence,
                                                geometry_seed, agent, environment_seed=env_seed)
        if episode['case_id'] != f'sandbox/{tier}/{sequence}/{geometry_seed}/{env_seed}' or episode['split'] != 'TRAIN':
            raise ValueError('Episode does not match the requested TRAIN case')
        trace = run_dir / 'agent_decisions.jsonl.gz'
        count = traffic_steps = forward_steps = inferred = changed = changed_in_traffic = steering_changes = 0
        lane_pose_steps = curvature_steps = curvature_outside_steps = 0
        guard_steps = bonus_steps = bonus_changed_steps = learned_steering_steps = learned_steering_changes = 0
        navigation_steps = navigation_speed_steps = cruise40_steps = 0
        bonus_blocks = Counter()
        cruise40_blocks = Counter()
        flow_steps, flow_blocks = 0, Counter()
        approach_steps = approach_applied = slow_steps = slow_streak = longest_slow = 0
        approach_blocks = Counter()
        plan_replay = policy() if policy.plan_control and agent == policy.agent_id else None
        last_change = None
        stopped_exit_steps = streak = longest = 0
        previous_passage = None
        with gzip.open(trace, 'rt', encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                obs = row['agent_input']
                if row['case_id'] != episode['case_id'] or row['step_index'] != count or obs['step_index'] != count:
                    raise ValueError('Traffic pilot trace is not aligned with the episode')
                traffic = obs['traffic_context']['active_count'] > 0
                traffic_steps += int(traffic)
                diag = row['decision']['diagnostics'] or {}
                motion = obs['ego_motion']
                slow = motion is not None and math.hypot(motion['velocity_x_mps'], motion['velocity_y_mps']) < .3
                slow_steps += int(slow)
                slow_streak = slow_streak+1 if slow else 0
                longest_slow = max(longest_slow, slow_streak)
                passage = diag.get('junction_passage_id')
                stopped_at_exit = (motion is not None and passage is not None
                                   and math.hypot(motion['velocity_x_mps'], motion['velocity_y_mps']) <= 0.3
                                   and diag.get('speed_limit_reason') == 'junction_phase'
                                   and diag.get('junction_exit_blocked') is True)
                stopped_exit_steps += int(stopped_at_exit)
                if not stopped_at_exit or passage != previous_passage:
                    streak = 0
                streak += int(stopped_at_exit)
                longest = max(longest, streak)
                previous_passage = passage
                if agent == policy.agent_id:
                    pdd = diag['pdd_command']
                    if policy.forward_only:
                        ahead = has_forward_corridor_actor(obs)
                        forward_steps += int(ahead)
                        if (pdd is not None) != (ahead and obs['ego_motion'] is not None):
                            raise ValueError(f'Forward-corridor inference mismatch at {count}')
                    if policy.traffic_only and not traffic and pdd is not None:
                        raise ValueError(f'Unexpected PDD control without public traffic at {count}')
                    rule = diag['stage1_command']
                    expected_steer = diag['model_steer'] if policy.plan_control else diag['stage1_steer']
                    if policy.plan_control:
                        from agent4_pdd_common_v2 import encode_public

                        features = encode_public(obs)[:90]
                        speed = features[0] if features[3] else None
                        past = plan_replay.history[0] if len(plan_replay.history) == 5 else None
                        plan_replay.history.append(speed)
                        features, _ = plan_replay.plan_features(obs, features, speed, past)
                        geometry_diagnostics = plan_replay.plan_geometry_diagnostics(features)
                        if policy.lane_conditioned:
                            observed_lane = geometry_diagnostics['lane_pose_valid']
                            if diag.get('lane_pose_valid') is not observed_lane:
                                raise ValueError(f'Lane observation validity replay mismatch at {count}')
                            lane_pose_steps += int(observed_lane)
                        if policy.curvature_conditioned:
                            if any(diag.get(key) != value for key, value in geometry_diagnostics.items()):
                                raise ValueError(f'Curvature observation replay mismatch at {count}')
                            curvature_steps += int(geometry_diagnostics['curvature_valid'])
                            curvature_outside_steps += int(geometry_diagnostics['curvature_outside_train_range'])
                        if hashlib.sha256(struct.pack(f'<{len(features)}f', *features)).hexdigest() != diag['pdd_feature_sha256']:
                            raise ValueError(f'Plan feature replay mismatch at {count}')
                        predicted = plan_replay.predict_plan(features) if speed is not None else None
                        if predicted is None:
                            if diag['pdd_plan'] is not None:
                                raise ValueError('Unexpected plan for invalid ego motion')
                        else:
                            if (diag['pdd_plan'] is None or len(diag['pdd_plan']) != 6
                                    or any(len(point) != 3 for point in diag['pdd_plan'])):
                                raise ValueError(f'Invalid logged plan shape at {count}')
                            if any(abs(x-y) > 1e-7 for a, b in zip(predicted, diag['pdd_plan']) for x, y in zip(a, b)):
                                raise ValueError(f'Plan prediction replay mismatch at {count}')
                        if policy.navigation40_enabled:
                            hint = diag['stage1_navigation_hint']
                            if (hint.get('behavior') != diag.get('stage1_behavior')
                                    or hint.get('junction_phase') != diag.get('junction_phase')
                                    or hint.get('junction_reason') != diag.get('junction_reason')):
                                raise ValueError(f'Navigation hint alignment mismatch at {count}')
                            plan_replay.reference_navigation = hint
                        steering, requested, support = plan_replay.control_plan(
                            obs, features, predicted, speed, plan_replay.previous_steering, diag['stage1_steer'], rule)
                        if policy.pace_support_enabled:
                            if any(diag.get(key) != value for key, value in support.items()):
                                raise ValueError(f'Guard/pace control replay mismatch at {count}')
                            guard_steps += int(support['steering_guard_active'])
                            bonus_steps += int(support['pace_bonus_active'])
                            bonus_changed_steps += int(support['pace_bonus_active'] and abs(requested-support['raw_plan_command']) > 1e-6)
                            if policy.navigation40_enabled:
                                navigation_steps += int(support['navigation_steering_active'])
                                navigation_speed_steps += int(support['navigation_speed_handoff'])
                                cruise40_steps += int(support['cruise40_active'])
                                cruise40_blocks.update(support['cruise40_block_reasons'])
                            if policy.flow40_enabled:
                                flow_steps += int(support['flow_progress_active'])
                                flow_blocks.update(support['flow_progress_block_reasons'])
                            if policy.visibility_approach_enabled:
                                approach_steps += int(support['visibility_approach']['active'])
                                approach_applied += int(support['visibility_progress_applied'])
                                approach_blocks.update(support['visibility_approach']['block_reasons'])
                            learned = support['steering_source'] == 'PDD_PLAN_TRACKER'
                            learned_steering_steps += int(learned)
                            learned_steering_changes += int(learned and abs(steering-diag['stage1_steer']) > 1e-6)
                            bonus_blocks.update(support['pace_bonus_block_reasons'])
                        if abs(steering-expected_steer) > 1e-7 or abs(requested-pdd) > 1e-7:
                            raise ValueError(f'Plan tracker replay mismatch at {count}')
                        plan_replay.previous_steering = steering
                    expected = [expected_steer, rule if pdd is None else min(rule, pdd)]
                    applied = row['canonical_action']
                    if len(applied) != 2 or not all(math.isclose(x, y, abs_tol=1e-6) for x, y in zip(expected, applied)):
                        raise ValueError(f'PDD applied action mismatch at {count}')
                    inferred += int(diag['pdd_plan'] is not None) if policy.plan_control else int(pdd is not None)
                    steering_changes += int(abs(expected_steer-diag['stage1_steer']) > 1e-6)
                    different = abs(applied[1] - rule) > 1e-6
                    if different:
                        last_change = count
                    changed += int(different)
                    changed_in_traffic += int(different and traffic)
                count += 1
        if plan_replay is not None:
            plan_replay.close()
        if count != int(episode['episode_steps']) or not count:
            raise ValueError(f'Incomplete driving trace for {agent}')
        reports[agent] = {'episode': episode, 'steps_with_public_traffic': traffic_steps,
                          'stopped_exit_steps': stopped_exit_steps, 'longest_stopped_exit_steps': longest,
                          'final_junction_phase': diag.get('junction_phase'),
                          'final_junction_reason': diag.get('junction_reason'),
                          'final_passage_id': previous_passage,
                          'trace_sha256': digest(trace), 'run_dir': str(run_dir)}
        with (run_dir / 'timing.csv').open(encoding='utf-8', newline='') as stream:
            timings = list(csv.DictReader(stream))
        if len(timings) != 1 or int(timings[0]['act_count']) != count:
            raise ValueError('Latency measurements do not match the episode')
        reports[agent]['p95_act_ms'] = float(timings[0]['p95_act_ms'])
        if agent == policy.agent_id:
            reports[agent].update(inference_steps=inferred, changed_longitudinal_steps=changed,
                                  changed_longitudinal_with_traffic=changed_in_traffic,
                                  last_pdd_change_step=last_change, steering_differences_from_stage1=steering_changes,
                                  stopped_s=slow_steps/10, longest_stop_s=longest_slow/10)
            if policy.plan_control:
                reports[agent]['plan_feature_prediction_tracker_replay'] = 'PASS'
                reports[agent]['plan_input_dimension'] = len(plan_replay.names)
                reports[agent]['navigation_conditioned'] = policy.navigation_conditioned
                reports[agent]['lane_conditioned'] = policy.lane_conditioned
                reports[agent]['lane_frame_conditioned'] = policy.lane_frame_conditioned
                reports[agent]['curvature_conditioned'] = policy.curvature_conditioned
                reports[agent]['tracker_geometry'] = plan_replay.tracker_geometry
                if policy.pace_support_enabled:
                    reports[agent]['controller_support'] = {
                        'parameters': dict(PACE_SUPPORT_PARAMETERS), 'trained_model_changed': False,
                        'steering_guard_steps': guard_steps, 'learned_plan_steering_steps': learned_steering_steps,
                        'learned_steering_differences_from_stage1': learned_steering_changes,
                        'speed_bonus_eligible_steps': bonus_steps, 'speed_bonus_changed_command_steps': bonus_changed_steps,
                        'bonus_block_reason_counts': dict(bonus_blocks),
                    }
                    if policy.navigation40_enabled:
                        reports[agent]['controller_support'].update(
                            navigation40_parameters=dict(NAV40_PARAMETERS), navigation_steering_steps=navigation_steps,
                            navigation_speed_handoff_steps=navigation_speed_steps, cruise40_steps=cruise40_steps,
                            cruise40_block_reason_counts=dict(cruise40_blocks))
                    if policy.flow40_enabled:
                        reports[agent]['controller_support'].update(
                            flow40_parameters=dict(FLOW40_PARAMETERS), flow_progress_steps=flow_steps,
                            flow_block_reason_counts=dict(flow_blocks))
                    if policy.visibility_approach_enabled:
                        reports[agent]['controller_support'].update(
                            visibility_approach_parameters=dict(VISIBILITY_APPROACH_PARAMETERS),
                            visibility_approach_eligible_steps=approach_steps, visibility_progress_applied_steps=approach_applied,
                            visibility_approach_block_reasons=dict(approach_blocks))
                if policy.lane_conditioned:
                    reports[agent]['lane_pose_observed_steps'] = lane_pose_steps
                    reports[agent]['lane_pose_masked_steps'] = count - lane_pose_steps
                if policy.curvature_conditioned:
                    reports[agent]['curvature_observed_steps'] = curvature_steps
                    reports[agent]['curvature_masked_steps'] = count - curvature_steps
                    reports[agent]['curvature_outside_train_range_steps'] = curvature_outside_steps
                reports[agent]['offline_status'] = plan_replay.offline_status
                reports[agent]['diagnostic_only'] = policy.diagnostic_only
                reports[agent]['plan_kind'] = plan_replay.plan_model['kind']
            if policy.forward_only:
                reports[agent]['forward_corridor_steps'] = forward_steps
        print(f"{agent}: {episode['terminal_reason']}, {episode['episode_time_s']} s, traffic steps={traffic_steps}",
              flush=True)
    hybrid = reports[policy.agent_id]
    observed = hybrid['changed_longitudinal_with_traffic'] > 0
    if policy.plan_control:
        observed = hybrid['inference_steps'] > 0 and hybrid['steering_differences_from_stage1'] > 0
        if policy.pace_support_enabled:
            observed = hybrid['controller_support']['learned_steering_differences_from_stage1'] > 0
    clean = hybrid['episode']['clean_success'].lower() == 'true'
    status = 'CLEAN_WITH_MODEL_IN_TRAFFIC' if observed and clean else 'REVIEW_REQUIRED'
    if policy.plan_control:
        status = 'CLEAN_WITH_MODEL_PLAN' if observed and clean else 'REVIEW_REQUIRED'
    report = {'scope': 'MATCHED_TRAIN_TRAFFIC_PILOT', 'status': status,
              'checkpoint_sha256': policy.checkpoint_hash, 'agents': reports}
    if policy.plan_control:
        report.update(offline_status=policy.expected_offline_status, diagnostic_only=policy.diagnostic_only)
        if policy.diagnostic_only:
            report['status'] = status = 'DIAGNOSTIC_CLEAN_ARRIVAL' if observed and clean else 'DIAGNOSTIC_REVIEW_REQUIRED'
    if prior_path is not None:
        report['reference_report'] = str(prior_path)
        if clean:
            seconds = float(hybrid['episode']['time_to_clean_success_s'])
            report['extra_seconds_to_clean_arrival'] = {
                name: round(seconds - float(reports[name]['episode']['time_to_clean_success_s']), 3)
                for name in ('stage1_rule_based', prior_agent)
                if reports[name]['episode']['clean_success'].lower() == 'true'}
    write_json(out / 'report.json', report, exclusive=True)
    print(f"{status}: PDD changed {hybrid['changed_longitudinal_steps']} actions, "
          f"{hybrid['changed_longitudinal_with_traffic']} with traffic. Read {out / 'report.json'}", flush=True)
    return report


def run_pdd_traffic_suite(project, out, policy=PDDTrafficOnlyPilot, *, cases=None):
    from src.logging.agent_experiment import write_json

    # Fix the development cases before observing the outcomes. The original
    # SCXCS/14/0 calibration pair is deliberately excluded from these results.
    reviewed_cases = cases is None
    cases = tuple(REVIEWED_TRAFFIC_PAIRS) if reviewed_cases else cases
    pairs = []
    for tier, sequence, geometry, env_seed in cases:
        case = (tier, sequence, geometry, env_seed)
        folder = out / f'{tier}_{sequence}_{geometry}_{env_seed}'
        folder.mkdir()
        case_id = f'sandbox/{tier}/{sequence}/{geometry}/{env_seed}'
        print(f'Running traffic comparison for {policy.agent_id} on {case_id}', flush=True)
        try:
            report = run_pdd_speed_traffic(project, folder, policy, case=case)
        except Exception as exc:
            traceback.print_exc()
            pairs.append({'case_id': case_id, 'error': f'{type(exc).__name__}: {exc}', 'folder': str(folder)})
            continue
        baseline = report['agents']['stage1_rule_based']['episode']
        hybrid = report['agents'][policy.agent_id]
        candidate = hybrid['episode']
        clean = candidate['clean_success'].lower() == 'true'
        baseline_clean = baseline['clean_success'].lower() == 'true'
        delta = None
        if clean and baseline_clean:
            delta = round(float(candidate['time_to_clean_success_s']) - float(baseline['time_to_clean_success_s']), 3)
        horizon_s = int(candidate['horizon_steps']) / 10.0
        pdd_cost = float(candidate['time_to_clean_success_s']) if clean else horizon_s
        baseline_cost = float(baseline['time_to_clean_success_s']) if baseline_clean else horizon_s
        pairs.append({'case_id': case_id, 'baseline_terminal': baseline['terminal_reason'],
                      'pdd_terminal': candidate['terminal_reason'], 'baseline_clean': baseline_clean,
                      'pdd_clean': clean, 'extra_seconds_if_both_clean': delta,
                      'extra_seconds_charging_horizon_on_failure': round(pdd_cost - baseline_cost, 3),
                      'pdd_changed_in_traffic': hybrid['changed_longitudinal_with_traffic'],
                      'pdd_steering_differences_from_stage1': hybrid['steering_differences_from_stage1'],
                      'pdd_vehicle_crash': candidate['crash_vehicle'].lower() == 'true',
                      'pdd_out_of_road': candidate['out_of_road'].lower() == 'true',
                      'pdd_timeout': candidate['terminal_reason'] == 'TIMEOUT',
                      'pdd_horizon_reached_event': candidate['timeout'].lower() == 'true',
                      'pdd_route_completion': float(candidate['final_route_completion']),
                      'pdd_p95_act_ms': hybrid['p95_act_ms'],
                      'pdd_mean_speed_kmh': float(candidate['mean_speed_kmh']),
                      'pdd_max_speed_kmh': float(candidate['max_speed_kmh']),
                      'pdd_episode_time_s': float(candidate['episode_time_s']),
                      'pdd_stopped_s': hybrid.get('stopped_s'), 'pdd_longest_stop_s': hybrid.get('longest_stop_s'),
                      'baseline_mean_speed_kmh': float(baseline['mean_speed_kmh']),
                      'baseline_max_speed_kmh': float(baseline['max_speed_kmh']),
                      'baseline_episode_time_s': float(baseline['episode_time_s']),
                      'report': str(folder / 'report.json')})
        if policy.plan_control and policy.pace_support_enabled:
            pairs[-1]['controller_support'] = hybrid['controller_support']
        if policy.forward_only:
            pairs[-1]['forward_corridor_steps'] = hybrid['forward_corridor_steps']
            if reviewed_cases:
                pairs[-1]['extra_seconds_vs_previous_if_both_clean'] = report.get(
                    'extra_seconds_to_clean_arrival', {}).get(PDDTrafficOnlyPilot.agent_id)
        if cases == EXTREME_EXIT_CASES:
            pairs[-1]['junction_exit'] = {
                agent: {key: values[key] for key in ('stopped_exit_steps', 'longest_stopped_exit_steps',
                                                     'final_junction_phase', 'final_junction_reason', 'final_passage_id')}
                for agent, values in report['agents'].items()}
            pairs[-1]['last_pdd_change_step'] = hybrid['last_pdd_change_step']
    complete = [p for p in pairs if 'error' not in p]
    deltas = [p['extra_seconds_if_both_clean'] for p in complete if p['extra_seconds_if_both_clean'] is not None]
    scope = 'THREE_REVIEWED_TRAIN_FORWARD_CORRIDOR_CASES' if policy.forward_only else 'THREE_NEW_PAIRED_TRAIN_TRAFFIC_CASES'
    if cases == FORWARD_TRANSFER_CASES:
        scope = 'THREE_NEW_PAIRED_TRAIN_FORWARD_TRANSFER_CASES'
    elif cases == EXTREME_EXIT_CASES:
        scope = 'PAIRED_TRAIN_EXTREME_JUNCTION_EXIT'
    if policy.reference_reports is not None:
        scope = 'EXPANDED_CHECKPOINT_EIGHT_TRAIN_CASES'
    if policy.plan_control:
        scope = 'PDD_PLAN_PAIRED_TRAIN_DIAGNOSTIC'
    summary = {'scope': scope, 'policy': policy.agent_id, 'checkpoint_sha256': policy.checkpoint_hash,
               'pairs': pairs, 'complete_pairs': len(complete), 'planned_pairs': len(cases),
               'baseline_clean': sum(p['baseline_clean'] for p in complete),
               'pdd_clean': sum(p['pdd_clean'] for p in complete),
               'pdd_clean_regressions': sum(p['baseline_clean'] and not p['pdd_clean'] for p in complete),
               'pdd_vehicle_crash_cases': sum(p['pdd_vehicle_crash'] for p in complete),
               'pdd_timeout_cases': sum(p['pdd_timeout'] for p in complete),
               'pairs_with_pdd_control_in_traffic': sum(p['pdd_changed_in_traffic'] > 0 for p in complete),
               'both_clean_pairs': len(deltas),
               'total_extra_seconds_charging_horizon_on_failure':
                   round(sum(p['extra_seconds_charging_horizon_on_failure'] for p in complete), 3) if complete else None,
               'mean_extra_seconds_if_both_clean': sum(deltas) / len(deltas) if deltas else None}
    total_time = sum(pair['pdd_episode_time_s'] for pair in complete)
    summary['pdd_time_weighted_mean_speed_kmh'] = (
        sum(pair['pdd_mean_speed_kmh']*pair['pdd_episode_time_s'] for pair in complete)/total_time if total_time else None)
    summary['pdd_max_speed_kmh'] = max((pair['pdd_max_speed_kmh'] for pair in complete), default=None)
    summary['pdd_clean_improvements_over_baseline'] = sum(pair['pdd_clean'] and not pair['baseline_clean'] for pair in complete)
    summary['tier_results'] = {}
    for tier in sorted({case[0] for case in cases}):
        group = [pair for pair in complete if pair['case_id'].split('/')[1] == tier]
        summary['tier_results'][tier] = {
            'completed_cases': len(group), 'planned_cases': sum(case[0] == tier for case in cases),
            'baseline_clean': sum(pair['baseline_clean'] for pair in group),
            'pdd_clean': sum(pair['pdd_clean'] for pair in group),
            'clean_regressions': sum(pair['baseline_clean'] and not pair['pdd_clean'] for pair in group),
            'clean_improvements': sum(pair['pdd_clean'] and not pair['baseline_clean'] for pair in group),
            'vehicle_crash_cases': sum(pair['pdd_vehicle_crash'] for pair in group),
            'out_of_road_cases': sum(pair['pdd_out_of_road'] for pair in group),
            'timeout_cases': sum(pair['pdd_timeout'] for pair in group),
        }
    if policy.reference_reports is not None:
        for pair in complete:
            report = json.loads(Path(pair['report']).read_bytes())
            pair['extra_seconds_vs_pilot02_if_both_clean'] = report.get(
                'extra_seconds_to_clean_arrival', {}).get(PDDForwardPilot.agent_id)
        summary['training_scope'] = '21 PDD TRAIN drivers, 4 DEV drivers'
        summary['capabilities_under_test'] = 'Learned longitudinal control with Stage 1 steering and constraints'
        if policy.plan_control:
            summary['capabilities_under_test'] = 'Learned plan controls steering and speed; Stage 1 caps longitudinal action'
            if policy.pace_support_enabled:
                summary['capabilities_under_test'] = 'HYBRID: PDD plan, explicit Stage1 lane-footprint steering guard, gated +2 km/h target bonus, Stage1 longitudinal cap'
                summary['controller_support_parameters'] = dict(PACE_SUPPORT_PARAMETERS)
    if policy.plan_control:
        summary.update(offline_status=policy.expected_offline_status, diagnostic_only=policy.diagnostic_only,
                       lane_conditioned=policy.lane_conditioned, lane_frame_conditioned=policy.lane_frame_conditioned,
                       curvature_conditioned=policy.curvature_conditioned,
                       tracker_origin='REAR_AXLE' if policy.rear_axle_offset_m else 'CHASSIS',
                       rear_axle_offset_m=policy.rear_axle_offset_m)
        if policy.frozen_transfer:
            summary['frozen_transfer_lineage'] = frozen_pace_lineage()
        if policy.navigation40_enabled:
            summary['navigation40_parameters'] = dict(NAV40_PARAMETERS)
            summary['capabilities_under_test'] = 'HYBRID: PDD plan with explicit maneuver authority and guarded open-road cruise40; Stage1 final longitudinal cap'
            completed_ids = {pair['case_id'] for pair in complete}
            summary['unevaluated_extreme_cases'] = [f'sandbox/{tier}/{seq}/{geometry}/{seed}'
                for tier, seq, geometry, seed in PDDExpandedForward.reference_reports
                if tier == 'Extreme' and f'sandbox/{tier}/{seq}/{geometry}/{seed}' not in completed_ids]
            summary['remaining_extreme_issue'] = ('See current Extreme outcomes; completion alone does not close deployment gates.'
                if policy.flow_transfer or policy.visibility_approach_enabled else
                'Previous frozen-transfer BLOCKED timeouts remain unresolved; raising cruise speed does not release blocked traffic.')
        if policy.flow40_enabled:
            summary['flow40_parameters'] = dict(FLOW40_PARAMETERS)
            summary['capabilities_under_test'] += '; bounded progress around reference-assessed benign traffic with TTC/headway gates'
        if policy.flow_transfer:
            summary['frozen_transfer_lineage'] = frozen_flow_lineage()
            summary['prior_flow40_development_clean'] = 5
            summary['prior_flow40_development_cases'] = 5
        if policy.diagnostic_only:
            summary['scope'] = policy.diagnostic_scope
            summary['deployment_eligible'] = False
        if policy.lane_conditioned and not policy.diagnostic_only:
            summary['scope'] = 'PDD_LANE_PLAN_PAIRED_TRAIN_DIAGNOSTIC'
            summary['deployment_eligible'] = False
        if (policy.diagnostic_only or policy.lane_conditioned) and policy.compare_previous_plan_run:
            if policy.flow_transfer:
                reference_name = 'guarded_pace_transfer'
                previous_run = 'run_20261120T155350_011690Z'
                previous_hash = 'ebe02402aa06007a8cfa13c2d956aaf7d74dad78cb202e37d2919f0772a2c072'
                expected_checkpoint = PDDCurvePaceTransferDiagnostic.checkpoint_hash
            elif policy.flow40_enabled:
                reference_name = 'navigation40'
                previous_run = 'run_20261120T165820_650923Z'
                previous_hash = '08839fc765e2723c32435f9cce4daf51de1d68652ea725ac867d9d43644326e7'
                expected_checkpoint = PDDCurveNavigation40Diagnostic.checkpoint_hash
            elif policy.pace_support_enabled:
                reference_name = 'curve06_rear'
                previous_run = 'run_20261120T150818_466529Z'
                previous_hash = '7bfcecd2a1c98101b2ffac8bf85d372b6331db6b06cd9d968c2bf493467b3200'
                expected_checkpoint = PDDCurveRearAxleDiagnostic.checkpoint_hash
            elif policy.rear_axle_offset_m:
                reference_name = 'curve06_chassis'
                previous_run = 'run_20261120T145456_767637Z'
                previous_hash = '6ee7dfe41e3fe8bb22b9c6677a3dc0482a8d380867b17bb80080d08a132288b5'
                expected_checkpoint = PDDCurvePlanDiagnostic.checkpoint_hash
            elif policy.curvature_conditioned:
                reference_name = 'frame05'
                previous_run = 'run_20261120T135335_843752Z'
                previous_hash = '6f50b3c040f6aca8cbd8dd5e105c25137f76e8a957607cd985fdf1789a7fac6b'
                expected_checkpoint = PDDLaneFramePlanDiagnostic.checkpoint_hash
            else:
                reference_name = 'lane04' if policy.lane_frame_conditioned else 'nav02'
                previous_run = ('run_20261120T131342_567885Z' if policy.lane_frame_conditioned
                                else 'run_20261120T092529_450244Z')
                previous_hash = ('1a5eaadc718dea4ce88f250181cb0a4a474e11cef8d610e26210518fa3c7a20b'
                                 if policy.lane_frame_conditioned else
                                 '9ebd626f67e155c86c0cbe07a40c8dd670bd8b4167e0b4e4ba750fbe64d2ba4d')
                expected_checkpoint = PDDLanePlanPilot.checkpoint_hash if policy.lane_frame_conditioned else PDDNavPlanPilot.checkpoint_hash
            previous_path = ROOT / 'agent4_integration_audit' / previous_run / 'report.json'
            if digest(previous_path) != previous_hash:
                raise ValueError('Previous plan runtime comparison report checksum mismatch')
            previous = json.loads(previous_path.read_bytes())
            if previous['checkpoint_sha256'] != expected_checkpoint:
                raise ValueError('Unexpected previous plan runtime comparison checkpoint')
            previous_pairs = {pair['case_id']: pair for pair in previous['pairs']}
            if policy.flow_transfer:
                # The old six-case batch contains exactly three Extreme counterparts.
                previous_pairs = {key: value for key, value in previous_pairs.items()
                                  if key.startswith('sandbox/Extreme/')}
            if set(previous_pairs) != {pair['case_id'] for pair in pairs}:
                raise ValueError('Plan comparison must use the same frozen TRAIN cases')
            for pair in complete:
                old = previous_pairs[pair['case_id']]
                pair[f'{reference_name}_terminal'] = old['pdd_terminal']
                pair[f'{reference_name}_route_completion'] = old['pdd_route_completion']
                pair[f'completion_change_vs_{reference_name}'] = pair['pdd_route_completion'] - old['pdd_route_completion']
                pair[f'{reference_name}_p95_act_ms'] = old['pdd_p95_act_ms']
                if pair['pdd_clean'] and old['pdd_clean']:
                    pair[f'time_change_vs_{reference_name}_if_both_clean_s'] = round(
                        pair['extra_seconds_if_both_clean']-old['extra_seconds_if_both_clean'], 3)
            summary[f'{reference_name}_comparison_report'] = {'path': str(previous_path), 'sha256': previous_hash,
                                                             'baseline_clean': sum(bool(pair['pdd_clean']) for pair in previous_pairs.values())}
    if policy.plan_control and policy.visibility_approach_enabled:
        historical = {}
        receipts = []
        for run, expected in (
            ('run_20261121T015940_628993Z', '2d66d0e9bbdef215f14e01f7202db6472919b7f921ca9029f533cf941ef8b022'),
            ('run_20261120T224022_728535Z', 'e84fa3315cd438639c568e8e32ec96ccc16f7f8971c558e892d4b237f2a66951'),
        ):
            source = ROOT / 'agent4_integration_audit' / run / 'report.json'
            if digest(source) != expected:
                raise ValueError('Historical flow40 report checksum mismatch')
            previous = json.loads(source.read_bytes())
            if previous['checkpoint_sha256'] != policy.checkpoint_hash:
                raise ValueError('Historical visibility comparison checkpoint mismatch')
            historical.update({pair['case_id']: (pair, previous['policy']) for pair in previous['pairs']})
            receipts.append({'path': str(source), 'sha256': expected})
        requested_ids = {pair['case_id'] for pair in pairs}
        if not requested_ids <= set(historical):
            raise ValueError('Visibility comparison requires reviewed fixed TRAIN cases')
        not_retested = sorted(set(historical)-requested_ids)
        for pair in complete:
            old, old_policy = historical[pair['case_id']]
            detail = json.loads(Path(old['report']).read_bytes())['agents'][old_policy]
            trace = Path(detail['run_dir']) / 'agent_decisions.jsonl.gz'
            if digest(trace) != detail['trace_sha256']:
                raise ValueError('Historical stop metric trace checksum mismatch')
            stops = recorded_stop_metrics(trace)
            pair.update(historical_flow40_terminal=old['pdd_terminal'], historical_flow40_stopped_s=stops['stopped_s'],
                        historical_flow40_longest_stop_s=stops['longest_stop_s'])
            pair.update(compare_stop_observations(pair, old, stops))
        summary.update(visibility_approach_parameters=dict(VISIBILITY_APPROACH_PARAMETERS),
            comparison_scope='HISTORICAL_SAME_CASE_IDS_NOT_PROVEN_IDENTICAL_TRAFFIC',
            historical_flow40_cases_not_retested=not_retested,
            historical_flow40_reports=receipts,
            capabilities_under_test='HYBRID: upstream junction_phase cap revision only; swept approach at <=2 km/h outside protected stop; no junction admission override',
            parent_flow40_lineage=frozen_flow_lineage())
    write_json(out / 'report.json', summary, exclusive=True)
    label = 'Completed' if policy.plan_control and policy.visibility_approach_enabled else 'Paired'
    print(f"{label} TRAIN cases {len(complete)}/{len(cases)}, clean arrivals Stage 1={summary['baseline_clean']} "
          f"PDD={summary['pdd_clean']}. Read {out / 'report.json'}", flush=True)
    if summary['pdd_time_weighted_mean_speed_kmh'] is not None:
        print(f"Measured speed including stops: mean={summary['pdd_time_weighted_mean_speed_kmh']:.2f} km/h, "
              f"max={summary['pdd_max_speed_kmh']:.2f} km/h", flush=True)
    if len(complete) < len(cases):
        raise SystemExit(2)


def run_junction_gap_review(project, out):
    """Replay the reference on existing inputs and inspect an unapplied timing candidate."""
    from scripts.replay_rule_based_trace import restore_input
    from src.agents.rule_based_agent import Stage1RuleBasedAgent
    from src.platform import agent_input as inputs
    from src.platform.agent import AgentPublicEpisodeContext
    from src.logging.agent_experiment import write_json

    reports = (
        ('run_20261120T205521_596750Z', '17c52c7e02b380aa28e0a1a26594c240b4c98b3c7b71dee311cfc607a04c7088'),
        ('run_20261120T224022_728535Z', 'e84fa3315cd438639c568e8e32ec96ccc16f7f8971c558e892d4b237f2a66951'),
    )
    evidence = []
    for run, expected in reports:
        path = ROOT / 'agent4_integration_audit' / run / 'report.json'
        if digest(path) != expected:
            raise ValueError('Junction review source report checksum mismatch')
        report = json.loads(path.read_bytes())
        if (report['checkpoint_sha256'] != PDDCurveFlow40Diagnostic.checkpoint_hash
                or report['complete_pairs'] != 3 or report['planned_pairs'] != 3
                or report['policy'] != PDDCurveFlowTransferDiagnostic.agent_id):
            raise ValueError('Expected completed frozen flow40 Extreme evidence')
        evidence.append((path, expected, report))
    old_by_case = {pair['case_id']: pair for pair in evidence[0][2]['pairs']}
    latest = evidence[1][2]
    if set(old_by_case) != {pair['case_id'] for pair in latest['pairs']}:
        raise ValueError('Repeat review cases differ')
    pairs = []
    for pair in latest['pairs']:
        print(f"Offline timing review {pair['case_id']} (no simulator/training)...", flush=True)
        detail_path = Path(pair['report'])
        detail = json.loads(detail_path.read_bytes())
        recorded = detail['agents'][latest['policy']]
        run_dir = Path(recorded['run_dir'])
        trace = run_dir / 'agent_decisions.jsonl.gz'
        old_detail = json.loads(Path(old_by_case[pair['case_id']]['report']).read_bytes())
        old_recorded = old_detail['agents'][latest['policy']]
        old_trace = Path(old_recorded['run_dir']) / 'agent_decisions.jsonl.gz'
        if digest(trace) != recorded['trace_sha256'] or digest(old_trace) != old_recorded['trace_sha256']:
            raise ValueError('Junction review trace checksum mismatch')
        if digest(run_dir / 'sources/agent_input.py') != digest(project / 'src/platform/agent_input.py'):
            raise ValueError('Public input source differs from the recorded trace')
        source_dir = Path(detail['agents']['stage1_rule_based']['run_dir']) / 'sources/agents'
        source_hashes = {}
        for source in source_dir.glob('*.py'):
            if digest(source) != digest(project / 'src/agents' / source.name):
                raise ValueError(f'Reference source changed: {source.name}')
            source_hashes[source.name] = digest(source)
        if 'rule_based_junction.py' not in source_hashes:
            raise ValueError('Missing frozen junction reference source')
        saved_config = json.loads((run_dir / 'agent_provenance.json').read_bytes())['configuration']['stage1']
        reference = Stage1RuleBasedAgent()
        known = reference.export_config()
        if set(saved_config) != set(known):
            raise ValueError('Recorded reference parameter schema changed')
        for name, value in saved_config.items():
            setattr(reference, name, value)
        reference.reset(AgentPublicEpisodeContext(horizon_steps=int(recorded['episode']['horizon_steps'])))
        original_gap = reference._junction._gap_check
        shadow, current = {}, {}

        def inspected_gap(corridor, tracks, speed, config):
            old = original_gap(corridor, tracks, speed, config)
            review = review_junction_gap(reference._junction, corridor, tracks, speed, config, current.get('plan'))
            if review['queue_blocked'] != old[0] or review['exit_known'] != old[2]:
                raise ValueError('Timing review changed the queue or visibility definition')
            shadow.update(review, original_conflicts=list(old[1]))
            return old  # The live reference FSM always receives its original result.

        reference._junction._gap_check = inspected_gap
        counters, block_reasons, stop_reasons = Counter(), Counter(), Counter()
        stop_segments, clear_windows = [], []
        stop_start = clear_start = None
        count, longest_stop, stop_run, clear_run = 0, 0, 0, 0
        with gzip.open(trace, 'rt', encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if row['step_index'] != count or row['case_id'] != pair['case_id']:
                    raise ValueError('Unaligned junction review trace')
                diag, obs = row['decision']['diagnostics'], row['agent_input']
                current['plan'] = diag.get('pdd_plan')
                shadow.clear()
                decision = reference.act(restore_input(obs, inputs))
                if (any(not math.isclose(float(a), float(b), abs_tol=1e-7, rel_tol=0.) for a, b in zip(
                        decision.action_payload, (diag['stage1_steer'], diag['stage1_command'])))
                        or decision.diagnostics.get('junction_phase') != diag.get('junction_phase')
                        or decision.diagnostics.get('junction_reason') != diag.get('junction_reason')):
                    raise ValueError(f'Reference replay mismatch at {pair["case_id"]} step {count}')
                motion = obs['ego_motion']
                speed = None if motion is None else math.hypot(motion['velocity_x_mps'], motion['velocity_y_mps'])
                stopped = speed is not None and speed < GAP_REVIEW_PARAMETERS['stop_threshold_mps']
                counters['stopped_steps'] += int(stopped)
                if stopped:
                    if stop_start is None:
                        stop_start = count
                    stop_run += 1
                    longest_stop = max(longest_stop, stop_run)
                    stop_reasons[decision.diagnostics.get('speed_limit_reason', 'UNKNOWN')] += 1
                elif stop_start is not None:
                    if stop_run >= 50:
                        stop_segments.append({'start_step': stop_start, 'end_step': count-1,
                                              'duration_s': stop_run/10, 'limits': dict(stop_reasons)})
                    stop_start, stop_run = None, 0
                    stop_reasons.clear()
                candidate, reasons = False, []
                if shadow:
                    counters['gap_checks'] += 1
                    original_conflict = bool(shadow['original_conflicts'])
                    nominal_relaxed = original_conflict and not shadow['nominal_conflicts']
                    bounded_relaxed = original_conflict and not shadow['bounded_conflicts']
                    counters['nominal_relaxed_steps'] += int(nominal_relaxed)
                    counters['bounded_relaxed_steps'] += int(bounded_relaxed)
                    counters['bounded_extra_conflict_steps'] += int(bool(
                        set(shadow['bounded_conflicts'])-set(shadow['original_conflicts'])))
                    if original_conflict:
                        if shadow['queue_blocked']:
                            reasons.append('EXIT_QUEUE')
                        if not shadow['exit_known']:
                            reasons.append('EXIT_NOT_VISIBLE')
                        if shadow['bounded_conflicts']:
                            reasons.append('BOUNDED_CLOCK_CONFLICT')
                        if not shadow['plan_clock_available']:
                            reasons.append('MODEL_CLOCK_UNAVAILABLE')
                        if not shadow['prediction_horizon_covered']:
                            reasons.append('TRAVERSAL_EXCEEDS_PREDICTION_HORIZON')
                        ref = decision.diagnostics
                        if (ref.get('emergency_brake') is not False or ref.get('conflict_distance_m') is not None
                                or ref.get('lidar_obstacle_gap_m') is not None):
                            reasons.append('REFERENCE_SHORT_HORIZON_HAZARD')
                        if (ref.get('traffic_overflow_count') != 0 or ref.get('actor_motion_path_overflow_count') != 0
                                or ref.get('ghost_track_count') != 0):
                            reasons.append('INCOMPLETE_OR_GHOST_TRAFFIC')
                        if (ref.get('junction_admission') or {}).get('queue_latch_s', 0) > 0:
                            reasons.append('QUEUE_RELEASE_LATCH')
                        if diag.get('raw_plan_command', -1) < -NAV40_PARAMETERS['maximum_model_brake']:
                            reasons.append('STRONG_MODEL_BRAKE')
                        block_reasons.update(reasons)
                        candidate = stopped and bounded_relaxed and not reasons
                if candidate:
                    if clear_start is None:
                        clear_start = count
                    clear_run += 1
                    counters['candidate_window_steps'] += 1
                elif clear_start is not None:
                    if clear_run/10 >= GAP_REVIEW_PARAMETERS['minimum_clear_window_s']:
                        clear_windows.append({'start_step': clear_start, 'end_step': count-1, 'duration_s': clear_run/10})
                    clear_start, clear_run = None, 0
                count += 1
                if count % 500 == 0:
                    print(f'  Reviewed {count} recorded steps', flush=True)
        if stop_start is not None and stop_run >= 50:
            stop_segments.append({'start_step': stop_start, 'end_step': count-1,
                                  'duration_s': stop_run/10, 'limits': dict(stop_reasons)})
        if clear_start is not None and clear_run/10 >= GAP_REVIEW_PARAMETERS['minimum_clear_window_s']:
            clear_windows.append({'start_step': clear_start, 'end_step': count-1, 'duration_s': clear_run/10})
        reference.close()
        if count != int(recorded['episode']['episode_steps']):
            raise ValueError('Incomplete reference replay')
        pairs.append({'case_id': pair['case_id'], 'recorded_terminal': pair['pdd_terminal'],
                      'recorded_completion': pair['pdd_route_completion'], 'steps': count,
                      'reference_replay': 'PASS', 'trace_sha256': recorded['trace_sha256'],
                      'source_sha256': source_hashes, 'counters': dict(counters),
                      'stopped_s': counters['stopped_steps']/10, 'longest_stop_s': longest_stop/10,
                      'stop_segments_at_least_5s': stop_segments, 'candidate_windows': clear_windows,
                      'clock_block_reasons': dict(block_reasons),
                      'repeat_prefix': compare_trace_prefixes(old_trace, trace)})
    summary = {'scope': 'PDD_JUNCTION_TIMING_SHADOW_REVIEW', 'status': 'REVIEW_REQUIRED',
               'actions_applied': False, 'simulator_used': False, 'training_performed': False,
               'deployment_eligible': False, 'checkpoint_sha256': latest['checkpoint_sha256'],
               'parameters': dict(GAP_REVIEW_PARAMETERS), 'pairs': pairs,
               'evidence_reports': [{'path': str(path), 'sha256': expected} for path, expected, _ in evidence],
               'reviewer_sha256': digest(__file__),
               'restore_input_source_sha256': digest(project / 'scripts/replay_rule_based_trace.py'),
               'limitations': ['The 0.8 m/s2 lower acceleration is a sensitivity assumption, not calibrated vehicle evidence.',
                   'PDD six-point speeds cover 1.5 seconds; their minimum is not a guaranteed cap throughout the junction.',
                   'Shadow windows are hypotheses on recorded observations, not saved time or safe closed-loop admissions.',
                   'Queue prediction and swept-footprint waiting-position changes are not implemented in this review.']}
    write_json(out / 'report.json', summary, exclusive=True)
    for pair in pairs:
        print(f"{pair['case_id']}: stopped={pair['stopped_s']:.1f}s, longest={pair['longest_stop_s']:.1f}s, "
              f"candidate windows={len(pair['candidate_windows'])}, repeat={pair['repeat_prefix']['status']}", flush=True)
    print(f"SHADOW_REVIEW_ONLY. No actions/checkpoint changed. Read {out / 'report.json'}", flush=True)


def advance_recorded_pose(previous_pose, motion):
    """Same measured-motion convention as LocalFrame; no estimate across missing odometry."""
    if motion is None:
        return None
    x, y, heading = previous_pose
    heading = math.atan2(math.sin(heading+motion['heading_delta_rad']), math.cos(heading+motion['heading_delta_rad']))
    dx, dy = motion['displacement_x_m'], motion['displacement_y_m']
    return (x+math.cos(heading)*dx-math.sin(heading)*dy,
            y+math.sin(heading)*dx+math.cos(heading)*dy, heading)


def recorded_footprints_intersect(first_pose, first_size, second_pose, second_size, margin=0.):
    """Four separating axes for two oriented rectangles, with explicit margin."""
    values = (*first_pose, *first_size, *second_pose, *second_size, margin)
    if not all(math.isfinite(value) for value in values) or min(*first_size, *second_size) <= 0 or margin < 0:
        raise ValueError('Invalid recorded footprint')
    dx, dy = second_pose[0]-first_pose[0], second_pose[1]-first_pose[1]
    for axis in (first_pose[2], first_pose[2]+math.pi/2, second_pose[2], second_pose[2]+math.pi/2):
        distance = abs(dx*math.cos(axis)+dy*math.sin(axis))
        extent = sum(size[0]/2*abs(math.cos(pose[2]-axis))+size[1]/2*abs(math.sin(pose[2]-axis))
                     for pose, size in ((first_pose, first_size), (second_pose, second_size)))
        if distance > extent+margin:
            return False
    return True


def recorded_hold_exposure(pose, size, frames, horizon_s, *, margin_m=.35):
    """Hindsight footprint exposure, never a counterfactual driving/safety score."""
    if not frames or not math.isfinite(horizon_s) or horizon_s <= 0:
        raise ValueError('A hold review needs recorded frames and a positive horizon')
    start, segment = frames[0]['step'], frames[0]['segment']
    requested_steps = round(horizon_s*10)
    selected = [frame for frame in frames if frame['step'] <= start+requested_steps]
    gaps, intrusion = Counter(), None
    for index, frame in enumerate(selected):
        if frame['step'] != start+index:
            gaps['non_contiguous_steps'] += 1
        if frame['pose'] is None or frame['segment'] != segment:
            gaps['odometry_gap'] += 1
            continue
        if frame['overflow']:
            gaps['traffic_capacity_overflow'] += 1
        radius = frame.get('radius_m')
        if radius is None or not math.isfinite(radius) or radius <= 0:
            gaps['unavailable_observation_radius'] += 1
        elif (math.hypot(pose[0]-frame['pose'][0], pose[1]-frame['pose'][1])
                +math.hypot(*size)/2+margin_m > radius):
            gaps['holding_footprint_outside_recorded_radius'] += 1
        for actor in frame['actors']:
            padding = margin_m+actor['speed_mps']*.05  # Half of the recorded 10 Hz interval.
            if recorded_footprints_intersect(pose, size, actor['pose'], actor['size'], padding) and intrusion is None:
                intrusion = {'step': frame['step'], 'after_s': (frame['step']-start)/10,
                             'actor_slot_at_that_frame': actor['slot']}
    complete = len(selected) == requested_steps+1 and not gaps
    status = ('OBSERVED_MARGIN_INTRUSION' if intrusion is not None else
              'NO_INTRUSION_OBSERVED_IN_WINDOW' if complete else 'INSUFFICIENT_OBSERVATIONS')
    return {'horizon_s': horizon_s, 'available_recorded_s': (selected[-1]['step']-start)/10,
            'observation_complete': complete, 'coverage_gaps': dict(gaps), 'status': status,
            'first_intrusion': intrusion, 'safety_certified': False}


WAITING_POSE_SOURCE_REPORT = 'agent4_integration_audit/run_20261121T014751_110937Z/report.json'
WAITING_POSE_SOURCE_SHA256 = 'aeb2ffe92d19daa0841d57bd403083f55fc0eeec9a658fe9cd49a05c196a155a'


def run_waiting_pose_review(project, out):
    """Inspect actual future actor observations around recorded waiting positions."""
    import numpy as np
    from src.logging.agent_experiment import write_json

    source_path = ROOT / WAITING_POSE_SOURCE_REPORT
    expected, actual = WAITING_POSE_SOURCE_SHA256, digest(source_path)
    if len(expected) != 64 or any(character not in '0123456789abcdef' for character in expected):
        raise ValueError('Invalid internal waiting-pose SHA256 pin; expected 64 lowercase hex characters')
    if actual != expected:
        raise ValueError(f'Waiting-pose report checksum mismatch: path={source_path}, expected={expected}, actual={actual}')
    source = json.loads(source_path.read_bytes())
    if (source['policy'] != PDDVisibilityApproachDiagnostic.agent_id or source['complete_pairs'] != 3
            or source['checkpoint_sha256'] != PDDVisibilityApproachDiagnostic.checkpoint_hash):
        raise ValueError('Unexpected waiting-pose review evidence')
    results = []
    for pair in source['pairs']:
        print(f"Reviewing recorded hold exposure {pair['case_id']} (no simulator)...", flush=True)
        detail = json.loads(Path(pair['report']).read_bytes())
        agent = detail['agents'][source['policy']]
        trace = Path(agent['run_dir']) / 'agent_decisions.jsonl.gz'
        if digest(trace) != agent['trace_sha256']:
            raise ValueError('Waiting-pose trace checksum mismatch')
        frames, pose, segment = [], (0., 0., 0.), 0
        first_assist = last_assist = None
        longest = streak = 0
        longest_start = 0
        with gzip.open(trace, 'rt', encoding='utf-8') as stream:
            for index, line in enumerate(stream):
                row = json.loads(line)
                if row['step_index'] != index or row['case_id'] != pair['case_id']:
                    raise ValueError('Waiting-pose trace steps/case are not aligned')
                obs, diag = row['agent_input'], row['decision']['diagnostics']
                motion = obs['ego_motion']
                if motion is None:
                    segment += 1
                    pose = None
                else:
                    pose = advance_recorded_pose((0., 0., 0.) if pose is None else pose, motion)
                traffic, actors = obs['traffic_context'], []
                if pose is not None:
                    c, s = math.cos(pose[2]), math.sin(pose[2])
                    for slot, (actor, valid) in enumerate(zip(traffic['actors'], traffic['validity_mask'])):
                        if not valid:
                            continue
                        x, y = actor['relative_position_x'], actor['relative_position_y']
                        actors.append({'slot': slot, 'pose': (pose[0]+c*x-s*y, pose[1]+s*x+c*y,
                                                            pose[2]+actor['relative_heading']),
                            'size': (actor['length'], actor['width']),
                            'speed_mps': math.hypot(motion['velocity_x_mps']+actor['relative_velocity_x'],
                                                    motion['velocity_y_mps']+actor['relative_velocity_y'])})
                speed = None if motion is None else math.hypot(motion['velocity_x_mps'], motion['velocity_y_mps'])
                streak = streak+1 if speed is not None and speed < .3 else 0
                if streak > longest:
                    longest, longest_start = streak, index-streak+1
                if diag.get('visibility_progress_applied'):
                    if first_assist is None:
                        first_assist = index
                    last_assist = index
                ref = diag['stage1_navigation_hint']['approach_reference']
                context = obs['task_context'].get('lane_context') or {}
                lanes = [lane for lane in context.get('lanes', []) if lane['side'] == 'CURRENT']
                frames.append({'step': index, 'pose': pose, 'segment': segment, 'actors': actors,
                    'overflow': traffic['overflow_count'], 'radius_m': traffic['radius_m'],
                    'lane': lanes[0] if len(lanes) == 1 else None,
                    'body_size': (ref['vehicle_length_m'], ref['vehicle_width_m']),
                    'phase': diag.get('junction_phase'), 'reason': diag.get('junction_reason')})
        if len(frames) != int(agent['episode']['episode_steps']):
            raise ValueError('Incomplete waiting-pose trace')
        anchors = {}
        for label, index in (('first_assist', first_assist), ('last_assist', last_assist), ('longest_slow_start', longest_start)):
            if index is not None:
                anchors.setdefault(index, []).append(label)
        reviewed = []
        for index, labels in sorted(anchors.items()):
            anchor = frames[index]
            if anchor['pose'] is None:
                reviewed.append({'step': index, 'labels': labels, 'status': 'MISSING_ANCHOR_ODOMETRY'})
                continue
            poses = [{'name': 'recorded_ego_pose', 'pose': anchor['pose']}]
            lane = anchor['lane']
            points = None if lane is None else np.asarray(lane['points'], dtype=float)
            valid_geometry = (points is not None and points.ndim == 2 and points.shape[1] == 4
                              and len(points) >= 2 and np.isfinite(points).all() and np.all(np.diff(points[:, 0]) > 0))
            for offset in (-2., -4., -6.):
                alternative = {'name': f'earlier_lane_center_{abs(offset):g}m', 'earlier_decision_only': True}
                if not valid_geometry or not points[0, 0] <= offset <= points[-1, 0]:
                    alternative['status'] = 'OBSERVED_REAR_GEOMETRY_UNAVAILABLE'
                else:
                    x, y = (float(np.interp(offset, points[:, 0], points[:, column])) for column in (1, 2))
                    heading = float(np.interp(offset, points[:, 0], np.unwrap(points[:, 3])))
                    px, py, ph = anchor['pose']
                    alternative['pose'] = (px+math.cos(ph)*x-math.sin(ph)*y,
                                           py+math.sin(ph)*x+math.cos(ph)*y, ph+heading)
                poses.append(alternative)
            for candidate in poses:
                if 'pose' in candidate:
                    candidate['exposure'] = [recorded_hold_exposure(candidate['pose'], anchor['body_size'],
                        frames[index:index+121], horizon) for horizon in (4., 8., 12.)]
            reviewed.append({'step': index, 'labels': labels, 'phase': anchor['phase'], 'reason': anchor['reason'],
                             'candidate_holding_poses': poses})
        results.append({'case_id': pair['case_id'], 'recorded_terminal': pair['pdd_terminal'],
            'trace_sha256': agent['trace_sha256'], 'recorded_steps': len(frames),
            'longest_slow_s': longest/10, 'anchors': reviewed})
    report = {'scope': 'RECORDED_WAITING_POSE_EXPOSURE_REVIEW', 'status': 'REVIEW_REQUIRED',
        'hindsight_only': True, 'actions_applied': False, 'simulator_used': False,
        'training_performed': False, 'deployment_eligible': False,
        'source_report': {'path': str(source_path), 'sha256': expected}, 'pairs': results,
        'reviewer_sha256': digest(__file__), 'margin_m': .35, 'horizons_s': [4., 8., 12.],
        'limitations': ['Observed actor trajectories are retrospective; they are not available to an online controller.',
            'Different ego waiting positions can change NPC reactions. This is not a counterfactual driving score.',
            'No observed intrusion is not a safety certificate; missing odometry, capacity overflow and trace truncation are reported.',
            'Holding footprint coverage is checked against recorded sensor radius; unseen actors outside it remain unknown.',
            'Earlier lane-center poses are design comparisons, not commands to reverse or proof of a feasible maneuver.',
            'No MetaDrive observation is used as a new training label.']}
    write_json(out / 'report.json', report, exclusive=True)
    for pair in results:
        print(f"{pair['case_id']}: reviewed {len(pair['anchors'])} anchors, longest recorded slow={pair['longest_slow_s']:.1f}s", flush=True)
    print(f"HINDSIGHT_REVIEW_ONLY. No control/checkpoint changed. Read {out / 'report.json'}", flush=True)


def validate_capacity_probe_prefix(public, observer):
    """The auditor may see more actors, but the driver's N8 input must stay exact."""
    if (public['capacity'] != 8 or observer['capacity'] != 32
            or public['radius_m'] != 50 or observer['radius_m'] != 50
            or len(public['actors']) != 8 or len(observer['actors']) != 32):
        raise ValueError('Unexpected capacity-probe observation shape')
    total = observer['active_count']+observer['overflow_count']
    expected_active = min(8, total)
    if (public['active_count'] != expected_active or public['overflow_count'] != max(0, total-8)
            or public['validity_mask'] != [True]*expected_active+[False]*(8-expected_active)
            or observer['validity_mask'] != [True]*observer['active_count']+[False]*(32-observer['active_count'])
            or public['actors'] != observer['actors'][:8]):
        raise ValueError('Auditor N32 observation does not preserve the exact driver N8 prefix')


def capacity_probe_builder(original_builder, observer_extractor, emit):
    """Passive SDK instrumentation: build the driver's input exactly once and return it unchanged."""
    def build(*args, **kwargs):
        public_input = original_builder(*args, **kwargs)
        observer = observer_extractor(kwargs['ego_vehicle'], kwargs['traffic_vehicles'], capacity=32, radius_m=50.)
        validate_capacity_probe_prefix(public_input.traffic_context.to_dict(), observer.to_dict())
        emit(public_input, observer)
        return public_input
    return build


def hidden_actors_near_recorded_route(snapshot, observer):
    """Geometric relevance only; absence here does not exclude future crossing intent."""
    waypoints = [(0., 0.)]+[(point['relative_x'], point['relative_y']) for point, valid in zip(
        snapshot['task_context']['waypoints'], snapshot['task_context']['validity_mask']) if valid]
    segments = [(start, end) for start, end in zip(waypoints, waypoints[1:]) if math.dist(start, end) > 1e-5]
    if not segments:
        return None
    selected = []
    for slot in range(snapshot['traffic_context']['active_count'], observer['active_count']):
        actor = observer['actors'][slot]
        x, y = actor['relative_position_x'], actor['relative_position_y']
        best = None
        for start, end in segments:
            dx, dy = end[0]-start[0], end[1]-start[1]
            fraction = max(0., min(1., ((x-start[0])*dx+(y-start[1])*dy)/(dx*dx+dy*dy)))
            distance = math.hypot(x-start[0]-fraction*dx, y-start[1]-fraction*dy)
            heading = math.atan2(dy, dx)
            if best is None or distance < best[0]:
                best = distance, heading
        relative = actor['relative_heading']-best[1]
        half_width = (actor['width']*abs(math.cos(relative))+actor['length']*abs(math.sin(relative)))/2
        if best[0] <= 1.852/2+half_width+.35:
            selected.append(slot)
    return selected


def run_traffic_capacity_probe(project, out):
    """One TRAIN case: observe N32 privately in the audit, keep actual driver N8."""
    import time
    from src.launcher import executor as launcher_executor
    from src.launcher.models import AgentRegistrationV1
    from src.launcher.registry import build_canonical_agent_registry
    from src.logging.agent_experiment import write_json
    from src.platform.agent_input import extract_traffic_context

    lineage = frozen_flow_lineage()
    policy = PDDCurveFlow40Diagnostic
    registry = build_canonical_agent_registry()
    registry.register(AgentRegistrationV1(agent_id=policy.agent_id, agent_version='0.1.0',
        method_family='HYBRID', purpose='CANDIDATE', implementation_ref=f'{policy.__module__}:{policy.__qualname__}',
        requires_checkpoint=True, stateful_within_episode=True), policy)
    case = ('Extreme', 'CrXROSTR', 3, 1)
    case_id = 'sandbox/Extreme/CrXROSTR/3/1'
    trace = out / 'observer_capacity32.jsonl.gz'
    counts, totals = Counter(), Counter()
    original = launcher_executor.build_agent_input
    started = time.perf_counter()
    with gzip.open(trace, 'xt', encoding='utf-8') as stream:
        def emit(public_input, observation):
            snapshot, wider = public_input.to_dict(), observation.to_dict()
            step = snapshot['step_index']
            if step != counts['frames']:
                raise ValueError('Capacity-probe steps are not consecutive')
            public = snapshot['traffic_context']
            total = wider['active_count']+wider['overflow_count']
            totals[total] += 1
            counts['frames'] += 1
            counts['n8_overflow_frames'] += int(public['overflow_count'] > 0)
            counts['n32_overflow_frames'] += int(wider['overflow_count'] > 0)
            counts['extra_observed_actor_frames'] += wider['active_count']-public['active_count']
            hidden = hidden_actors_near_recorded_route(snapshot, wider)
            counts['route_unavailable_frames'] += int(hidden is None)
            counts['frames_with_hidden_actor_near_route'] += int(bool(hidden))
            # No unrecorded wider road-path alternatives are represented as known-empty.
            wider.pop('motion_paths', None)
            wider.pop('motion_paths_overflow_count', None)
            public_hash = hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(',', ':'),
                                                    allow_nan=False).encode('utf-8')).hexdigest()
            stream.write(json.dumps({'case_id': case_id, 'step_index': step, 'public_input_sha256': public_hash,
                'observer_traffic_context': wider, 'observer_motion_paths_status': 'NOT_RECORDED',
                'hidden_actor_slots_near_current_route': hidden}, allow_nan=False)+'\n')
        try:
            launcher_executor.build_agent_input = capacity_probe_builder(original, extract_traffic_context, emit)
            print('N32 OBSERVER ONLY: flow40 still receives the original N8 public input; no visibility-approach assist.', flush=True)
            _, run_dir, episode = launch_train_case(project, out, registry, policy.agent_id,
                case[0], case[1], case[2], policy.agent_id, environment_seed=case[3])
        finally:
            launcher_executor.build_agent_input = original
    if counts['frames'] != int(episode['episode_steps']):
        raise ValueError('Capacity-probe recording does not cover the driver episode')
    # Verify against the launcher's own trace, not just our callback assertion.
    driver_trace = run_dir / 'agent_decisions.jsonl.gz'
    verified = 0
    with gzip.open(trace, 'rt', encoding='utf-8') as observer_stream, gzip.open(driver_trace, 'rt', encoding='utf-8') as driver_stream:
        for line in observer_stream:
            observer_row = json.loads(line)
            driver_line = driver_stream.readline()
            if not driver_line:
                raise ValueError('Driver trace shorter than capacity observer trace')
            driver_row = json.loads(driver_line)
            actual_hash = hashlib.sha256(json.dumps(driver_row['agent_input'], sort_keys=True, separators=(',', ':'),
                                                    allow_nan=False).encode('utf-8')).hexdigest()
            if (observer_row['step_index'] != driver_row['step_index']
                    or driver_row['case_id'] != case_id or actual_hash != observer_row['public_input_sha256']):
                raise ValueError('Capacity observer diverged from the actual driver public input')
            verified += 1
        if driver_stream.readline():
            raise ValueError('Driver trace longer than capacity observer trace')
    summary = {'scope': 'N8_DRIVER_N32_PASSIVE_TRAFFIC_CAPACITY_PROBE', 'status': 'RECORDED',
        'case_id': case_id, 'split': 'TRAIN', 'canonical_benchmark': False,
        'driver_policy': policy.agent_id, 'driver_input_capacity': 8, 'observer_capacity': 32, 'radius_m': 50.,
        'additional_information_passed_to_driver': False, 'driver_public_input_replay': 'PASS',
        'verified_input_steps': verified, 'counts': dict(counts), 'population_histogram': dict(totals),
        'maximum_observed_population': max(totals, default=0), 'driver_episode': episode,
        'driver_trace_sha256': digest(driver_trace), 'observer_trace_sha256': digest(trace),
        'observer_trace': str(trace), 'checkpoint_sha256': policy.checkpoint_hash, 'parent_flow40_lineage': lineage,
        'audit_source_sha256': digest(__file__),
        'sdk_input_source_sha256': digest(project / 'src/platform/agent_input.py'),
        'wall_time_s': time.perf_counter()-started, 'training_performed': False, 'deployment_eligible': False,
        'limitations': ['This is a fresh simulation for measurement, not an improved N32 driving result.',
            'Only the auditor receives N32. The driver and its model retain the original N8 contract and weights.',
            'The observer adds extraction/logging overhead outside agent act latency; this is not a latency benchmark.',
            'Hidden-near-route is current geometric relevance, not proven collision risk or future maneuver intention.',
            'Wider motion-path alternatives are not recorded. Capacity above 32 may still overflow.',
            'No MetaDrive observer record is used as a new training label.']}
    write_json(out / 'report.json', summary, exclusive=True)
    print(f"N8 overflow={counts['n8_overflow_frames']}/{counts['frames']} frames; "
          f"N32 overflow={counts['n32_overflow_frames']}/{counts['frames']}; "
          f"hidden-near-route frames={counts['frames_with_hidden_actor_near_route']}", flush=True)
    print(f"CAPACITY_MEASUREMENT_ONLY. Driver input unchanged. Read {out / 'report.json'}", flush=True)


def pdd_junction_source_context(info):
    """Source metadata for review, not an assertion of cross-simulator input parity."""
    from agent4_pdd_common_v2 import number

    inside, upcoming, signal = info['is_in_junction'], info['next_road_junction'], info['traffic_light_state']
    if type(inside) is not bool or type(upcoming) is not bool or not isinstance(signal, str) or not signal:
        raise ValueError('Invalid PDD junction context flags')
    raw_distance = info['distance_to_junction']
    distance = None if raw_distance is None else number(raw_distance)
    return {'is_in_junction': inside, 'next_road_junction': upcoming,
            'distance_to_junction_m': distance if distance is not None and distance >= 0 else None,
            'traffic_light_state': signal}


def pdd_junction_window(states, step):
    """Separate junction candidates; C0 scope and the frozen encoder remain unchanged."""
    from agent4_pdd_common_v2 import float32, project_carla

    current = states[step]
    context = current['context']
    nearby = context['next_road_junction'] and context['distance_to_junction_m'] is not None and context['distance_to_junction_m'] <= 50.
    if not (context['is_in_junction'] or nearby):
        return None, ['OUTSIDE_CURRENT_JUNCTION_CONTEXT']
    future = [states.get(step+5*offset) for offset in range(1, 7)]
    if any(state is None for state in future):
        return None, ['INCOMPLETE_HUMAN_HORIZON']
    window = [current, *future]
    invalid = set().union(*(state['geometry'] for state in window))
    invalid.update(set().union(*(state['scope'] for state in window))-{'JUNCTION_OR_SIGNAL_REVIEW_REQUIRED'})
    if invalid:
        return None, sorted(invalid)
    trajectory = []
    for state in future:
        x, y = project_carla(state['origin'][0]-current['origin'][0],
                             state['origin'][1]-current['origin'][1], current['yaw'])
        trajectory.append(float32([x, y, state['speed']]))
    speed, next_speed = current['speed'], trajectory[3][2]  # Human speed at +1 second.
    transition = ('WAIT' if speed < .3 and next_speed < .3 else
                  'LAUNCH' if speed < .3 and next_speed >= .3 else
                  'SLOWDOWN' if next_speed < speed-.5 else
                  'SPEEDUP' if next_speed > speed+.5 else 'STEADY')
    flags = ['HUMAN_BEHAVIOR_UNREVIEWED', 'JUNCTION_CONTEXT_PARITY_UNVERIFIED', 'PRIORITY_UNOBSERVED_BY_MAIN_INPUT']
    if any(state['context']['traffic_light_state'] != 'None' for state in window):
        flags.append('SIGNAL_UNOBSERVED_BY_MAIN_INPUT')
    return {'features': list(current['features']), 'human_trajectory': trajectory, 'target_mask': [True]*6,
        'label_source': 'HUMAN_RECORDED', 'human_speed_transition_1s': transition,
        'source_context_for_parity_review': dict(context),
        'future_context_for_label_review': [dict(state['context']) for state in future],
        'observability_review': flags, 'behavior_quality': 'UNREVIEWED', 'training_eligible': False}, []


def export_pdd_junction_candidates(root, out):
    """Build real-PDD junction review data, without changing frozen C0 sources or training."""
    from build_agent4_pdd_pilot import (
        REVISION, PAST_EXCEPTIONS, EgoLaneTypeConflict, read_source, control_index, pedals,
        paired_state, single_box, partition, HORIZONS,
    )
    from agent4_pdd_common_v2 import FEATURE_NAMES

    for name, expected in (
        ('build_agent4_pdd_pilot.py', '22b3de8370681320ecb721fd6f699930eec74797665a0a7a72cfe2bb75ee9936'),
        ('agent4_pdd_common_v2.py', 'cda499ff1c47fd23fef5a6b90eb26d4ebf29aab7badd80007137f4523b5eacc9'),
    ):
        if digest(ROOT / name) != expected:
            raise ValueError(f'Junction export requires the reviewed frozen helper: {name}')
    root = root.resolve()
    if not root.is_dir() or root == out or root in out.parents:
        raise ValueError('Expected an existing PDD root and an output directory outside raw PDD')
    revision = (root / '.cache/huggingface/download/README.md.metadata').read_text(encoding='utf-8').splitlines()[0]
    if revision != REVISION:
        raise ValueError('PDD revision differs from the surveyed source')
    output_path = out / 'junction_candidates.jsonl'
    counts = {key: Counter(candidates=0, inside_junction=0, joined_frames=0, valid_frames=0) for key in ('TRAIN', 'DEV')}
    transitions, flags, rejected = {}, {}, {}
    receipts, unexpected = [], []
    with output_path.open('x', encoding='utf-8') as stream:
        for driver in range(1, 26):  # Do not open held-out drivers 26..30.
            driver_name, split = f'driver_{driver:02d}', partition(driver)
            scenes = sorted((root / driver_name / 'data').iterdir())
            scenes = [scene for scene in scenes if scene.is_dir()]
            if len(scenes) != 21:
                raise ValueError(f'Expected 21 surveyed scenes: {driver_name}')
            for scene in scenes:
                inventory, notes, errors, exclusions, states = {}, Counter(), Counter(), Counter(), {}
                telemetry_raw = read_source(scene / 'metric/metric_info.json', inventory, root)
                telemetry = {}
                for key, value in telemetry_raw.items():
                    if not key.isdecimal() or str(int(key)) != key or int(key) % 5:
                        raise ValueError(f'Invalid telemetry step: {driver_name}/{scene.name}/{key}')
                    telemetry[int(key)] = value
                controls = control_index(read_source(scene / 'metric/metrics.json', inventory, root)['records'])
                boxes = {}
                for path in (scene / 'boxes').glob('*.json.gz'):
                    key = path.name[:-8]
                    if not key.isdecimal() or str(int(key)) != key:
                        raise ValueError(f'Invalid box frame: {path}')
                    boxes[int(key)*5] = path
                alignment = (len(telemetry.keys()-boxes.keys()), len(boxes.keys()-telemetry.keys()), min(telemetry, default=None))
                expected = PAST_EXCEPTIONS.get((driver, scene.name), (0, 0, 0))
                if alignment != expected or telemetry.keys()-controls.keys():
                    unexpected.append(f'{driver_name}/{scene.name}/alignment')
                joined = sorted(telemetry.keys() & controls.keys() & boxes.keys())
                for step in joined:
                    try:
                        control = controls[step]
                        pedals(control['steer'], control['throttle'], control['brake'])
                        modes = [control[name] for name in ('reverse', 'hand_brake', 'manual_gear_shift')]
                        if any(type(mode) is not bool for mode in modes):
                            raise ValueError('Expected boolean control modes')
                        if any(modes):
                            exclusions['UNSUPPORTED_CONTROL_MODE'] += 1
                            continue
                        frame = read_source(boxes[step], inventory, root, notes=notes)
                        geometry = set()
                        features, origin, yaw, speed, scope, _ = paired_state(telemetry[step], frame, geometry_flags=geometry)
                        context = pdd_junction_source_context(single_box(frame, 'ego_info'))
                        states[step] = {'features': features, 'origin': origin, 'yaw': yaw, 'speed': speed,
                                        'scope': scope, 'geometry': geometry, 'context': context}
                    except EgoLaneTypeConflict:
                        exclusions['EGO_LANE_TYPE_CONFLICT'] += 1
                    except (ValueError, KeyError, TypeError, OSError, EOFError, IndexError, OverflowError) as error:
                        errors[str(error)] += 1
                for step in sorted(states):
                    row, reasons = pdd_junction_window(states, step)
                    if row is None:
                        exclusions.update(reasons)
                        continue
                    identity = {'driver': driver_name, 'scenario': scene.name, 'partition': split,
                                'telemetry_step': step, 'frame_id': step//5}
                    stream.write(json.dumps({**identity, **row}, separators=(',', ':'), allow_nan=False)+'\n')
                    counts[split]['candidates'] += 1
                    counts[split]['inside_junction'] += int(row['source_context_for_parity_review']['is_in_junction'])
                    transitions.setdefault(split, Counter())[row['human_speed_transition_1s']] += 1
                    flags.setdefault(split, Counter()).update(row['observability_review'])
                counts[split]['joined_frames'] += len(joined)
                counts[split]['valid_frames'] += len(states)
                rejected.setdefault(split, Counter()).update(exclusions)
                if errors:
                    unexpected.append(f'{driver_name}/{scene.name}/source_errors')
                receipts.append({'episode': f'{driver_name}/{scene.name}', 'partition': split,
                    'alignment': alignment, 'exclusions': dict(exclusions), 'source_errors': dict(errors),
                    'annotation_notes': dict(notes), 'source_sha256': inventory})
            print(f"Exported junction candidates from {driver_name} ({split})", flush=True)
    report = {'schema': 'PDD_JUNCTION_REVIEW_V1', 'status': 'SOURCE_REVIEW_REQUIRED' if unexpected else 'REVIEW_REQUIRED',
        'training_ready': False, 'training_performed': False, 'deployment_eligible': False,
        'feature_schema': 'PDD_COMMON_V2', 'feature_names': list(FEATURE_NAMES), 'feature_dimension': len(FEATURE_NAMES),
        'horizons_s': list(HORIZONS), 'timebase_source': 'Existing reviewed PDD step/20 seconds; no interpolation',
        'revision': revision, 'label_source': 'HUMAN_RECORDED', 'drivers_read': list(range(1, 26)),
        'test_drivers_read': [], 'source': 'PDD_ONLY', 'simulator_used': False,
        'sampling': 'Current ego inside junction OR next road is junction at an observed distance of 0..50 m',
        'feature_policy': 'Only current 96D features. Current/future source junction/signal context is review metadata, not input.',
        'split_counts': {key: dict(value) for key, value in counts.items()},
        'human_transitions_1s': {key: dict(value) for key, value in transitions.items()},
        'observability_flags': {key: dict(value) for key, value in flags.items()},
        'excluded_windows_or_frames': {key: dict(value) for key, value in rejected.items()},
        'unexpected_alignment_or_source_errors': unexpected, 'episodes': receipts,
        'candidates_sha256': digest(output_path), 'exporter_sha256': digest(__file__),
        'frozen_builder_sha256': digest(ROOT / 'build_agent4_pdd_pilot.py'),
        'frozen_encoder_sha256': digest(ROOT / 'agent4_pdd_common_v2.py'),
        'limitations': ['These are junction review candidates, not automatically approved training rows.',
            'Human WAIT/LAUNCH/slowdown/speedup describe recorded motion; they are not legal right-of-way labels.',
            'A None traffic-light annotation does not prove an unsignalized junction or permission to go.',
            'Junction context needs a separate PDD/MetaDrive parity contract before training a junction policy.',
            'C0 filtering and frozen checkpoints are unchanged. No MetaDrive data supplies human labels.']}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f"PDD_JUNCTION_REVIEW_ONLY: {report['split_counts']}; no training. Read {out / 'report.json'}", flush=True)


JUNCTION_FORECAST_SOURCE_REPORT = 'agent4_integration_audit/run_20261121T080501_740128Z/report.json'
JUNCTION_FORECAST_SOURCE_SHA256 = '4fe52026780cf441e499a29bb4dd025e9e312336c746afa296d464d43e49feb3'
JUNCTION_MOTION_CLASSES = ('WAIT', 'LAUNCH', 'SLOWDOWN', 'SPEEDUP', 'STEADY')


def junction_forecast_features(features):
    """Current 90 state features plus two normalized navigation bearings; no junction metadata."""
    from agent4_pdd_common_v2 import number
    from train_pdd_baseline import encode_plan_navigation

    values = list(map(number, features))
    if len(values) != 96:
        raise ValueError('Junction forecast requires the current 96D source features')
    hints = []
    for offset in (90, 93):
        x, y, valid = values[offset:offset+3]
        if valid not in (0., 1.) or (valid == 0. and (x != 0. or y != 0.)):
            raise ValueError('Invalid observed navigation mask/padding')
        hints.append((x, y) if valid else None)
    return values[:90]+encode_plan_navigation(hints)


def junction_forecast_example(row):
    """Validate research targets; never promote UNREVIEWED motion into policy labels."""
    from agent4_pdd_common_v2 import number

    if (row['label_source'] != 'HUMAN_RECORDED' or row['target_mask'] != [True]*6
            or row['behavior_quality'] != 'UNREVIEWED' or row['training_eligible'] is not False
            or row['human_speed_transition_1s'] not in JUNCTION_MOTION_CLASSES):
        raise ValueError('Unexpected junction demonstration review/target contract')
    features = junction_forecast_features(row['features'])
    targets = row['human_trajectory']
    if len(targets) != 6 or any(len(point) != 3 for point in targets):
        raise ValueError('Expected six recorded human trajectory points')
    for point in targets:
        tuple(map(number, point))
        if point[2] < 0:
            raise ValueError('Human speed cannot be negative')
    contexts = [row['source_context_for_parity_review'], *row['future_context_for_label_review']]
    if len(contexts) != 7 or any(not isinstance(context['traffic_light_state'], str) for context in contexts):
        raise ValueError('Missing source signal coverage over the human horizon')
    signal = any(context['traffic_light_state'] != 'None' for context in contexts)
    flags = set(row['observability_review'])
    required = {'HUMAN_BEHAVIOR_UNREVIEWED', 'JUNCTION_CONTEXT_PARITY_UNVERIFIED', 'PRIORITY_UNOBSERVED_BY_MAIN_INPUT'}
    if not required <= flags or signal != ('SIGNAL_UNOBSERVED_BY_MAIN_INPUT' in flags):
        raise ValueError('Junction observability flags differ from the source context')
    if signal:
        return None  # Filter uses future only to screen labels, never as a model feature.
    return features, [float(point[2]) for point in targets], row['human_speed_transition_1s']


def junction_forecast_scores(truth, prediction, labels):
    import numpy as np

    truth, prediction = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    if (truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != 6
            or not np.isfinite(truth).all() or not np.isfinite(prediction).all() or len(labels) != len(truth)):
        raise ValueError('Invalid junction forecast evaluation arrays')
    scores = {}
    labels = np.asarray(labels)
    for name in ('all', *JUNCTION_MOTION_CLASSES):
        mask = np.ones(len(truth), dtype=bool) if name == 'all' else labels == name
        count = int(mask.sum())
        error = np.abs(prediction[mask]-truth[mask])
        scores[name] = {'n': count, 'mae_mps': float(error.mean()) if count else None,
                       'one_second_mae_mps': float(error[:, 3].mean()) if count else None}
    return scores


JUNCTION_NAV96_REPORT = 'agent4_integration_audit/run_20261121T091802_200039Z/report.json'
JUNCTION_NAV96_SHA256 = 'd955a1c243c1c157dc8372bd0fbff3737f4531a9785e85d4ccd222840defc345'
JUNCTION_HISTORY_NAMES = ['speed_delta_past_0p5s_mps', 'speed_delta_past_0p5s_valid',
    'speed_delta_past_1s_mps', 'speed_delta_past_1s_valid', 'stopped_span_past_2s_s', 'history_coverage_past_2s_s']


def causal_speed_history(speeds, tick, *, stride, seconds_per_tick):
    """Causal physical-time history for PDD (5/.05) and public control (1/.1)."""
    from agent4_pdd_common_v2 import number

    if (type(stride) is not int or stride <= 0 or type(tick) is not int
            or not math.isfinite(seconds_per_tick) or seconds_per_tick <= 0):
        raise ValueError('Invalid speed-history cadence')
    period = stride*seconds_per_tick
    if any(not math.isclose(seconds/period, round(seconds/period), abs_tol=1e-8) for seconds in (.5, 1., 2.)):
        raise ValueError('History cadence cannot represent the required physical horizons')
    if tick not in speeds:
        raise ValueError('Current motion sample is missing')
    observed = {}
    for offset in range(round(2./period)+1):
        key = tick-offset*stride
        if key not in speeds:
            break  # Never bridge a missing motion frame or read any future sample.
        value = number(speeds[key])
        if value < 0:
            raise ValueError('History speed cannot be negative')
        observed[offset] = value
    current, result = observed[0], []
    for seconds in (.5, 1.):
        offset = round(seconds/period)
        result.extend((current-observed[offset], 1.) if offset in observed else (0., 0.))
    stopped_span = 0.
    for offset, speed in observed.items():
        if speed >= .3:
            break
        stopped_span = offset*period
    return result+[stopped_span, (len(observed)-1)*period]


def add_junction_motion_history(groups, source, pdd):
    from collections import defaultdict
    from build_agent4_pdd_pilot import read_source, xyz
    import numpy as np

    pdd = pdd.resolve()
    receipts = {entry['episode']: entry for entry in source['episodes']}
    verified, summary = {}, {}
    for split, group in groups.items():
        episodes = defaultdict(list)
        history = np.zeros((len(group['identities']), 6), dtype=np.float32)
        for index, (driver, scene, step) in enumerate(group['identities']):
            episodes[(driver, scene)].append((index, step))
        for (driver, scene), positions in episodes.items():
            relative = f'{driver}/data/{scene}/metric/metric_info.json'
            inventory = {}
            raw = read_source(pdd / relative, inventory, pdd)
            receipt = receipts[f'{driver}/{scene}']
            if receipt['partition'] != split or inventory[relative] != receipt['source_sha256'][relative]:
                raise ValueError(f'PDD history source checksum/split mismatch: {relative}')
            speeds = {int(step): math.hypot(*xyz(row['velocity'])[:2]) for step, row in raw.items()}
            for index, step in positions:
                if not math.isclose(speeds[step], float(group['x'][index, 0]), abs_tol=1e-5, rel_tol=1e-6):
                    raise ValueError(f'Current motion differs from the candidate: {relative}/{step}')
                history[index] = causal_speed_history(speeds, step, stride=5, seconds_per_tick=.05)
            verified[relative] = inventory[relative]
        group['x'] = np.column_stack((group['x'], history))
        summary[split] = {'rows': len(history), 'valid_past_0p5s': int(history[:, 1].sum()),
                          'valid_past_1s': int(history[:, 3].sum()), 'full_2s_coverage': int((history[:, 5] == 2.).sum())}
    return {'features': list(JUNCTION_HISTORY_NAMES), 'summary': summary, 'verified_source_files': verified,
            'causal_only': True, 'missing_frame_policy': 'Stop history at the first missing motion frame',
            'pdd_cadence': {'stride': 5, 'seconds_per_tick': .05},
            'public_cadence': {'stride': 1, 'seconds_per_tick': .1}}


def junction_forecast_unmet(scores, baseline, classifier_f1, baseline_f1):
    """Keep the original WAIT and launch criteria unchanged across candidates."""
    unmet = []
    for name in ('all', 'LAUNCH', 'SLOWDOWN'):
        if scores[name]['one_second_mae_mps'] >= baseline[name]['one_second_mae_mps']:
            unmet.append(f'{name}/one_second_mae_not_better_than_persistence')
    if scores['WAIT']['one_second_mae_mps'] > baseline['WAIT']['one_second_mae_mps']+.1:
        unmet.append('WAIT/one_second_mae_regression_over_0p1_mps')
    if classifier_f1 <= baseline_f1:
        unmet.append('transition/macro_f1_not_better_than_persistence')
    return unmet


JUNCTION_HISTORY_REPORT = 'agent4_integration_audit/run_20261121T095240_560584Z/report.json'
JUNCTION_HISTORY_SHA256 = '579a91eb0acf557d6dccbf0de54e686b2561f95a80f6de138fa99c690d7e5c97'
JUNCTION_TRAFFIC_NAMES = ['lead_rear_distance_m', 'lead_valid', 'lead_forward_speed_mps',
    'lead_closing_speed_mps', 'lead_ttc_capped_20s', 'lead_ttc_valid', 'front_queue_count_over_8',
    'crossing_time_4s', 'crossing_x_m', 'crossing_valid', 'held_origin_clearance_4s_m', 'held_origin_clearance_valid']


def junction_traffic_features(features):
    """Shared current-state geometry, not right-of-way or an admission decision."""
    from agent4_pdd_common_v2 import number

    f = list(map(number, features))
    if len(f) != 96 or f[3] not in (0., 1.) or f[5] not in (0., 1.):
        raise ValueError('Unexpected shared input for junction traffic geometry')
    actors = []
    for offset in range(16, 88, 9):
        x, y, rvx, rvy, sine, cosine, length, width, valid = f[offset:offset+9]
        if valid not in (0., 1.) or (valid == 0. and any(f[offset:offset+8])):
            raise ValueError('Invalid shared actor mask/padding')
        if valid:
            if min(length, width) <= 0 or not math.isclose(math.hypot(sine, cosine), 1., abs_tol=1e-4):
                raise ValueError('Invalid shared actor size/orientation')
            actors.append((x, y, rvx, rvy, sine, cosine, length, width))
    if not math.isclose(f[88], len(actors)/8, abs_tol=1e-6):
        raise ValueError('Shared actor count does not match its validity flags')
    if f[3] == 0.:
        return [0.]*len(JUNCTION_TRAFFIC_NAMES)
    lane_half = f[4]/2 if f[5] == 1. and 0 < f[4] <= 6 else None
    leader = crossing = None
    queue = 0
    clearance = None
    for x, y, rvx, rvy, sine, cosine, length, width in actors:
        vx, vy = f[1]+rvx, f[2]+rvy
        half_x = (length*abs(cosine)+width*abs(sine))/2
        half_y = (width*abs(cosine)+length*abs(sine))/2
        if lane_half is not None and x > 0 and cosine > math.cos(.5) and abs(y) <= lane_half+half_y:
            rear = x-half_x  # Distance from ego origin, not a certified bumper gap.
            closing = max(0., -rvx)
            candidate = (rear, vx, closing)
            if leader is None or rear < leader[0]:
                leader = candidate
            queue += int(math.hypot(vx, vy) < 1.)
        if abs(vy) > .05:
            time_to_cross = -y/vy
            cross_x = x+vx*time_to_cross
            if 0 <= time_to_cross <= 4. and 0 <= cross_x <= 50.:
                candidate = (time_to_cross, cross_x)
                if crossing is None or candidate < crossing:
                    crossing = candidate
        # Constant observed actor velocity, fixed current heading, stationary
        # ego origin. A geometric covariate only, not a safe holding certificate.
        for index in range(9):
            px, py = x+vx*index*.5, y+vy*index*.5
            along = max(0., abs(px*cosine+py*sine)-length/2)
            across = max(0., abs(-px*sine+py*cosine)-width/2)
            distance = math.hypot(along, across)
            clearance = distance if clearance is None else min(clearance, distance)
    if leader is None:
        result = [0.]*6
    else:
        rear, vx, closing = leader
        ttc_valid = closing > .05
        result = [rear, 1., vx, closing, min(20., max(0., rear)/closing) if ttc_valid else 0., float(ttc_valid)]
    result += [queue/8]
    result += [*crossing, 1.] if crossing is not None else [0., 0., 0.]
    result += [clearance, 1.] if clearance is not None else [0., 0.]
    return result


def predict_junction_speed_residuals(model, features):
    """A learned near-stop expert may predict either waiting or launching."""
    import numpy as np

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError('Expected a batch of junction forecast features')
    if isinstance(model, dict):
        if model.get('kind') != 'CURRENT_SPEED_EXPERTS_V1':
            raise ValueError('Unknown junction forecast expert contract')
        stopped = features[:, 0] < model['threshold_mps']
        result = np.empty((len(features), 6), dtype=float)
        for mask, name in ((stopped, 'near_stop'), (~stopped, 'moving')):
            if mask.any():
                result[mask] = model[name].predict(features[mask])
    else:
        result = np.asarray(model.predict(features), dtype=float)
    if result.shape != (len(features), 6) or not np.isfinite(result).all():
        raise ValueError('Invalid learned junction speed residual output')
    return result


JUNCTION_TRAFFIC_REPORT = 'agent4_integration_audit/run_20261121T101635_351003Z/report.json'
JUNCTION_TRAFFIC_SHA256 = '0d856d55a2796365eb757612d45a0aea9c5c6cabe10a7f3654a1ddb069bbe5d1'


def junction_wait_cohorts(row):
    """Retrospective error groups only; future labels never become model inputs."""
    if row['motion_label'] != 'WAIT':
        return []
    speeds = row['true_speed_mps']
    if len(speeds) != 6:
        raise ValueError('Expected all six human speed targets for the error review')
    timing = ('intermediate_motion' if max(speeds[:4]) >= .3 else
              'launch_after_1s' if max(speeds[4:]) >= .3 else 'still_through_1p5s')
    return ['WAIT/all', f'WAIT/scenario/{row["scenario"]}', f'WAIT/driver/{row["driver"]}',
        'WAIT/inside_junction' if row['source_context']['is_in_junction'] else 'WAIT/approach',
        'WAIT/overflow' if row['overflow_count'] > 0 else 'WAIT/no_overflow',
        'WAIT/first_2s' if row['telemetry_step'] <= 40 else 'WAIT/after_first_2s', f'WAIT/{timing}']


def exact_junction_stop_groups(features, labels):
    """Exact encoded inputs, without rounding or claiming nearby points are identical."""
    import numpy as np

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] < 1 or not np.isfinite(features).all() or len(features) != len(labels):
        raise ValueError('Invalid feature/label arrays for exact-input review')
    groups = {}
    for index in np.flatnonzero(features[:, 0] < .3):
        groups.setdefault(tuple(features[index].tolist()), []).append(int(index))
    conflicting = {key for key, indices in groups.items() if {'WAIT', 'LAUNCH'} <= {labels[index] for index in indices}}
    return groups, conflicting


def review_junction_forecast_errors(out, groups):
    """Replay the rejected checkpoint and locate its errors; no estimator is fitted."""
    from collections import defaultdict
    import joblib
    import numpy as np
    from train_pdd_baseline import function_source_sha256
    from sklearn.metrics import confusion_matrix

    source_path = ROOT / JUNCTION_TRAFFIC_REPORT
    if digest(source_path) != JUNCTION_TRAFFIC_SHA256:
        raise ValueError('Expected the reviewed failed TRAFFIC114 forecast report')
    source = json.loads(source_path.read_bytes())
    model_path = source_path.parent / 'model_bundle.joblib'
    if digest(model_path) != source['model_bundle_sha256']:
        raise ValueError('TRAFFIC114 model checksum mismatch')
    bindings = {'feature_function_sha256': junction_forecast_features, 'history_function_sha256': causal_speed_history,
                'traffic_function_sha256': junction_traffic_features, 'speed_decoder_sha256': predict_junction_speed_residuals}
    if any(source[name] != function_source_sha256(function) for name, function in bindings.items()):
        raise ValueError('TRAFFIC114 forecast codec changed since training')
    if (source['schema'] != 'PDD_JUNCTION_MOTION_TRAFFIC_V3' or source['offline_status'] != 'FORECAST_NEEDS_FIX'
            or source['selected_model'] != 'near_stop_expert' or source['deployment_eligible'] is not False):
        raise ValueError('Unexpected checkpoint for the rejected forecast error review')
    bundle = joblib.load(model_path)
    selected = bundle['speed_regressor']
    models = {'selected_near_stop_expert': selected, 'full_absolute_error': selected['moving']}
    scores, predictions = {}, {}
    print('Replaying rejected TRAFFIC114 predictions and reviewing WAIT errors; no training/simulator...', flush=True)
    for split, group in groups.items():
        if len(group['x']) != source['counts'][split]['retained'] or group['x'].shape[1] != len(source['features']):
            raise ValueError('Forecast error-review input population/shape differs')
        predictions[split] = {name: np.maximum(0., group['x'][:, :1]+predict_junction_speed_residuals(model, group['x']))
                              for name, model in models.items()}
        scores[split] = {name: junction_forecast_scores(group['y'], values, group['labels'])
                         for name, values in predictions[split].items()}
    for name, recorded_name in (('selected_near_stop_expert', source['selected_model']), ('full_absolute_error', 'absolute_error_boosting')):
        for cohort, values in scores['DEV'][name].items():
            for metric in ('mae_mps', 'one_second_mae_mps'):
                if not math.isclose(values[metric], source['dev_speed'][recorded_name][cohort][metric], abs_tol=1e-8, rel_tol=0.):
                    raise ValueError('TRAFFIC114 error review did not reproduce recorded speed metrics')
    train, dev = groups['TRAIN'], groups['DEV']
    probabilities = bundle['transition_classifier'].predict_proba(dev['x'])
    predicted_classes = bundle['transition_classifier'].predict(dev['x'])
    if confusion_matrix(dev['labels'], predicted_classes, labels=list(JUNCTION_MOTION_CLASSES)).tolist() != source['dev_transition']['confusion_matrix']:
        raise ValueError('TRAFFIC114 classifier predictions did not reproduce recorded counts')
    rows, cohorts = [], defaultdict(list)
    archive = out / 'forecast_dev_predictions.jsonl.gz'
    with gzip.open(archive, 'xt', encoding='utf-8') as stream:
        for index, (driver, scene, step) in enumerate(dev['identities']):
            row = {'driver': driver, 'scenario': scene, 'telemetry_step': step, 'partition': 'DEV',
                'motion_label': dev['labels'][index], 'true_speed_mps': dev['y'][index].tolist(),
                'predicted_speed_mps': {name: values[index].tolist() for name, values in predictions['DEV'].items()},
                'predicted_motion_class': str(predicted_classes[index]),
                'motion_probabilities': {str(name): float(value) for name, value in zip(bundle['transition_classifier'].classes_, probabilities[index])},
                'source_context': dev['contexts'][index], 'overflow_count': round(math.expm1(float(dev['x'][index, 89]))),
                'features': dev['x'][index].tolist(),
                'absolute_error_1s_mps': float(abs(predictions['DEV']['selected_near_stop_expert'][index, 3]-dev['y'][index, 3]))}
            for cohort in junction_wait_cohorts(row):
                cohorts[cohort].append(index)
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    cohort_summary = {}
    for name, indices in cohorts.items():
        errors = [rows[index]['absolute_error_1s_mps'] for index in indices]
        cohort_summary[name] = {'n': len(indices), 'selected_1s_mae_mps': float(np.mean(errors)),
            'selected_1s_error_p95_mps': float(np.percentile(errors, 95)),
            'full_head_1s_mae_mps': float(np.abs(predictions['DEV']['full_absolute_error'][indices, 3]-dev['y'][indices, 3]).mean()),
            'predicted_LAUNCH_class_count': sum(rows[index]['predicted_motion_class'] == 'LAUNCH' for index in indices)}
    exact_groups, conflicting = exact_junction_stop_groups(train['x'], train['labels'])
    near_stop_indices = np.flatnonzero(train['x'][:, 0] < .3)
    scale = np.maximum(np.std(train['x'].astype(float), axis=0), 1.)  # TRAIN statistics only, no fitted estimator.
    top_indices = sorted(cohorts['WAIT/all'], key=lambda index: rows[index]['absolute_error_1s_mps'], reverse=True)[:20]
    examples = []
    for index in top_indices:
        row = rows[index]
        distances = np.sqrt(np.mean(((train['x'][near_stop_indices]-dev['x'][index])/scale)**2, axis=1))
        neighbors = []
        for position in np.argsort(distances, kind='mergesort')[:5]:
            neighbor = int(near_stop_indices[position])
            neighbors.append({'identity': train['identities'][neighbor], 'motion_label': train['labels'][neighbor],
                'true_speed_1s_mps': float(train['y'][neighbor, 3]),
                'predicted_speed_1s_mps': float(predictions['TRAIN']['selected_near_stop_expert'][neighbor, 3]),
                'scaled_rms_feature_distance': float(distances[position])})
        key = tuple(dev['x'][index].tolist())
        exact = exact_groups.get(key, [])
        examples.append({**row, 'nearest_train_near_stop': neighbors,
            'exact_train_motion_labels': dict(Counter(train['labels'][neighbor] for neighbor in exact)),
            'exact_conflicting_train_input': key in conflicting})
    report = {'scope': 'PDD_JUNCTION_FORECAST_ERROR_REVIEW', 'status': 'REPLAYED_ERROR_REVIEW',
        'source_report': str(source_path), 'source_report_sha256': JUNCTION_TRAFFIC_SHA256,
        'model_bundle_sha256': source['model_bundle_sha256'], 'prediction_replay': 'PASS',
        'training_performed': False, 'model_weights_changed': False, 'simulator_used': False, 'test_used': False,
        'deployment_eligible': False, 'actions_applied': False, 'features': source['features'],
        'speed_scores': scores, 'wait_cohorts': cohort_summary, 'largest_20_wait_errors': examples,
        'exact_train_near_stop_groups': len(exact_groups), 'exact_train_groups_with_WAIT_LAUNCH_conflict': len(conflicting),
        'dev_WAIT_exactly_matching_conflicting_train_input': sum(tuple(dev['x'][index].tolist()) in conflicting for index in cohorts['WAIT/all']),
        'neighbor_distance': 'RMS feature difference divided by max(TRAIN std, 1); top five TRAIN current-speed<0.3 examples; no rounding',
        'dev_predictions_sha256': digest(archive), 'reviewer_sha256': digest(__file__),
        'limitations': ['TRAIN errors are in-sample and are not generalization evidence.',
            'Future target-based cohorts are hindsight descriptions, not current model inputs or new training labels.',
            'Near neighbors are not identical observations. Their differing labels do not prove irreducible ambiguity.',
            'Only exact encoded-input conflicts establish conflicting demonstrations for that representation, not legal priority or a universal safety conclusion.',
            'No row is removed, target changed, threshold relaxed, model refitted or driving permission granted.']}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f"TRAFFIC114 prediction replay PASS; reviewed {len(cohorts['WAIT/all'])} DEV WAIT rows; "
          f"exact conflicting TRAIN groups={len(conflicting)}. Read {out / 'report.json'}", flush=True)


JUNCTION_ERROR_REVIEW_REPORT = 'agent4_integration_audit/run_20261121T103212_662871Z/report.json'
JUNCTION_ERROR_REVIEW_SHA256 = '8d8c65995130c67d2ed83e3adea7e0f97b2fbdcbe6b9838e1704719f73dbc509'
JUNCTION_STOP_QUANTILES = (.10, .25)


def junction_quantile_metrics(truth, prediction, quantile, near_stop):
    """Report quantile loss/coverage without reordering predictions or changing labels."""
    import numpy as np

    truth, prediction = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    near_stop = np.asarray(near_stop, dtype=bool)
    if (not 0 < quantile < 1 or truth.ndim != 2 or truth.shape[1] != 6 or truth.shape != prediction.shape
            or near_stop.shape != (len(truth),) or not len(truth)
            or not np.isfinite(truth).all() or not np.isfinite(prediction).all()):
        raise ValueError('Invalid quantile forecast evaluation data')
    difference = truth-prediction
    loss = np.maximum(quantile*difference, (quantile-1)*difference)
    return {'quantile_near_stop': quantile, 'near_stop_count': int(near_stop.sum()),
        'pinball_1s_mps_near_stop': float(loss[near_stop, 3].mean()) if near_stop.any() else None,
        'coverage_1s_near_stop': float((truth[near_stop, 3] <= prediction[near_stop, 3]).mean()) if near_stop.any() else None,
        'pinball_by_horizon_near_stop': loss[near_stop].mean(axis=0).tolist() if near_stop.any() else None,
        'coverage_by_horizon_near_stop': (truth[near_stop] <= prediction[near_stop]).mean(axis=0).tolist() if near_stop.any() else None}


def junction_stop_launch_tradeoff(labels, prediction):
    import numpy as np

    labels, prediction = np.asarray(labels), np.asarray(prediction, dtype=float)
    if prediction.shape != (len(labels), 6) or not np.isfinite(prediction).all():
        raise ValueError('Invalid prediction arrays for the stop/launch review')
    wait, launch = labels == 'WAIT', labels == 'LAUNCH'
    return {'WAIT_count': int(wait.sum()), 'LAUNCH_count': int(launch.sum()),
        'WAIT_predicted_speed_ge_0p3_count': int((prediction[wait, 3] >= .3).sum()),
        'LAUNCH_predicted_speed_lt_0p3_count': int((prediction[launch, 3] < .3).sum())}


def train_junction_quantile_forecast(out, groups):
    """Fit two predeclared lower-quantile heads; preserve moving head/classifier and original gates."""
    import joblib
    import numpy as np
    import sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.metrics import confusion_matrix, f1_score
    from train_pdd_baseline import function_source_sha256

    review_path = ROOT / JUNCTION_ERROR_REVIEW_REPORT
    if digest(review_path) != JUNCTION_ERROR_REVIEW_SHA256:
        raise ValueError('Expected the completed forecast-error review')
    review = json.loads(review_path.read_bytes())
    if (review['prediction_replay'] != 'PASS' or review['training_performed'] is not False
            or digest(review_path.parent / 'forecast_dev_predictions.jsonl.gz') != review['dev_predictions_sha256']):
        raise ValueError('Forecast-error review evidence is incomplete or changed')
    parent_path = ROOT / JUNCTION_TRAFFIC_REPORT
    if digest(parent_path) != JUNCTION_TRAFFIC_SHA256:
        raise ValueError('Expected the frozen TRAFFIC114 parent')
    parent_report = json.loads(parent_path.read_bytes())
    parent_model_path = parent_path.parent / 'model_bundle.joblib'
    if (digest(parent_model_path) != parent_report['model_bundle_sha256']
            or review['model_bundle_sha256'] != parent_report['model_bundle_sha256']):
        raise ValueError('Quantile forecast parent checkpoint mismatch')
    bindings = {'feature_function_sha256': junction_forecast_features, 'history_function_sha256': causal_speed_history,
                'traffic_function_sha256': junction_traffic_features, 'speed_decoder_sha256': predict_junction_speed_residuals}
    if any(parent_report[name] != function_source_sha256(function) for name, function in bindings.items()):
        raise ValueError('Quantile forecast parent observation/decoder contract changed')
    parent = joblib.load(parent_model_path)
    train, dev = groups['TRAIN'], groups['DEV']
    for split, group in groups.items():
        if len(group['x']) != parent_report['counts'][split]['retained'] or group['x'].shape[1] != len(parent['features']):
            raise ValueError('Quantile forecast source population/input shape changed')
    moving = parent['speed_regressor']['moving']
    models = {'median_expert_reference': parent['speed_regressor'], 'full_median_reference': moving}
    scores = {}
    for name, recorded in (('median_expert_reference', parent_report['selected_model']), ('full_median_reference', 'absolute_error_boosting')):
        predicted = np.maximum(0., dev['x'][:, :1]+predict_junction_speed_residuals(models[name], dev['x']))
        scores[name] = junction_forecast_scores(dev['y'], predicted, dev['labels'])
        for group, values in scores[name].items():
            for metric in ('mae_mps', 'one_second_mae_mps'):
                if not math.isclose(values[metric], parent_report['dev_speed'][recorded][group][metric], abs_tol=1e-8, rel_tol=0.):
                    raise ValueError('TRAFFIC114 parent speed metrics did not replay')
    classifier = parent['transition_classifier']
    predicted_classes = classifier.predict(dev['x'])
    if confusion_matrix(dev['labels'], predicted_classes, labels=list(JUNCTION_MOTION_CLASSES)).tolist() != parent_report['dev_transition']['confusion_matrix']:
        raise ValueError('Frozen parent classifier did not replay')
    classifier_f1 = float(f1_score(dev['labels'], predicted_classes, labels=list(JUNCTION_MOTION_CLASSES), average='macro', zero_division=0))
    baseline_labels = np.where(dev['x'][:, 0] < .3, 'WAIT', 'STEADY')
    baseline_f1 = float(f1_score(dev['labels'], baseline_labels, labels=list(JUNCTION_MOTION_CLASSES), average='macro', zero_division=0))
    persistence = np.repeat(dev['x'][:, :1], 6, axis=1)
    baseline = junction_forecast_scores(dev['y'], persistence, dev['labels'])
    stopped, stopped_dev = train['x'][:, 0] < .3, dev['x'][:, 0] < .3
    if int(stopped.sum()) < 40:
        raise ValueError('Insufficient TRAIN examples for the quantile heads')
    settings, diagnostics, predictions, candidates = {}, {}, {}, []
    for quantile in JUNCTION_STOP_QUANTILES:
        name = f'near_stop_q{round(quantile*100):02d}'
        parameters = {'loss': 'quantile', 'quantile': quantile, 'max_iter': 160, 'max_leaf_nodes': 15,
            'min_samples_leaf': 20, 'learning_rate': .08, 'l2_regularization': 1., 'early_stopping': False, 'random_state': 42}
        model = MultiOutputRegressor(HistGradientBoostingRegressor(**parameters), n_jobs=1)
        print(f'Fitting six q={quantile:.2f} heads on {int(stopped.sum())} TRAIN near-stop rows; moving/classifier frozen...', flush=True)
        model.fit(train['x'][stopped], train['y'][stopped]-train['x'][stopped, :1])
        models[name] = {'kind': 'CURRENT_SPEED_EXPERTS_V1', 'threshold_mps': .3, 'near_stop': model,
                        'moving': moving, 'near_stop_quantile': quantile}
        predictions[name] = np.maximum(0., dev['x'][:, :1]+predict_junction_speed_residuals(models[name], dev['x']))
        scores[name] = junction_forecast_scores(dev['y'], predictions[name], dev['labels'])
        diagnostics[name] = {'quantile_forecast': junction_quantile_metrics(dev['y'], predictions[name], quantile, stopped_dev),
            'persistence_at_same_quantile': junction_quantile_metrics(dev['y'], persistence, quantile, stopped_dev),
            'stop_launch_tradeoff': junction_stop_launch_tradeoff(dev['labels'], predictions[name])}
        settings[name] = parameters
        candidates.append(name)
    unmet = {name: junction_forecast_unmet(values, baseline, classifier_f1, baseline_f1) for name, values in scores.items()}
    eligible = [name for name in candidates if not unmet[name]]
    selected = min(eligible or candidates, key=lambda name: scores[name]['all']['one_second_mae_mps'])
    status = 'FORECAST_DEV_IMPROVED' if not unmet[selected] else 'FORECAST_NEEDS_FIX'
    crossing = predictions[candidates[0]][stopped_dev] > predictions[candidates[1]][stopped_dev]
    contract = {name: value for name, value in parent.items() if name not in ('speed_regressor', 'transition_classifier')}
    contract.update(schema='PDD_JUNCTION_MOTION_QUANTILE_V4', selected_model=selected, offline_status=status,
        forecast_interpretation='Lower conditional speed quantile near stop; inherited median head while moving',
        near_stop_quantile=models[selected]['near_stop_quantile'], quantiles_fitted=list(JUNCTION_STOP_QUANTILES),
        parent_traffic114_report_sha256=JUNCTION_TRAFFIC_SHA256, parent_model_sha256=parent_report['model_bundle_sha256'],
        error_review_sha256=JUNCTION_ERROR_REVIEW_SHA256, quantile_calibration_approved=False,
        policy_training_approved=False, deployment_eligible=False)
    model_path = out / 'model_bundle.joblib'
    joblib.dump({**contract, 'speed_regressor': models[selected], 'transition_classifier': classifier,
                 'near_stop_quantile_models': {name: models[name] for name in candidates}}, model_path)
    report = {**contract, 'training_performed': True, 'simulator_used': False, 'test_used': False,
        'parent_prediction_replay': 'PASS', 'counts': parent_report['counts'],
        'dev_speed': {'persistence': baseline, **scores}, 'quantile_diagnostics': diagnostics,
        'dev_transition': parent_report['dev_transition'], 'unmet_by_candidate': unmet,
        'unmet_forecast_criteria': unmet[selected], 'quantile_crossing_count_1s_near_stop': int(crossing[:, 3].sum()),
        'quantile_crossing_fraction_near_stop': float(crossing.mean()) if crossing.size else None,
        'fitting': {'near_stop_heads_only': True, 'parameters': settings, 'moving_head_refitted': False,
                    'classifier_refitted': False, 'near_stop_training_rows': int(stopped.sum()),
                    'near_stop_dev_rows': int(stopped_dev.sum()), 'sklearn_version': sklearn.__version__},
        'model_bundle_sha256': digest(model_path), 'implementation_sha256': digest(__file__),
        'limitations': ['Lower quantiles intentionally trade lower premature-motion predictions for more conservative launch predictions; both counts are reported.',
            'None of the original WAIT/LAUNCH/speed/classification gates is relaxed. No rows are dropped or targets changed.',
            'Independent quantile heads may cross; predictions are not reordered to hide this. Coverage is measured, not assumed calibrated.',
            'The inherited motion classifier is unchanged; a lower speed quantile is not a STOP/GO or legal priority decision.',
            'This remains an offline PDD human-motion experiment. A passing forecast status is not evidence of shorter waiting in MetaDrive or deployment approval.']}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for name in candidates:
        tradeoff = diagnostics[name]['stop_launch_tradeoff']
        print(f"{name}: all/+1s={scores[name]['all']['one_second_mae_mps']:.4f}, "
              f"WAIT={scores[name]['WAIT']['one_second_mae_mps']:.4f}, LAUNCH={scores[name]['LAUNCH']['one_second_mae_mps']:.4f} m/s; "
              f"WAIT>=0.3={tradeoff['WAIT_predicted_speed_ge_0p3_count']}/{tradeoff['WAIT_count']}, "
              f"LAUNCH<0.3={tradeoff['LAUNCH_predicted_speed_lt_0p3_count']}/{tradeoff['LAUNCH_count']}; unmet={unmet[name]}", flush=True)
    print(f"{status}, selected={selected}. Read {out / 'report.json'}", flush=True)


JUNCTION_QUANTILE_REPORT = 'agent4_integration_audit/run_20261122T021029_502784Z/report.json'
JUNCTION_QUANTILE_SHA256 = 'f645d3db5d5427ddd87d9032f8c33e013cfe8c3c75207b5b7f7c0474b4ace73d'
JUNCTION_TREE_GATE_REPORT = 'agent4_integration_audit/run_20261122T025257_618710Z/report.json'
JUNCTION_TREE_GATE_SHA256 = '566c995338694e7b4d9d8640d0a85350f3f8b0a2dbb48daec7568c2a6f53ed37'
JUNCTION_MLP_GATE_REPORT = 'agent4_integration_audit/run_20261122T035816_395278Z/report.json'
JUNCTION_MLP_GATE_SHA256 = 'f04b9d76a676a1d539f045e4afde89aae4ad438862b8656082c2169fe25d53a1'
JUNCTION_HISTORY_GATE_REPORT = 'agent4_integration_audit/run_20261122T042958_996819Z/report.json'
JUNCTION_HISTORY_GATE_SHA256 = 'b4f26ddd70da8e83407ee7868300640d0d9803c8e892fa523f238e46c1527c2a'
JUNCTION_STOP_REVIEW_REPORT = 'agent4_integration_audit/run_20261122T044931_688023Z/report.json'
JUNCTION_STOP_REVIEW_SHA256 = '82a1e8aeff52c0c098ed8c3c904df9dca30b28e06523f7842da309b871b61644'
STOP_GO_PROTOCOL = {'driver_folds': 5, 'maximum_WAIT_false_launch_rate': .05,
                    'minimum_LAUNCH_recall': .5, 'minimum_LAUNCH_mae_gain_fraction': .1}


def choose_stop_go_threshold(labels, probabilities):
    """Choose solely on TRAIN driver-out-of-fold probabilities; an all-hold gate cannot pass."""
    import numpy as np

    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if (labels.ndim != 1 or probabilities.shape != labels.shape or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1)) or set(labels.tolist()) != {0, 1}):
        raise ValueError('Expected both binary motion classes and finite probabilities')
    wait, launch = labels == 0, labels == 1
    candidates = []
    for threshold in np.unique(np.r_[0., probabilities, np.nextafter(1., 2.)]):
        chosen = probabilities >= threshold
        candidates.append({'threshold': float(threshold),
            'WAIT_false_launch_rate': float(chosen[wait].mean()), 'LAUNCH_recall': float(chosen[launch].mean())})
    within_budget = [point for point in candidates if point['WAIT_false_launch_rate'] <= STOP_GO_PROTOCOL['maximum_WAIT_false_launch_rate']]
    eligible = [point for point in within_budget if point['LAUNCH_recall'] >= STOP_GO_PROTOCOL['minimum_LAUNCH_recall']]
    selected = min(eligible, key=lambda point: (-point['LAUNCH_recall'], point['WAIT_false_launch_rate'], -point['threshold'])) if eligible else None
    return {'scope': 'TRAIN_DRIVER_OUT_OF_FOLD_ONLY', 'status': 'OOF_FEASIBLE' if selected else 'NO_FEASIBLE_TRAIN_OOF_THRESHOLD',
        'selected': selected, 'maximum_recall_within_WAIT_budget': max(point['LAUNCH_recall'] for point in within_budget),
        'WAIT_count': int(wait.sum()), 'LAUNCH_count': int(launch.sum()), 'operating_points': candidates}


def junction_stop_go_readiness(scores, baseline, classifier_f1, baseline_f1, tradeoff):
    """Add the requested ability to launch to the unchanged legacy forecast criteria."""
    unmet = junction_forecast_unmet(scores, baseline, classifier_f1, baseline_f1)
    if tradeoff['WAIT_count'] <= 0 or tradeoff['LAUNCH_count'] <= 0:
        raise ValueError('Stop/go readiness needs both held-out motion classes')
    false_launch = tradeoff['WAIT_predicted_speed_ge_0p3_count']/tradeoff['WAIT_count']
    recall = 1.-tradeoff['LAUNCH_predicted_speed_lt_0p3_count']/tradeoff['LAUNCH_count']
    gain = 1.-scores['LAUNCH']['one_second_mae_mps']/max(baseline['LAUNCH']['one_second_mae_mps'], 1e-12)
    if false_launch > STOP_GO_PROTOCOL['maximum_WAIT_false_launch_rate']:
        unmet.append('WAIT/false_motion_rate_over_0p05')
    if recall < STOP_GO_PROTOCOL['minimum_LAUNCH_recall']:
        unmet.append('LAUNCH/motion_recall_below_0p5')
    if gain < STOP_GO_PROTOCOL['minimum_LAUNCH_mae_gain_fraction']:
        unmet.append('LAUNCH/mae_gain_below_0p10')
    return {'unmet': unmet, 'WAIT_false_motion_rate': false_launch, 'LAUNCH_motion_recall': recall, 'LAUNCH_mae_gain_fraction': gain}


def predict_learned_stop_go_residuals(model, features):
    import numpy as np

    if model is None or model.get('kind') != 'LEARNED_STOP_GO_GATE_V1' or model.get('decision_threshold') is None:
        raise ValueError('No TRAIN-OOF-qualified stop/go speed model is available')
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] < 1 or not np.isfinite(features).all():
        raise ValueError('Invalid stop/go model observations')
    result = np.empty((len(features), 6), dtype=float)
    stopped = features[:, 0] < .3
    if (~stopped).any():
        result[~stopped] = model['moving'].predict(features[~stopped])
    if stopped.any():
        classifier = model['launch_classifier']
        index = list(classifier.classes_).index(1)
        launch = classifier.predict_proba(features[stopped])[:, index] >= model['decision_threshold']
        near = features[stopped]
        predicted = np.empty((len(near), 6), dtype=float)
        for mask, name in ((launch, 'launch'), (~launch, 'hold')):
            if mask.any():
                predicted[mask] = model[name].predict(near[mask])
        result[stopped] = predicted
    if result.shape != (len(features), 6) or not np.isfinite(result).all():
        raise ValueError('Invalid learned stop/go residual output')
    return result


def make_stop_go_gate(neural=False):
    """Keep normalization inside each fit, including every TRAIN driver fold."""
    if not neural:
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=96, min_samples_leaf=8,
                                    class_weight='balanced', random_state=42, n_jobs=-1)
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(
        hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=.01,
        batch_size=64, learning_rate_init=.001, max_iter=300,
        early_stopping=False, random_state=42))])


def validate_stop_go_oof_rows(rows, identities, labels):
    """Reject changed populations/order before using the archived tree comparison."""
    if len(rows) != len(identities) or len(rows) != len(labels):
        raise ValueError('Archived stop/go TRAIN population changed')
    for row, identity, label in zip(rows, identities, labels):
        if row['identity'] != list(identity) or row['motion_label'] != str(label):
            raise ValueError('Archived stop/go identity/order/label changed')
    return choose_stop_go_threshold([int(label == 'LAUNCH') for label in labels],
                                    [row['launch_probability_oof'] for row in rows])


JUNCTION_TRAFFIC_HISTORY_NAMES = [
    name for seconds in ('0p5', '1', '2')
    for name in [*[f'traffic_past_{seconds}s_{field}' for field in JUNCTION_TRAFFIC_NAMES],
                 f'traffic_past_{seconds}s_frame_valid']]


def causal_traffic_history(observations, tick, *, stride, seconds_per_tick):
    """Past traffic summaries in each frame's ego coordinates; no actor-slot tracking."""
    if (type(tick) is not int or type(stride) is not int or stride <= 0
            or not math.isfinite(seconds_per_tick) or seconds_per_tick <= 0):
        raise ValueError('Invalid traffic-history cadence')
    period = stride*seconds_per_tick
    if any(not math.isclose(seconds/period, round(seconds/period), abs_tol=1e-8)
           for seconds in (.5, 1., 2.)):
        raise ValueError('Traffic history cannot represent the physical horizons')
    if observations.get(tick) is None:
        raise ValueError('Missing current traffic observation')
    observed = {}
    for offset in range(round(2./period)+1):
        row = observations.get(tick-offset*stride)
        if row is None:
            break  # No future samples, interpolation, or bridging unavailable frames.
        summary = junction_traffic_features(row)
        if row[3] != 1.:
            break
        observed[offset] = summary
    result = []
    for seconds in (.5, 1., 2.):
        values = observed.get(round(seconds/period))
        result.extend([*values, 1.] if values is not None else [0.]*13)
    return result


def build_stop_go_traffic_history(groups, source, pdd):
    """Read only current/past raw PDD frames; candidate availability is not a history mask."""
    from collections import defaultdict
    import numpy as np
    from build_agent4_pdd_pilot import read_source, paired_state, EgoLaneTypeConflict
    from train_pdd_baseline import function_source_sha256

    pdd = pdd.resolve()
    receipts = {entry['episode']: entry for entry in source['episodes']}
    arrays, summaries, verified = {}, {}, {}
    for split, group in groups.items():
        episodes, reasons = defaultdict(list), Counter()
        extra = np.zeros((len(group['x']), len(JUNCTION_TRAFFIC_HISTORY_NAMES)), dtype=np.float32)
        near = np.flatnonzero(group['x'][:, 0] < .3)
        for index in near:
            driver, scene, tick = group['identities'][int(index)]
            episodes[(driver, scene)].append((int(index), tick))
        for (driver, scene), positions in episodes.items():
            receipt = receipts[f'{driver}/{scene}']
            if receipt['partition'] != split:
                raise ValueError('Traffic history source partition changed')
            inventory = {}
            def read_verified(relative):
                expected = receipt['source_sha256'].get(relative)
                if expected is None:
                    raise ValueError(f'Unreviewed traffic-history source: {relative}')
                value = read_source(pdd / relative, inventory, pdd)
                if inventory[relative] != expected:
                    raise ValueError(f'Traffic history source checksum mismatch: {relative}')
                verified[relative] = expected
                return value
            telemetry = read_verified(f'{driver}/data/{scene}/metric/metric_info.json')
            cache = {}
            def observation(tick):
                if tick in cache:
                    return cache[tick]
                relative = f'{driver}/data/{scene}/boxes/{tick//5}.json.gz'
                if tick < 0 or str(tick) not in telemetry:
                    reasons['MISSING_PAST_TELEMETRY'] += 1
                    cache[tick] = None
                    return None
                if relative not in receipt['source_sha256']:
                    reasons['NO_REVIEWED_PAST_BOX'] += 1
                    cache[tick] = None
                    return None
                boxes = read_verified(relative)
                geometry = set()
                try:
                    features, _, _, _, _, _ = paired_state(telemetry[str(tick)], boxes, geometry_flags=geometry)
                except EgoLaneTypeConflict:
                    reasons['PAST_EGO_LANE_CONFLICT'] += 1
                    cache[tick] = None
                else:
                    if geometry:
                        reasons['PAST_GEOMETRY_UNAVAILABLE'] += 1
                    cache[tick] = None if geometry else list(features)
                return cache[tick]
            for index, tick in positions:
                current = observation(tick)
                if current is None or not np.allclose(junction_forecast_features(current), group['x'][index, :96],
                                                      atol=1e-6, rtol=1e-6):
                    raise ValueError(f'Current traffic frame did not replay: {driver}/{scene}/{tick}')
                past = {}
                for offset in range(9):
                    key = tick-offset*5
                    row = observation(key)
                    if row is None:
                        break
                    past[key] = row
                extra[index] = causal_traffic_history(past, tick, stride=5, seconds_per_tick=.05)
            print(f'Loaded causal traffic history: {driver}/{scene} ({len(positions)} near-stop rows).', flush=True)
        arrays[split] = np.column_stack((group['x'], extra))
        summaries[split] = {'near_stop_rows': len(near), 'past_0p5s_valid': int(extra[:, 12].sum()),
            'past_1s_valid': int(extra[:, 25].sum()), 'past_2s_valid': int(extra[:, 38].sum()),
            'unavailable_frame_reasons': dict(reasons)}
    return arrays, {'features': list(JUNCTION_TRAFFIC_HISTORY_NAMES), 'summary': summaries,
        'verified_source_files': verified, 'feature_function_sha256': function_source_sha256(causal_traffic_history),
        'builder_function_sha256': function_source_sha256(build_stop_go_traffic_history),
        'source': 'RAW_PDD_CURRENT_AND_PAST_ONLY', 'future_frames_used': False,
        'candidate_presence_used_as_input': False, 'actor_slot_matching_used': False,
        'moving_rows_history': 'NOT_USED_BY_GATE_ZERO_PADDED',
        'pdd_cadence': {'stride': 5, 'seconds_per_tick': .05},
        'public_cadence': {'stride': 1, 'seconds_per_tick': .1},
        'public_history_runtime_implemented': False}


class PrefixFeatureRegressor:
    """Use an unchanged speed head while the launch classifier receives extra history."""
    def __init__(self, estimator, width):
        self.estimator, self.width = estimator, width

    def predict(self, features):
        return self.estimator.predict(features[:, :self.width])


def stop_go_diagnostic_points(selection):
    """Describe an unsuccessful ROC tradeoff; never approve a fallback threshold."""
    points = selection['operating_points']
    within_budget = [point for point in points
                     if point['WAIT_false_launch_rate'] <= STOP_GO_PROTOCOL['maximum_WAIT_false_launch_rate']]
    enough_recall = [point for point in points
                    if point['LAUNCH_recall'] >= STOP_GO_PROTOCOL['minimum_LAUNCH_recall']]
    return {'within_WAIT_budget': min(within_budget, key=lambda point:
        (-point['LAUNCH_recall'], point['WAIT_false_launch_rate'], -point['threshold'])),
        'minimum_WAIT_error_at_required_recall': min(enough_recall, key=lambda point:
        (point['WAIT_false_launch_rate'], -point['threshold'])), 'threshold_approved': False}


def stop_go_probability_counts(labels, probabilities, threshold):
    import numpy as np

    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if (labels.ndim != 1 or probabilities.shape != labels.shape or not len(labels)
            or not set(labels.tolist()) <= {0, 1} or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1)) or not math.isfinite(threshold)):
        raise ValueError('Invalid stop/go probability cohort')
    wait, launch, predicted = labels == 0, labels == 1, probabilities >= threshold
    return {'n': len(labels), 'WAIT_count': int(wait.sum()), 'LAUNCH_count': int(launch.sum()),
        'WAIT_false_launch_count': int((wait & predicted).sum()),
        'LAUNCH_detected_count': int((launch & predicted).sum()),
        'LAUNCH_missed_count': int((launch & ~predicted).sum()),
        'WAIT_false_launch_rate': float(predicted[wait].mean()) if wait.any() else None,
        'LAUNCH_recall': float(predicted[launch].mean()) if launch.any() else None,
        'brier_score': float(np.mean((probabilities-labels)**2))}


def observed_stop_age(speeds, tick, *, stride, seconds_per_tick, maximum_s=30.):
    """Observed stationary-span lower bound, with explicit gap/start/cap censoring."""
    from agent4_pdd_common_v2 import number

    if (type(tick) is not int or type(stride) is not int or stride <= 0
            or not math.isfinite(seconds_per_tick) or seconds_per_tick <= 0
            or not math.isfinite(maximum_s) or maximum_s <= 0):
        raise ValueError('Invalid stop-age cadence/cap')
    period = stride*seconds_per_tick
    if not math.isclose(maximum_s/period, round(maximum_s/period), abs_tol=1e-8):
        raise ValueError('Stop-age cap is not representable at this cadence')
    if tick not in speeds or not 0 <= number(speeds[tick]) < .3:
        raise ValueError('Stop-age review needs a current near-stop speed')
    lower_bound, boundary = 0., 'HORIZON_CAP'
    for offset in range(1, round(maximum_s/period)+1):
        key = tick-offset*stride
        if key not in speeds:
            boundary = 'EPISODE_START' if key < 0 else 'MISSING_PAST_SAMPLE'
            break
        value = number(speeds[key])
        if value < 0:
            raise ValueError('Negative history speed')
        if value >= .3:
            boundary = 'OBSERVED_MOVING'
            break
        lower_bound = offset*period
    return {'lower_bound_s': lower_bound, 'boundary': boundary,
            'left_censored': boundary != 'OBSERVED_MOVING', 'maximum_s': maximum_s}


def review_junction_stop_go(out, pdd):
    """Reuse archived OOF predictions; review label/visibility/stop-clock cohorts without fitting."""
    from collections import defaultdict
    import numpy as np
    from build_agent4_pdd_pilot import read_source, xyz

    sources = {'tree114': (JUNCTION_TREE_GATE_REPORT, JUNCTION_TREE_GATE_SHA256),
        'mlp114': (JUNCTION_MLP_GATE_REPORT, JUNCTION_MLP_GATE_SHA256),
        'tree153_history': (JUNCTION_HISTORY_GATE_REPORT, JUNCTION_HISTORY_GATE_SHA256)}
    models, reference_rows = {}, None
    for name, (relative, checksum) in sources.items():
        path = ROOT / relative
        if digest(path) != checksum:
            raise ValueError(f'Reviewed stop/go report changed: {name}')
        report = json.loads(path.read_bytes())
        archive = path.parent / 'train_driver_oof.jsonl.gz'
        if (report['offline_status'] != 'STOP_GO_TRAIN_OOF_NEEDS_FIX'
                or report['protocol'] != STOP_GO_PROTOCOL or report['decision_threshold'] is not None
                or report['speed_model_fitted'] or report['deployment_eligible']
                or digest(path.parent / 'model_bundle.joblib') != report['model_bundle_sha256']
                or digest(archive) != report['train_driver_oof']['archive_sha256']):
            raise ValueError(f'Stop/go review provenance/status failed: {name}')
        with gzip.open(archive, 'rt', encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream]
        if reference_rows is None:
            reference_rows = rows
        selection = validate_stop_go_oof_rows(rows, [row['identity'] for row in reference_rows],
                                             [row['motion_label'] for row in reference_rows])
        if selection != report['train_driver_oof']['operating_point_selection']:
            raise ValueError('Archived stop/go OOF operating points did not replay')
        models[name] = {'report_sha256': checksum, 'model_sha256': report['model_bundle_sha256'],
            'archive_sha256': digest(archive), 'diagnostic_points': stop_go_diagnostic_points(selection),
            'probabilities': np.asarray([row['launch_probability_oof'] for row in rows])}
    identities = [tuple(row['identity']) for row in reference_rows]
    if len(set(identities)) != len(identities) or any(not 1 <= int(driver.removeprefix('driver_')) <= 21
                                                   for driver, _, _ in identities):
        raise ValueError('OOF review needs unique TRAIN driver identities')
    labels = np.asarray([int(row['motion_label'] == 'LAUNCH') for row in reference_rows])
    source_path = ROOT / JUNCTION_FORECAST_SOURCE_REPORT
    if digest(source_path) != JUNCTION_FORECAST_SOURCE_SHA256:
        raise ValueError('Expected the reviewed original PDD junction export')
    source = json.loads(source_path.read_bytes())
    candidates = source_path.parent / 'junction_candidates.jsonl'
    if digest(candidates) != source['candidates_sha256']:
        raise ValueError('Stop/go label-review candidate checksum mismatch')
    wanted, metadata = set(identities), {}
    with candidates.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            identity = (row['driver'], row['scenario'], row['telemetry_step'])
            if identity not in wanted:
                continue
            example = junction_forecast_example(row)
            if identity in metadata or row['partition'] != 'TRAIN' or example is None:
                raise ValueError('Invalid/duplicate TRAIN stop/go source row')
            features, speeds, label = example
            if features[0] >= .3 or label != ('LAUNCH' if speeds[3] >= .3 else 'WAIT'):
                raise ValueError('Stop/go human-motion label failed source replay')
            metadata[identity] = {'identity': list(identity), 'motion_label': label,
                'current_speed_mps': features[0], 'human_speed_1s_mps': speeds[3],
                'human_speed_1p5s_mps': speeds[5], 'future_targets_for_review_only': speeds,
                'source_context_for_review_only': row['source_context_for_parity_review'],
                'overflow_count': round(math.expm1(features[89]))}
    if set(metadata) != wanted:
        raise ValueError('Stop/go review could not recover every TRAIN OOF identity')
    episodes, verified = defaultdict(list), {}
    for index, (driver, scene, tick) in enumerate(identities):
        if metadata[(driver, scene, tick)]['motion_label'] != reference_rows[index]['motion_label']:
            raise ValueError('Archived stop/go labels differ from original PDD')
        episodes[(driver, scene)].append((index, tick))
    receipts = {row['episode']: row for row in source['episodes']}
    pdd = pdd.resolve()
    for (driver, scene), positions in episodes.items():
        relative = f'{driver}/data/{scene}/metric/metric_info.json'
        inventory = {}
        telemetry = read_source(pdd / relative, inventory, pdd)
        receipt = receipts[f'{driver}/{scene}']
        if receipt['partition'] != 'TRAIN' or inventory[relative] != receipt['source_sha256'][relative]:
            raise ValueError(f'Stop-age source checksum/split mismatch: {relative}')
        verified[relative] = inventory[relative]
        speeds = {int(tick): math.hypot(*xyz(info['velocity'])[:2]) for tick, info in telemetry.items()}
        for index, tick in positions:
            row = metadata[identities[index]]
            if not math.isclose(speeds[tick], row['current_speed_mps'], abs_tol=1e-5, rel_tol=1e-6):
                raise ValueError('Stop-age current speed differs from archived source row')
            row['observed_stop_age'] = observed_stop_age(speeds, tick, stride=5, seconds_per_tick=.05)
    cohorts = defaultdict(list)
    for index, identity in enumerate(identities):
        row = metadata[identity]
        age = row['observed_stop_age']['lower_bound_s']
        age_bucket = 'lt_0p5s' if age < .5 else '0p5_to_2s' if age < 2. else '2_to_5s' if age < 5. else 'ge_5s'
        keys = ['all', f'driver/{identity[0]}', f'scenario/{identity[1]}', f'stop_age/{age_bucket}',
            f'stop_age_boundary/{row["observed_stop_age"]["boundary"]}',
            'inside_junction' if row['source_context_for_review_only']['is_in_junction'] else 'approach',
            'overflow' if row['overflow_count'] else 'no_overflow']
        if row['motion_label'] == 'WAIT':
            future = row['future_targets_for_review_only']
            timing = ('intermediate_motion' if max(future[:4]) >= .3 else
                      'launch_after_1s' if max(future[4:]) >= .3 else 'still_through_1p5s')
            keys.append(f'HINDSIGHT_WAIT/{timing}')
        else:
            keys.append('HINDSIGHT_LAUNCH/weak_under_1mps' if row['human_speed_1s_mps'] < 1.
                        else 'HINDSIGHT_LAUNCH/at_least_1mps')
        for key in keys:
            cohorts[key].append(index)
    cohort_scores = {}
    for name, model in models.items():
        threshold = model['diagnostic_points']['within_WAIT_budget']['threshold']
        cohort_scores[name] = {key: stop_go_probability_counts(labels[indices], model['probabilities'][indices], threshold)
                              for key, indices in cohorts.items()}
    latest = models['tree153_history']
    threshold = latest['diagnostic_points']['within_WAIT_budget']['threshold']
    predicted = latest['probabilities'] >= threshold
    errors = {}
    for kind, mask in (('missed_LAUNCH', (labels == 1) & ~predicted), ('false_LAUNCH_on_WAIT', (labels == 0) & predicted)):
        indices = np.flatnonzero(mask)
        order = sorted(indices.tolist(), key=lambda index: float(latest['probabilities'][index]),
                       reverse=kind == 'false_LAUNCH_on_WAIT')[:20]
        errors[kind] = [{**metadata[identities[index]], 'launch_probability_oof': float(latest['probabilities'][index])}
                        for index in order]
    result = {'schema': 'PDD_JUNCTION_STOP_GO_REVIEW_V1', 'status': 'TRAIN_OOF_REVIEW_ONLY',
        'scope': 'POSTHOC_TRAIN_ERROR_REVIEW_NOT_AN_INDEPENDENT_BENCHMARK',
        'training_performed': False, 'simulator_used': False, 'test_used': False,
        'dev_predictions_reanalyzed': False, 'threshold_selected_for_deployment': False,
        'policy_training_approved': False, 'deployment_eligible': False,
        'source_report_sha256': JUNCTION_FORECAST_SOURCE_SHA256, 'candidates_sha256': source['candidates_sha256'],
        'n_TRAIN_rows': len(identities), 'protocol_unchanged': dict(STOP_GO_PROTOCOL),
        'models': {name: {key: value for key, value in model.items() if key != 'probabilities'} for name, model in models.items()},
        'cohort_metrics_at_diagnostic_WAIT_budget': cohort_scores, 'error_examples': errors,
        'long_stop_clock': {'maximum_s': 30., 'TRAIN_rows_observed_still_over_2s':
            sum(metadata[identity]['observed_stop_age']['lower_bound_s'] > 2. for identity in identities),
            'verified_source_files': verified}, 'implementation_sha256': digest(__file__),
        'limitations': ['The diagnostic ROC points are not qualified decision thresholds and do not authorize moving the car.',
            'Human WAIT/LAUNCH labels describe timing; none of these records certifies right-of-way or a safe gap.',
            'Future motion and source junction metadata define hindsight cohorts only; they never enter model inputs.',
            'Stop age is an observed lower bound, censored by recording start, gaps or 30s cap; it is not permission to go.',
            'OOF predictions have already been inspected during development; these are not new unbiased validation results.',
            'No conclusion that missing stop duration explains errors, or that safe autonomous junction driving is learnable, follows without further evidence.']}
    (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for name, model in models.items():
        points = model['diagnostic_points']
        print(f"{name}: LAUNCH recall under WAIT budget={points['within_WAIT_budget']['LAUNCH_recall']:.3f}; "
              f"WAIT error needed for required recall={points['minimum_WAIT_error_at_required_recall']['WAIT_false_launch_rate']:.3f}", flush=True)
    print(f"TRAIN_OOF_REVIEW_ONLY; no training/control/checkpoint changed. Read {out / 'report.json'}", flush=True)


STOP_CLOCK_BOUNDARIES = ('OBSERVED_MOVING', 'EPISODE_START', 'MISSING_PAST_SAMPLE', 'HORIZON_CAP')
STOP_CLOCK_NAMES = ['observed_near_stop_span_capped_30s_s',
                   *[f'stop_clock_boundary_{name.lower()}' for name in STOP_CLOCK_BOUNDARIES]]


def stop_clock_features(speeds, tick, *, stride, seconds_per_tick):
    age = observed_stop_age(speeds, tick, stride=stride, seconds_per_tick=seconds_per_tick, maximum_s=30.)
    return [age['lower_bound_s'], *[float(age['boundary'] == name) for name in STOP_CLOCK_BOUNDARIES]]


def add_stop_clock_observations(groups, observations, source, pdd):
    from collections import defaultdict
    import numpy as np
    from build_agent4_pdd_pilot import read_source, xyz
    from train_pdd_baseline import function_source_sha256

    pdd = pdd.resolve()
    receipts = {row['episode']: row for row in source['episodes']}
    result, summary, verified = {}, {}, {}
    for split, group in groups.items():
        previous = np.asarray(observations[split], dtype=np.float32)
        if (previous.shape != (len(group['x']), 153) or not np.isfinite(previous).all()
                or not np.array_equal(previous[:, :114], group['x'])):
            raise ValueError('Stop-clock base153D observations/population changed')
        extra = np.zeros((len(previous), len(STOP_CLOCK_NAMES)), dtype=np.float32)
        episodes = defaultdict(list)
        for index in np.flatnonzero(group['x'][:, 0] < .3):
            driver, scene, tick = group['identities'][int(index)]
            episodes[(driver, scene)].append((int(index), tick))
        for (driver, scene), positions in episodes.items():
            relative = f'{driver}/data/{scene}/metric/metric_info.json'
            inventory = {}
            telemetry = read_source(pdd / relative, inventory, pdd)
            receipt = receipts[f'{driver}/{scene}']
            if receipt['partition'] != split or inventory[relative] != receipt['source_sha256'][relative]:
                raise ValueError(f'Stop-clock checksum/split mismatch: {relative}')
            verified[relative] = inventory[relative]
            speeds = {int(tick): math.hypot(*xyz(info['velocity'])[:2]) for tick, info in telemetry.items()}
            for index, tick in positions:
                if not math.isclose(speeds[tick], float(group['x'][index, 0]), abs_tol=1e-5, rel_tol=1e-6):
                    raise ValueError('Stop-clock current speed did not replay')
                extra[index] = stop_clock_features(speeds, tick, stride=5, seconds_per_tick=.05)
        result[split] = np.column_stack((previous, extra))
        summary[split] = {'near_stop_rows': int((group['x'][:, 0] < .3).sum()),
            'observed_span_over_2s': int((extra[:, 0] > 2.).sum()),
            'boundaries': {name: int(extra[:, index+1].sum()) for index, name in enumerate(STOP_CLOCK_BOUNDARIES)}}
    return result, {'features': list(STOP_CLOCK_NAMES), 'maximum_s': 30., 'near_stop_threshold_mps': .3,
        'summary': summary, 'verified_source_files': verified, 'causal_only': True,
        'episode_timestamp_used_as_feature': False, 'human_controls_used': False,
        'age_function_sha256': function_source_sha256(observed_stop_age),
        'feature_function_sha256': function_source_sha256(stop_clock_features),
        'builder_function_sha256': function_source_sha256(add_stop_clock_observations),
        'public_runtime_cache_implemented': False,
        'meaning': 'Observed low-speed span lower bound; boundaries encode censoring, not permission to move'}


def train_junction_stop_go(out, groups, *, neural_gate=False, traffic_history_pdd=None, source=None, stop_clock=False):
    """Learn a launch gate on PDD, qualify it across TRAIN drivers before fitting speed experts."""
    import joblib
    import numpy as np
    import sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.metrics import confusion_matrix
    from train_pdd_baseline import function_source_sha256

    parent_path = ROOT / JUNCTION_QUANTILE_REPORT
    if digest(parent_path) != JUNCTION_QUANTILE_SHA256:
        raise ValueError('Expected the reviewed q10 forecast report')
    parent_report = json.loads(parent_path.read_bytes())
    parent_model_path = parent_path.parent / 'model_bundle.joblib'
    if digest(parent_model_path) != parent_report['model_bundle_sha256']:
        raise ValueError('q10 source checkpoint changed')
    bindings = {'feature_function_sha256': junction_forecast_features, 'history_function_sha256': causal_speed_history,
                'traffic_function_sha256': junction_traffic_features, 'speed_decoder_sha256': predict_junction_speed_residuals}
    if any(parent_report[name] != function_source_sha256(function) for name, function in bindings.items()):
        raise ValueError('Stop/go source observation/decoder contract changed')
    parent = joblib.load(parent_model_path)
    train, dev = groups['TRAIN'], groups['DEV']
    for split, group in groups.items():
        if len(group['x']) != parent_report['counts'][split]['retained'] or group['x'].shape[1] != len(parent['features']):
            raise ValueError('Stop/go population/features differ from the q10 experiment')
    parent_prediction = np.maximum(0., dev['x'][:, :1]+predict_junction_speed_residuals(parent['speed_regressor'], dev['x']))
    parent_scores = junction_forecast_scores(dev['y'], parent_prediction, dev['labels'])
    for group, values in parent_scores.items():
        for metric in ('mae_mps', 'one_second_mae_mps'):
            if not math.isclose(values[metric], parent_report['dev_speed'][parent_report['selected_model']][group][metric], abs_tol=1e-8, rel_tol=0.):
                raise ValueError('q10 parent speed predictions did not replay')
    inherited_classifier = parent['transition_classifier']
    if confusion_matrix(dev['labels'], inherited_classifier.predict(dev['x']), labels=list(JUNCTION_MOTION_CLASSES)).tolist() != parent_report['dev_transition']['confusion_matrix']:
        raise ValueError('Parent motion classifier did not replay')
    baseline = junction_forecast_scores(dev['y'], np.repeat(dev['x'][:, :1], 6, axis=1), dev['labels'])
    f1 = parent_report['dev_transition']['extra_trees_macro_f1']
    baseline_f1 = parent_report['dev_transition']['persistence_macro_f1']
    parent_readiness = junction_stop_go_readiness(parent_scores, baseline, f1, baseline_f1,
                                                 junction_stop_launch_tradeoff(dev['labels'], parent_prediction))
    near_indices = np.flatnonzero(train['x'][:, 0] < .3)
    near_labels = np.asarray(train['labels'])[near_indices]
    if not set(near_labels.tolist()) <= {'WAIT', 'LAUNCH'}:
        raise ValueError('Near-stop TRAIN labels violate the observed-motion partition')
    labels = (near_labels == 'LAUNCH').astype(int)
    drivers = np.asarray([train['identities'][int(index)][0] for index in near_indices])
    if neural_gate and traffic_history_pdd is not None:
        raise ValueError('The traffic-history experiment keeps the archived ExtraTrees gate parameters')
    gate_observations = {split: group['x'] for split, group in groups.items()}
    history_contract, clock_contract, history_comparison = None, None, None
    if traffic_history_pdd is not None:
        gate_observations, history_contract = build_stop_go_traffic_history(groups, source, traffic_history_pdd)
    if stop_clock:
        if history_contract is None:
            raise ValueError('The clock experiment requires the unchanged153D traffic-history base')
        review_path, history_path = ROOT/JUNCTION_STOP_REVIEW_REPORT, ROOT/JUNCTION_HISTORY_GATE_REPORT
        if digest(review_path) != JUNCTION_STOP_REVIEW_SHA256 or digest(history_path) != JUNCTION_HISTORY_GATE_SHA256:
            raise ValueError('Expected the completed stop/go review and rejected history153D report')
        review = json.loads(review_path.read_bytes())
        reference = json.loads(history_path.read_bytes())
        if (review['status'] != 'TRAIN_OOF_REVIEW_ONLY' or review['training_performed']
                or review['deployment_eligible'] or reference['protocol'] != STOP_GO_PROTOCOL
                or history_contract != reference['traffic_history']
                or digest(history_path.parent/'model_bundle.joblib') != reference['model_bundle_sha256']):
            raise ValueError('Clock experiment provenance/history contract changed')
        old = joblib.load(history_path.parent/'model_bundle.joblib')
        near_dev_reference = np.flatnonzero(dev['x'][:, 0] < .3)
        recorded = reference['dev_launch_probability_diagnostics']['examples']
        if [list(dev['identities'][int(index)]) for index in near_dev_reference] != [row['identity'] for row in recorded]:
            raise ValueError('History153D DEV replay population changed')
        probabilities = old['launch_classifier'].predict_proba(gate_observations['DEV'][near_dev_reference])[:,
            list(old['launch_classifier'].classes_).index(1)]
        if not np.allclose(probabilities, [row['launch_probability'] for row in recorded], atol=1e-8, rtol=0.):
            raise ValueError('History153D parent probabilities did not replay')
        history_archive = history_path.parent/'train_driver_oof.jsonl.gz'
        if digest(history_archive) != reference['train_driver_oof']['archive_sha256']:
            raise ValueError('History153D OOF archive changed')
        with gzip.open(history_archive, 'rt', encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream]
        selection = validate_stop_go_oof_rows(rows, [train['identities'][int(index)] for index in near_indices], near_labels)
        if selection != reference['train_driver_oof']['operating_point_selection']:
            raise ValueError('History153D TRAIN OOF probabilities did not replay')
        history_comparison = {'report_sha256': JUNCTION_HISTORY_GATE_SHA256,
            'review_sha256': JUNCTION_STOP_REVIEW_SHA256, 'parent_probability_replay': 'PASS',
            'operating_point_selection': selection}
        gate_observations, clock_contract = add_stop_clock_observations(groups, gate_observations, source, traffic_history_pdd)
        if (clock_contract['summary']['TRAIN']['observed_span_over_2s'] != review['long_stop_clock']['TRAIN_rows_observed_still_over_2s']
                or clock_contract['summary']['TRAIN']['boundaries']['EPISODE_START'] !=
                   review['cohort_metrics_at_diagnostic_WAIT_budget']['tree153_history']['stop_age_boundary/EPISODE_START']['n']):
            raise ValueError('Stop-clock observations did not replay the completed TRAIN review')
        clock_contract['source_review_sha256'] = JUNCTION_STOP_REVIEW_SHA256
    x = gate_observations['TRAIN'][near_indices]
    tree_report, tree_comparison = None, None
    if neural_gate or history_contract is not None:
        tree_path = ROOT / JUNCTION_TREE_GATE_REPORT
        if digest(tree_path) != JUNCTION_TREE_GATE_SHA256:
            raise ValueError('Expected the reviewed ExtraTrees stop/go report')
        tree_report = json.loads(tree_path.read_bytes())
        tree_archive = tree_path.parent / 'train_driver_oof.jsonl.gz'
        if (tree_report['schema'] != 'PDD_JUNCTION_STOP_GO_V5'
                or tree_report['protocol'] != STOP_GO_PROTOCOL
                or tree_report['parent_q10_report_sha256'] != JUNCTION_QUANTILE_SHA256
                or digest(tree_path.parent / 'model_bundle.joblib') != tree_report['model_bundle_sha256']
                or digest(tree_archive) != tree_report['train_driver_oof']['archive_sha256']):
            raise ValueError('ExtraTrees comparison provenance/protocol changed')
        with gzip.open(tree_archive, 'rt', encoding='utf-8') as stream:
            tree_rows = [json.loads(line) for line in stream]
        tree_selection = validate_stop_go_oof_rows(tree_rows,
            [train['identities'][int(index)] for index in near_indices], near_labels)
        if tree_selection != tree_report['train_driver_oof']['operating_point_selection']:
            raise ValueError('ExtraTrees archived TRAIN operating points did not replay')
        tree_comparison = {'report_sha256': JUNCTION_TREE_GATE_SHA256,
            'archive_sha256': digest(tree_archive), 'population_replay': 'PASS',
            'operating_point_selection': tree_selection,
            'dev_brier_score': tree_report['dev_launch_probability_diagnostics']['brier_score']}
    oof, seen = np.zeros(len(x)), np.zeros(len(x), dtype=int)
    folds = []
    for fold, (fit, validation) in enumerate(GroupKFold(n_splits=STOP_GO_PROTOCOL['driver_folds']).split(x, labels, drivers), 1):
        fit_drivers, validation_drivers = set(drivers[fit].tolist()), set(drivers[validation].tolist())
        if fit_drivers & validation_drivers or set(labels[fit].tolist()) != {0, 1}:
            raise ValueError('Invalid TRAIN driver fold or missing fit class')
        fold_record = {'fold': fold, 'fit_drivers': sorted(fit_drivers), 'validation_drivers': sorted(validation_drivers),
                       'fit_rows': len(fit), 'validation_rows': len(validation)}
        if tree_report is not None and fold_record != tree_report['train_driver_oof']['folds'][fold-1]:
            raise ValueError('Stop/go comparison must use the exact archived TRAIN driver folds')
        gate = make_stop_go_gate(neural_gate)
        gate.fit(x[fit], labels[fit])
        oof[validation] = gate.predict_proba(x[validation])[:, list(gate.classes_).index(1)]
        seen[validation] += 1
        if neural_gate:
            fold_record['mlp_fit'] = {'n_iter': int(gate.named_steps['mlp'].n_iter_),
                                      'loss': float(gate.named_steps['mlp'].loss_)}
        folds.append(fold_record)
        print(f'TRAIN driver-fold {fold}/{STOP_GO_PROTOCOL["driver_folds"]} completed.', flush=True)
    if not np.all(seen == 1):
        raise ValueError('TRAIN OOF predictions do not cover each example exactly once')
    calibration = choose_stop_go_threshold(labels, oof)
    gate = make_stop_go_gate(neural_gate)
    gate.fit(x, labels)
    if not neural_gate:
        gate.n_jobs = 1
    threshold = None if calibration['selected'] is None else calibration['selected']['threshold']
    near_dev = np.flatnonzero(dev['x'][:, 0] < .3)
    dev_binary_labels = (np.asarray(dev['labels'])[near_dev] == 'LAUNCH').astype(int)
    dev_probabilities = gate.predict_proba(gate_observations['DEV'][near_dev])[:, list(gate.classes_).index(1)]
    probability_diagnostics = {'scope': 'DEV_DIAGNOSTIC_ONLY_NOT_THRESHOLD_SELECTION',
        'brier_score': float(np.mean((dev_probabilities-dev_binary_labels)**2)),
        'examples': [{'identity': dev['identities'][int(index)], 'motion_label': dev['labels'][int(index)],
                      'launch_probability': float(dev_probabilities[position])} for position, index in enumerate(near_dev)]}
    archive = out / 'train_driver_oof.jsonl.gz'
    with gzip.open(archive, 'xt', encoding='utf-8') as stream:
        for position, index in enumerate(near_indices):
            stream.write(json.dumps({'identity': train['identities'][int(index)], 'motion_label': str(near_labels[position]),
                                     'launch_probability_oof': float(oof[position])})+'\n')
    speed_model, speed_scores, readiness = None, None, None
    unmet = ['TRAIN/NO_FEASIBLE_DRIVER_OOF_THRESHOLD'] if threshold is None else []
    if threshold is not None:
        parameters = {'loss': 'absolute_error', 'max_iter': 160, 'max_leaf_nodes': 15, 'min_samples_leaf': 20,
                      'learning_rate': .08, 'l2_regularization': 1., 'early_stopping': False, 'random_state': 42}
        experts = {}
        for name, binary_label in (('hold', 0), ('launch', 1)):
            selected = near_indices[labels == binary_label]
            if len(selected) < 40:
                raise ValueError(f'Insufficient TRAIN examples for the {name} speed expert')
            expert = MultiOutputRegressor(HistGradientBoostingRegressor(**parameters), n_jobs=1)
            expert.fit(train['x'][selected], train['y'][selected]-train['x'][selected, :1])
            experts[name] = expert
        speed_model = {'kind': 'LEARNED_STOP_GO_GATE_V1', 'decision_threshold': threshold, 'launch_classifier': gate,
                       'moving': parent['speed_regressor']['moving'], **experts}
        if history_contract is not None:
            # A canonical import prevents CLI checkpoints from pickling a __main__ class.
            from agent4_integration_audit.audit_compatibility import PrefixFeatureRegressor as SpeedPrefixRegressor
            for name in ('moving', 'hold', 'launch'):
                speed_model[name] = SpeedPrefixRegressor(speed_model[name], train['x'].shape[1])
        predicted = np.maximum(0., dev['x'][:, :1]+predict_learned_stop_go_residuals(speed_model, gate_observations['DEV']))
        speed_scores = junction_forecast_scores(dev['y'], predicted, dev['labels'])
        readiness = junction_stop_go_readiness(speed_scores, baseline, f1, baseline_f1,
                                              junction_stop_launch_tradeoff(dev['labels'], predicted))
        unmet = readiness['unmet']
    contract = {'schema': ('PDD_JUNCTION_STOP_GO_CLOCK_V8' if clock_contract is not None
                           else 'PDD_JUNCTION_STOP_GO_HISTORY_V7' if history_contract is not None
                           else 'PDD_JUNCTION_STOP_GO_MLP_V6' if neural_gate else 'PDD_JUNCTION_STOP_GO_V5'),
        'scope': 'OFFLINE_HUMAN_MOTION_FORECAST_ONLY',
        'launch_gate_estimator': 'StandardScaler+MLPClassifier' if neural_gate else 'ExtraTreesClassifier',
        'features': (list(parent['features'])+(JUNCTION_TRAFFIC_HISTORY_NAMES if history_contract is not None else [])
                     +(STOP_CLOCK_NAMES if clock_contract is not None else [])),
        'stop_clock': clock_contract,
        'traffic_history': history_contract, 'speed_observation_width': train['x'].shape[1],
        'transition_observation_width': train['x'].shape[1],
        'speed_prefix_adapter_sha256': function_source_sha256(PrefixFeatureRegressor) if history_contract is not None else None,
        'horizons_s': parent['horizons_s'], 'decision_threshold': threshold,
        'offline_status': 'STOP_GO_TRAIN_OOF_NEEDS_FIX' if threshold is None else 'STOP_GO_DEV_NEEDS_FIX' if unmet else 'STOP_GO_DEV_IMPROVED',
        'source_report_sha256': parent['source_report_sha256'], 'candidates_sha256': parent['candidates_sha256'],
        'parent_q10_report_sha256': JUNCTION_QUANTILE_SHA256, 'parent_model_sha256': parent_report['model_bundle_sha256'],
        'feature_function_sha256': parent['feature_function_sha256'], 'history_function_sha256': parent['history_function_sha256'],
        'traffic_function_sha256': parent['traffic_function_sha256'], 'speed_decoder_sha256': function_source_sha256(predict_learned_stop_go_residuals),
        'protocol': dict(STOP_GO_PROTOCOL), 'policy_training_approved': False, 'deployment_eligible': False,
        'source_behavior_quality': 'UNREVIEWED', 'speed_model_fitted': speed_model is not None}
    if len(contract['features']) != x.shape[1]:
        raise ValueError('Stop/go feature names and trained input width differ')
    model_path = out / 'model_bundle.joblib'
    joblib.dump({**contract, 'launch_classifier': gate, 'speed_regressor': speed_model,
                 'transition_classifier': inherited_classifier}, model_path)
    gate_fit = ({'hidden_layer_sizes': [64, 32], 'activation': 'relu', 'solver': 'adam',
        'alpha': .01, 'batch_size': 64, 'learning_rate_init': .001, 'max_iter': 300,
        'early_stopping': False, 'random_state': 42, 'normalization': 'FIT_WITHIN_EACH_TRAIN_FOLD',
        'input_features': x.shape[1], 'n_iter': int(gate.named_steps['mlp'].n_iter_),
        'loss': float(gate.named_steps['mlp'].loss_),
        'parameter_count': int(sum(array.size for array in gate.named_steps['mlp'].coefs_)
                               +sum(array.size for array in gate.named_steps['mlp'].intercepts_))}
        if neural_gate else {'n_estimators': 96, 'min_samples_leaf': 8, 'class_weight': 'balanced',
                            'random_state': 42, 'fit_n_jobs': -1, 'predict_n_jobs': 1})
    report = {**contract, 'training_performed': True, 'test_used': False, 'simulator_used': False,
        'parent_prediction_replay': 'PASS', 'parent_legacy_offline_status': parent_report['offline_status'],
        'parent_actual_stop_go_readiness': parent_readiness, 'counts': parent_report['counts'],
        'train_driver_oof': {'folds': folds, 'operating_point_selection': calibration,
                             'archive_sha256': digest(archive)},
        'dev_launch_probability_diagnostics': probability_diagnostics,
        'dev_speed': speed_scores, 'dev_stop_go_readiness': readiness, 'unmet_criteria': unmet,
        'archived_tree_comparison': tree_comparison,
        'archived_history_comparison': history_comparison,
        'fitting': {'launch_gate': {**gate_fit, 'input_features': x.shape[1]},
                    'speed_experts': parameters if speed_model is not None else None,
                    'near_stop_TRAIN_rows': len(near_indices), 'moving_head_refitted': False,
                    'inherited_motion_classifier_refitted': False, 'sklearn_version': sklearn.__version__},
        'model_bundle_sha256': digest(model_path), 'implementation_sha256': digest(__file__),
        'limitations': ['The q10 legacy forecast status is preserved; the stronger stop/go protocol detects its nearly always-hold behavior.',
            'Threshold selection uses only TRAIN driver-OOF predictions. DEV never sets the threshold.',
            'If no TRAIN threshold qualifies, speed experts are not fitted and the serialized speed model is None; no fallback threshold is invented.',
            'The MLP experiment changes only the binary launch gate, not the observation schema, labels, driver folds, thresholds or driving controller.',
            'The traffic-history experiment appends past traffic summaries only to the launch gate; speed heads keep their original 114D inputs.',
            'Traffic-history validity describes observed past frames, not safe gaps. Public runtime history is not integrated by this offline experiment.',
            'The clock experiment adds only causal low-speed span/censoring inputs; no duration forces a launch, and missing right-of-way/traffic visibility remains unresolved.',
            'These labels describe human movement, not legal priority or a safe gap. OOF empirical rates are not safety guarantees.',
            'No simulator data, TEST drivers, current human controls or future context enter the model inputs. Nothing is deployed.']}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f"{contract['offline_status']}; TRAIN threshold={threshold}; maximum OOF recall within WAIT budget="
          f"{calibration['maximum_recall_within_WAIT_budget']:.3f}; unmet={unmet}. Read {out / 'report.json'}", flush=True)


def train_pdd_junction_forecast(out, *, temporal_pdd=None, traffic_features=False, error_review=False, quantile_forecast=False, stop_go=False, neural_stop_go=False, traffic_stop_go=False, clock_stop_go=False):
    """Offline human-motion experiment, not approval to train/deploy an admission policy."""
    import joblib
    import numpy as np
    import sklearn
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from build_agent4_pdd_pilot import partition
    from agent4_pdd_common_v2 import FEATURE_NAMES
    from train_pdd_baseline import PLAN_NAV_FEATURES, function_source_sha256

    source_path = ROOT / JUNCTION_FORECAST_SOURCE_REPORT
    if digest(source_path) != JUNCTION_FORECAST_SOURCE_SHA256:
        raise ValueError('Expected the reviewed PDD junction candidate export')
    source = json.loads(source_path.read_bytes())
    if (source['schema'] != 'PDD_JUNCTION_REVIEW_V1' or source['status'] != 'REVIEW_REQUIRED'
            or source['unexpected_alignment_or_source_errors'] or source['source'] != 'PDD_ONLY'
            or source['test_drivers_read'] or source['feature_names'] != list(FEATURE_NAMES)):
        raise ValueError('Junction source validity/split/schema gate failed')
    for name, expected in (
        ('train_pdd_baseline.py', '22c6a97d22b0b4ad239a16bbe22d5a1525f7f29b18d991d92ad6cf0c21454ea8'),
        ('agent4_pdd_common_v2.py', source['frozen_encoder_sha256']),
        ('build_agent4_pdd_pilot.py', source['frozen_builder_sha256']),
    ):
        if digest(ROOT / name) != expected:
            raise ValueError(f'Frozen forecast helper changed: {name}')
    candidates = source_path.parent / 'junction_candidates.jsonl'
    if digest(candidates) != source['candidates_sha256']:
        raise ValueError('Junction candidates checksum mismatch')
    groups = {split: {'x': [], 'y': [], 'labels': [], 'identities': [], 'contexts': []} for split in ('TRAIN', 'DEV')}
    seen, raw_counts, excluded = set(), Counter(), Counter()
    with candidates.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            driver = int(row['driver'].removeprefix('driver_'))
            split, step = row['partition'], row['telemetry_step']
            identity = (row['driver'], row['scenario'], step)
            if (not 1 <= driver <= 25 or row['driver'] != f'driver_{driver:02d}'
                    or split != partition(driver) or type(step) is not int or step < 0 or step % 5
                    or row['frame_id'] != step//5 or identity in seen):
                raise ValueError(f'Invalid junction identity/split: {identity}')
            seen.add(identity)
            raw_counts[split] += 1
            example = junction_forecast_example(row)
            if example is None:
                excluded[split] += 1
                continue
            x, y, label = example
            group = groups[split]
            group['x'].append(x)
            group['y'].append(y)
            group['labels'].append(label)
            group['identities'].append(identity)
            group['contexts'].append(row['source_context_for_parity_review'])
    if dict(raw_counts) != {split: source['split_counts'][split]['candidates'] for split in groups}:
        raise ValueError('Junction row counts differ from the reviewed export')
    for split, group in groups.items():
        if not group['x'] or set(group['labels']) != set(JUNCTION_MOTION_CLASSES):
            raise ValueError(f'Junction forecast lacks motion classes in {split}')
        group['x'], group['y'] = np.asarray(group['x'], dtype=np.float32), np.asarray(group['y'], dtype=np.float32)
    history_contract = add_junction_motion_history(groups, source, temporal_pdd) if temporal_pdd is not None else None
    if traffic_features and history_contract is None:
        raise ValueError('Traffic forecast experiment requires the existing causal history contract')
    train, dev = groups['TRAIN'], groups['DEV']
    reference_report = None
    if history_contract is not None:
        prior_path = ROOT / JUNCTION_NAV96_REPORT
        if digest(prior_path) != JUNCTION_NAV96_SHA256:
            raise ValueError('Expected the reviewed failed NAV96 junction forecast')
        prior = json.loads(prior_path.read_bytes())
        prior_model_path = prior_path.parent / 'model_bundle.joblib'
        if digest(prior_model_path) != prior['model_bundle_sha256']:
            raise ValueError('NAV96 forecast checkpoint checksum mismatch')
        if prior['feature_function_sha256'] != function_source_sha256(junction_forecast_features):
            raise ValueError('NAV96 reference feature helper changed')
        old = joblib.load(prior_model_path)
        old_prediction = np.maximum(0., dev['x'][:, :1]+old['speed_regressor'].predict(dev['x'][:, :96]))
        old_scores = junction_forecast_scores(dev['y'], old_prediction, dev['labels'])
        for name, scores in old_scores.items():
            for metric in ('mae_mps', 'one_second_mae_mps'):
                if not math.isclose(scores[metric], prior['dev_speed']['extra_trees'][name][metric], abs_tol=1e-8, rel_tol=0.):
                    raise ValueError('NAV96 reference does not reproduce its recorded DEV metrics')
        reference_report = {'path': str(prior_path), 'sha256': JUNCTION_NAV96_SHA256,
            'model_sha256': prior['model_bundle_sha256'], 'speed_scores': old_scores,
            'transition_macro_f1': prior['dev_transition']['extra_trees_macro_f1'], 'speed_prediction_replay': 'PASS'}
    history_reference = None
    if traffic_features:
        prior_path = ROOT / JUNCTION_HISTORY_REPORT
        if digest(prior_path) != JUNCTION_HISTORY_SHA256:
            raise ValueError('Expected the reviewed HISTORY102 junction forecast')
        prior = json.loads(prior_path.read_bytes())
        prior_model_path = prior_path.parent / 'model_bundle.joblib'
        if (digest(prior_model_path) != prior['model_bundle_sha256']
                or prior['history_function_sha256'] != function_source_sha256(causal_speed_history)):
            raise ValueError('HISTORY102 checkpoint/history codec changed')
        old = joblib.load(prior_model_path)
        old_prediction = np.maximum(0., dev['x'][:, :1]+predict_junction_speed_residuals(old['speed_regressor'], dev['x']))
        old_scores = junction_forecast_scores(dev['y'], old_prediction, dev['labels'])
        for name, scores in old_scores.items():
            for metric in ('mae_mps', 'one_second_mae_mps'):
                if not math.isclose(scores[metric], prior['dev_speed'][prior['selected_model']][name][metric], abs_tol=1e-8, rel_tol=0.):
                    raise ValueError('HISTORY102 speed metrics did not replay exactly')
        history_reference = {'path': str(prior_path), 'sha256': JUNCTION_HISTORY_SHA256,
            'model_sha256': prior['model_bundle_sha256'], 'speed_scores': old_scores, 'speed_prediction_replay': 'PASS'}
        for group in groups.values():
            derived = np.asarray([junction_traffic_features(row[:96]) for row in group['x'].tolist()], dtype=np.float32)
            group['x'] = np.column_stack((group['x'], derived))
    if error_review:
        if not traffic_features:
            raise ValueError('Error review requires the rejected TRAFFIC114 input contract')
        return review_junction_forecast_errors(out, groups)  # Return before any estimator fitting.
    if quantile_forecast:
        if not traffic_features:
            raise ValueError('Quantile forecasting requires the unchanged TRAFFIC114 input')
        return train_junction_quantile_forecast(out, groups)
    if stop_go or neural_stop_go or traffic_stop_go or clock_stop_go:
        if not traffic_features:
            raise ValueError('Stop/go training requires the unchanged TRAFFIC114 input')
        return train_junction_stop_go(out, groups, neural_gate=neural_stop_go,
            traffic_history_pdd=temporal_pdd if traffic_stop_go or clock_stop_go else None,
            source=source, stop_clock=clock_stop_go)
    regressor = ExtraTreesRegressor(n_estimators=96, min_samples_leaf=8, random_state=42, n_jobs=-1)
    classifier = ExtraTreesClassifier(n_estimators=96, min_samples_leaf=8, random_state=42, n_jobs=-1, class_weight='balanced')
    print(f"Training PDD junction motion forecast: TRAIN={len(train['y'])}, DEV={len(dev['y'])}, {train['x'].shape[1]} features.", flush=True)
    regressor.fit(train['x'], train['y']-train['x'][:, :1])
    classifier.fit(train['x'], train['labels'])
    # Avoid a parallel worker pool for each single-state runtime query later.
    regressor.n_jobs = classifier.n_jobs = 1
    regressors = {'extra_trees': regressor}
    candidate_settings = {'extra_trees': {'n_estimators': 96, 'min_samples_leaf': 8, 'criterion': 'squared_error'}}
    if history_contract is not None:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.multioutput import MultiOutputRegressor

        # An absolute-error objective learns conditional medians instead of
        # averaging the stationary and launching modes into a creeping speed.
        parameters = {'loss': 'absolute_error', 'max_iter': 160, 'max_leaf_nodes': 15,
                      'min_samples_leaf': 20, 'learning_rate': .08, 'l2_regularization': 1.,
                      'early_stopping': False, 'random_state': 42}
        robust = MultiOutputRegressor(HistGradientBoostingRegressor(**parameters), n_jobs=1)
        print('Fitting six absolute-error speed heads; TRAIN only, no random early-stopping split...', flush=True)
        robust.fit(train['x'], train['y']-train['x'][:, :1])
        regressors['absolute_error_boosting'] = robust
        candidate_settings['absolute_error_boosting'] = parameters
        if traffic_features:
            stopped = train['x'][:, 0] < .3
            if int(stopped.sum()) < 2*parameters['min_samples_leaf']:
                raise ValueError('Insufficient TRAIN near-stop examples for a learned specialist')
            specialist = MultiOutputRegressor(HistGradientBoostingRegressor(**parameters), n_jobs=1)
            print(f'Fitting learned near-stop expert on {int(stopped.sum())} TRAIN examples...', flush=True)
            specialist.fit(train['x'][stopped], train['y'][stopped]-train['x'][stopped, :1])
            regressors['near_stop_expert'] = {'kind': 'CURRENT_SPEED_EXPERTS_V1', 'threshold_mps': .3,
                'near_stop': specialist, 'moving': robust}
            candidate_settings['near_stop_expert'] = {'near_stop_parameters': parameters,
                'routing': 'current observed speed below 0.3 m/s; no target-based routing',
                'train_rows_near_stop': int(stopped.sum()), 'moving_expert': 'absolute_error_boosting'}
    persistence = np.repeat(dev['x'][:, :1], 6, axis=1)
    baseline_scores = junction_forecast_scores(dev['y'], persistence, dev['labels'])
    speed_scores = {name: junction_forecast_scores(dev['y'], np.maximum(0., dev['x'][:, :1]+predict_junction_speed_residuals(model, dev['x'])), dev['labels'])
                    for name, model in regressors.items()}
    predicted_labels = classifier.predict(dev['x'])
    baseline_labels = np.where(dev['x'][:, 0] < .3, 'WAIT', 'STEADY')
    classifier_f1 = float(f1_score(dev['labels'], predicted_labels, labels=list(JUNCTION_MOTION_CLASSES), average='macro', zero_division=0))
    baseline_f1 = float(f1_score(dev['labels'], baseline_labels, labels=list(JUNCTION_MOTION_CLASSES), average='macro', zero_division=0))
    unmet_candidates = {name: junction_forecast_unmet(scores, baseline_scores, classifier_f1, baseline_f1)
                        for name, scores in speed_scores.items()}
    eligible = [name for name in regressors if not unmet_candidates[name]]
    selected = min(eligible or list(regressors), key=lambda name: speed_scores[name]['all']['one_second_mae_mps'])
    regressor, model_scores, unmet = regressors[selected], speed_scores[selected], unmet_candidates[selected]
    status = 'FORECAST_DEV_IMPROVED' if not unmet else 'FORECAST_NEEDS_FIX'
    contract = {'schema': 'PDD_JUNCTION_MOTION_TRAFFIC_V3' if traffic_features else 'PDD_JUNCTION_MOTION_HISTORY_V2' if history_contract else 'PDD_JUNCTION_MOTION_NAV_V1',
        'scope': 'OFFLINE_HUMAN_MOTION_FORECAST_ONLY', 'selected_model': selected,
        'features': [*FEATURE_NAMES[:90], *PLAN_NAV_FEATURES, *(JUNCTION_HISTORY_NAMES if history_contract else []),
                     *(JUNCTION_TRAFFIC_NAMES if traffic_features else [])], 'horizons_s': source['horizons_s'],
        'output': 'six human speed residuals relative to current speed; motion class at +1 second',
        'motion_classes': list(JUNCTION_MOTION_CLASSES), 'offline_status': status,
        'source_report_sha256': JUNCTION_FORECAST_SOURCE_SHA256, 'candidates_sha256': source['candidates_sha256'],
        'feature_function_sha256': function_source_sha256(junction_forecast_features),
        'speed_decoder_sha256': function_source_sha256(predict_junction_speed_residuals),
        'frozen_trainer_sha256': digest(ROOT / 'train_pdd_baseline.py'),
        'encoder_sha256': source['frozen_encoder_sha256'], 'seed': 42,
        'policy_training_approved': False, 'deployment_eligible': False,
        'source_behavior_quality': 'UNREVIEWED', 'source_review_flags_preserved': True}
    if history_contract is not None:
        contract.update(history=history_contract, history_function_sha256=function_source_sha256(causal_speed_history),
                        nav96_reference=reference_report)
    if traffic_features:
        contract.update(traffic_function_sha256=function_source_sha256(junction_traffic_features),
            history102_reference=history_reference,
            traffic_feature_semantics='Ego-frame observed actors; current-lane aligned lead rear distance, kinematic crossing and stationary-origin distance. Not a safe gap or legal priority.')
    bundle_path = out / 'model_bundle.joblib'
    joblib.dump({**contract, 'speed_regressor': regressor, 'transition_classifier': classifier}, bundle_path)
    report = {**contract, 'training_performed': True, 'simulator_used': False, 'test_used': False,
        'source_report': str(source_path), 'source_labels': 'HUMAN_RECORDED_PDD_ONLY',
        'counts': {split: {'retained': len(group['y']), 'excluded_signal_windows': excluded[split],
            'motion_classes': dict(Counter(group['labels'])),
            'drivers': sorted({identity[0] for identity in group['identities']})} for split, group in groups.items()},
        'dev_speed': {'persistence': baseline_scores, **speed_scores},
        'dev_transition': {'persistence_macro_f1': baseline_f1, 'extra_trees_macro_f1': classifier_f1,
            'classification': classification_report(dev['labels'], predicted_labels, labels=list(JUNCTION_MOTION_CLASSES), output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(dev['labels'], predicted_labels, labels=list(JUNCTION_MOTION_CLASSES)).tolist()},
        'unmet_forecast_criteria': unmet, 'unmet_by_candidate': unmet_candidates, 'model_bundle_sha256': digest(bundle_path),
        'fitting': {'trees_per_extra_trees_head': 96, 'extra_trees_min_samples_leaf': 8, 'classifier_class_weight': 'balanced', 'extra_trees_fit_n_jobs': -1,
                    'predict_wrapper_n_jobs': 1, 'random_state': 42, 'sklearn_version': sklearn.__version__,
                    'speed_candidates': candidate_settings, 'selection': 'Eligible candidates first, then minimum all/+1s MAE on DEV'},
        'implementation_sha256': digest(__file__),
        'limitations': ['This experiment predicts observed human motion. It does not label legal right-of-way or approve unreviewed demonstrations as policy training data.',
            'None signal annotations are screened for absence of a recorded signal, not proof of green or an unsignalized junction.',
            'Only shared state/navigation, causal speed history and optional geometry derived from those observations enter the model. Junction/signal metadata and future context never enter features.',
            'No random row-level validation split, MetaDrive labels, held-out TEST data, runtime actions or Stage 1 overrides are used.',
            'FORECAST_DEV_IMPROVED is an offline forecasting result, not a driving or deployment approval. Existing curve06 remains unchanged.']}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for group in ('all', 'WAIT', 'LAUNCH', 'SLOWDOWN'):
        print(f"{group}: 1s speed MAE persistence={baseline_scores[group]['one_second_mae_mps']:.4f}, "
              f"learned={model_scores[group]['one_second_mae_mps']:.4f} m/s", flush=True)
    print(f"{status}, selected={selected}; transition macro F1={classifier_f1:.4f} vs {baseline_f1:.4f}. Read {out / 'report.json'}", flush=True)


CAPACITY_SHADOW_SOURCE_REPORT = 'agent4_integration_audit/run_20261121T030228_669283Z/report.json'
CAPACITY_SHADOW_SOURCE_SHA256 = 'e218e082064a25938657a9222c09bc29690d58b5e35afac2330e4cdb8a5ca6dc'


def capacity_shadow_observation(snapshot, observer_row):
    """A labelled partial N32 shadow, never an input for the checkpoint or actual driver."""
    from copy import deepcopy

    if observer_row['observer_motion_paths_status'] != 'NOT_RECORDED':
        raise ValueError('Unexpected wider motion-path recording contract')
    wider = observer_row['observer_traffic_context']
    validate_capacity_probe_prefix(snapshot['traffic_context'], wider)
    result = deepcopy(snapshot)
    result['traffic_context'] = deepcopy(wider)
    # The exact N8 prefix retains its observed alternatives; extra actors have
    # no recorded road alternatives. Do not invent alternatives or erase N8 ones.
    for field in ('motion_paths', 'motion_paths_overflow_count'):
        result['traffic_context'][field] = deepcopy(snapshot['traffic_context'][field])
    return result, wider['active_count']-snapshot['traffic_context']['active_count']


def run_traffic_capacity_shadow(project, out):
    """Paired perception sensitivity on an existing trajectory, with an exact N8 replay gate."""
    from scripts.replay_rule_based_trace import restore_input
    from src.agents.rule_based_agent import Stage1RuleBasedAgent
    from src.platform import agent_input as inputs
    from src.platform.agent import AgentPublicEpisodeContext
    from src.logging.agent_experiment import write_json

    source_path = ROOT / CAPACITY_SHADOW_SOURCE_REPORT
    if digest(source_path) != CAPACITY_SHADOW_SOURCE_SHA256:
        raise ValueError('Capacity shadow expected the reviewed passive observer report')
    source = json.loads(source_path.read_bytes())
    if (source['scope'] != 'N8_DRIVER_N32_PASSIVE_TRAFFIC_CAPACITY_PROBE'
            or source['driver_public_input_replay'] != 'PASS'
            or source['additional_information_passed_to_driver'] is not False
            or source['case_id'] != 'sandbox/Extreme/CrXROSTR/3/1'
            or source['checkpoint_sha256'] != PDDCurveFlow40Diagnostic.checkpoint_hash):
        raise ValueError('Unexpected capacity shadow evidence')
    if digest(project / 'src/platform/agent_input.py') != source['sdk_input_source_sha256']:
        raise ValueError('SDK input source changed since the capacity recording')
    run_dir = source_path.parent / 'launcher' / source['driver_policy']
    driver_path, observer_path = run_dir / 'agent_decisions.jsonl.gz', source_path.parent / 'observer_capacity32.jsonl.gz'
    for path, expected in ((driver_path, source['driver_trace_sha256']), (observer_path, source['observer_trace_sha256'])):
        if digest(path) != expected:
            raise ValueError(f'Capacity shadow trace checksum mismatch: {path}')
    provenance = json.loads((run_dir / 'agent_provenance.json').read_bytes())
    config = provenance['configuration']
    if hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest() != provenance['configuration_sha256']:
        raise ValueError('Recorded Stage 1 configuration checksum mismatch')
    reference, shadow = Stage1RuleBasedAgent(), Stage1RuleBasedAgent()
    for policy in (reference, shadow):
        if set(config['stage1']) != set(policy.export_config()):
            raise ValueError('Recorded Stage 1 configuration schema changed')
        for name, value in config['stage1'].items():
            setattr(policy, name, value)
        policy.reset(AgentPublicEpisodeContext(horizon_steps=int(source['driver_episode']['horizon_steps'])))
    counters, reference_stops, shadow_stops, cross_counts = Counter(), Counter(), Counter(), Counter()
    print('Offline N8 replay / partial N32 sensitivity on recorded inputs; no simulator or training.', flush=True)
    with gzip.open(driver_path, 'rt', encoding='utf-8') as driver_stream, gzip.open(observer_path, 'rt', encoding='utf-8') as observer_stream, gzip.open(out / 'capacity_shadow.jsonl.gz', 'xt', encoding='utf-8') as stream:
        for index, line in enumerate(driver_stream):
            row = json.loads(line)
            observer_line = observer_stream.readline()
            if not observer_line:
                raise ValueError('Capacity shadow observer trace ended early')
            observed = json.loads(observer_line)
            obs, recorded = row['agent_input'], row['decision']['diagnostics']
            public_hash = hashlib.sha256(json.dumps(obs, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            if (row['step_index'] != index or observed['step_index'] != index
                    or row['case_id'] != source['case_id'] or observed['case_id'] != source['case_id']
                    or public_hash != observed['public_input_sha256']):
                raise ValueError('Capacity shadow input alignment/checksum mismatch')
            wider, missing_paths = capacity_shadow_observation(obs, observed)
            original = reference.act(restore_input(obs, inputs))
            if (any(not math.isclose(float(a), float(b), abs_tol=1e-7, rel_tol=0.) for a, b in zip(
                    original.action_payload, (recorded['stage1_steer'], recorded['stage1_command'])))
                    or original.diagnostics.get('junction_phase') != recorded.get('junction_phase')
                    or original.diagnostics.get('junction_reason') != recorded.get('junction_reason')):
                raise ValueError(f'Capacity shadow N8 reference replay failed at step {index}')
            alternative = shadow.act(restore_input(wider, inputs))
            old, new = original.diagnostics, alternative.diagnostics
            motion = obs['ego_motion']
            stopped = motion is not None and math.hypot(motion['velocity_x_mps'], motion['velocity_y_mps']) < .3
            old_target, new_target = float(old['target_speed_kmh']), float(new['target_speed_kmh'])
            counters['frames'] += 1
            counters['actual_stopped_steps'] += int(stopped)
            counters['extra_actor_road_paths_unrecorded_steps'] += int(missing_paths > 0)
            counters['higher_shadow_target_steps'] += int(new_target > old_target+1e-7)
            counters['lower_shadow_target_steps'] += int(new_target < old_target-1e-7)
            counters['raw_reference_zero_shadow_positive_steps'] += int(old_target <= 1e-7 and new_target > 1e-7)
            if stopped:
                reference_stops[old['speed_limit_reason']+'/'+str(old.get('junction_reason'))] += 1
                shadow_stops[new['speed_limit_reason']+'/'+str(new.get('junction_reason'))] += 1
                cross_counts[old['speed_limit_reason']+' -> '+new['speed_limit_reason']] += 1
            stream.write(json.dumps({'step_index': index, 'actual_stopped': stopped,
                'reference_target_kmh': old_target, 'partial_shadow_target_kmh': new_target,
                'reference_reason': old['speed_limit_reason'], 'partial_shadow_reason': new['speed_limit_reason'],
                'reference_junction_reason': old.get('junction_reason'), 'partial_shadow_junction_reason': new.get('junction_reason'),
                'reference_ghost_tracks': old['ghost_track_count'], 'partial_shadow_ghost_tracks': new['ghost_track_count'],
                'extra_actors_without_recorded_road_paths': missing_paths,
                'permission_to_proceed_inferred': False}, allow_nan=False)+'\n')
        if observer_stream.readline() or counters['frames'] != source['verified_input_steps']:
            raise ValueError('Capacity shadow trace lengths differ')
    report = {'scope': 'RECORDED_N8_REFERENCE_N32_PARTIAL_PERCEPTION_SHADOW', 'status': 'SENSITIVITY_ONLY',
        'source_report': str(source_path), 'source_report_sha256': CAPACITY_SHADOW_SOURCE_SHA256,
        'case_id': source['case_id'], 'reference_replay': 'PASS', 'counts': dict(counters),
        'actual_stopped_reference_reasons': dict(reference_stops), 'actual_stopped_partial_shadow_reasons': dict(shadow_stops),
        'actual_stopped_limit_transitions': dict(cross_counts), 'actions_applied': False, 'simulator_used': False,
        'training_performed': False, 'deployment_eligible': False, 'permission_to_proceed_inferred': False,
        'checkpoint_sha256': source['checkpoint_sha256'], 'reviewer_sha256': digest(__file__),
        'shadow_trace_sha256': digest(out / 'capacity_shadow.jsonl.gz'),
        'live_reference_source_sha256': {path.name: digest(path) for path in sorted((project / 'src/agents').glob('rule_based*.py'))},
        'restore_input_source_sha256': digest(project / 'scripts/replay_rule_based_trace.py'),
        'limitations': ['Both stateful references see the recorded N8-driver trajectory, not their own hypothetical actions.',
            'N32 has current states but no recorded road alternatives for additional actors. Its shadow is partial.',
            'A higher target or a cleared hold does not establish a safe gap or reduced waiting in closed loop.',
            'No checkpoint is trained, no additional observation is delivered to the actual driver, no deployment is authorized.']}
    write_json(out / 'report.json', report, exclusive=True)
    print(f"N8 reference replay PASS: {counters['frames']} steps; partial shadow higher/lower target "
          f"{counters['higher_shadow_target_steps']}/{counters['lower_shadow_target_steps']} steps.", flush=True)
    print(f"SENSITIVITY_ONLY. No control/checkpoint changed. Read {out / 'report.json'}", flush=True)


def match_current_route_hints(paths, hints, tolerance_m=2.):
    """Match current planner hints to full local alternatives, preserving ambiguity."""
    if not math.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError('Invalid route-hint tolerance')
    used = [(float(x), float(y)) for x, y in hints if .5 <= math.hypot(x, y) <= 50.]
    if not used:
        return {'status': 'NO_HINT_WITHIN_50M', 'compatible_indices': [], 'hint_count': 0}
    compatible = []
    for index, path in enumerate(paths):
        previous, valid = -1., True
        for x, y in used:
            matches = []
            for first, second in zip(path, path[1:]):
                dx, dy = second[1]-first[1], second[2]-first[2]
                length2 = dx*dx+dy*dy
                if length2 <= 1e-10:
                    continue
                fraction = max(0., min(1., ((x-first[1])*dx+(y-first[2])*dy)/length2))
                along = first[0]+fraction*(second[0]-first[0])
                if along+.01 < previous:
                    continue
                matches.append((math.hypot(x-first[1]-fraction*dx, y-first[2]-fraction*dy), along))
            if not matches or min(matches)[0] > tolerance_m:
                valid = False
                break
            previous = min(matches)[1]
        if valid:
            compatible.append(index)
    return {'status': 'UNIQUE_GEOMETRIC_MATCH' if len(compatible) == 1 else
            'AMBIGUOUS' if compatible else 'NO_COMPATIBLE_PATH',
            'compatible_indices': compatible, 'hint_count': len(used), 'tolerance_m': tolerance_m}


def common_route_prefix(paths, tolerance_m=.01):
    """Keep only the identical sampled centerline before alternatives diverge."""
    if not paths:
        return []
    result = []
    for points in zip(*paths):
        reference = points[0]
        if any(abs(point[0]-reference[0]) > 1e-6 or
               math.hypot(point[1]-reference[1], point[2]-reference[2]) > tolerance_m or
               abs(math.atan2(math.sin(point[3]-reference[3]), math.cos(point[3]-reference[3]))) > 1e-3
               for point in points[1:]):
            break
        result.append(list(reference))
    return result


def encode_local_route_geometry(points):
    """Twenty 2.5m lookahead samples, ego-left axes, with explicit unknown padding."""
    result = []
    for index in range(1, 21):
        if index >= len(points):
            result.extend([0.]*5)
            continue
        distance, x, y, angle = map(float, points[index])
        if not all(math.isfinite(value) for value in (distance, x, y, angle)) or abs(distance-index*2.5) > 1e-6:
            raise ValueError('Expected finite route points on the shared 2.5m grid')
        result.extend([x/50., y/50., math.sin(angle), math.cos(angle), 1.])
    return result


def local_carla_route_candidates(waypoint, telemetry):
    from build_agent4_pdd_pilot import heading, xyz
    from agent4_pdd_common_v2 import project_carla
    origin, yaw = xyz(telemetry['location']), heading(telemetry)
    frontier, finished, expansions = [[waypoint]], [], 0
    while frontier:
        if len(frontier)+len(finished) > 32 or expansions >= 2000:
            return [], 'ENUMERATION_LIMIT'
        chain = frontier.pop()
        if len(chain) == 21:
            finished.append(chain)
            continue
        successors = chain[-1].next(2.5)
        expansions += 1
        if not successors:
            finished.append(chain)
        else:
            frontier.extend(chain+[successor] for successor in successors)
    paths, seen = [], set()
    for chain in finished:
        key = tuple((node.road_id, node.lane_id, round(node.s, 3)) for node in chain)
        if key in seen or len(chain) < 2:
            continue
        seen.add(key)
        points = []
        for index, node in enumerate(chain):
            point = node.transform.location
            x, y = project_carla(point.x-origin[0], point.y-origin[1], yaw)
            angle = yaw-math.radians(node.transform.rotation.yaw)
            points.append([index*2.5, x, y, math.atan2(math.sin(angle), math.cos(angle))])
        paths.append({'points': points, 'lane_chain': [list(value) for value in key]})
    return paths, 'COMPLETE_LOCAL_ENUMERATION'


def survey_pdd_junction_map(out, pdd, *, route_review=False):
    """Sample current junction poses and recorded road references; never infer a chosen route."""
    import xml.etree.ElementTree as ET
    from importlib.metadata import version
    from build_agent4_pdd_pilot import read_source, xyz, single_box, pdd_map_lane_pose

    source_path = ROOT/JUNCTION_FORECAST_SOURCE_REPORT
    if digest(source_path) != JUNCTION_FORECAST_SOURCE_SHA256:
        raise ValueError('Expected the frozen junction source report')
    source = json.loads(source_path.read_bytes())
    for name, checksum in (('build_agent4_pdd_pilot.py', source['frozen_builder_sha256']),
                           ('agent4_pdd_common_v2.py', source['frozen_encoder_sha256'])):
        if digest(ROOT/name) != checksum:
            raise ValueError(f'Frozen map-survey helper changed: {name}')
    candidates = source_path.parent/'junction_candidates.jsonl'
    map_path = ROOT/'runs/agent4_pdd_town12_map_01/Town12.xodr'
    if (digest(candidates) != source['candidates_sha256'] or
            digest(map_path) != '26fa53d4902ac3397d35f948969f0c77c1bf023755ede2c67edf00bef817151f'):
        raise ValueError('Junction source/map checksum mismatch')
    if version('carla') != '0.9.15':
        raise ValueError('Expected installed CARLA Python API0.9.15; no server is needed')
    import carla
    raw_map = map_path.read_text(encoding='utf-8-sig')
    tree = ET.fromstring(raw_map)
    roads = {int(node.attrib['id']): node for node in tree.findall('road')}
    junctions = {int(node.attrib['id']): node for node in tree.findall('junction')}
    road_map = carla.Map('Town12', raw_map)
    parent_hash = None
    if route_review:
        parent_path = ROOT/'agent4_integration_audit/run_20261122T084711_438186Z/report.json'
        parent_hash = '23cb028fe261915cdb9a80219030e4ccfb6586226308299c142aa15ee1a45178'
        if digest(parent_path) != parent_hash:
            raise ValueError('Expected the completed junction map survey')
    anchors = {}
    with candidates.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['partition'] != 'TRAIN':
                continue
            driver = int(row['driver'].removeprefix('driver_'))
            if not 1 <= driver <= 21:
                raise ValueError('Unexpected TRAIN driver in context survey')
            key = (row['driver'], row['scenario'], row['source_context_for_parity_review']['is_in_junction'])
            anchors.setdefault(key, row)  # First exported anchor per episode/inside-or-approach phase.
    receipts = {row['episode']: row for row in source['episodes']}
    pdd, verified, rows, counts = pdd.resolve(), {}, [], Counter()
    cached_episode, cached_telemetry = None, None
    for (driver, scene, inside), row in sorted(anchors.items()):
        receipt = receipts[f'{driver}/{scene}']
        if receipt['partition'] != 'TRAIN':
            raise ValueError('Source partition mismatch')
        def load(relative):
            inventory = {}
            result = read_source(pdd/relative, inventory, pdd)
            if inventory[relative] != receipt['source_sha256'][relative]:
                raise ValueError(f'Context survey source changed: {relative}')
            verified[relative] = inventory[relative]
            return result
        episode = (driver, scene)
        if episode != cached_episode:
            cached_telemetry = load(f'{driver}/data/{scene}/metric/metric_info.json')
            cached_episode = episode
        tick = row['telemetry_step']
        telemetry = cached_telemetry[str(tick)]
        boxes = load(f'{driver}/data/{scene}/boxes/{tick//5}.json.gz')
        info = single_box(boxes, 'ego_info')
        waypoint = road_map.get_waypoint(carla.Location(*xyz(telemetry['location'])),
                                         project_to_road=True, lane_type=carla.LaneType.Driving)
        pose, flags, match = pdd_map_lane_pose(telemetry, info, waypoint)
        flags = list(flags)
        if bool(info['is_in_junction']) != inside:
            raise ValueError('Current junction annotation differs from candidate')
        recorded_junction = int(info['junction_id'] if inside else info['next_junction_id'])
        junction = junctions.get(recorded_junction)
        connections = [] if junction is None else [
            {'incoming_road': int(node.attrib['incomingRoad']), 'connecting_road': int(node.attrib['connectingRoad'])}
            for node in junction.findall('connection')]
        if junction is None:
            flags.append('JUNCTION_ID_NOT_IN_MAP')
        if inside and waypoint is not None:
            actual = waypoint.get_junction() if waypoint.is_junction else None
            if actual is None or actual.id != recorded_junction:
                flags.append('CURRENT_JUNCTION_ID_MISMATCH')
        references = {key: info[key] for key in ('next_road_ids', 'next_next_road_ids_ego')}
        for key, ids in references.items():
            if not isinstance(ids, list) or any(type(value) is not int for value in ids):
                raise ValueError(f'Unexpected recorded road references: {key}')
            if any(value not in roads for value in ids):
                flags.append(f'{key.upper()}_NOT_IN_MAP')
        counts['anchors'] += 1
        counts['inside' if inside else 'approach'] += 1
        counts['pose_matched'] += int(pose is not None and not flags)
        counts.update(flags)
        route = None
        if route_review:
            if flags or waypoint is None:
                route = {'status': 'INVALID_CURRENT_POSE', 'selected_path': None}
            else:
                alternatives, enumeration = local_carla_route_candidates(waypoint, telemetry)
                hints = [(float(telemetry[key][0]), -float(telemetry[key][1])) for key in ('target_point', 'target_point_next')]
                matched = match_current_route_hints([path['points'] for path in alternatives], hints)
                selected = (alternatives[matched['compatible_indices'][0]] if enumeration == 'COMPLETE_LOCAL_ENUMERATION'
                            and matched['status'] == 'UNIQUE_GEOMETRIC_MATCH' else None)
                route = {**matched, 'enumeration': enumeration, 'status': matched['status'] if enumeration ==
                    'COMPLETE_LOCAL_ENUMERATION' else enumeration, 'alternative_count': len(alternatives),
                    'selected_path': selected, 'current_hints_left_m': hints}
                usable = ([alternatives[index]['points'] for index in matched['compatible_indices']]
                          if matched['compatible_indices'] else
                          [path['points'] for path in alternatives] if matched['status'] == 'NO_HINT_WITHIN_50M' else [])
                prefix = common_route_prefix(usable) if enumeration == 'COMPLETE_LOCAL_ENUMERATION' else []
                route['known_geometry_prefix'] = prefix
                route['geometry_features_100d'] = encode_local_route_geometry(prefix)
                route['known_lookahead_samples'] = max(0, len(prefix)-1)
                counts['route/with_known_prefix'] += int(len(prefix) >= 2)
            route.setdefault('known_geometry_prefix', [])
            route.setdefault('geometry_features_100d', [0.]*100)
            route.setdefault('known_lookahead_samples', 0)
            counts['route/'+route['status']] += 1
        rows.append({'identity': [driver, scene, tick], 'inside': inside, 'flags': flags,
            'pose_candidate': pose, 'map_match': match, 'recorded_road_id': info['road_id'],
            'recorded_lane_id': info['lane_id'], 'recorded_junction_id': recorded_junction,
            'recorded_distance_to_junction_m': info['distance_to_junction'],
            'recorded_road_references': references, 'map_connections': connections,
            'next_roads_are_connecting_roads': bool(references['next_road_ids']) and
                set(references['next_road_ids']) <= {connection['connecting_road'] for connection in connections},
            'chosen_branch': None, 'route_passage': None, 'actor_paths': None,
            'route_geometry_review': route,
            'context_training_eligible': False})
    result = {'schema': 'PDD_JUNCTION_ROUTE_REVIEW_V3' if route_review else 'PDD_JUNCTION_MAP_SURVEY_V1',
        'status': 'MAPPING_REVIEW_REQUIRED', 'parent_survey_sha256': parent_hash,
        'sampling': 'First exported candidate per TRAIN episode and inside/approach phase; not all frames',
        'counts': dict(counts), 'rows': rows, 'verified_source_files': verified,
        'source_report_sha256': JUNCTION_FORECAST_SOURCE_SHA256, 'map_sha256': digest(map_path),
        'training_performed': False, 'simulator_used': False, 'test_used': False,
        'deployment_eligible': False, 'implementation_sha256': digest(__file__),
        'limitations': ['Existing C0 lane matching does not validate junction frames; this is a new sampled survey.',
            'Road ID membership and junction connections do not establish a chosen route or public-context parity.',
            'PDD distance_to_junction is recorded for review, not assumed equivalent to signed route passage entry.',
            'No future human pose, controls, or scenario-derived intention is used to select a branch.',
            'Actor paths, passage exit/yield/conflict boundaries and legal priority remain unknown.']}
    if route_review:
        result['limitations'].append('A unique geometric hint match is a route hypothesis, not verified intent or public passage parity; enumeration limits remain unknown.')
    (out/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f"MAPPING_REVIEW_REQUIRED: {dict(counts)}. No training. Read {out/'report.json'}", flush=True)


ROUTE_DATA_SCHEMA = 'PDD_ROUTE196_V1'


def route196_feature_names():
    from agent4_pdd_common_v2 import FEATURE_NAMES
    from train_pdd_baseline import PLAN_NAV_FEATURES
    return [*FEATURE_NAMES[:90], *PLAN_NAV_FEATURES,
            *[f'route_{index}_{field}' for index in range(1, 21)
              for field in ('x_over_50m', 'y_left_over_50m', 'heading_sin', 'heading_cos', 'valid')]]


def encode_route196(base96, points):
    from agent4_pdd_common_v2 import float32
    return float32(junction_forecast_features(base96)+encode_local_route_geometry(points))


def public_route196(snapshot):
    """Inference adapter using only the public snapshot; no simulator calls or history."""
    from agent4_pdd_common_v2 import encode_public, number
    if hasattr(snapshot, 'to_dict'):
        snapshot = snapshot.to_dict()
    base = encode_public(snapshot)  # Validates the existing N8, frame and route-grid contract.
    task = snapshot['task_context']
    points = [[0., 0., 0., 0.]]
    for index, (row, valid) in enumerate(zip(task['waypoints'], task['validity_mask']), 1):
        if not valid:
            break  # A gap never becomes a known straight segment.
        points.append([index*2.5, number(row['relative_x']), number(row['relative_y']),
                       number(row['relative_heading'])])
    return encode_route196(base, points)


def current_pdd_route_prefix(waypoint, telemetry, invalid_pose=False):
    if waypoint is None or invalid_pose:
        return [], 'INVALID_CURRENT_POSE'
    alternatives, enumeration = local_carla_route_candidates(waypoint, telemetry)
    if enumeration != 'COMPLETE_LOCAL_ENUMERATION':
        return [], enumeration
    hints = [(float(telemetry[key][0]), -float(telemetry[key][1])) for key in ('target_point', 'target_point_next')]
    match = match_current_route_hints([path['points'] for path in alternatives], hints)
    compatible = ([alternatives[index]['points'] for index in match['compatible_indices']]
                  if match['compatible_indices'] else
                  [path['points'] for path in alternatives] if match['status'] == 'NO_HINT_WITHIN_50M' else [])
    return common_route_prefix(compatible), match['status']


def load_route196_rows(path, contract):
    """Actual training loader: enforce population, driver split, feature masks and receipt."""
    import numpy as np
    from build_agent4_pdd_pilot import partition
    if contract['schema'] != ROUTE_DATA_SCHEMA or digest(path) != contract['dataset_sha256']:
        raise ValueError('Route dataset schema/checksum mismatch')
    groups = {split: {'x': [], 'y': [], 'labels': [], 'identities': []} for split in ('TRAIN', 'DEV')}
    seen = set()
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            driver, scene, tick = row['identity']
            identity = (driver, scene, tick)
            index = int(driver.removeprefix('driver_'))
            if (driver != f'driver_{index:02d}' or not 1 <= index <= 25 or identity in seen
                    or type(tick) is not int or tick < 0 or tick % 5
                    or row['partition'] != partition(index) or row['partition'] not in groups
                    or row['label_source'] != 'HUMAN_RECORDED' or row['behavior_quality'] != 'UNREVIEWED'):
                raise ValueError('Route dataset identity/split/label contract failed')
            seen.add(identity)
            x, y = np.asarray(row['features'], dtype=np.float32), np.asarray(row['speed_targets_mps'], dtype=np.float32)
            if x.shape != (196,) or y.shape != (6,) or not np.isfinite(x).all() or not np.isfinite(y).all() or (y < 0).any():
                raise ValueError('Route dataset shape/finite-target contract failed')
            masked = False
            for point in x[96:].reshape(20, 5):
                if point[4] not in (0., 1.) or (point[4] == 0 and point[:4].any()) or (masked and point[4]):
                    raise ValueError('Invalid route padding or noncontiguous mask')
                masked |= point[4] == 0
                if point[4] and not math.isclose(float(point[2]**2+point[3]**2), 1., abs_tol=1e-5):
                    raise ValueError('Invalid route heading normalization')
            current_speed = float(row['label_current_speed_mps'])
            if not math.isfinite(current_speed) or current_speed < 0 or not math.isclose(current_speed, float(x[0]), abs_tol=1e-5, rel_tol=1e-6):
                raise ValueError('Original label current speed differs from encoded state')
            future_speed = float(y[3])
            label = ('WAIT' if current_speed < .3 and future_speed < .3 else 'LAUNCH' if current_speed < .3 else
                     'SLOWDOWN' if future_speed < current_speed-.5 else 'SPEEDUP' if future_speed > current_speed+.5 else 'STEADY')
            if row['motion_label'] != label:
                raise ValueError('Route dataset motion label did not replay')
            group = groups[row['partition']]
            for key, value in (('x', x), ('y', y), ('labels', row['motion_label']), ('identities', identity)):
                group[key].append(value)
    for split, group in groups.items():
        if len(group['x']) != contract['counts'][split]['rows']:
            raise ValueError('Route dataset population mismatch')
        group['x'] = np.asarray(group['x'], dtype=np.float32).reshape(-1, 196)
        group['y'] = np.asarray(group['y'], dtype=np.float32).reshape(-1, 6)
    return groups


def replay_public_route196():
    """Recorded TRAIN inputs verify executable inference conversion, never supply training rows."""
    import numpy as np
    runtime = ROOT/'agent4_integration_audit/run_20261120T135335_843752Z'
    root = json.loads((runtime/'report.json').read_bytes())
    receipts = []
    for pair in root['pairs']:
        folder = runtime/'_'.join(pair['case_id'].split('/')[1:])
        detail = json.loads((folder/'report.json').read_bytes())
        agent = detail['agents'][root['policy']]
        trace = folder/'launcher'/root['policy']/'agent_decisions.jsonl.gz'
        if agent['episode']['split'] != 'TRAIN' or digest(trace) != agent['trace_sha256']:
            raise ValueError('Recorded public TRAIN input integrity mismatch')
        count = 0
        with gzip.open(trace, 'rt', encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                features = np.asarray(public_route196(row['agent_input']))
                if features.shape != (196,) or not np.isfinite(features).all():
                    raise ValueError('Public inference adapter returned invalid features')
                count += 1
        if not count:
            raise ValueError('Empty public replay trace')
        receipts.append({'case_id': pair['case_id'], 'frames': count, 'sha256': digest(trace)})
    return {'status': 'STRUCTURAL_REPLAY_PASS', 'traces': receipts, 'used_as_training_data': False,
            'semantic_route_intent_parity_approved': False}


def export_route196_dataset(out, pdd):
    """One full export + real training-loader replay + recorded public inference replay."""
    from importlib.metadata import version
    import numpy as np
    from build_agent4_pdd_pilot import read_source, xyz, single_box, paired_state, pdd_map_lane_pose
    from train_pdd_baseline import function_source_sha256
    source_path = ROOT/JUNCTION_FORECAST_SOURCE_REPORT
    if digest(source_path) != JUNCTION_FORECAST_SOURCE_SHA256:
        raise ValueError('Expected the frozen PDD junction population')
    source = json.loads(source_path.read_bytes())
    for name, checksum in (('build_agent4_pdd_pilot.py', source['frozen_builder_sha256']),
                           ('agent4_pdd_common_v2.py', source['frozen_encoder_sha256'])):
        if digest(ROOT/name) != checksum:
            raise ValueError('Frozen route-export helper changed')
    candidates = source_path.parent/'junction_candidates.jsonl'
    map_path = ROOT/'runs/agent4_pdd_town12_map_01/Town12.xodr'
    if digest(candidates) != source['candidates_sha256'] or digest(map_path) != '26fa53d4902ac3397d35f948969f0c77c1bf023755ede2c67edf00bef817151f':
        raise ValueError('Route export source/map changed')
    if version('carla') != '0.9.15':
        raise ValueError('Expected CARLA0.9.15 Python API')
    # Fail quickly on real public inputs before the expensive map export.
    replay = replay_public_route196()
    import carla
    road_map = carla.Map('Town12', map_path.read_text(encoding='utf-8-sig'))
    pdd = pdd.resolve()
    receipts = {row['episode']: row for row in source['episodes']}
    counts = {split: Counter() for split in ('TRAIN', 'DEV')}
    verified, cached_episode, telemetry = {}, None, None
    dataset = out/'route196_training.jsonl.gz'
    with candidates.open(encoding='utf-8') as stream, gzip.open(dataset, 'xt', encoding='utf-8') as target:
        for line in stream:
            row = json.loads(line)
            split, driver, scene, tick = row['partition'], row['driver'], row['scenario'], row['telemetry_step']
            if split not in counts:
                raise ValueError('Unexpected TEST/other split in route source')
            example = junction_forecast_example(row)
            if example is None:
                counts[split]['signal_screened'] += 1
                continue
            _, speeds, label = example
            receipt = receipts[f'{driver}/{scene}']
            if receipt['partition'] != split:
                raise ValueError('Route source partition mismatch')
            def load(relative):
                inventory = {}
                value = read_source(pdd/relative, inventory, pdd)
                if inventory[relative] != receipt['source_sha256'][relative]:
                    raise ValueError(f'Route source checksum changed: {relative}')
                verified[relative] = inventory[relative]
                return value
            if cached_episode != (driver, scene):
                telemetry = load(f'{driver}/data/{scene}/metric/metric_info.json')
                cached_episode = (driver, scene)
            current = telemetry[str(tick)]
            boxes = load(f'{driver}/data/{scene}/boxes/{tick//5}.json.gz')
            geometry = set()
            actual, _, _, actual_speed, _, _ = paired_state(current, boxes, geometry_flags=geometry)
            if not np.allclose(actual, row['features'], atol=1e-6, rtol=1e-6):
                raise ValueError('Current PDD196 base input did not replay')
            waypoint = road_map.get_waypoint(carla.Location(*xyz(current['location'])),
                                             project_to_road=True, lane_type=carla.LaneType.Driving)
            _, flags, _ = pdd_map_lane_pose(current, single_box(boxes, 'ego_info'), waypoint)
            prefix, status = current_pdd_route_prefix(waypoint, current, bool(flags or geometry))
            features = encode_route196(row['features'], prefix)
            target.write(json.dumps({'identity': [driver, scene, tick], 'partition': split,
                'features': features, 'speed_targets_mps': speeds, 'human_trajectory': row['human_trajectory'],
                'label_current_speed_mps': actual_speed,
                'motion_label': label, 'label_source': 'HUMAN_RECORDED', 'behavior_quality': 'UNREVIEWED',
                'route_status': status, 'route_pose_flags': list(flags), 'known_route_samples': max(0, len(prefix)-1)},
                allow_nan=False)+'\n')
            counts[split]['rows'] += 1
            counts[split]['route/'+status] += 1
            counts[split]['with_route_prefix'] += int(len(prefix) > 1)
            if sum(group['rows'] for group in counts.values()) % 250 == 0:
                print(f'Exported route196: TRAIN={counts["TRAIN"]["rows"]}, DEV={counts["DEV"]["rows"]}', flush=True)
    if {split: group['rows'] for split, group in counts.items()} != {'TRAIN': 10900, 'DEV': 1633}:
        raise ValueError('Route export changed the retained population')
    contract = {'schema': ROUTE_DATA_SCHEMA, 'dataset_sha256': digest(dataset),
        'feature_names': route196_feature_names(), 'counts': {split: dict(group) for split, group in counts.items()}}
    loaded = load_route196_rows(dataset, contract)
    report = {**contract, 'status': 'EXPORTED_AND_STRUCTURALLY_VERIFIED',
        'training_loader_shapes': {split: list(group['x'].shape) for split, group in loaded.items()},
        'public_adapter_replay': replay, 'source_report_sha256': JUNCTION_FORECAST_SOURCE_SHA256,
        'map_sha256': digest(map_path), 'verified_source_files': verified,
        'encoder_sha256': function_source_sha256(encode_route196),
        'public_adapter_sha256': function_source_sha256(public_route196),
        'route_builder_sha256': function_source_sha256(current_pdd_route_prefix),
        'function_bindings': {function.__name__: function_source_sha256(function) for function in (
            encode_route196, public_route196, encode_local_route_geometry, common_route_prefix,
            match_current_route_hints, local_carla_route_candidates, current_pdd_route_prefix,
            junction_forecast_features, load_route196_rows)},
        'training_performed': False, 'simulator_used': False, 'test_used': False,
        'policy_training_approved': False, 'deployment_eligible': False,
        'limitations': ['No bad/ambiguous route row is dropped; its geometry is explicitly masked.',
            'Structural encoding parity does not prove PDD hint hypotheses equal the declared launcher route.',
            'This is a human-motion research dataset, not reviewed legal priority/safe-go supervision.',
            'No actor path/traffic capacity extension, runtime controller registration, or driving improvement is claimed.',
            'No public/MetaDrive sample enters the training dataset.']}
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f'EXPORTED_AND_STRUCTURALLY_VERIFIED. Read {out/"report.json"}', flush=True)


ROUTE196_DATA_REPORT = 'agent4_integration_audit/run_20261122T100956_914876Z/report.json'
ROUTE196_DATA_SHA256 = '2508d5f432a23262e2885cfa6e6332a50eda84b894c2b6588e65fb46b2ec9639'


def route196_reference(features):
    import numpy as np
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] < 3 or not np.isfinite(features).all():
        raise ValueError('Invalid route plan observations')
    horizons = np.arange(1, 7, dtype=float)*.25
    result = np.zeros((len(features), 6, 3))
    result[:, :, 0] = features[:, 1:2]*horizons
    result[:, :, 1] = features[:, 2:3]*horizons
    result[:, :, 2] = features[:, :1]
    return result


def decode_route196_plan(estimator, features):
    import numpy as np
    reference = route196_reference(features)
    residual = np.asarray(estimator.predict(features), dtype=float)
    if residual.shape != (len(reference), 18) or not np.isfinite(residual).all():
        raise ValueError('Invalid learned route plan output')
    plan = reference+residual.reshape(-1, 6, 3)
    plan[:, :, 2] = np.maximum(0., plan[:, :, 2])
    return plan


def predict_public_route196_plan(bundle, snapshot):
    """Ready inference entry point; returns six ego-left x/y/speed points, not actuator commands."""
    import numpy as np
    from train_pdd_baseline import function_source_sha256
    if bundle['schema'] != 'PDD_ROUTE196_PLAN_V1':
        raise ValueError('Unknown route196 checkpoint schema')
    for name, expected in bundle['frozen_inference_files'].items():
        if digest(ROOT/name) != expected:
            raise ValueError(f'Route196 frozen inference helper changed: {name}')
    for name, expected in bundle['inference_bindings'].items():
        if function_source_sha256(globals()[name]) != expected:
            raise ValueError(f'Route196 inference codec changed: {name}')
    features = np.asarray(public_route196(snapshot), dtype=np.float32).reshape(1, 196)
    return decode_route196_plan(bundle['plan_model'], features)[0].tolist()


def route196_plan_metrics(truth, predicted, labels):
    import numpy as np
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    if truth.shape != predicted.shape or truth.shape[1:] != (6, 3) or not np.isfinite(predicted).all():
        raise ValueError('Invalid route196 plan metrics input')
    return {'ade_m': float(np.linalg.norm(predicted[:, :, :2]-truth[:, :, :2], axis=2).mean()),
        'lateral_mae_m': float(np.abs(predicted[:, :, 1]-truth[:, :, 1]).mean()),
        'speed': junction_forecast_scores(truth[:, :, 2], predicted[:, :, 2], labels),
        'stop_launch_tradeoff': junction_stop_launch_tradeoff(labels, predicted[:, :, 2])}


def route196_plan_unmet(candidate, reference96, persistence):
    """Fixed development gates, declared before fitting; no threshold/target tuning."""
    unmet = []
    for metric in ('ade_m', 'lateral_mae_m'):
        if candidate[metric] > reference96[metric]*1.01:
            unmet.append('reference96/'+metric)
    if candidate['ade_m'] >= persistence['ade_m']:
        unmet.append('persistence/ade_m')
    for group in ('all', 'LAUNCH', 'SLOWDOWN'):
        if candidate['speed'][group]['one_second_mae_mps'] >= persistence['speed'][group]['one_second_mae_mps']:
            unmet.append('persistence/'+group+'/speed_1s')
    if candidate['speed']['WAIT']['one_second_mae_mps'] > persistence['speed']['WAIT']['one_second_mae_mps']+.1:
        unmet.append('WAIT/speed_regression_over_0p1')
    tradeoff = candidate['stop_launch_tradeoff']
    if not tradeoff['WAIT_count'] or not tradeoff['LAUNCH_count']:
        raise ValueError('Both motion classes are required for route plan qualification')
    if tradeoff['WAIT_predicted_speed_ge_0p3_count']/tradeoff['WAIT_count'] > STOP_GO_PROTOCOL['maximum_WAIT_false_launch_rate']:
        unmet.append('WAIT/false_motion_over_0p05')
    if 1.-tradeoff['LAUNCH_predicted_speed_lt_0p3_count']/tradeoff['LAUNCH_count'] < STOP_GO_PROTOCOL['minimum_LAUNCH_recall']:
        unmet.append('LAUNCH/recall_below_0p5')
    if candidate['speed']['LAUNCH']['one_second_mae_mps'] > .9*persistence['speed']['LAUNCH']['one_second_mae_mps']:
        unmet.append('LAUNCH/mae_gain_below_0p10')
    return unmet


def route_frame_reference(features):
    """Route centerline reference with CV fallback beyond observed prefix, no invented branch."""
    import numpy as np
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 196:
        raise ValueError('Route-frame reference requires196D')
    result = route196_reference(features)
    horizons = np.arange(1, 7)*.25
    for index, row in enumerate(features):
        geometry = row[96:].reshape(20, 5)
        valid = 0
        for point in geometry:
            if point[4] == 0:
                break
            if point[4] != 1:
                raise ValueError('Invalid route-frame mask')
            valid += 1
        if not valid:
            continue
        points = geometry[:valid]
        heading = math.atan2(float(points[0, 2]), float(points[0, 3]))
        x = np.r_[points[0, 0]*50.-2.5*math.cos(heading), points[:, 0]*50.]
        y = np.r_[points[0, 1]*50.-2.5*math.sin(heading), points[:, 1]*50.]
        arc = np.arange(valid+1)*2.5
        queries = float(row[0])*horizons
        known = (queries >= 0.) & (queries <= arc[-1])
        result[index, known, 0] = np.interp(queries[known], arc, x)
        result[index, known, 1] = np.interp(queries[known], arc, y)
    return result


def decode_route_frame_plan(estimator, features):
    import numpy as np
    reference = route_frame_reference(features)
    residual = np.asarray(estimator.predict(features), dtype=float)
    if residual.shape != (len(reference), 18) or not np.isfinite(residual).all():
        raise ValueError('Invalid route-frame residuals')
    result = reference+residual.reshape(-1, 6, 3)
    result[:, :, 2] = np.maximum(0., result[:, :, 2])
    return result


def predict_public_route_frame_plan(bundle, snapshot):
    import numpy as np
    from train_pdd_baseline import function_source_sha256
    if bundle['schema'] != 'PDD_ROUTE196_FRAME_PLAN_V2':
        raise ValueError('Unknown route-frame checkpoint schema')
    for name, expected in bundle['frozen_inference_files'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Route-frame frozen file changed')
    for name, expected in bundle['inference_bindings'].items():
        if function_source_sha256(globals()[name]) != expected:
            raise ValueError('Route-frame codec changed')
    return decode_route_frame_plan(bundle['plan_model'], np.asarray([public_route196(snapshot)], dtype=np.float32))[0].tolist()


def fit_route_frame_plan(out, source, groups, plans, frozen):
    import joblib
    import numpy as np
    from sklearn.ensemble import ExtraTreesRegressor
    from train_pdd_baseline import function_source_sha256
    path = ROOT/ROUTE196_PLAN_REPORT
    if digest(path) != ROUTE196_PLAN_SHA256:
        raise ValueError('Expected frozen parent196 report')
    parent = json.loads(path.read_bytes())
    if digest(path.parent/'model_bundle.joblib') != parent['model_bundle_sha256']:
        raise ValueError('Parent196 checkpoint changed')
    bundle = joblib.load(path.parent/'model_bundle.joblib')
    if parent['dataset_sha256'] != source['dataset_sha256']:
        raise ValueError('Route-reference candidate must use the same parent dataset')
    for name, expected in bundle['inference_bindings'].items():
        if function_source_sha256(globals()[name]) != expected:
            raise ValueError('Parent196 codec changed')
    train, dev = groups['TRAIN'], groups['DEV']
    replay = route196_plan_metrics(plans['DEV'], decode_route196_plan(bundle['plan_model'], dev['x']), dev['labels'])
    if not math.isclose(replay['ade_m'], parent['dev_metrics']['route196']['ade_m'], abs_tol=1e-8):
        raise ValueError('Parent196 prediction replay failed')
    params = {'n_estimators': 96, 'min_samples_leaf': 8, 'random_state': 42, 'n_jobs': -1}
    targets = (plans['TRAIN']-route_frame_reference(train['x'])).reshape(-1, 18)
    print('Fitting route-frame residuals on unchanged10900PDD TRAIN rows; same estimator/labels...', flush=True)
    model = ExtraTreesRegressor(**params).fit(train['x'], targets)
    model.n_jobs = 1
    scores = {'route_frame': route196_plan_metrics(plans['DEV'], decode_route_frame_plan(model, dev['x']), dev['labels']),
              'parent196': replay, 'reference96': parent['dev_metrics']['reference96'], 'persistence': parent['dev_metrics']['persistence']}
    unmet = route196_plan_unmet(scores['route_frame'], scores['reference96'], scores['persistence'])
    for metric in ('ade_m', 'lateral_mae_m'):
        if scores['route_frame'][metric] > replay[metric]*1.01:
            unmet.append('parent196/'+metric)
    contract = {'schema': 'PDD_ROUTE196_FRAME_PLAN_V2', 'offline_status': 'DEV_NEEDS_FIX' if unmet else 'DEV_IMPROVED',
        'source_behavior_quality': 'UNREVIEWED', 'output': '6x[x_forward_m,y_left_m,speed_mps]',
        'source_report_sha256': ROUTE196_DATA_SHA256, 'dataset_sha256': source['dataset_sha256'],
        'feature_names': source['feature_names'], 'horizons_s': [.25, .5, .75, 1., 1.25, 1.5],
        'frozen_inference_files': frozen, 'deployment_eligible': False, 'policy_training_approved': False,
        'reference': 'Map prefix at current speed*horizon; first center extrapolated2.5m from first tangent; CV outside known prefix',
        'inference_bindings': {function.__name__: function_source_sha256(function) for function in (
            public_route196, encode_route196, encode_local_route_geometry, junction_forecast_features,
            route196_reference, route_frame_reference, decode_route_frame_plan, predict_public_route_frame_plan)}}
    model_path = out/'model_bundle.joblib'
    joblib.dump({**contract, 'plan_model': model}, model_path)
    report = {**contract, 'training_performed': True, 'simulator_used': False, 'test_used': False,
        'counts': source['counts'], 'fitting': params, 'dev_metrics': scores, 'unmet_criteria': unmet,
        'parent_report_sha256': ROUTE196_PLAN_SHA256, 'parent_prediction_replay': 'PASS',
        'model_bundle_sha256': digest(model_path), 'implementation_sha256': digest(__file__),
        'limitations': ['Route reference is a decoder/target-normalization change, not new demonstrations or verified right-of-way.',
            'Unknown/out-of-prefix geometry uses the unchanged constant-velocity reference, not an invented road.',
            'Offline WAIT gates remain unchanged; guarded runtime still depends onStage1. No deployment approval.']}
    report_path = out/'report.json'
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f'{contract["offline_status"]}: {unmet}. Read {report_path}', flush=True)
    return report_path


def train_route196_plan(out, *, route_frame=False):
    import joblib
    import numpy as np
    import sklearn
    from sklearn.ensemble import ExtraTreesRegressor
    from train_pdd_baseline import function_source_sha256
    path = ROOT/ROUTE196_DATA_REPORT
    if digest(path) != ROUTE196_DATA_SHA256:
        raise ValueError('Expected the completed full route196 dataset report')
    source = json.loads(path.read_bytes())
    if (source['status'] != 'EXPORTED_AND_STRUCTURALLY_VERIFIED' or source['feature_names'] != route196_feature_names()
            or source['public_adapter_replay']['status'] != 'STRUCTURAL_REPLAY_PASS'
            or source['public_adapter_replay']['used_as_training_data']):
        raise ValueError('Route196 data/adapter contract failed')
    frozen = {'agent4_pdd_common_v2.py': 'cda499ff1c47fd23fef5a6b90eb26d4ebf29aab7badd80007137f4523b5eacc9',
              'train_pdd_baseline.py': '22c6a97d22b0b4ad239a16bbe22d5a1525f7f29b18d991d92ad6cf0c21454ea8'}
    if any(digest(ROOT/name) != expected for name, expected in frozen.items()):
        raise ValueError('Frozen route196 inference helpers changed')
    for name, expected in source['function_bindings'].items():
        if function_source_sha256(globals()[name]) != expected:
            raise ValueError(f'Exported route dataset codec changed: {name}')
    dataset = path.parent/'route196_training.jsonl.gz'
    groups = load_route196_rows(dataset, source)
    plans = {split: [] for split in groups}
    with gzip.open(dataset, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            split, index = row['partition'], len(plans[row['partition']])
            truth = np.asarray(row['human_trajectory'], dtype=float)
            if (tuple(row['identity']) != groups[split]['identities'][index] or truth.shape != (6, 3)
                    or not np.isfinite(truth).all() or not np.array_equal(truth[:, 2].astype(np.float32), groups[split]['y'][index])):
                raise ValueError('Original trajectory/speed labels did not replay')
            plans[split].append(truth)
    plans = {split: np.asarray(values) for split, values in plans.items()}
    train, dev = groups['TRAIN'], groups['DEV']
    if route_frame:
        return fit_route_frame_plan(out, source, groups, plans, frozen)
    residuals = (plans['TRAIN']-route196_reference(train['x'])).reshape(-1, 18)
    params = {'n_estimators': 96, 'min_samples_leaf': 8, 'random_state': 42, 'n_jobs': -1}
    models, scores = {}, {}
    for name, width in (('reference96', 96), ('route196', 196)):
        print(f'Fitting {name}: {len(train["x"])} original PDD rows,18 trajectory/speed residual outputs...', flush=True)
        estimator = ExtraTreesRegressor(**params).fit(train['x'][:, :width], residuals)
        estimator.n_jobs = 1
        models[name] = estimator
        scores[name] = route196_plan_metrics(plans['DEV'], decode_route196_plan(estimator, dev['x'][:, :width]), dev['labels'])
    scores['persistence'] = route196_plan_metrics(plans['DEV'], route196_reference(dev['x']), dev['labels'])
    unmet = route196_plan_unmet(scores['route196'], scores['reference96'], scores['persistence'])
    route_valid = dev['x'][:, 100::5].sum(axis=1) > 0
    cohorts = {}
    for name, mask in (('with_route_prefix', route_valid), ('route_unknown', ~route_valid)):
        if mask.any():
            cohorts[name] = {model_name: {'n': int(mask.sum()),
                'ade_m': float(np.linalg.norm(decode_route196_plan(model, dev['x'][mask, :width])[:, :, :2]-plans['DEV'][mask, :, :2], axis=2).mean())}
                for model_name, model, width in (('reference96', models['reference96'], 96), ('route196', models['route196'], 196))}
    contract = {'schema': 'PDD_ROUTE196_PLAN_V1', 'offline_status': 'DEV_NEEDS_FIX' if unmet else 'DEV_IMPROVED',
        'feature_names': source['feature_names'], 'horizons_s': [.25, .5, .75, 1., 1.25, 1.5],
        'output': '6x[x_forward_m,y_left_m,speed_mps]', 'source_report_sha256': ROUTE196_DATA_SHA256,
        'dataset_sha256': source['dataset_sha256'], 'deployment_eligible': False,
        'policy_training_approved': False, 'source_behavior_quality': 'UNREVIEWED',
        'frozen_inference_files': frozen,
        'inference_bindings': {function.__name__: function_source_sha256(function) for function in
            (public_route196, encode_route196, encode_local_route_geometry, junction_forecast_features,
             route196_reference, decode_route196_plan, predict_public_route196_plan)}}
    model_path = out/'model_bundle.joblib'
    joblib.dump({**contract, 'plan_model': models['route196'], 'reference96_model': models['reference96']}, model_path)
    report = {**contract, 'training_performed': True, 'simulator_used': False, 'test_used': False,
        'counts': source['counts'], 'fitting': {**params, 'predict_n_jobs': 1, 'sklearn_version': sklearn.__version__,
            'fixed_candidate': 'route196', 'development_parameter_search': False},
        'dev_metrics': scores, 'dev_route_cohorts': cohorts, 'unmet_criteria': unmet,
        'model_bundle_sha256': digest(model_path), 'implementation_sha256': digest(__file__),
        'qualification': 'Fixed regression gates plus retained WAIT/LAUNCH criteria; no fitted probability threshold',
        'limitations': ['This is a learned trajectory/speed model, not a direct steering/throttle/brake policy.',
            'A passing DEV forecast status does not approve right-of-way, semantic route parity or deployment.',
            'Historical development results have already been inspected; TEST/closed-loop validation remain open.',
            'Both candidate/reference use identical rows,targets,seed and estimator settings; unknown route rows are retained.']}
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for name, metric in scores.items():
        print(f'{name}: ADE={metric["ade_m"]:.4f}m, speed1s={metric["speed"]["all"]["one_second_mae_mps"]:.4f}m/s', flush=True)
    print(f'{contract["offline_status"]}; unmet={unmet}. Read {out/"report.json"}', flush=True)


ROUTE196_PLAN_REPORT = 'agent4_integration_audit/run_20261122T211809_402991Z/report.json'
ROUTE196_PLAN_SHA256 = 'a8857348f50fc5be0da10c8fd10bb8d16898a59e2bda766aef635d4e598589e8'


def route196_guarded_action(model_action, reference_action, previous_steering, config, maneuver, route_known):
    """Diagnostic hybrid authority, never a claim the learned WAIT policy is qualified."""
    steering, command = map(float, model_action)
    reference_steer, reference_command = map(float, reference_action)
    if not all(math.isfinite(value) for value in (steering, command, reference_steer, reference_command)):
        raise ValueError('Nonfinite route196 action')
    source = 'STAGE1_MANEUVER' if maneuver else 'STAGE1_ROUTE_UNKNOWN' if not route_known else 'MODEL_TRACKER'
    if source != 'MODEL_TRACKER':
        steering = reference_steer
    steering = max(-config.max_steering, min(config.max_steering, steering))
    limit = config.steering_rate_per_s*.1
    steering = max(previous_steering-limit, min(previous_steering+limit, steering))
    return [float(steering), max(-1., min(1., command, reference_command))], source


class Route196LauncherDiagnostic(PDDSpeedPilot):
    agent_id = 'agent4_pdd_route196_plan_diagnostic'
    checkpoint_schema = 'PDD_ROUTE196_PLAN_V1'

    def __init__(self, checkpoint_report=None, expected_sha=None):
        import joblib
        import numpy as np
        from src.agents.rule_based_agent import Stage1RuleBasedAgent
        from src.control_stabilizer.controller import StabilizerConfig
        from src.platform.agent import AgentDescriptor
        from train_pdd_baseline import function_source_sha256
        path = ROOT/ROUTE196_PLAN_REPORT if checkpoint_report is None else Path(checkpoint_report)
        expected = ROUTE196_PLAN_SHA256 if checkpoint_report is None else expected_sha
        if digest(path) != expected:
            raise ValueError('Expected the reviewed route196 checkpoint report')
        self.report = json.loads(path.read_bytes())
        self.checkpoint_hash = self.report['model_bundle_sha256']
        model_path = path.parent/'model_bundle.joblib'
        if digest(model_path) != self.checkpoint_hash:
            raise ValueError('Route196 diagnostic checkpoint changed')
        self.bundle = joblib.load(model_path)
        if self.bundle['schema'] != self.checkpoint_schema or self.bundle['offline_status'] not in ('DEV_NEEDS_FIX', 'DEV_IMPROVED'):
            raise ValueError('This diagnostic is bound to the reviewed unqualified checkpoint')
        for name, expected in self.bundle['inference_bindings'].items():
            if function_source_sha256(globals()[name]) != expected:
                raise ValueError(f'Route196 inference binding changed: {name}')
        for name, expected in self.bundle['frozen_inference_files'].items():
            if digest(ROOT/name) != expected:
                raise ValueError('Route196 inference file changed')
        self.rule, self.config = Stage1RuleBasedAgent(), StabilizerConfig()
        self.history, self.next_step = deque(maxlen=5), None
        self.previous_steering = 0.
        self.descriptor = AgentDescriptor(agent_id=self.agent_id, agent_version='0.1.0',
                                          stateful_within_episode=True, method_family='HYBRID')
        self.decode_plan(np.zeros((1, 196), dtype=np.float32))

    def decode_plan(self, features):
        return decode_route196_plan(self.bundle['plan_model'], features)

    def reset(self, public_context, agent_seed=None):
        super().reset(public_context, agent_seed)
        self.previous_steering = 0.

    def export_config(self):
        return {'schema': 'ROUTE196_LAUNCHER_DIAGNOSTIC_V1', 'checkpoint_sha256': self.checkpoint_hash,
            'checkpoint_schema': self.bundle['schema'],
            'offline_status': self.bundle['offline_status'], 'diagnostic_only': True, 'deployment_eligible': False,
            'authority': 'Stage1 maneuver/unknown-route steering; model tracker elsewhere; longitudinal min(model,Stage1)',
            'tracker_origin': 'CHASSIS', 'dt_s': .1, 'speed_bonus_enabled': False,
            'stage1': self.rule.export_config(), 'adapter_sha256': digest(__file__)}

    def act(self, agent_input):
        import numpy as np
        import time
        from src.platform.agent import AgentDecision
        started = time.perf_counter()
        snapshot, _, speed, _, reference = self._observe(agent_input)
        infer_started = time.perf_counter()
        features = np.asarray(public_route196(snapshot), dtype=np.float32).reshape(1, 196)
        plan = self.decode_plan(features)[0].tolist()
        inference_ms = (time.perf_counter()-infer_started)*1000.
        raw = ((0., -1.) if speed is None else track_pdd_plan(plan, speed, self.config,
                    self.previous_steering, .1, self.rule.k_speed, self.rule.max_throttle))
        diag = reference.diagnostics or {}
        lane = (diag.get('lane_change') or {}).get('phase')
        junction = diag.get('junction_phase')
        maneuver = (lane in ('PREPARE_CHANGE', 'CHANGE_LANE', 'ABORT_CHANGE', 'BLOCKED_CHANGE') or
                    junction in ('APPROACH_JUNCTION', 'COMMIT_JUNCTION', 'TRAVERSE_JUNCTION',
                                 'EXIT_JUNCTION', 'YIELD', 'BLOCKED', 'DEGRADED_STOP'))
        action, source = route196_guarded_action(raw, reference.action_payload, self.previous_steering,
            self.config, maneuver, bool(features[0, 100::5].any()))
        self.previous_steering = action[0]
        return AgentDecision(action, {'route196_plan': plan, 'model_raw_action': list(raw),
            'stage1_action': list(map(float, reference.action_payload)), 'steering_source': source,
            'stage1_command_cap_active': raw[1] > float(reference.action_payload[1])+1e-7,
            'model_limits_stage1': raw[1] < float(reference.action_payload[1])-1e-7,
            'model_inference_ms': inference_ms, 'policy_act_ms': (time.perf_counter()-started)*1000.,
            'junction_phase': junction, 'lane_change_phase': lane,
            'offline_status': self.bundle['offline_status'], 'diagnostic_only': True})


class RouteFrameLauncherDiagnostic(Route196LauncherDiagnostic):
    agent_id = 'agent4_pdd_route_frame_diagnostic'
    checkpoint_schema = 'PDD_ROUTE196_FRAME_PLAN_V2'

    def decode_plan(self, features):
        return decode_route_frame_plan(self.bundle['plan_model'], features)


def run_route196_runtime(project, out, *, policy=Route196LauncherDiagnostic, checkpoint_report=None, cases=None):
    import numpy as np
    from src.launcher.models import AgentRegistrationV1
    from src.launcher.registry import build_canonical_agent_registry
    registry = build_canonical_agent_registry()
    factory = policy if checkpoint_report is None else partial(policy, checkpoint_report=checkpoint_report,
                                                              expected_sha=digest(checkpoint_report))
    registry.register(AgentRegistrationV1(agent_id=policy.agent_id, agent_version='0.1.0',
        method_family='HYBRID', purpose='CANDIDATE', implementation_ref=f'{policy.__module__}:{policy.__qualname__}',
        requires_checkpoint=True, stateful_within_episode=True), factory)
    pairs = []
    for tier, sequence, geometry_seed, environment_seed in (cases or (('Medium', 'SCXCS', 14, 1), ('Medium', 'SCXCCS', 6, 2))):
        folder = out/f'{tier}_{sequence}_{geometry_seed}_{environment_seed}'
        folder.mkdir()
        agents = {}
        for agent in ('stage1_rule_based', policy.agent_id):
            print(f'DIAGNOSTIC route196: {agent} on {tier}/{sequence}/{geometry_seed}/{environment_seed}', flush=True)
            _, directory, episode = launch_train_case(project, folder, registry, agent, tier, sequence,
                geometry_seed, agent, environment_seed=environment_seed)
            trace = directory/'agent_decisions.jsonl.gz'
            counts, speed_values, latency = Counter(), [], []
            with gzip.open(trace, 'rt', encoding='utf-8') as stream:
                for line in stream:
                    row = json.loads(line)
                    if (row['case_id'] != episode['case_id'] or row['step_index'] != counts['frames']
                            or row['agent_input']['step_index'] != counts['frames']):
                        raise ValueError('Route196 runtime trace is not consecutive')
                    counts['frames'] += 1
                    traffic = row['agent_input']['traffic_context']['active_count'] > 0
                    counts['traffic_steps'] += int(traffic)
                    motion = row['agent_input']['ego_motion']
                    if motion is not None:
                        speed_values.append(math.hypot(motion['velocity_x_mps'], motion['velocity_y_mps'])*3.6)
                    if agent == policy.agent_id:
                        diag = row['decision']['diagnostics']
                        applied = row['decision']['action_payload']
                        if applied[1] > diag['stage1_action'][1]+1e-7:
                            raise ValueError('Route196 violated declared Stage1 longitudinal cap')
                        counts[diag['steering_source']] += 1
                        counts['stage1_command_caps'] += int(diag['stage1_command_cap_active'])
                        counts['model_limits_stage1'] += int(diag['model_limits_stage1'])
                        counts['moving_head_active_steps'] += int(diag.get('moving_head_active', False))
                        counts['speed_support_active_steps'] += int(diag.get('speed_support_active', False))
                        changed = any(abs(x-y)>1e-7 for x,y in zip(applied, diag['stage1_action']))
                        counts['changed_actions'] += int(changed)
                        counts['changed_actions_with_traffic'] += int(changed and traffic)
                        latency.append(diag['model_inference_ms'])
            if counts['frames'] != int(episode['episode_steps']):
                raise ValueError('Runtime trace length differs from episode')
            agents[agent] = {'episode': episode, 'run_dir': str(directory), 'trace_sha256': digest(trace),
                'counts': dict(counts), 'speed_mean_kmh_including_stops': float(np.mean(speed_values)) if speed_values else None,
                'speed_max_kmh': max(speed_values) if speed_values else None, 'stops': recorded_stop_metrics(trace),
                'model_inference_p95_ms': float(np.percentile(latency, 95)) if latency else None}
        baseline_clean = agents['stage1_rule_based']['episode']['clean_success'].lower() == 'true'
        candidate_clean = agents[policy.agent_id]['episode']['clean_success'].lower() == 'true'
        pair = {'case_id': episode['case_id'], 'agents': agents,
            'baseline_clean': baseline_clean, 'candidate_clean': candidate_clean,
            'time_change_s_if_both_clean': (float(agents[policy.agent_id]['episode']['time_to_clean_success_s'])-
                float(agents['stage1_rule_based']['episode']['time_to_clean_success_s'])) if baseline_clean and candidate_clean else None}
        (folder/'report.json').write_text(json.dumps(pair, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        pairs.append(pair)
    report = {'schema': 'ROUTE196_RUNTIME_DIAGNOSTIC_V1', 'status': 'DIAGNOSTIC_COMPLETE_NOT_DEPLOYABLE',
        'source_report_sha256': ROUTE196_PLAN_SHA256 if checkpoint_report is None else digest(checkpoint_report),
        'pairs': pairs, 'training_performed': False,
        'baseline_clean_arrivals': sum(pair['baseline_clean'] for pair in pairs),
        'candidate_clean_arrivals': sum(pair['candidate_clean'] for pair in pairs),
        'test_used': False, 'deployment_eligible': False,
        'policy': policy.agent_id,
        'limitations': ['Fresh paired TRAIN sandbox trials only, not final validation or autonomous learned junction authority.',
            'A guarded arrival never substitutes for offline WAIT qualification or independent validation.',
            'Stage1 steers maneuvers/unknown routes and caps every longitudinal command; report counts expose its contribution.',
            'Inference time excludes file/codec integrity checks performed once on load; it is not total launcher wall time.']}
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f'DIAGNOSTIC_COMPLETE_NOT_DEPLOYABLE. Read {out/"report.json"}', flush=True)


def run_route_frame_cycle(project, out):
    parent_runtime = ROOT/'agent4_integration_audit/run_20261122T212740_723289Z/report.json'
    if digest(parent_runtime) != '94e5efa1a465b9da2b679105f2f65a759b7e98a17df8fbe9ec61f16cd9ac58f9':
        raise ValueError('Expected the reviewed failed route196 runtime report')
    training, runtime = out/'training', out/'runtime'
    training.mkdir()
    runtime.mkdir()
    report_path = train_route196_plan(training, route_frame=True)
    run_route196_runtime(project, runtime, policy=RouteFrameLauncherDiagnostic, checkpoint_report=report_path)
    trained = json.loads(report_path.read_bytes())
    trial = json.loads((runtime/'report.json').read_bytes())
    summary = {'schema': 'ROUTE_FRAME_TRAIN_RUNTIME_CYCLE_V1', 'status': 'TRAIN_AND_DIAGNOSTIC_COMPLETE',
        'training_report': str(report_path), 'training_report_sha256': digest(report_path),
        'runtime_report': str(runtime/'report.json'), 'runtime_report_sha256': digest(runtime/'report.json'),
        'offline_status': trained['offline_status'], 'unmet_criteria': trained['unmet_criteria'],
        'baseline_clean_arrivals': trial['baseline_clean_arrivals'],
        'candidate_clean_arrivals': trial['candidate_clean_arrivals'],
        'training_performed': True, 'simulator_used': True, 'test_used': False, 'deployment_eligible': False,
        'limits': 'Original PDD labels only; guarded TRAIN trial even if DEV fails; no automatic promotion.'}
    (out/'report.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f'TRAIN_AND_DIAGNOSTIC_COMPLETE. Read {out/"report.json"}', flush=True)


FRAME_PARENT_REPORT = 'agent4_integration_audit/run_20261122T221041_687465Z/training/report.json'
FRAME_PARENT_SHA256 = '9e7b152a9fc66582e93896dca0d024b923beaa7da2f69a5a41cb962173e0f24f'


def speed_record_signature(features96, trajectory):
    values = [*features96, *[value for point in trajectory for value in point]]
    if len(values) != 114 or not all(math.isfinite(float(value)) for value in values):
        raise ValueError('Invalid speed-union observation/trajectory')
    return hashlib.sha256(struct.pack('<114f', *values)).hexdigest()


def prepare_moving_speed_union(out, pdd):
    """Reuse frozen C0/state and junction data; no map rebuild, boxes or expert controls."""
    import numpy as np
    from build_agent4_pdd_pilot import read_source, xyz, partition
    c0_path = ROOT/'runs/agent4_pdd_development_c0_02'
    if (digest(c0_path/'audit.json') != '3bd2be48eb74b6b869665eef4278625514d16030c43a11c86c7cd9ff4b22b5b9'
            or digest(c0_path/'c0_screened_candidates.jsonl') != '42eaf94fdb980c64e9d6047267b5383348778c820946cc4e26b606dbd676781e'):
        raise ValueError('Frozen original PDD C0 source changed')
    data_path = ROOT/ROUTE196_DATA_REPORT
    if digest(data_path) != ROUTE196_DATA_SHA256:
        raise ValueError('Frozen junction data report changed')
    data_report = json.loads(data_path.read_bytes())
    original = data_path.parent/'route196_training.jsonl.gz'
    if digest(original) != data_report['dataset_sha256']:
        raise ValueError('Frozen junction data changed')
    c0 = json.loads((c0_path/'audit.json').read_bytes())
    if digest(ROOT/JUNCTION_FORECAST_SOURCE_REPORT) != JUNCTION_FORECAST_SOURCE_SHA256:
        raise ValueError('Frozen junction source receipts changed')
    junction_source = json.loads((ROOT/JUNCTION_FORECAST_SOURCE_REPORT).read_bytes())
    if digest(ROOT/'build_agent4_pdd_pilot.py') != junction_source['frozen_builder_sha256']:
        raise ValueError('Frozen PDD history reader changed')
    receipts = {'C0': {row['episode']: row for row in c0['episodes']},
                'JUNCTION': {row['episode']: row for row in junction_source['episodes']}}
    groups = {split: {'x': [], 'y': [], 'labels': [], 'identities': []} for split in ('TRAIN', 'DEV')}
    counts = {split: Counter() for split in groups}
    seen, verified, cached, speeds = {}, {}, None, None
    pdd = pdd.resolve()
    for domain in ('JUNCTION', 'C0'):
        stream = (gzip.open(original, 'rt', encoding='utf-8') if domain == 'JUNCTION'
                  else (c0_path/'c0_screened_candidates.jsonl').open(encoding='utf-8'))
        with stream:
            for line in stream:
                row = json.loads(line)
                driver, scene, tick = row['identity'] if domain == 'JUNCTION' else (row['driver'], row['scenario'], row['telemetry_step'])
                identity, split = (driver, scene, tick), row['partition']
                driver_index = int(driver.removeprefix('driver_'))
                if (not 1 <= driver_index <= 25 or split != partition(driver_index)
                        or row['behavior_quality'] != 'UNREVIEWED' or (domain == 'C0' and row['target_mask'] != [True]*6)):
                    raise ValueError('Speed union split/source label contract failed')
                counts[split][domain+'_seen'] += 1
                base = row['features'][:96] if domain == 'JUNCTION' else encode_route196(row['features'], [])[:96]
                truth = row['human_trajectory']
                signature = speed_record_signature(base, truth)
                if identity in seen:
                    if seen[identity] != signature:
                        raise ValueError('Conflicting C0/junction duplicate, refusing silent replacement')
                    counts[split]['duplicates'] += 1
                    continue
                seen[identity] = signature
                receipt = receipts[domain][f'{driver}/{scene}']
                key = (domain, driver, scene)
                if key != cached:
                    relative = f'{driver}/data/{scene}/metric/metric_info.json'
                    inventory = {}
                    raw = read_source(pdd/relative, inventory, pdd)
                    if receipt['partition'] != split or inventory[relative] != receipt['source_sha256'][relative]:
                        raise ValueError('Speed history source checksum/split mismatch')
                    verified[relative] = inventory[relative]
                    speeds = {int(step): math.hypot(*xyz(info['velocity'])[:2]) for step, info in raw.items()}
                    del raw
                    cached = key
                current = speeds[tick]
                if not math.isclose(current, float(base[0]), abs_tol=1e-5, rel_tol=1e-6):
                    raise ValueError('Speed-union current speed mismatch')
                target = [float(point[2]) for point in truth]
                if len(target) != 6 or any(value < 0 for value in target):
                    raise ValueError('Invalid recorded speed targets')
                label = ('WAIT' if current < .3 and target[3] < .3 else 'LAUNCH' if current < .3 else
                         'SLOWDOWN' if target[3] < current-.5 else 'SPEEDUP' if target[3] > current+.5 else 'STEADY')
                if domain == 'JUNCTION' and label != row['motion_label']:
                    raise ValueError('Junction label changed during union')
                history = causal_speed_history(speeds, tick, stride=5, seconds_per_tick=.05)
                group = groups[split]
                group['x'].append([*base, *history])
                group['y'].append(target)
                group['labels'].append(label)
                group['identities'].append(identity)
                counts[split]['rows'] += 1
                counts[split][domain+'_added'] += 1
        print(f'Prepared {domain}: {dict(counts["TRAIN"])}, {dict(counts["DEV"])}', flush=True)
    expected = {'TRAIN': (10900, 24381), 'DEV': (1633, 3893)}
    arrays = {}
    for split, group in groups.items():
        junction_n, c0_n = expected[split]
        if (counts[split]['JUNCTION_seen'] != junction_n or counts[split]['C0_seen'] != c0_n
                or counts[split]['rows'] != junction_n+c0_n-counts[split]['duplicates']):
            raise ValueError('Speed-union population accounting failed')
        group['x'], group['y'] = np.asarray(group['x'], dtype=np.float32), np.asarray(group['y'], dtype=np.float32)
        if group['x'].shape != (counts[split]['rows'], 102) or not np.isfinite(group['x']).all() or not np.isfinite(group['y']).all():
            raise ValueError('Invalid union matrices')
        arrays[split+'_x'], arrays[split+'_y'] = group['x'], group['y']
        arrays[split+'_labels'] = np.asarray(group['labels'])
        arrays[split+'_identities'] = np.asarray([[driver, scene, str(tick)] for driver, scene, tick in group['identities']])
    data_file = out/'speed102_union.npz'
    np.savez_compressed(data_file, **arrays)
    with np.load(data_file, allow_pickle=False) as loaded:
        for key, values in arrays.items():
            if not np.array_equal(loaded[key], values):
                raise ValueError('Persisted speed-union matrix failed exact reload')
    manifest = {'schema': 'PDD_MOVING_SPEED102_UNION_V1', 'dataset_sha256': digest(data_file),
        'counts': {split: dict(values) for split, values in counts.items()}, 'verified_source_files': verified,
        'junction_report_sha256': ROUTE196_DATA_SHA256, 'c0_audit_sha256': digest(c0_path/'audit.json'),
        'map_rebuilt': False, 'expert_controls_used': False, 'test_used': False,
        'features': '96D normalized state/navigation +6 causal ego-speed history',
        'duplicate_policy': 'One identity once; conflicting state/trajectory aborts', 'reload': 'EXACT_ARRAY_PASS'}
    (out/'dataset_report.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return groups, manifest


def decode_moving_route_plan(bundle, features, history):
    import numpy as np
    features = np.asarray(features, dtype=np.float32)
    history = np.asarray(history, dtype=np.float32)
    if history.shape != (len(features), 6) or not np.isfinite(history).all():
        raise ValueError('Invalid causal moving-speed history')
    plan = decode_route_frame_plan(bundle['plan_model'], features)
    moving = (features[:, 0] >= .3) & (features[:, 3] == 1.)
    if moving.any():
        speed_x = np.column_stack((features[moving, :96], history[moving]))
        residual = np.asarray(bundle['moving_speed_model'].predict(speed_x), dtype=float)
        if residual.shape != (int(moving.sum()), 6) or not np.isfinite(residual).all():
            raise ValueError('Invalid learned moving-speed output')
        plan[moving, :, 2] = np.maximum(0., features[moving, :1]+residual)
    return plan


def predict_public_moving_route_plan(bundle, snapshot, observed_speeds):
    import numpy as np
    from train_pdd_baseline import function_source_sha256
    if bundle['schema'] != 'PDD_ROUTE_MOVING_SPEED_V3':
        raise ValueError('Unknown moving-route checkpoint schema')
    for name, expected in bundle['frozen_inference_files'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Moving-route inference helper changed')
    for name, expected in bundle['inference_bindings'].items():
        if function_source_sha256(globals()[name]) != expected:
            raise ValueError('Moving-route inference codec changed')
    if hasattr(snapshot, 'to_dict'):
        snapshot = snapshot.to_dict()
    features = np.asarray([public_route196(snapshot)], dtype=np.float32)
    cache = dict(observed_speeds)
    if features[0, 3] == 1.:
        cache[snapshot['step_index']] = float(features[0, 0])
        history = causal_speed_history(cache, snapshot['step_index'], stride=1, seconds_per_tick=.1)
    else:
        history = [0.]*6
    return decode_moving_route_plan(bundle, features, np.asarray([history]))[0].tolist()


class MovingRouteLauncherDiagnostic(RouteFrameLauncherDiagnostic):
    agent_id = 'agent4_pdd_route_moving_speed_diagnostic'
    checkpoint_schema = 'PDD_ROUTE_MOVING_SPEED_V3'

    def reset(self, public_context, agent_seed=None):
        super().reset(public_context, agent_seed)
        self.moving_history = {}

    def decode_plan(self, features):
        import numpy as np
        if self.next_step is None:
            return decode_route_frame_plan(self.bundle['plan_model'], features)
        step = self.next_step-1
        if features[0, 3] == 1.:
            self.moving_history[step] = float(features[0, 0])
            history = causal_speed_history(self.moving_history, step, stride=1, seconds_per_tick=.1)
        else:
            history = [0.]*6
        self.moving_history = {key: value for key, value in self.moving_history.items() if key >= step-20}
        self.moving_head_active = bool(features[0, 0] >= .3 and features[0, 3] == 1.)
        return decode_moving_route_plan(self.bundle, features, np.asarray([history]))

    def act(self, agent_input):
        from src.platform.agent import AgentDecision
        result = super().act(agent_input)
        return AgentDecision(result.action_payload, {**result.diagnostics, 'moving_head_active': self.moving_head_active})


def run_moving_route_cycle(project, out, pdd):
    import joblib
    import numpy as np
    from sklearn.ensemble import ExtraTreesRegressor
    from train_pdd_baseline import function_source_sha256
    parent_path = ROOT/FRAME_PARENT_REPORT
    if digest(parent_path) != FRAME_PARENT_SHA256:
        raise ValueError('Expected frozen2/2route-frame training report')
    parent = json.loads(parent_path.read_bytes())
    if digest(parent_path.parent.parent/'runtime/report.json') != '04a893125fc186e067f3f8f74f65ee3bc53dc95a407428c87d568228c0ca3124':
        raise ValueError('Frozen2/2geometry runtime evidence changed')
    if digest(parent_path.parent/'model_bundle.joblib') != parent['model_bundle_sha256']:
        raise ValueError('Frozen geometry checkpoint changed')
    bundle = joblib.load(parent_path.parent/'model_bundle.joblib')
    for name, expected in bundle['inference_bindings'].items():
        if function_source_sha256(globals()[name]) != expected:
            raise ValueError('Frozen geometry inference codec changed')
    for name, expected in bundle['frozen_inference_files'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen inference helper changed')
    groups, manifest = prepare_moving_speed_union(out, pdd)
    train, dev = groups['TRAIN'], groups['DEV']
    moving = (train['x'][:, 0] >= .3) & (train['x'][:, 3] == 1.)
    params = {'n_estimators': 96, 'min_samples_leaf': 8, 'random_state': 42, 'n_jobs': -1}
    print(f'Fitting moving-speed head on{int(moving.sum())}unique original PDD TRAIN rows...', flush=True)
    model = ExtraTreesRegressor(**params).fit(train['x'][moving], train['y'][moving]-train['x'][moving, :1])
    model.n_jobs = 1
    selected = (dev['x'][:, 0] >= .3) & (dev['x'][:, 3] == 1.)
    predicted = np.maximum(0., dev['x'][selected, :1]+model.predict(dev['x'][selected]))
    labels = np.asarray(dev['labels'])[selected]
    scores = junction_forecast_scores(dev['y'][selected], predicted, labels)
    baseline = junction_forecast_scores(dev['y'][selected], np.repeat(dev['x'][selected, :1], 6, axis=1), labels)
    unmet = []
    if scores['all']['one_second_mae_mps'] > .9*baseline['all']['one_second_mae_mps']:
        unmet.append('MOVING/mae_gain_below_0p10')
    if scores['SLOWDOWN']['one_second_mae_mps'] >= baseline['SLOWDOWN']['one_second_mae_mps']:
        unmet.append('MOVING/SLOWDOWN/not_better_than_persistence')
    # Preserve the original junction cohort, so C0 rows cannot dilute WAIT regressions.
    source = json.loads((ROOT/ROUTE196_DATA_REPORT).read_bytes())
    original = load_route196_rows((ROOT/ROUTE196_DATA_REPORT).parent/'route196_training.jsonl.gz', source)['DEV']
    lookup = {identity: index for index, identity in enumerate(dev['identities'])}
    indices = [lookup[identity] for identity in original['identities']]
    if (not np.array_equal(original['x'][:, :96], dev['x'][indices, :96]) or
            not np.array_equal(original['y'], dev['y'][indices])):
        raise ValueError('Original junction DEV state/targets changed in union')
    old_plan = decode_route_frame_plan(bundle['plan_model'], original['x'])
    bundle['moving_speed_model'] = model
    new_plan = decode_moving_route_plan(bundle, original['x'], dev['x'][indices, 96:])
    parent_speed = junction_forecast_scores(original['y'], old_plan[:, :, 2], original['labels'])
    new_speed = junction_forecast_scores(original['y'], new_plan[:, :, 2], original['labels'])
    if new_speed['all']['one_second_mae_mps'] > parent_speed['all']['one_second_mae_mps']*1.05:
        unmet.append('JUNCTION/all/speed_regression_over_5pct')
    if new_speed['SLOWDOWN']['one_second_mae_mps'] > parent_speed['SLOWDOWN']['one_second_mae_mps']*1.05:
        unmet.append('JUNCTION/SLOWDOWN/speed_regression_over_5pct')
    wait = np.asarray(original['labels']) == 'WAIT'
    if not np.array_equal(new_plan[wait, :, 2], old_plan[wait, :, 2]):
        unmet.append('JUNCTION/WAIT/predictions_changed')
    contract = {key: value for key, value in bundle.items() if key != 'plan_model' and key != 'moving_speed_model'}
    contract.update(schema='PDD_ROUTE_MOVING_SPEED_V3', offline_status='DEV_NEEDS_FIX',
        moving_head_status='MOVING_DEV_NEEDS_FIX' if unmet else 'MOVING_DEV_IMPROVED',
        moving_head_unmet=unmet, geometry_parent_sha256=parent['model_bundle_sha256'],
        geometry_parent_report_sha256=FRAME_PARENT_SHA256, geometry_input_width=196, moving_input_width=102,
        moving_speed_threshold_mps=.3, moving_head_features=[*route196_feature_names()[:96], *JUNCTION_HISTORY_NAMES],
        dataset_sha256=manifest['dataset_sha256'], deployment_eligible=False)
    contract['inference_bindings'] = {**contract['inference_bindings'],
        **{function.__name__: function_source_sha256(function) for function in
           (causal_speed_history, decode_moving_route_plan, predict_public_moving_route_plan)}}
    training = out/'training'
    training.mkdir()
    model_path = training/'model_bundle.joblib'
    joblib.dump({**contract, 'plan_model': bundle['plan_model'], 'moving_speed_model': model}, model_path)
    report = {**contract, 'training_performed': True, 'test_used': False, 'fitting': params,
        'parent_geometry_unmet_criteria': parent['unmet_criteria'],
        'counts': manifest['counts'], 'moving_DEV_scores': scores, 'moving_DEV_persistence': baseline,
        'junction_DEV_parent': parent_speed, 'junction_DEV_new': new_speed,
        'junction_stop_launch_tradeoff': junction_stop_launch_tradeoff(original['labels'], new_plan[:, :, 2]),
        'model_bundle_sha256': digest(model_path),
        'limitations': ['Only moving-speed head retrained; geometry weights fixed. No new cruiser/pace law.',
            'Near-stop keeps parent forecast; existing WAIT failures remain unqualified even if moving head improves.',
            'Original PDD C0/junction union only; future labels and expert controls never enter features.']}
    report_path = training/'report.json'
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    runtime = out/'runtime'
    runtime.mkdir()
    cases = (('Medium', 'SCXCS', 14, 1), ('Medium', 'SCXCCS', 6, 2), ('Medium', 'SCTCS', 11, 0),
             ('Hard', 'SCXOCS', 15, 0), ('Hard', 'XTOCS', 8, 2))
    run_route196_runtime(project, runtime, policy=MovingRouteLauncherDiagnostic, checkpoint_report=report_path, cases=cases)
    trial = json.loads((runtime/'report.json').read_bytes())
    summary = {'schema': 'MOVING_ROUTE_TRAIN_RUNTIME_CYCLE_V1', 'status': 'DATA_TRAIN_AND_DIAGNOSTIC_COMPLETE',
        'counts': manifest['counts'], 'moving_head_status': contract['moving_head_status'], 'moving_head_unmet': unmet,
        'training_report': str(report_path), 'runtime_report': str(runtime/'report.json'),
        'baseline_clean_arrivals': trial['baseline_clean_arrivals'], 'candidate_clean_arrivals': trial['candidate_clean_arrivals'],
        'test_used': False, 'deployment_eligible': False, 'parent_WAIT_qualification': 'STILL_NEEDS_FIX'}
    (out/'report.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f'DATA_TRAIN_AND_DIAGNOSTIC_COMPLETE. Read {out/"report.json"}', flush=True)


def supported_route_command(raw_command, assisted_command, stage1_command, speed, gain, max_throttle):
    """Existing40km/h goal, preserving strong model brake and every Stage1 stop."""
    if not all(math.isfinite(value) for value in (raw_command, assisted_command, stage1_command, speed, gain, max_throttle)):
        raise ValueError('Invalid supported longitudinal observation')
    requested = raw_command if raw_command < -NAV40_PARAMETERS['maximum_model_brake'] else assisted_command
    cap = max(-1., min(max_throttle, gain*(NAV40_PARAMETERS['target_kmh']-speed*3.6)))
    return float(max(-1., min(requested, stage1_command, cap)))


class SupportedMovingRouteDiagnostic(MovingRouteLauncherDiagnostic):
    agent_id = 'agent4_pdd_route_moving_supported40'
    pace_support_enabled = True
    rear_axle_offset_m = 0.

    def __init__(self, checkpoint_report=None, expected_sha=None):
        from train_pdd_baseline import PLAN_LANE_FEATURES
        self.support_lineage = frozen_flow_lineage()
        super().__init__(checkpoint_report=checkpoint_report, expected_sha=expected_sha)
        self.names = [*route196_feature_names()[:96], *PLAN_LANE_FEATURES]
        self.lane_guard_active, self.lane_guard_release_ticks = False, 0

    def reset(self, public_context, agent_seed=None):
        super().reset(public_context, agent_seed)
        self.lane_guard_active, self.lane_guard_release_ticks = False, 0

    def _observe(self, agent_input):
        observed = super()._observe(agent_input)
        self.support_reference = observed[-1].diagnostics or {}
        return observed

    def track_plan(self, plan, speed, previous_steering):
        return track_pdd_plan(plan, speed, self.config, previous_steering, .1,
                              self.rule.k_speed, self.rule.max_throttle)

    def export_config(self):
        return {**super().export_config(), 'controller_assistance': True,
            'speed_bonus_enabled': True,
            'authority': 'Reviewed lane-margin/pace/cruise/flow controller; Stage1 maneuver/unknown/guard steering and final longitudinal cap',
            'navigation40': dict(NAV40_PARAMETERS), 'flow40': dict(FLOW40_PARAMETERS),
            'pace_guard': dict(PACE_SUPPORT_PARAMETERS), 'support_logic_reference': self.support_lineage,
            'support_reference_weights_used': False,
            'claim': 'Frozen learned weights with explicit reviewed controller assistance, not learned40km/h authority'}

    def act(self, agent_input):
        import time
        from train_pdd_baseline import public_plan_lane_pose
        from src.platform.agent import AgentDecision
        started, previous = time.perf_counter(), self.previous_steering
        base = super().act(agent_input)
        snapshot = agent_input.to_dict()
        motion = snapshot['ego_motion']
        if motion is None:
            return AgentDecision(base.action_payload, {**base.diagnostics, 'controller_assistance': True,
                'speed_support_active': False, 'support_block_reasons': ['MOTION_UNAVAILABLE']})
        public_features = public_route196(snapshot)
        speed = float(public_features[0])
        features = [*public_features[:96], *public_plan_lane_pose(snapshot)]
        raw = base.diagnostics['model_raw_action']
        teacher = base.diagnostics['stage1_action']
        plan = base.diagnostics['route196_plan']
        steering, requested, support = PDDPlanPilot.control_plan(self, snapshot, features, plan, speed,
                                                                 previous, teacher[0], teacher[1])
        source = 'MODEL_TRACKER' if support['steering_source'] == 'PDD_PLAN_TRACKER' else support['steering_source']
        original_source = base.diagnostics['steering_source']
        if original_source != 'MODEL_TRACKER':
            steering, source = base.action_payload[0], original_source
            requested = raw[1]
            support['pace_bonus_active'] = False
        navigation = original_source == 'STAGE1_MANEUVER'
        route_known = any(public_features[100::5])
        ref = self.support_reference
        hint = {'junction_phase': ref.get('junction_phase'), 'junction_reason': ref.get('junction_reason'),
            'lane_change_phase': (ref.get('lane_change') or {}).get('phase'),
            'speed_limit_reason': ref.get('speed_limit_reason'), 'target_speed_kmh': ref.get('target_speed_kmh'),
            'traffic_risk': {name: ref[name] for name in FLOW_RISK_FIELDS if name in ref}}
        cruise_blocks = cruise40_reasons(snapshot, features, self.names, plan, speed, teacher[1],
            self.lane_guard_active, support['raw_plan_steer'], raw[1], navigation)
        flow_blocks = flow_progress_reasons(snapshot, plan, speed, raw[1], teacher[1], hint,
                                             self.lane_guard_active, navigation)
        if not route_known:
            cruise_blocks.append('MODEL_ROUTE_UNKNOWN')
            flow_blocks.append('MODEL_ROUTE_UNKNOWN')
        target = support['controller_target_speed_mps']
        cruise, flow = not cruise_blocks, False
        if cruise:
            target = NAV40_PARAMETERS['target_kmh']/3.6
            requested = max(-1., min(self.rule.max_throttle, self.rule.k_speed*3.6*(target-speed)))
        if not flow_blocks:
            maneuver = (hint['lane_change_phase'] in ('PREPARE_CHANGE', 'CHANGE_LANE', 'ABORT_CHANGE')
                        or hint['junction_phase'] in ('COMMIT_JUNCTION', 'TRAVERSE_JUNCTION', 'EXIT_JUNCTION'))
            target_kmh = hint['target_speed_kmh'] if maneuver else plan[3][2]*3.6+FLOW40_PARAMETERS['progress_increment_kmh']
            flow_target = min(target_kmh, hint['target_speed_kmh'], NAV40_PARAMETERS['target_kmh'])/3.6
            proposal = min(teacher[1], max(-1., min(self.rule.max_throttle, self.rule.k_speed*3.6*(flow_target-speed))))
            if proposal > requested:
                requested, target, flow = proposal, flow_target, True
        command = supported_route_command(raw[1], requested, teacher[1], speed, self.rule.k_speed, self.rule.max_throttle)
        self.previous_steering = steering
        return AgentDecision([float(steering), command], {**base.diagnostics,
            'steering_source': source, 'controller_assistance': True,
            'speed_support_active': command > base.action_payload[1]+1e-7,
            'unassisted_final_command': base.action_payload[1], 'cruise40_active': cruise,
            'flow_progress_active': flow, 'cruise40_block_reasons': cruise_blocks,
            'flow_progress_block_reasons': flow_blocks, 'lane_guard_active': self.lane_guard_active,
            'controller_target_speed_mps': min(target, NAV40_PARAMETERS['target_kmh']/3.6) if target is not None else None,
            'raw_model_target_speed_mps': plan[3][2],
            'policy_act_ms': (time.perf_counter()-started)*1000.})


def run_supported_route_trial(project, out):
    path = ROOT/'agent4_integration_audit/run_20261122T230515_925401Z/training/report.json'
    if digest(path) != 'ec58963c837dad6c4e5972da5038aa37b79398150b9f9144f1a038b1fb14e368':
        raise ValueError('Expected the reviewed moving-speed checkpoint report')
    cases = (('Medium', 'SCXCS', 14, 1), ('Medium', 'SCXCCS', 6, 2), ('Medium', 'SCTCS', 11, 0),
             ('Hard', 'SCXOCS', 15, 0), ('Hard', 'XTOCS', 8, 2))
    run_route196_runtime(project, out, policy=SupportedMovingRouteDiagnostic, checkpoint_report=path, cases=cases)
    report_path = out/'report.json'
    report = json.loads(report_path.read_bytes())
    source = json.loads(path.read_bytes())
    report.update(controller_assistance=True, learned_weights_changed=False,
        checkpoint_sha256=source['model_bundle_sha256'], offline_status=source['offline_status'],
        moving_head_status=source['moving_head_status'],
        navigation40_parameters=dict(NAV40_PARAMETERS), flow40_parameters=dict(FLOW40_PARAMETERS))
    report['limitations'].append('Any speed improvement here includes explicit controller assistance; it is not newly learned cruising or standalone policy qualification.')
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-root', required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--live', action='store_true')
    mode.add_argument('--pdd-runtime-smoke', action='store_true',
                      help='Check the PDD96 encoder through the TRAIN sandbox launcher')
    mode.add_argument('--pdd-speed-traffic', action='store_true',
                      help='Compare Stage 1 and the guarded pilot 02 checkpoint on matched TRAIN traffic')
    mode.add_argument('--pdd-speed-traffic-only', action='store_true',
                      help='Test pilot 02 longitudinal control only while a public actor is observed')
    mode.add_argument('--pdd-traffic-suite', action='store_true',
                      help='Compare the frozen traffic-only pilot and Stage 1 on three new TRAIN cases')
    mode.add_argument('--pdd-forward-suite', action='store_true',
                      help='Run three forward-corridor episodes using the reviewed same-case references')
    mode.add_argument('--pdd-forward-transfer', action='store_true',
                      help='Compare the frozen forward policy and Stage 1 on new Medium, Hard and Extreme TRAIN cases')
    mode.add_argument('--pdd-extreme-exit', action='store_true',
                      help='Compare Stage 1 and frozen PDD on the timeout geometry with traffic seeds 0 and 1')
    mode.add_argument('--pdd-expanded-suite', action='store_true',
                      help='Check the expanded checkpoint on eight TRAIN cases with reviewed references')
    mode.add_argument('--pdd-plan-runtime', action='store_true',
                      help='Track the learned plan for steering and speed on two paired TRAIN traffic cases')
    mode.add_argument('--pdd-lane-curvature-audit', action='store_true',
                      help='Offline Town12/current-public-lane curvature and speed-coverage audit; no simulator')
    mode.add_argument('--pdd-junction-gap-review', action='store_true',
                      help='Review coherent junction timing on recorded Extreme inputs; no simulator or model training')
    mode.add_argument('--pdd-waiting-pose-review', action='store_true',
                      help='Inspect recorded future actor footprints around waiting poses; offline hindsight only')
    mode.add_argument('--pdd-traffic-capacity-probe', action='store_true',
                      help='One TRAIN simulation with passive N32 observer; flow40 driver still receives original N8')
    mode.add_argument('--pdd-traffic-capacity-shadow', action='store_true',
                      help='Offline paired N8 replay and partial N32 perception sensitivity; no simulator')
    mode.add_argument('--pdd-junction-candidates', action='store_true',
                      help='Export HUMAN_RECORDED junction review windows from original PDD TRAIN/DEV; no simulator/train')
    mode.add_argument('--pdd-junction-forecast', action='store_true',
                      help='Train an offline PDD human speed/motion forecast experiment; no policy approval or simulator')
    mode.add_argument('--pdd-junction-forecast-temporal', action='store_true',
                      help='Train causal-history and absolute-error junction speed candidates, retaining the original WAIT gate')
    mode.add_argument('--pdd-junction-forecast-traffic', action='store_true',
                      help='Train shared traffic-geometry features and a learned near-stop expert; original gates unchanged')
    mode.add_argument('--pdd-junction-forecast-error-review', action='store_true',
                      help='Replay rejected TRAFFIC114 and review WAIT errors/nearest TRAIN observations; no fitting')
    mode.add_argument('--pdd-junction-forecast-quantile', action='store_true',
                      help='Fit two predeclared lower-quantile near-stop heads; parent moving head/classifier/gates unchanged')
    mode.add_argument('--pdd-junction-stop-go', action='store_true',
                      help='Learn a launch gate with TRAIN driver-OOF qualification; cannot pass by always holding')
    mode.add_argument('--pdd-junction-stop-go-mlp', action='store_true',
                      help='Compare a neural launch gate on the exact archived tree population/folds/protocol')
    mode.add_argument('--pdd-junction-stop-go-history', action='store_true',
                      help='Add causal raw-PDD traffic history to the tree launch gate; same labels/folds/protocol')
    mode.add_argument('--pdd-junction-stop-go-review', action='store_true',
                      help='Review archived TRAIN OOF errors, motion labels and observed stop age; no fitting or simulator')
    mode.add_argument('--pdd-junction-stop-go-clock', action='store_true',
                      help='Learn with causal low-speed span/censoring inputs on the unchanged history153D base')
    mode.add_argument('--pdd-junction-map-survey', action='store_true',
                      help='Check sampled TRAIN junction poses/road references against local Town12; no fitting or simulator')
    mode.add_argument('--pdd-junction-route-review', action='store_true',
                      help='Review bounded local route alternatives against current PDD planner hints; no fitting')
    mode.add_argument('--pdd-route-dataset', action='store_true',
                      help='Export full TRAIN/DEV route196 data, reload for training, and replay public inference conversion')
    mode.add_argument('--pdd-route-plan', action='store_true',
                      help='Train paired96D/196D trajectory-speed models on the verified PDD-only dataset')
    mode.add_argument('--pdd-route-runtime', action='store_true',
                      help='Diagnose guarded route196 checkpoint against fresh Stage1 on two TRAIN sandbox cases')
    mode.add_argument('--pdd-route-frame-cycle', action='store_true',
                      help='Train route-frame residual model from frozen PDD and run guarded paired TRAIN diagnostics in one cycle')
    mode.add_argument('--pdd-route-moving-cycle', action='store_true',
                      help='Deduplicate original C0+junction PDD, train moving-speed head with fixed geometry, run five paired TRAIN cases')
    mode.add_argument('--pdd-route-supported-runtime', action='store_true',
                      help='Diagnose frozen moving checkpoint with explicit existing pace/flow40/guard controller assistance')
    parser.add_argument('--pdd-root', type=Path, default=ROOT / 'PDD',
                        help='Original PDD directory, for junction export, causal history or forecast error review')
    plan_choice = parser.add_mutually_exclusive_group()
    plan_choice.add_argument('--plan-nav', action='store_true',
                             help='Use the navigation-conditioned plan 02 checkpoint with --pdd-plan-runtime')
    plan_choice.add_argument('--plan-factorized', action='store_true',
                             help='Diagnose frozen NEEDS_FIX factorized 03 on TRAIN cases; not a release check')
    plan_choice.add_argument('--plan-lane', action='store_true',
                             help='Track frozen 101D lane-observed plan 04 on the two paired TRAIN cases')
    plan_choice.add_argument('--plan-lane-frame', action='store_true',
                             help='Diagnose frozen NEEDS_FIX lane-frame 05 on the two paired TRAIN cases')
    plan_choice.add_argument('--plan-curve', action='store_true',
                             help='Diagnose frozen NEEDS_FIX curve06 on the two paired TRAIN cases')
    plan_choice.add_argument('--plan-curve-rear', action='store_true',
                             help='Diagnose the same curve06 checkpoint with a rear-axle tracker origin')
    plan_choice.add_argument('--plan-curve-pace', action='store_true',
                             help='Diagnose curve06 with explicit lane-margin guard and gated +2 km/h target bonus')
    plan_choice.add_argument('--plan-pace-transfer', action='store_true',
                             help='Run frozen guarded-pace on six additional Medium/Hard/Extreme TRAIN cases')
    plan_choice.add_argument('--plan-nav40', action='store_true',
                             help='Diagnose explicit maneuver authority and guarded 40 km/h cruise on five Medium/Hard TRAIN cases')
    plan_choice.add_argument('--plan-flow40', action='store_true',
                             help='Compare nav40 with reference-risk-gated progress around benign traffic on the same five cases')
    plan_choice.add_argument('--plan-flow-transfer', action='store_true',
                             help='Freeze successful flow40 and diagnose the three remaining Extreme TRAIN cases')
    plan_choice.add_argument('--plan-visibility-approach', action='store_true',
                             help='Diagnose guarded upstream visibility approach at <=2 km/h on all eight TRAIN cases')
    parser.add_argument('--visibility-extreme-only', action='store_true',
                        help='With --plan-visibility-approach, diagnose only the three Extreme cases; prior five are not retested')
    parser.add_argument('--steps', type=int, default=200)
    args = parser.parse_args()
    if args.visibility_extreme_only and not (args.pdd_plan_runtime and args.plan_visibility_approach):
        parser.error('--visibility-extreme-only requires --pdd-plan-runtime --plan-visibility-approach')
    if (args.plan_nav or args.plan_factorized or args.plan_lane or args.plan_lane_frame
            or args.plan_curve or args.plan_curve_rear or args.plan_curve_pace or args.plan_pace_transfer
            or args.plan_nav40 or args.plan_flow40 or args.plan_flow_transfer or args.plan_visibility_approach) and not args.pdd_plan_runtime:
        parser.error('Plan checkpoint options require --pdd-plan-runtime')
    root_arg = args.main_root
    api = Path('src/platform/agent_input.py')
    project = Path(root_arg).resolve()
    # cmd.exe passes single quotes literally when a PowerShell command is pasted.
    if (sys.platform == 'win32' and root_arg.startswith("'") and root_arg.endswith("'")
            and not (project / api).is_file()):
        project = Path(root_arg[1:-1]).resolve()
    if not (project / api).is_file():
        parser.error(f'--main-root: expected file "{project / api}" (working directory: "{Path.cwd()}")')
    out = ROOT / 'agent4_integration_audit' / ('run_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    sys.stdout.reconfigure(encoding='utf-8')
    sys.path.insert(0, str(project))
    if args.pdd_route_supported_runtime:
        sys.path.insert(0, str(ROOT))
        run_supported_route_trial(project, out)
        return
    if args.pdd_route_moving_cycle:
        sys.path.insert(0, str(ROOT))
        run_moving_route_cycle(project, out, args.pdd_root)
        return
    if args.pdd_route_frame_cycle:
        sys.path.insert(0, str(ROOT))
        run_route_frame_cycle(project, out)
        return
    if args.pdd_route_runtime:
        sys.path.insert(0, str(ROOT))
        run_route196_runtime(project, out)
        return
    if args.pdd_route_plan:
        sys.path.insert(0, str(ROOT))
        train_route196_plan(out)
        return
    if args.pdd_route_dataset:
        sys.path.insert(0, str(ROOT))
        export_route196_dataset(out, args.pdd_root)
        return
    if args.pdd_junction_route_review:
        sys.path.insert(0, str(ROOT))
        survey_pdd_junction_map(out, args.pdd_root, route_review=True)
        return
    if args.pdd_junction_map_survey:
        sys.path.insert(0, str(ROOT))
        survey_pdd_junction_map(out, args.pdd_root)
        return
    if args.pdd_junction_stop_go_clock:
        sys.path.insert(0, str(ROOT))
        train_pdd_junction_forecast(out, temporal_pdd=args.pdd_root, traffic_features=True, clock_stop_go=True)
        return
    if args.pdd_junction_stop_go_review:
        sys.path.insert(0, str(ROOT))
        review_junction_stop_go(out, args.pdd_root)
        return
    if args.pdd_junction_stop_go_history:
        sys.path.insert(0, str(ROOT))
        train_pdd_junction_forecast(out, temporal_pdd=args.pdd_root, traffic_features=True, traffic_stop_go=True)
        return
    if args.pdd_junction_stop_go_mlp:
        sys.path.insert(0, str(ROOT))
        train_pdd_junction_forecast(out, temporal_pdd=args.pdd_root, traffic_features=True, neural_stop_go=True)
        return
    if args.pdd_junction_stop_go:
        sys.path.insert(0, str(ROOT))
        train_pdd_junction_forecast(out, temporal_pdd=args.pdd_root, traffic_features=True, stop_go=True)
        return
    if args.pdd_junction_forecast_quantile:
        sys.path.insert(0, str(ROOT))
        train_pdd_junction_forecast(out, temporal_pdd=args.pdd_root, traffic_features=True, quantile_forecast=True)
        return
    if args.pdd_junction_forecast_error_review:
        sys.path.insert(0, str(ROOT))
        train_pdd_junction_forecast(out, temporal_pdd=args.pdd_root, traffic_features=True, error_review=True)
        return
    if args.pdd_junction_forecast_traffic:
        sys.path.insert(0, str(ROOT))
        train_pdd_junction_forecast(out, temporal_pdd=args.pdd_root, traffic_features=True)
        return
    if args.pdd_junction_forecast_temporal:
        sys.path.insert(0, str(ROOT))
        train_pdd_junction_forecast(out, temporal_pdd=args.pdd_root)
        return
    if args.pdd_junction_forecast:
        sys.path.insert(0, str(ROOT))
        train_pdd_junction_forecast(out)
        return
    if args.pdd_junction_candidates:
        sys.path.insert(0, str(ROOT))
        export_pdd_junction_candidates(args.pdd_root, out)
        return
    if args.pdd_traffic_capacity_shadow:
        sys.path.insert(0, str(ROOT))
        run_traffic_capacity_shadow(project, out)
        return
    if args.pdd_traffic_capacity_probe:
        sys.path.insert(0, str(ROOT))
        run_traffic_capacity_probe(project, out)
        return
    if args.pdd_waiting_pose_review:
        sys.path.insert(0, str(ROOT))
        run_waiting_pose_review(project, out)
        return
    if args.pdd_junction_gap_review:
        sys.path.insert(0, str(ROOT))
        run_junction_gap_review(project, out)
        return
    if args.pdd_lane_curvature_audit:
        sys.path.insert(0, str(ROOT))
        run_pdd_lane_curvature_audit(out)
        return
    if args.pdd_plan_runtime:
        sys.path.insert(0, str(ROOT))
        policy = PDDNavPlanPilot if args.plan_nav else PDDPlanPilot
        if args.plan_lane:
            policy = PDDLanePlanPilot
        if args.plan_lane_frame:
            policy = PDDLaneFramePlanDiagnostic
        if args.plan_curve:
            policy = PDDCurvePlanDiagnostic
        if args.plan_curve_rear:
            policy = PDDCurveRearAxleDiagnostic
        if args.plan_curve_pace:
            policy = PDDCurvePaceDiagnostic
        if args.plan_pace_transfer:
            policy = PDDCurvePaceTransferDiagnostic
        if args.plan_nav40:
            policy = PDDCurveNavigation40Diagnostic
        if args.plan_flow40:
            policy = PDDCurveFlow40Diagnostic
        if args.plan_flow_transfer:
            policy = PDDCurveFlowTransferDiagnostic
        if args.plan_visibility_approach:
            policy = PDDVisibilityApproachDiagnostic
        if args.plan_factorized:
            policy = PDDFactorizedPlanDiagnostic
        if policy.diagnostic_only:
            print(f'DIAGNOSTIC ONLY: {policy.agent_id} has offline_status=NEEDS_FIX; deployment remains ineligible.', flush=True)
        if policy.frozen_transfer:
            frozen_pace_lineage()
            print('Frozen transfer: six additional TRAIN cases, excluding the two completed development cases.', flush=True)
        if policy.flow_transfer:
            frozen_flow_lineage()
            print('Frozen flow40 transfer: three remaining Extreme TRAIN cases; no controller or checkpoint tuning.', flush=True)
        cases = tuple(policy.reference_reports)
        if args.visibility_extreme_only:
            cases = tuple(case for case in cases if case[0] == 'Extreme')
            print('Diagnostic subset: three Extreme cases; five Medium/Hard cases remain historical evidence only.', flush=True)
        run_pdd_traffic_suite(project, out, policy, cases=cases)
        return
    if args.pdd_expanded_suite:
        sys.path.insert(0, str(ROOT))
        run_pdd_traffic_suite(project, out, PDDExpandedForward, cases=tuple(PDDExpandedForward.reference_reports))
        return
    if args.pdd_traffic_suite or args.pdd_forward_suite or args.pdd_forward_transfer or args.pdd_extreme_exit:
        sys.path.insert(0, str(ROOT))
        policy = PDDTrafficOnlyPilot if args.pdd_traffic_suite else PDDForwardPilot
        cases = EXTREME_EXIT_CASES if args.pdd_extreme_exit else None
        if args.pdd_forward_transfer:
            cases = FORWARD_TRANSFER_CASES
        run_pdd_traffic_suite(project, out, policy, cases=cases)
        return
    if args.pdd_speed_traffic or args.pdd_speed_traffic_only:
        sys.path.insert(0, str(ROOT))
        run_pdd_speed_traffic(project, out, PDDTrafficOnlyPilot if args.pdd_speed_traffic_only else PDDSpeedPilot)
        return
    if args.pdd_runtime_smoke:
        sys.path.insert(0, str(ROOT))
        run_pdd_runtime_smoke(project, out)
        return
    sys.path.insert(0, str(ROOT / 'agent4_longitudinal_v3162'))
    from common_v3162 import dependencies, make_features, model_features, validate_v3162
    common, feat, geom = dependencies(ROOT)
    from src.platform.action_adapter import ContinuousBox2Adapter
    from src.platform.agent_input import build_agent_input
    import joblib
    import numpy as np
    from threadpoolctl import threadpool_limits

    report = {
        'audit_type': 'DEVELOPMENT_COMPATIBILITY_NOT_DRIVING_CERTIFICATION',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'main_project': str(project),
        'workspace': str(ROOT),
        'source_hashes': {}, 'checks': {}, 'live_trials': [],
        'integration_verdict': 'NOT_DIRECTLY_INTEGRATED',
        'limitations': [
            'Public-input to native-feature adapter does not exist in the inspected main sources.',
            'Action format acceptance is not evidence of driving quality.',
            'Live trials, when requested, are IDM-driven diagnostic sampling on TRAIN seed 170.',
            'Main launcher/mapsuite end-to-end evaluation is not executed by this audit.',
        ],
    }
    local_files = ['agent4_steering_v313/features_v313.py',
        'agent4_idm_dataset_v312b/collect_v312b.py',
        'agent4_longitudinal_v316/common_v316.py',
        'agent4_longitudinal_v3162/common_v3162.py']
    main_files = ['src/platform/agent_input.py', 'src/platform/action_adapter.py',
        'src/platform/agent.py', 'src/stage4_runtime/bridge.py',
        'src/learning/stage4_bc/checkpoint.py', 'src/learning/stage4_bc/encoder.py',
        'src/launcher/executor.py', 'src/launcher/registry.py']
    for base, names, prefix in [(ROOT, local_files, 'dataset/'), (project, main_files, 'main/')]:
        for name in names:
            report['source_hashes'][prefix + name] = digest(base / name)
    paths = {
        'steering': ROOT / 'agent4_v313_model/agent4_v313_steering.joblib',
        'actor': ROOT / 'agent4_v316_model/agent4_v316_longitudinal.joblib',
        'candidate': ROOT / 'agent4_v3162_model/agent4_v3162_longitudinal.joblib',
    }
    bundles = {k: joblib.load(p) for k, p in paths.items()}
    report['model_hashes'] = {k: digest(p) for k, p in paths.items()}
    original, splits, original_hash = common.original_manifest(ROOT / 'agent4_v312b_idm_dataset')
    common.check_baseline(bundles['steering'], original_hash, feat)
    validate_v3162(bundles['candidate'], original_hash,
                  digest(paths['steering']), digest(paths['actor']), feat)
    report['checks']['model_provenance'] = 'PASS'
    report['model_features'] = {'steering': list(feat.FEATURES),
                                'longitudinal': list(model_features(feat))}
    rows = []
    for ep in original['episodes']:
        if ep['split'] != 'train':
            continue
        with (ROOT / 'agent4_v312b_idm_dataset' / ep['clean_file']).open(encoding='utf-8-sig', newline='') as fh:
            values = list(csv.DictReader(fh))
        for index in sorted({0, len(values)//2, len(values)-1}):
            rows.append(values[index])
    with threadpool_limits(limits=1):
        xs = np.asarray([feat.row_features(row) for row in rows])
        xl = np.asarray([make_features(row, feat) for row in rows])
        steer = np.clip(bundles['steering']['estimator'].predict(xs), -1, 1)
        long = np.clip(bundles['candidate']['estimator'].predict(xl), -1, 1)
    adapter = ContinuousBox2Adapter()
    errors = []
    for s, l in zip(steer, long):
        adapted = adapter.to_canonical([float(s), float(l)]).to_numpy()
        errors.append(float(np.max(np.abs(adapted.astype(float)-[s,l]))))
    report['checks']['native_predictions_through_main_action_adapter'] = {
        'status': 'PASS', 'samples': len(rows), 'split': 'TRAIN_ONLY',
        'max_float32_rounding_error': max(errors),
        'steering_range': [float(min(steer)), float(max(steer))],
        'longitudinal_range': [float(min(long)), float(max(long))],
        'warning': 'Output representation only, not public-input conversion or closed-loop quality',
    }
    report['checks']['native_joblib_matches_main_checkpoint_layout'] = {
        'status': 'FAIL',
        'main_required_files': ['CHECKSUMS.sha256', 'weights.pt', 'model_config.json',
                                'encoder_schema.json', 'normalization.json', 'metadata.json'],
        'local_format': 'scikit-learn estimator bundles serialized with joblib',
        'main_format': 'Torch BC checkpoint directory and 10071D public-state encoder',
        'resolution': 'Add a separately versioned joblib AgentPolicy; do not rename or relabel weights.',
    }
    refs = []
    for directory in ['src', 'configs', 'scripts']:
        for p in (project / directory).rglob('*'):
            if p.suffix not in {'.py', '.json', '.yaml', '.yml'} or not p.is_file():
                continue
            body = p.read_text(encoding='utf-8-sig')
            if any(token in body for token in ['agent4_v313_steering', 'agent4_v3162', 'shared_adapter_v2']):
                refs.append(str(p.relative_to(project)))
    report['checks']['main_references_local_models'] = {'status': 'FAIL' if not refs else 'REVIEW', 'files': refs}
    commands = [
        [sys.executable, '-B', str(ROOT / 'agent4_pdd_v2/test_shared_adapter_v2.py')],
        [sys.executable, '-B', str(ROOT / 'agent4_calibration_v22/test_coordinate_v22.py')],
        [sys.executable, '-B', str(ROOT / 'agent4_calibration_v23/test_route_lookahead_v23.py')],
        [sys.executable, '-B', str(ROOT / 'agent4_calibration_v24/test_lane_v24.py')],
        [sys.executable, '-B', str(ROOT / 'agent4_calibration_v26/test_action_v26.py')],
        [sys.executable, '-B', str(ROOT / 'agent4_longitudinal_v3162/test_v3162.py')],
        [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_agent_contract.py'],
    ]
    report['tests'] = []
    for i, command in enumerate(commands):
        proc = subprocess.run(command, cwd=project if i == len(commands)-1 else ROOT,
                              capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120)
        (out / f'test_{i+1}.log').write_text(proc.stdout + proc.stderr, encoding='utf-8')
        report['tests'].append({'command': command, 'exit_code': proc.returncode,
                                'log': f'test_{i+1}.log'})
        print('TEST', i+1, 'PASS' if proc.returncode == 0 else 'FAIL', flush=True)
    if args.live:
        from metadrive import MetaDriveEnv
        from metadrive.policy.idm_policy import IDMPolicy
        for map_name in ['S', 'C', 'X']:
            trial = {'map': map_name, 'seed': 170, 'steps': 0, 'traffic_density': .15,
                     'driver': 'IDM_TEACHER_DIAGNOSTIC_ONLY', 'errors': [],
                     'circular_steps': 0, 'actor_steps': 0,
                     'max_speed_difference_mps': 0., 'max_heading_difference_rad': 0.,
                     'max_native_distance_to_lane_end_m': 0.,
                     'max_public_current_lane_remaining_m': 0.}
            env = None
            try:
                env = MetaDriveEnv(dict(map=map_name, use_render=False, num_scenarios=186,
                    agent_policy=IDMPolicy, traffic_density=.15, log_level=40,
                    physics_world_step_size=.02, decision_repeat=5, horizon=max(args.steps+10,1000)))
                obs, _ = env.reset(seed=170)
                for step in range(args.steps):
                    pre = geom.state_before_action(env.agent)
                    traffic = list(getattr(env.engine.traffic_manager, 'traffic_vehicles', []))
                    public = build_agent_input(obs, env.agent, traffic, env.agent.navigation,
                                               env.current_map.road_network, step)
                    public_dict = public.to_dict()
                    if 'direct_public_input_rejected' not in trial:
                        try:
                            feat.row_features(public_dict)
                            trial['direct_public_input_rejected'] = False
                        except ValueError as exc:
                            trial['direct_public_input_rejected'] = True
                            trial['direct_public_input_error'] = str(exc)
                        (out / f'{map_name}_paired_first_state.json').write_text(
                            json.dumps({'native': pre, 'public': public_dict}, ensure_ascii=False, indent=2), encoding='utf-8')
                    if public.ego_motion:
                        speed = math.hypot(public.ego_motion.velocity_x_mps, public.ego_motion.velocity_y_mps)
                        trial['max_speed_difference_mps'] = max(trial['max_speed_difference_mps'], abs(speed-pre['speed_mps']))
                        trial['max_heading_difference_rad'] = max(trial['max_heading_difference_rad'],
                            abs(public.ego_motion.lane_heading_error_rad-pre['heading_error_rad']))
                    trial['max_native_distance_to_lane_end_m'] = max(trial['max_native_distance_to_lane_end_m'], pre['distance_to_lane_end_m'])
                    if public.task_context.lane_context:
                        lane = next(x for x in public.task_context.lane_context.lanes if x.side == 'CURRENT')
                        trial['max_public_current_lane_remaining_m'] = max(trial['max_public_current_lane_remaining_m'], lane.remaining_m)
                    trial['circular_steps'] += int(pre['is_circular_lane'])
                    trial['actor_steps'] += int(public.traffic_context.active_count > 0)
                    obs, _, term, trunc, info = env.step([0., 0.])
                    trial['steps'] += 1
                    if term or trunc:
                        trial['termination'] = {k: bool(info.get(k)) for k in ['arrive_dest', 'out_of_road', 'crash']}
                        break
            except Exception:
                trial['errors'].append(traceback.format_exc())
            finally:
                if env is not None:
                    env.close()
            report['live_trials'].append(trial)
            print('LIVE', map_name, 'steps', trial['steps'], 'errors', len(trial['errors']), flush=True)
    report['completed_utc'] = datetime.now(timezone.utc).isoformat()
    report['source_hashes_unchanged'] = all(
        digest((ROOT if key.startswith('dataset/') else project) / key.split('/',1)[1]) == value
        for key, value in report['source_hashes'].items())
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('REPORT', out / 'report.json', flush=True)


if __name__ == '__main__':
    main()
