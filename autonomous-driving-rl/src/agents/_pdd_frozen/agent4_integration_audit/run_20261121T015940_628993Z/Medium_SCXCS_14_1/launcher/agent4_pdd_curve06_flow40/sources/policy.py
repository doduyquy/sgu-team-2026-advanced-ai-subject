"""Read-only compatibility audit of local models and the main project's public API.

Writes a new evidence directory in this workspace. Optional simulator trials use
TRAIN seed 170 and IDM control to compare extractors; they are NOT model-driving
or benchmark results. The PDD96 launcher smoke uses the existing rule policy
on one TRAIN sandbox case. The speed pilot compares a frozen PDD checkpoint
with Stage 1 in matched TRAIN traffic. No training or main-project writes occur.
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
                                  last_pdd_change_step=last_change, steering_differences_from_stage1=steering_changes)
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
                      'pdd_timeout': candidate['timeout'].lower() == 'true',
                      'pdd_route_completion': float(candidate['final_route_completion']),
                      'pdd_p95_act_ms': hybrid['p95_act_ms'],
                      'pdd_mean_speed_kmh': float(candidate['mean_speed_kmh']),
                      'pdd_max_speed_kmh': float(candidate['max_speed_kmh']),
                      'pdd_episode_time_s': float(candidate['episode_time_s']),
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
            summary['unevaluated_extreme_cases'] = [f'sandbox/{tier}/{seq}/{geometry}/{seed}'
                for tier, seq, geometry, seed in PDDExpandedForward.reference_reports if tier == 'Extreme']
            summary['remaining_extreme_issue'] = 'Previous frozen-transfer BLOCKED timeouts remain unresolved; raising cruise speed does not release blocked traffic.'
        if policy.flow40_enabled:
            summary['flow40_parameters'] = dict(FLOW40_PARAMETERS)
            summary['capabilities_under_test'] += '; bounded progress around reference-assessed benign traffic with TTC/headway gates'
        if policy.diagnostic_only:
            summary['scope'] = policy.diagnostic_scope
            summary['deployment_eligible'] = False
        if policy.lane_conditioned and not policy.diagnostic_only:
            summary['scope'] = 'PDD_LANE_PLAN_PAIRED_TRAIN_DIAGNOSTIC'
            summary['deployment_eligible'] = False
        if (policy.diagnostic_only or policy.lane_conditioned) and policy.compare_previous_plan_run:
            if policy.flow40_enabled:
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
                                                             'baseline_clean': previous['pdd_clean']}
    write_json(out / 'report.json', summary, exclusive=True)
    print(f"Paired TRAIN cases {len(complete)}/{len(cases)}, clean arrivals Stage 1={summary['baseline_clean']} "
          f"PDD={summary['pdd_clean']}. Read {out / 'report.json'}", flush=True)
    if summary['pdd_time_weighted_mean_speed_kmh'] is not None:
        print(f"Measured speed including stops: mean={summary['pdd_time_weighted_mean_speed_kmh']:.2f} km/h, "
              f"max={summary['pdd_max_speed_kmh']:.2f} km/h", flush=True)
    if len(complete) < len(cases):
        raise SystemExit(2)


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
    parser.add_argument('--steps', type=int, default=200)
    args = parser.parse_args()
    if (args.plan_nav or args.plan_factorized or args.plan_lane or args.plan_lane_frame
            or args.plan_curve or args.plan_curve_rear or args.plan_curve_pace or args.plan_pace_transfer
            or args.plan_nav40 or args.plan_flow40) and not args.pdd_plan_runtime:
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
        if args.plan_factorized:
            policy = PDDFactorizedPlanDiagnostic
        if policy.diagnostic_only:
            print(f'DIAGNOSTIC ONLY: {policy.agent_id} has offline_status=NEEDS_FIX; deployment remains ineligible.', flush=True)
        if policy.frozen_transfer:
            frozen_pace_lineage()
            print('Frozen transfer: six additional TRAIN cases, excluding the two completed development cases.', flush=True)
        run_pdd_traffic_suite(project, out, policy, cases=tuple(policy.reference_reports))
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
