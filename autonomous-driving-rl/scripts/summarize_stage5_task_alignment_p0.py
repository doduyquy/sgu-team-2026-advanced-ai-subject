import csv,gzip,hashlib,json,math
from pathlib import Path
root=Path('runs/stage5_task_alignment_p0')
out=Path('results/stage5/task_alignment_p0')
raw=root/'runtime_extended.json'
d=json.loads(raw.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def csv_sha(path):
 return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
def brief(p):
 rows=p['rows']; points=[point for row in rows for point in row['points']]
 seams=[seam for row in rows for seam in row['seams']]
 return {**{k:v for k,v in p.items() if k!='rows'},
 'finite_409':all(x['finite_409'] for x in rows),
 'max_frame_error_m':max(x['frame_error_m'] for x in points),
 'max_heading_error_rad':max(x['heading_error_rad'] for x in points),
 'max_reference_seam_gap_m':max(x['gap_m'] for x in seams),
 'max_reference_seam_tangent_rad':max(x['tangent_rad'] for x in seams),
 'max_speed_encoding_error':max(x['speed_encoding_error'] for x in rows),
 'max_latest_action_error':max(x['previous_action_error'] for x in rows),
 'max_native_navigation_error':max(x['navigation_error'] for x in rows),
 'traffic_padding_zero':all(x['traffic_padding_zero'] for x in rows),
 'task_padding_zero':all(x['task_padding_zero'] for x in rows),
 'maximum_traffic_actors':max(x['traffic_valid'] for x in rows),
 'min_lidar':min(x['lidar_min'] for x in rows),
 'min_first_waypoint_x_m':min(x['task'][0] for x in rows if x['valid_waypoints']),
 'checkpoint_changes':sum(a['checkpoints']!=b['checkpoints'] for a,b in zip(rows,rows[1:])),
 'terminal':rows[-1].get('outcome')}
parts={key:brief(d[key]) for key in ('historical_clean_replay','adjacent_lane_probe','localization_stress_probe',
                                    'traffic_probe','adjacent_lane_curve')}
curve_rows = [row for row in d['adjacent_lane_curve']['rows']
              if row['lane_type']=='CircularLane' and row['current_minus_reference_station_m'] is not None]
curve_witness = max(curve_rows, key=lambda x:abs(x['current_minus_reference_station_m']), default=None)
lane_one_witness = max((row for row in curve_rows if row['lane'][-1] == 1),
                       key=lambda x:abs(x['current_minus_reference_station_m']), default=None)
pulses={}
for steer,p in d['pulses'].items():
 r=p['rows']
 pulses[steer]={**brief(p),'kph_reset_accelerate_coast_brake':[r[i]['speed'] for i in (0,14,18,22)],
 'heading_delta_deg':math.degrees((r[18]['heading']-r[14]['heading']+math.pi)%(2*math.pi)-math.pi)}
r=d['historical_clean_replay']['rows']
select={0,len(r)-1}
select.update(i for i in range(1,len(r)) if r[i]['lane']!=r[i-1]['lane'] or r[i]['checkpoints']!=r[i-1]['checkpoints'])
selected=[('clean_replay',r[i]) for i in sorted(select|{max(0,i-1) for i in select}|{104,138,200,340,400})]
for key in ('adjacent_lane_probe','localization_stress_probe','traffic_probe','adjacent_lane_curve'):
 rows=d[key]['rows']
 selected.extend((key,rows[i]) for i in sorted({0,len(rows)-1,min(range(len(rows)),key=lambda i:rows[i]['lidar_min'])}))
for witness in (curve_witness, lane_one_witness):
 if witness:
  selected.append(('adjacent_lane_curve',witness))
with (out/'waypoint_trace_excerpt.csv').open('w',newline='',encoding='utf-8') as f:
 writer=csv.writer(f,lineterminator='\n')
 writer.writerow(('trace','step','lane','checkpoints','lane0_gap_m','first_x_m','first_y_m','first_heading_rad',
                  'valid_waypoints','speed_kmh','lidar_min','traffic_actors','first_world_xy_m','station_delta_m'))
 for key,x in selected:
  writer.writerow((key,x['step'],str(x['lane']),str(x['checkpoints']),x['lane0_gap_m'],*x['task'][:3],
                   x['valid_waypoints'],x['speed'],x['lidar_min'],x['traffic_valid'],
                   str(x['points'][0]['world']) if x['points'] else '',x['current_minus_reference_station_m']))
history=Path('runs/stage5_meter_reward/s5_5_meter_reward_seed101_20261120')
experiment=Path('results/stage5/s5_5_meter_reward_seed101_20261120')
expected=json.loads((experiment/'analysis_summary.json').read_text())['checkpoint_bundles']
bundles={}
for arm in 'ABC':
 for step in (10000,30000):
  p=history/'arms'/arm/'checkpoints'/f'step_{step:06d}'/'model.zip'
  expect=expected[arm][str(step)]['model_sha256']
  observed=sha(p) if p.exists() else None
  bundles[f'{arm}-{step}']={'sha256':observed,'expected':expect,'status':'PASS' if observed==expect else 'MISSING_OR_MISMATCH'}
traces={}
with gzip.open(history/'preflight/reward_traces.jsonl.gz','rt',encoding='utf-8') as f:
 for line in f:
  x=json.loads(line); traces.setdefault(x['trace_id'],[]).append(x)
end=min(min(map(len,traces.values()))-1,104)
start=end-50
credit={}
for label,rows in traces.items():
 window=rows[start:end]
 credit[label]={'window_start_inclusive_zero_based':start,'window_end_exclusive':end,
                'mean_speed_kmh':sum(x['speed_km_h'] for x in window)/len(window),
                'window_reward_v1':sum(x['reward_v1_env'] for x in window),
                'discounted_tail_at_window_start':sum(.99**i*x['reward_v1_env'] for i,x in enumerate(rows[start:]))}
summary={
 **{k:v for k,v in d.items() if not isinstance(v,dict)},
 'scope':'TRAIN only; no SAC training; replayed fixed reference actions in simulator for audit only',
 'raw_path':raw.as_posix(),'raw_sha256':sha(raw),
 'pulses':pulses,'scenes':parts,'reward':d['reward'],'matched_prefix_credit':credit,
 'checkpoint_inventory':bundles,
 'adjacent_curve_station':{'rows':len(curve_rows),
                           'negative_first_waypoint_count':sum(row['task'][0] < 0 for row in curve_rows),
                           'max_abs_witness':curve_witness,'lane_one_witness':lane_one_witness},
 'frozen_policy_audit':json.loads((root/'frozen_policies.json').read_text()),
 'frozen_policy_audit_sha256':sha(root/'frozen_policies.json'),
 'replay_buffers_found':[p.as_posix() for p in history.rglob('*replay*.pkl')],
 'artifact_hashes_lf_normalized':{p.name:csv_sha(p) for p in
                                  (out/'input_feature_manifest.csv',out/'waypoint_trace_excerpt.csv')},
}
(out/'evidence.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'credit':credit,'checkpoint_status':{k:v['status'] for k,v in bundles.items()},
                  'raw_sha256':summary['raw_sha256'],'artifacts':summary['artifact_hashes_lf_normalized'],
                  'threshold':d['reward']['positive_step_threshold_ideal_kmh']}))
