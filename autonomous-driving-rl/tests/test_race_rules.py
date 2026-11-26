"""Race fairness: identical safety rules, clean finishes and no recovery exploit."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import unittest

from src.visualization.race_rules import classify_race_outcome, ordered_racers, race_positions, limit_race_action, RACE_SPEED_LIMIT_KMH, cap_race_velocity
from src.visualization.race_environment import RaceEnvironment
from src.visualization.race_session import RaceComparison
import test_race_mode as fixtures


class RaceRulesTests(unittest.TestCase):
    def test_race_pace_raises_planner_cruise_without_changing_canonical_factories(self):
        import numpy as np
        from src.visualization.race_rules import race_planner_options, apply_race_planner_options
        from src.agents.astar_agent import AStarAgent
        from src.agents.graph_search_agents import BFSAgent, DijkstraAgent
        from src.agents.hybrid_astar_agent import HybridAStarAgent
        from src.agents.path_following import follow_local_path
        inputs=SimpleNamespace(core_observation=SimpleNamespace(ego_state=[0.,0.,0.,13./81.]))
        for factory in (AStarAgent,BFSAgent,DijkstraAgent,HybridAStarAgent):
            policy=factory();original=policy.config
            before=follow_local_path(np.array([[6.,0.]]),inputs,policy.config,{})
            apply_race_planner_options(policy,race_planner_options(80.))
            after=follow_local_path(np.array([[6.,0.]]),inputs,policy.config,{})
            self.assertEqual(before.action_payload,4)
            self.assertEqual(after.action_payload,7)
            self.assertEqual(original['target_normalized_speed'],.15)
            self.assertEqual(factory().config['target_normalized_speed'],.15)
            policy.close()

    def test_speed_governor_tapers_acceleration_and_preserves_driver_braking(self):
        from src.platform import CanonicalActionV1
        for practice in (True,False):
            half=limit_race_action(CanonicalActionV1(.2,.6),practice,RACE_SPEED_LIMIT_KMH-.5)
            self.assertAlmostEqual(half.throttle_brake,.3)
            at_limit=limit_race_action(CanonicalActionV1(.2,.6),practice,RACE_SPEED_LIMIT_KMH)
            self.assertEqual(at_limit.throttle_brake,0.)
            overspeed=limit_race_action(CanonicalActionV1(.2,.6),practice,RACE_SPEED_LIMIT_KMH+2.)
            self.assertLess(overspeed.throttle_brake,0.)
            braking=limit_race_action(CanonicalActionV1(.2,-.8),practice,RACE_SPEED_LIMIT_KMH+2.)
            self.assertEqual(braking.throttle_brake,-.8)
            self.assertEqual(braking.steering,.2)

    def test_speed_cap_keeps_heading_and_leaves_observation_speed_reference_intact(self):
        vehicle=SimpleNamespace(speed_km_h=50.,velocity=(3.,4.),
                                config={'max_speed_km_h':80},set_velocity=MagicMock())
        cap_race_velocity(vehicle)
        vehicle.set_velocity.assert_called_once_with((3.,4.),value=RACE_SPEED_LIMIT_KMH/3.6)
        self.assertEqual(vehicle.config['max_speed_km_h'],80)
        vehicle.speed_km_h=10.;vehicle.set_velocity.reset_mock()
        cap_race_velocity(vehicle)
        vehicle.set_velocity.assert_not_called()

    def test_solid_lines_fail_before_arrival_broken_lines_allow_lane_changes(self):
        for name in ('on_yellow_continuous_line','on_white_continuous_line'):
            with self.subTest(flag=name):
                result=classify_race_outcome({name:True,'arrive_dest':True})
                self.assertTrue(result.is_done)
                self.assertEqual(result.reason,'SOLID_LINE')
                self.assertFalse(result.clean_success)
        self.assertFalse(classify_race_outcome({'on_broken_line':True}).is_done)
        self.assertTrue(classify_race_outcome({'on_broken_line':True,'arrive_dest':True}).clean_success)

    def test_collision_and_off_road_beat_arrival_for_every_controller(self):
        for flag,reason in (('crash_vehicle','CRASH_VEHICLE'),('crash_object','CRASH_OBJECT'),
                            ('crash_sidewalk','CRASH_SIDEWALK'),('out_of_road','OUT_OF_ROAD')):
            with self.subTest(flag=flag):
                result=classify_race_outcome({flag:True,'arrive_dest':True,'on_yellow_continuous_line':True})
                self.assertEqual(result.reason,reason)
                self.assertTrue(result.is_done)
                self.assertFalse(result.clean_success)

    def test_practice_continues_but_never_earns_a_clean_win(self):
        flags={'crash_vehicle':True,'out_of_road':True,'on_yellow_continuous_line':True}
        self.assertFalse(classify_race_outcome(flags,practice=True).is_done)
        self.assertFalse(classify_race_outcome(dict(flags,arrive_dest=True),practice=True).clean_success)

    def test_eliminated_car_with_more_progress_cannot_outrank_active_car(self):
        failed=SimpleNamespace(index=5,route=.99,summary={'clean_success':False})
        active=SimpleNamespace(index=0,route=.1,summary=None)
        slow=SimpleNamespace(index=1,route=1.,summary={'clean_success':True,'simulation_time_s':20.})
        fast=SimpleNamespace(index=2,route=1.,summary={'clean_success':True,'simulation_time_s':15.})
        self.assertEqual(ordered_racers([failed,active,slow,fast]),[fast,slow,active,failed])

    def test_recovery_is_rejected_at_both_command_and_environment_boundaries(self):
        app=fixtures.TestRaceMode().running_app()
        app.selection.set_mode('human')
        app.racers['racer_5']=SimpleNamespace(summary=None)
        app.phone_control=MagicMock()
        app._phone_command('recover',None)
        app.env.recover.assert_not_called()
        env=RaceEnvironment.__new__(RaceEnvironment)
        self.assertFalse(env.recover('racer_5'))
        app.selection.set_mode('practice')
        app.recoveries=0
        app.env.recover.return_value=True
        app._phone_command('recover',None)
        app.env.recover.assert_called_once_with('racer_5')
        self.assertEqual(app.recoveries,1)

    def test_continuous_player_and_ai_get_same_limits_and_practice_is_unranked(self):
        from src.platform import CanonicalActionV1
        for steering in (-1.,1.):
            for throttle in (-1.,1.):
                action=CanonicalActionV1(steering,throttle)
                limited=limit_race_action(action)
                self.assertEqual(abs(limited.steering),.6)
                self.assertEqual(limited.throttle_brake,.6 if throttle>0 else -.8)
                self.assertIs(limit_race_action(action,practice=True),action)

    def test_equal_finish_times_share_rank_and_failures_have_no_rank(self):
        racers=[SimpleNamespace(index=i,vehicle_id=f'racer_{i}',route=1.,
            summary={'clean_success':i<2,'simulation_time_s':15.}) for i in range(3)]
        self.assertEqual(race_positions(racers),{'racer_0':1,'racer_1':1,'racer_2':None})

    def test_numpy_route_progress_produces_json_serializable_ranks(self):
        import json
        import numpy as np
        racers=[SimpleNamespace(index=i,vehicle_id=f'racer_{i}',route=np.float64(i/10),summary=None)
                for i in range(2)]
        ranks=race_positions(racers)
        self.assertEqual(ranks,{'racer_0':2,'racer_1':1})
        self.assertEqual(json.loads(json.dumps(ranks)),ranks)
        self.assertTrue(all(type(value) is int for value in ranks.values()))

    def test_realtime_applies_same_safety_rules_and_actuator_limits_to_player_and_ai(self):
        from collections import deque
        from io import StringIO
        import time
        import numpy as np
        from src.platform import CanonicalActionV1, ContinuousBox2Adapter
        from src.visualization.mobile_control import PhoneControl, HumanPolicy
        from src.visualization.realtime_race import RealtimeRace
        for flag,reason in (('on_yellow_continuous_line','SOLID_LINE'),('crash_vehicle','CRASH_VEHICLE')):
            app=fixtures.TestRaceMode().running_app()
            app.selection.set_mode('human')
            player=app.racers.pop('racer_1');player.index=5
            control=PhoneControl();control.claim('phone')
            control.receive('phone',{'type':'input','seq':1,'steering':1.,'throttle':1.})
            player.policy=HumanPolicy(control);player.adapter=ContinuousBox2Adapter()
            app.phone_control=control;app.racers['racer_5']=player
            app.env.agents['racer_5']=app.env.agents.pop('racer_1')
            app.comparison=RaceComparison((0,5),app.case.to_dict(),1)
            keys=tuple(app.racers)
            app.env.step.return_value=({}, {},dict.fromkeys(keys,False),dict.fromkeys(keys,False),
                {key:{flag:True,'arrive_dest':True,'route_completion':1.} for key in keys})
            app._next_decision=time.perf_counter()
            runtime=RealtimeRace.__new__(RealtimeRace)
            runtime.app=app;runtime.times=deque();runtime.applied=deque();runtime.log=StringIO()
            runtime._poll=MagicMock();runtime.request_decisions=MagicMock()
            runtime.actions={'racer_0':(CanonicalActionV1(-1.,1.),time.perf_counter(),0)}
            runtime.stale_count=0;runtime.last_hud=0;runtime.next_render=0;runtime.render_times=deque()
            runtime.step()
            for key,action in app.env.step.call_args.args[0].items():
                np.testing.assert_allclose(action,[-.6,.6])
                self.assertEqual(app.racers[key].summary['reason'],reason)
                self.assertFalse(app.racers[key].summary['clean_success'])

    @patch('src.platform.build_agent_input',return_value=SimpleNamespace(step_index=0))
    def test_lockstep_treats_player_and_ai_solid_line_and_clean_finish_identically(self,_):
        for event,reason,clean in (({'on_white_continuous_line':True},'SOLID_LINE',False),
                                   ({'arrive_dest':True},'SUCCESS',True)):
            app=fixtures.TestRaceMode().running_app()
            player=app.racers.pop('racer_1');player.index=5
            app.racers['racer_5']=player
            app.env.agents['racer_5']=app.env.agents.pop('racer_1')
            app.obs['racer_5']=app.obs.pop('racer_1')
            app.comparison=RaceComparison((0,5),app.case.to_dict(),1)
            keys=tuple(app.racers)
            app.env.step.return_value=(dict(app.obs),{},dict.fromkeys(keys,False),dict.fromkeys(keys,False),
                {key:dict(event,route_completion=.5) for key in keys})
            app._step()
            for racer in app.racers.values():
                self.assertEqual(racer.summary['reason'],reason)
                self.assertEqual(racer.summary['clean_success'],clean)
                self.assertTrue(racer.summary['ranked'])


if __name__=='__main__': unittest.main()
