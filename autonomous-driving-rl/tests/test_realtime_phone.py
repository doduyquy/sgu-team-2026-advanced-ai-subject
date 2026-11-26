"""Bounded AI execution, stale-action braking and real H.264 software fallback."""
from collections import deque
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import av
import numpy as np

from src.platform import AgentDecision, CanonicalActionV1, ContinuousBox2Adapter
from src.visualization.async_policy import AsyncPolicy
from src.visualization.mobile_control import PhoneControl, HumanPolicy
from src.visualization.phone_encoder import PhoneEncoder
from src.visualization.mobile_server import MobileHost, FrameMailbox
from src.visualization.race_mode import Racer
from src.visualization.realtime_race import RealtimeRace
from src.visualization.race_environment import DecisionObservation
import test_race_mode as fixtures


def slow_worker(factory, context, seed, requests, responses):
    responses.put(('ready',))
    while True:
        job=requests.get()
        if job is None: return
        tick,captured,inputs=job
        time.sleep(.2)
        responses.put(('decision',tick,captured,AgentDecision([0.,.5]),200.,None))


class WorkerTests(unittest.TestCase):
    def test_spawned_race_worker_uses_pace_override_for_real_bfs_decisions(self):
        from dataclasses import replace
        from test_agent_io_overlay import public_input
        from src.platform import AgentPublicEpisodeContext, CoreObservationV1
        from src.visualization.race_rules import race_planner_options
        inputs=public_input()
        features=inputs.core_observation.features.copy();features[3]=13./81.
        inputs=replace(inputs,core_observation=CoreObservationV1(features))
        options=race_planner_options(80.)
        worker=AsyncPolicy('src.agents.graph_search_agents:BFSAgent',
            AgentPublicEpisodeContext(action_adapter_id='discrete9_lowbranch_v1'),planner_options=options)
        try:
            deadline=time.perf_counter()+8
            while not worker.ready and not worker.error and time.perf_counter()<deadline:
                worker.poll();time.sleep(.01)
            self.assertIsNone(worker.error)
            self.assertEqual(worker.applied_planner_options,options)
            worker.submit(1,inputs)
            result=None
            while not result and not worker.error and time.perf_counter()<deadline:
                result=worker.poll();time.sleep(.01)
            self.assertIsNotNone(result,worker.error)
            # The Stage 2 controller may steer toward a tied BFS lattice path;
            # the worker override controls cruise speed independently of that.
            self.assertEqual(result[2].action_payload // 3, 2)
            self.assertAlmostEqual(result[2].diagnostics['target_speed_kmh'], 15.)
        finally:worker.close()

    def test_busy_ai_never_blocks_and_pending_snapshot_is_replaced(self):
        with patch.dict(os.environ,{'PYTHONUTF8':'on'}):
            worker=AsyncPolicy('unused',None,worker_target=slow_worker)
            self.assertEqual(os.environ['PYTHONUTF8'],'on')
        try:
            deadline=time.perf_counter()+8
            while not worker.ready and time.perf_counter()<deadline:
                worker.poll();time.sleep(.01)
            self.assertTrue(worker.ready)
            worker.submit(1,'first')
            started=time.perf_counter()
            for tick in range(2,101): worker.submit(tick,f'input-{tick}')
            self.assertLess(time.perf_counter()-started,.05)
            seen=[]
            while len(seen)<2 and time.perf_counter()<deadline:
                result=worker.poll()
                if result: seen.append(result[0])
                time.sleep(.005)
            self.assertEqual(seen,[1,100])
            self.assertEqual(worker.last_inputs,'input-100')
        finally:
            worker.close()
        self.assertFalse(worker.process.is_alive())

    def test_worker_startup_error_is_visible_and_cleanup_is_idempotent(self):
        worker=AsyncPolicy('module_that_does_not_exist:Agent',None)
        try:
            deadline=time.perf_counter()+8
            while not worker.error and time.perf_counter()<deadline:
                worker.poll();time.sleep(.01)
            self.assertIn('module_that_does_not_exist',worker.error)
        finally:
            worker.close();worker.close()


class RealtimeTests(unittest.TestCase):
    def test_observation_refreshes_at_decision_time_and_resets_cleanly(self):
        source=MagicMock()
        source.observe.side_effect=['initial','decision','after-reset']
        observation=DecisionObservation(source)
        self.assertEqual(observation.observe(None),'initial')
        for _ in range(4): self.assertEqual(observation.observe(None),'initial')
        self.assertEqual(observation.refresh(None),'decision')
        self.assertEqual(observation.observe(None),'decision')
        observation.reset(None,None)
        self.assertEqual(observation.observe(None),'after-reset')
        self.assertEqual(source.observe.call_count,3)

    def test_physics_samples_player_without_waiting_for_ai_and_brakes_stale_ai(self):
        app=fixtures.TestRaceMode().running_app()
        control=PhoneControl();control.claim('phone')
        control.receive('phone',{'type':'input','seq':1,'steering':.4,'throttle':1.})
        app.phone_control=control
        player=Racer(5,HumanPolicy(control),MagicMock(),ContinuousBox2Adapter())
        app.racers['racer_5']=player
        app.env.agents['racer_5']=MagicMock(speed_km_h=10.)
        keys=tuple(app.racers)
        app.env.step.return_value=({}, {},dict.fromkeys(keys,False),dict.fromkeys(keys,False),
                                   {key:{'route_completion':.1} for key in keys})
        app._finish_racer=MagicMock()
        app._next_decision=time.perf_counter()
        runtime=RealtimeRace.__new__(RealtimeRace)
        runtime.app=app;runtime.times=deque();runtime.applied=deque();runtime.log=StringIO()
        runtime._poll=MagicMock();runtime.request_decisions=MagicMock()
        runtime.actions={'racer_0':(CanonicalActionV1(0.,1.),time.perf_counter()-.5,0),
                         'racer_1':(CanonicalActionV1(.2,.7),time.perf_counter(),0)}
        runtime.stale_count=0;runtime.last_hud=0
        runtime.next_render=0
        runtime.render_times=deque()
        runtime.step()
        actions=app.env.step.call_args.args[0]
        np.testing.assert_allclose(actions['racer_0'],[0.,-.8])
        np.testing.assert_allclose(actions['racer_1'],[.2,.6])
        np.testing.assert_allclose(actions['racer_5'],[-.4,.6])
        for key in ('racer_0','racer_1'): app.racers[key].policy.act.assert_not_called()
        self.assertEqual(app.steps,1)
        self.assertEqual(runtime.stale_count,1)
        self.assertAlmostEqual(json.loads(runtime.log.getvalue())['simulation_time_s'],.02)

    def test_desktop_ai_moves_each_physics_tick_without_calling_policy_on_render_thread(self):
        app=fixtures.TestRaceMode().running_app()
        app.env.config={}
        runtime=RealtimeRace.__new__(RealtimeRace)
        runtime.app=app;runtime.times=deque();runtime.applied=deque();runtime.applied_ticks=deque()
        runtime.render_times=deque();runtime.log=StringIO()
        runtime._poll=MagicMock();runtime.request_decisions=MagicMock()
        runtime.stale_count=0;runtime.last_hud=0;runtime.next_render=0
        runtime.actions={key:(CanonicalActionV1(0.,.6),time.perf_counter(),0) for key in app.racers}
        app._next_decision=time.perf_counter()
        runtime.step()
        self.assertEqual(app.env.config['decision_repeat'],1)
        self.assertEqual(app.steps,1)
        self.assertEqual(len(runtime.render_times),1)
        for racer in app.racers.values(): racer.policy.act.assert_not_called()
        for value in app.env.step.call_args.args[0].values():
            np.testing.assert_allclose(value,[0.,.6])

    def test_pause_resume_resets_physics_deadline_and_replaces_old_ai_snapshot(self):
        runtime=RealtimeRace.__new__(RealtimeRace)
        runtime.app=MagicMock(steps=125)
        runtime.next_submit={'racer_0':150}
        runtime.request_decisions=MagicMock()
        with patch('src.visualization.realtime_race.time.perf_counter',return_value=10.):
            runtime.reset_clock()
        self.assertEqual(runtime.clock_sim,2.5)
        self.assertEqual(runtime.app._next_decision,10.)
        self.assertEqual(runtime.next_submit,{})
        runtime.request_decisions.assert_called_once()

    def test_start_waits_for_ai_workers_without_waiting_for_a_human_policy(self):
        runtime=RealtimeRace.__new__(RealtimeRace)
        runtime.app=fixtures.TestRaceMode().running_app()
        for r in runtime.app.racers.values():r.policy.ready=False
        self.assertFalse(runtime.workers_ready())
        for r in runtime.app.racers.values():r.policy.ready=True
        runtime.app.racers['racer_5']=Racer(5,object(),MagicMock(),MagicMock())
        self.assertTrue(runtime.workers_ready())


class EncoderTests(unittest.TestCase):
    def test_failed_nvidia_probe_does_not_open_host_or_publish_qr(self):
        with tempfile.TemporaryDirectory() as directory:
            host=MobileHost(PhoneControl(),FrameMailbox(),directory,port=0)
            self.assertEqual(host.encoder_preference,'nvenc')
            with patch.object(PhoneEncoder,'encode',side_effect=RuntimeError('NVENC unavailable')):
                with self.assertRaisesRegex(RuntimeError,'NVENC unavailable'): host.start()
            self.assertIsNone(host.thread)
            self.assertFalse((Path(directory)/'phone-qr.png').exists())

    def test_cpu_h264_decodes_and_resolution_change_produces_new_stream_headers(self):
        encoder=PhoneEncoder('cpu')
        decoder=av.CodecContext.create('h264','r')
        for i,(width,height) in enumerate(((320,180),(640,360),(320,180))):
            rgb=np.full((height,width,3),[30,100,180],dtype=np.uint8)
            packet=encoder.encode(rgb,i*3000)
            self.assertIsNotNone(packet)
            decoded=decoder.decode(packet)
            self.assertEqual((decoded[0].width,decoded[0].height),(width,height))
            self.assertLess(abs(decoded[0].to_ndarray(format='rgb24').mean()-rgb.mean()),5)
        self.assertFalse(encoder.stats()['hardware_encode'])

    def test_failed_hardware_falls_back_without_editing_aiortc(self):
        encoder=PhoneEncoder('auto')
        original=encoder._open
        def open_codec(name,width,height):
            if name!='libx264': raise RuntimeError('Hardware unavailable')
            return original(name,width,height)
        with patch.object(encoder,'_open',side_effect=open_codec):
            packet=encoder.encode(np.zeros((180,320,3),dtype=np.uint8),0)
        self.assertIsNotNone(packet)
        self.assertEqual(encoder.name,'libx264')
        self.assertEqual(set(encoder.failed),{'h264_nvenc'})

    def test_explicit_nvenc_never_falls_back_to_intel_or_cpu(self):
        encoder=PhoneEncoder('nvenc')
        with patch.object(encoder,'_open',side_effect=RuntimeError('NVIDIA unavailable')) as open_codec:
            with self.assertRaises(RuntimeError):
                encoder.encode(np.zeros((180,320,3),dtype=np.uint8),0)
        self.assertEqual(open_codec.call_count,1)
        self.assertEqual(open_codec.call_args.args[0],'h264_nvenc')


if __name__=='__main__': unittest.main()
