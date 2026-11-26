"""Player input, six-car integration and real local WebRTC negotiation."""

import asyncio
import json
import logging
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from src.visualization.mobile_control import PhoneControl, HumanPolicy
from src.visualization.mobile_server import MobileHost, FrameMailbox
from src.visualization.race_session import RaceSelection, RaceComparison, HUMAN_INDEX
from src.visualization.race_environment import build_grid, RaceEnvironment
from src.visualization.race_vehicle_models import build_car, CAR_MODELS
import test_race_mode as fixtures


class PhoneInputTests(unittest.TestCase):
    def setUp(self):
        self.now=0.
        self.control=PhoneControl(clock=lambda:self.now)
        self.control.claim('one')

    def send(self,seq=1,steering=.5,throttle=1.,owner='one'):
        return self.control.receive(owner,{'type':'input','seq':seq,'steering':steering,'throttle':throttle})

    def test_only_owner_and_increasing_finite_inputs_can_drive(self):
        self.assertFalse(self.control.claim('two'))
        self.assertFalse(self.send(owner='two'))
        self.assertTrue(self.send())
        self.assertFalse(self.send(throttle=-1))
        self.assertEqual(self.control.sample(),(.5,1.))
        for value in (float('nan'),float('inf'),1.01,'1',True,None):
            self.assertFalse(self.send(seq=2,steering=value))
        self.assertEqual(self.control.sample(),(.5,1.))

    def test_stale_input_disconnect_and_reset_brake_without_waiting_for_phone(self):
        self.send()
        self.now=.46
        self.assertEqual(self.control.sample(),(0.,-1.))
        self.send(seq=2)
        self.control.clear()
        self.assertEqual(self.control.sample(),(0.,-1.))
        self.send(seq=3)
        self.control.release('one')
        self.assertEqual(self.control.sample(),(0.,-1.))
        self.assertTrue(self.control.claim('two'))
        self.assertTrue(self.send(seq=1,owner='two'))

    def test_driver_right_maps_to_metadrive_negative_steering(self):
        self.send(steering=1.)
        self.assertEqual(HumanPolicy(self.control).act(None).action_payload,[-1.,1.])

    def test_reserving_an_offer_is_not_a_connected_driver_until_both_channels_open(self):
        self.assertFalse(self.control.snapshot()['connected'])
        self.assertTrue(self.control.snapshot()['pairing'])
        self.control.transport('one','connected',['commands'])
        self.assertFalse(self.control.snapshot()['connected'])
        self.control.transport('one','connected',['commands','drive'])
        self.assertTrue(self.control.snapshot()['connected'])
        self.control.release('one')
        self.control.claim('two')
        self.control.transport('one','connected',['commands','drive'])
        self.assertFalse(self.control.snapshot()['connected'])

    def test_only_explicit_commands_enter_main_thread_queue(self):
        self.assertFalse(self.control.receive('one',{'type':'command','command':'quit'}))
        self.assertTrue(self.control.receive('one',{'type':'command','command':'recover'}))
        self.assertEqual(self.control.commands.get_nowait(),('recover',None))

    def test_manual_modes_and_six_car_grid(self):
        s=RaceSelection(speed=5.)
        s.set_mode('human')
        self.assertEqual(s.speed,1.)
        self.assertEqual(s.planned_indices(),(5,0,1,2,3,4))
        grid=build_grid({},s.planned_indices(),3)
        self.assertEqual(len(grid),6)
        self.assertEqual(len({(tuple(c['spawn_lane_index']),c['spawn_longitude']) for c in grid.values()}),6)
        s.comparison_indices=()
        with self.assertRaises(ValueError): s.planned_indices()
        s.set_mode('practice')
        self.assertEqual(s.planned_indices(),(5,))
        self.assertEqual(s.chosen_indices,())
        s.player_model='vf3'
        self.assertEqual(s.model_for(5),'vf3')

    def test_human_result_is_distinct_from_ai_results(self):
        race=RaceComparison((5,0),{'road':1},1)
        race.record({'agent_id':'human_phone','case':{'road':1}})
        self.assertEqual(race.to_dict()['pending_agent_ids'],['stage2_astar'])
        race.record({'agent_id':'stage2_astar','case':{'road':1}})
        self.assertEqual(race.status,'COMPLETE')

    def test_three_models_have_distinct_real_geometry(self):
        from panda3d.core import NodePath
        shapes=[]
        for model in CAR_MODELS:
            parent=NodePath('test')
            mesh=build_car(parent,model,(.2,.8,1.,1.))
            lo,hi=mesh.getTightBounds()
            size=hi-lo
            self.assertTrue(all(d>0 for d in size))
            shapes.append(tuple(round(d,3) for d in size))
            triangles=sum(path.node().getGeom(i).getPrimitive(j).decompose().getNumPrimitives()
                          for path in mesh.findAllMatches('**/+GeomNode')
                          for i in range(path.node().getNumGeoms())
                          for j in range(path.node().getGeom(i).getNumPrimitives()))
            # Batching/LOD may merge meshes. Require actual car geometry,
            # rather than a minimum number of draw calls (which penalizes it).
            self.assertGreater(triangles,500)
            parent.removeNode()
        self.assertEqual(len(set(shapes)),3)

    def test_player_and_ai_stop_after_crash_except_in_unranked_practice(self):
        with patch('metadrive.envs.metadrive_env.MetaDriveEnv.done_function',return_value=(True,{'crash_vehicle':True})):
            env=RaceEnvironment.__new__(RaceEnvironment)
            env.agent_manager=SimpleNamespace(active_agents={
                'racer_0':SimpleNamespace(),'racer_5':SimpleNamespace()})
            self.assertEqual(env.done_function('racer_5')[0],True)
            self.assertEqual(env.done_function('racer_0')[0],True)
            env.race_practice=True
            self.assertEqual(env.done_function('racer_5')[0],False)
            self.assertEqual(env.done_function('racer_0')[0],True)


class HumanPhysicsTests(unittest.TestCase):
    def test_phone_preview_is_fresh_in_garage_and_stops_on_disconnect(self):
        from unittest.mock import MagicMock
        app=fixtures.TestRaceMode().running_app();app.state='garage';app._init_phone()
        app.phone_host=MagicMock();app.phone_camera=MagicMock(vehicle=None)
        app.phone_control.claim('phone')
        app.phone_control.transport('phone','connected',['drive','commands'])
        app._phone_tick()
        app.phone_camera.track.assert_called_once_with(app.env.agent)
        app.phone_camera.set_active.assert_called_with(True)
        self.assertEqual(app.ui.phone_status.__setitem__.call_args.args[1],'Đã kết nối tay lái · có thể xuất phát')
        app.phone_control.release('phone');app.last_phone_tick=0
        app._phone_tick()
        app.phone_camera.set_active.assert_called_with(False)

    def test_explicit_phone_join_returns_to_garage_before_changing_controller(self):
        from unittest.mock import MagicMock
        app=fixtures.TestRaceMode().running_app();app._garage=MagicMock()
        app._phone_command('join_driver','practice')
        app._garage.assert_called_once()
        self.assertEqual(app.selection.mode,'practice')
        app.ui.update_selection.assert_called_once()

    @patch('src.platform.build_agent_input',return_value=SimpleNamespace(step_index=0))
    def test_player_collision_ends_only_player_run_using_same_rules_as_ai(self,_):
        app=fixtures.TestRaceMode().running_app()
        player=app.racers.pop('racer_1')
        player.index=5
        app.racers['racer_5']=player
        app.env.agents['racer_5']=app.env.agents.pop('racer_1')
        app.obs['racer_5']=app.obs.pop('racer_1')
        app.comparison=RaceComparison((0,5),app.case.to_dict(),1)
        keys=tuple(app.racers)
        app.env.step.return_value=(dict(app.obs),{},dict.fromkeys(keys,False),dict.fromkeys(keys,False),
            {'racer_0':{'route_completion':.1},'racer_5':{'route_completion':.1,'crash_vehicle':True}})
        app._step()
        self.assertEqual(set(app.env.step.call_args.args[0]),{'racer_0','racer_5'})
        self.assertEqual(player.summary['reason'],'CRASH_VEHICLE')
        self.assertFalse(player.summary['clean_success'])
        self.assertTrue(player.flags['crash_vehicle'])
        self.assertIsNone(app.racers['racer_0'].summary)


class WebRTCTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_offer_video_control_auth_and_exclusive_ownership(self):
        logging.getLogger('aioice').setLevel(logging.WARNING)
        logging.getLogger('asyncio').setLevel(logging.ERROR)
        from aiohttp.test_utils import TestClient, TestServer
        from aiortc import RTCPeerConnection,RTCConfiguration,RTCSessionDescription
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            control=PhoneControl()
            frames=FrameMailbox()
            frames.put(np.full((180,320,3),[30,100,180],dtype=np.uint8))
            host=MobileHost(control,frames,directory)
            # Default is portable; opt in to the real RTX path for hardware QA.
            host.encoder_preference=os.environ.get('PHONE_TEST_ENCODER','cpu')
            client=TestClient(TestServer(host.make_application()))
            pc=RTCPeerConnection(RTCConfiguration(iceServers=[]))
            await client.start_server()
            received=asyncio.Future()
            @pc.on('track')
            def on_track(track):
                async def read():
                    try:
                        frame=await track.recv()
                        if not received.done(): received.set_result(frame)
                    except Exception as e:
                        if not received.done(): received.set_exception(e)
                asyncio.create_task(read())
            try:
                self.assertEqual((await client.get('/api/state')).status,403)
                auth={'X-Race-Token':host.token}
                self.assertEqual((await client.get('/api/state',headers=auth)).status,200)
                drive=pc.createDataChannel('drive',ordered=False,maxRetransmits=0)
                commands=pc.createDataChannel('commands')
                messages=[]
                commands.on('message',lambda value: messages.append(json.loads(value)))
                pc.addTransceiver('video',direction='recvonly')
                await pc.setLocalDescription(await pc.createOffer())
                response=await client.post('/offer',headers=auth,json={'type':'offer','sdp':pc.localDescription.sdp})
                self.assertEqual(response.status,200,await response.text())
                answer=await response.json()
                self.assertIn('H264/90000',answer['sdp'])
                self.assertNotIn('VP8/90000',answer['sdp'])
                await pc.setRemoteDescription(RTCSessionDescription(**answer))
                frame=await asyncio.wait_for(received,10)
                self.assertEqual((frame.width,frame.height),(320,180))
                self.assertGreater(frame.to_ndarray(format='rgb24').mean(),60)
                expected={'nvenc':'h264_nvenc','cpu':'libx264','qsv':'h264_qsv'}
                if host.encoder_preference in expected:
                    self.assertEqual(host.stream_stats['encoder'],expected[host.encoder_preference])
                for _ in range(100):
                    if drive.readyState=='open': break
                    await asyncio.sleep(.02)
                drive.send(json.dumps({'type':'input','seq':1,'steering':.4,'throttle':1.}))
                await asyncio.sleep(.1)
                self.assertEqual(control.sample(),(.4,1.))
                state=await (await client.get('/api/state',headers=auth)).json()
                self.assertTrue(state['connected'])
                self.assertEqual(state['connection_state'],'ready')
                self.assertEqual(state['open_channels'],['commands','drive'])
                # Real simulator route comparisons return numpy.int64. The
                # old telemetry loop silently closed the peer at race start.
                control.publish({'state':'running','position':np.int64(2),
                                 'performance':{'physics_hz':np.float32(50.)}})
                for _ in range(50):
                    if any(row.get('position')==2 for row in messages):break
                    await asyncio.sleep(.02)
                self.assertTrue(any(row.get('position')==2 for row in messages))
                self.assertTrue(control.snapshot()['connected'])
                state=await (await client.get('/api/state',headers=auth)).json()
                self.assertEqual(state['position'],2)
                self.assertEqual(state['performance']['physics_hz'],50.)
                self.assertEqual((await client.post('/offer',headers=auth,json={})).status,409)
                await pc.close()
                await asyncio.sleep(.1)
                self.assertEqual(control.sample(),(0.,-1.))
            finally:
                await pc.close()
                for peer in list(host.peers): await peer.close()
                await client.close()


if __name__=='__main__':
    unittest.main()
