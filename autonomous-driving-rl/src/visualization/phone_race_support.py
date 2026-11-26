"""Phone lifecycle and commands, kept separate from the race UI redesign."""

from queue import Empty
from src.visualization.mobile_control import PhoneControl
from src.visualization.mobile_server import MobileHost, FrameMailbox
from src.visualization.race_session import HUMAN_INDEX
from src.visualization.race_vehicle_models import CAR_MODELS, CAR_NAMES
from src.visualization.race_rules import RULESET_ID, race_positions, RACE_SPEED_LIMIT_KMH
from src.visualization.race_session import OUTCOME_LABELS

PHONE_STREAM_PROFILES={
    'auto':(960,540,3500000),'smooth':(640,360,1500000),
    'balanced':(960,540,3500000),'hd':(1280,720,6000000),
}


class PhoneRaceSupport:
    def _init_phone(self):
        self.phone_control = PhoneControl()
        self.phone_frames = FrameMailbox()
        self.phone_host = self.phone_camera = None
        self.phone_port = 8765
        self.phone_address = None
        self.recoveries = 0
        self.realtime = None
        self.realtime_enabled = True
        self.control_dt_s = .1
        self.execution_mode = 'shared_world_lockstep'
        self.phone_quality = 'auto'
        self.phone_encoder = 'nvenc'
        self.last_phone_tick = 0.

    def _open_phone(self, show=True):
        if self.phone_host is None:
            host = MobileHost(self.phone_control,self.phone_frames,self.output,
                              self.phone_port,self.phone_address)
            host.encoder_preference=self.phone_encoder
            width,height,host.bitrate=PHONE_STREAM_PROFILES[self.phone_quality]
            self.phone_host = host.start()
            from src.visualization.phone_camera import PhoneCamera
            self.phone_camera = PhoneCamera(self.env.engine,self.phone_frames,width,height)
            self.manifest['phone'] = {'port':self.phone_host.port,'address':self.phone_host.address,
                                      'video_size':[width,height],'input_timeout_s':self.phone_control.timeout,
                                      'encoder_preference':self.phone_encoder,
                                      'encoder_probe':self.phone_host.stream_stats}
            self.save_manifest()
        if show:
            self.ui.show_phone(self.phone_host)

    def _phone_ready(self):
        self._open_phone()
        if not self.phone_control.snapshot()['connected']:
            self.ui.status['text']='Quét QR và kết nối điện thoại trước khi xuất phát.'
            return False
        self.ui.phone_panel.hide()
        return True

    def _phone_before_reset(self):
        if self.phone_camera:
            self.phone_camera.detach()
        self.phone_control.clear()
        self.recoveries = 0

    def _phone_after_reset(self):
        if self.phone_camera and 'racer_5' in self.env.agents:
            self.phone_camera.track(self.env.agents['racer_5'])

    def _phone_command(self, command, arg):
        if command == 'join_driver':
            # Explicit phone button ends the current preview before switching;
            # merely connecting must not silently abort an existing AI race.
            if self.state != 'garage':
                self._garage()
            self.selection.set_mode(arg if arg in ('human','practice') else 'human')
            self.ui.phone_panel.hide()
            self.ui.update_selection(self.selection,self.roads)
            self.save_manifest()
        elif command == 'phone':
            self._open_phone()
        elif command == 'phone_close':
            self.ui.phone_panel.hide()
        elif command == 'phone_camera':
            if self.phone_camera and 'racer_5' in self.env.agents:
                views=self.phone_camera.VIEWS
                self.phone_camera.track(self.env.agents['racer_5'],views[(views.index(self.phone_camera.view)+1)%len(views)])
        elif command == 'recover':
            if self.selection.mode == 'practice' and self.state in ('running','paused') and 'racer_5' in self.racers:
                if self.racers['racer_5'].summary is None:
                    self.phone_control.clear()
                    if self.env.recover('racer_5'):
                        self.recoveries += 1
        elif command == 'car_model':
            if self.state == 'garage' and arg in CAR_MODELS:
                if self.selection.manual:
                    self.selection.player_model=arg
                else:
                    self.selection.vehicle_models[self.ui._preview_index]=arg
                self.ui.update_selection(self.selection,self.roads)
                self.save_manifest()
        elif command == 'quality':
            if arg in ('auto','smooth','balanced','hd'):
                self.phone_quality=arg
                if self.phone_camera:
                    from src.visualization.phone_camera import PhoneCamera
                    width,height,bitrate=PHONE_STREAM_PROFILES[arg]
                    old=self.phone_camera
                    vehicle,view=old.vehicle,old.view
                    if (old.width,old.height)!=(width,height):
                        old.destroy()
                        self.phone_camera=PhoneCamera(self.env.engine,self.phone_frames,width,height)
                        if vehicle:
                            self.phone_camera.track(vehicle,view)
                    if self.phone_host:
                        self.phone_host.bitrate=bitrate
        else:
            return False
        return True

    def _phone_tick(self):
        if not self.phone_host:
            return
        for _ in range(32):
            try:
                command,arg=self.phone_control.commands.get_nowait()
            except Empty:
                break
            # A phone cannot change the desktop into an AI-only mode.
            if command=='mode' and arg not in ('human','practice'):
                continue
            if command=='start' and self.state in ('results','comparison'):
                command='restart'
            self.enqueue(command,arg)
        import time
        now=time.perf_counter()
        # UI and telemetry do not need to rebuild text 50 times a second.
        if now-self.last_phone_tick<.1:
            return
        self.last_phone_tick=now
        player=self.racers.get('racer_5') if self.state!='garage' else None
        vehicle=self.env.agents.get('racer_5') if player else None
        position=None
        if player:
            if self.selection.mode != 'practice':
                position=race_positions(self.racers.values())[player.vehicle_id]
        self.phone_control.publish({'state':self.state,'mode':self.selection.mode,
            'model':self.selection.player_model,'model_name':CAR_NAMES[self.selection.player_model],
            'speed_kmh':float(vehicle.speed_km_h) if vehicle else 0.,
            'speed_limit_kmh':getattr(self,'race_speed_limit_kmh',RACE_SPEED_LIMIT_KMH),
            'progress':player.route if player else None,'position':position,
            'total':(1+len(self.selection.chosen_indices) if self.state=='garage' and self.selection.manual else len(self.racers)),
            'camera':self.phone_camera.view if self.phone_camera else 'chase',
            'finished':bool(player and player.summary),'recoveries':self.recoveries,
            'ruleset':RULESET_ID,'ranked':self.selection.mode != 'practice',
            'recover_allowed':self.selection.mode == 'practice',
            'result_reason':player.summary['reason'] if player and player.summary else None,
            'result_label':OUTCOME_LABELS.get(player.summary['reason'],player.summary['reason']) if player and player.summary else None,
            'clean_success':bool(player and player.summary and player.summary.get('clean_success')),
            'quality':self.phone_quality,'performance':self.realtime.stats() if self.realtime else {}})
        if self.phone_camera:
            target=vehicle or self.env.agent
            if self.phone_camera.vehicle is not target:
                self.phone_camera.track(target)
            self.phone_camera.set_active(self.phone_control.snapshot()['connected'])
        connection=self.phone_control.snapshot()
        self.ui.phone_status['text']=(
            'Đã kết nối tay lái · có thể xuất phát' if connection['connected'] else
            'Đang mở kênh tay lái…' if connection['pairing'] else 'Đang chờ điện thoại kết nối…')

    def _close_phone(self):
        if self.phone_camera:
            self.phone_camera.destroy()
            self.phone_camera=None
        if self.phone_host:
            self.phone_host.stop()
            self.phone_host=None
