"""Interactive 50 Hz physics, independent 10 Hz AI decisions and bounded age.

This is a demo execution mode. Diagnostic lockstep remains in RaceApp.
Workers consume immutable AgentInputV1 snapshots, never simulator objects.
"""

from collections import deque
import json
import os
import time

from src.platform import AgentPublicEpisodeContext, InputProfileId, build_agent_input, CanonicalActionV1
from src.visualization.race_rules import RACE_FLAGS, classify_race_outcome, limit_race_action
from src.visualization.async_policy import AsyncPolicy
from src.visualization.race_environment import racer_route_network
from src.visualization.race_session import HUMAN_INDEX


class RealtimeRace:
    dt = .02
    max_action_age_s = .35

    def __init__(self, app):
        self.app = app
        self.timer_period_started=False
        if os.name=='nt':
            import ctypes
            self.timer_period_started=ctypes.windll.winmm.timeBeginPeriod(1)==0
        self.actions = {}
        self.times = deque(maxlen=250)
        self.applied = deque(maxlen=250)
        self.applied_ticks = deque(maxlen=250)
        self.clock_wall=time.perf_counter()
        self.clock_sim=0.
        self.last_submit = -5
        self.next_submit = {}
        self.last_hud = 0.
        self.stale_count = 0
        self.warmup = time.perf_counter()
        self.next_render=0.
        self.render_times=deque(maxlen=250)
        pipe=getattr(app.env.engine,'pbrpipe',None)
        self.render_buffers=[app.env.engine.win]+list(pipe.manager.buffers if pipe else [])
        # Keep the laptop HUD at native resolution; reduce only its costly
        # HDR 3D buffer while a second camera is streaming to the phone.
        self.original_buffer_sizes=[]
        # The shared graphics profile already scales the 3D buffers. Preserve
        # the old fallback only for callers without the profile runtime.
        if pipe and not getattr(app, 'graphics_runtime', None):
            for buffer in pipe.manager.buffers:
                size=(buffer.getXSize(),buffer.getYSize())
                if hasattr(buffer,'setSize') and size[0]>800:
                    try:
                        buffer.setSize(int(size[0]*.75),int(size[1]*.75))
                        self.original_buffer_sizes.append((buffer,size))
                    except Exception:
                        pass
        self.log = (app.output/'applied_actions.jsonl').open('a',encoding='utf-8',buffering=8192)
        for racer in app.racers.values():
            if racer.index == HUMAN_INDEX or racer.summary:
                continue
            racer.policy.close()
            racer.policy = AsyncPolicy(app.registry.get_factory_ref(racer.agent_id),
                AgentPublicEpisodeContext(horizon_steps=app.case.horizon_steps,
                    input_profile_id=racer.registration.input_profile_id,
                    action_adapter_id=racer.registration.action_adapter_id),
                planner_options=racer.planner_options)
        self.request_decisions()

    def request_decisions(self, limit=None):
        app = self.app
        due=[(key,r) for key,r in app.racers.items() if r.index!=HUMAN_INDEX and not r.summary
             and app.steps>=self.next_submit.get(key,0)]
        if limit:
            due=sorted(due,key=lambda row:self.next_submit.get(row[0],0))[:limit]
        if not due:
            return
        actors = list(getattr(app.env.engine.traffic_manager,'traffic_vehicles',[])) + list(app.env.agents.values())
        now = time.perf_counter()
        for offset,(key,racer) in enumerate(due):
            vehicle = app.env.agents[key]
            raw=app.env.refresh_observation(key)
            inputs = build_agent_input(raw_obs=raw,ego_vehicle=vehicle,traffic_vehicles=actors,
                navigation=vehicle.navigation,road_network=racer_route_network(app.env.current_map.road_network,vehicle),
                step_index=app.steps//5,profile_id=InputProfileId.STATE_DECISION_V1,
                **({'task_lookahead_count':40,'task_lookahead_spacing_m':3.,'traffic_radius_m':120.}
                   if getattr(app,'race_speed_limit_kmh',20.)>20. else {}))
            racer.policy.submit(app.steps,inputs,now)
            self.next_submit[key]=app.steps+5+(offset if limit is None else 0)
        self.last_submit = app.steps

    def _poll(self):
        app = self.app
        for key,racer in app.racers.items():
            if racer.index==HUMAN_INDEX or racer.summary:
                continue
            worker = racer.policy
            result = worker.poll()
            if worker.error:
                app._finish_racer(racer,'TECHNICAL_ERROR',error=worker.error)
                continue
            if not worker.ready and time.perf_counter()-self.warmup>12:
                app._finish_racer(racer,'TECHNICAL_ERROR',error='AI startup timeout')
                continue
            if result:
                tick,captured,decision,latency = result
                try:
                    requested_action = racer.adapter.to_canonical(decision.action_payload)
                    from src.visualization.vietnam_speed import decision_action
                    vehicle=app.env.agents[key]
                    requested_action=decision_action(decision,requested_action,vehicle,
                        getattr(app,'race_speed_limit_kmh',20.)>20.)
                    action = limit_race_action(requested_action, app.selection.mode == 'practice',
                        vehicle.speed_km_h,getattr(app,'race_speed_limit_kmh',20.))
                except Exception as error:
                    app._finish_racer(racer,'TECHNICAL_ERROR',error=str(error))
                    continue
                self.actions[key] = (action,captured,tick)
                racer.latencies.append(latency)
                racer.last_control = (worker.last_inputs,decision,action,latency)
                app.decisions.write(json.dumps({'episode':app.episode,'race_tick':app.steps,
                    'physics_tick':app.steps,'simulation_time_s':app.steps*self.dt,
                    'execution':'async_ai_shared_world','vehicle_id':key,'agent_id':racer.agent_id,
                    'input_step':tick//5,'action':action.to_dict(),'requested_action':requested_action.to_dict(),'act_latency_ms':latency,
                    'decision_age_ms':round((time.perf_counter()-captured)*1000,2)},ensure_ascii=False)+'\n')

    def step(self):
        app = self.app
        started = time.perf_counter()
        self._poll()
        now = time.perf_counter()
        # Keep real-time motion even when a GPU render takes longer than one
        # 20 ms physical step. Bound catch-up to three fixed substeps, never
        # enlarge physics dt or wait for an AI result.
        speed=app.selection.speed
        target=getattr(self,'clock_sim',app.steps*self.dt)+(now-getattr(self,'clock_wall',now))*speed
        substeps=max(1,min(3,int((target-app.steps*self.dt)/self.dt)+1))
        self.request_decisions(limit=substeps)
        actions, ages = {}, {}
        active = [r for r in app.racers.values() if not r.summary]
        if not active:
            app._end_race()
            return
        for key,racer in app.racers.items():
            if racer.summary:
                actions[key]=[0.,0.]
            elif racer.index==HUMAN_INDEX:
                decision=racer.policy.act(None)
                from src.visualization.vietnam_speed import vehicle_speed_limit
                vehicle=app.env.agents[key]
                action=limit_race_action(racer.adapter.to_canonical(decision.action_payload), app.selection.mode == 'practice',vehicle.speed_km_h,
                                         vehicle_speed_limit(app.env,vehicle))
                actions[key]=action.to_numpy()
                app.phone_control.mark_applied()
            else:
                item=self.actions.get(key)
                age=now-item[1] if item else float('inf')
                ages[key]=round(age*1000,1) if item else None
                if item and age<=self.max_action_age_s:
                    from src.visualization.vietnam_speed import vehicle_speed_limit
                    vehicle=app.env.agents[key]
                    actions[key]=limit_race_action(item[0], app.selection.mode == 'practice',vehicle.speed_km_h,
                                                  vehicle_speed_limit(app.env,vehicle)).to_numpy()
                else:
                    actions[key]=[0.,-.8]
                    self.stale_count+=1
        render_due=now>=self.next_render
        self.set_render_enabled(render_due)
        if render_due:
            # A desktop AI race can display every 20 ms physics update. Keep
            # the phone's dual-camera GPU budget at its measured 30 FPS.
            streaming=getattr(app,'phone_camera',None) is not None and app.phone_camera.enabled
            render_hz=30 if streaming else 50
            self.next_render=max(self.next_render+1/render_hz,now)
        app.env.config['decision_repeat']=substeps
        app.obs,_,terminated,truncated,app.info = app.env.step(actions)
        if render_due:
            self.render_times.append(time.perf_counter())
        app.steps+=substeps
        for racer in active:
            key=racer.vehicle_id
            racer.steps+=substeps
            info=app.info[key]
            for flag in RACE_FLAGS:
                if info.get(flag): racer.flags[flag]=True
            outcome=classify_race_outcome(racer.flags,terminated=terminated[key],
                practice=app.selection.mode == 'practice',
                truncated=truncated[key] or app.steps>=app.case.horizon_steps*5)
            racer.route=info.get('route_completion')
            if outcome.is_done:
                app._finish_racer(racer,outcome.reason,outcome=outcome,
                                   termination_source='interactive_simulator')
        elapsed=(time.perf_counter()-started)*1000
        self.times.append(elapsed)
        self.applied.append(time.perf_counter())
        if not hasattr(self,'applied_ticks'): self.applied_ticks=deque(maxlen=250)
        self.applied_ticks.append(app.steps)
        self.log.write(json.dumps({'episode':app.episode,'physics_tick':app.steps,
            'simulation_time_s':round(app.steps*self.dt,3),'actions':{k:list(map(float,v)) for k,v in actions.items()},
            'ai_age_ms':ages,'physical_substeps':substeps,'step_ms':round(elapsed,2)})+'\n')
        if now-self.last_hud>=.1:
            app._update_hud(); app._update_standings()
            self.last_hud=now
        # Keep phase at 50 Hz without executing a burst of late physics steps.
        period=self.dt/speed
        app._next_decision=max(app._next_decision+period,time.perf_counter()-period)
        if all(r.summary for r in app.racers.values()):
            app._end_race()

    def stats(self):
        times=sorted(self.times)
        duration=self.applied[-1]-self.applied[0] if len(self.applied)>1 else 0
        hz=(self.applied_ticks[-1]-self.applied_ticks[0])/duration if duration else 0
        renders=getattr(self,'render_times',())
        render_duration=renders[-1]-renders[0] if len(renders)>1 else 0
        return {'physics_hz':round(hz,1),'physics_step_ms':round(sum(times)/len(times),2) if times else 0,
                'render_fps':round((len(renders)-1)/render_duration,1) if render_duration else 0,
                'physics_p95_ms':round(times[min(len(times)-1,int(len(times)*.95))],2) if times else 0,
                'control_hz':round((len(self.applied)-1)/duration,1) if duration else 0,
                'ai_workers':sum(isinstance(r.policy,AsyncPolicy) and r.policy.ready for r in self.app.racers.values()),
                'ai_stale_brakes':self.stale_count}

    def set_render_enabled(self,enabled):
        for buffer in getattr(self,'render_buffers',[]):
            buffer.setActive(enabled)
        if getattr(self.app,'phone_camera',None):
            self.app.phone_camera.render_gate=enabled

    def workers_ready(self):
        return all(r.summary or r.index==HUMAN_INDEX or r.policy.ready
                   for r in self.app.racers.values())

    def reset_clock(self):
        self.clock_wall=time.perf_counter()
        self.clock_sim=self.app.steps*self.dt
        self.app._next_decision=self.clock_wall
        # Paused/countdown snapshots are old in wall time. Replace them with
        # the current state instead of briefly applying their old controls.
        self.next_submit.clear()
        self.request_decisions()

    def close(self):
        self.set_render_enabled(True)
        for buffer,size in self.original_buffer_sizes:
            buffer.setSize(*size)
        if self.timer_period_started:
            import ctypes
            ctypes.windll.winmm.timeEndPeriod(1)
            self.timer_period_started=False
        self.log.close()
