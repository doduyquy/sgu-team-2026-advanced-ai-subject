"""Thread-safe phone input. Networking never calls Panda3D or steps physics."""

import math
from queue import SimpleQueue
import threading
import time


class PhoneControl:
    def __init__(self, timeout=.45, clock=time.perf_counter):
        self.timeout, self.clock = timeout, clock
        self.lock = threading.Lock()
        self.commands = SimpleQueue()
        self.owner = None
        self.transport_state = 'closed'
        self.open_channels = set()
        self.sequence = -1
        self.received = float('-inf')
        self.values = (0., -1.)
        self.status = {}
        self.applied_ms = None
        self.sampled_received = float('-inf')

    def claim(self, owner):
        with self.lock:
            if self.owner is not None and self.owner != owner:
                return False
            self.owner, self.sequence = owner, -1
            self.transport_state, self.open_channels = 'connecting', set()
            self.received, self.values = float('-inf'), (0., -1.)
            return True

    def release(self, owner):
        with self.lock:
            if self.owner == owner:
                self.owner = None
                self.transport_state, self.open_channels = 'closed', set()
                self.received, self.values = float('-inf'), (0., -1.)

    def clear(self):
        with self.lock:
            self.received, self.values = float('-inf'), (0., -1.)

    def transport(self, owner, state, channels):
        """A reserved offer is not a connected driver; both channels must open."""
        with self.lock:
            if self.owner == owner:
                self.transport_state, self.open_channels = state, set(channels)

    def receive(self, owner, message):
        if not isinstance(message, dict):
            return False
        if message.get('type') == 'input':
            seq, steering, throttle = (message.get(k) for k in ('seq', 'steering', 'throttle'))
            if type(seq) is not int or seq < 0:
                return False
            if any(type(v) not in (float, int) or not math.isfinite(v) or abs(v) > 1
                   for v in (steering, throttle)):
                return False
            with self.lock:
                if self.owner != owner or seq <= self.sequence:
                    return False
                self.sequence, self.received = seq, self.clock()
                self.values = (float(steering), float(throttle))
            return True
        command = message.get('command')
        if message.get('type') == 'command' and command in (
                'start', 'pause', 'recover', 'phone_camera', 'car_model', 'mode', 'quality', 'join_driver'):
            with self.lock:
                if self.owner != owner:
                    return False
            self.commands.put((command, message.get('value')))
            return True
        return False

    def sample(self):
        with self.lock:
            fresh = self.owner is not None and self.clock() - self.received <= self.timeout
            self.sampled_received=self.received if fresh else float('-inf')
            return self.values if fresh else (0., -1.)

    def publish(self, status):
        with self.lock:
            self.status = dict(status)

    def snapshot(self):
        with self.lock:
            ready = (self.owner is not None and self.transport_state == 'connected'
                     and {'drive','commands'}.issubset(self.open_channels))
            return dict(self.status, connected=ready, pairing=self.owner is not None and not ready,
                        connection_state='ready' if ready else ('connecting' if self.owner is not None else 'disconnected'),
                        transport_state=self.transport_state, open_channels=sorted(self.open_channels),
                        input_fresh=self.clock() - self.received <= self.timeout,
                        input_age_ms=round((self.clock()-self.received)*1000,1) if math.isfinite(self.received) else None,
                        input_apply_ms=self.applied_ms)

    def mark_applied(self):
        with self.lock:
            self.applied_ms=round((self.clock()-self.sampled_received)*1000,1) if math.isfinite(self.sampled_received) else None


class HumanPolicy:
    def __init__(self, control):
        self.control = control

    def reset(self, *_args, **_kwargs):
        self.control.clear()

    def act(self, _inputs):
        from src.platform.agent import AgentDecision
        steering, throttle = self.control.sample()
        # MetaDrive's physical positive steering turns left. The touch wheel
        # uses the driver's convention: positive means turning right.
        return AgentDecision([-steering, throttle], {'controller': 'phone',
                             'input_fresh': self.control.snapshot()['input_fresh']})

    def close(self):
        self.control.clear()
