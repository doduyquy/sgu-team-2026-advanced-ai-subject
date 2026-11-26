"""Bounded, persistent AI workers for native and phone interactive races."""

import importlib
import multiprocessing as mp
import os
from queue import Empty, Full, SimpleQueue
import threading
import time


def _policy_worker(factory_ref, context, seed, requests, responses, planner_options=None):
    # Prevent five independent NumPy workers each starting a BLAS thread pool.
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    policy = None
    try:
        module, name = factory_ref.split(':', 1)
        factory = importlib.import_module(module)
        for component in name.split('.'):
            factory = getattr(factory, component)
        policy = factory()
        if planner_options:
            from src.visualization.race_rules import apply_race_planner_options
            apply_race_planner_options(policy,planner_options)
        policy.reset(context, agent_seed=seed)
        responses.put(('ready',planner_options))
        while True:
            job = requests.get()
            if job is None:
                break
            tick, captured_at, inputs = job
            # NumPy pickle restores writable arrays; retain the public input's
            # immutable contract across the process boundary.
            for array in (inputs.core_observation.features, inputs.traffic_context.actors_array,
                          inputs.traffic_context.validity_mask, inputs.task_context.waypoints_array):
                array.setflags(write=False)
            started = time.perf_counter()
            try:
                from src.visualization.race_rules import prepare_race_policy
                prepare_race_policy(policy,inputs)
                decision = policy.act(inputs)
                result = ('decision', tick, captured_at, decision,
                          (time.perf_counter() - started) * 1000, None)
            except Exception as error:
                result = ('decision', tick, captured_at, None,
                          (time.perf_counter() - started) * 1000, str(error))
            responses.put(result)
    except Exception as error:
        responses.put(('error', str(error)))
    finally:
        if policy is not None:
            policy.close()


class AsyncPolicy:
    """Only one in-flight request; newest pending snapshot stays in the parent.

    Physics never waits for a process or a pipe write. Epochs are represented
    by new worker instances on reset, so previous-episode results cannot leak.
    """
    def __init__(self, factory_ref, context, seed=42, worker_target=_policy_worker, planner_options=None):
        ctx = mp.get_context('spawn')
        self.requests, self.responses = ctx.Queue(maxsize=1), ctx.Queue(maxsize=2)
        arguments=(factory_ref, context, seed, self.requests, self.responses)
        if planner_options:
            arguments=(*arguments,dict(planner_options))
        self.process = ctx.Process(target=worker_target,args=arguments,daemon=True)
        self.applied_planner_options=None
        self.ready = self.busy = self.closed = False
        self.pending = self.latest = self.error = None
        self.last_inputs = None
        self.submitted = {}
        self.started_at = time.perf_counter()
        # Pinned MetaDrive sets PYTHONUTF8="on" when loading assets. Python
        # rejects that spelling before a spawned interpreter can run any code.
        previous_utf8=os.environ.get('PYTHONUTF8')
        os.environ['PYTHONUTF8']='1'
        try:
            self.process.start()
        finally:
            if previous_utf8 is None:
                os.environ.pop('PYTHONUTF8',None)
            else:
                os.environ['PYTHONUTF8']=previous_utf8
        self.inbox=SimpleQueue()
        self.reader=threading.Thread(target=self._read_responses,daemon=True,name='ai-result-reader')
        self.reader.start()

    def _read_responses(self):
        while not self.closed:
            try:
                self.inbox.put(self.responses.get(timeout=.1))
            except Empty:
                pass
            except (ValueError,OSError,EOFError):
                return

    def submit(self, tick, inputs, captured_at=None):
        self.pending = (tick, captured_at or time.perf_counter(), inputs)
        self._dispatch()

    def _dispatch(self):
        if self.ready and not self.busy and self.pending is not None and not self.closed:
            try:
                self.requests.put_nowait(self.pending)
            except Full:
                return
            self.submitted = {self.pending[0]: self.pending[2]}
            self.pending = None
            self.busy = True

    def poll(self):
        result = None
        while not self.closed:
            try:
                message = self.inbox.get_nowait()
            except Empty:
                break
            if message[0] == 'ready':
                self.ready = True
                self.applied_planner_options=message[1] if len(message)>1 else None
            elif message[0] == 'error':
                self.error = message[1]
            else:
                _, tick, captured_at, decision, latency, error = message
                self.busy = False
                self.error = error
                self.last_inputs = self.submitted.get(tick)
                result = self.latest = (tick, captured_at, decision, latency)
        if not self.closed and not self.process.is_alive() and self.error is None:
            self.error = 'AI worker exited unexpectedly'
        self._dispatch()
        return result

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.requests.put_nowait(None)
        except Full:
            pass
        self.process.join(timeout=.15)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=.5)
        self.reader.join(timeout=.2)
        for queue in (self.requests, self.responses):
            queue.cancel_join_thread()
            queue.close()
