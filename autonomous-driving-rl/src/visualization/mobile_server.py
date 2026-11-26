"""Local-only WebRTC host. One phone owns the controls; one bounded video frame."""

import asyncio
import json
from pathlib import Path
import secrets
import socket
import threading
import time
from collections import deque


class FrameMailbox:
    def __init__(self):
        self.lock = threading.Lock()
        self.changed = threading.Condition(self.lock)
        self.frame = None
        self.produced = 0
        self.captured_at = 0.
        self.timestamps = deque(maxlen=90)

    def put(self, rgb):
        with self.lock:
            self.frame = rgb
            self.produced += 1
            self.captured_at = time.perf_counter()
            self.timestamps.append(self.captured_at)
            self.changed.notify_all()

    def get(self):
        with self.lock:
            return self.frame

    def latest(self):
        with self.lock:
            return self.produced,self.frame,self.captured_at

    def wait_latest(self, sequence, timeout=.25):
        with self.changed:
            self.changed.wait_for(lambda:self.produced!=sequence,timeout)
            return self.produced,self.frame,self.captured_at

    def stats(self):
        with self.lock:
            fps=(len(self.timestamps)-1)/(self.timestamps[-1]-self.timestamps[0]) if len(self.timestamps)>1 else 0
            return {'capture_fps':round(fps,1),'frame_age_ms':round((time.perf_counter()-self.captured_at)*1000,1) if self.captured_at else None,
                    'resolution':[self.frame.shape[1],self.frame.shape[0]] if self.frame is not None else None}


def local_address():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            # Route lookup only: connect() on UDP sends no packet.
            sock.connect(('192.0.2.1', 9))
            return sock.getsockname()[0]
    except OSError:
        return '127.0.0.1'


class MobileHost:
    def __init__(self, control, frames, output, port=8765, advertise_host=None):
        self.control, self.frames = control, frames
        self.output, self.port = Path(output), port
        self.address = advertise_host or local_address()
        self.token = secrets.token_urlsafe(24)
        self.ready = threading.Event()
        self.loop = self.thread = self.error = None
        self.peers = set()
        self.encoder_preference = 'nvenc'
        self.stream_stats = {}
        self.bitrate = 2500000

    @property
    def url(self):
        return f'http://{self.address}:{self.port}/#token={self.token}'

    def start(self):
        # Fail before displaying a QR if the requested hardware cannot encode.
        # This also makes the reported backend reflect a real encoded frame.
        import numpy as np
        from src.visualization.phone_encoder import PhoneEncoder
        probe = PhoneEncoder(self.encoder_preference, self.bitrate)
        try:
            probe.encode(np.zeros((180, 320, 3), dtype=np.uint8), 0)
            self.stream_stats = probe.stats()
        finally:
            probe.close()
        self.thread = threading.Thread(target=self._thread_main, name='phone-host', daemon=True)
        self.thread.start()
        if not self.ready.wait(8):
            raise RuntimeError('Phone host did not start in time')
        if self.error:
            raise RuntimeError(f'Không mở được cổng điện thoại: {self.error}')
        import qrcode
        self.output.mkdir(parents=True, exist_ok=True)
        qrcode.make(self.url).save(self.output / 'phone-qr.png')
        (self.output / 'phone-connection.json').write_text(json.dumps(
            {'url': self.url, 'loopback_url': f'http://127.0.0.1:{self.port}/#token={self.token}',
             'port': self.port}, indent=2), encoding='utf-8')
        return self

    def _thread_main(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._start())
            self.ready.set()
            self.loop.run_forever()
        except Exception as error:
            self.error = str(error)
            self.ready.set()
        finally:
            self.loop.run_until_complete(self._shutdown())
            self.loop.close()

    async def _start(self):
        from aiohttp import web
        self.runner = web.AppRunner(self.make_application(), access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, '0.0.0.0', self.port)
        await site.start()
        if self.port == 0:
            self.port = site._server.sockets[0].getsockname()[1]

    def make_application(self):
        from aiohttp import web

        @web.middleware
        async def headers(request, handler):
            response = await handler(request)
            response.headers.update({'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
                'Referrer-Policy': 'no-referrer', 'X-Frame-Options': 'DENY'})
            return response

        app = web.Application(client_max_size=128*1024, middlewares=[headers])
        async def index(_request):
            return web.FileResponse(Path(__file__).parent / 'mobile' / 'index.html')
        async def state(request):
            self._authorize(request)
            return web.json_response(self.snapshot())
        app.router.add_get('/', index)
        app.router.add_get('/api/state', state)
        app.router.add_post('/offer', self._offer)
        return app

    def _authorize(self, request):
        from aiohttp import web
        if not secrets.compare_digest(request.headers.get('X-Race-Token', ''), self.token):
            raise web.HTTPForbidden(text='Mã ghép nối không đúng. Quét lại QR trên laptop.')
        origin = request.headers.get('Origin')
        if origin and origin != f'{request.scheme}://{request.host}':
            raise web.HTTPForbidden(text='Origin rejected')

    def snapshot(self):
        from src.launcher.cases import make_jsonable
        return make_jsonable(dict(self.control.snapshot(),stream=dict(self.stream_stats,**self.frames.stats())))

    def _connection_event(self, event, **details):
        """Keep timings/errors for local diagnosis, without tokens or SDP."""
        row={'time_s':round(time.time(),3),'event':event,**details}
        with (self.output/'phone-events.jsonl').open('a',encoding='utf-8') as log:
            log.write(json.dumps(row,ensure_ascii=False)+'\n')

    async def _offer(self, request):
        from aiohttp import web
        from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration
        self._authorize(request)
        owner = secrets.token_hex(12)
        if not self.control.claim(owner):
            raise web.HTTPConflict(text='Đã có điện thoại lái xe. Ngắt kết nối máy đó trước.')
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        self.peers.add(pc)
        channels = set()
        closing = False
        started = time.perf_counter()
        def connection_status():
            self.control.transport(owner, pc.connectionState,
                [channel.label for channel in channels if channel.readyState == 'open'])
        async def close(reason='closed'):
            nonlocal closing
            if closing:
                return
            closing = True
            self._connection_event('closed',reason=reason,transport=pc.connectionState,
                                   channels={c.label:c.readyState for c in channels})
            self.control.release(owner)
            self.peers.discard(pc)
            await pc.close()
        @pc.on('connectionstatechange')
        async def connection_changed():
            connection_status()
            self._connection_event('transport',state=pc.connectionState,
                                   elapsed_ms=round((time.perf_counter()-started)*1000,1))
            if pc.connectionState in ('failed','closed'):
                await close('transport_'+pc.connectionState)
        @pc.on('datachannel')
        def datachannel(channel):
            channels.add(channel)
            channel.on('open', connection_status)
            connection_status()
            @channel.on('message')
            def message(data):
                if not isinstance(data, str) or len(data) > 2048:
                    return
                try:
                    message=json.loads(data)
                    if isinstance(message,dict) and message.get('type')=='ping' and isinstance(message.get('t'),(float,int)):
                        channel.send(json.dumps({'type':'pong','t':message['t']}))
                    else:
                        self.control.receive(owner, message)
                except (ValueError, TypeError):
                    pass
            @channel.on('close')
            def channel_closed():
                self.control.release(owner)
                asyncio.create_task(close('channel_'+channel.label))
        try:
            params = await request.json()
            if params.get('type') != 'offer' or not isinstance(params.get('sdp'), str):
                raise ValueError('Invalid offer')
            track=self.video_track()
            pc.addTrack(track)
            from aiortc import RTCRtpSender
            codecs=[c for c in RTCRtpSender.getCapabilities('video').codecs if c.mimeType=='video/H264']
            for transceiver in pc.getTransceivers():
                if transceiver.kind=='video':
                    transceiver.setCodecPreferences(codecs)
            # aiortc fixes the common codec list while applying the remote
            # offer, so preferences must be set before that operation.
            await asyncio.wait_for(pc.setRemoteDescription(RTCSessionDescription(sdp=params['sdp'], type='offer')),10)
            await asyncio.wait_for(pc.setLocalDescription(await pc.createAnswer()),10)
            self.stream_stats['signaling_ms'] = round((time.perf_counter()-started)*1000,1)
            async def telemetry():
                reason='telemetry_ended'
                try:
                    while pc in self.peers:
                        connection_status()
                        if not self.control.snapshot()['connected'] and time.perf_counter()-started > 15:
                            reason='ready_timeout'
                            break
                        for channel in tuple(channels):
                            if channel.label == 'commands' and channel.readyState == 'open':
                                channel.send(json.dumps(self.snapshot(), ensure_ascii=False))
                        await asyncio.sleep(.2)
                except Exception as error:
                    reason='telemetry_error: '+str(error)
                    raise
                finally:
                    await close(reason)
            task = asyncio.create_task(telemetry())
            task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
            return web.json_response({'sdp': pc.localDescription.sdp, 'type': pc.localDescription.type})
        except Exception as error:
            await close('offer_error: '+str(error))
            raise web.HTTPBadRequest(text=str(error)) from error

    def video_track(self):
        from aiortc import VideoStreamTrack
        from concurrent.futures import ThreadPoolExecutor
        from src.visualization.phone_encoder import PhoneEncoder
        frames = self.frames
        host = self
        class RaceVideoTrack(VideoStreamTrack):
            def __init__(self):
                super().__init__()
                self.encoder=PhoneEncoder(host.encoder_preference,host.bitrate)
                self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='phone-h264')
                self.sequence=0
                self.origin=time.perf_counter()
                self.last_send=0.
                self.last_pts=-1
                self.next_encode=0.

            async def recv(self):
                return await asyncio.get_running_loop().run_in_executor(self.executor,self.next_packet)

            def next_packet(self):
                from aiortc.mediastreams import MediaStreamError
                while self.readyState=='live':
                    sequence,rgb,captured=frames.wait_latest(self.sequence)
                    now=time.perf_counter()
                    if rgb is None:
                        continue
                    # Pace from the beginning of an encode, not its end. A
                    # condition wake avoids coarse Windows asyncio polling.
                    delay=self.next_encode-now
                    if delay>0:
                        time.sleep(delay)
                        sequence,rgb,captured=frames.latest()
                        now=time.perf_counter()
                    self.next_encode=max(self.next_encode+1/30,now)
                    self.sequence=sequence
                    pts=max(self.last_pts+1,int((now-self.origin)*90000))
                    self.encoder.bitrate=host.bitrate
                    try:
                        packet=self.encoder.encode(rgb,pts)
                    except Exception as error:
                        host.stream_stats={'encoder':'error','hardware_encode':False,'encoder_error':str(error)}
                        raise
                    self.last_send=time.perf_counter(); self.last_pts=pts
                    host.stream_stats=self.encoder.stats()
                    if packet is not None:
                        return packet
                raise MediaStreamError

            def stop(self):
                if self.readyState=='ended':
                    return
                super().stop()
                # Close the process on its encoding thread after any in-flight
                # frame, including when the phone disconnects during encoding.
                self.executor.submit(self.encoder.close)
                self.executor.shutdown(wait=False)
        return RaceVideoTrack()

    async def _shutdown(self):
        await asyncio.gather(*(pc.close() for pc in list(self.peers)), return_exceptions=True)
        if hasattr(self, 'runner'):
            await self.runner.cleanup()
        tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def stop(self):
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=5)
