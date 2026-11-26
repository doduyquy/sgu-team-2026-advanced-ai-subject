"""Persistent NVENC for drivers older than PyAV's bundled NVENC SDK.

The pinned imageio-ffmpeg wheel supplies FFmpeg 7.1. Raw RGB travels through
a pipe; local RTP supplies explicit frame boundaries without waiting for the
next frame. Only complete H.264 access units leave this adapter.
"""
from collections import deque
import queue
import socket
import subprocess
import threading

import av


class NvencProcess:
    backend = 'ffmpeg_nvenc'

    def __init__(self, width, height, bitrate):
        import imageio_ffmpeg
        self.width, self.height, self.bit_rate = width, height, bitrate
        self.frames = queue.Queue(maxsize=2)
        self.errors = deque(maxlen=12)
        self.closed = threading.Event()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)
        self.socket.bind(('127.0.0.1', 0))
        self.socket.settimeout(.2)
        port = self.socket.getsockname()[1]
        command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
                   '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
                   '-framerate', '30', '-i', 'pipe:0', '-an', '-c:v', 'h264_nvenc',
                   '-gpu', '0', '-preset', 'p1', '-tune', 'ull', '-rc', 'cbr',
                   '-b:v', str(bitrate), '-maxrate', str(bitrate), '-bufsize', str(bitrate//30),
                   '-pix_fmt', 'yuv420p', '-profile:v', 'baseline', '-bf', '0', '-g', '15',
                   '-zerolatency', '1', '-delay', '0', '-flush_packets', '1',
                   '-f', 'rtp', f'rtp://127.0.0.1:{port}?pkt_size=1200']
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                            stderr=subprocess.PIPE, bufsize=0,
                                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except Exception:
            self.socket.close()
            raise
        self.readers = [threading.Thread(target=f, daemon=True, name=name) for f,name in
                        ((self._receive, 'nvenc-rtp'), (self._read_errors, 'nvenc-stderr'))]
        for reader in self.readers:
            reader.start()

    def _read_errors(self):
        for line in self.process.stderr:
            self.errors.append(line.decode('utf-8', errors='replace').strip())

    def _receive(self):
        access_unit = bytearray()
        previous_sequence = timestamp = None
        damaged = False
        while not self.closed.is_set():
            try:
                packet = self.socket.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                return
            if len(packet) < 12 or packet[0] >> 6 != 2:
                continue
            sequence = int.from_bytes(packet[2:4], 'big')
            current_timestamp = packet[4:8]
            if current_timestamp != timestamp:
                access_unit.clear()
                timestamp, damaged = current_timestamp, False
            if previous_sequence is not None and sequence != (previous_sequence+1) % 65536:
                damaged = True
            previous_sequence = sequence
            offset = 12 + 4*(packet[0] & 15)
            if packet[0] & 16:
                if len(packet) < offset+4:
                    continue
                offset += 4 + 4*int.from_bytes(packet[offset+2:offset+4], 'big')
            payload = packet[offset:len(packet)-packet[-1] if packet[0] & 32 else len(packet)]
            if not payload:
                continue
            kind = payload[0] & 31
            if 1 <= kind <= 23:
                access_unit.extend(b'\x00\x00\x00\x01'+payload)
            elif kind == 24:  # STAP-A: several small NALs, e.g. SPS/PPS.
                at = 1
                while at+2 <= len(payload):
                    size = int.from_bytes(payload[at:at+2], 'big'); at += 2
                    if not size or at+size > len(payload):
                        damaged = True
                        break
                    access_unit.extend(b'\x00\x00\x00\x01'+payload[at:at+size]); at += size
            elif kind == 28 and len(payload) >= 2:  # FU-A.
                if payload[1] & 128:
                    access_unit.extend(b'\x00\x00\x00\x01'+bytes([(payload[0]&224)|(payload[1]&31)]))
                access_unit.extend(payload[2:])
            else:
                damaged = True
            if packet[1] & 128:
                if not damaged and access_unit:
                    try:
                        self.frames.put_nowait(bytes(access_unit))
                    except queue.Full:
                        pass
                access_unit.clear()

    def encode(self, frame):
        if self.closed.is_set():
            raise RuntimeError('NVENC encoder is closed')
        try:
            rgb = frame.to_ndarray(format='rgb24').tobytes()
            remaining = memoryview(rgb)
            while remaining:
                written = self.process.stdin.write(remaining)
                if not written:
                    raise BrokenPipeError('FFmpeg input closed')
                remaining = remaining[written:]
            encoded = self.frames.get(timeout=3)
            return [av.Packet(encoded)]
        except (OSError, queue.Empty) as error:
            details = '; '.join(self.errors)
            self.close()
            raise RuntimeError(f'RTX NVENC failed: {details or error}') from error

    def close(self):
        if self.closed.is_set():
            return
        self.closed.set()
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)
        self.socket.close()
        self.process.stdin.close()
        for reader in self.readers:
            if reader is not threading.current_thread():
                reader.join(timeout=1)
        self.process.stderr.close()
