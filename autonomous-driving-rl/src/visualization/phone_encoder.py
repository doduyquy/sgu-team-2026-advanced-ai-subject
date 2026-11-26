"""Per-stream H.264 encoder; hardware probe, zero B-frames and safe fallback.

    Encoded av.Packets use aiortc's public pre-encoded media path. This does
    not replace aiortc internals or edit the installed Python environment.
"""

from collections import deque
from fractions import Fraction
import time

import av


class PhoneEncoder:
    NAMES = {'auto': ('h264_nvenc', 'libx264'),
             'nvenc': ('h264_nvenc',),
             'qsv': ('h264_qsv', 'libx264'), 'cpu': ('libx264',)}

    def __init__(self, preference='auto', bitrate=2500000):
        if preference not in self.NAMES:
            raise ValueError('Unknown phone encoder')
        self.preference, self.bitrate = preference, bitrate
        self.codec = self.name = None
        self.failed = {}
        self.samples = deque(maxlen=120)
        self.count = 0
        self.last_frame_id = 0
        self.native_nvenc_error = ''

    def _open(self, name, width, height):
        if name != 'h264_nvenc':
            return self._open_pyav(name, width, height)
        try:
            with av.logging.Capture():
                return self._open_pyav(name, width, height)
        except Exception as error:
            # Keep using NVIDIA with the compatible FFmpeg binary if PyAV
            # was compiled against an NVENC SDK newer than this driver.
            self.native_nvenc_error = str(error)
            from src.visualization.nvenc_process import NvencProcess
            return NvencProcess(width, height, self.bitrate)

    def _open_pyav(self, name, width, height):
        codec = av.CodecContext.create(name, 'w')
        codec.width, codec.height = width, height
        codec.pix_fmt = 'nv12' if name == 'h264_qsv' else 'yuv420p'
        codec.bit_rate = self.bitrate
        codec.time_base, codec.framerate = Fraction(1, 90000), Fraction(30, 1)
        codec.max_b_frames, codec.gop_size = 0, 15
        codec.profile = 'Baseline'
        if name == 'h264_nvenc':
            codec.options = {'preset': 'p1', 'tune': 'ull', 'rc': 'cbr', 'zerolatency': '1', 'delay': '0'}
        elif name == 'h264_qsv':
            codec.options = {'async_depth': '1', 'look_ahead': '0', 'preset': 'veryfast'}
        else:
            codec.options = {'preset': 'ultrafast', 'tune': 'zerolatency', 'level': '31',
                             'x264-params': 'repeat-headers=1:scenecut=0'}
        codec.open()
        return codec

    def encode(self, rgb, pts):
        started = time.perf_counter()
        height, width = rgb.shape[:2]
        if self.codec is not None and (self.codec.width != width or self.codec.height != height or self.codec.bit_rate!=self.bitrate):
            self.close()
        frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
        frame.pts, frame.time_base = pts, Fraction(1, 90000)
        # Repeat IDR/SPS frequently for Wi-Fi loss and resolution changes.
        if self.count % 15 == 0:
            frame.pict_type = av.video.frame.PictureType.I
        for name in self.NAMES[self.preference]:
            if name in self.failed:
                continue
            try:
                if self.codec is None or self.name != name:
                    self.codec = self._open(name, width, height)
                    self.name = name
                    frame.pict_type = av.video.frame.PictureType.I
                packets = self.codec.encode(frame)
                self.count += 1
                self.samples.append((time.perf_counter()-started)*1000)
                if not packets:
                    return None
                packet = av.Packet(b''.join(bytes(packet) for packet in packets))
                packet.pts = packet.dts = pts
                packet.time_base = Fraction(1, 90000)
                return packet
            except Exception as error:
                self.failed[name] = str(error)
                self.close()
        raise RuntimeError(f'No H.264 encoder available: {self.failed}')

    def close(self):
        codec, self.codec = self.codec, None
        if codec is not None and callable(getattr(codec, 'close', None)):
            codec.close()

    def stats(self):
        samples = sorted(self.samples)
        return {'encoder': self.name or 'starting', 'hardware_encode': self.name in ('h264_nvenc','h264_qsv'),
                'encoder_backend':getattr(self.codec,'backend','pyav'),
                'native_nvenc_error':self.native_nvenc_error,
                'encode_ms': round(samples[-1] if len(samples)==1 else sum(samples)/len(samples),2) if samples else 0,
                'encode_p95_ms': round(samples[min(len(samples)-1,int(len(samples)*.95))],2) if samples else 0,
                'encoded_frames': self.count, 'encoder_fallbacks': dict(self.failed)}
