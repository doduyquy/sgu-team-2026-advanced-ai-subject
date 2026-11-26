"""Local rendering budgets, independent of policies and physics timing."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time


@dataclass(frozen=True)
class GraphicsProfile:
    name: str
    msaa_samples: int
    scene_scale: float
    preview_size: tuple
    texture_limit: int
    scene_pixel_budget: int = 0
    render_fps: int = 60
    garage_fps: int = 30

    def to_dict(self):
        return {**asdict(self), 'shadows': False, 'texture_compression': 'vehicle_artwork_only',
                'vehicle_asset_source': {'lamborghini':'cc_by_downloaded',
                                         'tesla':'cc_by_downloaded','vf3':'procedural_original'}}


PROFILES = {
    'lite': GraphicsProfile('lite', 2, .75, (960, 330), 512, scene_pixel_budget=1152*720),
    'balanced': GraphicsProfile('balanced', 2, 1., (1280, 440), 1024),
}
_profile = PROFILES['lite']


def scene_buffer_ratio(profile, width, height):
    """Cap only 3D pixels when maximizing, keeping all HUD text native-sized."""
    scale=profile.scene_scale
    if profile.scene_pixel_budget:
        scale=min(scale,math.sqrt(profile.scene_pixel_budget/max(1,width*height)))
    # Pinned FilterManager divides integer window sizes BEFORE multiplying.
    # A fixed denominator of 1000 would produce a zero-height scene at 800p.
    # Choose a small rational below the budget, preserving enough input rows.
    candidates=[(max(1,int(scale*div)),div) for div in range(1,33)]
    return max((pair for pair in candidates if pair[0]/pair[1]<=scale+1e-9),
               key=lambda pair:pair[0]/pair[1])


def get_graphics_profile():
    return _profile


def configure_graphics(profile='lite'):
    """Apply before creating any engine or loading textures."""
    global _profile
    _profile = PROFILES[profile]
    from panda3d.core import loadPrcFileData
    cache = Path(__file__).resolve().parents[2] / 'artifacts' / 'panda-visual-cache-v2'
    loadPrcFileData('race_local_graphics', '\n'.join((
        'sync-video false', 'win-size 1280 800',
        # Height/semantic maps are simulator data, not artwork. A global cap
        # would resize MetaDrive's numeric terrain input and change physics.
        'max-texture-dimension 0',
        # MetaDrive reads some texture RAM as numerical RGB arrays. Compress
        # only copied vehicle artwork below, never those simulator inputs.
        'compressed-textures false', 'model-cache-compressed-textures false',
        f'model-cache-dir {cache.as_posix()}',
    )))
    return _profile


_TEXTURE_CACHE = {}
_OPTIMIZED_TEXTURES = set()


def limit_visual_textures(root):
    """Resize only vehicle artwork; never terrain, sensors, UI or data maps."""
    from panda3d.core import PNMImage, Texture
    limit = get_graphics_profile().texture_limit
    for texture in root.findAllTextures():
        if (texture, limit) in _OPTIMIZED_TEXTURES:
            continue
        width, height = texture.getXSize(), texture.getYSize()
        key = (texture, limit)
        if key not in _TEXTURE_CACHE:
            replacement = texture.makeCopy()
            if max(width, height) > limit:
                source = PNMImage()
                if not texture.store(source):
                    continue
                scale = limit / max(width, height)
                reduced = PNMImage(max(1, round(width*scale)), max(1, round(height*scale)),
                                   source.getNumChannels())
                reduced.quickFilterFrom(source)
                replacement.load(reduced)
                replacement.setFormat(texture.getFormat())
            replacement.setCompression(Texture.CMOn)
            _TEXTURE_CACHE[key] = replacement
            _OPTIMIZED_TEXTURES.add((replacement, limit))
        root.replaceTexture(texture, _TEXTURE_CACHE[key])


class RaceGraphicsRuntime:
    """Gate draw outputs, keeping taskMgr, event processing and physics alive.

    Interactive phone racing owns its own draw gate; this task leaves it alone.
    FilterManager tracks resized dimensions so window resizes retain the scale.
    """

    def __init__(self, app):
        self.app = app
        self.engine = app.env.engine
        self.profile = get_graphics_profile()
        self.uncapped = getattr(app, 'graphics_benchmark_uncapped', False)
        self.next_render = 0.
        self.rendered_frames = 0
        self._task = f'race_draw_budget_{id(self)}'
        pipe = self.engine.pbrpipe
        self.manager=pipe.manager if pipe else None
        self._window_size=None
        self.buffers = [self.engine.win] + list(pipe.manager.buffers if pipe else [])
        self._resize_scene()
        self.engine.taskMgr.add(self._tick, self._task, sort=45)

    def _resize_scene(self):
        size=(self.engine.win.getXSize(),self.engine.win.getYSize())
        if size==self._window_size or self.manager is None:
            return
        self._window_size=size
        mul,div=scene_buffer_ratio(self.profile,*size)
        for i,(_,_,align) in enumerate(self.manager.sizes):
            self.manager.sizes[i]=(mul,div,align)
        self.manager.resizeBuffers()

    def _tick(self, task):
        self._resize_scene()
        if self.app.state == 'running' and getattr(self.app, 'realtime', None):
            return task.cont
        now = time.perf_counter()
        fps = self.profile.garage_fps if self.app.state == 'garage' else self.profile.render_fps
        due = self.uncapped or now >= self.next_render
        if due:
            self.rendered_frames += 1
            self.next_render = now + 1. / fps
        for buffer in self.buffers:
            buffer.setActive(due)
        preview = getattr(getattr(self.app, 'ui', None), 'preview', None)
        if preview:
            preview.set_active(due and self.app.state == 'garage')
        return task.cont

    def close(self):
        self.engine.taskMgr.remove(self._task)
        for buffer in self.buffers:
            buffer.setActive(True)
