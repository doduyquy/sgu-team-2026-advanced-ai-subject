"""Hold-drag input for the garage and race camera, scoped away from UI."""

from dataclasses import dataclass
import math
import time


@dataclass
class GarageOrbit:
    yaw: float = 10.
    elevation: float = 20.
    distance: float = 15.8

    def drag(self, dx, dy):
        self.yaw = (self.yaw + dx*.50) % 360
        self.elevation = max(8., min(42., self.elevation + dy*.18))

    def zoom(self, steps):
        self.distance = max(10., min(23., self.distance*math.exp(-steps*.10)))


def design_coordinates(mouse, size):
    """Convert Panda [-1,1] pointer to the letterboxed 1280×800 design."""
    width,height = size
    scale = min(width/1280.,height/800.)
    return (((mouse[0]+1)*width/2-(width-1280*scale)/2)/scale,
            ((1-mouse[1])*height/2-(height-800*scale)/2)/scale)


class GaragePointer:
    RECT = (500, 232, 1226, 458)

    def __init__(self, app):
        from direct.showbase.DirectObject import DirectObject
        self.app = app
        self.last = self.click = self.race_last = None
        self.last_click = -1.
        self.events = DirectObject()
        self.name = f'garage_orbit_{id(self)}'
        # This controller owns scene hold-drag. Never forward UI clicks to
        # MetaDrive's click-to-pan handler.
        self.camera = app.env.main_camera
        for event in ('mouse1','wheel_up','wheel_down'):
            app.env.engine.ignore(event)
        for event, callback, arguments in (
            ('mouse1',self.press,[]),('mouse1-up',self.release,[]),
            ('wheel_up',self.wheel,[1]),('wheel_down',self.wheel,[-1]),
        ):
            self.events.accept(event,callback,arguments)
        app.env.engine.taskMgr.add(self.tick,self.name,sort=30)

    def position(self):
        watcher = self.app.env.engine.mouseWatcherNode
        if not watcher or not watcher.hasMouse():
            return None
        win=self.app.env.engine.win
        return design_coordinates(watcher.getMouse(),(win.getXSize(),win.getYSize()))

    def inside(self, position):
        picker=getattr(self.app.ui,'track_picker',None)
        if picker is not None and picker.is_open is True:
            return False
        return position is not None and self.app.state=='garage' and (
            self.RECT[0] <= position[0] <= self.RECT[2] and
            self.RECT[1] <= position[1] <= self.RECT[3])

    def over_ui(self):
        ui = self.app.ui
        panel = getattr(ui,'phone_panel',None)
        if panel is not None and not panel.isHidden():
            return True
        engine = getattr(getattr(self.app,'env',None),'engine',None)
        if engine is not None:
            region = engine.mouseWatcherNode.getOverRegion()
            if region is not None and region.getName().startswith('pg'):
                return True
        return False

    def in_race_scene(self, position):
        if position is None or self.app.state not in ('running','paused') or self.over_ui():
            return False
        x,y = position
        if not (0<=x<=1280 and 0<=y<700):
            return False
        if (x<250 and y<210) or (280<=x<=1245 and y<85):
            return False
        if len(getattr(self.app,'racers',{}))>1 and x>=990 and 90<=y<=500:
            return False
        if getattr(self.app,'details_visible',False) and x>=650:
            return False
        if self.app.state=='paused' and 360<=x<=920 and 180<=y<=560:
            return False
        return True

    def button_held(self):
        from panda3d.core import MouseButton
        engine = getattr(getattr(self.app,'env',None),'engine',None)
        return engine is None or engine.mouseWatcherNode.isButtonDown(MouseButton.one())

    def press(self):
        self.release()
        position=self.position()
        if self.over_ui():
            return
        if self.inside(position):
            now=time.perf_counter()
            if now-self.last_click<.30 and self.click and math.dist(position,self.click)<7:
                self.app.ui.preview.reset_view()
                self.last=None
            else:
                self.last=position
            self.click,self.last_click=position,now
        elif self.in_race_scene(position):
            self.race_last=position

    def release(self):
        self.last=None
        self.race_last=None

    def wheel(self, steps):
        if self.inside(self.position()):
            self.app.ui.preview.orbit.zoom(steps)
            self.app.ui.preview.apply_view()
        elif self.app.state!='garage' and self.camera:
            (self.camera._wheel_up_height if steps>0 else self.camera._wheel_down_height)()

    def tick(self, task):
        if not self.button_held():
            self.release()
        if self.app.state!='garage':
            self.last=None
        elif self.last is not None:
            position=self.position()
            if position is None:
                self.last=None
            else:
                self.app.ui.preview.orbit.drag(position[0]-self.last[0],position[1]-self.last[1])
                self.app.ui.preview.apply_view()
                self.last=position
        if self.app.state not in ('running','paused'):
            self.race_last=None
        elif getattr(self,'race_last',None) is not None:
            position=self.position()
            if not self.in_race_scene(position):
                self.race_last=None
            else:
                control=getattr(self.app,'camera_input',None)
                if control is not None:
                    control.drag(position[0]-self.race_last[0],position[1]-self.race_last[1])
                self.race_last=position
        return task.cont

    def close(self):
        self.events.ignoreAll()
        self.app.env.engine.taskMgr.remove(self.name)
