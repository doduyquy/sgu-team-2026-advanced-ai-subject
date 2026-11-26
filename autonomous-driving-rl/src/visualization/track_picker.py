"""Garage modal for the 12 TEST maps and their 60 dataset cases."""
from direct.gui import DirectGuiGlobals as DGG
from direct.gui.DirectGui import DirectFrame
from panda3d.core import Filename, TextNode

from src.visualization.test_track_catalog import TIERS, TIER_LABELS, TrackPickerDraft


class TrackPicker:
    def __init__(self, ui, catalog):
        self.ui, self.catalog = ui, catalog
        self.draft = TrackPickerDraft(catalog)
        self.is_open = False
        self.textures = {}
        self.root = ui.root.attachNewNode('test_track_picker')
        self.root.setBin('fixed',800)
        g = self.root
        # Consume pointer input over the underlying garage, including outside
        # the modal. Keyboard commands are gated by RaceApp while it is open.
        self.blocker = DirectFrame(parent=g,frameColor=(.003,.008,.015,.91),
                                   frameSize=(-2000,3280,-2800,2000),state=DGG.NORMAL,relief=None)
        ui.frame(g,80,55,1120,690,ui.BG,ui.BORDER_BRIGHT,radius=16)
        ui.label(g,'CHỌN BẢN ĐỒ TEST',110,101,26,ui.ACCENT,bold=True)
        ui.label(g,'12 bản đồ · 4 độ khó · 5 kịch bản cho mỗi bản đồ',110,130,14,ui.MUTED)
        ui.button(g,'Đóng ×',1080,82,90,'track_close',h=36,size=14,align=TextNode.ACenter)
        self.tabs = {tier:ui.button(g,TIER_LABELS[tier],110+i*265,151,245,'track_tier',tier,
                                    h=42,size=16,align=TextNode.ACenter) for i,tier in enumerate(TIERS)}
        self.cards, self.thumbs, self.names, self.details = [],[],[],[]
        for i in range(3):
            y = 213+i*106
            card = ui.button(g,'',110,y,445,'track_map',h=94)
            self.cards.append(card)
            self.thumbs.append(DirectFrame(parent=card,frameColor=(1,1,1,1),
                                            frameSize=(-37,37,-37,37),pos=(48,0,-47),
                                            relief=None,state=DGG.DISABLED))
            self.names.append(ui.label(card,'',101,33,18,bold=True))
            self.details.append(ui.label(card,'',101,59,12,ui.MUTED,wrap=320))
        ui.frame(g,580,213,590,340,ui.PANEL,ui.BORDER)
        self.preview = DirectFrame(parent=g,frameColor=(1,1,1,1),frameSize=(-155,155,-155,155),
                                    pos=(875,0,-383),relief=None,state=DGG.DISABLED)
        self.description = ui.label(g,'',592,579,17,bold=True)
        self.geometry = ui.label(g,'',592,604,12,ui.MUTED,wrap=570)
        self.traffic = ui.label(g,'',592,630,13,ui.ACCENT)
        ui.label(g,'KỊCH BẢN TRONG TẬP TEST',110,563,12,ui.MUTED,bold=True)
        self.scenarios = [ui.button(g,str(i+1),110+i*90,582,80,'track_scenario',i,
                                     h=43,size=17,align=TextNode.ACenter) for i in range(5)]
        self.seed = ui.label(g,'',110,650,13,ui.MUTED)
        ui.label(g,'Chạy thử dữ liệu TEST · Không ghi đè kết quả benchmark',110,726,11,ui.DIMMED)
        ui.button(g,'Về map gốc',110,674,180,'track_original',h=38,size=14,align=TextNode.ACenter)
        ui.button(g,'DÙNG BẢN ĐỒ NÀY →',860,674,310,'track_apply',h=43,size=16,
                  primary=True,align=TextNode.ACenter)
        self.root.hide()

    def texture(self, path):
        if path not in self.textures:
            self.textures[path] = self.ui.engine.loader.loadTexture(Filename.fromOsSpecific(str(path.resolve())))
        return self.textures[path]

    def show(self, current_case):
        self.draft.reset(current_case)
        self.refresh()
        self.is_open = True
        self.root.show()

    def hide(self):
        self.is_open = False
        self.root.hide()

    def refresh(self):
        ui, selected = self.ui, self.draft.track
        for tier,button in self.tabs.items():
            active = tier==selected.tier
            ui._paint(button,ui.ACCENT_DIM if active else ui.BUTTON,ui.ACCENT if active else ui.BORDER)
            button['text_fg'] = ui.ACCENT if active else ui.MUTED
        for card,thumb,name,detail,track in zip(self.cards,self.thumbs,self.names,self.details,
                                                self.catalog.in_tier(selected.tier)):
            card['extraArgs'] = ['track_map',track.geometry_id]
            active = track==selected
            ui._paint(card,ui.ACCENT_DIM if active else ui.BUTTON,ui.ACCENT if active else ui.BORDER)
            thumb['image'],thumb['image_scale'] = self.texture(track.preview),(36,1,36)
            name['text'] = track.sequence
            name['text_fg'] = ui.ACCENT if active else ui.INK
            detail['text'] = f'{track.length_m:.0f} m · Seed map {track.geometry_seed}\n{track.curves} khúc cua · {track.intersections} nút giao'
        self.preview['image'],self.preview['image_scale'] = self.texture(selected.preview),(154,1,154)
        self.description['text'] = f'{TIER_LABELS[selected.tier]} / {selected.sequence} · {selected.length_m:.0f} m'
        self.geometry['text'] = f'Seed hình học: {selected.geometry_seed} · {selected.curves} khúc cua · {selected.intersections} nút giao'
        case = self.draft.case
        self.traffic['text'] = f'Giao thông TEST: {case.traffic_density:.0%} · Giới hạn: {case.horizon_steps} bước AI'
        self.seed['text'] = f'Kịch bản {self.draft.scenario+1}/5 · Seed môi trường {case.environment_seed}'
        for i,button in enumerate(self.scenarios):
            active = i==self.draft.scenario
            ui._paint(button,ui.ACCENT_DIM if active else ui.BUTTON,ui.ACCENT if active else ui.BORDER)
            button['text_fg'] = ui.ACCENT if active else ui.MUTED

    def handle(self, command, arg):
        if command=='track_tier':self.draft.tier(arg)
        elif command=='track_map':self.draft.map(arg)
        elif command=='track_scenario':self.draft.choose_scenario(arg)
        else:return False
        self.refresh()
        return True
