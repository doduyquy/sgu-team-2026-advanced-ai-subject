"""Add phone racing to the existing menu without replacing its visual design."""

from dataclasses import replace
from direct.gui import DirectGuiGlobals as DGG
from direct.gui.DirectGui import DirectFrame
from panda3d.core import TextNode
from src.visualization.race_menu import RaceMenu
from src.visualization.race_session import HUMAN_INDEX, race_participant, OUTCOME_LABELS, ROAD_NAMES
from src.visualization.race_style import RACE_COLORS
from src.visualization.race_vehicle_models import CAR_MODELS, CAR_NAMES
from src.visualization.race_rules import ordered_racers, race_positions


class PhoneRaceMenu(RaceMenu):
    def __init__(self, engine, enqueue):
        self.track_picker = None
        super().__init__(engine, enqueue)
        g = self.groups['garage']
        self.mode_human = self.button(g, 'Bạn đua với AI', 500, 145, 224, 'mode', 'human', h=38, size=13, align=TextNode.ACenter)
        self.mode_practice = self.button(g, 'Lái tự do (1 xe)', 728, 145, 224, 'mode', 'practice', h=38, size=13, align=TextNode.ACenter)
        self.phone_button = self.button(g, 'Kết nối Điện thoại · QR', 970, 142, 270, 'phone', h=44, size=13, align=TextNode.ACenter)
        self._paint(self.phone_button, self.BUTTON, border=self.BORDER_BRIGHT)
        self.phone_button['text_fg'] = self.INK

        for attr in ('preview_spec_mech', 'preview_spec_mem', 'preview_spec_app'):
            getattr(self, attr).hide()
        if hasattr(self, 'spec_bar'):
            self.spec_bar.hide()
        if hasattr(self, 'showroom_telemetry_label'):
            self.showroom_telemetry_label.hide()

        self.model_buttons = [self.button(g, CAR_NAMES[model], 504 + i * 242, 452, 234,
                              'car_model', model, h=38, size=13, align=TextNode.ACenter) for i, model in enumerate(CAR_MODELS)]

        self.model_plaque = self.frame(g, 504, 304, 240, 96, color=(.050, .058, .072, .95),
                                       border=self.BORDER, radius=6)
        self.model_brand = self.label(g, 'LAMBORGHINI RACING', 520, 324, 10, self.MUTED, bold=True)
        self.model_heading = self.label(g, 'AVENTADOR LP 700', 520, 348, 16, self.INK, bold=True)
        self.model_description = self.label(g, 'V12 6.5L · AWD · Thân sợi Carbon', 520, 372, 11, self.MUTED, wrap=210)

        self.orbit_guide = self.label(g, 'Kéo chuột: Xoay 360°  ·  Cuộn: Zoom  ·  Đúp: Mặc định',
                                      1220, 226, 10, self.MUTED, align=TextNode.ARight)
        self.model_credit = self.label(g, '', 1220, 248, 9, self.DIMMED, align=TextNode.ARight)
        self._build_phone_panel()
        self.test_track_summary = self.button(g, '', 506, 556, 712, 'track_open', h=76, size=13)
        self.test_track_summary.hide()
        self.vn_route_status=self.label(self.groups['running'],'',294,98,12,self.GOLD,bold=True)
        self.vn_attribution=self.label(self.groups['running'],'',18,220,9,self.MUTED)

    def show(self, state):
        super().show(state)
        if state!='garage' and self.track_picker:
            self.track_picker.hide()

    def open_tracks(self, catalog, case):
        if self.track_picker is None:
            from src.visualization.track_picker import TrackPicker
            self.track_picker = TrackPicker(self,catalog)
        self.phone_panel.hide()
        self.track_picker.show(case)

    def _build_running(self):
        super()._build_running()
        # Row 5 (6th row) for HUMAN player
        row = self.button(self.standings, '', 988, 144 + 5 * 48, 244,
                          'number', HUMAN_INDEX, h=44, size=12, left_stripe=RACE_COLORS[HUMAN_INDEX])
        self.standing_rows.append(row)
        # Phone pairing and car recovery buttons integrated cleanly into the bottom toolbar
        self.phone_run_button = self.button(self.groups['running'], 'Điện thoại / QR', 48, 714, 154, 'phone', h=40, size=13, align=TextNode.ACenter)
        self.recover_button = self.button(self.groups['running'], 'Về giữa đường · [H]', 210, 714, 178, 'recover', h=40, size=13, align=TextNode.ACenter)

    def _build_comparison(self):
        g = self.groups['comparison']
        self.frame(g, 25, 75, 1230, 650, self.BG, self.BORDER, radius=16)
        self.comparison_title = self.label(g, 'BẢNG SO SÁNH KẾT QUẢ ĐUA', 55, 120, 26, self.ACCENT, bold=True)
        self.comparison_context = self.label(g, '', 55, 158, 14, self.MUTED, wrap=1160)
        columns = ((60, 'Người lái / Thuật toán'), (300, 'Trạng thái'), (560, 'Tiến độ đường'),
                   (720, 'TG mô phỏng'), (880, 'Va chạm / Ra đường'), (1080, 'Độ trễ TB'))
        for x, title in columns:
            self.label(g, title, x, 206, 13, self.MUTED, bold=True)
        self.comparison_rows = []
        for i in range(6):
            y = 246 + i * 56
            self.frame(g, 45, y - 26, 1180, 50, (.042, .060, .088, 1) if i % 2 == 0 else self.PANEL,
                       border=self.BORDER, radius=10, left_stripe=RACE_COLORS[i])
            self.comparison_rows.append([self.label(g, '', x, y, 15, wrap=255 if x == 300 else None)
                                         for x, _ in columns])
        self.label(g, 'Mô phỏng MetaDrive · Cùng thế giới vật lý · Các xe tương tác và va chạm trực tiếp', 55, 595, 13, self.MUTED)
        self.button(g, 'Chạy lại cả nhóm [R]', 55, 636, 250, 'restart', primary=True, h=50, size=15, align=TextNode.ACenter)
        self.button(g, 'Về Garage [G]', 325, 636, 200, 'garage', h=50, size=15, align=TextNode.ACenter)
        self.button(g, 'Thoát [Esc]', 545, 636, 150, 'quit', h=50, size=15, align=TextNode.ACenter)

    def update_comparison(self, comparison):
        self.comparison_title['text'] = f"KẾT QUẢ CUỘC ĐUA · {len(comparison.results)}/{len(comparison.indices)} XE ĐÃ DỪNG"
        case = comparison.case
        source = 'Map gốc' if case.get('case_id', '').startswith('original/') else case.get('split', 'TRAIN')
        seq = case.get('sequence', '')
        seed = case.get('geometry_generation_seed', case.get('environment_seed', ''))
        self.comparison_context['text'] = f"Đường đua {seq} · Seed môi trường: {seed} · Phân vùng: {source}"
        rows = {r['agent_id']: r for r in comparison.results}
        finishers = [r for r in comparison.results if r.get('clean_success')]
        ranks = {r['agent_id']:1+sum(other['simulation_time_s'] < r['simulation_time_s'] for other in finishers)
                 for r in finishers}
        for i, labels in enumerate(self.comparison_rows):
            result = rows.get(race_participant(comparison.indices[i])[0]) if i < len(comparison.indices) else None
            values = [''] * 6
            if result:
                route, latency, flags = result['route_completion'], result['mean_act_latency_ms'], result['flags']
                crashes = any(v for k, v in flags.items() if k.startswith('crash_'))
                out_of_road = flags.get('out_of_road', False)
                reason_raw = result['reason']
                reason_str = OUTCOME_LABELS.get(reason_raw, reason_raw)
                values = [
                    (f"#{ranks[result['agent_id']]} " if result['agent_id'] in ranks else '— ') + race_participant(comparison.indices[i])[1].replace(' · thử nghiệm', ''),
                    reason_str,
                    f'{route * 100:.1f}%' if route is not None else '—',
                    f"{result['simulation_time_s']:.1f} s" if result.get('simulation_time_s') is not None else '—',
                    f"{'Có' if crashes else 'Không'} / {'Có' if out_of_road else 'Không'}",
                    f'{latency:.1f} ms' if latency is not None else '—',
                ]
            for col_idx, (label, value) in enumerate(zip(labels, values)):
                label['text'] = value
                if col_idx == 0 and result:
                    label['text_fg'] = RACE_COLORS[comparison.indices[i]]
                elif col_idx == 1 and result:
                    if result['reason'] == 'SUCCESS':
                        label['text_fg'] = self.SUCCESS
                    elif 'CRASH' in result['reason']:
                        label['text_fg'] = self.DANGER
                    elif result['reason'] in ('OUT_OF_ROAD', 'SOLID_LINE'):
                        label['text_fg'] = self.GOLD
                    else:
                        label['text_fg'] = self.INK
                else:
                    label['text_fg'] = self.INK

    def update_standings(self, racers, env):
        self.standings.show() if len(racers) > 1 else self.standings.hide()
        order = ordered_racers(racers.values())
        ranks = race_positions(order)
        for slot, button in enumerate(self.standing_rows):
            if slot >= len(order):
                button.hide()
                continue
            button.show()
            racer = order[slot]
            index, key = racer.index, racer.vehicle_id
            button['extraArgs'] = ['number', index]
            button['text_pos'] = (16, -18)
            progress = f'{racer.route * 100:.1f}%' if racer.route is not None else '—'
            status = OUTCOME_LABELS.get(racer.summary['reason'], racer.summary['reason']) if racer.summary else f'{env.agents[key].speed_km_h:.0f} km/h'
            prefix = '► ' if env.focus_id == key else '   '
            color = RACE_COLORS[index]
            name = race_participant(index)[1].replace(' · thử nghiệm', '')
            rank = str(ranks[key]) if ranks[key] is not None else 'LOẠI'
            button['text'] = f"{prefix}{rank}  {name}\n      Tiến độ: {progress}  ·  {status}"
            button['text_scale'] = 12
            button['text_fg'] = color
            self._paint(button, (.055, .15, .16, 1) if env.focus_id == key else self.BUTTON,
                        color if env.focus_id == key else self.BORDER,
                        left_stripe=color)

    def update_selection(self, selection, roads):
        # Base menu continues to own its algorithm and road widgets.
        if self._preview_index == HUMAN_INDEX:
            self._preview_index=0
        proxy=replace(selection,mode='compare' if selection.mode=='human' else 'single') if selection.manual else selection
        super().update_selection(proxy,roads)
        self.vn_route_status['text']=''
        vietnam=selection.test_case is None and selection.road_index==3
        self.vn_attribution['text']='© OpenStreetMap contributors · ODbL' if vietnam else ''
        for mode, button in [('single', self.mode_single), ('compare', self.mode_compare),
                             ('human', self.mode_human), ('practice', self.mode_practice)]:
            active = (selection.mode == mode)
            self._paint(button, (.18, .21, .27, 1.) if active else self.BUTTON,
                        self.BORDER_BRIGHT if active else self.BORDER,
                        border_width=1.5 if active else 1.0)
            button['text_fg'] = self.INK if active else self.MUTED
            button['text_font'] = self.bold_font if active else self.font
        self.mode_hint['text'] = {
            'single': 'Chế độ Chạy đơn: Xem 1 thuật toán AI lái xe. Phân tích chi tiết quỹ đạo, góc lái và độ trễ.',
            'compare': 'Chế độ Đua đồng thời: 2–5 thuật toán AI cùng xuất phát, tranh chấp đường đua và tương tác vật lý trực tiếp.',
            'human': 'Chế độ Bạn đua với AI: Bạn trực tiếp điều khiển xe đối đầu cùng các thuật toán AI qua điện thoại / bàn phím.',
            'practice': 'Chế độ Lái tự do: Một mình bạn làm chủ đường đua, làm quen góc cua và kiểm tra độ trễ điều khiển.',
        }[selection.mode]
        if selection.manual:
            self._preview_index = HUMAN_INDEX
            self.preview_name['text'] = 'BẠN / ' + CAR_NAMES[selection.player_model]
            self.preview_name['text_fg'] = RACE_COLORS[HUMAN_INDEX]
            self.preview_category['text'] = 'Điện thoại / Bàn phím cầm lái · Camera góc nhìn thứ ba · Laptop quan sát'
            self.selected['text'] = f'ĐÃ CHỌN: {len(selection.chosen_indices):02d} AI + BẠN' if selection.mode == 'human' else 'BẠN CẦM LÁI'
            self.speed_button['text'] = 'Bạn cầm lái · Cố định 1×'
            ready = selection.mode == 'practice' or bool(selection.chosen_indices)
            self.start_button['state'] = DGG.NORMAL if ready else DGG.DISABLED
            self.start_button['text'] = 'XUẤT PHÁT  →' if ready else 'CHỌN THÊM AI'
            self._paint(self.start_button, self.LAUNCH_RED if ready else self.BUTTON,
                        border=(1., .4, .4, 1.) if ready else self.BORDER, border_width=1.5)
            self.start_button['text_fg'] = self.INK if ready else self.DIMMED
            self.status['text'] = (
                f"[SẴN SÀNG]  {'Bạn lái tự do' if selection.mode == 'practice' else f'Bạn + {len(selection.chosen_indices):02d} AI'}  ·  Đường {ROAD_NAMES[selection.road_index]} ({roads[selection.road_index][0].sequence})"
                if ready else 'Chọn ít nhất 1 AI để bắt đầu đua đối đầu.'
            )
            self.status['text_fg'] = self.SUCCESS if ready else self.GOLD
            self.single_agent_spec_title['text'] = f"ĐIỀU KHIỂN TRỰC TIẾP · {CAR_NAMES[selection.player_model].upper()}"
            self.single_agent_spec_title['text_fg'] = RACE_COLORS[HUMAN_INDEX]
            self.single_agent_spec_detail['text'] = 'Điều khiển: Điện thoại qua Wi-Fi/QR  ·  Bàn phím [W, A, S, D] hoặc [Phím Mũi Tên]'
        for i, button in enumerate(self.agent_buttons):
            button['state'] = DGG.DISABLED if selection.mode == 'practice' else DGG.NORMAL
            if selection.mode == 'practice':
                marker, plus = self.agent_markers[i]
                marker.hide(); plus.show()
                self._paint(button, self.BUTTON, self.BORDER)
                self.agent_titles[i]['text_fg'] = self.MUTED
        for attr in ('preview_spec_mech', 'preview_spec_mem', 'preview_spec_app'):
            getattr(self, attr).hide()
        model = selection.model_for(self._preview_index)
        brand, title, description = {
            'lamborghini': ('LAMBORGHINI RACING', 'AVENTADOR LP 700', 'V12 6.5L · AWD · Thân sợi Carbon'),
            'tesla': ('TESLA', 'CYBERTRUCK', 'Thép bạc · Kính tối · Bánh xe chuyển động'),
            'vf3': ('VINFAST AUTO', 'VF 3 ELECTRIC', 'Mini EV đô thị · Gầm cao 175mm · Bán kính nhỏ'),
        }[model]
        self.model_brand['text'], self.model_heading['text'] = brand, title
        self.model_brand['text_fg'] = RACE_COLORS[self._preview_index]
        self.model_description['text'] = description
        self.model_credit['text'] = {'lamborghini': 'Arion Digital · CC BY 4.0',
                                     'tesla': 'Nieve5677 · CC BY 4.0',
                                     'vf3': 'Mẫu dựng minh họa'}[model]
        for key, button in zip(CAR_MODELS, self.model_buttons):
            active = (key == model)
            self._paint(button, (.20, .24, .30, 1.) if active else self.BUTTON,
                        self.BORDER_BRIGHT if active else self.BORDER,
                        border_width=1.5 if active else 1.0)
            button['text_fg'] = self.INK if active else self.MUTED
            button['text_font'] = self.bold_font if active else self.font
        if self.preview:
            self.preview.select(self._preview_index,model)
        self.recover_button.show() if selection.mode == 'practice' else self.recover_button.hide()
        test_case=selection.test_case
        self.test_track_summary.show() if test_case is not None else self.test_track_summary.hide()
        for button in self.road_buttons:
            button.hide() if test_case is not None else button.show()
        self.traffic_button['state']=DGG.DISABLED if test_case is not None else DGG.NORMAL
        if test_case is not None:
            from src.visualization.test_track_catalog import TIER_LABELS
            tier_str = TIER_LABELS[test_case.tier].upper()
            self.test_track_summary['text']=(
                f"BẢN ĐỒ TEST: {tier_str} · {test_case.sequence}     [BẤM ĐỔI BẢN ĐỒ ↓]\n"
                f"Seed map {test_case.geometry_generation_seed}  ·  Kịch bản {test_case.environment_seed-9100}/5  ·  Giao thông: {test_case.traffic_density:.0%}")
            self.test_track_summary['text_pos']=(20,-28)
            self.test_track_summary['text_scale']=13
            self._paint(self.test_track_summary, (.065, .075, .092, 1.), self.BORDER_BRIGHT, left_stripe=self.GOLD, stripe_width=4.0)
            self.traffic_button['text']=f'Giao thông TEST: {test_case.traffic_density:.0%} · Theo kịch bản'
            self.traffic_button['text_fg']=self.INK
            self.status['text']=f'TEST · {test_case.sequence} · Kịch bản {test_case.environment_seed-9100}/5'
        elif vietnam:
            self.status['text']='VIỆT NAM  ·  Võ Văn Kiệt / Sài Gòn\n10 km  ·  Hành trình 60  /  Tối đa 80 km/h'
            self.status['text_fg']=self.GOLD

    def _build_phone_panel(self):
        self.phone_panel=self.root.attachNewNode('phone_pairing')
        self.phone_panel.setBin('fixed',700)
        g=self.phone_panel
        self.frame(g,-2000,-2000,5280,4800,(.005,.012,.018,.86),radius=0)
        self.frame(g,365,95,550,610,self.PANEL,self.BORDER_BRIGHT,radius=16)
        self.label(g,'LÁI XE BẰNG ĐIỆN THOẠI',640,140,22,self.ACCENT,align=TextNode.ACenter,bold=True)
        self.label(g,'Cùng mạng Wi-Fi → Quét mã QR → Kết nối tay lái ảo',640,172,13,self.MUTED,align=TextNode.ACenter)
        self.frame(g,490,205,300,300,(1,1,1,1),radius=12)
        self.phone_qr=DirectFrame(parent=g,frameColor=(1,1,1,1),frameSize=(-138,138,-138,138),
                                  pos=(640,0,-355),relief=None)
        self.phone_address=self.label(g,'',640,535,15,align=TextNode.ACenter,wrap=470,bold=True)
        self.phone_status=self.label(g,'Đang mở máy chủ…',640,568,14,self.MUTED,align=TextNode.ACenter,wrap=470)
        self.button(g,'Đóng cửa sổ',400,625,480,'phone_close',h=46,primary=True,align=TextNode.ACenter)
        g.hide()

    def show_phone(self, host):
        from panda3d.core import Filename
        texture=self.engine.loader.loadTexture(Filename.fromOsSpecific(str((host.output/'phone-qr.png').resolve())))
        self.phone_qr['image']=texture
        self.phone_qr['image_scale']=(138,1,138)
        self.phone_address['text']=f'http://{host.address}:{host.port}'
        self.phone_panel.show()
