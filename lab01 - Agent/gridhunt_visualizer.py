"""
Module: gridhunt_visualizer
===========================
Cung cấp bộ công cụ trực quan hóa đồ họa Pixel-Art sinh động, nguyên bản cho GridHunt
(Single-Agent & Multi-Agent) trực tiếp bên trong Jupyter Notebook.

Tính năng nổi bật:
- Đồ họa Pixel-Art nguyên bản: Nhân vật Monster (quái vật đỏ), Hunter 1 (thám tử vàng),
  Hunter 2 (thám tử xanh dương) và nền gạch tối màu kèm viền lưới pixel chuẩn như game gốc.
- Tự động co giãn theo kích thước bàn cờ (n = 5, n = 15, n = 20, n = 30,...).
- Hỗ trợ cả chế độ 1 Thợ săn (như ảnh gốc) lẫn 2 Thợ săn (Multi-agent Assignment).
- 3 chế độ hiển thị:
  1. play_gridhunt: Trình phát tương tác HTML5/JS (Play/Pause/Tua từng bước/Thanh trượt Scrubber/Chỉnh tốc độ).
  2. record_gridhunt_video: Ghi hình thành video GIF mượt mà lưu vào thư mục `videos/`.
  3. arena_environment_live: Giả lập thời gian thực trực tiếp trong cell khi code đang chạy.
"""

import numpy as np
import os
import io
import json
import base64
from time import sleep
from PIL import Image, ImageDraw, ImageFont
from IPython.display import HTML, display, clear_output

# ==============================================================================
# 1. QUẢN LÝ TÀI NGUYÊN ĐỒ HỌA (SPRITES & TEXTURES)
# ==============================================================================

_ASSET_CACHE = {}

def _get_asset_path(filename):
    """Tìm đường dẫn file asset trong thư mục hiện tại hoặc thư mục assets/."""
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    candidates = [
        os.path.join(current_dir, "assets", filename),
        os.path.join(current_dir, filename),
        os.path.join(os.getcwd(), "assets", filename),
        os.path.join(os.getcwd(), filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _create_procedural_assets():
    """Tạo sprite dự phòng bằng code nếu thiếu file ảnh (không bao giờ để crash)."""
    # 1. Tile 60x60
    tile = Image.new('RGB', (60, 60), (35, 40, 28))
    draw = ImageDraw.Draw(tile)
    # Nhiễu texture giả gạch mossy
    np.random.seed(42)
    noise = np.random.randint(-12, 12, (60, 60, 3))
    arr = np.clip(np.array(tile, dtype=int) + noise, 0, 255).astype(np.uint8)
    tile = Image.fromarray(arr)

    # 2. Monster 60x60 (Pixel Art Demon)
    m_img = Image.new('RGB', (60, 60), (4, 13, 5))
    m_draw = ImageDraw.Draw(m_img)
    # Mặt đỏ
    m_draw.rectangle([14, 16, 46, 46], fill=(185, 55, 30))
    # Sừng
    m_draw.polygon([(14, 16), (10, 8), (20, 14)], fill=(210, 70, 40))
    m_draw.polygon([(46, 16), (50, 8), (40, 14)], fill=(210, 70, 40))
    # Mắt vàng sáng
    m_draw.rectangle([18, 24, 26, 30], fill=(255, 230, 140))
    m_draw.rectangle([34, 24, 42, 30], fill=(255, 230, 140))
    # Miệng răng nhọn
    m_draw.rectangle([20, 36, 40, 42], fill=(20, 20, 20))
    m_draw.rectangle([23, 36, 26, 39], fill=(240, 240, 240))
    m_draw.rectangle([34, 36, 37, 39], fill=(240, 240, 240))

    # 3. Hunter 1 60x60 (Vàng kim - Golden Fedora Detective)
    h1_img = Image.new('RGB', (60, 60), (4, 13, 5))
    h1_draw = ImageDraw.Draw(h1_img)
    gold = (215, 180, 115)
    # Nón Fedora
    h1_draw.rectangle([14, 20, 48, 23], fill=gold)
    h1_draw.rectangle([20, 14, 42, 20], fill=gold)
    # Mặt/Cổ
    h1_draw.rectangle([26, 23, 38, 30], fill=gold)
    # Áo măng tô và tay súng/đèn pin
    h1_draw.rectangle([22, 30, 44, 52], fill=gold)
    h1_draw.rectangle([12, 34, 24, 40], fill=gold)

    # 4. Hunter 2 60x60 (Xanh dương/Tím)
    h2_img = Image.new('RGB', (60, 60), (4, 13, 5))
    h2_draw = ImageDraw.Draw(h2_img)
    cyan = (56, 189, 248)
    h2_draw.rectangle([14, 20, 48, 23], fill=cyan)
    h2_draw.rectangle([20, 14, 42, 20], fill=cyan)
    h2_draw.rectangle([26, 23, 38, 30], fill=cyan)
    h2_draw.rectangle([22, 30, 44, 52], fill=cyan)
    h2_draw.rectangle([12, 34, 24, 40], fill=cyan)

    return {"tile": tile, "monster": m_img, "hunter1": h1_img, "hunter2": h2_img}


def load_assets():
    """Tải bộ sprite và texture (có cache để tốc độ cực nhanh)."""
    global _ASSET_CACHE
    if _ASSET_CACHE:
        return _ASSET_CACHE

    assets = {}
    tile_path = _get_asset_path("tile_60.png")
    m_path = _get_asset_path("monster_60.png")
    h1_path = _get_asset_path("hunter1_60.png")
    h2_path = _get_asset_path("hunter2_60.png")

    if tile_path and m_path and h1_path and h2_path:
        assets["tile"] = Image.open(tile_path).convert("RGB")
        assets["monster"] = Image.open(m_path).convert("RGB")
        assets["hunter1"] = Image.open(h1_path).convert("RGB")
        assets["hunter2"] = Image.open(h2_path).convert("RGB")
    else:
        assets = _create_procedural_assets()

    _ASSET_CACHE = assets
    return assets


# ==============================================================================
# 2. CORE SIMULATION LOGIC
# ==============================================================================

def _run_simulation(hunter_1_func, hunter_2_func=None, monster_func=None, n=20, max_steps=100):
    """
    Mô phỏng 1 ván đấu GridHunt và ghi nhận đầy đủ lịch sử tọa độ và hành động.
    Hỗ trợ cả chế độ:
    - 1 Hunter vs 1 Monster (nếu hunter_2_func là None)
    - 2 Hunters vs 1 Monster (Multi-agent)
    """
    if monster_func is None:
        def monster_func(m_pos, h1_pos, h2_pos):
            actions = ["north", "east", "west", "south", "stay"]
            return np.random.choice(actions, p=[0.125, 0.125, 0.125, 0.125, 0.5])

    # Khởi tạo vị trí ngẫu nhiên không trùng nhau
    monster_pos = np.random.randint(0, n, size=2)
    hunter_1_pos = np.random.randint(0, n, size=2)
    while np.array_equal(hunter_1_pos, monster_pos):
        hunter_1_pos = np.random.randint(0, n, size=2)

    has_h2 = (hunter_2_func is not None)
    if has_h2:
        hunter_2_pos = np.random.randint(0, n, size=2)
        while np.array_equal(hunter_2_pos, monster_pos):
            hunter_2_pos = np.random.randint(0, n, size=2)
    else:
        hunter_2_pos = np.array([-1, -1])

    def move(action, position):
        pos = position.copy()
        if action == 'north': pos[0] -= 1
        elif action == 'south': pos[0] += 1
        elif action == 'west': pos[1] -= 1
        elif action == 'east': pos[1] += 1
        return np.clip(pos, 0, n - 1)

    frames_data = []

    # Frame 0 (Trạng thái ban đầu)
    frames_data.append({
        'step': 0,
        'h1_pos': hunter_1_pos.copy(),
        'h2_pos': hunter_2_pos.copy() if has_h2 else None,
        'm_pos': monster_pos.copy(),
        'h1_act': 'start',
        'h2_act': 'start' if has_h2 else None,
        'm_act': 'start',
        'status': 'playing'
    })

    winner = "M"
    final_step = np.nan

    for step in range(max_steps):
        # 1. Monster đi trước
        m_action = monster_func(monster_pos, hunter_1_pos, hunter_2_pos if has_h2 else hunter_1_pos)
        if m_action != 'stay':
            monster_pos = move(m_action, monster_pos)

        # 2. Hunter 1 đi
        h1_action = hunter_1_func(hunter_1_pos, monster_pos)
        if h1_action == 'teleport':
            hunter_1_pos = np.random.randint(0, n, size=2)
        else:
            hunter_1_pos = move(h1_action, hunter_1_pos)

        # 3. Hunter 2 đi (nếu có)
        h2_action = None
        if has_h2:
            h2_action = hunter_2_func(hunter_2_pos, monster_pos)
            if h2_action == 'teleport':
                hunter_2_pos = np.random.randint(0, n, size=2)
            else:
                hunter_2_pos = move(h2_action, hunter_2_pos)

        # Kiểm tra kết quả sau lượt
        status = 'playing'
        if np.array_equal(hunter_1_pos, monster_pos):
            status = 'h1_win'
            winner, final_step = ("1", step + 1)
        elif has_h2 and np.array_equal(hunter_2_pos, monster_pos):
            status = 'h2_win'
            winner, final_step = ("2", step + 1)
        elif step == max_steps - 1:
            status = 'm_win'
            winner, final_step = ("M", np.nan)

        frames_data.append({
            'step': step + 1,
            'h1_pos': hunter_1_pos.copy(),
            'h2_pos': hunter_2_pos.copy() if has_h2 else None,
            'm_pos': monster_pos.copy(),
            'h1_act': h1_action,
            'h2_act': h2_action,
            'm_act': m_action,
            'status': status
        })

        if status != 'playing':
            break

    return frames_data, (winner, final_step)


# ==============================================================================
# 3. RENDERING ENGINE (PIXEL-ART TILES & SPRITES)
# ==============================================================================

def render_pixel_frame(frame_data, n=5, assets=None, show_trails=True, all_frames=None, frame_idx=0):
    """
    Dựng 1 khung hình Pixel-Art hoàn chỉnh (PIL Image) gồm:
    - Thanh tiêu đề thông tin trực quan.
    - Lưới bàn cờ vân ngói và viền ngăn cách sắc nét.
    - Sprite Quái vật, Thợ săn 1, Thợ săn 2.
    """
    if assets is None:
        assets = load_assets()

    # Kích thước cell theo kích thước lưới n
    if n <= 5:
        cell_px = 60
        border_px = 3
    elif n <= 8:
        cell_px = 48
        border_px = 2
    elif n <= 15:
        cell_px = 32
        border_px = 2
    elif n <= 20:
        cell_px = 24
        border_px = 2
    elif n <= 30:
        cell_px = 18
        border_px = 1
    else:
        cell_px = max(10, 520 // n)
        border_px = 1

    grid_w = n * cell_px + (n + 1) * border_px
    grid_h = n * cell_px + (n + 1) * border_px
    header_h = 56

    # Resize asset phù hợp với cell_px hiện tại (dùng NEAREST để giữ nguyên chất Pixel Art)
    tile_scaled = assets["tile"].resize((cell_px, cell_px), Image.Resampling.NEAREST)
    m_scaled = assets["monster"].resize((cell_px, cell_px), Image.Resampling.NEAREST)
    h1_scaled = assets["hunter1"].resize((cell_px, cell_px), Image.Resampling.NEAREST)
    h2_scaled = assets["hunter2"].resize((cell_px, cell_px), Image.Resampling.NEAREST)

    canvas = Image.new('RGB', (grid_w, grid_h + header_h), (9, 13, 22))
    draw = ImageDraw.Draw(canvas)

    # 1. Vẽ Header Bar
    draw.rectangle([0, 0, grid_w, header_h], fill=(15, 23, 42))
    draw.line([(0, header_h - 1), (grid_w, header_h - 1)], fill=(30, 41, 59), width=1)

    step_num = frame_data['step']
    status = frame_data['status']
    h1_act = frame_data.get('h1_act', '')
    h2_act = frame_data.get('h2_act', '')
    m_act = frame_data.get('m_act', '')

    # Nạp font hệ thống sắc nét (hỗ trợ Unicode tiếng Việt đầy đủ)
    font_main = None
    font_sub = None
    for font_path in [r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(font_path):
            try:
                font_main = ImageFont.truetype(font_path, 13)
                font_sub = ImageFont.truetype(font_path, 11)
                break
            except Exception:
                pass

    if font_main is None:
        title_text = f"STEP {step_num} | H1: {h1_act} | M: {m_act}"
        if h2_act is not None:
            title_text = f"STEP {step_num} | H1: {h1_act} | H2: {h2_act} | M: {m_act}"
        status_sub = "Hunting in progress..."
        if status == 'h1_win': status_sub = f"HUNTER 1 CAUGHT MONSTER! (Step {step_num})"
        elif status == 'h2_win': status_sub = f"HUNTER 2 CAUGHT MONSTER! (Step {step_num})"
        elif status == 'm_win': status_sub = "MONSTER ESCAPED!"
    else:
        title_text = f"BƯỚC {step_num} | H1: {h1_act} | M: {m_act}"
        if h2_act is not None:
            title_text = f"BƯỚC {step_num} | H1: {h1_act} | H2: {h2_act} | M: {m_act}"
        status_sub = "Đang săn lùng..."
        if status == 'h1_win': status_sub = f"HUNTER 1 ĐÃ BẮT ĐƯỢC QUÁI VẬT! (Bước {step_num})"
        elif status == 'h2_win': status_sub = f"HUNTER 2 ĐÃ BẮT ĐƯỢC QUÁI VẬT! (Bước {step_num})"
        elif status == 'm_win': status_sub = "QUÁI VẬT ĐÃ SỐNG SÓT THÀNH CÔNG!"

    status_color = (56, 189, 248) # Xanh cyan mặc định
    if status == 'h1_win':
        status_color = (250, 204, 21) # Vàng rực rỡ
    elif status == 'h2_win':
        status_color = (192, 132, 252) # Tím sáng
    elif status == 'm_win':
        status_color = (239, 68, 68) # Đỏ cảnh báo

    draw.text((12, 10), title_text, fill=(203, 213, 225), font=font_main)
    draw.text((12, 30), status_sub, fill=status_color, font=font_sub)

    # 2. Vẽ Lưới Gạch Nền
    m_pos = tuple(frame_data['m_pos'])
    h1_pos = tuple(frame_data['h1_pos'])
    h2_pos = tuple(frame_data['h2_pos']) if frame_data['h2_pos'] is not None else (-1, -1)

    for r in range(n):
        for c in range(n):
            x = border_px + c * (cell_px + border_px)
            y = header_h + border_px + r * (cell_px + border_px)
            canvas.paste(tile_scaled, (x, y))

    # 3. Vẽ Vệt Di Chuyển (Trails) nếu có lịch sử
    if show_trails and all_frames is not None and frame_idx > 0:
        trail_start = max(0, frame_idx - 10)
        dot_r = max(2, cell_px // 7)
        for idx in range(trail_start, frame_idx):
            f = all_frames[idx]
            # Vệt Hunter 1 (chấm vàng)
            hr1, hc1 = f['h1_pos']
            hx1 = border_px + hc1 * (cell_px + border_px) + cell_px // 2
            hy1 = header_h + border_px + hr1 * (cell_px + border_px) + cell_px // 2
            draw.ellipse([hx1 - dot_r, hy1 - dot_r, hx1 + dot_r, hy1 + dot_r], fill=(234, 179, 8))

            # Vệt Hunter 2 (chấm xanh)
            if f['h2_pos'] is not None and f['h2_pos'][0] >= 0:
                hr2, hc2 = f['h2_pos']
                hx2 = border_px + hc2 * (cell_px + border_px) + cell_px // 2
                hy2 = header_h + border_px + hr2 * (cell_px + border_px) + cell_px // 2
                draw.ellipse([hx2 - dot_r, hy2 - dot_r, hx2 + dot_r, hy2 + dot_r], fill=(168, 85, 247))

    # 4. Vẽ Sprites vào tọa độ
    # Quái vật
    if 0 <= m_pos[0] < n and 0 <= m_pos[1] < n:
        mx = border_px + m_pos[1] * (cell_px + border_px)
        my = header_h + border_px + m_pos[0] * (cell_px + border_px)
        canvas.paste(m_scaled, (mx, my))

    # Hunter 1
    if 0 <= h1_pos[0] < n and 0 <= h1_pos[1] < n:
        h1x = border_px + h1_pos[1] * (cell_px + border_px)
        h1y = header_h + border_px + h1_pos[0] * (cell_px + border_px)
        canvas.paste(h1_scaled, (h1x, h1y))

    # Hunter 2 (nếu có)
    if h2_pos[0] >= 0 and 0 <= h2_pos[0] < n and 0 <= h2_pos[1] < n:
        h2x = border_px + h2_pos[1] * (cell_px + border_px)
        h2y = header_h + border_px + h2_pos[0] * (cell_px + border_px)
        canvas.paste(h2_scaled, (h2x, h2y))

    return canvas


# ==============================================================================
# 4. CHẾ ĐỘ 1: PLAY_GRIDHUNT (INTERACTIVE HTML5 / JS ANIMATION PLAYER)
# ==============================================================================

def play_gridhunt(hunter_1_func, hunter_2_func=None, monster_func=None, n=20, max_steps=100, interval=250):
    """
    Trình phát tương tác HTML5 / JS Player nhúng trực tiếp vào Jupyter Notebook cell!
    Đặc điểm:
    - Đồ họa Pixel-Art nguyên bản, tự động co giãn.
    - Bộ điều khiển đầy đủ: Play, Pause, Tua tới/lùi, Về đầu/cuối, Điều chỉnh tốc độ, Thanh trượt Scrubber.
    - Hoàn toàn độc lập, không cần thư viện ngoài.
    """
    frames_data, result = _run_simulation(hunter_1_func, hunter_2_func, monster_func, n, max_steps)
    assets = load_assets()

    # Tạo mảng ảnh base64 cho từng frame
    base64_frames = []
    for idx, f in enumerate(frames_data):
        img = render_pixel_frame(f, n=n, assets=assets, show_trails=True, all_frames=frames_data, frame_idx=idx)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        encoded = base64.b64encode(buf.getvalue()).decode('ascii')
        base64_frames.append(f"data:image/png;base64,{encoded}")

    unique_id = f"gh_player_{np.random.randint(100000, 999999)}"
    frames_json = json.dumps(base64_frames)
    total_frames = len(base64_frames)

    # Giao diện HTML5 Player hiện đại, phong cách game retro sang trọng
    html_code = f"""
    <div id="{unique_id}_container" style="
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #0b0f19;
        color: #f1f5f9;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 16px;
        max-width: 620px;
        margin: 12px auto;
        box-shadow: 0 12px 30px rgba(0,0,0,0.6);
        text-align: center;
    ">
        <!-- Frame Canvas Image -->
        <div style="background: #05070d; border-radius: 8px; padding: 6px; display: inline-block; border: 1px solid #1e293b;">
            <img id="{unique_id}_img" src="{base64_frames[0]}" style="
                max-width: 100%;
                height: auto;
                display: block;
                border-radius: 4px;
                image-rendering: pixelated;
                image-rendering: crisp-edges;
            " />
        </div>

        <!-- Scrubber Progress Bar -->
        <div style="margin: 14px 10px 8px 10px; display: flex; align-items: center; gap: 10px;">
            <span id="{unique_id}_cur_step" style="font-size: 13px; font-weight: bold; color: #38bdf8; min-width: 60px; text-align: left;">0 / {total_frames - 1}</span>
            <input type="range" id="{unique_id}_slider" min="0" max="{total_frames - 1}" value="0" style="
                flex-grow: 1;
                cursor: pointer;
                accent-color: #38bdf8;
                height: 6px;
            " />
            <span style="font-size: 11px; color: #64748b;">Tổng: {total_frames} frames</span>
        </div>

        <!-- Player Controls Buttons -->
        <div style="display: flex; justify-content: center; align-items: center; gap: 8px; margin-top: 10px; flex-wrap: wrap;">
            <button id="{unique_id}_first" title="Về đầu" style="background:#1e293b; color:#cbd5e1; border:none; padding:6px 12px; border-radius:6px; cursor:pointer; font-weight:bold;">⏮</button>
            <button id="{unique_id}_prev" title="Lùi 1 bước" style="background:#1e293b; color:#cbd5e1; border:none; padding:6px 12px; border-radius:6px; cursor:pointer; font-weight:bold;">◀</button>
            <button id="{unique_id}_play" title="Phát / Dừng" style="background:#0284c7; color:white; border:none; padding:6px 18px; border-radius:6px; cursor:pointer; font-weight:bold; min-width:70px;">▶ Play</button>
            <button id="{unique_id}_next" title="Tiến 1 bước" style="background:#1e293b; color:#cbd5e1; border:none; padding:6px 12px; border-radius:6px; cursor:pointer; font-weight:bold;">▶</button>
            <button id="{unique_id}_last" title="Đến cuối" style="background:#1e293b; color:#cbd5e1; border:none; padding:6px 12px; border-radius:6px; cursor:pointer; font-weight:bold;">⏭</button>

            <!-- Speed selector -->
            <select id="{unique_id}_speed" style="background:#1e293b; color:#94a3b8; border:1px solid #334155; padding:5px 8px; border-radius:6px; cursor:pointer; font-size:12px; margin-left:6px;">
                <option value="500">0.5x</option>
                <option value="250" selected>1.0x</option>
                <option value="120">2.0x</option>
                <option value="50">5.0x</option>
            </select>
        </div>
    </div>

    <script>
    (function() {{
        const frames = {frames_json};
        let currentFrame = 0;
        let isPlaying = false;
        let timer = null;
        let intervalMs = {interval};

        const imgEl = document.getElementById("{unique_id}_img");
        const sliderEl = document.getElementById("{unique_id}_slider");
        const stepEl = document.getElementById("{unique_id}_cur_step");
        const playBtn = document.getElementById("{unique_id}_play");
        const speedSelect = document.getElementById("{unique_id}_speed");

        function showFrame(idx) {{
            if (idx < 0) idx = 0;
            if (idx >= frames.length) idx = frames.length - 1;
            currentFrame = idx;
            imgEl.src = frames[currentFrame];
            sliderEl.value = currentFrame;
            stepEl.innerText = currentFrame + " / " + (frames.length - 1);
        }}

        function play() {{
            if (isPlaying) return;
            isPlaying = true;
            playBtn.innerText = "⏸ Pause";
            playBtn.style.background = "#ea580c";
            
            if (currentFrame >= frames.length - 1) {{
                showFrame(0);
            }}

            timer = setInterval(() => {{
                if (currentFrame < frames.length - 1) {{
                    showFrame(currentFrame + 1);
                }} else {{
                    pause();
                }}
            }}, intervalMs);
        }}

        function pause() {{
            isPlaying = false;
            playBtn.innerText = "▶ Play";
            playBtn.style.background = "#0284c7";
            if (timer) clearInterval(timer);
        }}

        playBtn.onclick = () => isPlaying ? pause() : play();
        document.getElementById("{unique_id}_first").onclick = () => {{ pause(); showFrame(0); }};
        document.getElementById("{unique_id}_last").onclick = () => {{ pause(); showFrame(frames.length - 1); }};
        document.getElementById("{unique_id}_prev").onclick = () => {{ pause(); showFrame(currentFrame - 1); }};
        document.getElementById("{unique_id}_next").onclick = () => {{ pause(); showFrame(currentFrame + 1); }};

        sliderEl.oninput = (e) => {{
            pause();
            showFrame(parseInt(e.target.value));
        }};

        speedSelect.onchange = (e) => {{
            intervalMs = parseInt(e.target.value);
            if (isPlaying) {{
                pause();
                play();
            }}
        }};
    }})();
    </script>
    """
    return HTML(html_code)