"""
Demo script trực quan hóa vạch đích 3D Hologram Cyberpunk cuối bản đồ trên MetaDrive.

Cách chạy:
   python scripts/demo_finish_visual.py

Mặc định:
- Xe được đặt cách vạch đích 25m, nhìn thẳng vào vạch đích Hologram 3D để bạn thấy NGAY LẬP TỨC!
- Xe tự động lái tiến qua vạch đích bằng IDM Policy.

Tùy chọn:
- Lái xe thủ công bằng bàn phím (W/A/S/D):
   python scripts/demo_finish_visual.py --manual

- Chạy từ vạch xuất phát ban đầu của bản đồ (không teleport):
   python scripts/demo_finish_visual.py --from-start

- Đổi nội dung chữ hiển thị trên Hologram:
   python scripts/demo_finish_visual.py --text "VICTORY"

Phím điều khiển khi cửa sổ hiện ra:
- Phím 'C': Đổi góc nhìn camera (Third-person sau xe, Bird-eye từ trên cao, Top-down toàn cảnh)
- Phím 'W/A/S/D' hoặc phím mũi tên: Lái xe (khi bật --manual)
- Phím 'R': Reset lại vị trí xe
- Phím 'Esc': Thoát
"""

import argparse
from pathlib import Path
import sys
import time

# Đảm bảo đường dẫn gốc project có trong sys.path khi chạy trực tiếp
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy
from src.visualization import create_finish_visual


def teleport_near_finish(env: MetaDriveEnv, distance_before_m: float = 25.0) -> None:
    """Đưa xe đến vị trí ngay trước vạch đích để quan sát vạch đích lập tức."""
    try:
        final_lane = env.agent.navigation.final_lane
        target_long = max(0.0, final_lane.length - distance_before_m)
        spawn_pos = final_lane.position(target_long, 0.0)
        spawn_heading = final_lane.heading_theta_at(target_long)
        env.agent.set_position(spawn_pos)
        env.agent.set_heading_theta(spawn_heading)
    except Exception as e:
        print(f"[Cảnh báo] Không thể teleport xe: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo 3D Cyberpunk Hologram Finish Gate in MetaDrive")
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed bản đồ MetaDrive (mặc định: 1)"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="FINISH",
        help="Nội dung chữ hiển thị trên bảng Hologram (mặc định: 'FINISH')"
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        default=False,
        help="Tự lái xe bằng bàn phím (W/A/S/D hoặc mũi tên). Mặc định là xe tự lái (IDM)."
    )
    parser.add_argument(
        "--from-start",
        action="store_true",
        default=False,
        help="Bắt đầu từ vạch xuất phát đầu map thay vì xuất hiện ngay trước vạch đích."
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  KHỞI ĐỘNG DEMO: CỔNG ĐÍCH 3D HOLOGRAPHIC CYBERPUNK (METADRIVE)")
    print(f"  - Bản đồ Seed: {args.seed}")
    print(f"  - Nội dung Hologram: '{args.text}'")
    print(f"  - Chế độ điều khiển: {'Thủ công (W/A/S/D)' if args.manual else 'Tự lái (IDM Policy)'}")
    print(f"  - Vị trí xuất phát: {'Đầu bản đồ' if args.from_start else 'Cách vạch đích 25m (Thấy ngay)'}")
    print("=" * 70)
    print("\n[MẸO ĐIỀU KHIỂN]")
    print("  * Nhấn phím 'C': Chuyển đổi góc nhìn Camera (sau xe / trên cao / toàn cảnh)")
    if args.manual:
        print("  * Phím 'W'/'S': Tăng tốc / Phanh & Lùi")
        print("  * Phím 'A'/'D': Đánh lái Trái / Phải")
    print("  * Nhấn 'Esc': Thoát mô phỏng\n")

    env_config = dict(
        use_render=True,
        manual_control=args.manual,
        traffic_density=0.04,
        num_scenarios=1,
        start_seed=args.seed,
        window_size=(1280, 720),
    )
    if not args.manual:
        env_config["agent_policy"] = IDMPolicy

    env = MetaDriveEnv(env_config)

    try:
        obs, info = env.reset(seed=args.seed)

        if not args.from_start:
            teleport_near_finish(env, distance_before_m=25.0)

        finish_visual = create_finish_visual(env, text=args.text)

        print("[OK] Đã dựng Cổng vạch đích 3D Hologram bao phủ toàn bộ các làn đường!")
        print("Đang hiển thị... Bạn có thể nhấn 'C' để đổi góc camera nhìn toàn cảnh Hologram.\n")

        step = 0
        while step < 2000:
            if args.manual:
                obs, reward, terminated, truncated, step_info = env.step(env.agent.action)
            else:
                obs, reward, terminated, truncated, step_info = env.step([0.0, 0.0])

            if finish_visual is not None:
                finish_visual.update()

            step += 1
            if terminated or truncated:
                print(f"[Hoàn thành] Xe đã đến đích tại step {step}! (arrive_dest: {step_info.get('arrive_dest', False)})")
                time.sleep(2.0)
                obs, info = env.reset(seed=args.seed)
                if not args.from_start:
                    teleport_near_finish(env, distance_before_m=25.0)
                if finish_visual is not None:
                    finish_visual.destroy()
                finish_visual = create_finish_visual(env, text=args.text)
                step = 0

    except KeyboardInterrupt:
        print("\nĐã nhận tín hiệu dừng.")
    finally:
        if "finish_visual" in locals() and finish_visual is not None:
            finish_visual.destroy()
        env.close()
        print("[OK] Đã dọn dẹp sạch tài nguyên và đóng cửa sổ.")


if __name__ == "__main__":
    main()

