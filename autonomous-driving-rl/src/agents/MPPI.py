"""MPPI kết hợp Kinematic Bicycle Model, chạy trên CPU bằng NumPy.

Các class chính:
    1. MPPIConfig   : các tham số và ý nghĩa khi tăng/giảm.
    2. BicycleModel : dự đoán xe đi đâu khi thực hiện một chuỗi điều khiển.
    3. MPPIScene    : đường tham chiếu, xe xung quanh và vật cản LiDAR.
    4. cost_fn      : chấm điểm quỹ đạo; cost càng thấp càng tốt.
    5. MPPI         : lấy mẫu -> dự đoán -> tính cost -> cập nhật điều khiển.
    6. MPPIAgent    : nối thuật toán với giao diện của project.

KÝ HIỆU DÙNG TRONG CÔNG THỨC
    K       : số chuỗi điều khiển được thử trong một lần act().
    H       : số bước dự đoán của mỗi chuỗi.
    dt      : thời gian một bước điều khiển, lấy từ episode context.
    x0      : trạng thái ban đầu [x, y, theta, v], shape (4,).
    U       : chuỗi điều khiển danh định [steering, throttle_brake], (H, 2).
    eps     : nhiễu thêm vào U để khám phá các cách điều khiển khác.
    traj    : trạng thái dự đoán của K chuỗi, shape (K, H, 4).
    lambda  : temperature, quyết định mức tập trung trọng số vào mẫu tốt.

CHỖ CHỈNH THAM SỐ KHI CHẠY CLI / WORKBENCH
    Sửa configs/experiments/mppi_v1.json, sau đó khởi tạo/chạy lại agent.
    MPPIAgent() đọc JSON; các giá trị MPPIConfig bên dưới là mặc định của class.
    Ví dụ: target_speed_m_s = 10 nghĩa là mục tiêu 36 km/h (km/h = m/s * 3.6).

Biến thể này theo khung mẫu: clip action, chuẩn hóa cost và cập nhật bằng trọng
số softmax. Nó chưa có correction term của MPPI importance sampling đầy đủ.
Agent chỉ dùng public input; các rollout không gọi env.step().
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from src.platform.agent import (
    AgentDecision,
    AgentDescriptor,
    AgentPublicEpisodeContext,
)
from src.platform.agent_input import AgentInputV1
from src.platform.protocol import canonical_json_sha256


CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs/experiments/mppi_v1.json"
)


# =============================================================================
# 1. THAM SỐ: đọc nhóm A và B trước nếu muốn chỉnh hành vi của MPPI
# =============================================================================

@dataclass(frozen=True)
class MPPIConfig:
    """Cấu hình tĩnh; dùng MPPIConfig(...) để ghi đè khi tạo agent bằng Python."""

    # A. Khả năng tìm kiếm của MPPI
    # Tầm nhìn = H * dt. Ví dụ H=20, dt=0.1 s -> nhìn trước 2 giây.
    horizon: int = 20                # H: tăng -> nhìn xa hơn, tính toán nhiều hơn.
    num_samples: int = 512           # K: tăng -> thử nhiều phương án, tốn thời gian hơn.
    temperature: float = 0.5         # lambda: nhỏ -> tập trung vào ít mẫu có cost thấp.
    sigma_steering: float = 0.3      # Độ lệch chuẩn nhiễu lái; lớn -> thử góc lái khác U nhiều hơn.
    sigma_throttle: float = 0.5      # Độ lệch chuẩn nhiễu ga/phanh, cùng miền action [-1, 1].
    noise_correlation: float = 0.8   # rho: gần 1 -> nhiễu giữ cùng xu hướng lâu hơn; 0 -> độc lập.
    integration_dt_s: float = 0.02   # Bước tích phân tối đa; nhỏ -> mịn hơn, tốn tính hơn.

    # B. Tốc độ mục tiêu và hiệu chỉnh mô hình xe
    # max_speed_kmh còn dùng GIẢI MÃ observation; cần khớp giới hạn của xe MetaDrive.
    max_speed_kmh: float = 80.0
    target_speed_m_s: float = 8.0    # Tốc độ mong muốn (m/s); có thể giảm khi cua/gặp vật cản.
    wheelbase_m: float = 2.46894     # L: khoảng cách hai trục, KHÔNG phải chiều dài thân xe.
    max_steering_deg: float = 40.0   # delta_max: góc bánh trước khi steering = +/-1 (độ).
    steering_sign: float = 1.0       # +1: action lái dương -> quay trái trong hệ ego.
    acceleration_m_s2: float = 4.0   # a_max: gia tốc dự đoán ở ga tối đa, không chỉnh động cơ.
    braking_m_s2: float = 8.0        # b_max > 0: độ lớn gia tốc phanh giả định.
    drag_per_s: float = 0.05         # d: hệ số giảm tốc theo vận tốc, a_drag = -d * v.

    # C. Kích thước xe và khoảng đệm dùng trong hình học tránh va chạm
    vehicle_length_m: float = 4.515
    vehicle_width_m: float = 1.852
    lane_margin_m: float = 0.1      # Thu hẹp hành lang được dùng: width/2 - margin.
    obstacle_margin_m: float = 0.5  # Nới vùng vật cản để chừa khoảng cách; lớn -> dè chừng hơn.
    lidar_range_m: float = 50.0     # Phải khớp tầm LiDAR dùng để chuẩn hóa observation.
    lateral_accel_max_m_s2: float = 3.0  # Tốc độ khi cua: v_curve = sqrt(a_lat / |kappa|).

    # D. Trọng số cost: tăng trọng số -> ưu tiên tiêu chí đó hơn các tiêu chí khác
    lateral_weight: float = 2.0         # w_lat: bám đường tham chiếu.
    heading_weight: float = 1.0         # w_heading: hướng xe theo tiếp tuyến đường.
    speed_weight: float = 0.5           # w_v: bám tốc độ mục tiêu.
    steering_weight: float = 0.1        # w_steer: hạn chế đánh lái lớn.
    control_change_weight: float = 1.0  # w_du: hạn chế đổi lái/ga/phanh đột ngột.
    progress_weight: float = 1.0        # w_progress: thưởng tiến về phía trước dọc đường.
    outside_penalty: float = 50.0      # Phạt thân xe ra khỏi hành lang / vượt phần đường quan sát.
    collision_penalty: float = 500.0    # Phạt va chạm dự đoán; đây là cost mềm.

    def __post_init__(self):
        """Chặn tham số sai miền trước khi bắt đầu tính rollout."""
        for name in ("horizon", "num_samples"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")

        if self.num_samples < 3:
            raise ValueError(
                "num_samples must allow nominal, braking and random proposals"
            )

        for name, value in asdict(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Invalid MPPI setting: {name}")

            if name != "steering_sign" and value < 0:
                raise ValueError(f"Negative MPPI setting: {name}")

        for name in (
            "temperature",
            "sigma_steering",
            "sigma_throttle",
            "integration_dt_s",
            "max_speed_kmh",
            "target_speed_m_s",
            "wheelbase_m",
            "braking_m_s2",
            "vehicle_length_m",
            "vehicle_width_m",
            "lidar_range_m",
            "lateral_accel_max_m_s2",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

        if not 0 <= self.noise_correlation < 1:
            raise ValueError("noise_correlation must be in [0, 1)")

        if (
            self.steering_sign not in (-1.0, 1.0)
            or not 0 < self.max_steering_deg < 90
        ):
            raise ValueError("Invalid steering calibration")

        if self.target_speed_m_s > self.max_speed_kmh / 3.6:
            raise ValueError("Target speed exceeds model maximum speed")


def load_mppi_config() -> MPPIConfig:
    """Đọc parameters từ JSON; các giá trị này ghi đè mặc định của class."""
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    if payload["version"] != "mppi_planner_v1":
        raise ValueError("Unsupported MPPI configuration")

    return MPPIConfig(**payload["parameters"])


# =============================================================================
# 2. KINEMATIC BICYCLE MODEL: dự đoán chuyển động, chưa chọn action
# =============================================================================

class BicycleModel:
    """Xấp xỉ xe không trượt bánh bằng một bánh trước và một bánh sau.

    Trạng thái: [x, y, theta, v], đơn vị [m, m, rad, m/s].
    Action:     [s, q] trong [-1, 1]; s = lái, q = ga (+) / phanh (-).

    Công thức liên tục:
        delta     = steering_sign * delta_max * s
        dx/dt     = v * cos(theta)
        dy/dt     = v * sin(theta)
        dtheta/dt = (v / L) * tan(delta)
        dv/dt     = a_drive(q) - drag * v

    Đây là dạng tham chiếu trục sau, dùng để xấp xỉ vị trí tâm xe MetaDrive.
    Chưa mô tả lực lốp, trượt bánh hay độ trễ vô lăng.
    """

    def __init__(self, config: Optional[MPPIConfig] = None, dt: float = 0.1):
        self.config = MPPIConfig() if config is None else config

        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")

        self.dt = float(dt)

    def rollout(self, x0, U):
        """Dự đoán K quỹ đạo song song, mỗi quỹ đạo H bước.

        Input:
            x0: (4,)      trạng thái chung ở đầu lần dự đoán.
            U:  (K, H, 2) các chuỗi lái và ga/phanh cần thử.
        Output:
            traj: (K, H, 4), trạng thái SAU từng bước điều khiển.
        """
        U = np.asarray(U, dtype=float)
        x0 = np.asarray(x0, dtype=float)

        if U.ndim != 3 or U.shape[2] != 2 or min(U.shape[:2]) < 1:
            raise ValueError("U must have shape (K,H,2) with K,H positive")

        if x0.shape != (4,) or not np.all(np.isfinite(x0)) or x0[3] < 0:
            raise ValueError("x0 must be finite [x,y,yaw,nonnegative speed]")

        if not np.all(np.isfinite(U)) or np.any(np.abs(U) > 1):
            raise ValueError("U must contain finite controls in [-1,1]")

        cfg = self.config
        K, H, _ = U.shape
        state = np.broadcast_to(x0, (K, 4)).copy()
        traj = np.empty((K, H, 4))

        # Một action được giữ trong dt, nhưng mô phỏng bằng count bước nhỏ h.
        # Ví dụ dt=0.1, integration_dt_s=0.02 -> count=5, h=0.02 s.
        count = max(1, math.ceil(self.dt / cfg.integration_dt_s))
        h = self.dt / count

        for t in range(H):
            # (1) Lái chuẩn hóa -> góc bánh trước -> độ cong quỹ đạo.
            # delta = sign * delta_max * s; kappa = tan(delta) / L.
            steering_angle = (
                cfg.steering_sign
                * math.radians(cfg.max_steering_deg)
                * U[:, t, 0]
            )
            curvature = np.tan(steering_angle) / cfg.wheelbase_m

            # (2) a_drive = a_max*q nếu q>=0, ngược lại b_max*q (âm).
            throttle = U[:, t, 1]
            drive = np.where(
                throttle >= 0,
                cfg.acceleration_m_s2 * throttle,
                cfg.braking_m_s2 * throttle,
            )

            for _ in range(count):
                speed = state[:, 3].copy()
                acc = drive - cfg.drag_per_s * speed

                # (3) Nếu phanh dừng giữa bước h, chỉ tính quãng đường tới lúc dừng.
                # t_move = min(h, v/(-a)) khi a<0; còn lại t_move=h.
                active_h = np.minimum(
                    h,
                    np.divide(
                        speed,
                        -acc,
                        out=np.full(K, h),
                        where=acc < 0,
                    ),
                )
                distance = np.maximum(
                    0.0,
                    speed * active_h + 0.5 * acc * active_h ** 2,
                )

                # (4) Đi theo cung tròn với kappa cố định trong bước nhỏ.
                # dtheta = distance * kappa.
                # Độ dài dây cung = distance * sin(dtheta/2)/(dtheta/2).
                # np.sinc(z)=sin(pi*z)/(pi*z), nên vẫn đúng khi đi thẳng.
                turn = distance * curvature
                chord_distance = distance * np.sinc(turn / (2 * np.pi))
                mid_yaw = state[:, 2] + turn / 2

                state[:, 0] += chord_distance * np.cos(mid_yaw)
                state[:, 1] += chord_distance * np.sin(mid_yaw)
                state[:, 2] += turn

                # (5) v_next = clip(v + a*h, 0, v_max): không cho xe chạy lùi.
                state[:, 3] = np.clip(
                    speed + acc * h,
                    0.0,
                    cfg.max_speed_kmh / 3.6,
                )

            traj[:, t] = state

        return traj


# =============================================================================
# 3. CẢNH CỤC BỘ: đường tham chiếu và vật cản tại thời điểm gọi act()
# =============================================================================

class MPPIScene:
    """Giữ cố định hệ ego ở đầu act(): gốc tại xe, +x trước, +y trái.

    Waypoint/actor đã ở hệ này. Trong khi rollout, ego và actor di chuyển trong
    cùng hệ cố định; không xoay lại hệ tọa độ sau mỗi trạng thái dự đoán.
    """

    def __init__(
        self,
        agent_input: AgentInputV1,
        speed_m_s: float,
        config: MPPIConfig,
    ):
        self.config = config
        task = agent_input.task_context
        traffic = agent_input.traffic_context

        # 3.1. Đường tham chiếu: chỉ lấy waypoint hợp lệ, bỏ điểm liên tiếp trùng.
        route = task.waypoints_array[task.validity_mask].astype(float)
        if not len(route):
            raise ValueError("No public route")

        keep = np.r_[
            True,
            np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1) > 1e-6,
        ]
        route = route[keep]

        # Kéo dài lùi theo heading đường để đo độ lệch làn hiện tại của xe.
        # Nếu nối đường qua gốc ego [0,0], độ lệch làn tại gốc sẽ bị xóa sai.
        first_dir = np.array([np.cos(route[0, 2]), np.sin(route[0, 2])])
        last_dir = np.array([np.cos(route[-1, 2]), np.sin(route[-1, 2])])
        extension = config.vehicle_length_m + np.linalg.norm(route[0, :2])
        centers = np.vstack(
            (
                route[0, :2] - extension * first_dir,
                route[:, :2],
                route[-1, :2] + config.vehicle_length_m * last_dir,
            )
        )

        self.starts = centers[:-1]                   # Điểm đầu A của mỗi đoạn.
        self.vectors = np.diff(centers, axis=0)       # Vector B-A của mỗi đoạn.
        self.lengths = np.linalg.norm(self.vectors, axis=1)
        self.cumulative = np.r_[0.0, np.cumsum(self.lengths)]
        self.route_end = float(self.cumulative[-2])  # Cuối đường quan sát, trước đoạn kéo dài.
        self.root_progress = float(self.project([[0.0, 0.0]])[0][0])
        self.half_lane = max(
            0.0,
            float(task.current_lane_width) / 2 - config.lane_margin_m,
        )

        # 3.2. Xe xung quanh: actor = [x,y,vx_rel,vy_rel,heading,length,width].
        # v_actor = v_relative + [v_ego, 0], chỉ cộng một lần ở đầu act().
        # Dự đoán sau đó: p_actor(t) = p_actor(0) + v_actor * t.
        self.actors = traffic.actors_array[traffic.validity_mask].astype(float)
        self.velocities = self.actors[:, 2:4].copy()
        self.velocities[:, 0] += speed_m_s
        self.actor_cos = np.cos(self.actors[:, 4])
        self.actor_sin = np.sin(self.actors[:, 4])

        # 3.3. LiDAR chuẩn hóa -> các điểm vật cản trong hệ ego.
        # alpha_i = 2*pi*i/N; p_i = range*ray_i*[cos(alpha_i), sin(alpha_i)].
        # ray=1 nghĩa là KHÔNG có hit, không phải vật cản ở cuối tầm đo.
        rays = agent_input.core_observation.lidar_points
        valid = rays < 1.0 - 1e-6
        angles = np.arange(len(rays)) * (2 * np.pi / len(rays))
        hits = (
            config.lidar_range_m
            * rays[valid, None]
            * np.column_stack((np.cos(angles[valid]), np.sin(angles[valid])))
        )

        # Bỏ hit nằm trên actor đã theo dõi; tránh giữ thêm một "xe tĩnh" ở vị trí cũ.
        explained = np.zeros(len(hits), dtype=bool)
        for actor, c, s in zip(self.actors, self.actor_cos, self.actor_sin):
            delta = hits - actor[:2]
            explained |= (
                (np.abs(delta[:, 0] * c + delta[:, 1] * s) <= actor[5] / 2 + 0.5)
                & (np.abs(-delta[:, 0] * s + delta[:, 1] * c) <= actor[6] / 2 + 0.5)
            )

        self.static_hits = hits[~explained]

    def project(self, points):
        """Chiếu điểm P lên đoạn AB gần nhất của đường tham chiếu.

        r = clip(dot(P-A, B-A) / ||B-A||^2, 0, 1)
        Q = A + r*(B-A)                       : điểm gần nhất trên đoạn.
        d = ||P-Q||                          : khoảng cách tới đường.
        progress = cumulative + r*length     : vị trí dọc đường.

        Trả về progress, distance, heading tiếp tuyến, mỗi mảng shape (N,).
        """
        points = np.asarray(points, dtype=float).reshape((-1, 2))
        best = np.full(len(points), np.inf)
        progress = np.zeros(len(points))
        heading = np.zeros(len(points))

        # Tính song song trên N điểm, lặp qua <=21 đoạn để giảm mảng tạm lớn.
        for start, vector, length, cumulative in zip(
            self.starts,
            self.vectors,
            self.lengths,
            self.cumulative,
        ):
            dx = points[:, 0] - start[0]
            dy = points[:, 1] - start[1]
            fraction = np.clip(
                (dx * vector[0] + dy * vector[1]) / length ** 2,
                0.0,
                1.0,
            )
            squared = (
                (dx - fraction * vector[0]) ** 2
                + (dy - fraction * vector[1]) ** 2
            )

            closer = squared < best
            best[closer] = squared[closer]
            progress[closer] = cumulative + fraction[closer] * length
            heading[closer] = math.atan2(vector[1], vector[0])

        return progress, np.sqrt(best), heading

    def footprint_flags(self, states, times):
        """Kiểm tra thân xe tại từng trạng thái, không chỉ kiểm tra tâm xe.

        states: (N,4); times: (N,) thời gian dự đoán tính từ đầu act().
        Trả về collision và outside: hai mảng bool shape (N,).
        Đây là kiểm tra tại các thời điểm lấy mẫu, không phải va chạm liên tục.
        """
        cfg = self.config
        positions = states[:, :2]
        yaw = states[:, 2]
        c = np.cos(yaw)
        s = np.sin(yaw)
        half_l = cfg.vehicle_length_m / 2
        half_w = cfg.vehicle_width_m / 2

        # (1) Bốn góc thân xe = tâm +/- vector nửa chiều dài +/- nửa chiều rộng.
        along = np.column_stack((c, s)) * half_l
        across = np.column_stack((-s, c)) * half_w
        corners = np.stack(
            [
                positions + a * along + b * across
                for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1))
            ],
            axis=1,
        )
        corner_distances = self.project(corners.reshape((-1, 2)))[1]
        outside = np.any(
            corner_distances.reshape((-1, 4)) > self.half_lane,
            axis=1,
        )

        # (2) Actor chuyển động: kiểm tra hai hình chữ nhật có hướng (SAT).
        # Trên MỖI trục: |khoảng cách tâm chiếu| <= tổng hai bán kính chiếu.
        # Chỉ va chạm khi chồng nhau trên cả 4 trục: dọc/ngang ego và actor.
        collision = np.zeros(len(states), dtype=bool)
        for actor, velocity, actor_c, actor_s in zip(
            self.actors,
            self.velocities,
            self.actor_cos,
            self.actor_sin,
        ):
            delta = actor[:2] + times[:, None] * velocity - positions
            dx = delta[:, 0]
            dy = delta[:, 1]
            relative_c = np.abs(c * actor_c + s * actor_s)
            relative_s = np.abs(s * actor_c - c * actor_s)
            actor_half_l = actor[5] / 2 + cfg.obstacle_margin_m
            actor_half_w = actor[6] / 2 + cfg.obstacle_margin_m

            overlap_ego_long = (
                np.abs(dx * c + dy * s)
                <= half_l + actor_half_l * relative_c + actor_half_w * relative_s
            )
            overlap_ego_lat = (
                np.abs(-dx * s + dy * c)
                <= half_w + actor_half_l * relative_s + actor_half_w * relative_c
            )
            overlap_actor_long = (
                np.abs(dx * actor_c + dy * actor_s)
                <= actor_half_l + half_l * relative_c + half_w * relative_s
            )
            overlap_actor_lat = (
                np.abs(-dx * actor_s + dy * actor_c)
                <= actor_half_w + half_l * relative_s + half_w * relative_c
            )
            collision |= (
                overlap_ego_long
                & overlap_ego_lat
                & overlap_actor_long
                & overlap_actor_lat
            )

        # (3) Vật cản tĩnh: lọc điểm xa toàn bộ quỹ đạo để giảm phép tính.
        # radius bao trọn thân xe đã cộng khoảng đệm, kể cả khi xe xoay.
        radius = math.hypot(
            half_l + cfg.obstacle_margin_m,
            half_w + cfg.obstacle_margin_m,
        )
        near = np.all(
            (self.static_hits >= positions.min(axis=0) - radius)
            & (self.static_hits <= positions.max(axis=0) + radius),
            axis=1,
        )
        hits = self.static_hits[near]

        # Đổi điểm vật cản về hệ thân xe: [local_x,local_y] = R(-yaw)*(hit-pos).
        # Điểm trong hộp nếu |local_x|<=half_l+margin và |local_y|<=half_w+margin.
        # Xử lý từng nhóm 32 điểm để hạn chế bộ nhớ tạm.
        for offset in range(0, len(hits), 32):
            delta = hits[None, offset:offset + 32] - positions[:, None]
            local_x = delta[:, :, 0] * c[:, None] + delta[:, :, 1] * s[:, None]
            local_y = -delta[:, :, 0] * s[:, None] + delta[:, :, 1] * c[:, None]
            collision |= np.any(
                (np.abs(local_x) <= half_l + cfg.obstacle_margin_m)
                & (np.abs(local_y) <= half_w + cfg.obstacle_margin_m),
                axis=1,
            )

        return collision, outside


# =============================================================================
# 4. HÀM COST: quỹ đạo nào có tổng chi phí thấp hơn thì được ưu tiên hơn
# =============================================================================

def cost_fn(traj, U, scene: MPPIScene, last_action, dt):
    """Chấm K quỹ đạo; trả (cost shape (K,), có va chạm shape (K,)).

    Công thức cho mẫu k:
        S_k = sum_t [w_lat*d^2 + w_heading*e_heading^2 + w_v*(v-v_target)^2
                     + w_steer*s^2 + w_du*||u_t-u_prev||^2
                     + P_out*I_out + P_col*I_col + P_out*overrun^2]
              - w_progress*(progress_end - progress_start)

    d       : khoảng cách tâm xe tới đường; e_heading: sai lệch góc trong [-pi,pi).
    I_out   : 1 nếu có góc thân xe ngoài hành lang; I_col: 1 nếu dự đoán va chạm.
    overrun : phần vượt cuối đường quan sát. Tiến độ cuối được chặn tại cuối đường.
    """
    cfg = scene.config
    K, H, _ = traj.shape
    flat = traj.reshape((-1, 4))

    # 4.1. Sai lệch vị trí và hướng so với đoạn đường gần nhất.
    progress, lateral, heading = scene.project(flat[:, :2])
    progress = progress.reshape((K, H))
    lateral = lateral.reshape((K, H))
    heading = heading.reshape((K, H))
    heading_error = (traj[:, :, 2] - heading + np.pi) % (2 * np.pi) - np.pi

    # 4.2. Tốc độ mục tiêu: giảm khi cua và gần cuối phần đường được quan sát.
    # kappa ~= |d_heading / d_progress|; v_curve = sqrt(a_lat_max / |kappa|).
    curvature = (
        np.abs(np.diff(np.unwrap(heading, axis=1), axis=1))
        / np.maximum(np.abs(np.diff(progress, axis=1)), 0.1)
    )
    if H > 1:
        curvature = np.pad(curvature, ((0, 0), (0, 1)), mode="edge")
    else:
        curvature = np.zeros((K, H))

    target = np.minimum(
        cfg.target_speed_m_s,
        np.sqrt(cfg.lateral_accel_max_m_s2 / np.maximum(curvature, 1e-6)),
    )

    # Từ v^2 = 2*b*distance: giới hạn tốc độ để có thể phanh trước cuối snapshot.
    # Cuối snapshot là giới hạn thông tin, không phải cờ thành công của episode.
    remaining = np.maximum(0.0, scene.route_end - progress)
    target = np.minimum(target, np.sqrt(2 * cfg.braking_m_s2 * remaining))

    # 4.3. Độ mượt: delta_u[t] = u[t]-u[t-1]. Với t=0, dùng action thực trước đó.
    changes = np.diff(
        U,
        axis=1,
        prepend=np.broadcast_to(last_action, (K, 1, 2)),
    )
    times = np.tile(np.arange(1, H + 1) * dt, K)
    collision, outside = scene.footprint_flags(flat, times)
    collision = collision.reshape((K, H))
    outside = outside.reshape((K, H))

    # 4.4. Cộng cost tại mỗi bước, rồi cộng dọc horizon để có S_k.
    step_cost = (
        cfg.lateral_weight * lateral ** 2
        + cfg.heading_weight * heading_error ** 2
        + cfg.speed_weight * (traj[:, :, 3] - target) ** 2
        + cfg.steering_weight * U[:, :, 0] ** 2
        + cfg.control_change_weight * np.sum(changes ** 2, axis=2)
        + cfg.outside_penalty * outside
        + cfg.collision_penalty * collision
        + cfg.outside_penalty * np.maximum(0.0, progress - scene.route_end) ** 2
    )
    total_cost = step_cost.sum(axis=1)

    # Tiến được xa hơn -> giảm cost; không thưởng tiến độ ngoài đường quan sát.
    total_cost -= cfg.progress_weight * (
        np.minimum(progress[:, -1], scene.route_end) - scene.root_progress
    )

    return total_cost, collision.any(axis=1)


# =============================================================================
# 5. MPPI: lấy mẫu -> rollout -> cost -> trọng số -> action đầu tiên
# =============================================================================

class MPPI:
    """Tối ưu chuỗi điều khiển, dùng chuỗi lần trước để khởi tạo lần tiếp theo.

    Khi đọc thuật toán, tập trung vào act(): các bước 5.1 đến 5.7.
    Mỗi lần chỉ gửi một action ra ngoài; H-1 action còn lại là kế hoạch dự đoán.
    """

    def __init__(self, model: BicycleModel, seed: Optional[int] = 0):
        self.m = model
        self.config = model.config
        self.rng = np.random.default_rng(seed)
        self.U = np.zeros((self.config.horizon, 2))
        self.last_action = np.zeros(2)
        self.diagnostics = {}

    def _validated_plan(self, x0, plan, scene):
        """Kiểm tra lại va chạm ở bước thời gian mịn hơn nhịp điều khiển.

        Trung bình hai cách tránh vật cản có thể tạo ra cách đi vào giữa vật cản.
        Vì vậy phải rollout lại chuỗi vừa trộn, không dùng lại kết quả từng mẫu.
        """
        count = max(1, math.ceil(self.m.dt / self.config.integration_dt_s))
        fine = BicycleModel(self.config, self.m.dt / count)
        fine_controls = np.repeat(plan[None], count, axis=1)
        trajectory = fine.rollout(x0, fine_controls)[0]
        times = np.arange(1, len(trajectory) + 1) * fine.dt
        collision, _ = scene.footprint_flags(trajectory, times)

        return not np.any(collision)

    def act(self, x0, scene: MPPIScene):
        """Từ trạng thái hiện tại, tối ưu và trả [steering, throttle_brake]."""
        cfg = self.config

        # 5.1. Sinh nhiễu chuẩn z~N(0,1), shape (K,H,2).
        # Tương quan thời gian: z_t = rho*z_(t-1) + sqrt(1-rho^2)*noise_t.
        # Nhờ đó một mẫu có thể giữ xu hướng đánh lái qua nhiều bước.
        eps = self.rng.normal(size=(cfg.num_samples, cfg.horizon, 2))
        rho = cfg.noise_correlation
        for t in range(1, cfg.horizon):
            eps[:, t] = (
                rho * eps[:, t - 1]
                + math.sqrt(1 - rho ** 2) * eps[:, t]
            )
        eps *= [cfg.sigma_steering, cfg.sigma_throttle]

        # 5.2. Tạo chuỗi ứng viên: U_k = clip(U + eps_k, -1, 1).
        # Giữ riêng một mẫu danh định và một mẫu phanh thẳng để luôn thử chúng.
        proposals = np.clip(self.U[None] + eps, -1.0, 1.0)
        proposals[0] = self.U
        proposals[1, :, 0] = 0.0
        proposals[1, :, 1] = -1.0

        # Nhiễu HIỆU DỤNG sau clip, tránh cộng nhầm phần vượt giới hạn action.
        eps = proposals - self.U[None]

        # 5.3. Dự đoán K quỹ đạo bằng mô hình xe, rồi lấy K giá trị cost S_k.
        traj = self.m.rollout(x0, proposals)
        costs, collision = cost_fn(
            traj,
            proposals,
            scene,
            self.last_action,
            self.m.dt,
        )

        # 5.4. Chuẩn hóa cost, rồi tính trọng số softmax.
        # S_norm[k] = (S[k]-min(S)) / (std(S)+1e-8)
        # w[k] = exp(-S_norm[k]/lambda) / sum_j exp(-S_norm[j]/lambda)
        # Mẫu cost thấp nhận trọng số lớn; tổng mọi trọng số bằng 1.
        normalized = (costs - costs.min()) / (costs.std() + 1e-8)
        weights = np.exp(-normalized / cfg.temperature)
        weights /= weights.sum()

        # 5.5. Cập nhật CHUỖI ĐIỀU KHIỂN (không trung bình tọa độ quỹ đạo).
        # U_new[t] = clip(U[t] + sum_k w[k]*eps[k,t], -1, 1).
        weighted_noise = np.sum(weights[:, None, None] * eps, axis=0)
        plan = np.clip(self.U + weighted_noise, -1.0, 1.0)

        # 5.6. Nếu chuỗi trộn va chạm, thử mẫu không va chạm có cost thấp nhất.
        # Mẫu đó cũng phải qua kiểm tra mịn; nếu thất bại thì phanh thẳng.
        # Fallback là phản ứng của mô hình, không phải bảo đảm an toàn.
        status = "ACTION_SELECTED"
        if not self._validated_plan(x0, plan, scene):
            candidates = np.flatnonzero(~collision)
            best = (
                int(candidates[np.argmin(costs[candidates])])
                if len(candidates)
                else None
            )

            if best is not None and self._validated_plan(x0, proposals[best], scene):
                plan = proposals[best].copy()
                status = "BEST_SAMPLE_FALLBACK"
            else:
                plan = np.tile([0.0, -1.0], (cfg.horizon, 1))
                status = "NO_COLLISION_FREE_PLAN"

        # 5.7. Chỉ thực thi action đầu tiên (receding horizon).
        # U_next = [plan[1], ..., plan[H-1], plan[H-1]]: dịch trái, giữ đuôi.
        action = plan[0].copy()
        self.U[:-1] = plan[1:]
        self.U[-1] = plan[-1]
        self.last_action = action.copy()

        # ESS = 1/sum(w^2): gần 1 -> một mẫu chi phối; gần K -> trọng số gần đều.
        self.diagnostics = {
            "planner_status": status,
            "sample_count": cfg.num_samples,
            "horizon_steps": cfg.horizon,
            "lookahead_s": cfg.horizon * self.m.dt,
            "min_cost": float(costs.min()),
            "mean_cost": float(costs.mean()),
            "effective_sample_size": float(1.0 / np.sum(weights ** 2)),
            "collision_sample_fraction": float(collision.mean()),
        }

        return action


# =============================================================================
# 6. ADAPTER CỦA PROJECT: public input -> MPPI -> AgentDecision
# =============================================================================

class MPPIAgent:
    """Agent dùng action liên tục, có seed và kế hoạch riêng cho mỗi episode.

    reset(): tạo lại RNG, xóa kế hoạch episode cũ.
    act()  : đọc snapshot công khai, gọi optimizer, trả một action.
    close(): giải phóng trạng thái nội bộ.
    """

    agent_id = "stage3_mppi"
    agent_version = "0.1.0"
    method_family = "SIMULATION_PLANNING"

    def __init__(self, config: Optional[MPPIConfig] = None):
        # CLI/Workbench gọi MPPIAgent() -> nạp JSON, ghi đè mặc định của class.
        self.config = load_mppi_config() if config is None else config
        if not isinstance(self.config, MPPIConfig):
            raise TypeError("config must be MPPIConfig")

        self._controller = None
        self._step_counter = 0
        self._descriptor = AgentDescriptor(
            agent_id=self.agent_id,
            agent_version=self.agent_version,
            input_profile_id="STATE_DECISION_V1",
            action_adapter_id="continuous_box2_v1",
            inference_stochasticity="stochastic",
            stateful_within_episode=True,
            method_family=self.method_family,
        )

    @property
    def descriptor(self) -> AgentDescriptor:
        """Thông tin để launcher chọn đúng public input và action adapter."""
        return self._descriptor

    @classmethod
    def experiment_parameters(cls):
        """Ghi cấu hình và hash vào manifest để biết một lần chạy đã dùng gì."""
        parameters = {
            "schema_version": "mppi_parameters_v1",
            "search_method": "mppi",
            "model": "kinematic_bicycle_constant_actor_velocity_v1",
            "parameters": asdict(load_mppi_config()),
        }
        return dict(
            parameters,
            config_sha256=canonical_json_sha256(parameters),
        )

    def reset(
        self,
        public_context: AgentPublicEpisodeContext,
        agent_seed: Optional[int] = None,
    ) -> None:
        """Bắt đầu episode mới: giữ cấu hình, tạo lại RNG và chuỗi U bằng 0."""
        if public_context.input_profile_id != self.descriptor.input_profile_id:
            raise ValueError("MPPI requires STATE_DECISION_V1")

        if public_context.action_adapter_id != self.descriptor.action_adapter_id:
            raise ValueError("MPPI requires continuous_box2_v1")

        self._controller = MPPI(
            BicycleModel(self.config, public_context.control_dt_s),
            seed=42 if agent_seed is None else int(agent_seed),
        )
        self._step_counter = 0

    def close(self) -> None:
        """Sau close(), phải reset() trước khi gọi act() tiếp."""
        self._controller = None
        self._step_counter = 0

    def act(self, agent_input: AgentInputV1) -> AgentDecision:
        """Đọc trạng thái hiện tại và chọn một action cho bước thực thi kế tiếp."""
        if self._controller is None:
            raise RuntimeError("Call reset() before act().")

        if agent_input.profile_id != self.descriptor.input_profile_id:
            raise ValueError("MPPI requires the declared public input profile")

        self._step_counter += 1

        # 6.1. Giải mã tốc độ theo observation của MetaDrive 0.4.3.
        # z = (v_kmh + 1)/(v_max_kmh + 1)
        # => v_m_s = max(0, z*(v_max_kmh + 1) - 1) / 3.6.
        speed = max(
            0.0,
            float(agent_input.core_observation.ego_state[3])
            * (self.config.max_speed_kmh + 1.0)
            - 1.0,
        ) / 3.6

        # Action thực trước đó: observation [0,1] -> action [-1,1] bằng 2*z-1.
        # Dùng nó làm mốc tính cost độ mượt, gồm cả bước đầu của kế hoạch mới.
        ego = agent_input.core_observation.ego_state
        self._controller.last_action = np.clip(
            2 * np.asarray(ego[5:7], dtype=float) - 1,
            -1.0,
            1.0,
        )

        # 6.2. Tạo cảnh từ public input. Thiếu đường -> xóa kế hoạch, phanh thẳng.
        try:
            scene = MPPIScene(agent_input, speed, self.config)
        except ValueError:
            self._controller.U.fill(0.0)
            return AgentDecision(
                [0.0, -1.0],
                {
                    "step_counter": self._step_counter,
                    "planner_status": "NO_PUBLIC_ROUTE",
                },
            )

        # Ego là gốc hệ tọa độ hiện tại nên x=y=theta=0, chỉ cần điền vận tốc.
        x0 = np.array([0.0, 0.0, 0.0, speed])
        collision, _ = scene.footprint_flags(x0[None], np.zeros(1))
        if collision[0]:
            self._controller.U.fill(0.0)
            return AgentDecision(
                [0.0, -1.0],
                {
                    "step_counter": self._step_counter,
                    "planner_status": "ROOT_MODEL_COLLISION",
                },
            )

        # 6.3. MPPI tối ưu trong mô hình; platform sẽ thực thi action trả về.
        action = self._controller.act(x0, scene)
        return AgentDecision(
            action.tolist(),
            {
                "step_counter": self._step_counter,
                "speed_m_s": speed,
                **self._controller.diagnostics,
            },
        )
