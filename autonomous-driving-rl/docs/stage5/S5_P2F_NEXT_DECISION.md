# S5 P2-F — Quyết định tiếp theo

**Quyết định P2-F:** `CURRICULUM_NO_IMPROVEMENT`.

B hoàn thành lịch 60m → 120m → full-route và có nhiều route-progress fraction dương trong replay, nhưng không có clean training episode; full-route và curriculum evaluation đều stationary TIMEOUT. A chạy nhanh hơn nhưng gặp safety failure ở cả 6/6 full-route và 12/12 curriculum diagnostic episode. Không nhánh nào đáp ứng tiêu chí kỹ năng lái sạch.

**Hành động tiếp theo duy nhất:** dừng reset-distance treatment tại 10k và đưa kết quả này vào review nghiên cứu. Nếu nhóm quyết định tiếp tục, cần đăng ký riêng một thí nghiệm mới về phân phối episode/action sampling; không nối thêm steps, không biến curriculum hiện tại thành treatment thứ hai và không dùng kết quả curriculum để chọn benchmark checkpoint.

Không chạy P2E deterministic-episode mixture trong P2-F. Hướng đó vẫn là một đề xuất tách biệt cần protocol và phê duyệt riêng. Chưa có căn cứ GO sang multi-map, chưa có SAC benchmark candidate, và chưa bật benchmark eligibility.

Artifacts authoritative: `results/stage5/p2f_training_curriculum_20261121/result_summary.json` cùng SHA sidecar; model/replay/checkpoint bundles ở ignored `runs/stage5_p2f/s5_p2f_start_distance_seed101_20261121_retry3/`.
