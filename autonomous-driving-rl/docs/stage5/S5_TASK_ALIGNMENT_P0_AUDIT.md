# Stage 5 — Kiểm toán P0

**Quyết định: BLOCKER_INTEGRITY (F09).** Qua cua trên làn khác 0, Task Context dùng tọa độ dọc của một làn để lấy mẫu trên tim làn 0 và có thể trả waypoint nominal +2,5 m ở phía sau xe. P1 nên bắt đầu bằng regression TRAIN tái hiện lỗi và đề xuất sửa station trong lớp tương thích Stage 5. Cần change control nếu cách sửa đụng hợp đồng Platform V1. Chưa có bằng chứng lỗi này giải thích SAC đứng yên ở reset.

## Nguồn và phạm vi

Đã đọc 00–22 và SOURCE_MANIFEST theo thứ tự. Audit base d2458ed1eceb629b9e55d3dc7b4af63e69519af9. Sau fetch, PR #39 vẫn Draft/Open trên #38, PR #38 Draft/Open trên #37, PR #37 Open. MetaDrive 0.4.3 chạy từ checkout 85e5dadc6c7436d324348f6e3d8f8e680c06b4db. SB3 2.9.2. Geometry Easy SCS0 và Medium SCTCS11 đều được giải qua TRAIN allowlist, environment seed 0. Stage 1 chỉ phát lại action lịch sử để tạo trace đo, không có dữ liệu giáo viên trong replay SAC. Không có optimizer update từ P0 và không chạy P1/P2/P3.

## PASS, FAIL và UNKNOWN

- **PASS có phạm vi** — replay trajectory Stage 1 gồm 418 quyết định, SUCCESS sạch. Có reset tốc độ 0, straight, circular lane, năm lần đổi checkpoint và mẫu cuối còn một waypoint hợp lệ. Sai số inverse ego-frame tối đa 0,0701 mm so với lane 0 mà source thực sự chọn (dung sai 0,1 mm do làm tròn 4 chữ số). Sai số chỉ kiểm tra frame/rounding; giao seam kiểm thêm khoảng cách đầu-cuối lane 0 và tangent từ runtime geometry, không thay thế kiểm tra đường tham chiếu đúng.
- **PASS có phạm vi** — Navigation 10D khớp native float32. Ego speed và action slots 5–6 khớp source. Tại reset tốc độ bằng 0 cho kênh speed 1/81, hai kênh action là 0,5. Slot 7 là `acos(dot)/0.1` bị clip, không giữ dấu. Heading index 2 là dot với lane lateral đã chuẩn hóa. Native Navigation có 2×5 gồm hướng checkpoint, bend radius, bend direction, bend angle. Bảng 409 dòng có tọa độ và mask gốc trong CSV.
- **PASS có phạm vi** — Easy LiDAR 1 và Traffic rỗng. Medium có một actor hợp lệ, padding zero, LiDAR min 0,59407. Chưa kiểm multi-actor ordering thực nghiệm, toàn bộ thứ tự ray hoặc nhận thức của actor đối với vật cản. Source sắp traffic theo khoảng cách và tie break, có m/m/s/rad và padding. Ray là tỷ lệ range, không phải mét.
- **PASS Box2 / DOCUMENTATION_ISSUE** — 7 pulse, mỗi pulse 22 quyết định và 110 physics substeps. Sai số tối đa giữa lệnh và MetaDrive/info 2,39e-8. Steering ±0,1/0,3/0,6 cho đổi heading khoảng ±0,79/2,32/4,23 độ trong 0,4 s. Dương quay trái theo world yaw, ngược mô tả dấu trong action adapter. Ga +0,45 tăng 0→6,263 km/h, thả trôi còn 5,162–5,740 km/h, phanh −0,45 giảm còn 0,155–0,403 km/h. Kiểm bằng pulse hành vi, không thay adapter.
- **FAIL integrity trên làn khác 0 qua cua** — ở đường thẳng, tham chiếu tim làn 0 cách làn 1 đúng 3,5 m. Phép đo riêng đặt xe vào làn 1 cùng station ngay sau reset, sau đó cho đi liên tục ở khoảng 16 km/h bằng bộ điều khiển đo hình học. Có 315 mẫu cùng segment cong nhưng khác lane 0, trong đó 236 mẫu có first waypoint x âm. Step 614, ego làn 1 và cách tim làn hiện tại 1,595 m, station current trừ station chiếu lên reference bằng −7,685 m. Waypoint nominal +2,5 m do đó nằm ở −5,185 m dọc tuyến tham chiếu so với station của ego; ego-frame x thực tế −4,662 m. Sai lệch station tuyệt đối lớn nhất 15,223 m. Chuyến tham chiếu lane 0 vẫn PASS. Stress teleport vượt segment cũng sinh waypoint lùi, nhưng là tình huống đo riêng và không cần để chứng minh lỗi cua.
- **PASS RewardSpecV1 accounting** — xác minh SHA raw S5.5, p0=0,012258074, chiều dài 407,894416 m, H=1224, time cost 0,000204248/step. Ngưỡng lý tưởng để reward tiến độ tức thời dương là 2,9992 km/h nếu vận tốc chiếu tuyến bằng vận tốc xe. Gamma .99^1224 ≈ 4,56e-6 và bonus tại step 418 có trọng số .99^417 ≈ 0,01515. Timeout có terminal 0; signed progress giữ chiều âm. UNKNOWN với xếp hạng hai chuyến sạch nhanh/chậm vì chỉ có một trace sạch matched.

## Reward và credit

| Trace | V1 tổng | V1 chiết khấu | Scale tổng / chiết khấu | Mét tổng / chiết khấu |
|---|---:|---:|---:|---:|
| Clean, 418 bước | 1,8901 | 0,1814 | 80,2632 / 7,7038 | 78,6289 / 7,3915 |
| Stationary timeout | −0,2501 | −0,0205 | −10,6188 / −0,8697 | −10,6187 / −0,8696 |
| Unsafe seed 42 | −0,8185 | −0,2650 | −34,7585 / −11,2547 | −35,0977 / −11,4338 |
| Unsafe seed 101 | −0,8212 | −0,2769 | −34,8702 / −11,7575 | −35,2041 / −11,9347 |

Cùng cửa sổ zero-based [49,99), trước điểm preview cua của trace sạch tại 104, V1 tích lũy sạch +0,1210 so với unsafe +0,1341/+0,1379. Tốc độ trung bình lần lượt 38,81 / 43,16 / 43,77 km/h. Tính discounted tail từ cùng đầu cửa sổ cho sạch +0,2573, unsafe −0,4781/−0,4975. Đây là bằng chứng credit ngắn hạn có thể thuận lợi cho tiến nhanh; các unsafe trace kết thúc trước hoặc quanh vùng preview, không đủ để kết luận nguyên nhân là vượt tốc độ khi vào cua hay reward xếp sai chuyến sạch.

## Ledger F01–F18

| ID | Trạng thái / severity | Source hoặc phép đo | Hành động |
|---|---|---|---|
| F01 | CONFIRMED / high | S5.4 run_summary, 0/6 sạch và 6/6 timeout mỗi arm ở 100k | Giữ đối chứng lịch sử |
| F02 | CONFIRMED / high | S5.5 analysis_summary, 30k × 3; mỗi arm 6/6 timeout, stationary 100% | Loại gain bền vững từ hai reward đã thử |
| F03 | CONFIRMED / high | S5.5 C 10k full, 5,243 km/h, +0,113794 progress, 6/6 safety | Không chọn checkpoint này |
| F04 | CONFIRMED / medium | S5.5 A 10k full, 1,587 km/h, 4 safety, 2 timeout | So cùng geometry và outcome |
| F05 | CONFIRMED / medium | RAW preflight SHA và bốn full trajectories SCS0, RewardSpecV1 | Mở rộng bằng matched clean khi có |
| F06 | CONFIRMED / high | S5.5 fixed anchors, 30k self-probe SD 0 | Giới hạn kết luận theo tập mẫu |
| F07 | CONFIRMED / high | S5.5 checkpoint windows, 7.652–11.481 lần đảo dấu và 15–16 moving steps tối đa | Replay composition chưa biết; không suy nguyên nhân |
| F08 | CONFIRMED / medium | S5.5 runner callback rồi overwrite summary | Giữ outcome trong final summary và sidecar cho run mới, test bảo toàn file |
| F09 | CONFIRMED / blocker | extract_task_context kết hợp lane 0 với current lane s; arc-length mismatch 7,685 m ở step 614 trên chuyến đo liên tục | P1 regression cùng fixture và sửa phép chiếu station về đúng reference |
| F10 | CONFIRMED / low | state_obs.py và node_network_navigation.py so với comment CoreObservationV1 | CSV index/source theo pin |
| F11 | UNKNOWN / medium | Tính cùng prefix [49,99), unsafe có credit sớm cao hơn nhưng full thấp hơn | Chưa có matched clean fast/slow |
| F12 | CONFIRMED / low | reward.py timeout_penalty=0, stationary trace có −0,25 time cost | Giữ ngữ nghĩa timeout |
| F13 | CONFIRMED / medium | Easy no-hit, traffic vacant; Medium có actor và hit | Không suy rằng actor hiểu traffic |
| F14 | CONFIRMED / low | 7 physical pulses, adapter sign comment sai chiều đo được | Sửa diễn giải tại tài liệu P0, giữ contract |
| F15 | CONFIRMED / low | MlpPolicy, Ego 5–6 chứa action mới thực thi, yaw chỉ độ lớn | Chưa có state aliasing để thêm history |
| F16 | HYPOTHESIS / high | S5.3L soft Q gap +6,135 có phạm vi, S5.5 alpha cuối 2,18e-4 đến 8,90e-4 | Không đồng nhất Q gap, alpha và nguyên nhân collapse |
| F17 | CONFIRMED / medium | pilot_action_persistence.py và S5.3G runtime fidelity: đổi buffer rồi unscale cho env, reset done | Hold 5 đã giảm 461,2→91,1 flips/1.000 bước, vẫn 24/24 timeout; không lặp treatment |
| F18 | CONFIRMED / low | mppi_v1.json / mppi_v2_h30.json: 16/17 m/s, max 80 km/h | Không báo config là tốc độ đạt được |

## Artifacts và kiểm thử

- `results/stage5/task_alignment_p0/` giữ CSV manifest 409D, mẫu trace và evidence JSON kèm hashes; raw ở `runs/stage5_task_alignment_p0/`. SHA-256 raw runtime mở rộng `c1c3009e46867fb2ca580963a0febabfd659018c8133539cd6b1e209fed9eb07`.
- Thiếu replay buffer lịch sử S5.5, outcome từng episode train lịch sử và matched clean fast/slow. Sáu checkpoint A/B/C ở 10k và 30k được xác minh bundle, nạp lại và chạy inference trên 8 anchor TRAIN lịch sử. Evidence ghi action mean/std và mean Q1/Q2 tại deterministic action. Probe thu hai bước qua SB3 collect_rollouts trên SCS0 seed 386509, replay khớp tuyệt đối action đến MetaDrive ở warm-up và actor sampling, optimizer delta 0. Đây là phép đo buffer mới, không tái dựng buffer lịch sử.
- Chạy `conda run -n metadrive_sb3 python scripts/audit_stage5_task_alignment.py --output runs/stage5_task_alignment_p0/runtime_extended.json`, `conda run -n metadrive_sb3 python scripts/audit_stage5_task_alignment_checkpoints.py`, `python scripts/map_stage5_task_alignment_features.py` (ghi CSV), `python scripts/summarize_stage5_task_alignment_p0.py`. Lần đầu audit gặp lỗi serialize NumPy scalar; giữ output dở trong runs. Lần cuối ghi JSON hợp lệ. Torch import sau MetaDrive có một lần lỗi c10.dll WinError 1114; môi trường chạy thành công khi Torch được nạp trước.
- Regression: `conda run -n metadrive_sb3 python -m unittest tests.test_stage5_task_alignment_p0 tests.test_stage5_training_foundation tests.test_stage5_sac tests.test_stage5_pilot tests.test_stage5_pilot_monitor tests.test_stage5_learnability tests.test_stage5_tau_ab tests.test_stage5_meter_reward tests.test_stage5_s55_evaluation tests.test_stage5_s55_analysis tests.test_reward_metrics -q` → 92 PASS, 0 FAIL/SKIP, log local `tests.log`.
