# Sửa lỗi Unicode của Workbench — 15/11/2026

Run gốc `run_5987f1d680e3` thất bại trước hành động đầu tiên với
`'charmap' codec can't encode character '\xe3' in position 16`.
Manifest ghi agent `stage0_random_naive` v0.2.0 và MetaDrive 0.4.3, đúng
commit đã ghim `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`.

## Nguyên nhân và bản sửa

Thông báo `[FinishVisual] Đã dựng...` trong `src/launcher/executor.py`
chứa ký tự `ã` ở vị trí 16, không ghi được bằng Windows cp1258. Đã tái hiện
đúng lỗi này bằng stream cp1258 với chế độ lỗi strict. Cờ `-X utf8` của
tiến trình GUI không tự truyền sang worker do QProcess mở.

- `process_runner.py` đặt `PYTHONUTF8=1` và `PYTHONIOENCODING=utf-8` cho
  worker; giữ nguyên các biến môi trường khác.
- `worker.py` cấu hình các standard stream thành UTF-8 khi gọi trực tiếp,
  để gửi được cả báo cáo và lỗi chứa đường dẫn tiếng Việt.
- Thông báo FINISH dùng ASCII để CLI với bảng mã cũ cũng ghi được.
- Test chạy toàn luồng dùng run ID mới mỗi lượt, tránh đụng kết quả của
  lần chạy trước đã hoàn tất mô phỏng nhưng lỗi lúc gửi báo cáo.

Các policy, phiên bản agent, đầu vào, benchmark và mã MetaDrive không đổi.
Run lỗi gốc được giữ nguyên.

## Kiểm chứng mô phỏng

Hai lượt SANDBOX qua `WorkbenchProcessRunner` thật, render `NATIVE`, W&B
`DISABLED`; case TRAIN `sandbox/Easy/SCCS/0/0`, geometry seed 0,
environment seed 0. Mỗi lượt có thư mục riêng:

| Agent | Agent seed | Run ID | Trạng thái run | Kết quả episode |
|---|---:|---|---|---|
| Stage 0 v0.2.0 | 101 | `utf8_stage0_fd2da2459b4a` | COMPLETE, VERIFIED | OUT_OF_ROAD; 9,65% tuyến; 80 bước |
| Stage 1 v0.6.1 | 42 | `utf8_stage1_61b148e89fde` | COMPLETE, VERIFIED | SUCCESS; 99,18% tuyến; 597 bước; 59,7 giây mô phỏng |

Raw artifacts nằm tại `runs/<run_id>/`, gồm manifest, state, episodes,
timing, summary và integrity. Stage 1 còn có provenance, snapshot nguồn
và decision trace. `RunArtifactRepository` xác nhận cả hai run COMPLETE,
VERIFIED, đã ghi đủ 1/1 episode. Đây là kiểm tra phát triển trên một case
TRAIN; không phải kết quả benchmark VALIDATION/TEST.

Hai lượt mô phỏng kiểm chứng môi trường QProcess và thông báo FINISH.
Phần cấu hình stream khi gọi worker trực tiếp được bổ sung và kiểm chứng
trong bộ test cuối.

## Kiểm thử

**137 tests qua, 0 lỗi**, gồm `test_finish_visual`, `test_workbench`,
`test_workbench_hardening`, `test_workbench_results` (83,759 giây).
Ba regression tests mới kiểm tra thông báo FINISH trên cp1258, QProcess
thật với stdout/stderr và file tiếng Việt, và worker gọi trực tiếp báo lỗi
chứa đường dẫn tiếng Việt. Các regression test đã thất bại trước bản sửa.

Lệnh đã chạy từ `autonomous-driving-rl/`, bằng Python môi trường `metadrive`:

```powershell
python -X utf8 -u -c "from PySide6.QtWidgets import QApplication; import sys, unittest; sys.path.insert(0, 'tests'); suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in ['test_finish_visual', 'test_workbench', 'test_workbench_hardening', 'test_workbench_results']); result=unittest.TextTestRunner(verbosity=1).run(suite); raise SystemExit(not result.wasSuccessful())"
```

Các lượt đầu trong sandbox bị chặn quyền tạo thư mục run và đọc texture
ở thư mục tạm. Các lượt mô phỏng và bộ test cuối chạy ngoài sandbox.
Lần thử bị chặn `utf8_stage0_cdb73a84d6f7` không tạo được thư mục run;
không có episode được thực thi ở lần này.

## Mở lại Workbench

Đóng GUI cũ để nạp lại mã, dùng Python của môi trường `metadrive`, rồi
chọn SANDBOX → agent → Easy → NATIVE → Resolve & Preflight → Run.

