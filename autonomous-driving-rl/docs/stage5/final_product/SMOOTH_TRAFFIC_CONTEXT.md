# Smooth TrafficContext thử nghiệm

**Quyết định: loại wrapper khỏi phương án bàn giao.** Smooth TrafficContext gần như triệt tiêu bước nhảy ở 50 m trên hai input đã khóa, nhưng trên đúng ca Extreme đang xét, xe đổi từ timeout ở 39,73% sang ra khỏi đường ở 35,44%. Thử nghiệm dừng tại SANDBOX.

## Phương pháp và kết quả

Wrapper dùng V2–5.000 đóng băng, giữ nguyên raw steering. Với actor xa và có envelope chuyển động đủ tách khỏi hành lang trong horizon ngắn, wrapper tạo các tập actor con hợp lệ, rồi tích phân lệnh ga/phanh qua các ngưỡng loại actor lồng nhau. Mỗi observation đối chứng được pack lại theo slot, mask và count, giữ LiDAR nguyên bản. Khi producer báo overflow, khi thiếu dải waypoint hoặc ở terminal preview, wrapper giữ action nguyên bản. Không co giãn tọa độ NPC hay đổi tầm cảm biến.

| Input Extreme | Bước nhảy ga/phanh của V2 | Sau blend | Mức giảm |
|---|---:|---:|---:|
| Bắt đầu pha dừng, bước 2.456 | 1,24658 | 0 | 100% |
| Input terminal, bước 2.973 | 1,15607 | 0,000000119 | >99,9999% |

Trên toàn bộ 6.423 observation đã lưu, độ lệch steering offline bằng 0. Các probe xe gần chắn đường, xe thu hẹp nhanh và xe cắt ngang giữ ga/phanh nguyên bản. Đối chứng này chỉ kiểm tra wrapper giữ action trong các probe ấy; một số action gốc vốn gần bão hòa nên kết quả không chứng nhận phản ứng an toàn. Ở nhóm Medium cua, p95 |Δ ga/phanh| là 0,031, với hai frame đi từ phanh sang ga và không có frame ngược chiều.

| Ca ghép cặp | V2 | Wrapper | Tiến tuyến wrapper |
|---|---|---|---:|
| Extreme gốc | Timeout | Ra khỏi đường | 35,44% |
| Medium SCXCCS | Va chạm | Va chạm | 99,28% |
| Medium SCTCS | Va chạm | Ra khỏi đường | 61,25% |
| Hard XTOCS | Clean success | Clean success | 99,49% |
| Extreme CrXROSTR | Ra khỏi đường | Va chạm | 22,78% |

Ca Extreme gốc có 348 lần can thiệp, vẫn còn 20 NPC pending và kết thúc sớm vì ra khỏi đường. Thời gian đứng yên ngắn hơn không phải bằng chứng giải freezing. Ngân sách pool là tám worker; năm ca thực tế sử dụng năm process đồng thời và đạt 151,56 quyết định/s đầu-cuối, tương đương ×15,16 ở tần số 10 Hz. Không chạy panel 48 ca cho wrapper sau lỗi an toàn, không thu teacher data và không train luồng này.

JSON đối chiếu nằm trong `runs/kaggle/smooth-traffic/offline.json` và `sandbox.json`. Mã thực nghiệm tách khỏi launcher và benchmark chuẩn. V2 tiếp tục làm chuẩn so sánh của Consolidation V3.
