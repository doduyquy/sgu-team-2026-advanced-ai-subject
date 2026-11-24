# Tóm tắt dự án mô phỏng xe tự hành

## 1. Nhiệm vụ của xe và bối cảnh mô phỏng

Xe tự hành di chuyển từ điểm xuất phát đến đích theo tuyến được giao, bảo đảm an toàn và hoàn thành trong thời gian cho phép. Xe cần giữ làn, điều chỉnh tốc độ, qua đường cong, giao lộ, vòng xuyến và tương tác với xe khác.

Môi trường là các tuyến đường đô thị/cao tốc tổng hợp, được sinh bằng seed cố định để tái lập và so sánh các thuật toán. Agent nhận trạng thái xe, điều hướng, LiDAR và ngữ cảnh giao thông; điều khiển góc lái và ga/phanh ở tần số **10 Hz**.

## 2. Simulator và công cụ sử dụng

- **MetaDrive 0.4.3:** mô phỏng xe, bản đồ và giao thông; ghim commit `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`.
- **Python 3.10, Conda, giao diện Gymnasium:** môi trường phát triển và tương tác mô phỏng.
- **Research Platform V1, Launcher/CLI:** chuẩn hóa thí nghiệm, kiểm tra cấu hình, đánh giá và ghi log.
- **Research Workbench (PySide6):** giao diện thiết lập, giám sát và xem kết quả.
- **NumPy, pandas, matplotlib:** xử lý dữ liệu và trực quan hóa; **W&B** theo dõi thí nghiệm tùy chọn.
- **scikit-learn 1.7.2, joblib:** nạp model ExtraTrees của [Agent 4 PDD 1.0.0](docs/AGENT4_PDD_AGENT.md).

## 3. Các đối tượng trong môi trường

- Xe tự hành và các xe giao thông.
- Đường thẳng, đường cong, ngã ba, ngã tư, vòng xuyến, đường vào/ra cao tốc, đoạn thu/mở rộng làn.
- Làn đường, vạch kẻ, biên đường, vỉa hè; điểm xuất phát, checkpoint và vùng đích.

Hệ thống nhận diện thêm va chạm với vật thể tĩnh, công trình và người đi bộ, nhưng benchmark hiện chưa có nhóm kịch bản riêng cho người đi bộ/tai nạn ngẫu nhiên (`accident_prob = 0.0`). Số làn cơ sở thực tế là **3**, rộng **3,5 m/làn**; thông tin 2 làn trong một số tài liệu cũ đã được [đính chính](docs/environment/MAPSUITE_V1_LANE_METADATA_ERRATUM.md).

## 4. Điều kiện thành công và thất bại

| Kết quả | Điều kiện |
|---|---|
| **Thành công an toàn (`clean_success`)** | Vào vùng đích hợp lệ và không có cờ vi phạm an toàn. |
| **Thất bại do an toàn** | Va chạm xe, vật thể, công trình, người, vỉa hè hoặc ra khỏi đường. |
| **Hết thời gian (`TIMEOUT`)** | Chưa hoàn thành khi hết ngân sách bước; ghi nhận bằng `truncated=True`. |
| **Kết thúc chưa rõ nguyên nhân** | Simulator kết thúc nhưng không có cờ nguyên nhân đã biết; không tính thành công an toàn. |

Lỗi an toàn được ưu tiên: **tới đích đồng thời va chạm vẫn là thất bại**. Giới hạn lượt chạy tính theo chiều dài tuyến, trong khoảng **1.000–4.000 bước** (100–400 giây mô phỏng). Tổng reward không quyết định thành công.

## 5. Danh sách kịch bản

Các kịch bản đã được đóng gói trong **MapSuite V1**, gồm 12 họ:

| Độ khó | Kịch bản | Nội dung chính | Mật độ giao thông |
|---|---|---|---|
| Easy | `SCS`, `SCSS`, `SCCS` | Bám làn trên đường thẳng và đường cong; không có xe giao thông. | 0,0 |
| Medium | `SCXCS`, `SCTCS`, `SCXCCS` | Ngã ba/ngã tư kết hợp đường cong, giao thông thưa. | 0,08 |
| Hard | `SCXOCS`, `SCTXrCS`, `XTOCS` | Nhiều giao lộ, vòng xuyến hoặc đường nhập cao tốc. | 0,15 |
| Extreme | `CrXROSTR`, `SCXOCrTYCS`, `SCTXORyCCS` | Kết hợp giao lộ, vòng xuyến, đường vào/ra và thu/mở rộng làn trong giao thông dày. | 0,25 |

Mật độ là tham số của simulator, không phải số xe cố định. Mỗi họ có **20 seed hình học**, tổng cộng **240 cấu hình bản đồ**: **180 TRAIN / 48 VALIDATION / 12 TEST**. Kết hợp seed môi trường cố định tạo **96 lượt VALIDATION / 60 lượt TEST**.

Nguồn đối chiếu: [README dự án](README.md), [MapSuite V1](datasets/mapsuite_v1/README.md), [danh sách kịch bản](datasets/mapsuite_v1/scenario_families.csv), [luật kết thúc](src/platform/episode.py), [cấu hình episode](configs/platform/episode_spec_v1.json).
