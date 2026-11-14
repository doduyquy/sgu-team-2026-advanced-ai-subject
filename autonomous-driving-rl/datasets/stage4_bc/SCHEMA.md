# Cấu trúc dữ liệu hành vi mẫu Stage 4 — Đề xuất phiên bản 1

Trạng thái: **cấu trúc BC đề xuất, cần xem xét tại G1**. Tài liệu quy định yêu
cầu tối thiểu cho bộ xây dữ liệu. G1 bước 2 đã có [dữ liệu thô pilot](pilot_raw_v1.json)
được kiểm tra ghép bước/nguồn gốc; chưa duyệt chất lượng. Cách chia theo hình
học đã chốt tại [manifest FIT/DEV v1](fit_dev_v1/manifest.json).

## Ghép quan sát/hành động và truy xuất nguồn gốc

Dùng bản ghi `DecisionTrace`: `agent_input(t) -> canonical_action(t)`.
Bộ thực thi ghi sau khi chuyển đổi hành động và trước `env.step()`. Nhãn là giá
trị float32 thực tế gửi đến MetaDrive. Giữ dấu lái và thang ga/phanh. Không lấy
quan sát kế tiếp, giá trị float Python chưa chuyển đổi, chẩn đoán của tác nhân
hướng dẫn hoặc chỉ số sau bước làm đặc trưng đầu vào.

Yêu cầu lượt nguồn đã hoàn tất, bước liên tục từ 0, đúng tình huống và số dòng
khớp kết quả. Vết có thể chứa quyết định cuối ngay trước một `env.step()` bị lỗi;
không tự nhận lượt lỗi hoặc thiếu vào dữ liệu. BC không yêu cầu `(s', r)`.

Tệp kê khai ghi phiên bản dữ liệu/cấu trúc/quy tắc chất lượng, lượt nguồn, mã băm
mã nguồn/cấu hình tác nhân hướng dẫn, bộ trích xuất đầu vào/bộ chuyển hành động,
hình học, cách chia tập, các phần dữ liệu và sổ đoạn nhận/loại. Mỗi mẫu trỏ tới
`(run_id, episode_id, step_index)`. Mỗi lượt ghi đúng seed hình học/môi trường,
giới hạn bước, cấu hình, môi trường thực thi, kết quả và trạng thái can thiệp.
Bằng chứng 0.6.0 giữ phiên bản đó, không đổi nhãn thành dữ liệu thu bằng 0.6.1.

## Chia tập

Chia hình học TRAIN trước khi cắt đoạn, chuẩn hóa hoặc cân bằng. Trong mỗi họ,
xếp theo SHA-256 của chuỗi UTF-8 `stage4-fit-dev-v1|<geometry_id>`; nếu trùng thì
so định danh. Ba hình học đầu thuộc DEV, mười hai còn lại thuộc FIT: tổng 36 DEV
và 144 FIT trên 12 họ. Người dùng đã chốt ngày 14/11/2026; G1 đóng băng đúng
danh sách G0 đã đề xuất, xem [quyết định và phạm vi](../../docs/STAGE4_G1_SPLIT_KAGGLE_VI.md).
Tất cả lượt, đoạn và lịch sử của cùng hình học thuộc cùng một phía.

Không dùng canonical VALIDATION/TEST làm dữ liệu mẫu. DEV ở đây là phần giữ lại
từ TRAIN, khác bộ VALIDATION chuẩn có 96 tình huống.

## Sổ xem xét chất lượng

Mỗi khoảng được cân nhắc cho BC phải ghi:

| Thông tin | Yêu cầu |
|---|---|
| Định danh nguồn | Mã băm vết; tác nhân hướng dẫn, nguồn/cấu hình, lượt và hình học |
| Khoảng bước | Bắt đầu được tính, kết thúc không tính; ngữ cảnh trước và sau |
| Loại chất lượng | Tiến triển hợp lệ / chờ an toàn có lý do / bế tắc hoặc lặp vô ích / chưa rõ |
| Bằng chứng | Quan sát/ngữ cảnh công khai, chẩn đoán đã đối chiếu, mâu thuẫn và giới hạn |
| Xem xét | Người xem, thời điểm thực tế, lý do, phiên bản quy tắc và trạng thái nhận/loại/chưa rõ |
| Đưa vào dữ liệu | BC-A, BC-B hoặc không dùng; giữ lịch sử quyết định bất biến |

SUCCESS vẫn phải kiểm định đầu vào/hành động/nguồn gốc và chất lượng điều khiển.
TIMEOUT cần xem từng đoạn: chờ an toàn có thể hữu ích, bế tắc vô lý kéo dài thì
không. Nhãn YIELD/BLOCKED hoặc việc không va chạm ở cuối lượt chưa tự chứng
minh toàn đoạn an toàn hoặc chất lượng tốt.

Đối chiếu tính khả thi của làn/tuyến, vị trí/chuyển động/tràn dung lượng của xe
khác, lý do phanh/chờ, tính liên tục điều khiển và diễn biến xung quanh. Kiểm tra
vật cản có còn tồn tại trước khi phân loại đoạn đứng yên. Nếu thiếu bằng chứng,
giữ nhãn chưa xác định. Độ dịch chuyển từ vết chưa phải tiến độ dọc tuyến.

**Hiệu chỉnh tiêu chí:** thống kê FIT; xem ví dụ tiến triển/chờ/bế tắc cùng ngữ
cảnh; đề xuất quy tắc giải thích được; xem trường hợp giáp ranh; ghi bất đồng và
lý do loại; đóng băng phiên bản trước khi tạo BC-B. Kiểm chứng quy tắc trên DEV
nội bộ nhưng không huấn luyện hoặc fit chuẩn hóa bằng DEV. G0 chưa đặt ngưỡng
cắt, trọng số, tỷ lệ lấy mẫu hoặc ranh giới đoạn chất lượng cuối cùng.

## Giới hạn đặc trưng đầu vào

Bộ mã hóa chỉ nhận `AgentInputV1` công khai. Bộ đọc tách quan sát/nhãn khỏi siêu
dữ liệu phân tích. Các nhóm có kích thước cố định và mặt nạ hợp lệ: lõi 259D;
8 xe giao thông; tối đa 8 đường khả dĩ/xe với 22 điểm/đường; 20 điểm tuyến;
làn hiện tại/trái/phải tối đa 28 điểm/làn; điểm quyết định, đoạn giao cắt, chuyển
động cục bộ của xe, loại biên và khả năng nối tuyến.

Góc dùng sin/cos; loại rời rạc dùng one-hot; giá trị thiếu hoặc không hợp lệ có
mặt nạ riêng. Chuẩn hóa chỉ bằng phần tử FIT hợp lệ; phần đệm giữ 0. Không nhúng
định danh hoặc mã băm làn/nhánh/đoạn/tình huống thành đặc trưng. Vị trí xe trong
mảng không được xem là định danh bền vững giữa các khung quan sát.

Lịch sử chỉ dùng quan sát hiện tại và quá khứ trong cùng lượt, có mặt nạ và đệm.
Không bổ sung nhãn hành động hướng dẫn từ vết làm đặc trưng; giữ lõi công khai
theo hợp đồng. Không mượn khung tương lai qua biên đoạn. G1 phải ghi rõ việc dùng
ngữ cảnh trước đoạn mà không biến các bước đó thành nhãn huấn luyện được nhận.

Kết quả, phần thưởng, trạng thái riêng của bộ đánh giá, chẩn đoán/bộ nhớ tác nhân
hướng dẫn và nhãn chất lượng chỉ phục vụ phân tích. So sánh BC-A/BC-B dùng cùng
bộ mã hóa/quy trình chuẩn hóa, báo riêng số mẫu, số bước cập nhật và mức phủ
hành vi/hình học.
