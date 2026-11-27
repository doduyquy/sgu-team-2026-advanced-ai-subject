# Tổng hợp kinh nghiệm SAC trên máy local

## Thiết kế thực nghiệm

Sử dụng bản sao toàn trạng thái S01 để giữ trọng số, trạng thái ba optimizer, alpha và 122.880 transition trong replay nền. Trong đó có 5.300 transition guided được bảo vệ. Số transition lịch sử nhập từ các learner khác là 0 vì các replay ring đã lưu thiếu provenance ở cấp transition để xác minh khi hợp nhất.

Chạy lại ba chính sách từ chiến dịch đầu tiên trong MetaDrive ở chế độ headless. S01 là chính sách tham chiếu. S09 từng hoàn thành hai ca Medium mà S01 bỏ lỡ trên panel xác nhận. S13 có kết quả Medium khả quan trên panel phát triển và được dùng như một nguồn thu bổ sung. Các policy chạy trên cùng các cặp geometry và seed TRAIN trong từng đợt. Bộ kiểm tra giữ cả ca thành công và thất bại hợp lệ, xác minh observation 409D, hành động Box2 đã thực thi, native reward, observation kế tiếp và trạng thái kết thúc trước khi nhận dữ liệu. Các bộ dữ liệu chẩn đoán, VALIDATION và TEST nằm ngoài quá trình thu thập và cập nhật.

Mỗi minibatch 256 mẫu gồm 192 lượt từ replay nền và 64 lượt từ replay bổ sung. 64 lượt được chia gần đều theo chính sách thu thập rồi theo tier hiện có của chính sách đó. Bộ học cập nhật actor, critics, target critics và hệ số entropy. Đây là học off-policy từ dữ liệu đã thu. Lượt cập nhật không chạy thêm environment interaction. Lượt mẫu được đếm riêng với số transition độc nhất vì một transition có thể xuất hiện nhiều lần.

## Replay thu thập

Ba đợt thu mới tạo 53.724 transition. Cộng với 5.586 transition của đợt pilot ban đầu, tổng dữ liệu bổ sung là 59.310 transition từ 118 episode và 40 geometry TRAIN phân theo 41 cặp geometry–seed.

| Policy thu thập | Easy | Medium | Hard | Extreme | Tổng |
|---|---:|---:|---:|---:|---:|
| S01 | 2.015 | 9.902 | 6.065 | 8.464 | 26.446 |
| S09 | 1.617 | 9.186 | 3.687 | 2.561 | 17.051 |
| S13 | 1.630 | 7.132 | 3.410 | 3.641 | 15.813 |
| Tổng | 5.262 | 26.220 | 13.162 | 14.666 | 59.310 |

Có 20 episode thành công trên Easy, 35 trên Medium và 2 trên Hard. Không có episode thành công trên Extreme. Tập replay Extreme gồm 18 episode lỗi an toàn và 2 timeout. Kết quả phản ánh độ khó của việc thu thập trải nghiệm cuối tuyến ở tier cao.

## Cập nhật SAC

| Mốc tích lũy | Replay bổ sung đủ điều kiện | Lượt mẫu bổ sung tích lũy |
|---:|---:|---:|
| 128 | 5.586 | 8.192 |
| 2.000 | 20.085 | 128.000 |
| 5.000 | 36.532 | 320.000 |
| 10.000 | 59.310 | 640.000 |

Không phát hiện mẫu trùng trong bốn batch đã nhập hoặc episode bị loại vì lỗi định vị trong lượt thu. Ba optimizer đều tăng từ bước 640.940 lên 650.940. Trong toàn bộ lượt cập nhật, learner lấy 1.920.000 lượt mẫu từ replay nền và 640.000 lượt mẫu từ replay bổ sung. Replay nền được đối chiếu toàn bộ tại cả bốn checkpoint. Thời gian cập nhật gradient khoảng 352 giây trên CPU, ngoài thời gian thu dữ liệu, nạp replay, xuất checkpoint và đánh giá.

## Phương pháp đánh giá

Mỗi checkpoint chạy trên đúng 48 ca của panel xác nhận nội bộ S01. Mỗi tier gồm sáu geometry, mỗi geometry chạy với hai seed môi trường đã khóa. Policy chạy deterministic với steering gốc. Kết quả đo clean success, lỗi an toàn, timeout, tiến độ, tốc độ và thời gian tới đích. Chênh lệch thời gian được tính riêng trên các ca cả hai policy hoàn thành an toàn. Đây là đánh giá mô phỏng phát triển.

## Kết quả ghép cặp

Các ô ghi số ca clean và số ca lỗi an toàn, mỗi tier có 12 ca.

| Checkpoint | Easy | Medium | Hard | Extreme |
|---|---:|---:|---:|---:|
| S01 | 12 / 0 | 10 / 2 | 2 / 10 | 0 / 10 |
| 128 updates | 12 / 0 | 11 / 1 | 1 / 11 | 0 / 12 |
| 2.000 updates | 1 / 11 | 11 / 1 | 4 / 8 | 0 / 12 |
| 5.000 updates | 11 / 1 | 7 / 5 | 4 / 8 | 0 / 12 |
| 10.000 updates | 11 / 1 | 7 / 5 | 3 / 8 | 0 / 7 |

Pilot hoàn thành thêm hai ca Medium từng thất bại ở S01, đồng thời mất một ca Medium từng thành công. Trên chín ca Medium mà cả hai hoàn thành an toàn, pilot nhanh hơn trung bình 1,31 giây. Mốc 2.000 updates đạt thêm hai clean success trên Hard về tổng số nhưng có 11 ca Easy văng đường dù tiến độ Easy trung bình đạt 97,95%. Mốc 5.000 giữ 4/12 trên Hard và nâng tiến độ Hard lên 72,53%, so với 61,42% của S01. Cùng mốc đó, Medium giảm còn 7/12.

Mốc 10.000 giảm lỗi an toàn Extreme từ 10 xuống 7 và tăng tiến độ trung bình từ 19,43% lên 26,28%. Timeout tăng từ 2 lên 5 và chưa có ca hoàn thành an toàn. Không có lỗi kỹ thuật ở các checkpoint trong phép đánh giá này.

## Model bàn giao và giới hạn

Giữ S01 với steering nguyên bản cho bản bàn giao. Pilot 128 updates là checkpoint đã học giữ kết quả Easy và Medium tốt nhất theo thứ tự ưu tiên retention của thử nghiệm. Chênh lệch an toàn trên Hard và Extreme khiến pilot chưa đủ điều kiện thay S01. Các mốc nhiều updates hơn cũng giảm kết quả Easy hoặc Medium.

Thử nghiệm xác nhận learner đã học từ replay của nhiều chính sách. Nó cho thấy cải thiện cục bộ trên từng tier và dao động qua các checkpoint. Replay được mở rộng cùng với số updates nên thiết kế này chưa tách riêng ảnh hưởng của quy mô replay khỏi thời lượng học. Kết quả chỉ áp dụng cho panel mô phỏng đã dùng.

## Dữ liệu làm báo cáo

- [So sánh trên panel 48 ca](local_consolidation_comparison.csv). Cột `mean_progress` dùng tỷ lệ từ 0 đến 1. Ô thời gian để trống khi không có ca clean chung.
- [Transition theo nguồn và tier](local_consolidation_replay.csv) và [kết cục từng nhóm episode TRAIN](local_consolidation_outcomes.csv).
- [Ngân sách học và kết quả lấy mẫu](local_consolidation_learning.csv).
