# Stage 5 — Đặc tả nhiệm vụ lái

Policy SAC from scratch chọn đồng thời steering và throttle/brake ∈ [-1,1] mỗi 0,1 giây. Đầu vào là snapshot public AgentInputV1 409D, với encoder hiện hành và transform có định danh nếu dùng. Trạng thái simulator đầy đủ, tiến độ, reward và outcome dành cho trainer hoặc evaluator. Policy MLP có quan sát một phần. Latest action có sẵn trong Ego, qua hai kênh của một quyết định, không có chuỗi lịch sử riêng.

Mục tiêu là tự bám tuyến, phanh hoặc giảm ga trước cua và vật cản khi cần, hoàn thành sạch trong thời gian ngắn. Báo clean success, safety failures và timeout riêng theo Platform V1. So thời gian đến đích trong các chuyến sạch và luôn ghi mẫu số. Khi chưa có chuyến sạch, thời gian đến đích là N/A. Route progress có dấu, tốc độ trung bình/đỉnh, stationary fraction, lệch làn, heading và thời gian chuyển động là chỉ số chẩn đoán.

Khoảng 60 km/h là mục tiêu cruising có điều kiện, khi hình học và khả năng phanh cho phép. 80 km/h là tham số model MPPI trong cấu hình hiện hành, không phải giới hạn benchmark hay tín hiệu gửi cho SAC. Ví dụ ở gia tốc ngang giả định 3 m/s², bán kính 25 m và 60 m cho tốc độ khoảng 31 và 48 km/h. Các số đó minh họa quan hệ vật lý và chưa phải ngưỡng hiệu chuẩn.

Năng lực phát triển đòi hỏi ít nhất hai geometry Easy diagnostic khác nhau có chuyến sạch đầy đủ, safety không áp đảo và hành động phụ thuộc trạng thái. Actor frozen tự tạo cả hai lệnh. Probe bị cắt thời lượng ghi PROBE_CENSORED. TRAIN diagnostic tách khỏi replay huấn luyện. Stage 3 chỉ là nguồn tham chiếu tài liệu. Kết quả P0 giữ benchmark_eligible=false.
