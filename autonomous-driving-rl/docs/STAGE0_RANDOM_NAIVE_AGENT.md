# Stage 0 Random / Naive Agent

Agent: `stage0_random_naive`, phiên bản `0.2.0`, trong
`src/agents/random_naive_agent.py`.

## Cách chọn hành động

1. Chọn hành động ngẫu nhiên ngay ở lần `act()` đầu tiên.
2. Giữ hành động trong 2–4 giây mô phỏng, rồi chọn lại. Chu kỳ được đổi sang
   số bước bằng `AgentPublicEpisodeContext.control_dt_s`; ở 10 Hz là 20–40 bước.
3. Khi một ngã rẽ thật sự nằm trong khoảng 8 m phía trước, chọn lại ngay dù
   chưa hết chu kỳ. Chọn đều giữa những hướng có thật: trái, thẳng, phải.
   Ví dụ ngã ba chỉ có trái/phải thì không chọn đi thẳng.
4. Sau quyết định tại ngã rẽ, bắt đầu lại toàn bộ chu kỳ 2–4 giây.
   Không random liên tục mỗi frame ở cùng một ngã rẽ. Khi đã rời vùng tiếp cận
   trong ba bước liên tiếp, cho phép kích hoạt lại nếu quay về cùng ngã rẽ.
   Bộ đếm theo thời gian vẫn chạy như bình thường.

Chu kỳ thông thường giữ phân phối lái cũ: thẳng 60%, hai phía mỗi bên 20%.
Ga/phanh vẫn được random: phanh 10%, thả trôi 15%, tăng ga 75%.
`discrete9_lowbranch_v1` cung cấp ba mức lái chuẩn hóa `-0.6`, `0`, `0.6`;
đây là lệnh góc lái bánh xe, không bảo đảm xe hoàn tất một góc quay thân xe cụ thể.
Theo chiều lái vật lý MetaDrive hiện tại, giá trị dương rẽ trái.
Agent này chưa có điều khiển bám làn hoặc điều khiển để hoàn tất một nhánh rẽ.

## Cấu hình

```python
agent = Stage0RandomNaiveAgent(
    min_hold_s=2.0,
    max_hold_s=4.0,
    junction_trigger_distance_m=8.0,
)
```

`reset()` xóa bộ đếm, trạng thái ngã rẽ và khởi tạo RNG từ `agent_seed`.
Cùng seed và cùng chuỗi đầu vào sẽ tái tạo cùng quyết định và chu kỳ giữ.
Diagnostics có `resample_reason` (`initial`, `timer`, `junction`, hoặc `None`),
`steps_until_resample`, `seconds_until_resample` và `junction_trigger`.

## Dữ liệu ngã rẽ

Lớp platform trích xuất `TaskContextV1.decision_point`, một snapshot bất biến
`RoadDecisionPointV1(point_id, distance_m, available_turns)`.
Agent không giữ hoặc đọc trực tiếp đối tượng simulator.

Bộ trích xuất xét các nhánh nối từ làn hiện tại và chỉ đi tiếp qua đoạn nối
có một lối ra trong phạm vi lookahead 50 m. Dùng hướng cuối của đoạn nối để
phân biệt trái/thẳng/phải vì các nhánh ở giao lộ có thể cùng tiếp tuyến đầu.
Đường chỉ có một lối tiếp tục, đường cong và các làn song song không tự tạo
decision point. Loại nhánh quay đầu từ 150°, coi lệch hướng tối đa 20° là thẳng.
Thông tin không phụ thuộc vào hướng mà route đích yêu cầu xe đi.
`CORE_ONLY_V1` che thông tin này bằng `None`.

Schema công khai được tăng lên `1.1.0` với trạng thái
`SANDBOX-EXTENSION-PENDING-AUDIT`. Các kết quả/hash audit Gate 6 đã lưu thuộc
schema `1.0.0`; cần audit lại các hợp đồng phụ thuộc trước khi dùng phần mở rộng
cho benchmark. Agent vẫn không được đánh dấu `benchmark_eligible`.

## Xác minh

```powershell
python -X utf8 -m unittest tests.test_random_naive_agent tests.test_agent_contract tests.test_rule_based_agent
```

Kiểm tra gồm ngắt chu kỳ ở ngã rẽ, reset thời gian, không kích hoạt mỗi bước,
seed replay, reset episode, biến đổi tần số điều khiển, lựa chọn đúng tập hướng,
schema JSON và che thông tin trong `CORE_ONLY_V1`.
Bộ trích xuất cũng đã được kiểm tra với topology bản đồ MetaDrive `X`, `T`, `C`.
