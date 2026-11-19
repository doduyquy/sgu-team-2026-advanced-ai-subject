# Stage 5 — câu hỏi phản biện và câu trả lời

## Cơ bản

### 1. Reinforcement Learning là gì?

Là cách học trong đó agent quan sát môi trường, chọn hành động, nhận reward và observation mới rồi điều chỉnh policy để tối đa hóa return dài hạn. Trong Stage 5, MetaDrive tạo transition và RewardSpecV1 cung cấp training signal.

### 2. “Model-Free” nghĩa là gì?

Thuật toán không cần một mô hình động lực học tường minh do nhóm xây để dự đoán mọi trạng thái tương lai trước khi hành động. Nó học Q-value và policy từ trải nghiệm simulator. Nó vẫn dùng neural networks và MetaDrive.

### 3. SAC khác Rule-Based thế nào?

Rule-Based thi hành các luật do người thiết kế. SAC điều chỉnh policy từ dữ liệu tương tác và reward. Rule-Based dễ giải thích hơn; SAC có khả năng học tương tác phức tạp nhưng cần dữ liệu, kiểm soát và bằng chứng đánh giá.

### 4. SAC khác DQN và PPO thế nào?

DQN tiêu chuẩn dùng action rời rạc nên không trực tiếp khớp action Box liên tục của xe. PPO là policy-gradient on-policy, thường thu trajectory mới cho mỗi vòng cập nhật. SAC là actor-critic off-policy, tái sử dụng transition trong replay buffer và phù hợp với action liên tục. Đây là khác biệt thuật toán, không phải kết luận rằng SAC luôn tốt hơn.

### 5. State và observation có giống nhau không?

Không nhất thiết. State có thể là toàn bộ trạng thái simulator; observation là phần agent nhận. Agent Stage 5 chỉ thấy public `AgentInputV1` được mã hóa thành 409D, không nhận mọi trường evaluator.

### 6. Episode là gì?

Một lần mô phỏng từ reset đến một kết thúc: success, safety termination hoặc time-limit truncation. 10,000 training steps không phải 10,000 episodes; đó là 10,000 lần agent ra quyết định.

### 7. Return khác reward thế nào?

Reward là tín hiệu của một bước. Return cộng nhiều reward về sau, thường chiết khấu bằng gamma. SAC nhắm vào giá trị dài hạn chứ không chỉ chọn hành động có reward tức thời cao nhất.

### 8. Policy là gì?

Policy là ánh xạ từ observation sang phân phối hành động. SAC stochastic có thể lấy mẫu các action khác nhau; inference deterministic dùng action trung tâm/mean đã squashed để có hành vi lặp lại ổn định hơn.

## Observation, action và reward

### 9. Vì sao observation có 409 chiều?

Đó là phép ghép cố định: Core 259D, Traffic 66D, Task 84D. Các phần này tái sử dụng cấu trúc `AgentInputV1` để cùng chia sẻ quyền thông tin với Platform V1.

### 10. 259 chiều Core gồm những gì?

9 đặc trưng ego vehicle, 10 đặc trưng hướng/checkpoint và 240 tia LiDAR 360° được chuẩn hóa, theo schema trong `CoreObservationV1`.

### 11. Vì sao không dùng camera?

Stage 5 hiện nghiên cứu quyết định/điều khiển từ input cấu trúc chung của Platform V1. Dùng camera sẽ thêm bài toán perception và dữ liệu ảnh, làm thay đổi phạm vi so sánh. Đây là giới hạn thiết kế hiện tại, không phải khẳng định camera không thể dùng trong RL.

### 12. Validity mask để làm gì?

Traffic actor và waypoint có số lượng tối đa cố định. Mask phân biệt slot thật với slot zero-padding; không có mask, mạng có thể hiểu nhầm padding thành một đối tượng thật.

### 13. Hai giá trị action điều khiển ra sao?

`action[0]` là steering: âm trái, dương phải. `action[1]` là throttle/brake: âm phanh, dương ga. Cả hai trong `[-1,1]`; adapter không đổi thứ tự hoặc dấu.

### 14. 10 Hz và năm physics substeps nghĩa là gì?

Mỗi quyết định cách nhau danh nghĩa 0.1 giây: 5 substeps × 0.02 giây. Trong khoảng đó MetaDrive giữ một action qua các substeps vật lý; SAC vẫn ghi nhận một transition cho decision đó.

### 15. Reward công thức là gì?

Mặc định mỗi bước: `delta_route_completion - 0.25 / horizon_steps + terminal_reward`. Clean success cộng 1; safety failure trừ 1; timeout có terminal penalty 0. Progress có thể âm và reward có thể âm ngay cả khi xe tiến một lượng rất nhỏ.

### 16. Vì sao benchmark metric không phải reward?

Reward là scalar để tối ưu khi train. Benchmark đánh giá kết quả có ý nghĩa khoa học hơn như clean success, safety, route completion và time-to-success. Thay reward thành metric có thể khiến thuật toán tối ưu proxy thay vì mục tiêu đánh giá.

## SAC và cập nhật mạng

### 17. Actor và critic khác nhau thế nào?

Actor sinh action. Critic ước lượng giá trị của observation-action. Actor học cách chọn action có Q tốt nhưng đồng thời tính entropy; critic học theo soft Bellman target.

### 18. Vì sao có hai critic?

SAC lấy min giữa hai Q estimates trong target/actor objective để giảm tác động của overestimation phổ biến ở Q-learning. Twin critics vẫn có thể cùng sai; cần kiểm tra với return thực nghiệm.

### 19. Replay buffer làm gì?

Nó lưu các transition rồi cho SB3 lấy minibatch ngẫu nhiên để cập nhật. Điều đó giúp tái sử dụng dữ liệu và giảm tương quan theo thứ tự thời gian. Action gắn trong transition phải là action đã tạo ra transition đó.

### 20. Off-policy là gì?

Học policy hiện tại từ transition có thể được sinh ra bởi policy/behavior trước đó, thay vì chỉ dùng dữ liệu chính policy mới nhất. SAC nhờ đó sử dụng lại replay; các sai lệch trong dữ liệu vẫn có thể ảnh hưởng learning.

### 21. Entropy giúp gì? Alpha là gì?

Entropy khuyến khích policy chưa quá deterministic khi đang học. Alpha điều chỉnh mức ảnh hưởng của entropy so với Q. Cấu hình dùng `ent_coef="auto"`, để SB3 tự tối ưu alpha quanh target entropy.

### 22. Gamma ảnh hưởng gì?

Gamma chiết khấu reward xa. Gamma 0.99 làm tín hiệu xa nhiều trăm steps nhỏ đi đáng kể. Tăng gamma là thay đổi objective/horizon hiệu dụng, không phải “sửa đúng lỗi” nếu chưa có đối chứng.

### 23. Tau ảnh hưởng gì?

Tau điều khiển Polyak update của target network. Tau nhỏ làm target chậm đổi; tau lớn làm target theo online critic nhanh hơn. S5.3M so sánh 0.005 với 0.001, chỉ đổi tau.

### 24. Vì sao loss giảm không chứng minh xe lái tốt?

Loss đo mức phù hợp với target hoặc objective mà chính quá trình học tạo ra. Critic có thể nhất quán với target sai hoặc dữ liệu thiếu hữu ích. Cần xem route/safety/success và hành vi thực tế.

### 25. Nhóm tự viết SAC phải không?

Không viết lại optimizer SAC. Stable-Baselines3 cung cấp SAC core. Nhóm xây environment/encoder/sampler, tích hợp RewardSpecV1, telemetry/callback, checkpoint provenance, Platform inference adapter, diagnostic runners và Workbench.

## Dữ liệu và kết quả

### 26. MapSuite chia thế nào?

240 geometry: 180 TRAIN, 48 VALIDATION, 12 TEST. TRAIN phục vụ phát triển; VALIDATION chỉ dùng theo protocol đã duyệt; TEST là holdout cuối, không dùng để debug hay tune.

### 27. Internal split 156/24 có phải split chính thức không?

Không. Nó là split phát triển nội bộ bên trong 180 TRAIN: 156 `PILOT_TRAIN`, 24 `PILOT_DIAGNOSTIC`, 39/6 mỗi tier. Nó không sửa manifest chính thức và không thay validation/test.

### 28. Vì sao 10k steps không đồng nghĩa 10k episodes?

Mỗi episode có nhiều decision steps; một episode chỉ kết thúc khi xảy ra terminal/truncation. S5.3M mỗi arm có 10k decisions nhưng chỉ năm episode kết thúc. Nhiều transition trong replay đến từ một số ít episode.

### 29. Tại sao S5.3A–L thử nhiều điều nhưng chưa tìm root cause?

Các diagnostic hẹp từng tầng: action path, steering/speed, reward, critic, timeout và Q–return. Chúng cho biết triệu chứng và một số cơ chế phù hợp, nhưng nhiều nguyên nhân khác nhau có thể tạo cùng policy stationary. S5.3M là đối chứng tau có giới hạn; nó cũng chưa giải thích nguyên nhân.

### 30. S5.3L thấy Q–return gap bao nhiêu và nghĩa là gì?

Trên 96 rollout frozen-policy, mean min-Q khoảng 15.232 và mean entropy-aware soft return khoảng 9.097, gap khoảng +6.135. Đây là bằng chứng critic ước lượng cao hơn return theo phép đo và assumptions cụ thể. Nó không chứng minh code bug hay chứng minh gap đó khiến deterministic policy phanh.

### 31. Vì sao B hơn A mà không gọi B là winner?

B có paired route-progress delta cao hơn trung bình 0.002675 trên tám case và tám chênh lệch cùng chiều, nhưng chỉ một training root. Cả hai vẫn stationary 100%, zero clean success. Chênh lệch nhỏ không thay đổi kết luận khả năng lái.

### 32. 8/8 `PROBE_CENSORED` có nghĩa là gì?

Mỗi probe dừng ở giới hạn 300 quyết định mà không có terminal outcome. Nó không phải success, safety success hay official timeout. Không đủ quan sát để biết episode dài hơn sẽ ra sao.

### 33. Vì sao chưa chạy benchmark chính thức?

Stage 5 chưa eligible: chưa có resolved-plan binding cho checkpoint SHA, candidate chưa được xác nhận qua training roots độc lập, và chưa có sign-off/protocol hoàn chỉnh. Dùng validation/test trước khi đạt điều kiện sẽ làm hỏng ý nghĩa holdout.

### 34. Agent đã học lái chưa?

Chưa có bằng chứng clean driving. Pipeline đã train và cập nhật weights, nhưng ở S5.3M cả A/B stationary theo ngưỡng dưới 1 km/h, zero clean success và 8/8 censored. Câu trả lời khoa học hiện tại là “chưa chứng minh đã học lái hữu ích.”

### 35. Nếu tiếp tục, điều kiện nào cần trước?

Một protocol được duyệt cho xác nhận qua training roots độc lập, checkpoint SHA được bind bằng schema versioned đã review, candidate được freeze trước evaluation và holdout protocol được tuân thủ. Chưa nên nói trước rằng một hyperparameter cụ thể sẽ sửa được policy.
