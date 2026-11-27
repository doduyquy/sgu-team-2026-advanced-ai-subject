# Tổng hợp kinh nghiệm SAC V3 từ chiến dịch thứ hai

## Thu hồi và đánh giá nguồn

Đã xác minh model và metadata checkpoint cuối cho đủ **22/22 learner** theo receipt Kaggle có version. Mỗi nguồn được đối chiếu danh tính job, checksum và hợp đồng observation 409D, Box2, native reward, Terminal Preview, F09 và transform. Bốn lượt dispatch ban đầu không có version được đối chiếu với các recovery tương ứng; chúng không được tính thành learner bổ sung. Không nhập ring của các phiên cloud vào replay học chéo vì ring thiếu provenance TRAIN ở cấp transition.

Manifest cloud của cả 22 learner đều ghi `COMPLETE`. `learners22.csv` ghi riêng policy, full state cuối đã xác minh, periodic full state, dấu vết screening và version. Listing API giới hạn 200 file mỗi trang; số marker từ trang đầu chỉ là cận dưới khi danh sách còn phân trang.

Từng policy chạy một ca cố định ở mỗi tier, tổng cộng 88 episode sàng lọc ghép cặp. Easy có 20/22 policy hoàn thành, Medium 19/22. Ca Hard và ca Extreme được chọn không có policy nào hoàn thành sạch; tiến tuyến cao nhất lần lượt là 68,76% và 28,42%. Đây là một ca trên mỗi tier cho mỗi policy. Kết quả không đại diện cho tỷ lệ hoàn thành toàn tier, cũng không phủ nhận hai episode Hard thành công khi thu trên geometry TRAIN khác.

Toàn bộ 22 policy được chạy lại trên các observation đã lưu để đo nhảy ga ở 50 m, phản ứng với xe rời xa, xe thu hẹp, xe cắt ngang và ca Medium cua. Một số policy có bước nhảy dưới 0,001 nhưng vẫn phát gần hết ga trên các probe nguy cơ. Chỉ số liên tục không được dùng một mình để chọn nguồn hoặc model.

## Replay và huấn luyện

Hai mươi hai policy thu 88 episode dự kiến trên các geometry TRAIN độc lập. Có 87 episode hợp lệ và một ca lỗi localization bị loại. Thu được **32.318 transition** từ policy nguồn, trong đó **27.125 transition duy nhất** được nạp sau xác minh và giới hạn số hàng mỗi episode. Dữ liệu có 9 ca Medium thành công và 2 ca Hard thành công; các ca Extreme đóng góp transition thất bại và timeout hợp lệ cho critic. Đường thu replay chỉ chạy policy deterministic nguyên bản, giữ successor thật, cờ terminate/truncate và action thực thi. Các observation phản thực tế của hai luồng chẩn đoán không đi vào replay.

Hai trajectory Hard thành công đến từ S09 trên họ XTOCS (439 hàng) và S15 trên họ SCXOCS (548 hàng). Đây là bằng chứng về hai ca TRAIN cụ thể. Không dùng chúng để suy rộng khả năng hoàn thành giao lộ hoặc vòng xuyến trên toàn tier.

S03 đóng góp trajectory từ checkpoint periodic đã xác minh, có trước khi policy cuối sẵn sàng. Các bảng lưu checksum riêng của policy nguồn thu dữ liệu và policy cuối dùng để khảo sát.

V3 tiếp tục đầy đủ trạng thái actor, critic, alpha, optimizer và RNG từ V2–5.000. Minibatch 256 gồm 192 mẫu retention, 38 mẫu nguồn mới và 26 mẫu online. Phần retention lấy từ ring S01, external Easy/Medium của V2 bao gồm dữ liệu V1 tương ứng, online V2 và hai pool V1 Hard/Extreme hợp lệ được giữ riêng. Warm-up online có 7.269 hàng trên tám episode, trước lần gradient update đầu. Steering retention dùng tham chiếu V2 đóng băng chỉ trên phần minibatch retention, không bắt buộc khớp ga/phanh.

Pilot chưa dùng regularization và Actor LR 2,5×10⁻⁵ dừng ở 16 updates khi p95 drift steering lên 0,480 và 265/2.143 observation đảo dấu. Pilot sau giảm Actor LR xuống 2,5×10⁻⁶, giữ Critic LR 10⁻⁴, và cộng MSE giữ steering với trọng số 4. Tại 64 updates, pilot thứ hai giữ 12/12 Easy và Medium nhưng có 9 lỗi an toàn Hard so với 7 của V2. Hai timeout Hard đã biến thành lỗi an toàn, nên ứng viên này không đủ điều kiện bàn giao.

| Tier | Hàng thu từ 22 policy | Hàng donor nạp vào V3 | Hàng online mới |
|---|---:|---:|---:|
| Easy | 0 | 0 | 296 |
| Medium | 9.868 | 9.368 | 1.640 |
| Hard | 11.818 | 10.417 | 613 |
| Extreme | 10.632 | 7.340 | 4.720 |
| Tổng | 32.318 | 27.125 | 7.269 |

Nhánh đủ 22 nguồn chạy 512 updates. Mốc 512 dừng ở cổng sàng lọc do thêm lỗi an toàn trên Hard và Extreme. Tổng 131.072 lượt lấy mẫu gồm 98.304 lượt retention, 19.456 lượt từ nguồn đợt hai và 13.312 lượt online mới. Tỷ lệ lượt lấy mẫu trên số hàng online đã nạp là 1,83. UTD thực tế tính trên 7.269 hàng online là 0,0704. Giai đoạn thu dữ liệu tổng cộng 39.587 quyết định với tám worker, đo đầu-cuối đạt 173,02 quyết định/s (×17,30), đợt cao nhất 240,52 quyết định/s. Tại lần đo thấp nhất sau một wave vẫn còn khoảng 5,88 GiB RAM khả dụng. Các chỉ số này gồm reset, inference và xuất chunk nên không so trực tiếp với throughput steady-state của V2.

## Kết quả ghép cặp 48 ca và khóa model

Mỗi ô ghi clean success / lỗi an toàn / timeout, trong 12 ca của tier tương ứng.

| Checkpoint | Easy | Medium | Hard | Extreme |
|---|---:|---:|---:|---:|
| **V2–5.000** | **12 / 0 / 0** | **12 / 0 / 0** | **3 / 7 / 2** | **0 / 7 / 5** |
| Pilot retention, 64 updates, 10 donor | 12 / 0 / 0 | 12 / 0 / 0 | 3 / 9 / 0 | 0 / 7 / 5 |
| V3, 32 updates, 22 donor | 12 / 0 / 0 | 11 / 1 / 0 | 2 / 10 / 0 | 0 / 9 / 3 |
| V3, 128 updates, 22 donor | 12 / 0 / 0 | 12 / 0 / 0 | 2 / 10 / 0 | 0 / 6 / 6 |

Trên 2.143 observation đối chứng, V3–128 có p95 drift steering 0,042 và không có đảo dấu steering qua deadband. Tuy vậy, Hard mất một ca clean và tăng ba lỗi an toàn. Ở đúng ca Extreme bị dừng, V3–128 ra khỏi đường tại 34,09%, so với 39,73% và timeout ở V2. Mức chênh đầu ra tại biên 50 m trên probe bắt đầu phanh là 1,2523 ở V3–128, so với 1,2466 ở V2. Replay thêm và loss retention giữ các thao tác gần V2 trên mẫu offline; độ liên tục và an toàn vòng kín vẫn không đạt cổng.

**Model bàn giao tiếp tục là V2–5.000.** V3–128 được giữ làm checkpoint nghiên cứu. Full state của nó khôi phục được bộ đếm **646.068** của cả ba optimizer, replay và minibatch kế tiếp. SHA-256 manifest là `f1a27aa11b8560bab15f56105bef4d0dfbb7c34d6aa58daf513999d2b75cbfe1`. Demo V2 qua launcher SANDBOX chung chạy lại thành công một ca Medium với 99,22% tiến tuyến trong 63,9 giây mô phỏng. [Kết quả wrapper riêng](SMOOTH_TRAFFIC_CONTEXT.md)

## Tái lập và đường dẫn

- `python -m cloud.kaggle.second_window_harvest --mode small` chạy bằng Python có Kaggle CLI/SDK, dùng operator slot và DPAPI đã cấu hình. `--mode full` chỉ thu trạng thái còn thiếu theo giới hạn dung lượng. Bước này không cấp quyền hoặc sửa notebook.
- `python -m cloud.kaggle.second_window_inventory --screen` xác minh đủ danh mục, chạy sensitivity và sàng lọc nguồn. `python -m cloud.kaggle.v3_collect --warmup` thu trajectory TRAIN. `python -m cloud.kaggle.v3_train --deadline <ISO-8601>` học trên dữ liệu đã xác minh và áp dụng cổng kiểm chứng.
- Receipt kỹ thuật, bảng CSV chi tiết, dải checkpoint và proof resume nằm trong `runs/kaggle/second-window-v3/`. State hoàn chỉnh từ Kaggle được giữ trên ổ C trong `Stage5CampaignHarvest`; trajectory và state V3 mới nằm trên ổ dự án.
- `python -m cloud.kaggle.v3_evidence` cập nhật `learners22.csv`, `admitted-tiers.csv`, `comparison.csv` và manifest SHA-256 từ các receipt và kết quả đã có, rồi kiểm tra lại V2–5.000.
