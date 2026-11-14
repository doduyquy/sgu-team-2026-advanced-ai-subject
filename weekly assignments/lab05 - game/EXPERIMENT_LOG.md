# Lab 05 — Nhật ký thực nghiệm (Tic-Tac-Toe)

Phạm vi: phần 2, Bài 1–7. Các notebook Connect 4, Mean Connect 4 và Dots and Boxes thuộc phần mở rộng, không nằm trong phạm vi log này.

## Thiết lập

- Baseline: `random_player` đấu `random_player`, 1.000 ván; seed NumPy `20261112`.
- Bảng đấu: mỗi tác tử đấu `random_player` 100 ván, tác tử làm X; dùng cùng seed `20261112`. Thời gian là trung bình mỗi lần tác tử chọn nước, đo bằng `perf_counter`; số nút là trung bình mỗi lần gọi. Với Monte Carlo, ghi ngân sách playout thay cho số nút.
- Kết quả có tính ngẫu nhiên ở các playout và nước đi random. Thời gian phụ thuộc máy; các phép đo trong bảng là lần chạy của project này.

## Kết quả

### Bài 1 — Mô hình và baseline

Trạng thái là list 9 ô, đánh số 0–8 theo hàng; nước hợp lệ là chỉ số ô trống. `result` trả về bàn mới, `terminal`/`check_board` nhận diện X thắng, O thắng, hòa; `utility` trả +1/0/−1 theo góc nhìn người chơi truyền vào.

| X thắng | O thắng | Hòa | Tỷ lệ X thắng |
|---:|---:|---:|---:|
| 601 | 284 | 115 | 60,1% |

X có lợi thế đi trước trong baseline random (chênh 317 ván thắng). Đây là kết quả thực nghiệm của một seed, không phải bảo đảm cho mọi lần chạy.

### Bài 2 — AND-OR Search

Các ca kiểm thử trong notebook cho thấy: khi X có thể thắng, kế hoạch chọn nước 2 rồi chỉ định phản ứng tiếp theo cho từng nước của O; trong ca chỉ có thể hòa, kế hoạch rẽ nhánh theo phản ứng của đối thủ và kết thúc bằng thắng hoặc hòa. Ở bàn rỗng, nước đầu là 0 và kế hoạch điều kiện đảm bảo ít nhất kết quả hòa. Ca bất lợi có thể trả `None` nếu yêu cầu buộc phải thắng.

Với fallback random khi không có kế hoạch: 100 ván, X thắng 93, O thắng 0, hòa 7; trung bình 20,30 ms và 311 nút mỗi lần chọn nước.

### Bài 3 — Minimax, Alpha-Beta, move ordering

Trên cùng trạng thái, Minimax và Alpha-Beta giữ nguyên giá trị tối ưu; có thể chọn ô khác nhau khi nhiều nước đồng hạng. Ở bàn rỗng, Minimax duyệt 549.946 nút (8,61 s); Alpha-Beta không sắp thứ tự duyệt 18.297 nút (373 ms); sau move ordering còn 7.275 nút (200 ms), giảm khoảng 60,2% so với Alpha-Beta chưa sắp thứ tự. Giá trị vẫn là hòa (0). Thứ tự ưu tiên nước tốt giúp Alpha-Beta sớm có cận chặt để cắt các nhánh kém.

Trong loạt 100 ván dùng Alpha-Beta không move ordering: X thắng 100, O thắng 0, hòa 0; trung bình 404,57 ms và 5.501 nút mỗi nước. Thời gian cao hơn cho thấy tầm quan trọng của move ordering.

### Bài 4 — Heuristic Alpha-Beta và cutoff

| Trạng thái kiểm thử | Cấu hình | Nước đi | Giá trị | Nút | Thời gian |
|---|---|---:|---:|---:|---:|
| X có thể hòa | cutoff 2 | 7 | −0,4 | 21 | 8,81 ms |
| X có thể hòa | cutoff 4 | 7 | 0 | 81 | 26,9 ms |
| X có thể hòa | đầy đủ | 7 | 0 | 101 | 48,6 ms |
| Bàn rỗng | cutoff 2 | 0 | 0 | 26 | 9,28 ms |
| Bàn rỗng | cutoff 4 | 4 | 0 | 541 | 182 ms |
| Bàn rỗng | đầy đủ | 0 | 0 | 18.297 | 3,76 s |

Độ sâu lớn hơn tốn thêm nút/thời gian nhưng cải thiện đánh giá ở ca X có thể hòa (cutoff 2 đánh giá −0,4, sâu hơn nhận ra giá trị 0). Ở bàn rỗng, nhiều nước đồng giá trị nên nước được chọn có thể khác mà chất lượng tối ưu không đổi.

### Bài 5–6 — Monte Carlo và UCB1

| Tác tử (X) | Thắng | Hòa | Thua | ms/lần chọn | Ngân sách mô phỏng |
|---|---:|---:|---:|---:|---:|
| Pure MC, N=10 | 92 | 3 | 5 | 1,48 | 10 playout |
| Pure MC, N=100 | 95 | 5 | 0 | 37,03 | 100 playout |
| UCB1, N=10 | 95 | 2 | 3 | 1,42 | 10 playout |
| UCB1, N=100 | 98 | 2 | 0 | 26,82 | 100 playout |

Thêm mô phỏng thường làm kết quả ổn định hơn nhưng tăng thời gian; một lần chạy không chứng minh xu hướng thắng luôn tăng. Pure MC chia gần đều ngân sách cho các nước. UCB1 ưu tiên nước có điểm trung bình cao (exploitation), đồng thời cộng thưởng cho nước ít được thử (exploration); do đó phân bổ không đều. Trong notebook, UCB1 chọn trung tâm ở bàn rỗng và cấp phần lớn playout cho nước đó.

### Đối chiếu tác tử với Random

Các số dưới đây là loạt có seed cố định phía trên; kết quả tính theo vai X. PMCS chia `N` theo số nước còn trống bằng `floor(N / số_nước)`, nên tổng playout thực tế có thể thấp hơn N.

| Tác tử | Thắng–Hòa–Thua | ms/lần chọn | Nút hoặc playout/lần |
|---|---:|---:|---:|
| Random | 601–115–284 / 1.000 ván | — | — |
| AND-OR | 93–7–0 / 100 ván | 20,30 | 311 nút |
| Alpha-Beta | 100–0–0 / 100 ván | 404,57 | 5.501 nút |
| Heuristic Alpha-Beta, cutoff 2 | 99–1–0 / 100 ván | 2,79 | 22 nút |
| Heuristic Alpha-Beta, cutoff 4 | 98–2–0 / 100 ván | 50,00 | 225 nút |
| Pure MC, N=10 | 92–3–5 / 100 ván | 1,48 | tối đa 10 playout |
| Pure MC, N=100 | 95–5–0 / 100 ván | 37,03 | tối đa 100 playout |
| UCB1, N=10 | 95–2–3 / 100 ván | 1,42 | 10 playout |
| UCB1, N=100 | 98–2–0 / 100 ván | 26,82 | 100 playout |

## Bài 7 — Ván tương tác mẫu

Đã replay `interactive_player` bằng đầu vào giả lập ở cả hai vai X và O, với nước phản hồi cố định để kiểm tra luồng nhập. Transcript hợp lệ: `X0, O4, X1, O2, X3, O6`; O thắng đường chéo `2–4–6`, ván kết thúc ngay sau nước O6. Hai cell trong notebook vẫn cho phép chơi trực tiếp bằng bàn phím trong Jupyter.

## Nhận xét và câu hỏi tổng kết

1. AND-OR xem nước đối thủ là mọi kết quả có thể có và chỉ trả kế hoạch thắng/đạt hòa nếu tất cả nhánh phù hợp; Minimax gán lượt đối thủ là nút Min và tối đa hóa giá trị tối thiểu.
2. Alpha-Beta không đổi giá trị tối ưu của Minimax; nó chỉ bỏ qua nhánh không thể cải thiện quyết định. Move ordering làm việc cắt nhánh hiệu quả hơn.
3. Dùng heuristic và cutoff khi cây đầy đủ quá lớn hoặc độ trễ bị giới hạn; đánh đổi là đánh giá gần đúng và phụ thuộc chất lượng heuristic.
4. Pure MC không cần tự thiết kế heuristic nhưng cần nhiều rollout, có nhiễu và dễ bỏ sót nước chiến thuật khi ngân sách nhỏ.
5. UCB1 cân bằng khám phá và khai thác bằng thưởng bất định; nó dồn mô phỏng vào nước triển vọng thay vì chia đều.
6. Không gian tăng nhanh chủ yếu do số trạng thái hợp lệ và branching factor nhân qua độ sâu; số nước hợp lệ giảm dần trong Tic-Tac-Toe nhưng số chuỗi nước đi vẫn lớn hơn số trạng thái vì có nhiều thứ tự dẫn tới cùng bàn.
