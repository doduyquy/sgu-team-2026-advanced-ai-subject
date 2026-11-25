# Báo cáo phân tích 29 cột dữ liệu MapSuite V1

Ngày biên soạn: 12/10/2026. Ngôn ngữ: tiếng Việt.

Phạm vi: `datasets/mapsuite_v1/geometries.csv` và ba bảng `splits/train_geometries.csv`, `splits/validation_geometries.csv`, `splits/test_geometries.csv`.

Cơ sở bằng chứng: nội dung CSV, code và tài liệu đã đọc trực tiếp trong phiên phân tích ngày 08/10/2026. Công cụ shell gặp lỗi khởi tạo ngày biên soạn nên chưa xác nhận lại trạng thái repo ngày 12/10/2026. Các số liệu sau phản ánh lần kiểm tra trước; báo cáo không tuyên bố đã chạy lại mô phỏng.

## Ghi chú quan trọng cần đọc trước

> **1. Số làn có lỗi metadata:** `base_lane_num=2` là khai báo lịch sử; cấu hình thực tế của benchmark dùng **3 làn cơ sở**. Không ép cấu hình hai làn khi tái tạo bản đồ đã đóng băng.

> **2. 180 dòng TRAIN là 180 bản đồ:** mỗi episode tạo nhiều bước tương tác để học. Các cột metadata không tự động là đầu vào của agent.

> **3. Phân biệt seed:** geometry seed tạo hình học; environment seed tạo điều kiện môi trường trên hình học đã giữ cố định; agent/training seed điều khiển ngẫu nhiên của thuật toán.

> **4. 100 giây là mốc thống kê:** `episode_budget_seconds` không thay thế `horizon_steps` thực tế. Cột tốc độ trung bình cho horizon 1000 không phải tốc độ an toàn hay giới hạn tốc độ.

> **5. Số xe có hai lớp ý nghĩa:** `traffic_density` là tham số mô phỏng; `planned_traffic_vehicle_count` là số xe trong lần audit, gồm cả xe chờ kích hoạt. Cả hai không trực tiếp cho biết số xe agent đang nhìn thấy.

> **6. Sinh bản đồ thành công khác kết quả lái:** `generation_success=True` xác nhận bước sinh/thu thập dữ liệu thành công; không chứng minh agent đã đến đích hoặc tránh va chạm.

> **7. Giữ đúng tập và nguồn chuẩn:** dùng TRAIN để học, VALIDATION để chọn phương án, TEST để đánh giá cuối. Launcher dùng manifest chuẩn và hash; sửa CSV đóng gói không tự đổi benchmark.

> **8. Phạm vi bằng chứng:** báo cáo dựa trên dữ liệu/code đã đọc ngày 08/10/2026; chưa xác nhận lại repo ngày 12/10/2026 do lỗi công cụ. Hash phản ánh cấu hình geometry được tuần tự hóa, không bao phủ toàn bộ trạng thái mô phỏng.

## 1. Bản chất và đơn vị dữ liệu

Mỗi dòng trong bốn bảng trên mô tả một hình học bản đồ đường (geometry). Một dòng chứa định danh, phân tập, cấu trúc đường, số đo hình học, mật độ giao thông và thống kê tham chiếu. Đây là danh mục kịch bản dùng để dựng môi trường, không phải một mẫu quan sát của xe tại một thời điểm.

Ba đơn vị cần phân biệt:

| Đơn vị | Thành phần | Vai trò |
|---|---|---|
| Geometry | Hình học đường cố định; nhận diện bằng ID, sequence, geometry seed và hash | Quy định agent được học/đánh giá trên đường nào |
| Evaluation case | Geometry + environment seed + traffic density + horizon | Quy định một điều kiện chạy đánh giá cụ thể |
| Transition | Observation, action, reward, next observation, trạng thái kết thúc | Trải nghiệm phát sinh khi agent tương tác; thuật toán RL có thể dùng để cập nhật mô hình |

180 dòng TRAIN không phải 180 transition. Một geometry có thể chạy nhiều episode; mỗi episode tạo nhiều bước tương tác.

CSV không cung cấp hành động lái đúng, trajectory chuyên gia hay nhãn điều khiển. `tier`, `candidate_role`, `generation_success` cũng không phải nhãn hành động.

Trong hệ thống hiện có, launcher/workbench là nền tảng thực thi và đánh giá. Tài liệu phát triển agent yêu cầu xây dựng pipeline huấn luyện learned agent riêng, dùng TRAIN và xuất checkpoint để đánh giá. Danh mục TRAIN tồn tại không chứng minh rằng một mô hình đã được huấn luyện.

## 2. Dữ liệu được sinh từ đâu?

Luồng nguồn gốc đã đối chiếu:

```text
12 họ bản đồ × geometry seed 0..19
    ↓ MetaDrive tạo môi trường và hình học
scripts/audit_mapsuite_candidates.py
    ↓ đo tuyến đường, hộp bao, đếm block, ghi giao thông
results/audits/mapsuite/candidate_metrics.csv
    + canonical_candidates.json
    ↓ phân tập và xác định hash hình học
src/platform/protocol.py
results/audits/evaluation_protocol/geometry_split_manifest.csv
    ↓ ghép thông tin theo định danh ứng viên
scripts/export_mapsuite_v1_dataset.py
    ↓
datasets/mapsuite_v1/geometries.csv
    ↓ lọc split
train_geometries.csv / validation_geometries.csv / test_geometries.csv
```

Hai nhóm nguồn có vai trò khác nhau:

- Manifest phân tập cung cấp định danh chuẩn, split, tier, sequence, geometry seed, candidate role, hash, nguồn hash, block IDs, chiều dài tuyến và density.
- Bảng đo ứng viên cung cấp trạng thái sinh, số block, kích thước hộp bao, metadata làn, các bộ đếm block, điểm phân nhánh, số xe dự kiến và chỉ số ngân sách 100 giây.

Exporter kiểm tra sự nhất quán khi ghép dữ liệu; không có nghĩa mọi trường đều được đo mới trong lúc xuất package.

Các file trong `datasets/mapsuite_v1` là bản đóng gói dẫn xuất. Launcher hiện đọc các manifest chuẩn tại `results/audits/evaluation_protocol` và kiểm tra hash khóa của chúng. Sửa bản CSV đóng gói không mặc nhiên sửa dữ liệu mà launcher thực thi.

Nguồn: [script audit](../scripts/audit_mapsuite_candidates.py), [script export](../scripts/export_mapsuite_v1_dataset.py), [protocol](../src/platform/protocol.py), [launcher cases](../src/launcher/cases.py).

## 3. Cách tổ chức train/validation/test

| Tập | Số geometry | Mỗi tier | Mỗi sequence | Mục đích |
|---|---:|---:|---:|---|
| TRAIN | 180 | 45 | 15 | Huấn luyện, phát triển thuật toán |
| VALIDATION | 48 | 12 | 4 | Chọn tham số, checkpoint, phương án |
| TEST | 12 | 3 | 1 | Đánh giá cuối sau khi chốt phương pháp |
| Tổng | 240 | 60 | 20 | 4 tier, 12 sequence |

Mỗi sequence có 20 geometry seed. Một geometry chuẩn đã chọn được dành cho TEST. Với 19 geometry còn lại, code tính SHA-256 từ salt, tier, sequence, seed rồi sắp theo hash; 15 geometry đầu thuộc TRAIN, 4 geometry còn lại thuộc VALIDATION.

Ví dụ họ `SCS`: TEST có seed 11; VALIDATION có seed 6, 9, 10, 13; 15 seed khác thuộc TRAIN. Không có quy tắc “seed 0..14 luôn là train”.

Lần kiểm tra trước xác nhận không giao nhau về geometry hash giữa ba tập. Tuy vậy, các tập chia sẻ cùng 12 họ cấu trúc. Kết quả phản ánh khả năng tổng quát hóa sang geometry giữ riêng trong các họ này; không tự chứng minh khả năng lái trên loại đường hoàn toàn mới hoặc đường thực tế.

Validation có 96 case = 48 geometry × 2 environment seed (5101, 5102). Test có 60 case = 12 geometry × 5 environment seed (9101..9105). Số episode thực thi còn phụ thuộc số replicate. Không dùng TEST để cập nhật mô hình hoặc chọn tham số/checkpoint.

## 4. Từ điển chi tiết 29 cột

Kiểu dữ liệu dưới đây là kiểu ngữ nghĩa. Trong CSV, dữ liệu được lưu bằng văn bản; chương trình đọc phải chuyển kiểu phù hợp.

### 01. geometry_id

**Ý nghĩa và kiểu:** Chuỗi định danh một geometry, ví dụ `geom_easy_SCS_seed0`.

**Nguồn:** Hàm `assign_geometry_splits()` ghép tên tier viết thường, sequence và geometry seed theo mẫu `geom_<tier>_<sequence>_seed<seed>`.

**Cách đọc:** Ví dụ trên chỉ biến thể seed 0 của họ SCS thuộc tier Easy. “0” không phải số lượng block hoặc độ khó.

**Tác dụng:** Tra cứu bản đồ, liên kết bảng, lưu log, lập lịch episode, truy lại một kết quả tới đúng geometry. Quy tắc sinh seed môi trường cho training có thể dùng geometry_id làm một thành phần đầu vào.

**Giới hạn:** ID phản ánh tên cấu hình; bản thân tên không chứng minh nội dung hình học không đổi. Kiểm tra hình học còn cần hash và engine đúng phiên bản. Không dùng chuỗi ID như tín hiệu để agent học thuộc bản đồ.

### 02. split

**Ý nghĩa và kiểu:** Biến phân loại với ba giá trị `TRAIN`, `VALIDATION`, `TEST`.

**Nguồn:** `assign_geometry_splits()` áp dụng quy tắc giữ các canonical geometry cho test và phân chia có salt như phần 3; exporter chép từ manifest.

**Cách đọc:** `TRAIN` nghĩa là geometry được phép dùng trong phát triển/huấn luyện. `VALIDATION` dùng để lựa chọn phương án. `TEST` dành cho đánh giá cuối.

**Tác dụng:** Kiểm soát sử dụng dữ liệu, tránh rò rỉ tập đánh giá. Launcher sandbox kiểm tra tập được phép dùng.

**Giới hạn:** Đây là quyền sử dụng trong thiết kế thí nghiệm, không phải đặc tính vật lý. Trong từng file split, cột có cùng một giá trị nhưng vẫn hữu ích khi ghép bảng hoặc kiểm tra dữ liệu.

### 03. tier

**Ý nghĩa và kiểu:** Nhóm độ khó thiết kế: Easy, Medium, Hard, Extreme.

**Nguồn:** Nhóm ứng viên của MapSuite được định nghĩa trước, gắn với các sequence và traffic density; được truyền qua audit, manifest và package.

**Cách đọc:** Easy gồm đường thẳng/cua và density 0; Medium thêm cấu trúc giao lộ, density 0.08; Hard phối hợp nhiều cấu trúc, density 0.15; Extreme phối hợp phức tạp hơn, density 0.25.

**Tác dụng:** Báo cáo kết quả theo mức, cân bằng dữ liệu và xây dựng curriculum nếu pipeline nghiên cứu lựa chọn cách đó. Chỉ có cột tier không làm curriculum tự xảy ra.

**Giới hạn:** Tier không phải độ khó được đo bằng một công thức duy nhất và không đảm bảo mọi geometry Hard đều khó hơn mọi geometry Medium với mọi thuật toán. Cấu trúc đường và density cùng thay đổi theo tier; so sánh tier không tách riêng được tác động của từng yếu tố.

### 04. sequence

**Ý nghĩa và kiểu:** Chuỗi ký hiệu cấu trúc đường, ví dụ `SCXCS`.

**Nguồn:** Danh sách họ bản đồ dùng trong đợt audit. Chuỗi được truyền qua cấu hình `map=<sequence>` cho MetaDrive.

**Cách đọc:** S = thẳng; C = cua; X = ngã tư; T = ngã ba; O = vòng xuyến; r/R = đường dẫn nhập/ra; y/Y = giảm/tăng số làn. Hoa/thường có ý nghĩa khác nhau. SCXCS là thẳng → cua → ngã tư → cua → thẳng.

**Tác dụng:** Chỉ định topology cấp block, tái tạo bản đồ cùng seed và engine, phân nhóm thí nghiệm theo loại cấu trúc.

**Giới hạn:** Không chứa đầy đủ độ dài, góc cua, bán kính hoặc tọa độ. Cùng sequence có thể có nhiều geometry. Block khởi đầu I không nằm trong chuỗi yêu cầu này.

### 05. geometry_generation_seed

**Ý nghĩa và kiểu:** Số nguyên từ 0 đến 19 trong MapSuite V1.

**Nguồn:** Vòng quét seed của script audit; dùng khi reset/generate để lấy biến thể hình học. Được giữ lại trong manifest và ID.

**Cách đọc:** Seed 13 là đầu vào xác định trạng thái bộ sinh ngẫu nhiên, không phải góc 13 độ hay điểm khó 13.

**Tác dụng:** Tái tạo biến thể, đa dạng hóa hình học và định danh geometry khi kết hợp sequence. Giúp chia dữ liệu có thể tái lập.

**Giới hạn:** Seed đơn lẻ không nhận diện duy nhất bản đồ vì mỗi sequence dùng lại 0..19. Hai seed gần nhau không có nghĩa bản đồ gần giống nhau. Khả năng tái tạo phụ thuộc phiên bản engine và cấu hình. Geometry seed không được thay bằng agent seed hay environment seed.

### 06. candidate_role

**Ý nghĩa và kiểu:** Nhãn vai trò trong benchmark.

| Giá trị | Ý nghĩa | Phân bố đã kiểm tra |
|---|---|---|
| pool_candidate | Geometry trong pool thông thường | TRAIN và VALIDATION |
| primary_canonical | Geometry chuẩn chính của tier | TEST; một geometry mỗi tier |
| alternate_canonical | Geometry chuẩn bổ sung | TEST; hai geometry mỗi tier |

**Nguồn:** Các geometry được chọn trong canonical candidates manifest nhận vai trò chính/bổ sung; các geometry còn lại được `assign_geometry_splits()` gán pool_candidate.

**Tác dụng:** Truy xuất vai trò mẫu đại diện và lịch sử chọn benchmark, liên kết audit với dữ liệu.

**Giới hạn:** Alternate không có nghĩa chất lượng thấp hơn hoặc có thể tùy ý bỏ khỏi đánh giá. Primary không tự tạo trọng số lớn hơn nếu protocol không quy định. Đây không phải nhãn dự đoán.

### 07. geometry_sha256

**Ý nghĩa và kiểu:** Chuỗi hexadecimal 64 ký tự biểu diễn SHA-256, ví dụ bắt đầu `3bbbf995...`.

**Nguồn:** Với cấu hình exact đã lưu, code băm `exact_block_sequence`. Với đường tái sinh, launcher lấy `current_map.get_meta_data()["block_sequence"]`, chuyển các kiểu dữ liệu về dạng JSON rồi băm bằng `canonical_json_sha256()`. Hàm canonical hóa sắp khóa, dùng dấu phân cách cố định và không cho NaN.

**Cách đọc:** Hash là dấu vân tay của cấu hình chuỗi block được tuần tự hóa. Cần gọi đúng đối tượng được băm; không diễn giải thành hash của toàn bộ trạng thái mô phỏng, mọi xe giao thông hoặc toàn bộ tọa độ vật lý.

**Tác dụng:** Xác nhận tái tạo đúng cấu hình hình học đóng băng, phát hiện geometry drift, kiểm tra giao nhau giữa các tập, gắn kết quả với geometry.

**Giới hạn:** Không phải số đo độ khó, không có đơn vị; khoảng cách giữa hai hash không đo độ giống bản đồ. Hash giống nhau không chứng minh traffic, reward, input profile và toàn bộ môi trường đều giống nhau. Không dùng làm feature điều khiển.

### 08. geometry_hash_source

**Ý nghĩa và kiểu:** Chuỗi mô tả nguồn bằng chứng của geometry hash.

- `gate2_stored_exact`: hash được lấy từ cấu hình block exact đã lưu trong canonical audit.
- `pinned_regeneration`: hash từ việc tái sinh với engine được cố định phiên bản.

**Nguồn:** `assign_geometry_splits()` chọn nhánh nguồn tương ứng rồi exporter ghi ra package.

**Tác dụng:** Truy nguồn khi kiểm tra sai lệch, giải thích cách một geometry có hash chuẩn.

**Giới hạn:** Không phải hai loại bản đồ có độ khó khác nhau. Code còn có nhánh dự phòng `fallback_identity` khi thiếu bằng chứng, băm chuỗi nhận diện thay vì exact block; không được đánh đồng với hash cấu hình thật. Nhánh này là khả năng của code, không phải khẳng định nó xuất hiện trong package đang phân tích.

### 09. generation_success

**Ý nghĩa và kiểu:** Boolean được biểu diễn bằng văn bản True/False.

**Nguồn:** Script audit chạy reset, lấy map và tính metrics trong khối xử lý lỗi. Nhánh thành công ghi True; nhánh lỗi ghi False cùng một số giá trị thay thế.

**Cách đọc:** True cho biết đợt sinh/thu thập ứng viên đó đã hoàn tất theo script. Nó không nói agent đã lái về đích, không va chạm hay tuyến đã được chứng minh dễ đi.

**Tác dụng:** Phân biệt ứng viên có metrics sinh thành công với ứng viên lỗi, kiểm soát chất lượng danh mục.

**Giới hạn:** Khi làm việc với bảng audit rộng hơn, giá trị 0 ở các trường đo có thể là giá trị thay thế do lỗi, không nhất thiết là kết quả vật lý thật. Không ép Boolean bằng `bool("False")` trong Python vì kết quả là True; cần chuyển kiểu tường minh.

### 10. block_ids

**Ý nghĩa và kiểu:** Chuỗi các ID block thực tế trong bản đồ, ví dụ ISCS.

**Nguồn:** Script đọc `[b.ID for b in env.current_map.blocks]`; manifest/package lưu dạng chuỗi.

**Cách đọc:** I là block khởi đầu được thêm ngầm. `sequence=SCS` cho cấu trúc thực tế ISCS: I, S, C, S.

**Tác dụng:** Kiểm tra cấu trúc được dựng, tính các bộ đếm block, đối chiếu sequence với kết quả thực tế.

**Giới hạn:** Đây vẫn chỉ là loại block; không thay cho toàn bộ cấu hình hình học. Một ký tự không tương đương một đoạn đường có chiều dài cố định hoặc một đoạn centerline duy nhất.

### 11. block_count

**Ý nghĩa và kiểu:** Số nguyên đếm tổng block thực tế.

**Nguồn/cách tính:** Độ dài danh sách block IDs của map; bao gồm I.

**Ví dụ:** ISCS có 4 block; ISCXCCS có 7 block.

**Tác dụng:** Chỉ số quy mô cấu trúc, kiểm tra nhất quán và nhóm geometry theo độ dài chuỗi.

**Giới hạn:** Không phải số làn, số giao lộ hoặc số đoạn lane nội bộ. Hai bản đồ bằng block_count vẫn có thể khác nhiều về chiều dài và khó khăn. Không suy chiều dài đường bằng cách nhân số block với một chiều dài cố định.

### 12. route_length_m

**Ý nghĩa và kiểu:** Số thực, đơn vị mét; chiều dài tuyến tham chiếu navigation.

**Nguồn:** Đợt audit đọc `float(env.agent.navigation.total_length)`. Exporter dùng giá trị trong geometry split manifest; bước tạo manifest có thể lấy số đo tái sinh độ chính xác đầy đủ.

**Ví dụ:** `407.89441610698384` nghĩa là tuyến tham chiếu dài khoảng 407.89 m.

**Tác dụng:** Tính horizon theo tuyến, tính yêu cầu tốc độ tham chiếu, so sánh quy mô nhiệm vụ.

**Giới hạn:** Không phải khoảng cách thẳng từ điểm đầu tới đích; không phải tổng chiều dài mọi nhánh của mạng đường; không phải quãng đường agent thực tế đã chạy. Xe đi vòng hoặc lệch làn có thể chạy quãng đường khác. Độ dài CSV nhiều chữ số không đồng nghĩa độ chính xác đo ngoài đời tương ứng.

### 13. bbox_width_m

**Ý nghĩa và kiểu:** Số thực mét; độ trải rộng hộp bao mạng đường theo trục x.

**Nguồn/cách tính:** `m.road_network.get_bounding_box()`; script tính `round(b_box[1] - b_box[0], 2)`.

**Ví dụ:** 192.22 nghĩa là miền chứa mạng đường trải rộng khoảng 192.22 m theo trục này.

**Tác dụng:** Đánh giá không gian bản đồ chiếm dụng, đặt khung hình preview, so sánh hình dạng tổng thể.

**Giới hạn:** Không phải bề rộng mặt đường hay làn. Đây là hộp bao theo trục tọa độ; hướng xoay bản đồ ảnh hưởng chỉ số. Nhân width với height chỉ cho diện tích hộp bao, không phải diện tích mặt đường.

### 14. bbox_height_m

**Ý nghĩa và kiểu:** Số thực mét; độ trải rộng theo trục y trên mặt phẳng bản đồ.

**Nguồn/cách tính:** Cùng bounding box, script tính `round(b_box[3] - b_box[2], 2)`.

**Ví dụ:** 246.46 là độ trải rộng 246.46 m theo trục y.

**Tác dụng:** Kết hợp bbox_width_m mô tả phạm vi không gian, hỗ trợ scale/khung nhìn.

**Giới hạn:** “Height” là chiều cao trên hình chiếu mặt bằng; không phải cao độ địa hình, độ cao cầu hay độ dốc.

### 15. lane_width_m

**Ý nghĩa và kiểu:** Số thực mét; bề rộng làn cơ sở được mô tả trong dataset, hiện ghi 3.5.

**Nguồn:** Audit ghi literal `lane_width=3.5`; exporter đổi tên thành lane_width_m. Tài liệu erratum xác nhận bề rộng làn của cấu hình thực tế là 3.5 m và không randomize lane width.

**Tác dụng:** Mô tả kích thước làn, giải thích khoảng cách ngang, hỗ trợ đối chiếu cấu hình.

**Giới hạn:** Nguồn ghi của cột là metadata khai báo, không phải phép đo lại mọi làn. Không nhân tùy ý với base_lane_num để kết luận bề rộng mọi giao lộ/ramp. Khi điều khiển, dùng thông tin làn hiện tại trong đầu vào được phép của agent.

### 16. base_lane_num

**Ý nghĩa và kiểu:** Số nguyên khai báo số làn cơ sở. CSV ghi 2 trên 240 dòng đã kiểm tra.

**Nguồn:** Script audit ghi cứng 2 vào metrics nhưng lúc dựng môi trường dùng `map=<sequence>` và không áp dụng `map_config.lane_num=2`. Exporter kế thừa giá trị này.

**Ý nghĩa đúng trong bản V1:** Theo erratum, declared_base_lane_num = 2, effective_base_lane_num = 3. MetaDrive được ghim phiên bản giữ mặc định 3 trong đường dựng này. Báo cáo kiểm chứng preview trong repo ghi nhận 240/240 geometry hash khớp với đường tái tạo thực tế ba làn cơ sở.

**Tác dụng:** Ghi lại mô tả cấu hình lịch sử và xác định lỗi metadata. Khi mô tả benchmark phải nói rõ khai báo và cấu hình hiệu lực.

**Giới hạn:** Không diễn giải “2” thành bằng chứng bản đồ thực tế có hai làn. Không ép lane_num=2 khi tái tạo benchmark vì có thể làm đổi hình học/hash. Số làn cơ sở cũng không khẳng định mọi đoạn ramp/merge đều có cùng số làn.

Nguồn bắt buộc đọc kèm: [lane metadata erratum](environment/MAPSUITE_V1_LANE_METADATA_ERRATUM.md).

### 17. straight_count

**Ý nghĩa và kiểu:** Số nguyên đếm block thẳng, tính cả block khởi đầu.

**Nguồn/cách tính:** `block_ids.count("S") + block_ids.count("I")`.

**Ví dụ:** ISCS có hai S và một I nên straight_count = 3.

**Tác dụng:** Mô tả thành phần đường thẳng, kiểm tra cấu trúc và phân tích lỗi theo dạng đường.

**Giới hạn:** Không phải số mét đường thẳng. Block phức tạp có thể chứa lane thẳng nội bộ nhưng không được cộng thêm vào bộ đếm này.

### 18. curve_count

**Ý nghĩa và kiểu:** Số nguyên đếm block C.

**Nguồn/cách tính:** `block_ids.count("C")`.

**Ví dụ:** ISCXCCS có ba C nên curve_count = 3.

**Tác dụng:** Mô tả lượng block cua; có thể dùng trong phân tích hậu nghiệm về yêu cầu đánh lái.

**Giới hạn:** Không chứa bán kính, góc cua, chiều rẽ hay tốc độ an toàn. Các đường cong nội bộ của vòng xuyến/giao lộ không được đếm thành C.

### 19. intersection_count

**Ý nghĩa và kiểu:** Số nguyên đếm block ngã tư X.

**Nguồn/cách tính:** `block_ids.count("X")`.

**Ví dụ:** ISCXCCS có intersection_count = 1.

**Tác dụng:** Thống kê cấu trúc ngã tư và đóng góp vào decision_block_count, branching_choice_score.

**Giới hạn:** Tên intersection_count trong bộ này chỉ đếm X; ngã ba được tách riêng. Một ngã tư không có nghĩa agent được tùy chọn đích; nhiệm vụ navigation vẫn có tuyến được chỉ định. Cột không mô tả đèn tín hiệu hay quyền ưu tiên.

### 20. t_intersection_count

**Ý nghĩa và kiểu:** Số nguyên đếm block ngã ba T.

**Nguồn/cách tính:** `block_ids.count("T")`.

**Ví dụ:** Chuỗi có một T thì trường bằng 1.

**Tác dụng:** Tách ngã ba khỏi ngã tư để mô tả topology và tính điểm cấu trúc.

**Giới hạn:** Không cho biết hướng vào, hướng ra yêu cầu hoặc mật độ xe tại ngã ba. Giá trị 0 chỉ nói không có block T, không nói toàn bộ bản đồ không có điểm nhập/ra.

### 21. roundabout_count

**Ý nghĩa và kiểu:** Số nguyên đếm block vòng xuyến O.

**Nguồn/cách tính:** `block_ids.count("O")`.

**Ví dụ:** SCXOCS có một O.

**Tác dụng:** Nhận diện nhiệm vụ có vòng xuyến, hỗ trợ phân tích xử lý đường cong và tương tác giao thông tại đây.

**Giới hạn:** Không phải số vòng xe cần chạy; không mã hóa lối ra cần chọn, bán kính hoặc số làn của vòng xuyến.

### 22. ramp_count

**Ý nghĩa và kiểu:** Số nguyên tổng block đường dẫn nhập và ra.

**Nguồn/cách tính:** `block_ids.count("r") + block_ids.count("R")`.

**Ví dụ:** CrXROSTR có một r và hai R nên ramp_count = 3.

**Tác dụng:** Mô tả số cấu trúc ramp, nhóm thí nghiệm có nhập/tách khỏi đường chính.

**Giới hạn:** Trường tổng không phân biệt ramp nhập với ramp ra. Muốn biết từng loại phải đọc sequence/block_ids. Không phải số xe đang nhập làn hay số lần agent chắc chắn đổi làn.

### 23. merge_split_count

**Ý nghĩa và kiểu:** Số nguyên tổng block giảm/tăng số làn y và Y.

**Nguồn/cách tính:** `block_ids.count("y") + block_ids.count("Y")`.

**Ví dụ:** SCXOCrTYCS có một Y, trường bằng 1; SCTXORyCCS có một y, trường cũng bằng 1.

**Tác dụng:** Mô tả các đoạn thay đổi số làn, hỗ trợ phân tích hành vi tại nút thắt.

**Giới hạn:** Hai ví dụ có cùng giá trị nhưng hướng thay đổi làn ngược nhau. Không phải số làn tăng/giảm, không tính mọi cấu trúc nhập xe trong ramp.

### 24. decision_block_count

**Ý nghĩa và kiểu:** Số nguyên đếm block thuộc tập X, T, O, r, R, y, Y.

**Nguồn/cách tính:** Hàm `count_decision_blocks()` cộng 1 cho mỗi block thuộc tập trên. Trong vocabulary hiện có:
`intersection_count + t_intersection_count + roundabout_count + ramp_count + merge_split_count`.

**Ví dụ:** CrXROSTR chứa X, O, T và ba ramp; decision_block_count = 6.

**Tác dụng:** Chỉ số đơn giản về số cấu trúc có phân nhánh hoặc yêu cầu xử lý maneuver, dùng mô tả benchmark.

**Giới hạn:** Đây là quy ước đếm trong code; không phải số quyết định thực tế mà agent đã đưa ra. Cua C vẫn yêu cầu điều khiển nhưng không được tính. Merge/split y/Y có được tính, dù phần mô tả schema ngắn có thể không liệt kê hết.

### 25. branching_choice_score

**Ý nghĩa và kiểu:** Điểm số nguyên không có đơn vị, dùng trọng số topology.

**Nguồn/cách tính:** Hàm `compute_branching_choice_score()`:
`3×X + 2×T + 3×O + 1×(r+R) + 1×(y+Y)`.

**Ví dụ:** CrXROSTR có 1 X, 1 T, 1 O, 3 ramp, 0 y/Y; điểm = 3 + 2 + 3 + 3 = 11. Geometry có một X và không có decision block khác có điểm 3.

**Tác dụng:** Tổng hợp cấu trúc thành một chỉ số dễ so sánh, phân nhóm và trình bày bảng thống kê.

**Giới hạn:** Trọng số là quy ước thiết kế, không phải đo trực tiếp mọi đường đi hợp lệ. Không phải tích số lộ trình, số action, xác suất lỗi hay reward. Điểm cao hơn không đảm bảo khó hơn với mọi policy.

### 26. traffic_density

**Ý nghĩa và kiểu:** Tham số số thực không có đơn vị vật lý được khai báo trong bảng.

| Tier | Density |
|---|---:|
| Easy | 0.00 |
| Medium | 0.08 |
| Hard | 0.15 |
| Extreme | 0.25 |

**Nguồn:** Cấu hình ứng viên theo tier; được truyền vào MetaDrive. Exporter dùng giá trị từ split manifest và kiểm tra khớp bảng metrics.

**Cách đọc:** 0.08 là tham số tạo giao thông của simulator. Nó không trực tiếp có nghĩa 8 xe, 8% diện tích đường bị chiếm, 8 xe/km hoặc xác suất va chạm 8%.

**Tác dụng:** Điều chỉnh mức giao thông trong ca chạy và bảo đảm các thuật toán được so trên cùng cấu hình.

**Giới hạn:** Số xe cụ thể còn phụ thuộc geometry và logic traffic manager. Không có công thức chuyển density thành số xe cố định chỉ từ bảng này. Báo cáo chưa kiểm tra lại mã nội bộ traffic manager nên không tự gán công thức chi tiết. Density bằng 0 không loại bỏ nguy cơ xe tự đâm lề hoặc đi ra đường.

### 27. planned_traffic_vehicle_count

**Ý nghĩa và kiểu:** Số nguyên đếm xe nền đã lên kế hoạch trong lần audit.

**Nguồn/cách tính:** Script lấy traffic manager:
`len(tm.traffic_vehicles) + sum(len(bv.vehicles) for bv in tm.block_triggered_vehicles)`.

**Cách đọc:** Gồm xe đang được traffic manager quản lý như xe hoạt động và xe được giữ để kích hoạt theo block. Ví dụ SCXCCS seed 13 có giá trị 12 trong bảng.

**Tác dụng:** Giúp người đọc hiểu quy mô giao thông thực tế được audit tạo ra dưới density tương ứng; so sánh các geometry có cùng density nhưng chiều dài/cấu trúc khác nhau.

**Giới hạn:** Không phải 12 xe luôn đồng thời ở cạnh ego, không phải số xe agent nhìn thấy, không gồm ego theo công thức này. Đây là thống kê của lần audit gốc, không phải cam kết số xe y hệt cho mọi environment seed của evaluation case. Không dùng nó để thay thế traffic_context tại runtime.

### 28. episode_budget_seconds

**Ý nghĩa và kiểu:** Số thực giây; mốc ngân sách tham chiếu của đợt audit.

**Nguồn/cách tính:** Script gán `episode_budget_s = 100.0`, từ horizon tham chiếu 1000 bước ở tần số điều khiển 10 Hz.

**Cách đọc:** 100.0 nghĩa là giả định có 100 giây mô phỏng để tính chỉ số tốc độ ở cột kế tiếp.

**Tác dụng:** Tạo một mốc chung để so chiều dài tuyến dưới cùng thời lượng; giải thích nguồn gốc required_avg_speed.

**Giới hạn:** Không phải giới hạn episode thực tế của mọi ca Platform V1. Runtime evaluation dùng horizon_steps theo route length. Cũng không phải 100 giây thời gian đồng hồ trên máy; tốc độ mô phỏng và rendering có thể khác thời gian mô phỏng.

### 29. required_avg_speed_kmh_for_horizon_1000

**Ý nghĩa và kiểu:** Số thực km/h, được làm tròn hai chữ số thập phân.

**Nguồn/cách tính:** Script audit:
`round(route_len / 100.0 * 3.6, 2)`.
route_len ở đây là chiều dài navigation được đo trong audit; cột route_length_m trong package đến từ manifest tái sinh nên cần lưu ý nguồn và độ chính xác khi đối chiếu sát ranh làm tròn.

**Ví dụ:** Tuyến 407.894416 m: 407.894416 / 100 × 3.6 ≈ 14.68 km/h. Tuyến 688.370129 m: ≈ 24.78 km/h.

**Tác dụng:** Ước lượng yêu cầu thời gian của một horizon cố định 1000 bước và phát hiện việc dùng một horizon chung có thể gây bất lợi cho đường dài.

**Giới hạn:** Đây là mức trung bình lý thuyết trên toàn tuyến, không phải tốc độ hiện tại, tốc độ giới hạn, tốc độ an toàn ở cua, hay giá trị mục tiêu bắt buộc cho policy. Dừng tránh xe làm giảm trung bình; đi vòng làm quãng đường thực tế khác. Runtime hiện dùng horizon route-aware nên không lấy cột này làm chuẩn đánh giá timeout thực tế.

## 5. Đọc trọn một dòng TRAIN

Dòng đã đọc từ train_geometries.csv:

| Cột | Giá trị |
|---|---|
| geometry_id | geom_easy_SCS_seed0 |
| split | TRAIN |
| tier | Easy |
| sequence | SCS |
| geometry_generation_seed | 0 |
| candidate_role | pool_candidate |
| geometry_sha256 | 3bbbf995e00279527132ce1594ffe36e8b0236c47afdbec7fdb6bb287e9687d8 |
| geometry_hash_source | pinned_regeneration |
| generation_success | True |
| block_ids | ISCS |
| block_count | 4 |
| route_length_m | 407.89441610698384 |
| bbox_width_m | 192.22 |
| bbox_height_m | 246.46 |
| lane_width_m | 3.5 |
| base_lane_num | 2 |
| straight_count | 3 |
| curve_count | 1 |
| intersection_count | 0 |
| t_intersection_count | 0 |
| roundabout_count | 0 |
| ramp_count | 0 |
| merge_split_count | 0 |
| decision_block_count | 0 |
| branching_choice_score | 0 |
| traffic_density | 0.0 |
| planned_traffic_vehicle_count | 0 |
| episode_budget_seconds | 100.0 |
| required_avg_speed_kmh_for_horizon_1000 | 14.68 |

Diễn giải: Đây là bản đồ được phép huấn luyện của họ SCS, biến thể seed 0. Cấu trúc thực tế gồm block đầu I cùng S, C, S, nên có bốn block: ba block thẳng và một block cua. Tuyến tham chiếu dài khoảng 407.89 m, nằm trong hộp bao 192.22 × 246.46 m. Không có block thuộc nhóm decision nên hai chỉ số decision/branching bằng 0. Density và số xe nền audit bằng 0. Nếu lấy mốc 100 giây thì tốc độ trung bình lý thuyết là 14.68 km/h.

Giá trị generation_success chỉ xác nhận sinh ứng viên thành công. Giá trị base_lane_num phải đọc cùng erratum: bảng khai báo 2 nhưng cấu hình hiệu lực là 3. Không có cột nào trong dòng này nói agent đã chạy, đạt reward bao nhiêu hay hoàn thành nhiệm vụ.

## 6. Nối geometry với một ca TEST

Geometry `geom_medium_SCXCCS_seed13` thuộc TEST, vai trò alternate_canonical, có block IDs ISCXCCS, 7 block, 3 block thẳng, 3 block cua, 1 ngã tư. Decision count = 1, branching score = 3, route length ≈ 688.370129 m, density = 0.08, planned traffic audit = 12.

Một ca tương ứng là `test/Medium/SCXCCS/geom-13/env-9103`. Case thêm environment_seed 9103 và horizon_steps 2066. Bản đồ giống nhau có thể xuất hiện trong năm ca test với năm seed môi trường.

Horizon runtime được tính bởi `compute_route_aware_horizon()`:

```text
reference_speed = 18 km/h = 5 m/s
safety_margin = 1.5
control_frequency = 10 Hz
nominal_steps = ceil(route_length_m / 5 × 1.5 × 10)
horizon_steps = max(1000, min(4000, nominal_steps))
```

Với 688.370129 m, kết quả 2066 bước = 206.6 giây mô phỏng. Cột episode_budget_seconds trong bảng geometry vẫn là 100 vì đó là mốc thống kê cũ; hai trường không thay thế nhau.

## 7. Các cột có vai trò gì đối với mô hình?

| Nhóm | Cột | Vai trò chủ yếu |
|---|---|---|
| Quản lý và truy nguồn | geometry_id, split, candidate_role, geometry_sha256, geometry_hash_source, generation_success | Kiểm soát thí nghiệm, kiểm tra dữ liệu, tái lập |
| Mô tả/dựng kịch bản | tier, sequence, geometry_generation_seed, traffic_density, metadata làn | Chọn điều kiện môi trường, đối chiếu cấu hình |
| Mô tả hình học | block_ids, block_count, route_length_m, bbox và các bộ đếm/điểm cấu trúc | Phân tích độ phức tạp, báo cáo, một phần tính budget |
| Thống kê tham chiếu | planned_traffic_vehicle_count, episode_budget_seconds, required_avg_speed... | Giải thích đợt audit và yêu cầu nhiệm vụ |

Phân nhóm thể hiện công dụng ngữ nghĩa, không khẳng định runtime trực tiếp đọc mọi cột. Ví dụ trường làn lịch sử không được phép lấn át cấu hình geometry đóng băng.

Agent runtime có hợp đồng đầu vào riêng: core observation 259 giá trị, cùng traffic/task context tùy information profile. Toàn bộ 29 cột không được tự động đưa vào policy. Thêm split, ID, hash, toàn tuyến hoặc thống kê toàn bản đồ vào agent có thể tạo đường ghi nhớ dữ liệu hoặc thay đổi information profile so với benchmark.

Sử dụng metadata để phân tích hậu nghiệm khác với cấp metadata làm quan sát điều khiển. Nếu nghiên cứu cần đầu vào bổ sung, phải xác định rõ profile và khả năng so sánh kết quả.

## 8. Quan hệ giữa các trường và cách kiểm tra

Các quan hệ có thể kiểm tra từ bảng, với vocabulary hiện tại:

```text
block_count =
    straight_count + curve_count + intersection_count
    + t_intersection_count + roundabout_count
    + ramp_count + merge_split_count

decision_block_count =
    intersection_count + t_intersection_count + roundabout_count
    + ramp_count + merge_split_count

branching_choice_score =
    3×intersection_count + 2×t_intersection_count + 3×roundabout_count
    + ramp_count + merge_split_count

required_avg_speed... ≈ round(route_length_m / 100 × 3.6, 2)
```

Quan hệ cuối cần xét việc hai trường lấy từ hai tầng nguồn và độ chính xác đo. Sai khác phải được truy nguồn, không sửa giá trị để ép khớp.

Các bộ đếm phần lớn suy ra từ block_ids; branching score suy ra từ bộ đếm; required speed suy ra từ chiều dài và budget. Vì vậy 29 cột không phải 29 nguồn thông tin độc lập. Nếu phân tích tương quan, cần nhận ra các quan hệ định nghĩa này.

Từ thống kê bảng không thể kết luận quan hệ nhân quả giữa một trường và hiệu suất agent. Ví dụ tier, density và loại block cùng thay đổi; cần thí nghiệm kiểm soát mới tách được từng tác động.

## 9. Những điểm cần nêu rõ khi viết báo cáo nghiên cứu

1. Đơn vị chia tập là geometry; đơn vị đánh giá là case; đơn vị trải nghiệm RL là transition.
2. Cùng sequence nhưng khác geometry seed là khác biến thể; cùng geometry nhưng khác environment seed là khác điều kiện mô phỏng.
3. Train/validation/test tách geometry nhưng dùng chung họ cấu trúc.
4. base_lane_num=2 là metadata khai báo sai so với cấu hình hiệu lực 3 đã được erratum giải thích.
5. generation_success không phải success của agent.
6. traffic_density không có phép đổi đơn giản thành số xe hoặc phần trăm xác suất.
7. planned traffic là thống kê audit, không phải số xe quan sát tại mọi bước.
8. episode_budget_seconds=100 và required_avg_speed... mô tả mốc tham chiếu; horizon thực tế lấy từ protocol.
9. branching_choice_score là chỉ số có trọng số quy ước, không phải số tất cả đường đi hoặc reward.
10. Hash kiểm tra serialized geometry configuration, không bao phủ toàn bộ trạng thái runtime.
11. Tài liệu schema phải đọc cùng code và erratum: schema mô tả gọn có thể bỏ sót y/Y trong decision block hoặc vai trò pool_candidate ở validation.
12. Preview, metadata và chỉ số cấu trúc không phải bằng chứng hiệu suất thuật toán; cần episode outcomes và metrics riêng.

## 10. Nguồn kiểm chứng trong repository

- [Dataset README](../datasets/mapsuite_v1/README.md): loại dữ liệu, quy mô, mục đích sử dụng.
- [SCHEMA.md](../datasets/mapsuite_v1/SCHEMA.md): tên và kiểu các cột; đọc cùng erratum và code.
- [geometries.csv](../datasets/mapsuite_v1/geometries.csv): danh mục tổng.
- [train_geometries.csv](../datasets/mapsuite_v1/splits/train_geometries.csv), [validation_geometries.csv](../datasets/mapsuite_v1/splits/validation_geometries.csv), [test_geometries.csv](../datasets/mapsuite_v1/splits/test_geometries.csv): dữ liệu từng tập.
- [scenario_families.csv](../datasets/mapsuite_v1/scenario_families.csv): các họ bản đồ và tier.
- [audit_mapsuite_candidates.py](../scripts/audit_mapsuite_candidates.py): đo geometry, đếm block, tính branching, traffic count và chỉ số 100 giây.
- [export_mapsuite_v1_dataset.py](../scripts/export_mapsuite_v1_dataset.py): ghép nguồn, ánh xạ và xuất 29 cột.
- [protocol.py](../src/platform/protocol.py): phân tập, seed, hash và tạo evaluation case.
- [specs.py](../src/platform/specs.py): horizon theo chiều dài tuyến.
- [cases.py](../src/launcher/cases.py): đọc manifest chuẩn, kiểm tra hash, dựng lại geometry.
- [scenario_split_v1.json](../configs/platform/scenario_split_v1.json), [seed_plan_v1.json](../configs/platform/seed_plan_v1.json), [evaluation_protocol_v1.json](../configs/platform/evaluation_protocol_v1.json): hợp đồng split/seed/evaluation.
- [agent_contract_v1.json](../configs/platform/agent_contract_v1.json): đầu vào được phép của agent.
- [AGENT_DEVELOPMENT_GUIDE.md](AGENT_DEVELOPMENT_GUIDE.md): ranh giới training pipeline với launcher/evaluator.
- [MAPSUITE_V1_LANE_METADATA_ERRATUM.md](environment/MAPSUITE_V1_LANE_METADATA_ERRATUM.md): lỗi khai báo số làn và cấu hình thực tế.

