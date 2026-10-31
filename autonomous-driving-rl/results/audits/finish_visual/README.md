# Kiểm tra lỗi không hiển thị FINISH hologram

Ngày kiểm tra: 2026-10-30. MetaDrive 0.4.3, Panda3D.

## Tình huống tái hiện

```powershell
python -X utf8 -m src.launcher.cli run --mode SANDBOX --agent stage1_rule_based --tier Medium --sequence SCXCS --geom-seed 14 --env-seed 0 --agent-seed 42 --render NATIVE --wandb DISABLED --finish-visual hologram
```

Đã kiểm tra đường truyền CLI → biến môi trường visual style → executor →
factory → `FinishFloatingHologram`. Flag được truyền đúng.

Với đúng geometry SCXCS/14 và environment seed 0:

- Final lane: `('4C0_1_', '5S0_0_', 2)`, dài khoảng 73.243 m.
- Hologram nằm ở longitudinal `lane.length - 5`, cùng ngưỡng tiếp cận đích.
- Tâm road tại marker: `(159.5787, 230.4598, 0)`.
- Road width 10.5 m, heading của node khoảng 16.423°.
- Không thấy sai vị trí đặt marker trong case này.

## Nguyên nhân được xác nhận bằng render

`engine.render` gán shader PBR cho các node con. Hai mặt phẳng RGBA từ
`CardMaker` kế thừa shader đó. `setLightOff()` chỉ thay trạng thái ánh sáng,
không loại bỏ shader đã kế thừa. Node và texture đều tồn tại nhưng bảng và
laser không xuất hiện trong ảnh.

Đặt `setShaderOff(1)` trên root visual làm cả bảng và laser xuất hiện ngay,
không cần thay đổi vị trí, góc quay hoặc texture. Đồng thời đặt light off,
depth write off cho các mặt phẳng trong suốt; giữ depth test mặc định.

Nguồn API: [Panda3D NodePath.setShaderOff](https://docs.panda3d.org/1.10/python/reference/panda3d.core.NodePath#panda3d.core.NodePath.setShaderOff).

## Bằng chứng ảnh

Ảnh chụp bằng main camera offscreen 960×540, dùng cùng pipeline PBR và đúng
geometry của lệnh trên. Camera đặt cách tâm marker 25 m về phía trước đích,
cao 4.5 m, nhìn vào điểm cao 3 m. Không chạy policy hoặc thay đổi physics
giữa các ảnh so sánh; đây là kiểm tra render, không phải một episode benchmark.

| Ảnh | Mô tả | Số pixel cyan theo ngưỡng kiểm tra |
|---|---|---:|
| [before.png](before.png) | Code trước sửa, kế thừa PBR | 0 |
| [shader_off.png](shader_off.png) | Probe chỉ tắt shader trên code cũ | 2668 |
| [after.png](after.png) | Code sau sửa đầy đủ | 2668 |
| [hidden.png](hidden.png) | Ẩn marker trên code sau sửa để đối chiếu | 0 |

Ngưỡng cyan: R < 100, G > 130, B > 130. Ảnh sau sửa so với ảnh ẩn marker có
25,653 pixel thay đổi trên 25 đơn vị ở ít nhất một kênh RGB. Đã xem trực tiếp
ảnh sau sửa để xác nhận chữ FINISH và laser mặt đường.

## Sửa các lỗi phụ trong cùng luồng

- Executor trước đó dùng `except Exception: finish_visual = None`, làm mất
  hoàn toàn nguyên nhân khi khởi tạo thất bại. Nay ghi log exception/traceback.
- Kiểm tra kết quả load texture; thiếu texture báo lỗi cụ thể.
- Dọn node đã dựng dở nếu constructor thất bại.
- Visual chỉ hiện với MainCam và MiniMap, ẩn khỏi RGB/depth/semantic camera
  và shadow pass. Không thêm Bullet node; observation/LiDAR giữ nguyên.

## Kiểm thử

```powershell
python -X utf8 -m unittest tests.test_finish_visual tests.test_random_naive_agent tests.test_agent_contract tests.test_rule_based_agent tests.test_launcher_core
```

187 tests đạt, gồm regression shader kế thừa, camera mask, load texture lỗi,
cleanup, log lỗi launcher và observation/LiDAR invariance. Các test visual
cũ chỉ kiểm tra node tồn tại nên chưa bắt được lỗi render này.

Có thể chạy lại nguyên lệnh NATIVE phía trên sau khi đóng process cũ.
Hologram là vật thể tại cuối tuyến; nhìn thấy khi camera tới gần đích và
hướng về marker, không phải overlay luôn hiện trên màn hình từ lúc xuất phát.

## Cập nhật giao diện: phủ cả hai chiều, chỉ giữ FINISH

Theo yêu cầu tiếp theo, bảng và laser mở rộng qua cả hai chiều của đoạn đường
đích. Lấy thêm road ngược bằng `-Road(u, v)` và chiếu cùng một mặt cắt thế giới
vào từng làn, vì làn chiều ngược có mốc longitudinal ở đầu đối diện.

Ở SCXCS/14, bề ngang tăng từ 10.5 m lên 21 m; tâm mới khoảng
`(154.5429, 228.9754, 0)`. Laser phủ đúng bề ngang mặt đường; bảng thêm 0.6 m
mỗi bên. Xóa tiêu đề phụ, dòng nhiệm vụ và biểu tượng cờ, chỉ giữ chữ FINISH.
Texture mới có tên chứa phiên bản layout và hash nội dung để không dùng lại
ảnh cache còn chữ cũ.

Ảnh xác nhận: [wide_finish_only.png](wide_finish_only.png).
9 kiểm thử visual đạt, bao gồm mặt cắt đường có làn đi và làn về cùng hiện diện.
