# Xe 3D, garage xoay và cảnh đua

Trong cửa sổ Race Mode, chọn Lamborghini hoặc Tesla Cybertruck để xem model
glTF. Giữ chuột trái trong ô garage rồi kéo để xoay; kéo dọc đổi góc nhìn.
Cuộn chuột để zoom, nhấp đúp trong ô để đặt lại góc. Nút chọn xe và các nút
đường đua không bắt đầu thao tác xoay. Đổi chế độ/rời garage kết thúc kéo.
Trong cuộc đua, **giữ chuột trái rồi kéo trong vùng 3D** để xoay camera
quanh xe; kéo dọc đổi độ cao góc nhìn. Thả chuột thì ngừng xoay, xe vẫn
được camera theo dõi. Ở góc nhìn từ trên xuống, giữ–kéo để dịch vùng nhìn.
Không giữ nút thì chuột di chuyển tự do, không tự canh về giữa màn hình.
`C`/`V` đổi góc/xe theo dõi và đặt lại góc kéo; cuộn chuột vẫn chỉnh camera.
Bấm nút giao diện, bản đồ nhỏ hoặc bảng xếp hạng không bắt đầu kéo.
Kiểm tra native xác nhận giữ–kéo, thả dừng, rê không xoay và không gọi
`movePointer`, đồng thời giữ nguyên vị trí xe trong phép kiểm tra render:
`artifacts/drag_camera_20261120/native/drag-audit.json`.

Hai model được đóng gói để chạy offline:

| Mẫu | Nguồn | Tam giác sau xử lý |
|---|---|---:|
| Lamborghini Aventador | Arion Digital, CC BY 4.0 | 10.252 cả bốn bánh |
| Tesla Cybertruck | Nieve5677, CC BY 4.0 | 42.161, từ 72.528 |
| VinFast VF3 | Mẫu dựng theo ảnh tham chiếu | 5.500 cả bốn bánh |

Nguồn, giấy phép, các sửa đổi và giới hạn ở
[assets/vehicles/ATTRIBUTION.md](../assets/vehicles/ATTRIBUTION.md).
Attribution cũng hiện trong garage. Lamborghini và Cybertruck có bốn bánh
quay riêng, bánh trước theo góc đánh lái Bullet. Ngoại hình được
căn lại theo khung chung, không phải mô phỏng động lực học hãng.

Camera mặc định dùng góc bám sau xe của MetaDrive (cao 2,2 m, cách 7,5 m).
`C` chuyển sang nhìn từ trên xuống; `V` hoặc `1`–`5` đổi xe theo dõi. Bộ
`original` giữ hình học đường gốc, mặc định SCTCS/0. Mây, địa hình, vỉa hè
dùng renderer MetaDrive hiện có. Độ dày vạch đường được quy đổi theo độ
phân giải raster để tránh vạch quá rộng khi giảm tải GPU. Lọc texture góc
xiên và vật liệu/normal giúp mặt đường và xe rõ hơn. Các thay đổi camera
và hiển thị này không thay đổi thuật toán lái hay vật lý.

Từ repository:

```powershell
.\CHAY_DUA_XE.cmd --graphics balanced --map-set original --road medium
# Giảm tải GPU:
.\CHAY_DUA_XE.cmd --graphics lite
# Mở trực tiếp VF3 trong garage:
.\CHAY_DUA_XE.cmd --vehicle vf3
```

`balanced` hiện là mặc định của launcher, cảnh 3D vẽ ở độ phân giải cửa sổ.
`lite` vẽ cảnh 3D ở 75%, giữ chữ/nút ở độ phân giải cửa sổ. Cả hai dùng MSAA
2×; garage có môi trường phản chiếu nhỏ tính sẵn, không dùng camera phản
chiếu realtime hay shadow map nặng. Texture mỹ thuật xe tối đa 512 px ở
Lite, 1024 px ở Balanced; dữ liệu terrain/sensor không bị nén/thu nhỏ chung.

## Kiểm chứng ngày 20/11/2026

Sau khi thêm camera giữ–kéo: 39 kiểm tra cho input, garage, vạch đường,
Race Mode và realtime/phone đều đạt. Native smoke chạy 4 nhóm trên 3 map,
pause, focus, đổi camera và reset; kết quả ở
`artifacts/drag_camera_20261120/smoke/session.json`.

Đã render và xem ba mẫu xe, góc xoay khác nhau và cảnh đua có trời/địa hình.
Native smoke hoàn tất các nhóm 5/2/1/3 xe trên ba đường, pause, đổi xe/camera
và reset cùng cửa sổ. Kiểm thử riêng kiểm tra phạm vi chuột, zoom có giới
hạn, nhấp đúp và giữ nguyên node Bullet/vị trí bánh khi thay model.

Ảnh/log nằm trong `artifacts/car_orbit_map_20261120/`. Phép đo render tĩnh
Lite ở `landscape-v2/graphics.json` đạt khoảng 10,8 ms/vòng trong mẫu 4 giây.
Nó không đo toàn bộ cuộc đua hay GPU timestamp riêng, và không phải cam kết
FPS trên mọi máy. Các kết quả đo của bản xe tự dựng cũ trong
`LOCAL_GPU_LITE.md` là lịch sử, không thay thế phép đo của model mới.

Công cụ điều khiển Windows không khởi tạo được trong phiên này. Ảnh được
renderer xuất trực tiếp; thao tác chuột được kiểm tra ở mức controller,
chưa có kiểm tra kéo chuột qua công cụ Windows hoặc iPhone thật.

Để dựng lại Tesla, cài các công cụ chỉ trong thư mục artifact:

```powershell
cd autonomous-driving-rl
npm.cmd install --prefix artifacts/vehicle-asset-tools --no-audit --no-fund @gltf-transform/core@4.4.0 @gltf-transform/extensions@4.4.0 @gltf-transform/functions@4.4.0 draco3dgltf@1.5.7 meshoptimizer@0.25.0
node scripts/optimize_vehicle_assets.mjs artifacts/cybertruck_20261121/sources/cybertruck-source.glb artifacts/cybertruck_20261121/sources/cybertruck-decoded.glb 1
& '..\.venv\metadrive\python.exe' -B scripts/prepare_cybertruck_asset.py artifacts/cybertruck_20261121/sources/cybertruck-decoded.glb artifacts/cybertruck_20261121/sources/cybertruck-materials.glb
node scripts/optimize_vehicle_assets.mjs artifacts/cybertruck_20261121/sources/cybertruck-materials.glb assets/vehicles/tesla/model.glb 0.35
```

GLB nguồn và credit đã lưu trong thư mục artifact ở lệnh trên;
runtime chỉ cần hai `assets/vehicles/*/model.glb`, không cần npm hoặc mạng.

## Cybertruck theo ảnh tham chiếu ngày 21/11/2026

Tesla Model 3 được thay bằng Cybertruck: thân thép bạc có các mặt phẳng,
kính tối, dải đèn ngang và bánh xe đen. ID `tesla` được giữ để cấu hình
xe trên laptop và điện thoại tiếp tục khớp. Model đóng gói khoảng 2,07 MB,
không có texture ngoài; bốn bánh được tách và gắn vào node vật lý sẵn có.
Giới hạn 20 km/h và luật đua chung vẫn áp dụng.

Kiểm tra native xuất ảnh garage, góc xoay, xe chạy cùng AI và ba camera
điện thoại trong `artifacts/cybertruck_20261121/native-final/`. Xe đi được
6,36 m qua 28 bước, đạt 16,70 km/h, cả bốn node bánh chuyển động. 27 kiểm
tra Python và 15 JavaScript đạt. Đây là kiểm tra model, chuyển động và
render trong MetaDrive; chưa phải thử bằng điện thoại thật.

## VF3 theo ảnh tham chiếu

VF3 được dựng lại trong `vf3_mesh.py` theo ảnh xanh ngọc/mui đen người dùng
cung cấp: cabin đứng, mui đen, một cửa bên mỗi phía, dải mặt trước chữ V,
đèn pha hình chữ nhật, ốp vè vuông rộng và mâm thép đen. Hốc bánh được khoét
trong thân, hình bánh chỉ dịch trong node visual; tâm bánh Bullet không đổi.
Garage có góc nhìn riêng cho VF3 để mui không chạm tiêu đề.

Tỷ lệ đối chiếu từ [tài liệu VinFast](https://vinfastauto.com/vn_vi/node/11419).
Model vẫn là bản minh họa nhẹ theo ảnh, được căn theo khung mô phỏng chung;
không phải CAD hãng hoặc bản sao chính xác từng chi tiết.

Thân gần camera có 4 Geom, 1.724 tam giác; mỗi bánh 944 tam giác, tổng 5.500.
Ảnh và log mới: `artifacts/vf3_reference_20261120/`. 39 kiểm thử race/graphics/
orbit/realtime đã qua, native smoke hoàn tất 5/2/1/3 xe và ba đường. Các số đo
render cũ ở phần trên không phải số đo hiệu năng của toàn bộ model VF3 mới.
