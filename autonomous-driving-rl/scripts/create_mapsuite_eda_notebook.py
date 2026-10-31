"""Build and execute a Vietnamese learning notebook without changing MapSuite.

Run from any directory with the project-local Python interpreter.
Only notebooks/mapsuite_v1_eda.ipynb and notebooks/eda_figures/ are written.
No simulator, launcher, training, or external service is used.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path
import textwrap


PROJECT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT / "notebooks" / "mapsuite_v1_eda.ipynb"
CELLS: list[dict] = []


def markdown(source: str) -> None:
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip()})


def code(source: str) -> None:
    CELLS.append({"cell_type": "code", "metadata": {}, "source": textwrap.dedent(source).strip(),
                  "execution_count": None, "outputs": []})


markdown("""
# Hiểu MapSuite V1 qua EDA

**Mục tiêu:** hiểu dữ liệu trước khi xây 8 tầng agent. Đọc từng phần và chạy từng ô.

MapSuite chứa mô tả **đường mô phỏng** và **ca đánh giá**. Một dòng `geometries.csv`
là một bản đồ; một dòng `evaluation_cases/*.csv` là một bản đồ kết hợp seed môi trường.
Dữ liệu trạng thái–hành động để học phải được thu riêng khi agent chạy.

Notebook chỉ đọc dữ liệu gốc. Các ô lưu biểu đồ vào `eda_figures/` cạnh notebook.
Phân tích đặc điểm hình học dùng TRAIN/VALIDATION. TEST chỉ tham gia kiểm tra cấu trúc,
số lượng và tính tách biệt; không chạy agent hay dùng kết quả TEST để chọn thuật toán.

**Cách chạy:** chọn kernel Python của `.venv/metadrive/python.exe` trong repository
rồi Run All. Cần pandas và matplotlib. Không cần khởi động MetaDrive hay Workbench.

**Nguồn:** `datasets/mapsuite_v1/{README.md,SCHEMA.md,geometries.csv,scenario_families.csv,
splits/,evaluation_cases/,CHECKSUMS.sha256}`; `configs/platform/seed_plan_v1.json`;
`docs/environment/MAPSUITE_V1_LANE_METADATA_ERRATUM.md`.
""")

code("""
from pathlib import Path
import hashlib
import json
import pandas as pd
import matplotlib.pyplot as plt

def find_project():
    here = Path.cwd().resolve()
    for ancestor in [here, *here.parents]:
        for candidate in [ancestor, ancestor / 'autonomous-driving-rl']:
            if (candidate / 'datasets/mapsuite_v1/geometries.csv').is_file():
                return candidate
    raise FileNotFoundError('Mở notebook từ repository hoặc autonomous-driving-rl.')

root = find_project()
data = root / 'datasets/mapsuite_v1'
figures = root / 'notebooks/eda_figures'
figures.mkdir(parents=True, exist_ok=True)
tiers = ['Easy', 'Medium', 'Hard', 'Extreme']
splits_order = ['TRAIN', 'VALIDATION', 'TEST']
colors = ['#137c8b', '#4c78a8', '#dc9b29', '#b65050']
plt.rcParams.update({'figure.dpi': 110, 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.2})

geometries = pd.read_csv(data / 'geometries.csv')
split_tables = {s: pd.read_csv(data / f'splits/{s.lower()}_geometries.csv') for s in splits_order}
validation_cases = pd.read_csv(data / 'evaluation_cases/validation_cases.csv')
test_cases = pd.read_csv(data / 'evaluation_cases/test_cases.csv')
families = pd.read_csv(data / 'scenario_families.csv')
seed_plan = json.loads((root / 'configs/platform/seed_plan_v1.json').read_text())
development = geometries[geometries['split'].isin(['TRAIN', 'VALIDATION'])].copy()
print(f'{len(geometries)} bản đồ, {len(families)} họ đường; phân tích hình học trên {len(development)} bản đồ TRAIN/VALIDATION.')
""")

markdown("""
## 1. Đọc một dòng dữ liệu

Ví dụ TRAIN được chọn theo ID, để việc sắp xếp CSV không làm đổi ví dụ.
`geometry_id` là nhãn; `geometry_sha256` là dấu vân tay của cấu hình hình học.
`sequence` cho biết loại khối đường, seed xác định biến thể trong cùng họ.

Mã khối: **I** xuất phát, **S** thẳng, **C** cong, **X** ngã tư, **T** ngã ba,
**O** vòng xuyến, **r/R** ramp vào/ra, **y/Y** merge/split.
Seed là khóa tái lập, không phải điểm khó: seed 19 không mặc nhiên khó hơn seed 0.
""")
code("""
example = geometries.loc[geometries['geometry_id'].eq('geom_easy_SCS_seed0')].iloc[0]
columns = ['geometry_id', 'split', 'tier', 'sequence', 'geometry_generation_seed',
           'block_ids', 'route_length_m', 'decision_block_count', 'traffic_density']
display(example[columns].to_frame('Giá trị'))
print('SCS: thẳng → cong → thẳng; block_ids có thêm đoạn xuất phát I.')
""")

markdown("""
## 2. Hiểu các nhóm cột

| Nhóm | Cột | Cách đọc |
|---|---|---|
| Danh tính | geometry_id, split, tier, sequence, geometry_generation_seed, candidate_role | Bản đồ nào, thuộc tập/họ nào, vai trò lựa chọn nào |
| Nguồn và tái lập | geometry_sha256, geometry_hash_source, generation_success | Dấu vân tay, nguồn của hash, kết quả dựng hình học đã ghi nhận |
| Kích thước | route_length_m, bbox_width_m, bbox_height_m | Tuyến dài bao nhiêu; bản đồ chiếm vùng rộng/cao bao nhiêu (m) |
| Khối đường | block_ids, block_count, straight_count, curve_count, intersection_count, t_intersection_count, roundabout_count, ramp_count, merge_split_count | Cấu trúc đường; I được tính trong block_count |
| Quyết định | decision_block_count, branching_choice_score | Số khối có lựa chọn nhánh và chỉ số lựa chọn nhánh được ghi nhận |
| Làn | lane_width_m, base_lane_num | Rộng làn (m); base_lane_num là metadata lịch sử, đọc cùng đính chính |
| Giao thông | traffic_density, planned_traffic_vehicle_count | Tham số mật độ và số xe nền dự kiến; không phải số xe quan sát ở mọi bước |
| Thời gian lịch sử | episode_budget_seconds, required_avg_speed_kmh_for_horizon_1000 | Tham chiếu ngân sách 1000 bước; không thay thế horizon_steps của ca benchmark |

**Đính chính số làn:** CSV ghi 2; cấu hình cơ sở thực tế của hình học đóng băng dùng 3.
Giữ dữ liệu gốc và đọc tài liệu erratum; không ép mô phỏng về 2 làn.
Một số đoạn merge/ramp có thể có số làn khác cấu hình cơ sở.
""")

markdown("""
## 3. Kiểm tra chất lượng dữ liệu

Kiểm tra thiếu dữ liệu, ID trùng, các tập giao nhau và sự khớp của bảng split với bảng gốc.
Giá trị 0 ở số ngã tư hay traffic_density là dữ liệu hợp lệ, không phải missing.
Kiểm tra này đánh giá các bảng đã lưu; không chứng minh đã dựng lại mọi bản đồ bằng simulator.
""")
code("""
ids = {s: set(table['geometry_id']) for s, table in split_tables.items()}
quality = {
    'Số ô thiếu trong bảng hình học': int(geometries.isna().sum().sum()),
    'ID bản đồ trùng': int(geometries['geometry_id'].duplicated().sum()),
    'Khóa (tier, sequence, seed) trùng': int(geometries.duplicated(['tier', 'sequence', 'geometry_generation_seed']).sum()),
    'TRAIN giao VALIDATION': len(ids['TRAIN'] & ids['VALIDATION']),
    'TRAIN giao TEST': len(ids['TRAIN'] & ids['TEST']),
    'VALIDATION giao TEST': len(ids['VALIDATION'] & ids['TEST']),
    'Số ô thiếu trong bảng VALIDATION cases': int(validation_cases.isna().sum().sum()),
    'Số ô thiếu trong bảng TEST cases': int(test_cases.isna().sum().sum()),
    'ID ca VALIDATION trùng': int(validation_cases['case_id'].duplicated().sum()),
    'ID ca TEST trùng': int(test_cases['case_id'].duplicated().sum()),
}
display(pd.Series(quality, name='Số lượng').to_frame())
assert geometries['geometry_id'].is_unique
assert set.union(*ids.values()) == set(geometries['geometry_id'])
assert not any(ids[a] & ids[b] for a, b in [('TRAIN','VALIDATION'),('TRAIN','TEST'),('VALIDATION','TEST')])
for split, table in split_tables.items():
    expected = geometries.loc[geometries['split'].eq(split)].sort_values('geometry_id').reset_index(drop=True)
    pd.testing.assert_frame_equal(table.sort_values('geometry_id').reset_index(drop=True), expected)
print('Các bảng split khớp bảng gốc và không giao nhau.')
""")

markdown("""
## 4. Cân bằng số lượng giữa các tập

240 = 12 họ × 20 seed hình học. Các tập tách theo **bản đồ cụ thể**, vẫn có thể chứa cùng họ đường.
Số lượng cân bằng theo tier không bảo đảm mọi đặc điểm hình học có phân phối giống nhau.
Đây là benchmark được thiết kế và đóng băng; EDA không tự cho phép chia lại data.
""")
code("""
split_counts = geometries['split'].value_counts().reindex(splits_order)
tier_split = pd.crosstab(geometries['tier'], geometries['split']).reindex(index=tiers, columns=splits_order)
display(tier_split)
display(pd.crosstab(geometries['sequence'], geometries['split']).reindex(columns=splits_order))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
split_counts.plot.bar(ax=axes[0], color=['#137c8b', '#4c78a8', '#dc9b29'], rot=0)
axes[0].set(title='Geometry counts by split', ylabel='Geometries', xlabel='')
for i, value in enumerate(split_counts): axes[0].text(i, value + 2, str(value), ha='center')
tier_split.plot.bar(ax=axes[1], rot=0, color=['#137c8b', '#4c78a8', '#dc9b29'])
axes[1].set(title='Geometry counts by difficulty', ylabel='Geometries', xlabel='Difficulty tier')
axes[1].set_ylim(0, 55)
axes[1].legend(title=None, loc='upper center', ncol=3, fontsize=9, frameon=False)
fig.tight_layout()
fig.savefig(figures / '01_split_counts.png', dpi=150)
plt.show()
""")

markdown("""
## 5. Các họ đường khác nhau ở đâu?

Bảng dưới mô tả các bản đồ TRAIN/VALIDATION. Chú ý độ dài, số khối quyết định và mật độ.
`traffic_density` được đặt theo tier, nên quan hệ giữa tier và mật độ là thiết kế sẵn.
Không suy ra quan hệ nhân quả từ tương quan của các cột này.
""")
code("""
family_summary = development.groupby(['tier', 'sequence']).agg(
    n=('geometry_id', 'size'), min_route_m=('route_length_m', 'min'),
    mean_route_m=('route_length_m', 'mean'), max_route_m=('route_length_m', 'max'),
    decision_blocks=('decision_block_count', 'median'), traffic_density=('traffic_density', 'first'))
display(family_summary.round(2))
display(development.groupby('tier')[['route_length_m', 'block_count', 'decision_block_count', 'planned_traffic_vehicle_count']]
        .agg(['min', 'median', 'max']).reindex(tiers).round(2))
""")

markdown("""
## 6. Độ dài và cấu trúc đường

Boxplot: đường giữa hộp là median; hộp là khoảng 25%–75%; các điểm ngoài râu là giá trị xa theo quy tắc boxplot.
Điểm xa không mặc nhiên là bản đồ sai. Không xóa hay đổi split chỉ vì tuyến dài.
So sánh TRAIN/VALIDATION để hiểu độ phủ; TEST được giữ ngoài biểu đồ đặc điểm.
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
values = [development.loc[development['tier'].eq(t), 'route_length_m'] for t in tiers]
boxes = axes[0].boxplot(values, patch_artist=True)
axes[0].set_xticks(range(1, 5), tiers)
for box, color in zip(boxes['boxes'], colors): box.set_facecolor(color); box.set_alpha(0.65)
axes[0].set(title='Route length by tier (TRAIN + VALIDATION)', ylabel='Route length (m)')
for split, color in [('TRAIN', '#137c8b'), ('VALIDATION', '#dc9b29')]:
    axes[1].hist(development.loc[development['split'].eq(split), 'route_length_m'],
                 bins=range(0, int(development['route_length_m'].max()) + 101, 100),
                 density=True, alpha=0.45, label=split, color=color)
axes[1].set(title='Route-length distributions', xlabel='Route length (m)', ylabel='Probability density (1/m)')
axes[1].legend()
fig.tight_layout()
fig.savefig(figures / '02_route_lengths.png', dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(9, 4.8))
for tier, color in zip(tiers, colors):
    subset = development.loc[development['tier'].eq(tier)]
    ax.scatter(subset['decision_block_count'], subset['route_length_m'], label=tier,
               color=color, alpha=0.55, s=28)
ax.set(title='Route length and decision blocks (TRAIN + VALIDATION)',
       xlabel='Decision blocks (count)', ylabel='Route length (m)')
ax.set_xticks(sorted(development['decision_block_count'].unique()))
ax.legend()
fig.tight_layout()
fig.savefig(figures / '03_structure.png', dpi=150)
plt.show()
""")

markdown("""
## 7. Từ bản đồ đến ca đánh giá: hiểu seed

- Geometry seed quyết định hình dạng đường.
- Environment seed quyết định ngẫu nhiên mô phỏng trên đường đó.
- Agent seed quyết định ngẫu nhiên bên trong thuật toán, được cấu hình khi chạy.

Ví dụ lấy một bản đồ VALIDATION và hiển thị hai ca của nó. Chúng có cùng geometry hash,
nhưng environment seed khác. 48 bản đồ × 2 seed = 96 ca VALIDATION;
12 bản đồ × 5 seed = 60 ca TEST. Một ca thường tương ứng một episode trong một lần lặp agent.
""")
code("""
key_columns = ['tier', 'sequence', 'geometry_generation_seed']
v = validation_cases.iloc[0]
mask = (validation_cases[key_columns] == v[key_columns]).all(axis=1)
paired = validation_cases.loc[mask].sort_values('environment_seed')
display(paired[['case_id', 'geometry_generation_seed', 'environment_seed', 'horizon_steps', 'protocol_order_index']])
assert paired['geometry_sha256'].nunique() == 1
for label, cases, expected_split, seeds in [
    ('VALIDATION', validation_cases, 'VALIDATION', seed_plan['validation_environment_seeds']),
    ('TEST', test_cases, 'TEST', seed_plan['test_environment_seeds'])]:
    allowed = set(map(tuple, split_tables[expected_split][key_columns].to_numpy()))
    assert set(map(tuple, cases[key_columns].to_numpy())) == allowed
    for key, group in cases.groupby(key_columns):
        assert sorted(group['environment_seed'].tolist()) == sorted(seeds)
        assert group['geometry_sha256'].nunique() == 1
    assert sorted(cases['protocol_order_index']) == list(range(1, len(cases) + 1))
    print(label, ':', len(allowed), 'bản đồ ×', len(seeds), 'seed =', len(cases), 'ca')
print('Seed agent cho các lần lặp:', seed_plan['agent_replicate_seeds'])
print('Không nhầm 3 lần lặp agent với 3 bản đồ hoặc 3 seed môi trường.')
""")

markdown("""
## 8. Giới hạn thời gian của một episode

Control 10 Hz: `horizon_steps / 10` là thời gian **mô phỏng** cho phép.
Tốc độ trung bình tối thiểu lý tưởng = độ dài tuyến / thời gian × 3.6 (km/h).
Đây là phép tính hình học; phanh/chờ/rẽ khiến yêu cầu lái thực tế phức tạp hơn.
Ngân sách 100 giây trong `geometries.csv` là tham chiếu lịch sử 1000 bước;
ca đánh giá dùng `horizon_steps` riêng theo tuyến.

Mỗi bản đồ VALIDATION chỉ được tính một lần trong thống kê hình học dưới đây,
tránh nhân đôi mẫu bởi hai environment seed.
""")
code("""
validation_geometry_cases = validation_cases.drop_duplicates(key_columns)
budget = validation_geometry_cases.merge(split_tables['VALIDATION'][key_columns + ['route_length_m']],
                                          on=key_columns, validate='one_to_one')
budget['allowed_simulation_seconds'] = budget['horizon_steps'] / 10
budget['ideal_required_avg_speed_kmh'] = budget['route_length_m'] / budget['allowed_simulation_seconds'] * 3.6
display(budget[['tier', 'sequence', 'geometry_generation_seed', 'route_length_m',
                'horizon_steps', 'allowed_simulation_seconds', 'ideal_required_avg_speed_kmh']].head().round(2))
display(budget.groupby('tier')[['allowed_simulation_seconds', 'ideal_required_avg_speed_kmh']]
        .agg(['min', 'median', 'max']).reindex(tiers).round(2))
fig, ax = plt.subplots(figsize=(9, 4.8))
for tier, color in zip(tiers, colors):
    subset = budget.loc[budget['tier'].eq(tier)]
    ax.scatter(subset['route_length_m'], subset['allowed_simulation_seconds'], color=color, label=tier)
ax.set(title='Canonical VALIDATION time budget', xlabel='Route length (m)', ylabel='Allowed simulation time (s)')
ax.legend()
fig.tight_layout()
fig.savefig(figures / '04_time_budget.png', dpi=150)
plt.show()
""")

markdown("""
## 9. Kiểm tra dấu vân tay của gói

Hash byte của file thay đổi khi CRLF đổi thành LF, dù giá trị CSV không đổi.
Bảng này chỉ chẩn đoán. Không sửa nguồn hay viết lại checksum.
`lf_normalized_match=True` cho biết chuẩn hóa xuống dòng trong bộ nhớ khôi phục đúng hash đã ghi.
Checksum của file khác với geometry hash của cấu hình bản đồ.
""")
code("""
checksum_rows = []
for line in (data / 'CHECKSUMS.sha256').read_text().splitlines():
    expected_hash, relative = line.split(maxsplit=1)
    content = (data / relative).read_bytes()
    checksum_rows.append({'file': relative, 'raw_match': hashlib.sha256(content).hexdigest() == expected_hash,
                          'lf_normalized_match': hashlib.sha256(content.replace(b'\\r\\n', b'\\n')).hexdigest() == expected_hash})
checksum_table = pd.DataFrame(checksum_rows)
display(checksum_table.loc[~checksum_table['raw_match']])
print('Số file lệch raw checksum:', int((~checksum_table['raw_match']).sum()))
""")

markdown("""
## 10. Tự giải thích data trước khi làm agent

1. Một dòng hình học và một dòng ca đánh giá khác nhau thế nào?
2. Đổi environment seed nhưng giữ geometry seed có đổi đường không?
3. Vì sao 12 bản đồ TEST tạo thành 60 ca, và agent ngẫu nhiên có thể chạy 180 episode qua 3 lần lặp?
4. Tầng 4 cần thu dữ liệu trạng thái–hành động thế nào từ TRAIN?
5. Cột traffic_density có phải tỷ lệ xe mà agent nhìn thấy ở một bước không?
6. Vì sao không dùng cột ngân sách 100 giây thay horizon của ca benchmark?
7. Vì sao số bản đồ cân bằng theo tier chưa chứng minh agent chạy tốt trên mọi tier?

Sau EDA, bước tiếp theo là xác minh runtime, hoàn thiện Stage 0 rồi xây Stage 1.
Success rate, crash rate, tốc độ thực tế và thời gian tới đích cần kết quả episode của agent;
không tính được từ các bảng hình học này. Notebook không tạo kết quả benchmark hay kiểm chứng
live geometry reconstruction. Không thay đổi TEST từ quan sát EDA.
""")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source_dir = PROJECT / "datasets" / "mapsuite_v1"
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_dir.rglob("*") if p.is_file()}
    environment = {"__name__": "__notebook__"}
    for count, cell in enumerate((c for c in CELLS if c["cell_type"] == "code"), 1):
        outputs: list[dict] = []
        def display(value):
            payload = {"text/plain": str(value)}
            if hasattr(value, "to_html"):
                payload["text/html"] = value.to_html()
            outputs.append({"output_type": "display_data", "data": payload, "metadata": {}})

        def show():
            for number in plt.get_fignums():
                figure = plt.figure(number)
                buffer = io.BytesIO()
                figure.savefig(buffer, format="png", dpi=110)
                outputs.append({"output_type": "display_data", "data": {
                    "image/png": base64.b64encode(buffer.getvalue()).decode("ascii"),
                    "text/plain": "Matplotlib figure"}, "metadata": {}})
                plt.close(figure)

        environment["display"] = display
        plt.show = show
        captured = io.StringIO()
        with contextlib.chdir(PROJECT) if hasattr(contextlib, "chdir") else project_directory(PROJECT):
            with contextlib.redirect_stdout(captured):
                exec(compile(cell["source"], f"notebook cell {count}", "exec"), environment)
        if captured.getvalue():
            outputs.append({"output_type": "stream", "name": "stdout", "text": captured.getvalue()})
        cell["execution_count"] = count
        cell["outputs"] = outputs
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_dir.rglob("*") if p.is_file()}
    assert before == after, "Dataset changed during notebook execution"
    notebook = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python (metadrive)",
        "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}},
        "nbformat": 4, "nbformat_minor": 4}
    try:
        import nbformat
    except ImportError:
        pass  # JSON is still valid; nbformat is not a dependency of the analysis.
    else:
        nbformat.validate(nbformat.from_dict(notebook))
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Executed {sum(c['cell_type'] == 'code' for c in CELLS)} code cells: {DESTINATION}")
    print("Dataset bytes preserved.")
    print("Split counts:", environment["split_counts"].to_dict())
    print("Quality findings:", environment["quality"])


@contextlib.contextmanager
def project_directory(path):
    import os
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    main()
