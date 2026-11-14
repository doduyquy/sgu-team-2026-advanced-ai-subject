# Documentation Guide

This directory has two layers of documentation.

**Known MapSuite V1 erratum:** the historical metadata declares two base lanes, while the frozen geometry uses the pinned effective three-lane reconstruction. Read the [lane metadata erratum](environment/MAPSUITE_V1_LANE_METADATA_ERRATUM.md) before interpreting map configuration or reproducing geometry.

## 1. Team-facing guides

Read these when joining or developing the project:

- [`GETTING_STARTED.md`](GETTING_STARTED.md) — machine setup and smoke verification.
- [`PLATFORM_V1_OVERVIEW.md`](PLATFORM_V1_OVERVIEW.md) — what Platform V1 already provides.
- [`EXPERIMENT_WORKFLOW.md`](EXPERIMENT_WORKFLOW.md) — experiment modes, evaluation protocol, and result interpretation.
- [`AGENT_DEVELOPMENT_GUIDE.md`](AGENT_DEVELOPMENT_GUIDE.md) — how to implement and register agents.
- [`STAGE4_G0_TECHNICAL_AUDIT_VI.md`](STAGE4_G0_TECHNICAL_AUDIT_VI.md) — Kiểm định BC Stage 4, thiết kế dữ liệu/mô hình và nghiệm thu từng cổng; G0 đã nghiệm thu.
- [`STAGE4_G1_TELEMETRY_VI.md`](STAGE4_G1_TELEMETRY_VI.md) — G1 bước 1: bằng chứng TRAIN trước/sau sửa vận tốc, xác minh dấu lái; G1 đang thực hiện.
- [`STAGE4_G1_PILOT_VI.md`](STAGE4_G1_PILOT_VI.md) — G1 bước 2: 5 lượt TRAIN, 5,744 cặp dữ liệu thô đã kiểm tra nguồn gốc/ghép bước và phát lại; chưa duyệt chất lượng BC.
- [`STAGE4_G1_RESOURCES_VI.md`](STAGE4_G1_RESOURCES_VI.md) — G1 bước 3: thời gian/dung lượng/RAM thực đo, kịch bản ngân sách TRAIN và môi trường huấn luyện NOT_READY.
- [`STAGE4_G1_SPLIT_KAGGLE_VI.md`](STAGE4_G1_SPLIT_KAGGLE_VI.md) — G1 bước 4: danh sách 144 FIT / 36 DEV đã chốt, ánh xạ pilot và đích huấn luyện Kaggle.
- [`STAGE4_G1_QUALITY_VI.md`](STAGE4_G1_QUALITY_VI.md) — G1 bước 5: hiệu chỉnh bằng FIT, kiểm định SUCCESS và từng đoạn TIMEOUT, quy tắc giữ/loại có phiên bản.
- [`STAGE4_G1_DATASETS_VI.md`](STAGE4_G1_DATASETS_VI.md) — G1 bước 6: bộ thử BC-A/BC-B, danh sách mẫu/nguồn gốc/mã băm, cách đọc độc lập và đơn vị bản đồ/lượt/mẫu.
- [`STAGE4_G0_REVIEW_VI.md`](STAGE4_G0_REVIEW_VI.md) — Quyết định nghiệm thu G0 ngày 13/11/2026, bằng chứng, giới hạn và công việc chuyển tiếp.
- [`STAGE_ROADMAP.md`](STAGE_ROADMAP.md) — Stage 0–7 research map and expected integration boundaries.
- [`WORKBENCH_GUIDE.md`](WORKBENCH_GUIDE.md) — practical Research Workbench usage.
- [`comparison/`](comparison/COMPARISON_SPEC_V1.md) — Gate C1 scientific comparison specification, field catalog, and comparability rules (draft).
- [`../datasets/mapsuite_v1/README.md`](../datasets/mapsuite_v1/README.md) — dataset card for the packaged MetaDrive MapSuite V1 procedural benchmark.

## 2. Detailed technical evidence

[`environment/`](environment/) contains the detailed Platform V1 audit/specification reports produced while the research foundation was built. They are valuable as technical references and evidence, but a new team member should normally read the six guides above first.

The audit sequence covers:

- observation/action design;
- scenario/map suite;
- episode lifecycle;
- reward and metrics;
- evaluation protocol;
- agent contract;
- logging/W&B/provenance;
- launcher/preflight;
- Workbench, results browser, and final hardening.

[`survey/`](survey/) contains literature-survey material and course research references. It is not an authoritative source for Platform V1 runtime behavior.

## Platform V1 knowledge and derived artifacts

- [Build record](PLATFORM_V1_BUILD_RECORD.md) -- chronological gates, verified PRs/merges and dependencies.
- [Contract index](PLATFORM_V1_CONTRACT_INDEX.md) -- hashes, producing modules and downstream locks.
- [Evidence index](PLATFORM_V1_EVIDENCE_INDEX.md) -- implementation, audits, tests and committed machine evidence.
- [Legacy and boundaries](LEGACY_AND_BOUNDARIES.md) -- prototypes, fixtures and result authority.
- [Course research scope](COURSE_RESEARCH_SCOPE.md) -- conceptual stages and practical priorities.
- [Map artifact distribution](MAP_ARTIFACT_DISTRIBUTION.md) -- reproducible Level 1 previews and mirror policy.
