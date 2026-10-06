# MapSuite V1 lane metadata erratum

**Known historical metadata defect: `declared_base_lane_num = 2`, while `effective_base_lane_num = 3`.** This predates PR #27. It is a descriptive metadata defect in Gate 2 and the derived package, not a new geometry defect introduced by preview export.

## Root cause and affected records

Gate 2's [`audit_mapsuite_candidates.py`](../../scripts/audit_mapsuite_candidates.py) constructed each calibration environment with `map=<sequence>` and no `map_config.lane_num=2` override. Its metrics and configuration writers nevertheless recorded the literal `base_lane_num = 2`. That value describes the intended/recorded setup, not the effective runtime configuration.

Pinned MetaDrive 0.4.3, commit `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`, supplies `BaseMap.LANE_NUM: 3` in `metadrive/envs/metadrive_env.py:METADRIVE_DEFAULT_CONFIG`. Its shorthand parser preserves that default. The lane width is 3.5 m and lane randomization remains disabled. A base lane count does not assert that every ramp/merge segment has that same count.

| Affected historical field/document | Recorded declaration | Effective interpretation |
|---|---|---|
| [`candidate_metrics.csv`](../../results/audits/mapsuite/candidate_metrics.csv), `base_lane_num` | 2 | Calibration actually used the pinned default of 3. |
| [`mapsuite_v1_candidates.json`](../../configs/maps/mapsuite_v1_candidates.json), `metadata.base_lane_num` | 2 | Historical generator description; not an applied lane override. |
| [Gate 2 human audit](PLATFORM_V1_MAPSUITE_AUDIT.md), geometry generator parameters | `base_lane_num = 2` | Read with this erratum. |
| [`geometries.csv`](../../datasets/mapsuite_v1/geometries.csv), `base_lane_num` | 2 | Inherited descriptive field; use frozen geometry identity. |
| [`generation_config.json`](../../datasets/mapsuite_v1/generation_config.json), `lane_configuration.base_lane_num` | 2 | Inherited description, not authority to reconstruct different geometry. |
| [Dataset schema](../../datasets/mapsuite_v1/SCHEMA.md), description of `base_lane_num` | Describes the base lane field | Read the packaged value as a historical declaration, not the effective runtime value. |

The explicit distinction is:

```text
declared_base_lane_num = 2
effective_base_lane_num = 3
```

The preview manifest retains `dataset_declared_base_lane_num = 2` as its name for the historical declaration, separately records `effective_base_lane_num = 3`, and emits a warning. These are different meanings, not interchangeable identities.

## Evidence and benchmark authority

Gate 5's [`audit_evaluation_protocol.py`](../../scripts/audit_evaluation_protocol.py) regenerated the 240 geometries through the same pinned shorthand path and froze hashes over the actual serialized block sequences. The geometry split/case manifests, exact canonical block configurations and `geometry_sha256` values describe those actual geometries.

The real Level 1 exporter reads the effective runtime `env.config['map_config']['lane_num']` and requires every reconstructed block sequence to match its recorded hash before rendering. All **240/240** frozen geometry hashes matched the effective three-lane path, with **zero reconstruction/hash failures**. Runtime values and the full export result are committed in [`verification.json`](../../results/audits/mapsuite_previews/verification.json) and explained in the [export audit](PLATFORM_V1_MAPSUITE_PREVIEW_EXPORT_AUDIT.md). The evidence covers geometry reconstruction only, not scientific agent performance.

The actual benchmark remains the frozen serialized geometry/hash identity. Forcing `lane_num=2` now would generate different geometry rather than repair a label; the exporter must reject any resulting hash mismatch. PNGs and a descriptive CSV field cannot override that identity.

## Why PR #27 preserves frozen metadata

Rewriting the package fields would change checksummed files. Rewriting Gate 2 artifacts or Gate 5 manifests would also change locked provenance or scientific contract identities. This correction pass leaves those files, raw checksums, splits, seeds and all 17 recorded scientific/registry/manifest hashes untouched. It adds an explicit interpretive erratum and verifiable export provenance.

## Requirements for a future versioned correction

A separately reviewed correction must identify every affected field/document and choose whether it is only correcting descriptions or proposing different geometries. A descriptive correction must preserve all 240 serialized geometries, IDs, seeds, splits and canonical evaluation cases; version the package metadata; regenerate its package checksums/provenance; document old/new identities and update any downstream contracts that bind changed source bytes through explicit review. Preserve the original version for reproducibility.

If actual two-lane geometry is desired, treat that as a new benchmark version with new geometry/hash identities and scientific review. Do not call it a non-semantic metadata correction, and do not alter Platform V1 silently.
