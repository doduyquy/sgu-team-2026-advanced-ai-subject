# Datasets Registry

This directory serves as the structured registry of research datasets for the autonomous-driving subproject.

## Current Packaged Datasets

### 1. [`mapsuite_v1/`](mapsuite_v1/) — MetaDrive MapSuite V1
- **Dataset ID:** `mapsuite_v1`
- **Type:** Procedural autonomous-driving scenario benchmark dataset
- **Status:** **FROZEN / ACTIVE** (derived from Platform V1 benchmark artifacts)
- **Scale:** 12 scenario families × 20 seeds = 240 procedural geometries
- **Splits:** 180 TRAIN / 48 VALIDATION / 12 TEST geometries
- **Evaluation Suites:** 96 canonical validation cases / 60 canonical test cases
- **Documentation:** [`mapsuite_v1/README.md`](mapsuite_v1/README.md) (Dataset Card), [`mapsuite_v1/SCHEMA.md`](mapsuite_v1/SCHEMA.md), [`mapsuite_v1/REPRODUCE.md`](mapsuite_v1/REPRODUCE.md)

---

## Future Dataset Categories (Planning / Not Yet Existing)

Future research stages may introduce separate dataset packages. These are distinct research products and do **not** replace the MapSuite V1 scenario benchmark:

1. **Demonstration & Imitation Datasets (Future Stage 4):**
   - Expert driving trajectories collected from human drivers, heuristic rule policies, or privileged planners for behavioral cloning and offline imitation.
   - *Status: Not yet collected or implemented.*
2. **Offline Reinforcement Learning Datasets (Future Stages 4/5):**
   - Fixed replay buffers of transition tuples `(s, a, r, s', done)` across diverse driving behaviors for offline policy optimization.
   - *Status: Not yet collected or implemented.*
3. **Vietnam-Oriented Scenario & Map Datasets (Future Scope / Extensions):**
   - Localized urban traffic scenarios, mixed motorcycle interactions, roundabout patterns, or OSM-derived road geometries representing Vietnamese traffic conditions.
   - *Status: Exploration only; not part of Platform V1 MapSuite.*

Any future dataset will receive its own directory under `datasets/`, versioned manifest, and dataset card.
