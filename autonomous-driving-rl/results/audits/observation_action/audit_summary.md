# Gate 1 Observation and Action Audit Summary

## 1. Verified MetaDrive Source
- **Pinned Commit:** `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`
- **Version:** MetaDrive 0.4.3

## 2. Key Observation Discoveries
- **Ego Last Action Ordering:** The MetaDrive docstring claimed throttle precedes steering. Source code and empirical validation prove that index 5 is STEERING and index 6 is THROTTLE/BRAKE.
- **LiDAR Sweep Direction:** The MetaDrive docstring claimed clockwise sweep. Source code and empirical cone testing prove that ray 0 is Forward (0 deg), ray 60 is Left (+90 deg), ray 120 is Rear (180 deg), ray 180 is Right (270 deg), meaning an exact COUNTER-CLOCKWISE sweep.
- **LiDAR Static Detection:** LiDAR dynamic world detects dynamic vehicles and static traffic obstacles (cones, barriers) but DOES NOT detect road borders, curbs, or sidewalks.
- **Surrounding Vehicles:** With `num_others=4`, 16 features are added (longitudinal/lateral relative position and velocity). These features use broad-phase cylinder query bypassing ray occlusion and directly read physics state (privileged state leakage). Padding for absent vehicles is `0.0`, whereas zero relative displacement/velocity normalizes to `0.5`.

## 3. Control Frequency Findings
- `physics_world_step_size = 0.02` s
- `decision_repeat = 5`
- Nominal step dt = 0.1 s (~10 Hz agent decision rate).
- Empirically verified effective dt: 0.0955 s.

## 4. Determinism Findings
- Exact bit-level determinism confirmed: True.
- Max position deviation across runs with identical scenario seed and actions: 0.00e+00 m.

## 5. Candidate Spaces Evaluated
- **Candidate A (259D):** MetaDrive default (9 ego + 10 nav + 240 lidar). Dynamic-object information is raw point cloud only; no surrounding vehicle state.
- **Candidate B (275D):** MetaDrive with `num_others=4` (9 ego + 10 nav + 16 surrounding + 240 lidar). High information sufficiency, but surrounding vehicle features leak privileged physics state.
- **Candidate C (35D):** CourseEnv compressed (9 ego + 10 nav + 16 min-pooled lidar sectors). Low dimensional, fast, but suffers significant loss of fine-grained spatial and dynamic object details.

## 6. Action Space Findings
- Native MetaDrive action is `Box(-1, 1, shape=(2,))` representing `[steering, throttle_brake]`.
- Native discrete mapping `Discrete(25)` implements a symmetric 5x5 grid using `steering = (a % 5) * 0.5 - 1.0` and `throttle = (a // 5) * 0.5 - 1.0`.
- `CourseEnvV1`'s `Discrete(5)` locks steering to throttle, prevents braking while steering, and omits idle/coasting.
