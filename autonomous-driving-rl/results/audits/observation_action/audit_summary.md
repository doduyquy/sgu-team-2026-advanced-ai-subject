# Gate 1 Observation and Action Audit Summary

## 1. Verified MetaDrive Source
- **Pinned Commit:** `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`
- **Package Version:** `0.4.3`
- **Git Status:** Verified clean working tree.

## 2. Key Observation Discoveries
- **Ego Last Action Ordering:** The MetaDrive docstring claimed throttle precedes steering. Source code and empirical validation prove that index 5 is STEERING and index 6 is THROTTLE/BRAKE.
- **Observation Semantics:** Road borders (`dist_to_left/right_side`), lane offsets, and heading deviation are geometric state abstractions, not side-detector rays. `heading_diff` is a normalized signed proxy based on dot product with the lane lateral vector.
- **LiDAR Sweep Direction:** The MetaDrive docstring claimed clockwise sweep. Source code and empirical testing prove that ray 0 is Forward (0 deg), ray 60 is Left (+90 deg), ray 120 is Rear (180 deg), ray 180 is Right (270 deg) — an exact COUNTER-CLOCKWISE sweep.
- **LiDAR Detection Targets:** LiDAR detects dynamic vehicles and static traffic obstacles (cones, barriers). Sidewalks and lane lines are excluded because Lidar.mask is set to CollisionGroup.can_be_lidar_detected(), which excludes Sidewalk, ContinuousLaneLine, and BrokenLaneLine.
- **Surrounding Vehicles:** With `num_others=4`, 16 features are added. These features query physics state directly via a 50m cylinder, bypassing LiDAR ray occlusion (privileged state leakage). Padding for absent vehicles is `0.0`, whereas zero relative delta normalizes to `0.5`.

## 3. Control Frequency Findings
- `physics_world_step_size = 0.02` s
- `decision_repeat = 5`
- Nominal step dt = 0.1 s (~10 Hz agent decision rate).
- Empirically verified effective dt: 0.0955 s.

## 4. Determinism & Repeatability Findings
- **Finding:** Exact repeatability was observed under the tested configuration across multiple scenario seeds (seeds 42 and 101).
- All observation components, rewards, positions, headings, velocities, and termination signals matched across repeated runs.
- Scenario seed (`env_seed`) is formally distinguished from agent exploratory RNG seed.

## 5. Candidate Spaces Evaluated
- **Candidate A (259D):** MetaDrive default local state + LiDAR. State-based local observation without explicit surrounding vehicle tracks.
- **Candidate B (275D):** MetaDrive state + navigation + 4 surrounding vehicles + LiDAR. High dynamic obstacle information, but leaks privileged physics state.
- **Candidate C (35D):** CourseEnv compressed (9 ego + 10 nav + 16 min-pooled LiDAR sectors). Irreversibly loses spatial resolution and dynamic velocities.
- **Candidate D (Open Concept):** Structured AgentInput concept separating CoreObservation (259D), optional TrafficContext (with validity mask), and TaskContext (read-only map context).

## 6. Action Space Findings
- Native MetaDrive action is continuous `Box(-1, 1, shape=(2,))` representing `[steering, throttle_brake]`.
- Native discrete mapping `Discrete(25)` implements a symmetric 5x5 grid.
- `CourseEnvV1` Discrete(5) embeds throttle into steering, lacks evasive braking, and omits idle/coasting.
- Mirror-symmetric steering response was observed in the tested 15-step configuration.
- Controlled braking from ~19.9 km/h confirms monotonic stopping distances: 3.92m (-0.25), 2.06m (-0.50), and 1.48m (-1.00).
