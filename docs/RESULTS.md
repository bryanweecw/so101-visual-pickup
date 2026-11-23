# Wrist RGB pickup training results - 2026-11-23

Selected by development rollouts: `rgb-pickup-refinement-001` (8/12), then frozen before a fresh final test. All results below are simulation only. No real arm, camera or motor settings were accessed.

| Candidate | Unseen pickups | Wider/noisier stress |
|---|---:|---:|
| rgb-pickup-training-001 | 26/64 | 13/32 |
| rgb-pickup-latent-001 | 0/64 | 2/32 |
| rgb-pickup-recurrent-001 | 31/64 | 11/32 |
| rgb-pickup-refinement-001 | 50/64 | 18/32 |

Selected model: **50/64 (78.1%)** standard test pickups; Wilson 95% interval **66.6-86.5%**. Stress: **18/32 (56.3%)**. These are conditional simulation success rates, not estimates of real transfer confidence. Each candidate has a separate predeclared final-test seed set; this is not a paired causal comparison.

On the first 16 selected-model cases: normal RGB **10/16**, blank RGB **0/16**. Removing the image is an out-of-distribution ablation; it demonstrates visual dependence, not real-world robustness.

The selected model was trained on **8,904 frames from 175 retained training rollouts**: 111 successful initial teacher episodes and 64 mixed student/teacher recovery episodes. Initial collection attempted 192 episodes, with 81 failures logged rather than silently discarded. Validation used 902 frames from 18 successful teacher episodes out of 24 attempted, on separate seeds. Student-only recovery data collected for the rejected latent candidate did not enter the selected model.

Every standard test varies actual cube/arm initial positions, camera/geometry residuals, friction, mass, servo gain, colors, lighting, image corruption and small occlusion. The exact bounds are in `manifest.json` and the training guide. Standard positions are only +/-25 mm around one nominal reach; this does not qualify arbitrary tables or clutter. The assumed wrist camera is 105 degrees FOV with +/-3 degree variation; it is not a measured replica of the real wrist camera. Its geometry and image compatibility must be checked before hardware transfer.

Success requires a >=50 mm lift and continuous two-pad contact for >=2 seconds. No release or placement is qualified. The simulator uses approximate pad/servo physics and disables imported arm-mesh collisions.

The latent-position candidate failed even in a privileged exact-position motor diagnostic (0/12); no oracle result is counted as visual-policy success. Command smoothing did not improve the original direct policy on development cases. A GRU improved the development score modestly but was weaker than the refined direct candidate. The useful improvement came from continuing training and selecting checkpoints by completed development pickups rather than minimum frame-prediction loss.

## Artifacts

- `policy.npz`: selected frozen CNN/MLP weights; trained entirely in simulation.
- `manifest.json`: observation/action protocol, randomization, splits, source data and model path.
- `FROZEN.json`: weight digest and selected update count.
- `results.json`: every test, stress and ablation outcome.
- `RELEASE-VALIDATION.json`: isolated package tests and eight-seed smoke replay; the full benchmark was not rerun for publication.
- `first-four-test-cases.mp4`: 26.65 seconds, H.264/yuv420p, about 1.04 MB. First four assigned test cases, with two passes and two failures. External view is recording only; inset is actual policy RGB input.

Replayed recording outcomes match the original first four test outcomes and peak lifts. 32 focused tests passed; one legacy two-camera RGB-D sensor test was intentionally excluded. Runtime LLM calls and tokens are zero; development/conversation usage is separate.

**Not ready for unattended real transfer.** Keep this checkpoint frozen for future calibration and physical verification. Broader generalization and the 14 standard-test failures remain unresolved. Do not interpret repeated model attempts as a universally calibrated SO-101 policy.
