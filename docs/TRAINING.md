# Simulation-trained wrist RGB pickup baseline

This experiment trains a compact MLX neural network locally on the Mac. It uses the installed public RUA SO-101 assets and our standalone MuJoCo contact approximations. It accesses no real robot, leader, follower or camera. It does not apply changes to the existing hardware configuration.

The [completed results](RESULTS.md) select `runs/rgb-pickup-refinement-001`: 50/64 standard held-out pickups and 18/32 stress cases. The initial direct, latent-position and recurrent candidates are retained for comparison. The selected model uses the direct CNN/MLP architecture; it does not use recurrent memory or the rejected latent motor decoder.

## What the policy sees and controls

The neural network receives one 80x80 RGB image from a camera rigidly mounted on the rotating gripper, six encoder positions including the jaw, six scaled encoder velocities, and its previous six-dimensional action. It outputs five bounded absolute joint targets and a jaw command at 10 Hz. It uses a CNN and an MLP, trained from scratch; there are no downloaded vision foundation weights. An auxiliary cube-relative-position loss is used only in training. The deployed actor reads neither simulator object coordinates, segmentation, depth, contact counts, expert phases nor rewards. No IK or teacher runs in final evaluation.

The objective is one 30 mm cube lifted at least 50 mm with continuous two-pad contact for two seconds. This experiment does not qualify release, placement, selecting among multiple cubes, full-arm collision safety or hardware transfer. Imported robot mesh collisions are disabled in this inherited contact model; finger-pad, cube and table contacts are enabled. Finger collision boxes and servos remain physical approximations. There are no object attachments or imposed object-lift forces.

## Randomization and data quality

Training starts the cube within +/-25 mm in X and Y of the nominal 235 mm reach position, and varies five arm starting joints by up to [0.20, 0.12, 0.12, 0.15, 0.25] radians. It also varies initial jaw opening, cube yaw (+/-0.18 rad), mass (25-60 g), contact friction (0.45-0.9), servo gain (+/-10%), residual joint-spacing scale (+/-1.5%), wrist-camera translation (+/-2 mm) and field of view (+/-3 degrees). Cube colors are red, white and orange. Table/arm colors, lighting, brightness, gamma, gradients, Gaussian noise, blur, sparse pixel artifacts and occasional small 5x5-pixel occluders vary. Photometric augmentation is regenerated on optimizer steps.

These are bounded randomizations, not coverage of arbitrary real environments. Camera rotation errors, prolonged/full occlusion, clutter and multiple similar objects are not qualified. A calibration step is still necessary before hardware trials. Domain randomization does not replace physical verification.

Initial training keeps only successful deterministic teacher rollouts, while retaining the outcomes of every attempted episode. The subsequent DAgger pass keeps teacher-labelled states visited by a mixture of student execution (65% probability) and teacher execution (35%). This teacher assistance exists during simulation training only. A large frame count alone is not an acceptance condition: closed-loop held-out success is the relevant result.

## Splits and validation

The run manifest fixes non-overlapping seeds before collection: training 100000+, label-validation 200000+, development rollouts 300000+, DAgger training 400000+, final test 500000+, and wider stress cases 600000+. Episode boundaries, rather than random neighboring frames, define the split. Only successful teacher label-validation episodes enter imitation-loss validation. Final evaluation includes all assigned cases, with no filtering for teacher success.

Imitation-loss validation selects a candidate within each round. Separate development rollouts choose between rounds. Checkpoint SHA256 is frozen before final test cases run, and checked afterward. Test reporting includes counts and a Wilson 95% interval. Stress tests increase position radius to 40 mm and residual/image noise to 1.5 times the training setting. An additional zero-RGB ablation measures the same actor with its visual input removed; compare this subset with its paired normal-image cases before attributing success to vision.

A reset bug found in the initial pilot was fixed before the main dataset: `mj_setConst` restored the cube's default pose. The fixed reset preserves the sampled cube pose across that call. Automated tests check actual cube displacement across seeds, deterministic resets without accumulated scale, sensor exclusion of hidden object state, corruption reproducibility, teacher-free inference and checkpoint loading.

## Reproduce

From the project directory, with local macOS graphics and MLX GPU access:

```sh
.venv/bin/python -m so101_mimic.rgb_pickup_learning \
  --output runs/rgb-pickup-NEW \
  --episodes 192 --steps 3000 --dagger-episodes 64 --test-episodes 64
```

Use a fresh output directory; existing runs are preserved. Dataset NPZ files store RGB, proprioception, expert action labels, training-only auxiliary labels and episode IDs. `outcomes.json` records teacher/mixture failures. The export contains `policy.npz`, `manifest.json`, `FROZEN.json`, and `results.json`. Do not call a checkpoint ready based only on its existence or training loss.

Then refine the direct policy, choosing among 0/1000/2000/3000 additional updates by completed development pickups instead of imitation loss. This stage preserves the source run and assigns a fresh final test:

```sh
.venv/bin/python -m so101_mimic.rgb_pickup_refinement \
  --source runs/rgb-pickup-NEW --output runs/rgb-pickup-refined-NEW
```

Load the frozen checkpoint:

```python
from so101_mimic.rgb_pickup_learning import load_rgb_actor
actor, metadata = load_rgb_actor("runs/rgb-pickup-refinement-001")
action = actor.act(rgb_uint8_80x80x3, proprio_float32_length18)
```

The loader verifies the frozen checkpoint digest. This is an inference interface, not a hardware motion adapter. Physical execution must retain native robot authority, calibrated coordinate/action mapping, speed limits and guards. No automatic hardware execution is provided by this training command.

The first four assigned test cases are included in `evidence/first-four-test-cases.mp4`. Use the evaluation command in the root README to replay declared seeds. Runtime LLM calls and token counts are zero; development/conversation usage is separate.
