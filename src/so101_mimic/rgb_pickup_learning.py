"""MLX RGB behavioral cloning with simulation-only DAgger and held-out rollouts."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .physics import HOME
from .policy_world import JOINT_SCALE
from .rgb_pickup import PickupWorld, WristSensor, prepare, teacher_episode


class RGBActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = [
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.Conv2d(16, 24, 3, stride=2, padding=1),
            nn.Conv2d(24, 32, 3, stride=2, padding=1),
        ]
        self.visual = nn.Linear(3200, 128)
        self.layers = [nn.Linear(128 + 18, 128), nn.Linear(128, 128)]
        self.action_head = nn.Linear(128, 6)
        self.auxiliary = nn.Linear(128, 3)

    def forward(self, image, proprio):
        h = image / 127.5 - 1
        for conv in self.convs:
            h = nn.relu(conv(h))
        h = nn.relu(self.visual(h.reshape(h.shape[0], -1)))
        # Encoder noise is explicit; no object positions or phase supplied.
        h = mx.concatenate([h, proprio], axis=-1)
        for layer in self.layers:
            h = nn.relu(layer(h))
        logits = self.action_head(h)
        current = (proprio[:, :5] - mx.array(HOME[:5])) / mx.array(JOINT_SCALE)
        action = mx.concatenate(
            [
                mx.clip(current + 0.12 * mx.tanh(logits[:, :5]), -0.98, 0.98),
                mx.tanh(logits[:, 5:6]),
            ],
            axis=-1,
        )
        return action, self.auxiliary(h), logits

    def __call__(self, image, proprio):
        return self.forward(image, proprio)[0]

    def act(self, image, proprio):
        return np.array(
            self(mx.array(image[None].astype(np.float32)), mx.array(proprio[None]))
        )[0]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(model, cfg, folder, seeds, *, policy=None, mix=0.0, radius=0.025):
    folder.mkdir(parents=True, exist_ok=False)
    world = PickupWorld(model, cfg, radius=radius)
    outcomes = []
    for idx, seed in enumerate(seeds):
        sensor = WristSensor(world.model, seed)
        try:
            records, row = teacher_episode(
                world, seed, sensor=sensor, policy=policy, mix=mix
            )
        finally:
            sensor.close()
        # Initial demonstrations must complete pickup. DAgger retains recovery labels.
        keep = bool(records) and (row["success"] or policy is not None)
        row.update(kept=keep, labels=len(records), seed=int(seed))
        outcomes.append(row)
        if keep:
            rgb, p, a, aux = zip(*records, strict=True)
            np.savez_compressed(
                folder / f"episode-{seed}.npz",
                rgb=np.array(rgb),
                proprio=np.array(p),
                action=np.array(a),
                auxiliary=np.array(aux),
                episode=np.full(len(rgb), seed),
            )
        if (idx + 1) % 8 == 0:
            print(
                json.dumps(
                    {
                        "collection": folder.name,
                        "episodes": idx + 1,
                        "successful": sum(r["success"] for r in outcomes),
                        "retained": sum(r["kept"] for r in outcomes),
                    }
                ),
                flush=True,
            )
    (folder / "outcomes.json").write_text(json.dumps(outcomes, indent=2))
    return outcomes


def load_data(folders):
    chunks = []
    for folder in folders:
        for path in sorted(folder.glob("episode-*.npz")):
            with np.load(path) as f:
                chunks.append({k: f[k] for k in f.files})
    if not chunks:
        raise ValueError("No successful demonstrations")
    return {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}


def evaluate(
    model, cfg, actor, seeds, *, radius=0.025, noise=1.0, output=None, blank_rgb=False
):
    world = PickupWorld(model, cfg, radius=radius, noise=noise)
    rows = []
    for seed in seeds:
        world.reset(seed)
        sensor = WristSensor(world.model, seed, noise=noise)
        done = False
        try:
            for _ in range(180):
                image, p = sensor.capture(world.data, world.previous_action)
                if blank_rgb:
                    image = np.zeros_like(image)
                action = actor.act(image, p)
                _, _, done = world.step(action)
                if done:
                    break
        finally:
            sensor.close()
        row = {
            "seed": int(seed),
            "spec": world.episode_spec,
            "success": world.held_seconds >= 2.0,
            "peak_lift_m": world.peak_lift,
            "hold_s": world.held_seconds,
            "steps": world.steps,
        }
        rows.append(row)
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            (output / f"trace-{seed}.json").write_text(json.dumps(world.trace))
        if len(rows) % 8 == 0:
            print(
                f"Evaluation {len(rows)}/{len(seeds)}: {sum(r['success'] for r in rows)} successful",
                flush=True,
            )
    n = len(rows)
    s = sum(r["success"] for r in rows)
    # Wilson 95% interval, to avoid presenting tiny evaluation sets as certainty.
    z = 1.96
    mid = (s / n + z * z / (2 * n)) / (1 + z * z / n)
    half = (
        z * np.sqrt((s / n) * (1 - s / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    )
    return {
        "successes": s,
        "count": n,
        "rate": s / n,
        "wilson95": [mid - half, mid + half],
        "radius_m": radius,
        "noise": noise,
        "blank_rgb": blank_rgb,
        "episodes": rows,
    }


def train(actor, data, validation, steps, seed, output):
    rng = np.random.default_rng(seed)
    optimizer = optim.Adam(learning_rate=3e-4)
    arrays = {k: mx.array(v) for k, v in data.items() if k != "episode"}
    val_idx = np.arange(
        0, len(validation["rgb"]), max(1, len(validation["rgb"]) // 512)
    )[:512]
    val = {k: mx.array(v[val_idx]) for k, v in validation.items() if k != "episode"}

    def loss_fn(net, image, p, target, aux):
        action, position, logits = net.forward(image, p)
        arm = mx.mean(((action[:, :5] - target[:, :5]) / 0.12) ** 2)
        closed = (target[:, 5] < 0).astype(mx.float32)
        # tanh jaw logit; closed is negative.
        jaw = mx.mean(
            mx.logaddexp(0.0, 2 * logits[:, 5]) - 2 * logits[:, 5] * (1 - closed)
        )
        return arm + 0.3 * jaw + 0.2 * mx.mean((position - aux) ** 2)

    grad = nn.value_and_grad(actor, loss_fn)
    best = float("inf")
    history = []
    started = time.time()
    for step in range(steps):
        ids = mx.array(rng.integers(0, len(data["rgb"]), 96))
        image = arrays["rgb"][ids].astype(mx.float32)
        # Fresh photometric augmentation each optimizer step, beyond stored noise.
        image = mx.clip(
            image * mx.random.uniform(0.85, 1.15, (96, 1, 1, 1))
            + mx.random.normal(image.shape) * 2,
            0,
            255,
        )
        loss, grads = grad(
            actor,
            image,
            arrays["proprio"][ids],
            arrays["action"][ids],
            arrays["auxiliary"][ids],
        )
        optimizer.update(actor, grads)
        mx.eval(actor.parameters(), optimizer.state, loss)
        if (step + 1) % 250 == 0 or step == steps - 1:
            vl = float(
                loss_fn(
                    actor,
                    val["rgb"].astype(mx.float32),
                    val["proprio"],
                    val["action"],
                    val["auxiliary"],
                )
            )
            row = {
                "step": step + 1,
                "loss": float(loss),
                "validation_loss": vl,
                "seconds": time.time() - started,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            if vl < best:
                best = vl
                actor.save_weights(str(output / "candidate.npz"))
    actor.load_weights(str(output / "candidate.npz"))
    (output / "optimization.json").write_text(json.dumps(history, indent=2))
    return history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=192)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--dagger-episodes", type=int, default=64)
    p.add_argument("--test-episodes", type=int, default=64)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    mx.random.seed(8101)
    model, cfg = prepare(out)
    manifest = {
        "train": list(range(100000, 100000 + args.episodes)),
        "validation": list(range(200000, 200024)),
        "development": list(range(300000, 300012)),
        "dagger": list(range(400000, 400000 + args.dagger_episodes)),
        "test": list(range(500000, 500000 + args.test_episodes)),
        "stress": list(range(600000, 600032)),
        "sensor": "single rotating-gripper wrist RGB camera, 80x80, plus measured joints/velocities/previous action",
        "privileged_state_at_inference": False,
        "task": "30mm cube pickup >=50mm, two-pad hold >=2s",
        "runtime_llm_calls": 0,
        "runtime_llm_input_tokens": 0,
        "runtime_llm_output_tokens": 0,
        "physical_hardware_used": False,
        "randomization": {
            "cube_xy_radius_m": 0.025,
            "arm_start_jitter_rad": [0.2, 0.12, 0.12, 0.15, 0.25],
            "cube_yaw_rad": [-0.18, 0.18],
            "mass_kg": [0.025, 0.06],
            "friction": [0.45, 0.9],
            "servo_gain_multiplier": [0.9, 1.1],
            "link_spacing_scale": [0.985, 1.015],
            "camera_translation_m": [-0.002, 0.002],
            "camera_fov_jitter_deg": [-3, 3],
            "colors": ["red", "white", "orange"],
            "image_noise_std": 0.018,
            "partial_occlusion_probability": 0.15,
            "occluder_pixels": [5, 5],
        },
        "selection": "development rollouts only; final test seeds evaluated once after checkpoint freeze",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    started = time.time()
    collect(model, cfg, out / "train", manifest["train"])
    collect(model, cfg, out / "validation", manifest["validation"])
    data = load_data([out / "train"])
    validation = load_data([out / "validation"])
    actor = RGBActor()
    round0 = out / "round-0"
    round0.mkdir()
    train(actor, data, validation, args.steps, 8101, round0)
    baseline = evaluate(model, cfg, actor, manifest["development"])
    (round0 / "development.json").write_text(json.dumps(baseline, indent=2))
    actor.save_weights(str(out / "policy.npz"))
    chosen = baseline
    chosen_round = 0
    if args.dagger_episodes:
        collect(model, cfg, out / "dagger", manifest["dagger"], policy=actor, mix=0.65)
        data = load_data([out / "train", out / "dagger"])
        round1 = out / "round-1"
        round1.mkdir()
        train(actor, data, validation, args.steps // 2, 8102, round1)
        dev = evaluate(model, cfg, actor, manifest["development"])
        (round1 / "development.json").write_text(json.dumps(dev, indent=2))
        if (dev["successes"], np.mean([r["peak_lift_m"] for r in dev["episodes"]])) > (
            chosen["successes"],
            np.mean([r["peak_lift_m"] for r in chosen["episodes"]]),
        ):
            actor.save_weights(str(out / "policy.npz"))
            chosen = dev
            chosen_round = 1
    actor.load_weights(str(out / "policy.npz"))
    frozen = digest(out / "policy.npz")
    (out / "FROZEN.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": frozen,
                "selected_round": chosen_round,
                "development_successes": chosen["successes"],
                "test_started_after_freeze": True,
            },
            indent=2,
        )
    )
    tests = evaluate(model, cfg, actor, manifest["test"], output=out / "test-traces")
    stress = evaluate(
        model,
        cfg,
        actor,
        manifest["stress"],
        radius=0.04,
        noise=1.5,
        output=out / "stress-traces",
    )
    ablation = evaluate(model, cfg, actor, manifest["test"][:16], blank_rgb=True)
    if digest(out / "policy.npz") != frozen:
        raise RuntimeError("Checkpoint changed during test")
    result = {
        "test": tests,
        "stress": stress,
        "blank_rgb_ablation": ablation,
        "training_frames": len(data["rgb"]),
        "validation_frames": len(validation["rgb"]),
        "checkpoint_sha256": frozen,
        "selected_round": chosen_round,
        "seconds": time.time() - started,
        "physical_transfer_verified": False,
        "ready_for_real_transfer": False,
    }
    (out / "results.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("test", "stress", "blank_rgb_ablation")
            }
        ),
        flush=True,
    )
    print(
        f"FINAL held-out pickup: {tests['successes']}/{tests['count']}; stress {stress['successes']}/{stress['count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()


def load_rgb_actor(directory):
    """Load a frozen export, checking its checkpoint digest before inference."""
    directory = Path(directory)
    metadata = json.loads((directory / "manifest.json").read_text())
    frozen = json.loads((directory / "FROZEN.json").read_text())
    if metadata.get("privileged_state_at_inference") is not False:
        raise ValueError("Unsupported or privileged policy input schema")
    if digest(directory / "policy.npz") != frozen["checkpoint_sha256"]:
        raise ValueError("Checkpoint digest mismatch")
    actor = RGBActor()
    actor.load_weights(str(directory / "policy.npz"))
    return actor, metadata
