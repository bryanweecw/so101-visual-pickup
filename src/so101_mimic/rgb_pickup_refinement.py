"""Continue RGB BC, selecting checkpoints by closed-loop development pickups."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .grasp_physics import GraspConfig
from .losses import action_loss
from .rgb_pickup_learning import RGBActor, digest, evaluate, load_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source / "manifest.json").read_text())
    model = source / "model/learning.xml"
    manifest.update(
        architecture="rgb-direct-v1",
        source_run=str(source),
        source_checkpoint_sha256=digest(source / "policy.npz"),
        model_path=str(model),
        test=list(range(1300000, 1300064)),
        stress=list(range(1400000, 1400032)),
        selection="closed-loop development pickups after each 1000 updates; final test once after checkpoint freeze",
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "task.json").write_text((source / "task.json").read_text())
    cfg = GraspConfig(**json.loads((source / "task.json").read_text()))
    data = load_data([source / "train", source / "dagger"])
    arrays = {k: mx.array(data[k]) for k in ("rgb", "proprio", "action", "auxiliary")}
    actor = RGBActor()
    actor.load_weights(str(source / "policy.npz"))
    actor.save_weights(str(out / "policy.npz"))
    baseline = json.loads((source / "round-1/development.json").read_text())

    def key(r):
        return (
            r["successes"],
            float(np.mean([x["peak_lift_m"] for x in r["episodes"]])),
        )

    best = key(baseline)
    selection = [{"step": 0, "evaluation": baseline}]
    chosen = 0
    opt = optim.Adam(learning_rate=1e-4)
    rng = np.random.default_rng(18101)
    mx.random.seed(18101)

    def loss_fn(net, rgb, p, target, aux):
        a, pos, logits = net.forward(rgb, p)
        return action_loss(a, logits, target) + 0.2 * mx.mean((pos - aux) ** 2)

    grad = nn.value_and_grad(actor, loss_fn)
    for step in range(3000):
        ix = mx.array(rng.integers(0, len(data["rgb"]), 96))
        image = arrays["rgb"][ix].astype(mx.float32)
        image = mx.clip(
            image * mx.random.uniform(0.85, 1.15, (96, 1, 1, 1))
            + mx.random.normal(image.shape) * 2,
            0,
            255,
        )
        loss, g = grad(
            actor,
            image,
            arrays["proprio"][ix],
            arrays["action"][ix],
            arrays["auxiliary"][ix],
        )
        opt.update(actor, g)
        mx.eval(actor.parameters(), opt.state, loss)
        if (step + 1) % 1000 == 0:
            dev = evaluate(model, cfg, actor, manifest["development"])
            selection.append({"step": step + 1, "loss": float(loss), "evaluation": dev})
            print("DEVELOPMENT", step + 1, dev["successes"], dev["count"], flush=True)
            if key(dev) > best:
                best = key(dev)
                chosen = step + 1
                actor.save_weights(str(out / "policy.npz"))
    (out / "selection.json").write_text(json.dumps(selection, indent=2))
    actor.load_weights(str(out / "policy.npz"))
    frozen = digest(out / "policy.npz")
    (out / "FROZEN.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": frozen,
                "selected_step": chosen,
                "development_successes": best[0],
            },
            indent=2,
        )
    )
    test = evaluate(model, cfg, actor, manifest["test"], output=out / "test-traces")
    stress = evaluate(
        model,
        cfg,
        actor,
        manifest["stress"],
        radius=0.04,
        noise=1.5,
        output=out / "stress-traces",
    )
    blank = evaluate(model, cfg, actor, manifest["test"][:16], blank_rgb=True)
    assert digest(out / "policy.npz") == frozen
    result = {
        "test": test,
        "stress": stress,
        "blank_rgb_ablation": blank,
        "training_frames": len(data["rgb"]),
        "checkpoint_sha256": frozen,
        "selected_step": chosen,
        "physical_transfer_verified": False,
        "ready_for_real_transfer": False,
    }
    (out / "results.json").write_text(json.dumps(result, indent=2))
    print(
        "FINAL",
        test["successes"],
        test["count"],
        stress["successes"],
        stress["count"],
        flush=True,
    )


if __name__ == "__main__":
    main()
