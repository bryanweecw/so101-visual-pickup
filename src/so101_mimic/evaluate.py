"""Replay frozen visual policy on declared simulation seeds."""

import argparse
import json
from pathlib import Path

from .rgb_pickup import prepare
from .rgb_pickup_learning import evaluate, load_rgb_actor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("evidence"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    actor, manifest = load_rgb_actor(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    model, cfg = prepare(args.output)
    result = {}
    for name, radius, noise in [("test", 0.025, 1.0), ("stress", 0.04, 1.5)]:
        seeds = manifest[name][: args.limit]
        result[name] = evaluate(model, cfg, actor, seeds, radius=radius, noise=noise)
    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                k: {"successes": v["successes"], "count": v["count"]}
                for k, v in result.items()
            }
        )
    )


if __name__ == "__main__":
    main()
