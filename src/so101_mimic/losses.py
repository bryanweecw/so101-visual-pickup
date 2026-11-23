"""Balanced joint and jaw imitation loss."""

import mlx.core as mx


def action_loss(action, logits, target):
    arms = mx.mean(((action[:, :5] - target[:, :5]) / 0.12) ** 2)
    opened = (target[:, 5] > 0).astype(mx.float32)
    jaw = mx.mean(mx.logaddexp(0, 2 * logits[:, 5]) - 2 * logits[:, 5] * opened)
    return arms + 0.3 * jaw
