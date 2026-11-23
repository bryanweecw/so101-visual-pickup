import json
from types import SimpleNamespace

import numpy as np
import pytest

from so101_mimic.rgb_pickup import PickupWorld, WristSensor, corrupt_rgb, prepare


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    folder = tmp_path_factory.mktemp("rgb-pickup")
    model, cfg = prepare(folder)
    return PickupWorld(model, cfg, radius=0.025)


def test_randomization_changes_actual_cube_and_arm_positions(world):
    positions = []
    joints = []
    for seed in range(31, 41):
        world.reset(seed)
        positions.append(world.data.xpos[world.block, :2].copy())
        joints.append(world.data.qpos[:5].copy())
        np.testing.assert_allclose(
            world.data.xpos[world.block, :2],
            [world.config.object_x, world.config.object_y],
            atol=0.001,
        )
        assert world.config.width == 0.03
    assert np.ptp(positions, axis=0).min() > 0.025
    assert np.ptp(joints, axis=0).min() > 0.08


def test_resets_repeat_exactly_without_accumulating_scale(world):
    world.reset(42)
    q = world.data.qpos.copy()
    body = world.model.body_pos.copy()
    spec = json.dumps(world.episode_spec, sort_keys=True)
    world.reset(73)
    world.reset(42)
    np.testing.assert_allclose(world.data.qpos, q, atol=1e-10)
    np.testing.assert_array_equal(world.model.body_pos, body)
    assert json.dumps(world.episode_spec, sort_keys=True) == spec


def test_rgb_sensor_excludes_hidden_object_state(monkeypatch, world):
    class FakeRenderer:
        def __init__(self, *args, **kwargs):
            pass

        def update_scene(self, data, camera):
            assert camera == "pickup_wrist"

        def render(self):
            return np.full((80, 80, 3), 100, dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr("so101_mimic.rgb_pickup.mujoco.Renderer", FakeRenderer)
    a = WristSensor(world.model, 9)
    b = WristSensor(world.model, 9)
    first = SimpleNamespace(qpos=np.arange(13, dtype=float), qvel=np.zeros(12))
    second = SimpleNamespace(
        qpos=np.r_[np.arange(6, dtype=float), [900] * 7], qvel=np.zeros(12)
    )
    rgb1, p1 = a.capture(first, np.zeros(6))
    rgb2, p2 = b.capture(second, np.zeros(6))
    np.testing.assert_array_equal(rgb1, rgb2)
    np.testing.assert_array_equal(p1, p2)
    assert p1.shape == (18,)


def test_corruption_is_seeded_and_changes_image():
    rgb = np.full((80, 80, 3), 100, dtype=np.uint8)
    a = corrupt_rgb(rgb, np.random.default_rng(1))
    b = corrupt_rgb(rgb, np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, rgb)
    assert a.dtype == np.uint8
    np.testing.assert_array_equal(corrupt_rgb(rgb, np.random.default_rng(1), 0), rgb)


def test_actor_checkpoint_roundtrip_no_teacher_calls(tmp_path, monkeypatch):
    from so101_mimic.rgb_pickup_learning import RGBActor

    def forbidden(*args, **kwargs):
        raise AssertionError("Teacher/IK invoked during inference")

    monkeypatch.setattr("so101_mimic.rgb_pickup.feedback_expert", forbidden)
    monkeypatch.setattr("so101_mimic.grasping.grasp_ik", forbidden)
    actor = RGBActor()
    rgb = np.full((80, 80, 3), 120, dtype=np.uint8)
    p = np.zeros(18, dtype=np.float32)
    first = actor.act(rgb, p)
    path = tmp_path / "weights.npz"
    actor.save_weights(str(path))
    loaded = RGBActor()
    loaded.load_weights(str(path))
    np.testing.assert_array_equal(first, loaded.act(rgb, p))
    assert first.shape == (6,) and np.isfinite(first).all()
    assert np.max(np.abs(first)) <= 1


def test_frozen_loader_verifies_checkpoint_hash(tmp_path):
    from so101_mimic.rgb_pickup_learning import RGBActor, digest, load_rgb_actor

    actor = RGBActor()
    actor.save_weights(str(tmp_path / "policy.npz"))
    (tmp_path / "manifest.json").write_text(
        json.dumps({"privileged_state_at_inference": False})
    )
    (tmp_path / "FROZEN.json").write_text(
        json.dumps({"checkpoint_sha256": digest(tmp_path / "policy.npz")})
    )
    loaded, _ = load_rgb_actor(tmp_path)
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    p = np.zeros(18, dtype=np.float32)
    np.testing.assert_array_equal(actor.act(image, p), loaded.act(image, p))
    (tmp_path / "FROZEN.json").write_text(json.dumps({"checkpoint_sha256": "invalid"}))
    with pytest.raises(ValueError, match="digest"):
        load_rgb_actor(tmp_path)
