"""Wrist RGB-only learned cube pickup. Simulation only; no hardware imports.

Simulator object state is confined to teacher labels, reset and scoring.
Policy inputs are RGB, encoder readings and its previous action.
"""

import json
import xml.etree.ElementTree as ET
from dataclasses import asdict

import cv2
import mujoco
import numpy as np

from .grasp_physics import GraspConfig, build_grasp_model
from .physics import HOME
from .policy_world import (
    ManipulationWorld,
    feedback_expert,
    prepare_learning_model,
)

IMAGE_SIZE = 80


def prepare(output):
    cfg = GraspConfig(width=0.03, depth=0.03, height=0.03)
    build_grasp_model(output / "geometry", cfg)
    path = prepare_learning_model(output / "geometry/grasp.xml", output / "model")
    tree = ET.parse(path)
    hand = tree.getroot().find(".//body[@name='gripper_frame_link']")
    # Camera is rigid on rotating gripper. Its local -Z looks past the fingertips.
    position = np.array([0.0, -0.045, -0.075])
    direction = np.array([0.0, 0.0, 0.08]) - position
    z = -direction / np.linalg.norm(direction)
    x = np.cross([0, 1, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    ET.SubElement(
        hand,
        "camera",
        name="pickup_wrist",
        pos=" ".join(map(str, position)),
        xyaxes=" ".join(map(str, np.r_[x, y])),
        fovy="105",
    )
    tree.write(path)
    (output / "task.json").write_text(json.dumps(asdict(cfg), indent=2))
    return path, cfg


class PickupWorld(ManipulationWorld):
    def __init__(self, path, config, *, radius=0.035, noise=1.0):
        self.radius, self.noise = radius, noise
        self.defaults = None
        super().__init__(path, config)
        self.defaults = {
            name: getattr(self.model, name).copy()
            for name in (
                "body_pos",
                "geom_rgba",
                "cam_pos",
                "cam_quat",
                "cam_fovy",
                "light_diffuse",
                "light_ambient",
            )
        }

    def reset(self, seed, **kwargs):
        if self.defaults is not None:
            for name, array in self.defaults.items():
                getattr(self.model, name)[:] = array
        super().reset(seed, radius=self.radius)
        rng = np.random.default_rng(seed + 710000)
        strength = self.noise
        # Small residual geometric error after calibration; fixed-size motor meshes.
        scale = float(rng.uniform(1 - 0.015 * strength, 1 + 0.015 * strength))
        if self.defaults is not None:
            for name in (
                "upper_arm_link",
                "lower_arm_link",
                "wrist_link",
                "gripper_link",
                "gripper_frame_link",
            ):
                i = self.model.body(name).id
                self.model.body_pos[i] *= scale
            camera = self.model.camera("pickup_wrist").id
            self.model.cam_pos[camera] += rng.uniform(-0.002, 0.002, 3) * strength
            self.model.cam_fovy[camera] += rng.uniform(-3, 3) * strength
            for light in range(self.model.nlight):
                self.model.light_diffuse[light] *= rng.uniform(0.65, 1.3)
                self.model.light_ambient[light] = rng.uniform(0.05, 0.25, 3)
            palette = np.array(
                [[0.85, 0.12, 0.08, 1], [0.9, 0.85, 0.75, 1], [0.95, 0.4, 0.05, 1]]
            )
            self.model.geom_rgba[self.geom] = palette[rng.integers(3)]
            table = self.model.geom("table").id
            self.model.geom_rgba[table, :3] = rng.uniform(0.12, 0.4, 3)
            for geom in range(self.model.ngeom):
                if self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                    self.model.geom_rgba[geom, :3] = rng.uniform(0.4, 0.85, 3)
        object_pose = self.data.qpos[6:13].copy()
        mujoco.mj_setConst(self.model, self.data)
        self.data.qpos[6:13] = object_pose
        # Diverse actual arm starts, not a trajectory replay from one home pose.
        q = HOME[:5] + rng.uniform(-1, 1, 5) * np.array([0.20, 0.12, 0.12, 0.15, 0.25])
        self.data.qpos[:6] = np.r_[q, rng.uniform(0.7, 1.05)]
        yaw = rng.uniform(-0.18, 0.18)
        self.data.qpos[9:13] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.data.qvel[:] = 0
        self.data.ctrl[:] = self.data.qpos[:6]
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data, nstep=200)
        mujoco.mj_forward(self.model, self.data)
        self.initial_z = float(self.data.xpos[self.block, 2])
        self.rng = rng
        self.episode_spec = {
            "seed": int(seed),
            "cube_xy_m": self.data.xpos[self.block, :2].tolist(),
            "initial_joints_rad": self.data.qpos[:6].tolist(),
            "scale": scale,
            "mass_kg": self.config.mass,
            "friction": self.config.friction,
            "cube_yaw_rad": yaw,
        }
        return self.observe()

    def step(self, action, **kwargs):
        state, reward, done = super().step(action, **kwargs)
        # Pickup, not placement: require 5cm lift and continuous two-pad hold for 2s.
        success = self.held_seconds >= 2.0
        return state, float(success), bool(done or success)


def corrupt_rgb(rgb, rng, strength=1.0):
    if strength == 0:
        return rgb.copy()
    x = rgb.astype(np.float32) / 255
    x = np.maximum(
        x * rng.uniform(0.65, 1.3) + rng.uniform(-0.06, 0.06), 0
    ) ** rng.uniform(0.8, 1.25)
    x = np.nan_to_num(x)
    x += rng.normal(0, 0.018 * strength, x.shape)
    # Localized lighting gradient, sparse pixel artifacts, blur and small occluders.
    gradient = np.linspace(rng.uniform(0.65, 1.15), rng.uniform(0.65, 1.15), x.shape[1])
    x *= gradient[None, :, None]
    if rng.random() < 0.25 * strength:
        x = cv2.GaussianBlur(x, (3, 3), rng.uniform(0.3, 0.9))
    if rng.random() < 0.15 * strength:
        h, w = x.shape[:2]
        y, z = rng.integers(0, h - 5), rng.integers(0, w - 5)
        x[y : y + 5, z : z + 5] = rng.uniform(0, 1, 3)
    mask = rng.random(x.shape[:2]) < 0.002 * strength
    x[mask] = rng.uniform(0, 1, (mask.sum(), 3))
    return np.uint8(np.clip(x, 0, 1) * 255)


class WristSensor:
    def __init__(self, model, seed, *, noise=1.0):
        self.renderer = mujoco.Renderer(model, height=IMAGE_SIZE, width=IMAGE_SIZE)
        self.rng = np.random.default_rng(seed + 900000)
        self.noise = noise

    def capture(self, data, previous):
        # No object body/geom lookup, segmentation or depth rendering.
        self.renderer.update_scene(data, camera="pickup_wrist")
        image = corrupt_rgb(self.renderer.render(), self.rng, self.noise)
        q = data.qpos[:6] + self.rng.normal(0, 0.0015 * self.noise, 6)
        velocity = data.qvel[:6] * 0.1 + self.rng.normal(0, 0.001 * self.noise, 6)
        proprio = np.r_[q, velocity, previous].astype(np.float32)
        return image, proprio

    def close(self):
        self.renderer.close()


def teacher_episode(world, seed, *, sensor=None, policy=None, mix=0.0):
    world.reset(seed)
    records = []
    for _ in range(180):
        try:
            label = feedback_expert(world)
        except ValueError as e:
            return records, {
                "spec": world.episode_spec,
                "success": False,
                "steps": world.steps,
                "error": str(e),
            }
        if sensor is not None:
            image, p = sensor.capture(world.data, world.previous_action)
            auxiliary = (
                world.data.xpos[world.block] - world.data.site_xpos[world.site]
            ) * 10
            records.append((image, p, label, auxiliary.astype(np.float32)))
        action = (
            policy.act(image, p)
            if policy is not None and world.rng.random() < mix
            else label
        )
        _, _, done = world.step(action)
        if done:
            break
    return records, {
        "spec": world.episode_spec,
        "success": world.held_seconds >= 2,
        "steps": world.steps,
        "peak_lift_m": world.peak_lift,
        "hold_s": world.held_seconds,
    }
