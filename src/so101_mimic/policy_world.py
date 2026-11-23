"""A contact manipulation task for demonstration-augmented policy learning.

Position servos run inside MuJoCo's C stepper. Neural policies choose bounded
absolute joint targets and jaw opening at 10 Hz. No IK, trajectory or phase label is
given to the deployed policy. Privileged state is used only by teacher training.
"""

import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, replace

import mujoco
import numpy as np

from .grasping import grasp_ik
from .physics import HOME

CONTROL_DT = 0.1
ACTION_SCALE = 0.08
JOINT_SCALE = np.array([1.0, 1.5, 1.5, 1.0, 1.0])
HORIZON = 220
STATE_DIM = 36


def prepare_learning_model(source_xml, output):
    output.mkdir(parents=True, exist_ok=False)
    tree = ET.parse(source_xml)
    root = tree.getroot()
    actuator = root.find("actuator")
    actuator.clear()
    names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    for i, name in enumerate(names):
        joint = next(j for j in root.iter("joint") if j.get("name") == name)
        joint.set("damping", ".8" if i < 5 else ".02")
        ET.SubElement(
            actuator,
            "position",
            joint=name,
            kp="60" if i < 5 else "2",
            kv="3" if i < 5 else ".04",
            ctrlrange=joint.get("range"),
            ctrllimited="true",
            forcerange="-10 10" if i < 5 else "-.18 .18",
            forcelimited="true",
        )
    # A fixed robot reference must not encode a sampled object width.
    site = next(s for s in root.iter("site") if s.get("name") == "grasp_centre")
    site.set("pos", ".0055 -.000218 -.087")
    world = root.find("worldbody")
    for name, pos, xyaxes in [
        ("learning_front", ".43 -.32 .40", ".86 .51 0 -.30 .51 .80"),
        ("learning_side", ".25 .32 .42", "-1 0 0 0 -.75 .66"),
    ]:
        ET.SubElement(world, "camera", name=name, pos=pos, xyaxes=xyaxes, fovy="55")
    tree.write(output / "learning.xml")
    return output / "learning.xml"


class ManipulationWorld:
    def __init__(self, model_path, config, *, goal_xy=None):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.base = config
        self.config = config
        self.goal_xy = np.array(
            goal_xy
            if goal_xy is not None
            else [config.object_x, config.object_y + 0.035]
        )
        self.block = self.model.body("block").id
        self.geom = self.model.geom("block_geom").id
        self.site = self.model.site("grasp_centre").id
        self.pads = [self.model.geom(n).id for n in ("fixed_pad", "moving_pad")]
        self.initial_z = config.height / 2
        self.reset(0)

    def reset(self, seed, *, radius=0.006, randomize=True):
        rng = np.random.default_rng(seed)
        cfg = replace(
            self.base,
            object_x=self.base.object_x + float(rng.uniform(-radius, radius)),
            object_y=self.base.object_y + float(rng.uniform(-radius, radius)),
            mass=float(rng.uniform(0.025, 0.06)) if randomize else self.base.mass,
            friction=float(rng.uniform(0.45, 0.9)) if randomize else self.base.friction,
        )
        self.config = cfg
        self.model.body_mass[self.block] = cfg.mass
        self.model.body_inertia[self.block] = (
            cfg.mass
            * np.array(
                [
                    cfg.depth**2 + cfg.height**2,
                    cfg.width**2 + cfg.height**2,
                    cfg.width**2 + cfg.depth**2,
                ]
            )
            / 12
        )
        for geom in [self.geom, *self.pads]:
            self.model.geom_friction[geom, 0] = cfg.friction
        gain = float(rng.uniform(0.9, 1.1)) if randomize else 1.0
        self.model.actuator_gainprm[:5, 0] = 60 * gain
        self.model.actuator_biasprm[:5, 1] = -60 * gain
        self.model.actuator_biasprm[:5, 2] = -3.0
        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = [*HOME[:5], 0.65]
        self.data.qpos[6:9] = [cfg.object_x, cfg.object_y, cfg.height / 2 + 0.001]
        self.data.qpos[9:13] = [1.0, 0.0, 0.0, 0.0]
        self.data.ctrl[:] = self.data.qpos[:6]
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data, nstep=200)
        mujoco.mj_forward(self.model, self.data)
        self.initial_z = float(self.data.xpos[self.block, 2])
        self.steps, self.held_seconds, self.success_dwell = 0, 0.0, 0.0
        self.peak_lift = 0.0
        self.previous_action = np.zeros(6)
        self.success = False
        self.trace = []
        return self.observe()

    def contact_count(self):
        touched = set()
        for c in self.data.contact:
            pair = {int(c.geom1), int(c.geom2)}
            for p in self.pads:
                if pair == {p, self.geom}:
                    touched.add(p)
        return len(touched)

    def observe(self):
        object_xyz = self.data.xpos[self.block]
        hand = self.data.site_xpos[self.site]
        goal = np.r_[self.goal_xy, self.config.height / 2]
        # Memory is derived from observed task progress, not an expert phase.
        return np.r_[
            self.data.qpos[:6] - np.r_[HOME[:5], 0.5],
            self.data.qvel[:6] * 0.1,
            (object_xyz - hand) * 10,
            self.data.xquat[self.block],
            self.data.qvel[6:9],
            (goal - object_xyz) * 10,
            hand * 3,
            min(self.peak_lift / 0.065, 1.0),
            min(self.held_seconds / 2.0, 1.0),
            self.previous_action,
        ].astype(np.float32)

    def action_target(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("Invalid learned action")
        target = self.data.qpos[:6].copy()
        target[:5] = HOME[:5] + JOINT_SCALE * np.clip(action[:5], -1, 1)
        target[5] = (np.clip(action[5], -1, 1) + 1) * 0.55
        return np.clip(target, self.model.jnt_range[:6, 0], self.model.jnt_range[:6, 1])

    def step(self, action, *, target_override=None):
        action = np.asarray(action, dtype=float)
        target = (
            self.action_target(action)
            if target_override is None
            else np.asarray(target_override)
        )
        self.data.ctrl[:] = target
        mujoco.mj_step(
            self.model, self.data, nstep=round(CONTROL_DT / self.model.opt.timestep)
        )
        mujoco.mj_forward(self.model, self.data)
        return self.after_physics(action, target)

    def after_physics(self, action, target):
        self.steps += 1
        obj = self.data.xpos[self.block].copy()
        lift = float(obj[2] - self.initial_z)
        self.peak_lift = max(self.peak_lift, lift)
        if self.held_seconds >= 2.0 and lift > 0.015 and self.contact_count() < 2:
            self.held_seconds = 0.0
        elif self.held_seconds < 2.0:
            self.held_seconds = (
                self.held_seconds + CONTROL_DT
                if lift >= 0.05 and self.contact_count() == 2
                else 0.0
            )
        goal_error = float(np.linalg.norm(obj[:2] - self.goal_xy))
        placed = (
            self.held_seconds >= 2.0
            and goal_error < 0.02
            and abs(obj[2] - self.config.height / 2) < 0.003
            and self.data.qpos[5] > 0.6
            and self.contact_count() == 0
            and np.linalg.norm(self.data.qvel[6:9]) < 0.04
        )
        self.success_dwell = self.success_dwell + CONTROL_DT if placed else 0.0
        self.success = self.success_dwell >= 0.3
        self.previous_action = np.clip(action, -1, 1)
        finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        if not finite:
            raise RuntimeError("Nonfinite policy environment")
        done = (
            self.success
            or self.steps >= HORIZON
            or obj[2] < -0.03
            or np.linalg.norm(obj[:2]) > 0.6
        )
        # Sparse success objective; demonstration BC supplies exploration bias.
        reward = 10.0 if self.success else 0.0
        self.trace.append(
            {
                "step": self.steps,
                "time": self.steps * CONTROL_DT,
                "joints": self.data.qpos[:6].tolist(),
                "object_xyz": obj.tolist(),
                "object_quaternion": self.data.xquat[self.block].tolist(),
                "action": action.tolist(),
                "target": target.tolist(),
                "held_seconds": self.held_seconds,
                "goal_xy": self.goal_xy.tolist(),
                "success": self.success,
            }
        )
        return self.observe(), reward, done

    def snapshot(self):
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "held_seconds": self.held_seconds,
            "peak_lift": self.peak_lift,
            "steps": self.steps,
            "previous_action": self.previous_action.copy(),
            "initial_z": self.initial_z,
        }

    def restore(self, state):
        self.data.qpos[:] = state["qpos"]
        self.data.qvel[:] = state["qvel"]
        self.data.ctrl[:] = state["ctrl"]
        self.held_seconds, self.peak_lift = state["held_seconds"], state["peak_lift"]
        self.steps, self.initial_z = state["steps"], state["initial_z"]
        self.previous_action = state["previous_action"].copy()
        self.success_dwell, self.success = 0.0, False
        mujoco.mj_forward(self.model, self.data)
        return self.observe()


def expert_schedule(world):
    cfg = world.config
    origin = np.array([cfg.object_x, cfg.object_y, cfg.height / 2 + 0.003])
    destination = np.r_[world.goal_xy, cfg.height / 2 + 0.003]
    q = HOME[:5].copy()
    above = grasp_ik(world.model, origin + [0, 0, 0.065], q)
    down = grasp_ik(world.model, origin, above)
    goal_above = grasp_ik(world.model, destination + [0, 0, 0.065], above)
    goal_down = grasp_ik(world.model, destination + [0, 0, 0.005], goal_above)
    stages = [
        ("approach", 1.5, above, 1.1),
        ("descend", 2.0, down, 1.1),
        ("close", 1.0, down, 0.0),
        ("lift", 1.5, above, 0.0),
        ("hold", 2.4, above, 0.0),
        ("transport", 1.5, goal_above, 0.0),
        ("lower", 1.5, goal_down, 0.0),
        ("release", 1.2, goal_down, 1.1),
        ("retreat", 1.2, goal_above, 1.1),
    ]
    previous = np.r_[q, 1.1]
    schedule, current = [], q.copy()
    for name, duration, arm, jaw in stages:
        end = np.r_[arm, jaw]
        n = round(duration / CONTROL_DT)
        for i in range(n):
            t = (i + 1) / n
            blend = 10 * t**3 - 15 * t**4 + 6 * t**5
            command = previous + (end - previous) * blend
            if name in ("descend", "lift", "transport", "lower"):
                start_xyz, end_xyz = (
                    (origin + [0, 0, 0.065], origin)
                    if name == "descend"
                    else (origin, origin + [0, 0, 0.065])
                    if name == "lift"
                    else (origin + [0, 0, 0.065], destination + [0, 0, 0.065])
                    if name == "transport"
                    else (destination + [0, 0, 0.065], destination + [0, 0, 0.005])
                )
                current = grasp_ik(
                    world.model, start_xyz + (end_xyz - start_xyz) * blend, current
                )
                command[:5] = current
            else:
                current = command[:5].copy()
            schedule.append((name, command))
        previous = end
    return schedule


def collect_experts(model_path, config, output, *, count=24, first_seed=71):
    output.mkdir(parents=True, exist_ok=False)
    world = ManipulationWorld(model_path, config)
    observations, actions, reset_states, episodes = [], [], [], []
    for i in range(count):
        observation = world.reset(first_seed + i, radius=0.004)
        local_obs, local_act, local_states = [], [], []
        try:
            schedule = expert_schedule(world)
        except ValueError as exc:
            episodes.append(
                {"seed": first_seed + i, "success": False, "error": str(exc)}
            )
            continue
        for phase, target in schedule:
            if phase == "hold" and world.held_seconds >= 2.0:
                continue
            if phase == "close" and world.contact_count() == 2:
                continue
            opened = phase in ("approach", "descend", "release", "retreat")
            action = np.r_[
                np.clip((target[:5] - HOME[:5]) / JOINT_SCALE, -0.98, 0.98),
                0.98 if opened else -0.98,
            ]
            local_obs.append(observation.copy())
            local_act.append(action)
            local_states.append(world.snapshot())
            observation, _, done = world.step(action)
            if done:
                break
        episodes.append(
            {"seed": first_seed + i, "success": world.success, "steps": world.steps}
        )
        if world.success:
            observations.extend(local_obs)
            actions.extend(local_act)
            reset_states.extend(local_states[::5])
            np.savez_compressed(
                output / f"episode-{i:03}.npz", observation=local_obs, action=local_act
            )
        print(f"Expert {i + 1}/{count}: {world.success}", flush=True)
    if not observations:
        raise RuntimeError("No successful physical demonstrations")
    np.savez_compressed(
        output / "demonstrations.npz",
        observation=np.array(observations),
        action=np.array(actions),
    )
    # Numeric arrays, not pickle: physically attained curriculum states.
    np.savez_compressed(
        output / "reset-bank.npz",
        **{k: np.array([s[k] for s in reset_states]) for k in reset_states[0]},
    )
    report = {
        "episodes": episodes,
        "successful": sum(r["success"] for r in episodes),
        "transitions": len(observations),
        "config": asdict(config),
        "action_schema": 2,
        "action": "five normalized absolute joint targets and normalized jaw opening",
        "curriculum": "states physically attained in successful demonstrations; no object attachments",
        "source": "autonomous simulation expert, not human or real-world demonstrations",
    }
    (output / "demonstrations.json").write_text(json.dumps(report, indent=2))
    return report


def feedback_expert(world):
    obj = world.data.xpos[world.block].copy()
    hand = world.data.site_xpos[world.site].copy()
    q = world.data.qpos[:5].copy()
    contact = world.contact_count() == 2
    goal_error = np.linalg.norm(obj[:2] - world.goal_xy)
    grasped = contact and world.data.qpos[5] < 0.4
    if world.held_seconds >= 2.0 and goal_error < 0.009:
        if obj[2] > world.config.height / 2 + 0.01 and grasped:
            desired = (
                hand
                + np.r_[
                    world.goal_xy - obj[:2], world.config.height / 2 + 0.005 - obj[2]
                ]
            )
            jaw = -0.98
        elif world.data.qpos[5] < 0.7:
            desired, jaw = hand, 0.98
        else:
            desired, jaw = hand + [0, 0, 0.04], 0.98
    elif grasped:
        desired = hand + np.array([0.0, 0.0, world.initial_z + 0.065 - obj[2]])
        if world.held_seconds >= 2.0:
            desired[:2] += world.goal_xy - obj[:2]
        jaw = -0.98
    else:
        aligned = np.linalg.norm(hand[:2] - obj[:2]) < 0.004
        desired = obj + np.array([0.0, 0.0, 0.003 if aligned else 0.065])
        jaw = -0.98 if aligned and abs(hand[2] - (obj[2] + 0.003)) < 0.008 else 0.98
    try:
        goal_q = grasp_ik(world.model, desired, q)
    except ValueError:
        desired[2] = min(desired[2], 0.083)
        goal_q = grasp_ik(world.model, desired, q)
    # Keep teaching targets near the current state for smooth, recoverable
    # feedback; the learned outputs remain absolute and cannot integrate drift.
    target_q = q + np.clip(0.45 * (goal_q - q), -0.10, 0.10)
    return np.r_[np.clip((target_q - HOME[:5]) / JOINT_SCALE, -0.98, 0.98), jaw]


def collect_feedback_experts(model_path, config, output, *, count=96, first_seed=100):
    output.mkdir(parents=True, exist_ok=False)
    world = ManipulationWorld(model_path, config)
    observations, actions, reset_states, episodes = [], [], [], []
    for i in range(count):
        obs = world.reset(first_seed + i, radius=0.008)
        local_obs, local_act, local_states = [], [], []
        done = False
        try:
            while not done:
                action = feedback_expert(world)
                local_obs.append(obs.copy())
                local_act.append(action)
                local_states.append(world.snapshot())
                obs, _, done = world.step(action)
        except ValueError:
            pass
        episodes.append(
            {"seed": first_seed + i, "success": world.success, "steps": world.steps}
        )
        if world.success:
            observations.extend(local_obs)
            actions.extend(local_act)
            reset_states.extend(local_states[::5])
        if (i + 1) % 8 == 0:
            print(
                f"Feedback experts {i + 1}/{count}: {sum(r['success'] for r in episodes)} successes",
                flush=True,
            )
    if not observations:
        raise RuntimeError("No successful feedback demonstrations")
    np.savez_compressed(
        output / "demonstrations.npz", observation=observations, action=actions
    )
    np.savez_compressed(
        output / "reset-bank.npz",
        **{k: np.array([s[k] for s in reset_states]) for k in reset_states[0]},
    )
    report = {
        "episodes": episodes,
        "successful": sum(r["success"] for r in episodes),
        "transitions": len(observations),
        "config": asdict(config),
        "action_schema": 2,
        "action": "normalized absolute joint targets + jaw opening",
        "source": "autonomous state-feedback expert in simulation; not physical demonstrations",
    }
    (output / "demonstrations.json").write_text(json.dumps(report, indent=2))
    return report
