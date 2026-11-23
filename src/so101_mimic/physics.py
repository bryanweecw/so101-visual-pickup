"""Public RUA robot geometry, original MuJoCo position-servo approximation."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
HOME = np.array([0, -0.8, 0.3, 0.8, 0, 0.0])
DT = 0.02


def build_model(work: Path):
    work.mkdir(parents=True, exist_ok=True)
    assets = Path(sys.prefix) / "share/hebbian-robot-use/robots/so101"
    if not (assets / "robot.urdf").is_file():
        raise RuntimeError("Install the public RUA wheel with its SO-101 assets")
    tree = ET.parse(assets / "robot.urdf")
    ext = ET.SubElement(tree.getroot(), "mujoco")
    ET.SubElement(
        ext, "compiler", meshdir=str(assets), discardvisual="false", fusestatic="false"
    )
    tree.write(work / "source.urdf")
    imported = mujoco.MjModel.from_xml_path(str(work / "source.urdf"))
    mujoco.mj_saveLastXML(str(work / "import.xml"), imported)
    tree = ET.parse(work / "import.xml")
    root = tree.getroot()
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
    # Approximate gravity-compensated servos. These are assumptions, not a
    # calibrated STS3215 model; there is no contact-manipulation claim.
    for body in root.iter("body"):
        body.set("gravcomp", "1")
    for joint in root.iter("joint"):
        joint.set("damping", "0.8")
        joint.set("armature", "0.015")
    for geom in root.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
        geom.set("rgba", "0.72 0.76 0.82 1")
    world = root.find("worldbody")
    ET.SubElement(world, "geom", type="plane", size="2 2 .01", rgba=".16 .21 .25 1")
    ET.SubElement(world, "light", pos="0 -1 2", diffuse=".9 .9 .9")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth="960", offheight="640")
    actuator = ET.SubElement(root, "actuator")
    for name in JOINTS:
        ET.SubElement(
            actuator, "motor", joint=name, ctrlrange="-15 15", ctrllimited="true"
        )
    tree.write(work / "model.xml")
    model = mujoco.MjModel.from_xml_path(str(work / "model.xml"))
    if [model.joint(i).name for i in range(model.njnt)] != JOINTS:
        raise ValueError("Unexpected public model joint order")
    return model


class Physics:
    def __init__(self, model, *, strength=1.0, damping=1.0):
        self.model = model
        self.data = mujoco.MjData(model)
        self.strength, self.damping = strength, damping
        self.reset(HOME, np.zeros(6))

    def reset(self, q, v):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = q
        self.data.qvel[:] = v
        mujoco.mj_forward(self.model, self.data)

    def state(self):
        return np.concatenate((self.data.qpos, self.data.qvel * 0.1)).copy()

    def step(self, target, seconds=DT, powered=True):
        target = np.asarray(target, dtype=float)
        if (
            target.shape != (6,)
            or not np.isfinite(target).all()
            or not np.isfinite(seconds)
            or not 0 < seconds <= 0.5
        ):
            raise ValueError("Invalid simulator request")
        if np.any(target < self.model.jnt_range[:, 0]) or np.any(
            target > self.model.jnt_range[:, 1]
        ):
            raise ValueError("Target outside model limits")
        for _ in range(max(1, round(seconds / self.model.opt.timestep))):
            self.data.ctrl[:] = (
                np.clip(
                    self.strength * 40 * (target - self.data.qpos)
                    - 3 * self.damping * self.data.qvel,
                    -15,
                    15,
                )
                if powered
                else 0
            )
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.state()).all():
            raise RuntimeError("Nonfinite physics")
        return self.state()

    def wire_state(self):
        return {
            "positions": self.data.qpos[:5].tolist(),
            "velocities": self.data.qvel[:5].tolist(),
            "efforts": [0.0] * 5,
            "gripper_angle": float(self.data.qpos[5]),
        }
