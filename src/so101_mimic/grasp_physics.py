"""Contact-based SO-101 grasp plant; no attachments, object forces or pose writes.

Public robot geometry and the real swinging-jaw joint are retained. Finger-pad
collision boxes are explicit approximations, not measured hardware geometry.
Only pad/object/table collision pairs are enabled; full-arm collision physics
is outside this first experiment.
"""

import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from .physics import DT, HOME, build_model


@dataclass(frozen=True)
class GraspConfig:
    mass: float = 0.04
    friction: float = 0.7
    closing_torque: float = 0.18
    width: float = 0.026
    depth: float = 0.026
    height: float = 0.030
    object_x: float = 0.235
    object_y: float = 0.0
    pad_contacts: bool = True

    def validate(self):
        values = np.array(
            [
                self.mass,
                self.friction,
                self.closing_torque,
                self.width,
                self.depth,
                self.height,
                self.object_x,
                self.object_y,
            ]
        )
        if (
            not np.isfinite(values).all()
            or not 0.001 <= self.mass <= 0.5
            or not 0 <= self.friction <= 2
            or not 0 < self.closing_torque <= 0.5
            or not 0.012 <= self.width <= 0.045
            or not 0.012 <= self.depth <= 0.05
            or not 0.02 <= self.height <= 0.06
        ):
            raise ValueError("Invalid grasp physics configuration")


def build_grasp_model(work, config=GraspConfig()):
    config.validate()
    build_model(work)
    tree = ET.parse(work / "model.xml")
    root = tree.getroot()
    root.find("option").attrib.update(
        timestep="0.001", cone="elliptic", iterations="80", tolerance="1e-9"
    )
    world = root.find("worldbody")
    floor = next(g for g in world.findall("geom") if g.get("type") == "plane")
    floor.attrib.update(
        name="table", contype="1", conaffinity="3", friction=".6 .002 .0001"
    )
    # The imported meshes are visual-only: their convex hulls fill the jaw gap.
    # Contact boxes follow the actual fixed link and rotating jaw, not two
    # fictitious symmetric slider fingers.
    fixed = next(b for b in root.iter("body") if b.get("name") == "gripper_link")
    moving = next(
        b for b in root.iter("body") if b.get("name") == "moving_jaw_so101_v1_link"
    )
    source = mujoco.MjModel.from_xml_path(str(work / "model.xml"))
    data = mujoco.MjData(source)
    data.qpos[:] = HOME
    mujoco.mj_forward(source, data)
    f, m = source.body("gripper_link").id, source.body("moving_jaw_so101_v1_link").id
    fixed_pos = np.array([-0.0135, -0.000218, -0.087])
    moving_rest = np.array([0.0135, -0.000218, -0.087])
    relative_rotation = data.xmat[m].reshape(3, 3).T @ data.xmat[f].reshape(3, 3)
    moving_pos = data.xmat[m].reshape(3, 3).T @ (
        data.xpos[f] + data.xmat[f].reshape(3, 3) @ moving_rest - data.xpos[m]
    )
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, relative_rotation.reshape(-1))
    common = {
        "type": "box",
        "size": ".003 .008 .012",
        "mass": ".001",
        "contype": "2" if config.pad_contacts else "0",
        "conaffinity": "1" if config.pad_contacts else "0",
        "condim": "4",
        "friction": f"{config.friction} .002 .0001",
        "solref": ".006 1",
        "solimp": ".95 .99 .001",
        "rgba": ".12 .19 .23 1",
    }
    ET.SubElement(
        fixed, "geom", name="fixed_pad", pos=" ".join(map(str, fixed_pos)), **common
    )
    ET.SubElement(
        moving,
        "geom",
        name="moving_pad",
        pos=" ".join(map(str, moving_pos)),
        quat=" ".join(map(str, quat)),
        **common,
    )
    # The fixed pad's inner face is at x=-.0105 in the gripper-link frame.
    # Centre the requested object between that face and the moving pad.
    # Leave 3 mm clearance at the fixed finger during descent; closure pushes
    # the object across that gap instead of scraping its top with the fixed pad.
    grasp_centre = [-0.0105 + config.width / 2 + 0.003, -0.000218, -0.087]
    ET.SubElement(
        fixed,
        "site",
        name="grasp_centre",
        pos=" ".join(map(str, grasp_centre)),
        size=".002",
        rgba="0 .8 .5 .4",
    )
    obj = ET.SubElement(
        world,
        "body",
        name="block",
        pos=f"{config.object_x} {config.object_y} {config.height / 2 + 0.001}",
        gravcomp="0",
    )
    ET.SubElement(obj, "freejoint", name="block_free")
    ET.SubElement(
        obj,
        "geom",
        name="block_geom",
        type="box",
        size=f"{config.width / 2} {config.depth / 2} {config.height / 2}",
        mass=str(config.mass),
        contype="1",
        conaffinity="3",
        condim="4",
        friction=f"{config.friction} .002 .0001",
        solref=".006 1",
        rgba=".9 .25 .08 1",
    )
    for joint in root.iter("joint"):
        if joint.get("name") == "gripper":
            joint.set("damping", ".02")
            joint.set("armature", ".0003")
    tree.write(work / "grasp.xml")
    (work / "grasp-config.json").write_text(json.dumps(asdict(config), indent=2))
    model = mujoco.MjModel.from_xml_path(str(work / "grasp.xml"))
    if model.nq != 13 or model.nv != 12 or model.neq:
        raise ValueError("Unexpected grasp model layout or attachment constraint")
    return model


class GraspPhysics:
    def __init__(self, model, config=GraspConfig()):
        self.model, self.config = model, config
        self.data = mujoco.MjData(model)
        self.block = model.body("block").id
        self.object_geom = model.geom("block_geom").id
        self.pad_ids = [model.geom(n).id for n in ("fixed_pad", "moving_pad")]
        guards = [
            i
            for i in range(model.ngeom)
            if (model.geom(i).name or "").endswith("_tip_guard")
        ]
        floors = [
            i
            for i in range(model.ngeom)
            if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
        ]
        self.tip_floor_pairs = [(guard, floor) for guard in guards for floor in floors]
        self.phase = "initial"
        self.trace = []
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = [*HOME[:5], 0.65]
        mujoco.mj_forward(self.model, self.data)
        self.trace.clear()

    def state(self):
        return np.r_[self.data.qpos[:6], self.data.qvel[:6] * 0.1].copy()

    def wire_state(self):
        return {
            "positions": self.data.qpos[:5].tolist(),
            "velocities": self.data.qvel[:5].tolist(),
            "efforts": [0.0] * 5,
            "gripper_angle": float(self.data.qpos[5]),
        }

    def contact_forces(self):
        forces = np.zeros(2)
        for contact_id in range(self.data.ncon):
            c = self.data.contact[contact_id]
            for i, pad in enumerate(self.pad_ids):
                if {int(c.geom1), int(c.geom2)} == {pad, self.object_geom}:
                    force = np.empty(6)
                    mujoco.mj_contactForce(self.model, self.data, contact_id, force)
                    forces[i] += max(0.0, force[0])
        return forces

    def step(self, target, seconds=DT, powered=True):
        target = np.asarray(target, dtype=float)
        if (
            target.shape != (6,)
            or not np.isfinite(target).all()
            or not np.isfinite(seconds)
            or not 0 < seconds <= 0.5
            or np.any(target < self.model.jnt_range[:6, 0])
            or np.any(target > self.model.jnt_range[:6, 1])
        ):
            raise ValueError("Invalid grasp actuator request")
        for _ in range(max(1, round(seconds / self.model.opt.timestep))):
            if powered:
                self.data.ctrl[:5] = np.clip(
                    60 * (target[:5] - self.data.qpos[:5]) - 3 * self.data.qvel[:5],
                    -10,
                    10,
                )
                self.data.ctrl[5] = np.clip(
                    2 * (target[5] - self.data.qpos[5]) - 0.04 * self.data.qvel[5],
                    -self.config.closing_torque,
                    self.config.closing_torque,
                )
            else:
                self.data.ctrl[:] = 0
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if (
            not np.isfinite(self.data.qpos).all()
            or not np.isfinite(self.data.qvel).all()
        ):
            raise RuntimeError("Nonfinite grasp physics")
        self.trace.append(
            {
                "time": float(self.data.time),
                "phase": self.phase,
                "command": target.tolist(),
                "powered": bool(powered),
                "joints": self.data.qpos[:6].tolist(),
                "object_xyz": self.data.xpos[self.block].tolist(),
                "object_quaternion": self.data.xquat[self.block].tolist(),
                "object_velocity": self.data.qvel[6:9].tolist(),
                "normal_forces": self.contact_forces().tolist(),
                "fingertip_table_clearance_m": min(
                    (
                        float(
                            mujoco.mj_geomDistance(
                                self.model, self.data, a, b, 1.0, None
                            )
                        )
                        for a, b in self.tip_floor_pairs
                    ),
                    default=None,
                ),
                "gripper_torque": float(self.data.qfrc_actuator[5]),
            }
        )
        return self.state()
