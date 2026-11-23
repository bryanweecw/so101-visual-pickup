"""Analytical approach/close/lift/hold/place baseline with physical outcomes."""

import hashlib
import json
from dataclasses import asdict

import mujoco
import numpy as np

from .grasp_physics import GraspConfig, GraspPhysics, build_grasp_model
from .physics import DT, HOME


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grasp_ik(model, xyz, initial):
    """Solve reachable top-down grasp-centre poses, with wrist roll near zero."""
    data = mujoco.MjData(model)
    q = np.asarray(initial, dtype=float).copy()
    site = model.site("grasp_centre").id
    body = model.body("gripper_link").id
    desired = np.asarray(xyz, dtype=float)
    if desired.shape != (3,) or not np.isfinite(desired).all():
        raise ValueError("Invalid grasp position")
    for _ in range(250):
        data.qpos[:6] = [*q, 0.0]
        mujoco.mj_forward(model, data)
        z = data.xmat[body].reshape(3, 3)[:, 2]
        error = np.r_[
            desired - data.site_xpos[site], 0.1 * np.cross(z, [0, 0, 1]), -0.03 * q[4]
        ]
        if np.linalg.norm(error) < 0.0001:
            return q
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, site)
        angular = (np.eye(3) - np.outer(z, z)) @ jr[:, :5]
        jac = np.vstack((jp[:, :5], 0.1 * angular, [0, 0, 0, 0, 0.03]))
        delta = jac.T @ np.linalg.solve(jac @ jac.T + 0.002**2 * np.eye(7), error)
        q = np.clip(
            q + np.clip(delta, -0.15, 0.15),
            model.jnt_range[:5, 0] + 0.001,
            model.jnt_range[:5, 1] - 0.001,
        )
    if (
        np.linalg.norm(desired - data.site_xpos[site]) > 0.001
        or np.linalg.norm(np.cross(z, [0, 0, 1])) > 0.015
    ):
        raise ValueError(f"Grasp pose unreachable: {desired.tolist()}")
    return q


def grasp_schedule(model, config, *, offset=(0.0, 0.0), keep_open=False):
    target = np.array(
        [
            config.object_x + offset[0],
            config.object_y + offset[1],
            config.height / 2 + 0.003,
        ]
    )
    q = HOME[:5].copy()
    approach = grasp_ik(model, target + [0, 0, 0.065], q)
    down = grasp_ik(model, target, approach)
    lift = grasp_ik(model, target + [0, 0, 0.065], down)
    opened, closed = 0.65, 0.65 if keep_open else 0.0
    waypoints = [
        ("settle", 0.5, np.r_[q, opened]),
        ("approach", 1.5, np.r_[approach, opened]),
        ("descend", 2.0, np.r_[down, opened]),
        ("close", 1.0, np.r_[down, closed]),
        ("lift", 1.5, np.r_[lift, closed]),
        ("hold", 2.3, np.r_[lift, closed]),
        ("lower", 1.5, np.r_[down, closed]),
        ("release", 0.8, np.r_[down, opened]),
        ("retreat", 1.2, np.r_[approach, opened]),
        ("placed", 0.5, np.r_[approach, opened]),
    ]
    previous = np.r_[q, opened]
    schedule = []
    current_q = q.copy()
    for phase, duration, end in waypoints:
        steps = round(duration / DT)
        for i in range(steps):
            t = (i + 1) / steps
            blend = 10 * t**3 - 15 * t**4 + 6 * t**5
            command = previous + (end - previous) * blend
            if phase in ("descend", "lift", "lower"):
                start_z, end_z = (
                    (0.065, 0.0) if phase in ("descend", "lower") else (0.0, 0.065)
                )
                xyz = target + np.array([0.0, 0.0, start_z + (end_z - start_z) * blend])
                current_q = grasp_ik(model, xyz, current_q)
                command[:5] = current_q
            else:
                current_q = command[:5].copy()
            schedule.append((phase, command))
        previous = end
    return schedule


def assess_grasp(trace, config):
    if not trace:
        raise ValueError("Empty grasp trace")
    initial = np.median([r["object_xyz"][2] for r in trace if r["phase"] == "settle"])
    hold = [r for r in trace if r["phase"] == "hold"]
    # Use the final two seconds of the designated hold, not its best frame.
    final_hold = (
        [r for r in hold if r["time"] >= hold[-1]["time"] - 2.0 + DT / 2]
        if hold
        else []
    )
    lift = [r["object_xyz"][2] - initial for r in final_hold]
    supported = [min(r["normal_forces"]) > 0.01 for r in final_hold]
    held = len(final_hold) >= 99 and min(lift) >= 0.05 and all(supported)
    final = trace[-1]
    lowered_in_air = [
        r
        for r in trace
        if r["phase"] == "lower" and r["object_xyz"][2] > initial + 0.01
    ]
    controlled_lower = bool(lowered_in_air) and all(
        min(r["normal_forces"]) > 0.01 for r in lowered_in_air
    )
    placement_error = float(
        np.linalg.norm(
            np.array(final["object_xyz"][:2]) - [config.object_x, config.object_y]
        )
    )
    placed = (
        abs(final["object_xyz"][2] - config.height / 2) < 0.003
        and np.linalg.norm(final["object_velocity"]) < 0.03
        and placement_error < 0.02
        and max(final["normal_forces"]) < 0.01
    )
    return {
        "success": bool(held and placed and controlled_lower),
        "controlled_lower": controlled_lower,
        "lift_and_hold": bool(held),
        "placed_and_released": bool(placed),
        "minimum_hold_lift_m": float(min(lift)) if lift else 0.0,
        "maximum_object_lift_m": float(
            max(r["object_xyz"][2] - initial for r in trace)
        ),
        "two_pad_contact_fraction_during_hold": float(np.mean(supported))
        if supported
        else 0.0,
        "hold_frames": len(final_hold),
        "hold_seconds": len(final_hold) * DT,
        "placement_error_m": placement_error,
        "final_object_xyz": final["object_xyz"],
        "finite": bool(all(np.isfinite(r["object_xyz"]).all() for r in trace)),
        "peak_gripper_torque_nm": float(max(abs(r["gripper_torque"]) for r in trace)),
    }


def grasp_trial(output, config=GraspConfig(), *, offset=(0.0, 0.0), keep_open=False):
    output.mkdir(parents=True, exist_ok=False)
    model = build_grasp_model(output / "model", config)
    plant = GraspPhysics(model, config)
    schedule = grasp_schedule(model, config, offset=offset, keep_open=keep_open)
    for phase, command in schedule:
        plant.phase = phase
        plant.step(command)
    result = assess_grasp(plant.trace, config)
    (output / "trace.json").write_text(json.dumps(plant.trace))
    report = {
        **result,
        "config": asdict(config),
        "offset": list(offset),
        "keep_open": keep_open,
        "simulation_only": True,
        "backend": "direct MuJoCo contact dynamics",
        "model_sha256": file_hash(output / "model/grasp.xml"),
        "trace_sha256": file_hash(output / "trace.json"),
        "object_attachment_constraints": model.neq,
        "success_definition": "object stays at least 5 cm above its initial centre for final 2 s of hold, with two-pad contact, then is placed and released",
        "assumptions": "approximate pad geometry, gravity-compensated arm, assumed friction/mass/servo parameters; no hardware qualification",
    }
    (output / "grasp.json").write_text(json.dumps(report, indent=2))
    return report
