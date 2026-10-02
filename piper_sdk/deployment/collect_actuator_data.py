"""Collect PiPER actuator data, or preview joint references without opening CAN.

Examples (run in the sim environment)::

    python -m piper_sdk.deployment.collect_actuator_data --mode move_j \
        --source trajectory --preview --output_dir logs/actuator_preview
    python -m piper_sdk.deployment.collect_actuator_data --mode mit \
        --source trajectory --max_steps 1000 --output_dir logs/actuator_mit_short
    python -m piper_sdk.deployment.collect_actuator_data --mode move_j \
        --source policy --checkpoint_path /path/to/model.pt \
        --target_pos_b 0.5 -0.2 0.4 --output_dir logs/actuator_move_j_policy

Without --preview this command operates the arm. Each output directory is a
new session. Edit a preview's trajectory_config.json and pass it through
--trajectory_config to use different centers or segment durations.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from piper_sdk.deployment.actuator_recording import FrameRecorder, plot_response, write_json
from piper_sdk.deployment.actuator_trajectory import (
    TrajectoryConfig, TrajectoryPoint, generate_trajectory, trajectory_summary, write_trajectory,
)
from piper_sdk.deployment.manipulation import (
    ACTION_CLIP, ACTION_SCALE, DEFAULT_TARGET_POS_B, JOINT_LIMITS_RAD,
    MIT_JOINT_KD, MIT_JOINT_KP, PIPER_MAX_TARGET_DELTA_RAD_PER_POLICY_TICK,
    POLICY_CONTROL_PERIOD, POLICY_INIT_JOINT_POS, SPEED_PERCENT,
)
from piper_sdk.deployment.record_manipulation_real import RecordingManipulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("move_j", "mit"), required=True)
    parser.add_argument("--source", choices=("trajectory", "policy"), required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--preview", action="store_true", help="Generate trajectory files without connecting to CAN.")
    parser.add_argument("--trajectory_config", type=Path)
    parser.add_argument("--max_steps", type=int, help="Collect only the first N trajectory steps for a short session.")
    parser.add_argument("--checkpoint_path", type=Path)
    parser.add_argument("--policy_steps", type=int, default=500)
    targets = parser.add_mutually_exclusive_group()
    targets.add_argument("--target_pos_b", type=float, nargs=3, default=DEFAULT_TARGET_POS_B)
    targets.add_argument("--random_target", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--can_name", default="can_piper")
    parser.add_argument("--can_host", default="192.168.123.162")
    parser.add_argument("--can_port", type=int, default=29536)
    args = parser.parse_args()
    if args.source == "policy":
        if args.checkpoint_path is None:
            parser.error("--source policy requires --checkpoint_path")
        if args.preview or args.trajectory_config is not None or args.max_steps is not None:
            parser.error("--preview, --trajectory_config and --max_steps are trajectory options")
        args.checkpoint_path = args.checkpoint_path.expanduser().resolve()
        if not args.checkpoint_path.is_file():
            parser.error(f"Checkpoint not found: {args.checkpoint_path}")
    if args.policy_steps <= 0 or (args.max_steps is not None and args.max_steps <= 0):
        parser.error("Step counts must be positive")
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def record_step(
    arm: RecordingManipulation, recorder: FrameRecorder, action: torch.Tensor,
    point: TrajectoryPoint, step: int, previous_start: int | None,
) -> int:
    recorder.check_error()
    recorder.set_context(point.phase, point.segment, point.pose_index, step)
    requested = arm.default_arm_joint_pos + ACTION_SCALE * action
    started = time.monotonic_ns()
    arm.step(action)
    finished = time.monotonic_ns()
    vectors = (
        action.detach().cpu().tolist(), requested.detach().cpu().tolist(),
        arm.arm_joint_pos_target.detach().cpu().tolist(), list(recorder.sent_targets),
        arm.arm_joint_pos.detach().cpu().tolist(), arm.arm_joint_vel.detach().cpu().tolist(),
        list(arm.motor_speed_latest_rad_s), list(arm.motor_currents_a),
        list(arm.motor_efforts_nm), list(arm.motor_feedback_timestamps),
    )
    row = [
        step, point.phase, point.segment, point.pose_index, point.joint_index,
        started, finished, (started - previous_start) * 1e-9 if previous_start is not None else "",
        (finished - started) * 1e-9, arm.joint_feedback_timestamp,
        arm.status_feedback_timestamp, arm.ctrl_mode, arm.move_mode, arm.arm_status_code,
    ]
    for vector in vectors:
        row.extend(vector)
    recorder.record_step(row)
    return started


def collect(args: argparse.Namespace, metadata: dict, points: list[TrajectoryPoint]) -> None:
    recorder = FrameRecorder(args.output_dir)
    arm = None
    returned = False
    outcome, failure = "complete", None
    try:
        arm = RecordingManipulation(
            checkpoint_path=args.checkpoint_path if args.source == "policy" else None,
            device=args.device, can_name=args.can_name, can_host=args.can_host, can_port=args.can_port,
            target_pos_b=None if args.random_target else args.target_pos_b,
            policy_control_mode=args.mode, can_frame_observer=recorder.on_frame,
        )
        metadata["firmware"] = arm.piper.GetPiperFirmwareVersion()
        metadata["target_pose_b"] = arm.pose_command_b.detach().cpu().tolist()
        write_json(args.output_dir / "metadata.json", metadata)

        # Prime the selected mode while holding its takeover position. These
        # samples are separate from training segments, including for MOVE_J.
        takeover = arm.arm_joint_pos.clone()
        takeover_action = (takeover - arm.default_arm_joint_pos) / ACTION_SCALE
        switch = TrajectoryPoint("mode_switch", "mode_switch", -1, 0, tuple(takeover.tolist()))
        previous = None
        for step in range(-100, 0):
            previous = record_step(arm, recorder, takeover_action, switch, step, previous)

        total = len(points) if args.source == "trajectory" else args.policy_steps
        last_segment = None
        for step in range(total):
            if args.source == "trajectory":
                point = points[step]
                target = torch.tensor(point.target_rad, dtype=torch.float32, device=arm.device)
                action = (target - arm.default_arm_joint_pos) / ACTION_SCALE
            else:
                action = arm.arm_policy.get_action(arm.arm_history_obs_buf)
                point = TrajectoryPoint("policy", "policy", -1, 0, ())
            if point.segment != last_segment:
                print(f"[{args.mode}] {step}/{total} {point.segment}", flush=True)
                last_segment = point.segment
            previous = record_step(arm, recorder, action, point, step, previous)

        recorder.set_context("return", "return_to_zero")
        arm.move_j_to_zero()
        returned = True
        recorder.check_error()
    except BaseException as exc:
        outcome = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if arm is not None:
            if not returned and arm.control_started:
                recorder.set_context("stop", "stop")
                arm.quick_stop()
            # Wait for the receiver before closing its recording queue.
            arm.piper.DisconnectPort(thread_timeout=1.5)
            arm.disconnect()
        recorder.close()
        if recorder.error is not None:
            outcome = "failed"
            failure = str(recorder.error)
        metadata.update(outcome=outcome, failure=failure, finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output_dir / "metadata.json", metadata)
        summary = recorder.summarize(outcome, failure)
        write_json(args.output_dir / "summary.json", summary)
        plot_response(args.output_dir)
        print(f"采集结果 {outcome}: {args.output_dir}", flush=True)
    recorder.check_error()


def main() -> None:
    args = parse_args()
    points: list[TrajectoryPoint] = []
    cfg = TrajectoryConfig()
    if args.source == "trajectory":
        if args.trajectory_config is not None:
            with args.trajectory_config.open() as stream:
                cfg = TrajectoryConfig(**json.load(stream))
        points = generate_trajectory(cfg, POLICY_CONTROL_PERIOD, POLICY_INIT_JOINT_POS)
        # A trajectory must survive the same action and joint limits as deployment.
        for point in points:
            for joint, (q, (lower, upper)) in enumerate(zip(point.target_rad, JOINT_LIMITS_RAD)):
                if not math.isfinite(q) or not lower <= q <= upper or abs((q - POLICY_INIT_JOINT_POS[joint]) / ACTION_SCALE) > ACTION_CLIP:
                    raise ValueError(f"Trajectory target outside deployment limits: {point.segment}, J{joint + 1}={q}")
        if args.max_steps is not None:
            points = points[:args.max_steps]

    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "mode": args.mode, "source": args.source, "preview": args.preview,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "local_clock_anchor": {"monotonic_ns": time.monotonic_ns(), "wall_time_ns": time.time_ns()},
        "setup": {"base": "stationary", "end_load": "gripper_only"},
        "connection": {"interface": "socketcand", "channel": args.can_name, "host": args.can_host, "port": args.can_port},
        "control": {
            "period_s": POLICY_CONTROL_PERIOD,
            "max_target_delta_rad_per_tick": PIPER_MAX_TARGET_DELTA_RAD_PER_POLICY_TICK,
            "ctrl_mode": 0x01, "move_mode": 0x01 if args.mode == "move_j" else 0x04,
            "is_mit_mode": 0xAD, "speed_percent": SPEED_PERCENT if args.mode == "move_j" else 0,
            "kp": list(MIT_JOINT_KP) if args.mode == "mit" else None,
            "kd": list(MIT_JOINT_KD) if args.mode == "mit" else None,
            "vel_ref": 0.0 if args.mode == "mit" else None,
            "t_ref": 0.0 if args.mode == "mit" else None,
        },
        "checkpoint_path": str(args.checkpoint_path) if args.source == "policy" else None,
        "policy_steps": args.policy_steps if args.source == "policy" else None,
        "trajectory_config": asdict(cfg) if args.source == "trajectory" else None,
        "max_steps": args.max_steps,
        "collection_phases": ["hold", "sweep", "multijoint", "policy"],
        "data_notes": {
            "time": "Local monotonic timestamps share one host clock. CAN source timestamps may use a remote clock; do not subtract the clocks without alignment.",
            "tx": "TX times bracket the host send call, not motor reception. sent_target_rad is decoded from successful outgoing CAN bytes.",
            "feedback": "steps.csv contains asynchronous SDK snapshots. can_frames.csv preserves individual original feedback frames and their timestamps.",
            "effort": "SDK current-derived estimate, not calibrated joint output torque; raw signed current is preserved in CAN bytes.",
            "phases": "Initialization, mode_switch, transition, return and stop are not collection segments. Return and initialization use MOVE_J in both sessions.",
        },
    }
    write_json(args.output_dir / "metadata.json", metadata)
    if args.source == "trajectory":
        write_json(args.output_dir / "trajectory_config.json", asdict(cfg))
        write_trajectory(args.output_dir / "trajectory.csv", points, POLICY_CONTROL_PERIOD)
        planned = trajectory_summary(points, POLICY_CONTROL_PERIOD)
        metadata["planned_trajectory"] = planned
        write_json(args.output_dir / "metadata.json", metadata)
        print(f"轨迹: {planned['steps']} 步, {planned['duration_s'] / 60:.2f} 分钟", flush=True)
    if args.preview:
        write_json(args.output_dir / "summary.json", {"outcome": "preview", **planned})
        print(f"离线预览已保存: {args.output_dir}")
        return

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    collect(args, metadata, points)


if __name__ == "__main__":
    main()
