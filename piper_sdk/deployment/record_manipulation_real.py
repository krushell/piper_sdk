"""Record a Piper rollout to CSV and its exact policy inputs to NPZ."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
import torch

from piper_sdk.deployment.manipulation import (
    DEFAULT_TARGET_POS_B,
    TARGET_REACHED_KEYPOINT_ERROR_M,
    Manipulation,
)


JOINT_COUNT = 6


class RecordingManipulation(Manipulation):
    """Manipulation controller that retains the feedback used by observations."""

    def __init__(
        self,
        checkpoint_path: str | Path | None,
        device: str,
        can_name: str,
        can_host: str,
        can_port: int,
        target_pos_b: Sequence[float] | None,
        policy_control_mode: Literal["move_j", "mit"],
        can_frame_observer: Callable | None = None,
    ) -> None:
        self.joint_feedback_timestamp = 0.0
        self.joint_feedback_hz = 0.0
        self.motor_feedback_hz = 0.0
        self.motor_feedback_timestamps = (0.0,) * JOINT_COUNT
        self.motor_speed_latest_rad_s = (0.0,) * JOINT_COUNT
        self.motor_speed_average_sample_counts = (0,) * JOINT_COUNT
        self.motor_positions_rad = (0.0,) * JOINT_COUNT
        self.motor_currents_a = (0.0,) * JOINT_COUNT
        self.motor_efforts_nm = (0.0,) * JOINT_COUNT
        self.status_feedback_timestamp = 0.0
        self.status_feedback_hz = 0.0
        self.arm_status_code = 0
        self.ctrl_mode = 0
        self.move_mode = 0
        self.motion_status = 0
        super().__init__(
            checkpoint_path=checkpoint_path,
            device=device,
            can_name=can_name,
            can_host=can_host,
            can_port=can_port,
            target_pos_b=target_pos_b,
            policy_control_mode=policy_control_mode,
            can_frame_observer=can_frame_observer,
        )

    def update_feedback_observation(self) -> None:
        joint_msg = self.piper.GetArmJointMsgs()
        joint_state = joint_msg.joint_state
        joint_positions_millideg = (
            joint_state.joint_1,
            joint_state.joint_2,
            joint_state.joint_3,
            joint_state.joint_4,
            joint_state.joint_5,
            joint_state.joint_6,
        )
        self.joint_feedback_timestamp = float(joint_msg.time_stamp)
        self.joint_feedback_hz = float(joint_msg.Hz)

        motor_high, averaged_motor_speed_rad_s = self._read_motor_speed_observation(
            self.joint_feedback_timestamp
        )
        motors = tuple(
            getattr(motor_high, f"motor_{index}")
            for index in range(1, JOINT_COUNT + 1)
        )
        self.motor_feedback_hz = float(motor_high.Hz)
        self.motor_feedback_timestamps = tuple(
            float(motor.time_stamp) for motor in motors
        )
        self.motor_positions_rad = tuple(motor.pos * 0.001 for motor in motors)
        self.motor_currents_a = tuple(motor.current * 0.001 for motor in motors)
        self.motor_efforts_nm = tuple(motor.effort * 0.001 for motor in motors)

        status_msg = self.piper.GetArmStatus()
        arm_status = status_msg.arm_status
        self.status_feedback_timestamp = float(status_msg.time_stamp)
        self.status_feedback_hz = float(status_msg.Hz)
        self.arm_status_code = int(arm_status.arm_status)
        self.ctrl_mode = int(arm_status.ctrl_mode)
        self.move_mode = int(arm_status.mode_feed)
        self.motion_status = int(arm_status.motion_status)

        self.arm_joint_pos.copy_(
            torch.tensor(
                joint_positions_millideg,
                dtype=torch.float32,
                device=self.device,
            )
            * (math.pi / 180000.0)
        )
        self.arm_joint_vel.copy_(
            torch.tensor(
                averaged_motor_speed_rad_s,
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.update_current_ee_pose()
        self.update_keypoint_error_command()
        self.compute_arm_observations()
        self.update_arm_history_obs()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a fixed or randomly sampled target policy rollout."
    )
    parser.add_argument(
        "--can_name",
        default="can_piper",
        help="PC2 上由 socketcand 暴露的 CAN 设备名称。",
    )
    parser.add_argument(
        "--can_host",
        default="192.168.123.162",
        help="PC2 socketcand 地址。",
    )
    parser.add_argument(
        "--can_port",
        type=int,
        default=29536,
        help="PC2 socketcand TCP 端口。",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        required=True,
        help="RSL-RL Manipulation checkpoint path.",
    )
    parser.add_argument("--device", default="cpu", help="Policy inference device.")
    parser.add_argument(
        "--policy_control_mode",
        choices=("move_j", "mit"),
        default="move_j",
        help="Policy sender: MOVE J + 0xAD or MOVE M + 0xAD (MIT).",
    )
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--target_pos_b",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Fixed ee_gripper target position in the command frame, in meters.",
    )
    target_group.add_argument(
        "--random_target",
        action="store_true",
        help="Randomly sample one target pose from the training ranges.",
    )
    parser.add_argument(
        "--policy_steps",
        "--steps",
        dest="policy_steps",
        type=int,
        default=500,
        help="Number of policy steps to execute and record.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New CSV output path; an existing file is never overwritten.",
    )
    parser.add_argument(
        "--run_policy",
        action="store_true",
        help="Confirm that policy commands may be sent to the real arm.",
    )
    args = parser.parse_args()

    if not args.run_policy:
        parser.error("This script controls the real arm; pass --run_policy to confirm.")
    if args.policy_steps < 1:
        parser.error("--policy_steps must be positive.")
    if args.random_target:
        args.target_pos_b = None
    elif args.target_pos_b is None:
        args.target_pos_b = DEFAULT_TARGET_POS_B
    args.checkpoint_path = args.checkpoint_path.expanduser().resolve()
    if not args.checkpoint_path.is_file():
        parser.error(f"--checkpoint_path does not exist: {args.checkpoint_path}")
    args.output = args.output.expanduser().resolve()
    return args


def _csv_header() -> list[str]:
    header = [
        "step",
        "time_s",
        "loop_dt_s",
        "inference_dt_s",
        "step_dt_s",
        "wall_time_s",
    ]
    header += [f"target_pos_{axis}" for axis in "xyz"]
    header += [f"target_quat_{axis}" for axis in ("w", "x", "y", "z")]
    for prefix in (
        "action_raw",
        "action_clipped",
        "action_applied",
        "joint_target",
        "joint_pos",
        "joint_vel",
        "motor_speed_latest",
        "motor_speed_avg_sample_count",
        "joint_vel_fd",
        "joint_acc_fd",
        "motor_pos",
        "motor_current_a",
        "motor_effort_nm",
        "motor_timestamp_s",
        "motor_age_s",
    ):
        header += [f"{prefix}_j{joint}" for joint in range(1, JOINT_COUNT + 1)]
    header += [f"ee_pos_{axis}" for axis in "xyz"]
    header += [f"ee_quat_{axis}" for axis in ("w", "x", "y", "z")]
    header += [
        "ee_position_error_m",
        "ee_orientation_error_rad",
        "keypoint_rms_m",
        "target_reached",
    ]
    header += [f"keypoint_error_{index}" for index in range(9)]
    header += [
        "joint_feedback_timestamp_s",
        "joint_feedback_age_s",
        "joint_feedback_hz",
        "motor_feedback_hz",
        "motor_speed_avg_start_timestamp_s",
        "motor_speed_avg_end_timestamp_s",
        "motor_speed_avg_window_s",
        "status_feedback_timestamp_s",
        "status_feedback_age_s",
        "status_feedback_hz",
        "arm_status_code",
        "ctrl_mode",
        "move_mode",
        "motion_status",
    ]
    for prefix in ("obs_joint_pos", "obs_joint_vel", "obs_last_action"):
        header += [f"{prefix}_j{joint}" for joint in range(1, JOINT_COUNT + 1)]
    header += [f"obs_keypoint_error_{index}" for index in range(9)]
    return header


def _values(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().to(device="cpu", dtype=torch.float64).flatten().tolist()


def _tracking_errors(arm: RecordingManipulation) -> tuple[float, float, float]:
    position_error_m = torch.linalg.vector_norm(
        arm.pose_command_b[:3] - arm.current_ee_pose_b[:3]
    )
    target_quat = torch.nn.functional.normalize(arm.pose_command_b[3:], dim=0)
    current_quat = torch.nn.functional.normalize(arm.current_ee_pose_b[3:], dim=0)
    quat_dot = torch.clamp(
        torch.abs(torch.dot(target_quat, current_quat)), max=1.0
    )
    orientation_error_rad = 2.0 * torch.acos(quat_dot)
    keypoint_rms_m = (
        torch.linalg.vector_norm(arm.keypoint_error_command_b)
        / math.sqrt(arm.target_keypoints_b.shape[0])
    )
    return (
        float(position_error_m),
        float(orientation_error_rad),
        float(keypoint_rms_m),
    )


def _run_and_record(
    arm: RecordingManipulation,
    writer: csv.writer,
    policy_steps: int,
    policy_observations: list[np.ndarray],
) -> int:
    header_length = len(_csv_header())
    previous_joint_pos = arm.arm_joint_pos.clone()
    previous_joint_vel_fd = torch.zeros_like(previous_joint_pos)
    start_time = time.monotonic()
    previous_sample_time = start_time
    rows_written = 0

    for step_index in range(policy_steps):
        policy_observation = arm.arm_history_obs_buf.detach().clone()
        inference_start = time.monotonic()
        action_raw = (
            arm.arm_policy.get_action(policy_observation).detach().clone()
        )
        command_start = time.monotonic()
        arm.step(action_raw)
        sample_time = time.monotonic()
        wall_time = time.time()

        loop_dt = sample_time - previous_sample_time
        if loop_dt <= 0.0:
            raise RuntimeError(f"Non-positive control period: {loop_dt}")
        joint_vel_fd = (arm.arm_joint_pos - previous_joint_pos) / loop_dt
        joint_acc_fd = (joint_vel_fd - previous_joint_vel_fd) / loop_dt
        motor_ages = [
            wall_time - timestamp for timestamp in arm.motor_feedback_timestamps
        ]
        position_error_m, orientation_error_rad, keypoint_rms_m = (
            _tracking_errors(arm)
        )
        if arm.policy_control_mode == "move_j":
            sent_joint_target = torch.tensor(
                arm.target_millideg, dtype=torch.float32, device=arm.device
            ) * (math.pi / 180000.0)
        else:
            sent_joint_target = arm.arm_joint_pos_target
        sent_action = (sent_joint_target - arm.default_arm_joint_pos) / (
            arm.arm_policy.cfg.ActionCfg.action_scale
        )

        row: list[float | int] = [
            step_index + 1,
            sample_time - start_time,
            loop_dt,
            command_start - inference_start,
            sample_time - command_start,
            wall_time,
        ]
        row += _values(arm.pose_command_b[:3])
        row += _values(arm.pose_command_b[3:])
        row += _values(action_raw.squeeze(0))
        row += _values(arm.arm_last_action)
        row += _values(sent_action)
        row += _values(sent_joint_target)
        row += _values(arm.arm_joint_pos)
        row += _values(arm.arm_joint_vel)
        row += list(arm.motor_speed_latest_rad_s)
        row += list(arm.motor_speed_average_sample_counts)
        row += _values(joint_vel_fd)
        row += _values(joint_acc_fd)
        row += list(arm.motor_positions_rad)
        row += list(arm.motor_currents_a)
        row += list(arm.motor_efforts_nm)
        row += list(arm.motor_feedback_timestamps)
        row += motor_ages
        row += _values(arm.current_ee_pose_b[:3])
        row += _values(arm.current_ee_pose_b[3:])
        row += [
            position_error_m,
            orientation_error_rad,
            keypoint_rms_m,
            int(keypoint_rms_m <= TARGET_REACHED_KEYPOINT_ERROR_M),
        ]
        row += _values(arm.keypoint_error_command_b)
        row += [
            arm.joint_feedback_timestamp,
            wall_time - arm.joint_feedback_timestamp,
            arm.joint_feedback_hz,
            arm.motor_feedback_hz,
            arm.motor_speed_average_start_timestamp_s,
            arm.motor_speed_average_end_timestamp_s,
            arm.motor_speed_average_window_s,
            arm.status_feedback_timestamp,
            wall_time - arm.status_feedback_timestamp,
            arm.status_feedback_hz,
            arm.arm_status_code,
            arm.ctrl_mode,
            arm.move_mode,
            arm.motion_status,
        ]
        row += _values(policy_observation[-arm.arm_current_obs_buf.numel():])
        if len(row) != header_length:
            raise RuntimeError(
                f"CSV row has {len(row)} values, expected {header_length}."
            )
        writer.writerow(row)
        policy_observations.append(policy_observation.cpu().numpy())
        rows_written += 1

        previous_joint_pos.copy_(arm.arm_joint_pos)
        previous_joint_vel_fd.copy_(joint_vel_fd)
        previous_sample_time = sample_time

    return rows_written


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    arm: RecordingManipulation | None = None
    rows_written = 0
    policy_completed_normally = False
    returned_to_default = False
    policy_observations: list[np.ndarray] = []
    try:
        with args.output.open("x", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(_csv_header())
            output_file.flush()

            arm = RecordingManipulation(
                checkpoint_path=args.checkpoint_path,
                device=args.device,
                can_name=args.can_name,
                can_host=args.can_host,
                can_port=args.can_port,
                target_pos_b=args.target_pos_b,
                policy_control_mode=args.policy_control_mode,
            )
            try:
                rows_written = _run_and_record(
                    arm, writer, args.policy_steps, policy_observations
                )
                policy_completed_normally = True
            except KeyboardInterrupt:
                print("用户中断实机 policy 记录")
            finally:
                try:
                    arm.print_target_status()
                    if policy_completed_normally:
                        arm.move_j_to_zero()
                        returned_to_default = True
                finally:
                    if not returned_to_default and arm.control_started:
                        arm.quick_stop()
                    arm.disconnect()
    except FileExistsError as exc:
        raise FileExistsError(
            f"Output CSV already exists and was not overwritten: {args.output}"
        ) from exc
    finally:
        if policy_observations:
            np.savez_compressed(
                args.output.with_suffix(".npz"),
                policy_observation=np.stack(policy_observations),
            )

    print(
        f"已记录 {rows_written} 个实机策略步: {args.output}; "
        f"完整策略观测: {args.output.with_suffix('.npz')}"
    )


if __name__ == "__main__":
    main()
