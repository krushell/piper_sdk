from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence
from piper_sdk import C_PiperInterface_V2, C_PiperForwardKinematics
import torch
from piper_sdk.deployment import math_utils
from piper_sdk.deployment.policy import ManipulationPolicy

# 关节限位
JOINT_LIMITS_RAD = (
    (math.radians(-150.0), math.radians(150.0)),
    (math.radians(0.0), math.radians(180.0)),
    (math.radians(-170.0), math.radians(0.0)),
    (math.radians(-100.0), math.radians(100.0)),
    (math.radians(-70.0), math.radians(70.0)),
    (math.radians(-120.0), math.radians(120.0)),
)
JOINT_LIMITS_DEG = (
    (-150.0, 150.0),
    (0.0, 180.0),
    (-170.0, 0.0),
    (-100.0, 100.0),
    (-70.0, 70.0),
    (-120.0, 120.0),
)
# 末端偏移量
EE_OFFSET = 0.1358
# 策略初始化关节位置
POLICY_INIT_JOINT_POS = (
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)
# q = [0, 0.8, -0.8, 0, 0, 0] 对应的 ee_gripper 命令坐标系位置。
DEFAULT_TARGET_POS_B = (0.30474148, 0.0, 0.29298553)
POSE_COMMAND_RANGES = (
    (0.4, 0.7), # radius
    (0.0, 1.2), # pitch
    (-1.0, 1.0), # yaw
)

EE_LOCAL_YAW_RANGE = (-0.25, 0.25)
ACTION_CLIP = 10.0
ACTION_SCALE = 0.25
SIMULATION_DT = 0.005
SIMULATION_DECIMATION = 4
POLICY_CONTROL_PERIOD = SIMULATION_DT * SIMULATION_DECIMATION
PIPER_MAX_TARGET_DELTA_RAD_PER_POLICY_TICK = 0.006

# MIT_JOINT_KP = (1.7, 1.5, 1.5, 1.5, 1.7, 1.4)
# MIT_JOINT_KD = (0.75, 0.8, 0.8, 0.4, 0.55, 0.45)
MIT_JOINT_KP = (4, 4, 4, 4, 4, 2)
MIT_JOINT_KD = (0.75, 0.8, 0.8, 0.4, 0.55, 0.45)

SPEED_PERCENT = 5
MOTION_TIMEOUT = 25.0
TARGET_REACHED_KEYPOINT_ERROR_M = 0.02
ARM_RECOVERY_TIMEOUT = 5.0
ARM_RECOVERY_STABLE_SAMPLES = 10
RESET_SETTLE_TIME = 1.0
ENABLE_TIMEOUT = 8.0
ENABLE_COMMAND_INTERVAL = 1.0
ENABLE_POST_COMMAND_SETTLE_TIME = 0.5
ENABLE_STABLE_SAMPLES = 10
CAN_CONTROL_READY_TIMEOUT = 5.0
CAN_CONTROL_STABLE_SAMPLES = 20
REQUIRED_ENABLE_COMMANDS = 2
JOINT_FEEDBACK_TIMEOUT = 0.5
MOVE_J_POSITION_TOLERANCE_DEG = 2.0
MOVE_J_POSITION_STABLE_SAMPLES = 10
# J2/J3 的机械零位反馈可能略微越过名义边界。该容差只用于检查当前反馈，
# 不会放宽发送给机械臂的目标关节限位。
INITIAL_JOINT_LIMIT_TOLERANCE_DEG = 5.0





class Manipulation:
    def __init__(
        self, 
        checkpoint_path: str | Path | None = None,
        device: str = "cpu",
        can_name: str = "can_piper",
        can_host: str = "192.168.123.162",
        can_port: int = 29536,
        target_pos_b: Sequence[float] | torch.Tensor | None = DEFAULT_TARGET_POS_B,
        policy_control_mode: Literal["move_j", "mit"] = "move_j",
        can_frame_observer: Callable | None = None,
    )->None:
        self.set_policy_control_mode(policy_control_mode)
        self.device = torch.device(device)
        self.arm_policy =  ManipulationPolicy(device=self.device)
        self.checkpoint_path: Path | None = None
        if checkpoint_path is not None:
            self.arm_policy.load_policy(checkpoint_path)

        self.piper = C_PiperInterface_V2(
            can_name,
            judge_flag=False,
            can_auto_init=False,
        )
        self.piper.CreateCanBus(
            can_name=can_name,
            bustype="socketcand",
            judge_flag=False,
            host=can_host,
            port=can_port,
            tcp_tune=True,
            frame_observer=can_frame_observer,
        )
        self.fk = C_PiperForwardKinematics(dh_is_offset=1)
        self.control_started = False
        self._enable_attempted = False
        self.initialized = False
        self.target_init_joint = [math.degrees(angle) for angle in POLICY_INIT_JOINT_POS]
        # 检查初始目标位置是否越界
        for index, (angle, limits) in enumerate(
                zip(self.target_init_joint, JOINT_LIMITS_DEG), start=1
            ):
                lower, upper = limits
                if not lower <= angle <= upper:
                    raise ValueError(
                        f"J{index} 目标角度越界：{angle:.3f}°，"
                        f"允许范围 [{lower:.1f}, {upper:.1f}]°"
                    )
        try:
            self.piper.ConnectPort()
            # 等待3s 关节和机械臂状态反馈
            feedback_deadline = time.monotonic() + 3.0
            while True:
                joint_msg = self.piper.GetArmJointMsgs()
                status_msg = self.piper.GetArmStatus()
                # 收到piper的反馈 可以退出等待
                low_spd_msg = self.piper.GetArmLowSpdInfoMsgs()
                if (
                    joint_msg.Hz > 0
                    and status_msg.Hz > 0
                    and low_spd_msg.Hz > 0
                ):
                    break
                if time.monotonic() > feedback_deadline:
                    raise RuntimeError("没有收到完整反馈，请检查 CAN、机械臂电源和从臂模式")
                time.sleep(0.02)

            firmware_version = self.piper.GetPiperFirmwareVersion()
            if isinstance(firmware_version, str):
                print(f"机械臂固件版本: {firmware_version}")

            self.reset()
            # self.check_zero_link6() # 检查实机的link6位置是否与urdf一致

        except KeyboardInterrupt:
            if self._enable_attempted:
                self.quick_stop()
            self.piper.DisconnectPort(thread_timeout=1.5)
            raise
        except Exception as e:
            if self._enable_attempted:
                self.quick_stop()
            self.piper.DisconnectPort(thread_timeout=1.5)
            raise RuntimeError(f"初始化机械臂失败：{e}") from e

        self.arm_joint_pos = torch.zeros(6, dtype=torch.float32,device=self.device)
        self.arm_joint_vel = torch.zeros(6, dtype=torch.float32,device=self.device)
        self.arm_last_action = torch.zeros(6, dtype=torch.float32,device=self.device)
        self.arm_command_action = torch.zeros_like(self.arm_last_action)
        self.default_arm_joint_pos = torch.tensor(
            POLICY_INIT_JOINT_POS, dtype=torch.float32, device=self.device
        )
        self.arm_joint_pos_target = self.default_arm_joint_pos.clone()
        self._last_sent_arm_joint_pos_target = self.default_arm_joint_pos.clone()
        self.arm_joint_limits_rad = torch.tensor(
            JOINT_LIMITS_RAD, dtype=torch.float32, device=self.device
        )

        self._next_observation_deadline: float | None = None
        self._previous_joint_feedback_timestamp_s: float | None = None
        self.motor_speed_average_start_timestamp_s = 0.0
        self.motor_speed_average_end_timestamp_s = 0.0
        self.motor_speed_average_window_s = POLICY_CONTROL_PERIOD
        self.motor_speed_average_sample_counts = (0,) * 6
        self.motor_speed_latest_rad_s = (0.0,) * 6
        # 目标位置
        self.pose_command_b = torch.zeros(7, dtype=torch.float32,device=self.device)
        self.current_ee_pose_b = torch.zeros_like(self.pose_command_b)
        self.current_ee_pose_b[3] = 1.0
        self.pose_command_b[3] = 1.0
        half_keypoint_side = 0.05 # 姿态控制权重
        self.keypoint_offsets_ee = torch.tensor(
            (
                (half_keypoint_side, 0.0, 0.0),
                (0.0, half_keypoint_side, 0.0),
                (0.0, 0.0, half_keypoint_side),
            ),
            device=self.device,
        )

        self.target_keypoints_b = torch.zeros(3, 3, dtype=torch.float32,device=self.device)
        self.current_ee_keypoints_b = torch.zeros_like(self.target_keypoints_b)
        self.keypoint_error_command_b = torch.zeros(9, dtype=torch.float32,device=self.device)

        self.arm_current_obs_buf = torch.zeros(
            self.arm_policy.cfg.ObsCfg.num_single_observations, dtype=torch.float32, device=self.device
        )
        self.arm_history_obs_buf = torch.zeros(
            self.arm_policy.cfg.ObsCfg.num_observations,
            dtype=torch.float32,
            device=self.device,
        )
        self._arm_history_initialized = False

        self.set_arm_command(target_pos_b)
        self.update_feedback_observation()
        self.arm_joint_pos_target.copy_(self.arm_joint_pos)
        self._last_sent_arm_joint_pos_target.copy_(self.arm_joint_pos)
        self.initialized = True


    def set_policy_control_mode(self, mode: Literal["move_j", "mit"]) -> None:
        """Select the next policy action's control mode; call between policy steps."""
        if mode not in ("move_j", "mit"):
            raise ValueError(
                f"Unknown policy control mode: {mode!r}; expected 'move_j' or 'mit'."
            )
        self.policy_control_mode = mode
        self._policy_control_mode_changed = True
        description = (
            "MOVE_J 0xAD / JointCtrl"
            if mode == "move_j"
            else "MOVE_M 0xAD / JointMitCtrl"
        )
        print(
            f"策略控制模式设为 {description}，下一次策略动作生效；"
            "初始化和结束回位仍使用 MOVE_J 0xAD"
        )

    def run_policy(self, num_steps: int | None = None) -> None:
        if not self.initialized:
            raise RuntimeError("机械臂尚未成功初始化，禁止运行 policy")
        if self.arm_policy is None:
            raise RuntimeError("Arm policy is not loaded")
        if num_steps is not None and num_steps < 1:
            raise ValueError("num_steps must be positive or None")

        step_count = 0
        while num_steps is None or step_count < num_steps:
            action = self.arm_policy.get_action(self.arm_history_obs_buf)
            self.step(action)
            step_count += 1


    # 当前ee gripper的[x,y,z,qw,qx,qy,qz]
    def update_current_ee_pose(self)->None:
        joint_rad = [float(value) for value in self.arm_joint_pos]
        # link6 相对基座坐标系的位姿 list[x,y,z,roll,pitch,yaw]，单位 mm, deg
        link6_sdk_pose = self.fk.CalFK(joint_rad)[-1]
        link6_x_mm, link6_y_mm, link6_z_mm, link6_roll_deg, link6_pitch_deg, link6_yaw_deg = link6_sdk_pose

        link6_rpy = torch.deg2rad(
            torch.tensor(
                (link6_roll_deg, link6_pitch_deg, link6_yaw_deg),
                dtype=torch.float32,
                device=self.device,
            )
        )
        link6_quat_b = math_utils.quat_from_euler_xyz(link6_rpy[0], link6_rpy[1], link6_rpy[2])
        link6_rotation_b = math_utils.matrix_from_rpy(link6_rpy)
        link6_pos_b = torch.tensor(
            (link6_x_mm * 0.001, link6_y_mm * 0.001, link6_z_mm * 0.001),
            dtype=torch.float32,
            device=self.device,
        )
        ee_offset_6 = torch.tensor(
            (0.0, 0.0, EE_OFFSET),
            dtype=torch.float32,
            device=self.device,
        )
        ee_position_b = link6_pos_b + link6_rotation_b @ ee_offset_6
        # isaaclab中 ee_position_b += root_height -fix_height =  0.5559 - 0.6789 = -0.123
        ee_position_b[2] -= 0.123
        self.current_ee_pose_b[:3] = ee_position_b
        self.current_ee_pose_b[3:] = link6_quat_b
        self.current_ee_keypoints_b.copy_(
            self.compute_keypoints_b(
                self.current_ee_pose_b[:3],
                self.current_ee_pose_b[3:],
            )
        )

    def update_keypoint_error_command(self) -> None:
        self.keypoint_error_command_b.copy_(
            (self.target_keypoints_b - self.current_ee_keypoints_b).reshape(-1)
        )

    def print_target_status(self) -> bool:
        """Refresh feedback and print the final target-reaching result once."""
        self.update_feedback_observation()

        position_error_m = torch.linalg.vector_norm(
            self.pose_command_b[:3] - self.current_ee_pose_b[:3]
        )
        target_quat = torch.nn.functional.normalize(
            self.pose_command_b[3:], dim=0
        )
        current_quat = torch.nn.functional.normalize(
            self.current_ee_pose_b[3:], dim=0
        )
        quat_dot = torch.clamp(
            torch.abs(torch.dot(target_quat, current_quat)), max=1.0
        )
        orientation_error_deg = torch.rad2deg(2.0 * torch.acos(quat_dot))
        keypoint_rms_m = (
            torch.linalg.vector_norm(self.keypoint_error_command_b)
            / math.sqrt(self.target_keypoints_b.shape[0])
        )
        position_error_mm = position_error_m.item() * 1000.0
        orientation_error_deg_value = orientation_error_deg.item()
        keypoint_rms_mm = keypoint_rms_m.item() * 1000.0
        reached = keypoint_rms_mm <= TARGET_REACHED_KEYPOINT_ERROR_M * 1000.0

        target_pose = self.pose_command_b.detach().cpu().tolist()
        current_pose = self.current_ee_pose_b.detach().cpu().tolist()
        target_pose_format = (
            "[" + ", ".join(f"{value:.6f}" for value in target_pose) + "]"
        )
        current_pose_format = (
            "[" + ", ".join(f"{value:.6f}" for value in current_pose) + "]"
        )
        pose_fields = "[x, y, z, qw, qx, qy, qz]（位置单位 m）"
        print(f"目标位姿_b {pose_fields}: {target_pose_format}")
        print(f"当前位姿_b {pose_fields}: {current_pose_format}")
        print(
            f"是否到达目标: {'是' if reached else '否'} "
            f"(关键点 RMS 阈值 {TARGET_REACHED_KEYPOINT_ERROR_M * 1000.0:.1f} mm)"
        )
        print(
            f"目标误差: 位置 {position_error_mm:.3f} mm, "
            f"姿态 {orientation_error_deg_value:.3f}°, "
            f"关键点 RMS {keypoint_rms_mm:.3f} mm"
        )
        return reached


    def compute_keypoints_b(
        self,
        pos_b: torch.Tensor,
        quat_b: torch.Tensor,
    ) -> torch.Tensor:
        """Compute three EE keypoints in command frame.

        Args:
            pos_b: EE position, shape (3,).
            quat_b: EE quaternion in (w, x, y, z), shape (4,).

        Returns:
            Keypoint positions, shape (3, 3).
        """

        offsets = self.keypoint_offsets_ee
        quat = quat_b.unsqueeze(0).expand(offsets.shape[0], -1)

        quat_xyz = quat[:, 1:]
        cross_1 = 2.0 * torch.cross(quat_xyz, offsets, dim=-1)
        rotated_offsets = (
            offsets
            + quat[:, :1] * cross_1
            + torch.cross(quat_xyz, cross_1, dim=-1)
        )
        return pos_b.unsqueeze(0) + rotated_offsets

    # 采样self.target_keypoints_b
    def set_arm_command(self, target_pos_b: torch.Tensor | None = None) -> None:
        radius_range, pitch_range, yaw_range = POSE_COMMAND_RANGES
        dtype = self.pose_command_b.dtype

        if target_pos_b is None:
            radius = torch.empty((), dtype=dtype, device=self.device).uniform_(
                *radius_range
            )
            pitch = torch.empty((), dtype=dtype, device=self.device).uniform_(
                *pitch_range
            )
            yaw = torch.empty((), dtype=dtype, device=self.device).uniform_(*yaw_range)
            cos_pitch = torch.cos(pitch)
            target_pos = torch.stack(
                (
                    radius * cos_pitch * torch.cos(yaw),
                    radius * cos_pitch * torch.sin(yaw),
                    radius * torch.sin(pitch),
                )
            )
            local_yaw = torch.empty((), dtype=dtype, device=self.device).uniform_(
                *EE_LOCAL_YAW_RANGE
            )
        else:
            target_pos = torch.as_tensor(
                target_pos_b, dtype=dtype, device=self.device
            ).clone()
            if target_pos.shape != (3,):
                raise ValueError(
                    f"Expected target_pos_b shape (3,), got {target_pos.shape}."
                )
            radius = torch.linalg.vector_norm(target_pos)
            horizontal_radius = torch.linalg.vector_norm(target_pos[:2])
            pitch = torch.atan2(target_pos[2], horizontal_radius)
            yaw = torch.atan2(target_pos[1], target_pos[0])
            spherical = (radius, pitch, yaw)
            ranges = (radius_range, pitch_range, yaw_range)
            names = ("radius", "pitch", "yaw")
            for name, value, value_range in zip(names, spherical, ranges):
                if not value_range[0] <= float(value) <= value_range[1]:
                    raise ValueError(
                        f"Target {name} {float(value):.4f} is outside training range "
                        f"[{value_range[0]}, {value_range[1]}]."
                    )
            local_yaw = torch.zeros((), dtype=dtype, device=self.device)

        zero = torch.zeros((), dtype=dtype, device=self.device)
        half_pi = torch.full((), math.pi / 2.0, dtype=dtype, device=self.device)
        reference_quat = math_utils.quat_from_euler_xyz(zero, half_pi, yaw)
        local_quat = math_utils.quat_from_euler_xyz(zero, zero, local_yaw)
        target_quat = math_utils.quat_unique(
            math_utils.quat_mul(reference_quat, local_quat)
        )

        self.pose_command_b[:3].copy_(target_pos)
        self.pose_command_b[3:].copy_(target_quat)
        self.target_keypoints_b.copy_(
            self.compute_keypoints_b(
                self.pose_command_b[:3], self.pose_command_b[3:]
            )
        )

    def apply_action(self, action: torch.Tensor) -> None:
        if not self.initialized:
            raise RuntimeError("机械臂尚未成功初始化，禁止发送 policy 动作")
        action = torch.as_tensor(
            action, dtype=self.arm_last_action.dtype, device=self.device
        )
        if action.shape == (1, 6):
            action = action.squeeze(0)
        if action.shape != (6,):
            raise ValueError(
                f"Expected action shape (6,) or (1, 6), got {action.shape}."
            )
        if not torch.isfinite(action).all():
            raise ValueError("Policy action must contain only finite values")

        joint_msg = self.piper.GetArmJointMsgs()
        status_msg = self.piper.GetArmStatus()
        if joint_msg.Hz <= 0 or status_msg.Hz <= 0:
            raise RuntimeError("CAN 反馈频率为零")
        current_joint_deg = self._validate_joint_feedback(
            joint_msg, context="policy 控制"
        )
        arm_status = status_msg.arm_status.arm_status
        arm_ctrl_mode = status_msg.arm_status.ctrl_mode
        if int(arm_status) != 0 or int(arm_ctrl_mode) != 0x01:
            raise RuntimeError(
                "policy 控制前机械臂状态异常："
                f"{self._arm_status_diagnostics(status_msg)}"
            )
        if self._policy_control_mode_changed:
            self._last_sent_arm_joint_pos_target.copy_(
                torch.deg2rad(
                    torch.tensor(
                        current_joint_deg,
                        dtype=self.arm_joint_pos_target.dtype,
                        device=self.device,
                    )
                )
            )

        self.arm_last_action.copy_(torch.clamp(action, -ACTION_CLIP, ACTION_CLIP))
        self.arm_joint_pos_target.copy_(
            self.default_arm_joint_pos + ACTION_SCALE * self.arm_last_action
        )
        self.arm_joint_pos_target.clamp_(
            self.arm_joint_limits_rad[:, 0], self.arm_joint_limits_rad[:, 1]
        )
        target_delta = torch.clamp(
            self.arm_joint_pos_target - self._last_sent_arm_joint_pos_target,
            min=-PIPER_MAX_TARGET_DELTA_RAD_PER_POLICY_TICK,
            max=PIPER_MAX_TARGET_DELTA_RAD_PER_POLICY_TICK,
        )
        self.arm_joint_pos_target.copy_(
            self._last_sent_arm_joint_pos_target + target_delta
        )
        self.arm_command_action.copy_(
            (self.arm_joint_pos_target - self.default_arm_joint_pos) / ACTION_SCALE
        )

        if self.policy_control_mode == "move_j":
            self.target_millideg = (
                torch.rad2deg(self.arm_joint_pos_target)
                .mul(1000.0)
                .round()
                .to(dtype=torch.int64, device="cpu")
                .tolist()
            )
            self._send_joint_position_target(self.target_millideg)
        else:
            self._send_mit_joint_target(
                self.arm_joint_pos_target.detach().cpu().tolist()
            )
        self._last_sent_arm_joint_pos_target.copy_(
            self.arm_joint_pos_target
        )
        self._policy_control_mode_changed = False


    def wait_for_policy_period(self) -> None:
        now = time.monotonic()
        if self._next_observation_deadline is None:
            deadline = now + POLICY_CONTROL_PERIOD
        else:
            deadline = self._next_observation_deadline + POLICY_CONTROL_PERIOD
            if deadline <= now:
                deadline = now + POLICY_CONTROL_PERIOD
        self._next_observation_deadline = deadline

        remaining = deadline - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)

    def step(self, action: torch.Tensor) -> torch.Tensor:
        # 裁剪策略动作 并转为角度 交给机械臂控制
        self.apply_action(action)
        self.wait_for_policy_period()
        self.update_feedback_observation()
        return self.arm_history_obs_buf.clone()

    def _read_motor_speed_observation(
        self, joint_feedback_timestamp_s: float
    ) -> tuple[object, tuple[float, ...]]:
        window_end = float(joint_feedback_timestamp_s)
        if not math.isfinite(window_end) or window_end <= 0.0:
            window_end = time.time()

        previous_timestamp = self._previous_joint_feedback_timestamp_s
        is_initial_observation = previous_timestamp is None
        elapsed = (
            window_end - previous_timestamp
            if not is_initial_observation
            else 0.0
        )
        if not 0.5 * POLICY_CONTROL_PERIOD <= elapsed <= 2.0 * POLICY_CONTROL_PERIOD:
            window_start = window_end - POLICY_CONTROL_PERIOD
        else:
            window_start = previous_timestamp

        averaged = self.piper.GetArmHighSpdInfoAverage(window_start, window_end)
        self._previous_joint_feedback_timestamp_s = window_end
        self.motor_speed_average_start_timestamp_s = averaged.start_time
        self.motor_speed_average_end_timestamp_s = averaged.end_time
        self.motor_speed_average_window_s = averaged.end_time - averaged.start_time
        self.motor_speed_average_sample_counts = averaged.sample_count

        latest_motors = tuple(
            getattr(averaged.latest, f"motor_{index}") for index in range(1, 7)
        )
        self.motor_speed_latest_rad_s = tuple(
            motor.motor_speed * 0.001 for motor in latest_motors
        )
        missing_motors = [
            index
            for index, sample_count in enumerate(averaged.sample_count, start=1)
            if sample_count == 0
        ]
        if missing_motors and not is_initial_observation:
            raise RuntimeError(
                "策略速度平均窗口内缺少高速电机反馈："
                f"joints={missing_motors}, counts={averaged.sample_count}, "
                f"window={self.motor_speed_average_window_s:.6f}s"
            )

        averaged_rad_s = tuple(speed * 0.001 for speed in averaged.motor_speed)
        return averaged.latest, averaged_rad_s

    def update_feedback_observation(self) -> None:
        joint_msg = self.piper.GetArmJointMsgs()
        joint_state = joint_msg.joint_state
        joint_pos = (
            joint_state.joint_1, joint_state.joint_2, joint_state.joint_3,
            joint_state.joint_4, joint_state.joint_5, joint_state.joint_6,
        )
        self.arm_joint_pos.copy_(
            torch.tensor(joint_pos, dtype=torch.float32, device=self.device)
            * (math.pi / 180000.0)
        )
        _, averaged_motor_speed_rad_s = self._read_motor_speed_observation(
            joint_msg.time_stamp
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

    def update_arm_history_obs(self) -> None:
        history = self.arm_history_obs_buf.view(
            self.arm_policy.cfg.ObsCfg.obs_history_length, self.arm_policy.cfg.ObsCfg.num_single_observations
        )
        if not self._arm_history_initialized:
            history.copy_(
                self.arm_current_obs_buf.unsqueeze(0).expand_as(history)
            )
            self._arm_history_initialized = True
            return

        history[:-1].copy_(history[1:].clone())
        history[-1].copy_(self.arm_current_obs_buf)

    def compute_arm_observations(self) -> None:
        current_observation = torch.cat(
            (
                self.arm_joint_pos - self.default_arm_joint_pos,
                self.arm_joint_vel,
                self.arm_last_action,
                self.keypoint_error_command_b,
            )
        )
        self.arm_current_obs_buf.copy_(current_observation)

    def check_zero_link6(self) -> None:
        joint_state = self.piper.GetArmJointMsgs().joint_state
        joint_pos = (
            joint_state.joint_1,
            joint_state.joint_2,
            joint_state.joint_3,
            joint_state.joint_4,
            joint_state.joint_5,
            joint_state.joint_6,
        )

        # Piper feedback: 0.001 degree
        joint_deg = [value / 1000.0 for value in joint_pos]
        joint_rad = [math.radians(value) for value in joint_deg]

        print("实际关节角(deg):", joint_deg)

        if max(abs(value) for value in joint_deg) > 0.5:
            raise RuntimeError("机械臂尚未稳定到零位，停止 FK 零位检查")
        # link6 相对基座坐标系的位姿 list[x,y,z,roll,pitch,yaw]，单位 mm, deg
        sdk_pose = self.fk.CalFK(joint_rad)[-1]
        print("SDK link6 [mm, deg]:", sdk_pose)
        urdf_zero_pose = (
            56.1424716,
            -0.0000563,
            213.193093,
            0.0,
            85.003598,
            0.0,
        )
        position_error_mm = math.sqrt(
            sum(
                (sdk_pose[index] - urdf_zero_pose[index]) ** 2
                for index in range(3)
            )
        )
        print(f"link6 零位位置差: {position_error_mm:.4f} mm")        


    @staticmethod
    def _joint_positions_deg(joint_msg: object) -> list[float]:
        joint_state = joint_msg.joint_state
        return [
            getattr(joint_state, f"joint_{index}") / 1000.0
            for index in range(1, 7)
        ]

    @staticmethod
    def _arm_status_diagnostics(status_msg: object) -> str:
        arm_status = status_msg.arm_status
        err_status = arm_status.err_status
        communication_joints = [
            f"J{index}"
            for index in range(1, 7)
            if getattr(err_status, f"communication_status_joint_{index}", False)
        ]
        angle_limit_joints = [
            f"J{index}"
            for index in range(1, 7)
            if getattr(err_status, f"joint_{index}_angle_limit", False)
        ]
        communication_text = ",".join(communication_joints) or "无"
        angle_limit_text = ",".join(angle_limit_joints) or "无"
        err_code = int(arm_status.err_code) & 0xFFFF
        return (
            f"status={arm_status.arm_status}, ctrl_mode={arm_status.ctrl_mode}, "
            f"mode_feed={arm_status.mode_feed}, "
            f"err_code=0x{err_code:04X}, 通信异常关节={communication_text}, "
            f"角度超限关节={angle_limit_text}"
        )

    def _validate_joint_feedback(
        self,
        joint_msg: object,
        *,
        context: str,
        warn_near_limit: bool = False,
    ) -> list[float]:
        current_joint_deg = self._joint_positions_deg(joint_msg)
        violations = []
        near_limit = []
        for index, (current, limits) in enumerate(
            zip(current_joint_deg, JOINT_LIMITS_DEG), start=1
        ):
            lower, upper = limits
            if not math.isfinite(current):
                violations.append(f"J{index}=非有限值")
            elif not (
                lower - INITIAL_JOINT_LIMIT_TOLERANCE_DEG
                <= current
                <= upper + INITIAL_JOINT_LIMIT_TOLERANCE_DEG
            ):
                violations.append(
                    f"J{index}={current:.3f}° "
                    f"(允许启动范围 "
                    f"[{lower - INITIAL_JOINT_LIMIT_TOLERANCE_DEG:.1f}, "
                    f"{upper + INITIAL_JOINT_LIMIT_TOLERANCE_DEG:.1f}]°)"
                )
            elif not lower <= current <= upper:
                near_limit.append(
                    f"J{index}={current:.3f}° (名义范围 [{lower:.1f}, {upper:.1f}]°)"
                )

        if violations:
            raise RuntimeError(
                f"{context}关节反馈严重越界：{'；'.join(violations)}。"
                "禁止发送运动指令；请保持失能并检查零点/编码器圈数。"
            )
        if warn_near_limit and near_limit:
            print(
                "警告：启动关节反馈略超名义边界，已按 "
                f"{INITIAL_JOINT_LIMIT_TOLERANCE_DEG:.1f}° 容差放行："
                + "；".join(near_limit)
            )
        return current_joint_deg

    def _wait_for_arm_recovery(self, *, reset_sent_at: float) -> None:
        deadline = time.monotonic() + ARM_RECOVERY_TIMEOUT
        settle_deadline = reset_sent_at + RESET_SETTLE_TIME
        stable_samples = 0
        last_status_timestamp: float | None = None
        last_joint_timestamp = float(self.piper.GetArmJointMsgs().time_stamp)
        last_low_spd_timestamp = float(
            self.piper.GetArmLowSpdInfoMsgs().time_stamp
        )
        fresh_joint_feedback_seen = False
        fresh_low_spd_feedback_seen = False
        last_diagnostics = "尚未收到新的机械臂状态帧"

        while time.monotonic() <= deadline:
            status_msg = self.piper.GetArmStatus()
            joint_msg = self.piper.GetArmJointMsgs()
            low_spd_msg = self.piper.GetArmLowSpdInfoMsgs()
            if (
                status_msg.Hz <= 0
                or joint_msg.Hz <= 0
                or low_spd_msg.Hz <= 0
            ):
                stable_samples = 0
                last_diagnostics = (
                    "状态、关节或六轴驱动器低速反馈频率为零"
                )
                time.sleep(0.01)
                continue

            joint_timestamp = float(joint_msg.time_stamp)
            if joint_timestamp != last_joint_timestamp:
                last_joint_timestamp = joint_timestamp
                fresh_joint_feedback_seen = True

            low_spd_timestamp = float(low_spd_msg.time_stamp)
            if low_spd_timestamp != last_low_spd_timestamp:
                last_low_spd_timestamp = low_spd_timestamp
                fresh_low_spd_feedback_seen = True

            status_timestamp = float(status_msg.time_stamp)
            if status_timestamp == last_status_timestamp:
                time.sleep(0.005)
                continue
            last_status_timestamp = status_timestamp

            arm_status = int(status_msg.arm_status.arm_status)
            err_code = int(status_msg.arm_status.err_code)
            last_diagnostics = (
                f"{self._arm_status_diagnostics(status_msg)}, "
                f"driver_hz={low_spd_msg.Hz:.1f}, "
                f"enable_status={self.piper.GetArmEnableStatus()}"
            )

            recovered = (
                time.monotonic() >= settle_deadline
                and arm_status == 0
                and err_code == 0
                and fresh_joint_feedback_seen
                and fresh_low_spd_feedback_seen
            )
            if recovered:
                stable_samples += 1
                if stable_samples >= ARM_RECOVERY_STABLE_SAMPLES:
                    return
            else:
                stable_samples = 0
                # reset 后急停和关节通信异常可以短暂存在；其它故障不可自动恢复。
                if arm_status not in (0x00, 0x01, 0x05):
                    raise RuntimeError(
                        "reset 后出现不可恢复状态："
                        f"{last_diagnostics}"
                    )

            time.sleep(0.01)

        raise RuntimeError(
            "等待 reset 后机械臂恢复超时："
            f"{last_diagnostics}, "
            f"fresh_joint_feedback={fresh_joint_feedback_seen}, "
            f"fresh_driver_feedback={fresh_low_spd_feedback_seen}"
        )

    def _enable_arm_and_wait(self) -> None:
        """Enable twice with feedback between commands; do not move meanwhile."""
        deadline = time.monotonic() + ENABLE_TIMEOUT
        enable_command_count = 0
        last_enable_command_time: float | None = None
        last_status_timestamp: float | None = None
        last_low_spd_timestamp = float(
            self.piper.GetArmLowSpdInfoMsgs().time_stamp
        )
        fresh_driver_feedback_after_enable = False
        stable_samples = 0
        last_diagnostics = "尚未收到使能后的新反馈"

        while time.monotonic() <= deadline:
            now = time.monotonic()
            enable_status = self.piper.GetArmEnableStatus()
            need_enable_command = (
                enable_command_count < REQUIRED_ENABLE_COMMANDS
                or not all(enable_status)
            )
            command_interval_elapsed = (
                last_enable_command_time is None
                or now - last_enable_command_time >= ENABLE_COMMAND_INTERVAL
            )
            if need_enable_command and command_interval_elapsed:
                self.piper.EnableArm(7)
                enable_command_count += 1
                last_enable_command_time = now
                last_low_spd_timestamp = float(
                    self.piper.GetArmLowSpdInfoMsgs().time_stamp
                )
                fresh_driver_feedback_after_enable = False

            status_msg = self.piper.GetArmStatus()
            low_spd_msg = self.piper.GetArmLowSpdInfoMsgs()
            low_spd_timestamp = float(low_spd_msg.time_stamp)
            if low_spd_timestamp != last_low_spd_timestamp:
                last_low_spd_timestamp = low_spd_timestamp
                fresh_driver_feedback_after_enable = True

            arm_status = int(status_msg.arm_status.arm_status)
            err_code = int(status_msg.arm_status.err_code)
            enable_status = self.piper.GetArmEnableStatus()
            last_diagnostics = (
                f"{self._arm_status_diagnostics(status_msg)}, "
                f"driver_hz={low_spd_msg.Hz:.1f}, "
                f"enable_status={enable_status}, "
                f"enable_commands={enable_command_count}"
            )

            status_timestamp = float(status_msg.time_stamp)
            new_status_frame = status_timestamp != last_status_timestamp
            if new_status_frame:
                last_status_timestamp = status_timestamp

            enabled = (
                enable_command_count >= REQUIRED_ENABLE_COMMANDS
                and last_enable_command_time is not None
                and now - last_enable_command_time
                >= ENABLE_POST_COMMAND_SETTLE_TIME
                and fresh_driver_feedback_after_enable
                and low_spd_msg.Hz > 0
                and all(enable_status)
                and arm_status == 0
                and err_code == 0
            )
            if enabled and new_status_frame:
                stable_samples += 1
                if stable_samples >= ENABLE_STABLE_SAMPLES:
                    return
            elif not enabled:
                stable_samples = 0
                if arm_status not in (0x00, 0x05):
                    raise RuntimeError(
                        "重新使能期间机械臂状态异常："
                        f"{last_diagnostics}"
                    )

            time.sleep(0.01)

        raise RuntimeError(
            "重新使能机械臂超时："
            f"{last_diagnostics}, "
            f"fresh_driver_feedback={fresh_driver_feedback_after_enable}"
        )

    def _send_joint_position_target(self, target_millideg: Sequence[int]) -> None:
        self.piper.MotionCtrl_2(
            ctrl_mode=0x01,
            move_mode=0x01,
            move_spd_rate_ctrl=SPEED_PERCENT,
            is_mit_mode=0xAD,
        )
        self.piper.JointCtrl(*target_millideg)

    def _send_mit_joint_target(self, target_joint_rad: Sequence[float]) -> None:
        self.piper.MotionCtrl_2(
            ctrl_mode=0x01,
            move_mode=0x04,
            move_spd_rate_ctrl=0,
            is_mit_mode=0xAD,
        )
        for index, position in enumerate(target_joint_rad, start=1):
            self.piper.JointMitCtrl(
                motor_num=index,
                pos_ref=float(position),
                vel_ref=0.0,
                kp=MIT_JOINT_KP[index - 1],
                kd=MIT_JOINT_KD[index - 1],
                t_ref=0.0,
            )

    def reset(self) -> None:
        joint_msg = self.piper.GetArmJointMsgs()
        status_msg = self.piper.GetArmStatus()
        low_spd_msg = self.piper.GetArmLowSpdInfoMsgs()
        if (
            joint_msg.Hz <= 0
            or status_msg.Hz <= 0
            or low_spd_msg.Hz <= 0
        ):
            raise RuntimeError("CAN 反馈频率为零")

        current_joint_deg = self._validate_joint_feedback(
            joint_msg,
            context="启动前",
            warn_near_limit=True,
        )
        initial_arm_status = int(status_msg.arm_status.arm_status)
        if initial_arm_status not in (0x00, 0x01, 0x05):
            raise RuntimeError(
                "机械臂启动前存在不可自动恢复故障："
                f"{self._arm_status_diagnostics(status_msg)}"
            )
        print(
            "启动关节反馈(°): "
            + "[" + ", ".join(f"{value:.3f}" for value in current_joint_deg) + "]"
        )

        self._enable_attempted = True

        initial_ctrl_mode = int(status_msg.arm_status.ctrl_mode)
        initial_err_code = int(status_msg.arm_status.err_code)
        can_control_state_is_healthy = (
            initial_arm_status == 0
            and initial_err_code == 0
            and initial_ctrl_mode == 0x01
        )
        reset_performed = False

        if can_control_state_is_healthy:
            print("控制器处于正常 CAN_CTRL，跳过控制器恢复复位；仍执行初始化回位")
        else:
            # S-V1.7-3 的 STANDBY 可能仍锁存上一次 stop/控制器状态：此时
            # enable_status 即使为 True，0x151 也可能被忽略。必须先 reset，
            # 再重新使能两次。真实急停/通信故障同样走这条恢复路径。
            reset_sent_at = time.monotonic()
            self.piper.MotionCtrl_1(0x02, 0, 0)
            self._wait_for_arm_recovery(reset_sent_at=reset_sent_at)
            reset_performed = True
            print("机械臂控制器恢复复位完成")

        # reset 后即使低速反馈暂时仍显示 True，也必须重新发送两次使能。
        if reset_performed or not all(self.piper.GetArmEnableStatus()):
            self._enable_arm_and_wait()
            print("机械臂重新使能完成")

        self.control_started = True
        self._move_j_to_joint_target(
            self.target_init_joint,
            context="机械臂初始化 MOVE J 回位",
        )

    def move_j_to_zero(self) -> None:
        """Return all joints to the policy default position using MOVE J."""
        if not self.initialized or not self.control_started:
            raise RuntimeError("机械臂尚未成功初始化，禁止执行回零运动")
        print("策略控制完成，开始使用 MOVE J 返回默认关节位置")
        self._move_j_to_joint_target(
            self.target_init_joint,
            context="策略结束 MOVE J 归零",
        )

    def _move_j_to_joint_target(
        self,
        target_joint_deg: Sequence[float],
        *,
        context: str,
    ) -> None:
        self._policy_control_mode_changed = True
        target_deg = tuple(float(value) for value in target_joint_deg)
        if len(target_deg) != 6:
            raise ValueError("MOVE J 目标必须包含 6 个关节角")
        for index, (angle, limits) in enumerate(zip(target_deg, JOINT_LIMITS_DEG), start=1):
            lower, upper = limits
            if not math.isfinite(angle) or not lower <= angle <= upper:
                raise ValueError(
                    f"J{index} MOVE J 目标角度越界：{angle:.3f}°，"
                    f"允许范围 [{lower:.1f}, {upper:.1f}]°"
                )
        target = tuple(math.radians(value) for value in target_deg)
        joint_msg = self.piper.GetArmJointMsgs()
        if joint_msg.Hz <= 0:
            raise RuntimeError(f"{context}前 CAN 关节反馈频率为零")
        current_deg = self._validate_joint_feedback(joint_msg, context=f"{context}前")
        command_target = tuple(math.radians(value) for value in current_deg)
        print(
            f"{context}：MOVE_J 0xAD 从实测位置接管，"
            f"current_deg={current_deg}, target_deg={target_deg}"
        )
        control_deadline = time.monotonic() + CAN_CONTROL_READY_TIMEOUT
        motion_deadline: float | None = None
        final_target_sent = False
        control_stable_samples = 0
        last_status_timestamp = None
        stable_samples = 0
        last_feedback_timestamp = joint_msg.time_stamp
        last_feedback_time = time.monotonic()

        while True:
            self._send_joint_position_target(
                [round(math.degrees(value) * 1000.0) for value in command_target]
            )
            time.sleep(POLICY_CONTROL_PERIOD)

            joint_msg = self.piper.GetArmJointMsgs()
            status_msg = self.piper.GetArmStatus()
            now = time.monotonic()

            fresh_joint_feedback = joint_msg.time_stamp != last_feedback_timestamp
            if fresh_joint_feedback:
                last_feedback_timestamp = joint_msg.time_stamp
                last_feedback_time = now
            elif now - last_feedback_time > JOINT_FEEDBACK_TIMEOUT:
                raise RuntimeError(f"{context}关节反馈超时")

            status = status_msg.arm_status
            if (
                joint_msg.Hz <= 0
                or status_msg.Hz <= 0
                or int(status.arm_status) != 0
                or int(status.err_code) != 0
            ):
                raise RuntimeError(
                    f"{context}机械臂状态异常："
                    f"{self._arm_status_diagnostics(status_msg)}"
                )

            current_deg = self._validate_joint_feedback(
                joint_msg,
                context=context,
            )
            mode_ready = (
                int(status.ctrl_mode) == 0x01
                and int(status.mode_feed) == 0x01
            )
            if motion_deadline is None:
                if status_msg.time_stamp != last_status_timestamp:
                    last_status_timestamp = status_msg.time_stamp
                    control_stable_samples = (
                        control_stable_samples + 1 if mode_ready else 0
                    )
                if (
                    control_stable_samples >= CAN_CONTROL_STABLE_SAMPLES
                    and fresh_joint_feedback
                ):
                    motion_deadline = now + MOTION_TIMEOUT
                    print(f"{context}：MOVE_J 0xAD 接管完成，开始渐进回位")
                elif now > control_deadline:
                    raise RuntimeError(
                        f"{context} MOVE_J 接管超时："
                        f"{self._arm_status_diagnostics(status_msg)}"
                    )
                continue

            if not mode_ready:
                raise RuntimeError(
                    f"{context} MOVE_J 模式丢失："
                    f"{self._arm_status_diagnostics(status_msg)}"
                )

            if command_target == target and not final_target_sent:
                final_target_sent = True
                print(
                    f"{context}：主机已下发最终目标，"
                    f"command_deg={target_deg}, current_deg={current_deg}"
                )

            max_error = max(
                abs(current - desired)
                for current, desired in zip(current_deg, target_deg)
            )

            if fresh_joint_feedback:
                if command_target == target and max_error <= MOVE_J_POSITION_TOLERANCE_DEG:
                    stable_samples += 1
                    if stable_samples >= MOVE_J_POSITION_STABLE_SAMPLES:
                        print(f"{context}成功，最大误差 {max_error:.3f}°")
                        return
                else:
                    stable_samples = 0

            if now > motion_deadline:
                motor_msg = self.piper.GetArmHighSpdInfoMsgs()
                motor_feedback = [
                    (
                        round(getattr(motor_msg, f"motor_{index}").motor_speed * 0.001, 4),
                        getattr(motor_msg, f"motor_{index}").current,
                    )
                    for index in range(1, 7)
                ]
                raise RuntimeError(
                    f"{context}超时：current_deg={current_deg}, "
                    f"target_deg={target_deg}, "
                    f"command_deg={tuple(math.degrees(value) for value in command_target)}, "
                    f"最大误差 {max_error:.3f}°，"
                    f"enable_status={self.piper.GetArmEnableStatus()}, "
                    f"motor_feedback(speed_rad_s, current_raw)={motor_feedback}, "
                    f"motor_feedback_timestamp={motor_msg.time_stamp}, "
                    f"motor_feedback_hz={motor_msg.Hz}"
                )

            if fresh_joint_feedback:
                command_target = tuple(
                    desired
                    if abs(desired - commanded) <= PIPER_MAX_TARGET_DELTA_RAD_PER_POLICY_TICK
                    else commanded + math.copysign(
                        PIPER_MAX_TARGET_DELTA_RAD_PER_POLICY_TICK,
                        desired - commanded,
                    )
                    for commanded, desired in zip(command_target, target)
                )

    def quick_stop(self):
        """Emergency-only damped stop for interruption or failures."""
        stop_errors = []
        for _ in range(5):
            try:
                self.piper.MotionCtrl_1(0x01, 0, 0)
            except Exception as exc:
                stop_errors.append(exc)
            time.sleep(0.01)

        self.control_started = False
        self._enable_attempted = False
        self.initialized = False
        if stop_errors:
            print(f"警告：快速急停指令发送失败：{stop_errors[-1]}")

    def disconnect(self) -> None:
        try:
            self.piper.DisconnectPort()
        except Exception as exc:
            print(f"警告：关闭 CAN 连接失败：{exc}")
        finally:
            self.control_started = False
            self._enable_attempted = False
            self.initialized = False


def _run_policy_from_cli(
    manipulation: Manipulation,
    policy_steps: int,
) -> None:
    policy_completed_normally = False
    parked_for_normal_exit = False
    try:
        manipulation.run_policy(
            num_steps=policy_steps if policy_steps > 0 else None
        )
        policy_completed_normally = True
    except KeyboardInterrupt:
        print("用户中断 policy 控制")
    finally:
        try:
            manipulation.print_target_status()
            if policy_completed_normally and policy_steps > 0:
                manipulation.move_j_to_zero()
                parked_for_normal_exit = True
        finally:
            if not parked_for_normal_exit and manipulation.control_started:
                manipulation.quick_stop()
            manipulation.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--can_name",
        type=str,
        default="can_piper",
        help="PC2 上由 socketcand 暴露的 CAN 设备名称",
    )
    parser.add_argument(
        "--can_host",
        type=str,
        default="192.168.123.162",
        help="PC2 socketcand 地址",
    )
    parser.add_argument(
        "--can_port",
        type=int,
        default=29536,
        help="PC2 socketcand TCP 端口",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=None,
        help="RSL-RL manipulation checkpoint path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Policy inference device, for example cpu or cuda:0",
    )
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--target_pos_b",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Target ee_gripper position in command frame, in meters",
    )
    target_group.add_argument(
        "--random_target",
        action="store_true",
        help="Randomly sample one target pose from the training ranges",
    )
    parser.add_argument(
        "--run_policy",
        action="store_true",
        help="Run the loaded policy and send its actions to the arm",
    )
    parser.add_argument(
        "--policy_control_mode",
        choices=("move_j", "mit"),
        default="move_j",
        help="Policy sender: MOVE J + 0xAD or MOVE M + 0xAD (MIT); homing always uses MOVE J",
    )
    parser.add_argument(
        "--policy_steps",
        type=int,
        default=0,
        help="Number of policy steps; 0 runs until interrupted",
    )
    args = parser.parse_args()
    if args.run_policy and args.checkpoint_path is None:
        parser.error("--run_policy requires --checkpoint_path")
    if args.policy_steps < 0:
        parser.error("--policy_steps must be non-negative")

    if args.random_target:
        target_pos_b = None
    elif args.target_pos_b is not None:
        target_pos_b = args.target_pos_b
    else:
        target_pos_b = DEFAULT_TARGET_POS_B

    try:
        manipulation = Manipulation(
            checkpoint_path=args.checkpoint_path,
            device=args.device,
            can_name=args.can_name,
            can_host=args.can_host,
            can_port=args.can_port,
            target_pos_b=target_pos_b,
            policy_control_mode=args.policy_control_mode,
        )
    except KeyboardInterrupt:
        print("用户中断机械臂初始化", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)

    if args.run_policy:
        _run_policy_from_cli(manipulation, args.policy_steps)
    else:
        manipulation.disconnect()
