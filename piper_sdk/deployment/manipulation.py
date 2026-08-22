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
from typing import Iterable, Sequence
from piper_sdk import C_PiperInterface_V2, C_PiperForwardKinematics
import torch
from piper_sdk.deployment import math_utils
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
SINGLE_OBSERVATION_DIM = 27
OBS_HISTORY_LENGTH = 5
SPEED_PERCENT = 5
MOTION_TIMEOUT = 25.0
TARGET_REACHED_KEYPOINT_ERROR_M = 0.02


class ArmPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(SINGLE_OBSERVATION_DIM * OBS_HISTORY_LENGTH, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 6),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.mlp(observation)


class Manipulation:
    def __init__(
        self, 
        checkpoint_path: str | Path | None = None,
        device: str = "cpu",
        can_name: str = "can0",
        target_pos_b: Sequence[float] | torch.Tensor | None = DEFAULT_TARGET_POS_B,
    )->None:
        self.device = torch.device(device)
        self.arm_policy: ArmPolicy | None = None
        self.checkpoint_path: Path | None = None
        if checkpoint_path is not None:
            self.load_arm_policy(checkpoint_path)

        self.piper = C_PiperInterface_V2(can_name)
        self.fk = C_PiperForwardKinematics(dh_is_offset=1)
        self.control_started = False
        self.target_init_joint = [math.degrees(angle) for angle in POLICY_INIT_JOINT_POS]
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
            # 等待3s 关节和机械臂状态反馈，避免读取 SDK 的初始零值。
            feedback_deadline = time.monotonic() + 3.0
            while True:
                joint_msg = self.piper.GetArmJointMsgs()
                status_msg = self.piper.GetArmStatus()
                # 收到piper的反馈 可以退出等待
                if joint_msg.Hz > 0 and status_msg.Hz > 0:
                    break
                if time.monotonic() > feedback_deadline:
                    raise RuntimeError("没有收到完整反馈，请检查 CAN、机械臂电源和从臂模式")
                time.sleep(0.02)

            self.reset()
            # self.check_zero_link6()
            
        except KeyboardInterrupt:
            print("用户中断")
            if self.control_started:
                self.quick_stop()
        except Exception as e:
            print(f"初始化机械臂失败：{e}")
            if self.control_started:
                self.quick_stop()
        
        self.arm_joint_pos = torch.zeros(6, dtype=torch.float32,device=self.device)
        self.arm_joint_vel = torch.zeros(6, dtype=torch.float32,device=self.device)
        self.arm_last_action = torch.zeros(6, dtype=torch.float32,device=self.device)
        self.arm_command_action = torch.zeros_like(self.arm_last_action)
        self.default_arm_joint_pos = torch.tensor(
            POLICY_INIT_JOINT_POS, dtype=torch.float32, device=self.device
        )
        self.arm_joint_pos_target = self.default_arm_joint_pos.clone()
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
        self.target_pos_b = torch.zeros(3, dtype=torch.float32,device=self.device)
        self.pose_command_b = torch.zeros(7, dtype=torch.float32,device=self.device)
        self.current_ee_pose_b = torch.zeros_like(self.pose_command_b)
        self.current_ee_pose_b[3] = 1.0
        self.pose_command_b[3] = 1.0
        half_keypoint_side = 0.05
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
            SINGLE_OBSERVATION_DIM, dtype=torch.float32, device=self.device
        )
        self.arm_history_obs_buf = torch.zeros(
            SINGLE_OBSERVATION_DIM * OBS_HISTORY_LENGTH,
            dtype=torch.float32,
            device=self.device,
        )
        self._arm_history_initialized = False

        self.set_arm_command(target_pos_b)
        self.update_feedback_observation()

    def load_arm_policy(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Policy checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict) or "actor_state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint does not contain actor_state_dict: {path}")
        actor_state = checkpoint["actor_state_dict"]
        if not isinstance(actor_state, dict):
            raise TypeError("actor_state_dict must be a mapping of tensor names to tensors")

        actor = ArmPolicy().to(self.device)
        expected_keys = set(actor.state_dict())
        mlp_state = {
            key: value for key, value in actor_state.items() if key.startswith("mlp.")
        }
        missing_keys = expected_keys - set(mlp_state)
        unexpected_keys = set(mlp_state) - expected_keys
        unsupported_keys = set(actor_state) - set(mlp_state) - {"distribution.std_param"}
        if missing_keys or unexpected_keys or unsupported_keys:
            raise RuntimeError(
                "Policy architecture does not match the deployment actor: "
                f"missing={sorted(missing_keys)}, "
                f"unexpected={sorted(unexpected_keys)}, "
                f"unsupported={sorted(unsupported_keys)}"
            )

        actor.load_state_dict(mlp_state, strict=True)
        actor.eval().requires_grad_(False)
        self.arm_policy = actor
        self.checkpoint_path = path

    def get_action(
        self, observation: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.arm_policy is None:
            raise RuntimeError("Arm policy is not loaded")

        if observation is None:
            if not self._arm_history_initialized:
                raise RuntimeError("Arm observation history is not initialized")
            observation = self.arm_history_obs_buf
        observation = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        )
        if observation.shape == (SINGLE_OBSERVATION_DIM * OBS_HISTORY_LENGTH,):
            observation = observation.unsqueeze(0)
        expected_shape = (1, SINGLE_OBSERVATION_DIM * OBS_HISTORY_LENGTH)
        if observation.shape != expected_shape:
            raise ValueError(
                f"Expected observation shape {expected_shape} or "
                f"({expected_shape[1]},), got {observation.shape}."
            )

        with torch.inference_mode():
            action = self.arm_policy(observation)
        if action.shape != (1, 6):
            raise RuntimeError(f"Expected policy action shape (1, 6), got {action.shape}.")
        return action

    def run_policy(self, num_steps: int | None = None) -> None:
        if self.arm_policy is None:
            raise RuntimeError("Arm policy is not loaded")
        if num_steps is not None and num_steps < 1:
            raise ValueError("num_steps must be positive or None")

        step_count = 0
        while num_steps is None or step_count < num_steps:
            action = self.get_action()
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

        self.target_pos_b.copy_(target_pos)
        self.pose_command_b[:3].copy_(target_pos)
        self.pose_command_b[3:].copy_(target_quat)
        self.target_keypoints_b.copy_(
            self.compute_keypoints_b(
                self.pose_command_b[:3], self.pose_command_b[3:]
            )
        )

    def apply_action(self, action: torch.Tensor) -> None:
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
            raise ValueError("Action must contain only finite values.")
        self.arm_last_action.copy_(torch.clamp(action, -ACTION_CLIP, ACTION_CLIP))
        self.arm_joint_pos_target.copy_(
            self.default_arm_joint_pos + ACTION_SCALE * self.arm_last_action
        )
        self.arm_joint_pos_target.clamp_(
            self.arm_joint_limits_rad[:, 0], self.arm_joint_limits_rad[:, 1]
        )
        self.arm_command_action.copy_(
            (self.arm_joint_pos_target - self.default_arm_joint_pos) / ACTION_SCALE
        )

        joint_msg = self.piper.GetArmJointMsgs()
        status_msg = self.piper.GetArmStatus()
        if joint_msg.Hz <= 0 or status_msg.Hz <= 0:
            raise RuntimeError("CAN 反馈频率为零")
        arm_status = status_msg.arm_status.arm_status
        if int(arm_status) != 0:
            raise RuntimeError(f"机械臂状态异常：{arm_status}")

        target_millideg = (
            torch.rad2deg(self.arm_joint_pos_target)
            .mul(1000.0)
            .round()
            .to(dtype=torch.int64, device="cpu")
            .tolist()
        )
        # Piper keeps the MOVE J target in its internal servo loop.  At the
        # policy rate we only need to refresh it once, not once per sim substep.
        self.piper.MotionCtrl_2(0x01, 0x01, SPEED_PERCENT, 0x00)
        self.piper.JointCtrl(*target_millideg)

    def wait_for_policy_period(self) -> None:
        now = time.monotonic()
        if self._next_observation_deadline is None:
            deadline = now + POLICY_CONTROL_PERIOD
        else:
            deadline = self._next_observation_deadline + POLICY_CONTROL_PERIOD
            if deadline <= now:
                # Do not issue catch-up commands faster than the trained rate.
                deadline = now + POLICY_CONTROL_PERIOD
        self._next_observation_deadline = deadline

        remaining = deadline - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)

    def step(self, action: torch.Tensor) -> torch.Tensor:
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
            OBS_HISTORY_LENGTH, SINGLE_OBSERVATION_DIM
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
        if current_observation.shape != (SINGLE_OBSERVATION_DIM,):
            raise RuntimeError(
                f"Expected {SINGLE_OBSERVATION_DIM} observations, "
                f"got {current_observation.shape}."
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


    # 重置机械臂到初始位置
    def reset(self)->None:
        try:
            joint_msg = self.piper.GetArmJointMsgs()
            status_msg = self.piper.GetArmStatus()
            if joint_msg.Hz <= 0 or status_msg.Hz <= 0:
                raise RuntimeError("CAN 反馈频率为零")
            arm_status = status_msg.arm_status.arm_status # 0x00字段表示状态正常
            if int(arm_status) != 0:
                raise RuntimeError(f"机械臂启动前状态异常：{arm_status}")

            self.control_started = True
            enable_timeout = time.monotonic() + 5.0
            # 使能机械臂
            while not self.piper.EnablePiper():
                if time.monotonic() > enable_timeout:
                    raise RuntimeError("机械臂使能超时")
                time.sleep(0.01)
            motion_timeout = time.monotonic() + MOTION_TIMEOUT
            last_feedback_time_stamp = joint_msg.time_stamp
            last_feedback_time = time.monotonic()
            while True:
                # 发送 运动模式的配置指令 使用CAN指令 MOVE J关节运动模式 设置速度百分比 使用位置速度模式
                self.piper.MotionCtrl_2(0x01, 0x01, SPEED_PERCENT, 0x00)
                
                target = [round(angle * 1000) for angle in self.target_init_joint]
                self.piper.JointCtrl(*target)

                joint_msg = self.piper.GetArmJointMsgs()
                status_msg = self.piper.GetArmStatus()
                now = time.monotonic()

                if joint_msg.time_stamp != last_feedback_time_stamp:
                    last_feedback_time_stamp = joint_msg.time_stamp
                    last_feedback_time = now
                elif now - last_feedback_time > 0.5:
                    raise RuntimeError("关节状态反馈超时")

                if joint_msg.Hz <= 0 or status_msg.Hz <= 0:
                    raise RuntimeError("CAN 反馈频率为零")
                arm_status = status_msg.arm_status.arm_status # 0x00字段表示状态正常
                arm_ctrl_mode = status_msg.arm_status.ctrl_mode # 0x01字段表示CAN 控制模式

                if int(arm_status) != 0:
                    raise RuntimeError(f"机械臂状态异常：{arm_status, arm_ctrl_mode}")
                

                current_joint_deg = [
                    joint_msg.joint_state.joint_1 / 1000.0,
                    joint_msg.joint_state.joint_2 / 1000.0,
                    joint_msg.joint_state.joint_3 / 1000.0,
                    joint_msg.joint_state.joint_4 / 1000.0,
                    joint_msg.joint_state.joint_5 / 1000.0,
                    joint_msg.joint_state.joint_6 / 1000.0,
                ]

                deg_error = [
                    abs(current - target)
                    for current, target in zip(current_joint_deg, self.target_init_joint)
                ]
                max_error = max(deg_error)
                if max_error <= 0.5:
                    print(f"机械臂初始化到目标关节位置成功，最大误差 {max_error:.3f}°")
                    break
                if now > motion_timeout:
                    raise RuntimeError(
                        f"关节运动超时"
                    )
                time.sleep(0.02)
        except KeyboardInterrupt:
            print("用户中断")
            if self.control_started:
                self.quick_stop()
        except Exception as e:
            print(f"初始化机械臂失败：{e}")
            if self.control_started:
                self.quick_stop()

    


    def quick_stop(self):
        stop_errors = []
        for _ in range(5):
            try:
                self.piper.MotionCtrl_1(0x01, 0, 0)
            except Exception as exc:
                stop_errors.append(exc)
            time.sleep(0.01)
    
        if stop_errors:
            print(f"警告：快速急停指令发送失败：{stop_errors[-1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--can_name",
        type=str,
        default="can0",
        help="CAN 设备名称，默认 can0",
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

    manipulation = Manipulation(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        can_name=args.can_name,
        target_pos_b=target_pos_b,
    )
    if args.run_policy:
        try:
            manipulation.run_policy(
                num_steps=args.policy_steps if args.policy_steps > 0 else None
            )
        except KeyboardInterrupt:
            print("用户中断 policy 控制")
        finally:
            try:
                manipulation.print_target_status()
            finally:
                if manipulation.control_started:
                    manipulation.quick_stop()
