#!/usr/bin/env python3
"""PiPER + D435i eye-in-hand calibration with one AprilTag.

This program runs on the workstation.  It reads PiPER joint feedback through
PC2's socketcand service and keeps one SSH camera worker running on PC2, where
the D435i is physically connected.  It never sends a motion command.

The calibrated transform is ``T_link_camera``::

    p_link = T_link_camera @ p_color_camera

The current target defaults are the observed tag36h11 ID 2 with a measured
black-square side length of 157 mm.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import getpass
import json
import math
import os
import select
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from piper_sdk import C_PiperForwardKinematics, C_PiperInterface_V2


DEFAULT_CAN_NAME = "can_piper"
DEFAULT_PC2_HOST = "192.168.123.162"
DEFAULT_CAN_PORT = 29536
DEFAULT_PC2_USER = "unitree"
DEFAULT_TAG_ID = 2
DEFAULT_TAG_SIZE_MM = 157.0
DEFAULT_MOUNT_LINK = 6
DEFAULT_METHOD = "park"
MAX_USABLE_TRANSLATION_RMS_MM = 10.0
MAX_USABLE_ROTATION_RMS_DEG = 1.0


REMOTE_CAMERA_WORKER = r'''
import base64
import json
import sys
import time

import cv2
import numpy as np
from pyrealsense2 import pyrealsense2 as rs


def emit(message):
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


warmup_frames = int(sys.argv[1])
frames_per_capture = int(sys.argv[2])
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

try:
    profile = pipeline.start(config)
    for _ in range(warmup_frames):
        pipeline.wait_for_frames(5000)

    stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = stream.get_intrinsics()
    device = profile.get_device()
    emit({
        "type": "ready",
        "camera_name": device.get_info(rs.camera_info.name),
        "camera_serial": device.get_info(rs.camera_info.serial_number),
        "resolution": [intrinsics.width, intrinsics.height],
        "intrinsics": {
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "ppx": intrinsics.ppx,
            "ppy": intrinsics.ppy,
            "coeffs": list(intrinsics.coeffs),
            "distortion_model": str(intrinsics.model),
        },
    })

    for command in sys.stdin:
        command = command.strip()
        if command == "quit":
            break
        if command != "capture":
            emit({"type": "error", "error": "unknown command: " + command})
            continue

        processing_started = time.monotonic()
        best_candidate = None
        for _ in range(frames_per_capture):
            frame = pipeline.wait_for_frames(5000).get_color_frame()
            image = np.asanyarray(frame.get_data()).copy()
            preview = cv2.resize(
                image, (480, 270), interpolation=cv2.INTER_AREA
            )
            gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
            candidate = (
                sharpness,
                image,
                int(frame.get_frame_number()),
                float(frame.get_timestamp()),
            )
            if best_candidate is None or sharpness > best_candidate[0]:
                best_candidate = candidate

        sharpness, image, frame_number, timestamp_ms = best_candidate
        ok, jpeg = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 96]
        )
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        emit({
            "type": "frame",
            "frame_number": frame_number,
            "camera_timestamp_ms": timestamp_ms,
            "sharpness": sharpness,
            "processing_ms": (time.monotonic() - processing_started) * 1000.0,
            "jpeg_b64": base64.b64encode(jpeg.tobytes()).decode("ascii"),
        })
except Exception as exc:
    emit({"type": "fatal", "error": repr(exc)})
finally:
    pipeline.stop()
'''


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


def rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def matrix_from_sdk_pose(pose_mm_deg: list[float]) -> np.ndarray:
    """Convert SDK [x,y,z,roll,pitch,yaw] into base-from-link SE(3)."""
    x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg = pose_mm_deg
    roll, pitch, yaw = map(math.radians, (roll_deg, pitch_deg, yaw_deg))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll)
    transform[:3, 3] = np.array((x_mm, y_mm, z_mm), dtype=np.float64) * 0.001
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ transform[:3, 3]
    return inverse


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def average_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    u, _, vh = np.linalg.svd(np.sum(rotations, axis=0))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vh)
    return u @ correction @ vh


def rpy_degrees_from_matrix(rotation: np.ndarray) -> list[float]:
    horizontal = math.hypot(rotation[0, 0], rotation[1, 0])
    if horizontal > 1e-9:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], horizontal)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], horizontal)
        yaw = 0.0
    return [math.degrees(value) for value in (roll, pitch, yaw)]


def quaternion_xyzw_from_matrix(rotation: np.ndarray) -> list[float]:
    # Eigenvector of the symmetric Davenport matrix; scalar component is last.
    r = rotation
    k = np.array(
        [
            [r[0, 0] - r[1, 1] - r[2, 2], r[0, 1] + r[1, 0], r[0, 2] + r[2, 0], r[2, 1] - r[1, 2]],
            [r[0, 1] + r[1, 0], r[1, 1] - r[0, 0] - r[2, 2], r[1, 2] + r[2, 1], r[0, 2] - r[2, 0]],
            [r[0, 2] + r[2, 0], r[1, 2] + r[2, 1], r[2, 2] - r[0, 0] - r[1, 1], r[1, 0] - r[0, 1]],
            [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1], r[0, 0] + r[1, 1] + r[2, 2]],
        ],
        dtype=np.float64,
    ) / 3.0
    values, vectors = np.linalg.eigh(k)
    quaternion = vectors[:, int(np.argmax(values))]
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion.tolist()


def transform_record(transform: np.ndarray, meaning: str) -> dict[str, Any]:
    return {
        "meaning": meaning,
        "matrix": transform.tolist(),
        "translation_m": transform[:3, 3].tolist(),
        "rotation_matrix": transform[:3, :3].tolist(),
        "rotation_rpy_deg_xyz": rpy_degrees_from_matrix(transform[:3, :3]),
        "quaternion_xyzw": quaternion_xyzw_from_matrix(transform[:3, :3]),
    }


class RemoteRealSense:
    def __init__(
        self,
        host: str,
        user: str,
        auth: str,
        ssh_key: Path | None,
        password_env: str,
        warmup_frames: int,
        frames_per_capture: int,
        stderr_path: Path,
    ) -> None:
        self._capture_timeout = 5.0 * frames_per_capture + 15.0
        payload = base64.b64encode(REMOTE_CAMERA_WORKER.encode("utf-8")).decode("ascii")
        remote_command = (
            "python3 -u -c \"import base64;"
            f"exec(base64.b64decode('{payload}'))\" "
            f"{warmup_frames} {frames_per_capture}"
        )
        command: list[str] = []
        environment = os.environ.copy()
        if auth == "password":
            if shutil.which("sshpass") is None:
                raise RuntimeError("password 模式需要系统命令 sshpass")
            password = environment.get(password_env)
            if password is None:
                password = getpass.getpass(f"{user}@{host} SSH 密码: ")
            environment["SSHPASS"] = password
            command.extend(("sshpass", "-e"))

        command.extend(
            (
                "ssh",
                "-T",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                "ConnectTimeout=8",
                "-o",
                "ServerAliveInterval=5",
            )
        )
        if auth == "key":
            command.extend(("-o", "BatchMode=yes"))
            if ssh_key is not None:
                command.extend(("-i", str(ssh_key)))
        command.extend((f"{user}@{host}", remote_command))

        self._stderr_file = stderr_path.open("w", encoding="utf-8")
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            text=True,
            bufsize=1,
            env=environment,
        )
        ready = self._read_message(timeout=20.0)
        if ready.get("type") != "ready":
            raise RuntimeError(f"PC2 相机启动失败: {ready}")
        self.camera_info = ready

    def _read_message(self, timeout: float) -> dict[str, Any]:
        assert self._process.stdout is not None
        readable, _, _ = select.select([self._process.stdout], [], [], timeout)
        if not readable:
            raise TimeoutError("等待 PC2 相机响应超时")
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError(
                f"PC2 相机进程已退出，returncode={self._process.poll()}"
            )
        message = json.loads(line)
        if message.get("type") in ("error", "fatal"):
            raise RuntimeError(f"PC2 相机错误: {message.get('error')}")
        return message

    def capture(self) -> tuple[np.ndarray, bytes, dict[str, Any]]:
        assert self._process.stdin is not None
        self._process.stdin.write("capture\n")
        self._process.stdin.flush()
        message = self._read_message(timeout=self._capture_timeout)
        if message.get("type") != "frame":
            raise RuntimeError(f"PC2 返回了非图像消息: {message}")
        jpeg_bytes = base64.b64decode(message.pop("jpeg_b64"))
        image = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("无法解码 PC2 返回的 JPEG")
        return image, jpeg_bytes, message

    def close(self) -> None:
        if self._process.poll() is None and self._process.stdin is not None:
            try:
                self._process.stdin.write("quit\n")
                self._process.stdin.flush()
                self._process.wait(timeout=5.0)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self._process.terminate()
                self._process.wait(timeout=5.0)
        self._stderr_file.close()


class PiperStateReader:
    def __init__(
        self,
        can_name: str,
        can_host: str,
        can_port: int,
        dh_is_offset: int,
        mount_link: int,
    ) -> None:
        self._piper = C_PiperInterface_V2(
            can_name,
            judge_flag=False,
            can_auto_init=False,
            dh_is_offset=dh_is_offset,
        )
        self._piper.CreateCanBus(
            can_name=can_name,
            bustype="socketcand",
            judge_flag=False,
            host=can_host,
            port=can_port,
            tcp_tune=True,
        )
        self._piper.ConnectPort(piper_init=False)
        self._fk = C_PiperForwardKinematics(dh_is_offset=dh_is_offset)
        self._mount_link = mount_link

        deadline = time.monotonic() + 5.0
        while self._piper.GetArmJointMsgs().Hz <= 0.0:
            if time.monotonic() >= deadline:
                raise RuntimeError("5 秒内没有收到 PiPER 关节反馈")
            time.sleep(0.05)

    def read_joint_degrees(self) -> tuple[list[float], float]:
        message = self._piper.GetArmJointMsgs()
        state = message.joint_state
        joints = [getattr(state, f"joint_{index}") / 1000.0 for index in range(1, 7)]
        return joints, float(message.time_stamp)

    def link_pose_from_joints(
        self, joint_degrees: list[float]
    ) -> tuple[list[float], np.ndarray]:
        joint_radians = [math.radians(value) for value in joint_degrees]
        pose = [float(value) for value in self._fk.CalFK(joint_radians)[self._mount_link - 1]]
        return pose, matrix_from_sdk_pose(pose)

    def close(self) -> None:
        self._piper.DisconnectPort(thread_timeout=1.2)


class AprilTagPoseEstimator:
    def __init__(
        self,
        tag_id: int,
        tag_size_mm: float,
        camera_info: dict[str, Any],
    ) -> None:
        self.tag_id = tag_id
        self.tag_size_m = tag_size_mm * 0.001
        intrinsics = camera_info["intrinsics"]
        self.camera_matrix = np.array(
            (
                (intrinsics["fx"], 0.0, intrinsics["ppx"]),
                (0.0, intrinsics["fy"], intrinsics["ppy"]),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        self.distortion = np.asarray(intrinsics["coeffs"], dtype=np.float64)
        half = self.tag_size_m * 0.5
        # OpenCV IPPE_SQUARE order: top-left, top-right, bottom-right, bottom-left.
        self.object_points = np.array(
            ((-half, half, 0.0), (half, half, 0.0),
             (half, -half, 0.0), (-half, -half, 0.0)),
            dtype=np.float64,
        )
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        parameters.cornerRefinementWinSize = 7
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self._detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    def estimate(self, image: np.ndarray) -> dict[str, Any]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None or self.tag_id not in ids[:, 0]:
            seen = [] if ids is None else ids[:, 0].tolist()
            raise RuntimeError(f"未检测到 tag36h11 ID {self.tag_id}，当前识别到 {seen}")
        matches = np.flatnonzero(ids[:, 0] == self.tag_id)
        if len(matches) != 1:
            raise RuntimeError(f"画面中出现了 {len(matches)} 个 ID {self.tag_id}")
        image_points = corners[int(matches[0])].reshape(4, 2).astype(np.float64)

        pnp_result = cv2.solvePnPGeneric(
            self.object_points,
            image_points,
            self.camera_matrix,
            self.distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not pnp_result[0]:
            raise RuntimeError("AprilTag solvePnP 失败")

        candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
        for rotation_vector, translation_vector in zip(pnp_result[1], pnp_result[2]):
            if float(translation_vector[2, 0]) <= 0.0:
                continue
            projected, _ = cv2.projectPoints(
                self.object_points,
                rotation_vector,
                translation_vector,
                self.camera_matrix,
                self.distortion,
            )
            delta = projected.reshape(4, 2) - image_points
            rms = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
            candidates.append((rms, rotation_vector, translation_vector))
        if not candidates:
            raise RuntimeError("AprilTag PnP 没有相机前方的有效解")

        reprojection_rms_px, rotation_vector, translation_vector = min(
            candidates, key=lambda candidate: candidate[0]
        )
        rotation, _ = cv2.Rodrigues(rotation_vector)
        camera_from_target = np.eye(4, dtype=np.float64)
        camera_from_target[:3, :3] = rotation
        camera_from_target[:3, 3] = translation_vector[:, 0]
        return {
            "corners_uv": image_points.tolist(),
            "center_uv": image_points.mean(axis=0).tolist(),
            "side_lengths_px": np.linalg.norm(
                np.roll(image_points, -1, axis=0) - image_points, axis=1
            ).tolist(),
            "reprojection_rms_px": reprojection_rms_px,
            "rotation_vector": rotation_vector[:, 0].tolist(),
            "translation_m": translation_vector[:, 0].tolist(),
            "T_camera_target": camera_from_target,
        }

    def annotate(self, image: np.ndarray, pose: dict[str, Any]) -> np.ndarray:
        annotated = image.copy()
        corners = np.round(pose["corners_uv"]).astype(np.int32)
        cv2.polylines(annotated, [corners], True, (0, 255, 0), 4, cv2.LINE_AA)
        rotation_vector = np.asarray(pose["rotation_vector"], dtype=np.float64).reshape(3, 1)
        translation = np.asarray(pose["translation_m"], dtype=np.float64).reshape(3, 1)
        cv2.drawFrameAxes(
            annotated,
            self.camera_matrix,
            self.distortion,
            rotation_vector,
            translation,
            self.tag_size_m * 0.5,
            4,
        )
        text = (
            f"tag36h11 ID {self.tag_id}  "
            f"z={translation[2, 0]:.3f}m  "
            f"reproj={pose['reprojection_rms_px']:.2f}px"
        )
        cv2.rectangle(annotated, (20, 20), (930, 75), (20, 20, 20), -1)
        cv2.putText(
            annotated,
            text,
            (35, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated


HAND_EYE_METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def solve_dataset(
    session_dir: Path,
    method_name: str,
    min_samples: int,
) -> dict[str, Any]:
    session_path = session_dir / "session.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    samples = session["samples"]
    if len(samples) < min_samples:
        raise RuntimeError(
            f"只有 {len(samples)} 组样本，当前要求至少 {min_samples} 组"
        )

    base_from_link = [
        np.asarray(sample["T_base_link"], dtype=np.float64) for sample in samples
    ]
    camera_from_target = [
        np.asarray(sample["T_camera_target"], dtype=np.float64) for sample in samples
    ]
    rotation_link_to_base = [transform[:3, :3] for transform in base_from_link]
    translation_link_to_base = [transform[:3, 3:4] for transform in base_from_link]
    rotation_target_to_camera = [transform[:3, :3] for transform in camera_from_target]
    translation_target_to_camera = [transform[:3, 3:4] for transform in camera_from_target]

    rotation_camera_to_link, translation_camera_to_link = cv2.calibrateHandEye(
        rotation_link_to_base,
        translation_link_to_base,
        rotation_target_to_camera,
        translation_target_to_camera,
        method=HAND_EYE_METHODS[method_name],
    )
    if not (
        np.all(np.isfinite(rotation_camera_to_link))
        and np.all(np.isfinite(translation_camera_to_link))
    ):
        raise RuntimeError("手眼标定返回了非有限值；采样姿态旋转变化不足")

    link_from_camera = np.eye(4, dtype=np.float64)
    link_from_camera[:3, :3] = rotation_camera_to_link
    link_from_camera[:3, 3] = translation_camera_to_link[:, 0]

    base_from_targets = [
        base_link @ link_from_camera @ camera_target
        for base_link, camera_target in zip(base_from_link, camera_from_target)
    ]
    target_mean = np.eye(4, dtype=np.float64)
    target_mean[:3, :3] = average_rotation(
        [transform[:3, :3] for transform in base_from_targets]
    )
    target_mean[:3, 3] = np.mean(
        [transform[:3, 3] for transform in base_from_targets], axis=0
    )
    translation_residual_mm = np.array(
        [
            np.linalg.norm(transform[:3, 3] - target_mean[:3, 3]) * 1000.0
            for transform in base_from_targets
        ]
    )
    rotation_residual_deg = np.array(
        [
            rotation_angle_deg(target_mean[:3, :3].T @ transform[:3, :3])
            for transform in base_from_targets
        ]
    )

    pair_rotation_deg: list[float] = []
    pair_translation_mm: list[float] = []
    for first in range(len(base_from_link)):
        for second in range(first + 1, len(base_from_link)):
            relative = invert_transform(base_from_link[first]) @ base_from_link[second]
            pair_rotation_deg.append(rotation_angle_deg(relative[:3, :3]))
            pair_translation_mm.append(float(np.linalg.norm(relative[:3, 3]) * 1000.0))

    reprojection = np.asarray(
        [sample["tag"]["reprojection_rms_px"] for sample in samples],
        dtype=np.float64,
    )
    mount_link = int(session["configuration"]["mount_link"])
    inverse = invert_transform(link_from_camera)
    result = {
        "schema_version": 1,
        "solved_at_utc": utc_timestamp(),
        "method": method_name,
        "sample_count": len(samples),
        "tag": session["configuration"]["tag"],
        "camera": session["camera"],
        f"T_link{mount_link}_color_camera": transform_record(
            link_from_camera,
            f"p_link{mount_link} = T_link{mount_link}_color_camera @ p_color_camera",
        ),
        f"T_color_camera_link{mount_link}": transform_record(
            inverse,
            f"p_color_camera = T_color_camera_link{mount_link} @ p_link{mount_link}",
        ),
        "estimated_T_base_target": transform_record(
            target_mean,
            "Mean stationary AprilTag pose in the PiPER base frame",
        ),
        "quality": {
            "target_translation_residual_mm": {
                "rms": float(np.sqrt(np.mean(translation_residual_mm**2))),
                "mean": float(np.mean(translation_residual_mm)),
                "max": float(np.max(translation_residual_mm)),
                "per_sample": translation_residual_mm.tolist(),
            },
            "target_rotation_residual_deg": {
                "rms": float(np.sqrt(np.mean(rotation_residual_deg**2))),
                "mean": float(np.mean(rotation_residual_deg)),
                "max": float(np.max(rotation_residual_deg)),
                "per_sample": rotation_residual_deg.tolist(),
            },
            "tag_reprojection_rms_px": {
                "mean": float(np.mean(reprojection)),
                "max": float(np.max(reprojection)),
            },
            "robot_pose_pairwise_motion": {
                "rotation_deg_min_median_max": [
                    float(np.min(pair_rotation_deg)),
                    float(np.median(pair_rotation_deg)),
                    float(np.max(pair_rotation_deg)),
                ],
                "translation_mm_min_median_max": [
                    float(np.min(pair_translation_mm)),
                    float(np.median(pair_translation_mm)),
                    float(np.max(pair_translation_mm)),
                ],
            },
        },
    }
    translation_rms_mm = result["quality"]["target_translation_residual_mm"]["rms"]
    rotation_rms_deg = result["quality"]["target_rotation_residual_deg"]["rms"]
    quality_passed = (
        translation_rms_mm <= MAX_USABLE_TRANSLATION_RMS_MM
        and rotation_rms_deg <= MAX_USABLE_ROTATION_RMS_DEG
    )
    result["quality"]["acceptance"] = {
        "passed": quality_passed,
        "translation_rms_limit_mm": MAX_USABLE_TRANSLATION_RMS_MM,
        "rotation_rms_limit_deg": MAX_USABLE_ROTATION_RMS_DEG,
        "meaning": (
            "A failed result is diagnostic only and must not be used for robot control."
        ),
    }
    output_path = session_dir / "hand_eye_calibration.json"
    save_json(output_path, result)

    translation = link_from_camera[:3, 3] * 1000.0
    rpy = rpy_degrees_from_matrix(link_from_camera[:3, :3])
    quality = result["quality"]
    print("\n标定完成")
    print(f"  方法: {method_name}, 样本: {len(samples)}")
    print(
        f"  T_link{mount_link}_camera 平移 [mm]: "
        f"{translation[0]:+.3f}, {translation[1]:+.3f}, {translation[2]:+.3f}"
    )
    print(
        "  T_link_camera RPY [deg]: "
        f"{rpy[0]:+.3f}, {rpy[1]:+.3f}, {rpy[2]:+.3f}"
    )
    print(
        "  固定标签一致性: "
        f"translation RMS={quality['target_translation_residual_mm']['rms']:.2f} mm, "
        f"rotation RMS={quality['target_rotation_residual_deg']['rms']:.3f} deg"
    )
    if quality_passed:
        print("  质量判定: PASS")
    else:
        print("  质量判定: FAIL；该外参仅供诊断，禁止用于机械臂控制")
    print(f"  输出: {output_path}")
    return result


def residual_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "rms": float(np.sqrt(np.mean(array**2))),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
        "per_sample": array.tolist(),
    }


def validate_calibration(args: argparse.Namespace) -> None:
    dataset_dir = args.dataset.resolve()
    session = json.loads(
        (dataset_dir / "session.json").read_text(encoding="utf-8")
    )
    calibration_path = dataset_dir / "hand_eye_calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not calibration["quality"]["acceptance"]["passed"]:
        raise RuntimeError("该标定结果未通过质量门槛，不能作为验证基准")
    if args.samples < 3:
        raise ValueError("验证至少需要 3 个新姿态")

    mount_link = int(session["configuration"]["mount_link"])
    dh_is_offset = int(session["configuration"]["dh_is_offset"])
    tag_config = session["configuration"]["tag"]
    link_from_camera = np.asarray(
        calibration[f"T_link{mount_link}_color_camera"]["matrix"],
        dtype=np.float64,
    )
    reference_base_from_target = np.asarray(
        calibration["estimated_T_base_target"]["matrix"], dtype=np.float64
    )

    if args.output is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        validation_dir = dataset_dir / "validation" / stamp
    else:
        validation_dir = args.output.resolve()
    validation_dir.mkdir(parents=True, exist_ok=False)
    (validation_dir / "samples").mkdir()
    (validation_dir / "rejected").mkdir()

    camera: RemoteRealSense | None = None
    arm: PiperStateReader | None = None
    validation: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": utc_timestamp(),
        "calibration_dataset": str(dataset_dir),
        "calibration_file": str(calibration_path),
        "configuration": {
            "required_samples": args.samples,
            "mount_link": mount_link,
            "dh_is_offset": dh_is_offset,
            "tag": tag_config,
            "translation_rms_limit_mm": args.translation_rms_limit_mm,
            "rotation_rms_limit_deg": args.rotation_rms_limit_deg,
        },
        f"T_link{mount_link}_color_camera": calibration[
            f"T_link{mount_link}_color_camera"
        ],
        "reference_T_base_target": calibration["estimated_T_base_target"],
        "samples": [],
    }
    output_path = validation_dir / "validation.json"

    try:
        camera = RemoteRealSense(
            host=args.camera_host,
            user=args.camera_user,
            auth=args.camera_auth,
            ssh_key=args.ssh_key,
            password_env=args.camera_password_env,
            warmup_frames=args.warmup_frames,
            frames_per_capture=args.frames_per_sample,
            stderr_path=validation_dir / "remote_camera_stderr.log",
        )
        calibrated_serial = calibration["camera"]["camera_serial"]
        live_serial = camera.camera_info["camera_serial"]
        if live_serial != calibrated_serial:
            raise RuntimeError(
                f"相机序列号不一致：标定使用 {calibrated_serial}，当前为 {live_serial}"
            )
        validation["camera"] = camera.camera_info

        arm = PiperStateReader(
            can_name=args.can_name,
            can_host=args.can_host,
            can_port=args.can_port,
            dh_is_offset=dh_is_offset,
            mount_link=mount_link,
        )
        estimator = AprilTagPoseEstimator(
            tag_id=int(tag_config["id"]),
            tag_size_mm=float(tag_config["black_square_size_mm"]),
            camera_info=camera.camera_info,
        )
        save_json(output_path, validation)

        print(f"\n验证数据目录: {validation_dir}")
        print("脚本只读取机械臂状态，不发送任何运动指令。")
        print("AprilTag 只需在本轮验证期间固定，不能移动或扶动。")
        print("旧基座标签坐标仅用于判断安装场景是否改变，不参与手眼判定。")
        print(
            f"将机械臂移到 {args.samples} 个未参与标定的新姿态；"
            "每次完全停稳后按 Enter。"
        )
        print("命令: [Enter]采集当前姿态  q保存并退出\n")

        while len(validation["samples"]) < args.samples:
            index = len(validation["samples"])
            command = input(f"验证样本 {index}/{args.samples}: ").strip().lower()
            if command == "q":
                break
            if command:
                print("请输入 Enter 或 q")
                continue

            joints_before, joint_timestamp_before = arm.read_joint_degrees()
            image, jpeg_bytes, frame_meta = camera.capture()
            joints_after, joint_timestamp_after = arm.read_joint_degrees()
            joint_motion = float(
                np.max(np.abs(np.asarray(joints_after) - np.asarray(joints_before)))
            )
            if joint_motion > args.stationary_tolerance_deg:
                print(
                    f"拒绝：抓图期间关节变化 {joint_motion:.3f} deg，"
                    f"要求 <= {args.stationary_tolerance_deg:.3f} deg"
                )
                continue

            try:
                tag_pose = estimator.estimate(image)
            except RuntimeError as exc:
                rejected_name = f"{dt.datetime.now().strftime('%H%M%S_%f')}.jpg"
                (validation_dir / "rejected" / rejected_name).write_bytes(jpeg_bytes)
                print(f"拒绝：{exc}；图像保存为 rejected/{rejected_name}")
                continue
            if tag_pose["reprojection_rms_px"] > args.max_reprojection_error_px:
                print(
                    "拒绝：AprilTag 重投影误差 "
                    f"{tag_pose['reprojection_rms_px']:.3f} px > "
                    f"{args.max_reprojection_error_px:.3f} px"
                )
                continue

            joint_degrees = (
                (np.asarray(joints_before) + np.asarray(joints_after)) * 0.5
            ).tolist()
            link_pose, base_from_link = arm.link_pose_from_joints(joint_degrees)
            camera_from_target = tag_pose.pop("T_camera_target")
            estimated_base_from_target = (
                base_from_link @ link_from_camera @ camera_from_target
            )
            translation_error_mm = float(
                np.linalg.norm(
                    estimated_base_from_target[:3, 3]
                    - reference_base_from_target[:3, 3]
                )
                * 1000.0
            )
            rotation_error_deg = rotation_angle_deg(
                reference_base_from_target[:3, :3].T
                @ estimated_base_from_target[:3, :3]
            )

            relative_dir = Path("samples") / f"{index:03d}"
            sample_dir = validation_dir / relative_dir
            sample_dir.mkdir()
            (sample_dir / "image.jpg").write_bytes(jpeg_bytes)
            annotated = estimator.annotate(image, tag_pose)
            cv2.imwrite(str(sample_dir / "annotated.jpg"), annotated)

            sample = {
                "index": index,
                "captured_at_utc": utc_timestamp(),
                "files": {
                    "image": str(relative_dir / "image.jpg"),
                    "annotated": str(relative_dir / "annotated.jpg"),
                },
                "camera_frame": frame_meta,
                "joint_feedback": {
                    "before_deg": joints_before,
                    "after_deg": joints_after,
                    "used_deg": joint_degrees,
                    "motion_during_capture_deg": joint_motion,
                    "timestamp_before_s": joint_timestamp_before,
                    "timestamp_after_s": joint_timestamp_after,
                },
                "link_pose_base_mm_deg": link_pose,
                "T_base_link": base_from_link.tolist(),
                "tag": tag_pose,
                "T_camera_target": camera_from_target.tolist(),
                "estimated_T_base_target": estimated_base_from_target.tolist(),
                "reference_error": {
                    "translation_mm": translation_error_mm,
                    "rotation_deg": rotation_error_deg,
                },
            }
            validation["samples"].append(sample)
            save_json(sample_dir / "sample.json", sample)
            save_json(output_path, validation)
            print(
                f"已保存 #{index:03d}: 旧基座参考偏移 "
                f"{translation_error_mm:.2f} mm / {rotation_error_deg:.3f} deg, "
                f"tag z={tag_pose['translation_m'][2]:.3f} m"
            )
    finally:
        if camera is not None:
            camera.close()
        if arm is not None:
            arm.close()

    samples = validation["samples"]
    if not samples:
        print(f"未采集有效验证样本；目录已保存: {validation_dir}")
        return

    estimated_targets = [
        np.asarray(sample["estimated_T_base_target"], dtype=np.float64)
        for sample in samples
    ]
    validation_mean = np.eye(4, dtype=np.float64)
    validation_mean[:3, :3] = average_rotation(
        [transform[:3, :3] for transform in estimated_targets]
    )
    validation_mean[:3, 3] = np.mean(
        [transform[:3, 3] for transform in estimated_targets], axis=0
    )
    reference_translation_errors = [
        float(sample["reference_error"]["translation_mm"]) for sample in samples
    ]
    reference_rotation_errors = [
        float(sample["reference_error"]["rotation_deg"]) for sample in samples
    ]
    internal_translation_errors = [
        float(np.linalg.norm(transform[:3, 3] - validation_mean[:3, 3]) * 1000.0)
        for transform in estimated_targets
    ]
    internal_rotation_errors = [
        rotation_angle_deg(validation_mean[:3, :3].T @ transform[:3, :3])
        for transform in estimated_targets
    ]
    reference_translation = residual_summary(reference_translation_errors)
    reference_rotation = residual_summary(reference_rotation_errors)
    internal_translation = residual_summary(internal_translation_errors)
    internal_rotation = residual_summary(internal_rotation_errors)
    completed = len(samples) == args.samples
    hand_eye_passed = (
        completed
        and internal_translation["rms"] <= args.translation_rms_limit_mm
        and internal_rotation["rms"] <= args.rotation_rms_limit_deg
    )
    calibration_scene_matches = (
        completed
        and reference_translation["rms"] <= args.translation_rms_limit_mm
        and reference_rotation["rms"] <= args.rotation_rms_limit_deg
    )
    validation["completed_at_utc"] = utc_timestamp()
    validation["sample_count"] = len(samples)
    validation["validation_T_base_target_mean"] = transform_record(
        validation_mean, "Mean AprilTag pose estimated from new validation poses"
    )
    validation["quality"] = {
        "error_against_calibration_target": {
            "translation_mm": reference_translation,
            "rotation_deg": reference_rotation,
        },
        "new_pose_internal_consistency": {
            "translation_mm": internal_translation,
            "rotation_deg": internal_rotation,
        },
        "acceptance": {
            "passed": hand_eye_passed,
            "criterion": "new_pose_internal_consistency",
            "completed_required_samples": completed,
            "translation_rms_limit_mm": args.translation_rms_limit_mm,
            "rotation_rms_limit_deg": args.rotation_rms_limit_deg,
        },
        "calibration_scene_reference_match": {
            "passed": calibration_scene_matches,
            "meaning": (
                "False means the PiPER base or AprilTag moved relative to the "
                "original calibration scene; it does not invalidate T_link_camera."
            ),
        },
    }
    save_json(output_path, validation)

    print("\n独立姿态验证完成")
    print(f"  有效样本: {len(samples)}/{args.samples}")
    print(
        "  手眼外参新姿态一致性: "
        f"translation RMS={internal_translation['rms']:.2f} mm, "
        f"max={internal_translation['max']:.2f} mm; "
        f"rotation RMS={internal_rotation['rms']:.3f} deg, "
        f"max={internal_rotation['max']:.3f} deg"
    )
    print(
        "  旧基座参考偏移: "
        f"translation RMS={reference_translation['rms']:.2f} mm, "
        f"max={reference_translation['max']:.2f} mm; "
        f"rotation RMS={reference_rotation['rms']:.3f} deg, "
        f"max={reference_rotation['max']:.3f} deg"
    )
    print(f"  手眼外参判定: {'PASS' if hand_eye_passed else 'FAIL'}")
    print(
        "  原安装场景: "
        + ("MATCH" if calibration_scene_matches else "CHANGED（底座或标签已移动）")
    )
    print(f"  输出: {output_path}")


def collect(args: argparse.Namespace) -> None:
    if args.output is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = Path(__file__).resolve().parent / "hand_eye_data" / stamp
    else:
        session_dir = args.output.resolve()
    session_dir.mkdir(parents=True, exist_ok=False)
    (session_dir / "samples").mkdir()
    (session_dir / "rejected").mkdir()

    camera: RemoteRealSense | None = None
    arm: PiperStateReader | None = None
    solve_after_collection = False
    try:
        camera = RemoteRealSense(
            host=args.camera_host,
            user=args.camera_user,
            auth=args.camera_auth,
            ssh_key=args.ssh_key,
            password_env=args.camera_password_env,
            warmup_frames=args.warmup_frames,
            frames_per_capture=args.frames_per_sample,
            stderr_path=session_dir / "remote_camera_stderr.log",
        )
        arm = PiperStateReader(
            can_name=args.can_name,
            can_host=args.can_host,
            can_port=args.can_port,
            dh_is_offset=args.dh_is_offset,
            mount_link=args.mount_link,
        )
        estimator = AprilTagPoseEstimator(
            tag_id=args.tag_id,
            tag_size_mm=args.tag_size_mm,
            camera_info=camera.camera_info,
        )
        session: dict[str, Any] = {
            "schema_version": 1,
            "created_at_utc": utc_timestamp(),
            "configuration": {
                "mode": "eye_in_hand",
                "mount_link": args.mount_link,
                "dh_is_offset": args.dh_is_offset,
                "tag": {
                    "family": "tag36h11",
                    "id": args.tag_id,
                    "black_square_size_mm": args.tag_size_mm,
                },
                "can": {
                    "name": args.can_name,
                    "host": args.can_host,
                    "port": args.can_port,
                },
            },
            "camera": camera.camera_info,
            "samples": [],
        }
        save_json(session_dir / "session.json", session)

        print(f"\n数据目录: {session_dir}")
        print("脚本只读取机械臂状态，不发送任何运动指令。")
        print("固定 AprilTag；每次将机械臂移动到新姿态并完全停稳。")
        print("建议采集 15-25 组，并让相机绕多个方向转动 15-35 度。")
        print("命令: [Enter]采集当前姿态  s求解并退出  q仅保存并退出\n")

        while True:
            command = input(f"样本 {len(session['samples'])}: ").strip().lower()
            if command == "q":
                break
            if command == "s":
                solve_after_collection = True
                break
            if command:
                print("请输入 Enter、s 或 q")
                continue

            joints_before, joint_timestamp_before = arm.read_joint_degrees()
            image, jpeg_bytes, frame_meta = camera.capture()
            joints_after, joint_timestamp_after = arm.read_joint_degrees()
            joint_motion = float(
                np.max(np.abs(np.asarray(joints_after) - np.asarray(joints_before)))
            )
            if joint_motion > args.stationary_tolerance_deg:
                print(
                    f"拒绝：抓图期间关节变化 {joint_motion:.3f} deg，"
                    f"要求 <= {args.stationary_tolerance_deg:.3f} deg"
                )
                continue

            try:
                tag_pose = estimator.estimate(image)
            except RuntimeError as exc:
                rejected_name = f"{dt.datetime.now().strftime('%H%M%S_%f')}.jpg"
                (session_dir / "rejected" / rejected_name).write_bytes(jpeg_bytes)
                print(f"拒绝：{exc}；图像保存为 rejected/{rejected_name}")
                continue
            if tag_pose["reprojection_rms_px"] > args.max_reprojection_error_px:
                print(
                    "拒绝：AprilTag 重投影误差 "
                    f"{tag_pose['reprojection_rms_px']:.3f} px > "
                    f"{args.max_reprojection_error_px:.3f} px"
                )
                continue

            joint_degrees = (
                (np.asarray(joints_before) + np.asarray(joints_after)) * 0.5
            ).tolist()
            link_pose, base_from_link = arm.link_pose_from_joints(joint_degrees)
            index = len(session["samples"])
            relative_dir = Path("samples") / f"{index:03d}"
            sample_dir = session_dir / relative_dir
            sample_dir.mkdir()
            (sample_dir / "image.jpg").write_bytes(jpeg_bytes)
            annotated = estimator.annotate(image, tag_pose)
            cv2.imwrite(str(sample_dir / "annotated.jpg"), annotated)

            camera_from_target = tag_pose.pop("T_camera_target")
            sample = {
                "index": index,
                "captured_at_utc": utc_timestamp(),
                "files": {
                    "image": str(relative_dir / "image.jpg"),
                    "annotated": str(relative_dir / "annotated.jpg"),
                },
                "camera_frame": frame_meta,
                "joint_feedback": {
                    "before_deg": joints_before,
                    "after_deg": joints_after,
                    "used_deg": joint_degrees,
                    "motion_during_capture_deg": joint_motion,
                    "timestamp_before_s": joint_timestamp_before,
                    "timestamp_after_s": joint_timestamp_after,
                },
                "link_pose_base_mm_deg": link_pose,
                "T_base_link": base_from_link.tolist(),
                "tag": tag_pose,
                "T_camera_target": camera_from_target.tolist(),
            }
            session["samples"].append(sample)
            save_json(sample_dir / "sample.json", sample)
            save_json(session_dir / "session.json", session)
            print(
                f"已保存 #{index:03d}: tag z={tag_pose['translation_m'][2]:.3f} m, "
                f"reproj={tag_pose['reprojection_rms_px']:.3f} px, "
                f"joint motion={joint_motion:.3f} deg"
            )
    finally:
        if camera is not None:
            camera.close()
        if arm is not None:
            arm.close()

    if solve_after_collection:
        solve_dataset(session_dir, args.method, args.min_samples)
    else:
        print(f"采集数据已保存: {session_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PiPER link6-mounted D435i + AprilTag eye-in-hand calibration"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser(
        "collect", help="交互采集 PiPER 姿态和 PC2 D435i 图像，结束后求解"
    )
    collect_parser.add_argument("--output", type=Path)
    collect_parser.add_argument("--tag-id", type=int, default=DEFAULT_TAG_ID)
    collect_parser.add_argument("--tag-size-mm", type=float, default=DEFAULT_TAG_SIZE_MM)
    collect_parser.add_argument(
        "--mount-link", type=int, choices=(5, 6), default=DEFAULT_MOUNT_LINK
    )
    collect_parser.add_argument("--dh-is-offset", type=int, choices=(0, 1), default=1)
    collect_parser.add_argument("--can-name", default=DEFAULT_CAN_NAME)
    collect_parser.add_argument("--can-host", default=DEFAULT_PC2_HOST)
    collect_parser.add_argument("--can-port", type=int, default=DEFAULT_CAN_PORT)
    collect_parser.add_argument("--camera-host", default=DEFAULT_PC2_HOST)
    collect_parser.add_argument("--camera-user", default=DEFAULT_PC2_USER)
    collect_parser.add_argument(
        "--camera-auth", choices=("password", "key"), default="password"
    )
    collect_parser.add_argument("--ssh-key", type=Path)
    collect_parser.add_argument(
        "--camera-password-env",
        default="A2_PC2_PASSWORD",
        help="可选的 PC2 SSH 密码环境变量名；未设置时安全交互输入",
    )
    collect_parser.add_argument("--warmup-frames", type=int, default=60)
    collect_parser.add_argument("--frames-per-sample", type=int, default=5)
    collect_parser.add_argument(
        "--stationary-tolerance-deg", type=float, default=0.10
    )
    collect_parser.add_argument(
        "--max-reprojection-error-px", type=float, default=1.0
    )
    collect_parser.add_argument("--min-samples", type=int, default=15)
    collect_parser.add_argument(
        "--method", choices=tuple(HAND_EYE_METHODS), default=DEFAULT_METHOD
    )
    collect_parser.set_defaults(handler=collect)

    solve_parser = subparsers.add_parser(
        "solve", help="使用已有 session.json 重新求解手眼外参"
    )
    solve_parser.add_argument("dataset", type=Path)
    solve_parser.add_argument("--min-samples", type=int, default=15)
    solve_parser.add_argument(
        "--method", choices=tuple(HAND_EYE_METHODS), default=DEFAULT_METHOD
    )
    solve_parser.set_defaults(
        handler=lambda args: solve_dataset(
            args.dataset.resolve(), args.method, args.min_samples
        )
    )

    validate_parser = subparsers.add_parser(
        "validate", help="使用固定标签和新机械臂姿态独立验证已有外参"
    )
    validate_parser.add_argument("dataset", type=Path)
    validate_parser.add_argument("--output", type=Path)
    validate_parser.add_argument("--samples", type=int, default=5)
    validate_parser.add_argument("--can-name", default=DEFAULT_CAN_NAME)
    validate_parser.add_argument("--can-host", default=DEFAULT_PC2_HOST)
    validate_parser.add_argument("--can-port", type=int, default=DEFAULT_CAN_PORT)
    validate_parser.add_argument("--camera-host", default=DEFAULT_PC2_HOST)
    validate_parser.add_argument("--camera-user", default=DEFAULT_PC2_USER)
    validate_parser.add_argument(
        "--camera-auth", choices=("password", "key"), default="password"
    )
    validate_parser.add_argument("--ssh-key", type=Path)
    validate_parser.add_argument(
        "--camera-password-env",
        default="A2_PC2_PASSWORD",
        help="可选的 PC2 SSH 密码环境变量名；未设置时安全交互输入",
    )
    validate_parser.add_argument("--warmup-frames", type=int, default=60)
    validate_parser.add_argument("--frames-per-sample", type=int, default=5)
    validate_parser.add_argument(
        "--stationary-tolerance-deg", type=float, default=0.10
    )
    validate_parser.add_argument(
        "--max-reprojection-error-px", type=float, default=1.0
    )
    validate_parser.add_argument(
        "--translation-rms-limit-mm", type=float, default=10.0
    )
    validate_parser.add_argument(
        "--rotation-rms-limit-deg", type=float, default=1.0
    )
    validate_parser.set_defaults(handler=validate_calibration)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断，已停止采集。", file=sys.stderr)
        raise SystemExit(130)
