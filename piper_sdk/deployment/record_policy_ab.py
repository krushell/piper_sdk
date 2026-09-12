"""Record and compare Piper joint feedback before and after a code change.

``before`` loads ``piper_sdk/deployment/manipulation.py`` directly from a Git
revision (``HEAD`` by default). ``after`` imports the current working-tree
module. Git files are only read; this script never checks out or rewrites the
working tree.

The recorder samples joint positions, motor velocities, finite-difference
velocities, and arm mode/status in a background thread. The policy control
loop itself is left unchanged. By default both variants stop immediately after
the requested policy steps so the comparison isolates policy-time behavior.
The current variant can additionally record the new return-to-zero transition
with ``--include-return``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence, TextIO


JOINT_COUNT = 6
DEFAULT_SAMPLE_HZ = 200.0
DEFAULT_POLICY_STEPS = 500
DEFAULT_COMPARE_WINDOW_S = 2.0
MANIPULATION_REPO_PATH = "piper_sdk/deployment/manipulation.py"


def _csv_header() -> list[str]:
    header = [
        "sample_index",
        "elapsed_s",
        "sample_dt_s",
        "wall_time_s",
        "label",
        "variant",
        "source_ref",
        "source_commit",
        "source_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "target_pos_b_x",
        "target_pos_b_y",
        "target_pos_b_z",
        "policy_steps_requested",
        "sample_hz_requested",
        "phase",
        "policy_step",
        "joint_feedback_timestamp_s",
        "joint_feedback_age_s",
        "joint_feedback_hz",
        "motor_feedback_hz",
        "status_feedback_timestamp_s",
        "status_feedback_age_s",
        "status_feedback_hz",
        "status_arm_status",
        "status_ctrl_mode",
        "status_move_mode",
        "status_motion_status",
        "status_err_code",
        "max_abs_motor_velocity_rad_s",
        "max_abs_motor_velocity_deg_s",
        "max_velocity_joint",
    ]
    for prefix in (
        "joint_pos_deg",
        "joint_pos_rad",
        "joint_vel_motor_rad_s",
        "joint_vel_motor_deg_s",
        "joint_vel_fd_rad_s",
        "joint_vel_fd_deg_s",
        "motor_timestamp_s",
        "motor_age_s",
    ):
        header.extend(
            f"{prefix}_j{joint}" for joint in range(1, JOINT_COUNT + 1)
        )
    return header


@dataclass(frozen=True)
class ControllerSpec:
    controller_class: type[Any]
    default_target_pos_b: tuple[float, float, float]
    source_ref: str
    source_commit: str
    source_sha256: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_git(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=_repo_root(),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise RuntimeError(f"Git command failed: git {' '.join(args)}: {detail}") from exc
    return completed.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_controller(variant: str, baseline_ref: str) -> ControllerSpec:
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if variant == "after":
        module = importlib.import_module("piper_sdk.deployment.manipulation")
        source_path = Path(module.__file__).resolve()
        expected_path = (repo_root / MANIPULATION_REPO_PATH).resolve()
        if source_path != expected_path:
            raise RuntimeError(
                "after 控制器没有从当前工作树加载："
                f"expected={expected_path}, actual={source_path}"
            )
        source = source_path.read_bytes()
        return ControllerSpec(
            controller_class=module.Manipulation,
            default_target_pos_b=tuple(module.DEFAULT_TARGET_POS_B),
            source_ref=f"working-tree:{source_path}",
            source_commit=_run_git("rev-parse", "HEAD^{commit}"),
            source_sha256=hashlib.sha256(source).hexdigest(),
        )

    commit = _run_git("rev-parse", f"{baseline_ref}^{{commit}}")
    source_text = _run_git(
        "show",
        f"{baseline_ref}:{MANIPULATION_REPO_PATH}",
    )
    source_bytes = source_text.encode("utf-8")
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    module_name = f"piper_sdk.deployment._manipulation_before_{source_hash[:12]}"
    module = ModuleType(module_name)
    module.__file__ = f"{commit}:{MANIPULATION_REPO_PATH}"
    module.__package__ = "piper_sdk.deployment"
    sys.modules[module_name] = module
    exec(compile(source_text, module.__file__, "exec"), module.__dict__)
    return ControllerSpec(
        controller_class=module.Manipulation,
        default_target_pos_b=tuple(module.DEFAULT_TARGET_POS_B),
        source_ref=f"{baseline_ref}:{MANIPULATION_REPO_PATH}",
        source_commit=commit,
        source_sha256=source_hash,
    )


class PolicyFeedbackRecorder:
    """Read-only, fixed-rate sampler for SDK feedback aggregates."""

    def __init__(
        self,
        *,
        piper: Any,
        output_file: TextIO,
        label: str,
        variant: str,
        controller: ControllerSpec,
        sample_hz: float,
        checkpoint_path: Path,
        checkpoint_sha256: str,
        target_pos_b: Sequence[float],
        policy_steps: int,
    ) -> None:
        self.piper = piper
        self.output_file = output_file
        self.writer = csv.DictWriter(
            output_file,
            fieldnames=_csv_header(),
            extrasaction="raise",
        )
        self.writer.writeheader()
        self.output_file.flush()

        self.label = label
        self.variant = variant
        self.controller = controller
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = checkpoint_sha256
        self.target_pos_b = tuple(float(value) for value in target_pos_b)
        self.policy_steps = int(policy_steps)
        self.sample_hz = float(sample_hz)
        self.sample_period_s = 1.0 / sample_hz
        self._state_lock = threading.Lock()
        self._phase = "policy"
        self._policy_step = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._start_monotonic_s = math.nan
        self._previous_sample_s: float | None = None
        self._previous_joint_timestamp_s: float | None = None
        self._previous_joint_pos_rad: tuple[float, ...] | None = None
        self.rows_written = 0

    @property
    def error(self) -> BaseException | None:
        return self._error

    def set_phase(self, phase: str) -> None:
        with self._state_lock:
            self._phase = phase

    def set_policy_step(self, step: int) -> None:
        with self._state_lock:
            self._policy_step = int(step)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._start_monotonic_s = time.monotonic()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="piper-policy-ab-recorder",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.output_file.flush()
        os.fsync(self.output_file.fileno())

    def ensure_healthy(self) -> None:
        if self._error is not None:
            raise RuntimeError(
                "后台关节反馈记录已停止；为避免无记录运动，停止策略控制："
                f"{self._error}"
            ) from self._error

    def _joint_snapshot(self) -> tuple[float, float, tuple[float, ...]]:
        snapshot = (0.0, 0.0, (math.nan,) * JOINT_COUNT)
        for _ in range(3):
            joint_msg = self.piper.GetArmJointMsgs()
            timestamp_before = float(joint_msg.time_stamp)
            joint_state = joint_msg.joint_state
            positions_deg = tuple(
                float(getattr(joint_state, f"joint_{index}")) * 0.001
                for index in range(1, JOINT_COUNT + 1)
            )
            snapshot = (timestamp_before, float(joint_msg.Hz), positions_deg)
            if timestamp_before == float(joint_msg.time_stamp):
                break
        return snapshot

    def _status_snapshot(self) -> tuple[float, float, int, int, int, int, int]:
        snapshot = (0.0, 0.0, -1, -1, -1, -1, -1)
        for _ in range(3):
            status_msg = self.piper.GetArmStatus()
            timestamp_before = float(status_msg.time_stamp)
            status = status_msg.arm_status
            snapshot = (
                timestamp_before,
                float(status_msg.Hz),
                int(status.arm_status),
                int(status.ctrl_mode),
                int(status.mode_feed),
                int(status.motion_status),
                int(status.err_code),
            )
            if timestamp_before == float(status_msg.time_stamp):
                break
        return snapshot

    def _write_sample(self, sample_monotonic_s: float) -> None:
        wall_time_s = time.time()
        joint_timestamp_s, joint_hz, joint_pos_deg = self._joint_snapshot()
        joint_pos_rad = tuple(math.radians(value) for value in joint_pos_deg)
        motor_feedback = self.piper.GetArmHighSpdInfoSnapshot()
        motors = tuple(
            getattr(motor_feedback, f"motor_{index}")
            for index in range(1, JOINT_COUNT + 1)
        )
        motor_velocity_rad_s = tuple(
            float(motor.motor_speed) * 0.001 for motor in motors
        )
        motor_velocity_deg_s = tuple(
            math.degrees(value) for value in motor_velocity_rad_s
        )
        motor_timestamp_s = tuple(float(motor.time_stamp) for motor in motors)
        motor_age_s = tuple(
            wall_time_s - timestamp for timestamp in motor_timestamp_s
        )
        (
            status_timestamp_s,
            status_hz,
            status_arm_status,
            status_ctrl_mode,
            status_move_mode,
            status_motion_status,
            status_err_code,
        ) = self._status_snapshot()

        joint_velocity_fd_rad_s = (math.nan,) * JOINT_COUNT
        previous_timestamp = self._previous_joint_timestamp_s
        previous_joint_pos = self._previous_joint_pos_rad
        if (
            previous_timestamp is not None
            and previous_joint_pos is not None
            and joint_timestamp_s > previous_timestamp
        ):
            feedback_dt_s = joint_timestamp_s - previous_timestamp
            joint_velocity_fd_rad_s = tuple(
                (current - previous) / feedback_dt_s
                for current, previous in zip(joint_pos_rad, previous_joint_pos)
            )
        if previous_timestamp is None or joint_timestamp_s > previous_timestamp:
            self._previous_joint_timestamp_s = joint_timestamp_s
            self._previous_joint_pos_rad = joint_pos_rad
        joint_velocity_fd_deg_s = tuple(
            math.degrees(value) for value in joint_velocity_fd_rad_s
        )

        with self._state_lock:
            phase = self._phase
            policy_step = self._policy_step

        elapsed_s = sample_monotonic_s - self._start_monotonic_s
        sample_dt_s = (
            math.nan
            if self._previous_sample_s is None
            else sample_monotonic_s - self._previous_sample_s
        )
        self._previous_sample_s = sample_monotonic_s
        max_velocity_index = max(
            range(JOINT_COUNT),
            key=lambda index: abs(motor_velocity_rad_s[index]),
        )

        row: dict[str, float | int | str] = {
            "sample_index": self.rows_written + 1,
            "elapsed_s": elapsed_s,
            "sample_dt_s": sample_dt_s,
            "wall_time_s": wall_time_s,
            "label": self.label,
            "variant": self.variant,
            "source_ref": self.controller.source_ref,
            "source_commit": self.controller.source_commit,
            "source_sha256": self.controller.source_sha256,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "target_pos_b_x": self.target_pos_b[0],
            "target_pos_b_y": self.target_pos_b[1],
            "target_pos_b_z": self.target_pos_b[2],
            "policy_steps_requested": self.policy_steps,
            "sample_hz_requested": self.sample_hz,
            "phase": phase,
            "policy_step": policy_step,
            "joint_feedback_timestamp_s": joint_timestamp_s,
            "joint_feedback_age_s": wall_time_s - joint_timestamp_s,
            "joint_feedback_hz": joint_hz,
            "motor_feedback_hz": float(motor_feedback.Hz),
            "status_feedback_timestamp_s": status_timestamp_s,
            "status_feedback_age_s": wall_time_s - status_timestamp_s,
            "status_feedback_hz": status_hz,
            "status_arm_status": status_arm_status,
            "status_ctrl_mode": status_ctrl_mode,
            "status_move_mode": status_move_mode,
            "status_motion_status": status_motion_status,
            "status_err_code": status_err_code,
            "max_abs_motor_velocity_rad_s": abs(
                motor_velocity_rad_s[max_velocity_index]
            ),
            "max_abs_motor_velocity_deg_s": abs(
                motor_velocity_deg_s[max_velocity_index]
            ),
            "max_velocity_joint": max_velocity_index + 1,
        }
        per_joint_values = {
            "joint_pos_deg": joint_pos_deg,
            "joint_pos_rad": joint_pos_rad,
            "joint_vel_motor_rad_s": motor_velocity_rad_s,
            "joint_vel_motor_deg_s": motor_velocity_deg_s,
            "joint_vel_fd_rad_s": joint_velocity_fd_rad_s,
            "joint_vel_fd_deg_s": joint_velocity_fd_deg_s,
            "motor_timestamp_s": motor_timestamp_s,
            "motor_age_s": motor_age_s,
        }
        for prefix, values in per_joint_values.items():
            for joint, value in enumerate(values, start=1):
                row[f"{prefix}_j{joint}"] = value

        self.writer.writerow(row)
        self.rows_written += 1
        if self.rows_written % 100 == 0:
            self.output_file.flush()

    def _sample_loop(self) -> None:
        next_sample_s = time.monotonic()
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                if now < next_sample_s:
                    self._stop.wait(next_sample_s - now)
                    continue
                self._write_sample(now)
                next_sample_s += self.sample_period_s
                if next_sample_s <= now:
                    next_sample_s = now + self.sample_period_s
        except BaseException as exc:
            self._error = exc


def _default_record_output(variant: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return Path("logs") / f"policy_{variant}_{timestamp}.csv"


def _record(args: argparse.Namespace) -> None:
    controller = _load_controller(args.variant, args.baseline_ref)
    target_pos_b: Sequence[float] = (
        args.target_pos_b
        if args.target_pos_b is not None
        else controller.default_target_pos_b
    )
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else _default_record_output(args.variant).resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_sha256 = _sha256_file(args.checkpoint_path)

    arm: Any | None = None
    recorder: PolicyFeedbackRecorder | None = None
    interrupted = False
    with output_path.open("x", newline="", encoding="utf-8") as output_file:
        try:
            arm = controller.controller_class(
                checkpoint_path=args.checkpoint_path,
                device=args.device,
                can_name=args.can_name,
                target_pos_b=target_pos_b,
            )
            recorder = PolicyFeedbackRecorder(
                piper=arm.piper,
                output_file=output_file,
                label=args.label or args.variant,
                variant=args.variant,
                controller=controller,
                sample_hz=args.sample_hz,
                checkpoint_path=args.checkpoint_path,
                checkpoint_sha256=checkpoint_sha256,
                target_pos_b=target_pos_b,
                policy_steps=args.policy_steps,
            )
            recorder.start()
            for step in range(1, args.policy_steps + 1):
                recorder.ensure_healthy()
                action = arm.arm_policy.get_action(arm.arm_history_obs_buf)
                arm.step(action)
                recorder.set_policy_step(step)
            recorder.ensure_healthy()
            recorder.set_phase("policy_complete")
            arm.print_target_status()
            if args.include_return:
                recorder.set_phase("return_zero")
                arm.move_j_to_zero()
        except KeyboardInterrupt:
            interrupted = True
            print("用户中断 A/B 策略记录", file=sys.stderr)
        finally:
            if recorder is not None:
                recorder.set_phase("quick_stop")
            if arm is not None and arm.control_started:
                arm.quick_stop()
            if recorder is not None:
                if args.post_stop_seconds > 0.0:
                    time.sleep(args.post_stop_seconds)
                recorder.stop()

    rows_written = recorder.rows_written if recorder is not None else 0
    print(f"记录完成: {output_path}")
    print(
        f"variant={args.variant}, rows={rows_written}, "
        f"source={controller.source_ref}, sha256={controller.source_sha256[:16]}"
    )
    if recorder is not None and recorder.error is not None and not interrupted:
        raise RuntimeError(f"后台反馈记录失败: {recorder.error}") from recorder.error
    if interrupted:
        raise SystemExit(130)


def _finite_values(rows: Sequence[dict[str, str]], field: str) -> list[float]:
    values = []
    for row in rows:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _rms(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    return math.sqrt(sum(value * value for value in values) / len(values))


def _percentile_abs(values: Sequence[float], percentile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(abs(value) for value in values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _ratio(after: float, before: float) -> float:
    if not math.isfinite(after) or not math.isfinite(before):
        return math.nan
    if abs(before) < 1e-12:
        return math.inf if abs(after) >= 1e-12 else 1.0
    return after / before


def _load_terminal_policy_window(
    path: Path,
    window_s: float,
) -> tuple[list[dict[str, str]], dict[str, str], int]:
    with path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    policy_rows = [row for row in rows if row.get("phase") == "policy"]
    if not policy_rows:
        raise ValueError(f"CSV 没有 policy 阶段样本: {path}")
    elapsed = _finite_values(policy_rows, "elapsed_s")
    if not elapsed:
        raise ValueError(f"CSV 的 policy elapsed_s 无有效值: {path}")
    window_start = max(elapsed) - window_s
    selected = [
        row
        for row in policy_rows
        if math.isfinite(float(row["elapsed_s"]))
        and float(row["elapsed_s"]) >= window_start
    ]
    if not selected:
        raise ValueError(f"CSV 的末端时间窗没有样本: {path}")
    return selected, policy_rows[0], len(policy_rows)


def _joint_metrics(rows: Sequence[dict[str, str]], joint: int) -> dict[str, float]:
    positions = _finite_values(rows, f"joint_pos_deg_j{joint}")
    motor_velocity = _finite_values(rows, f"joint_vel_motor_deg_s_j{joint}")
    fd_velocity = _finite_values(rows, f"joint_vel_fd_deg_s_j{joint}")
    return {
        "position_p2p_deg": (
            max(positions) - min(positions) if positions else math.nan
        ),
        "motor_velocity_rms_deg_s": _rms(motor_velocity),
        "motor_velocity_p95_deg_s": _percentile_abs(motor_velocity, 0.95),
        "motor_velocity_peak_deg_s": (
            max((abs(value) for value in motor_velocity), default=math.nan)
        ),
        "fd_velocity_rms_deg_s": _rms(fd_velocity),
    }


def _format_metric(value: float) -> str:
    if math.isinf(value):
        return "inf"
    if not math.isfinite(value):
        return "nan"
    return f"{value:.3f}"


def _compare(args: argparse.Namespace) -> None:
    before_rows, before_meta, before_total = _load_terminal_policy_window(
        args.before, args.window_seconds
    )
    after_rows, after_meta, after_total = _load_terminal_policy_window(
        args.after, args.window_seconds
    )
    comparison_fields = (
        "checkpoint_sha256",
        "target_pos_b_x",
        "target_pos_b_y",
        "target_pos_b_z",
        "policy_steps_requested",
        "sample_hz_requested",
    )
    mismatches = [
        field
        for field in comparison_fields
        if before_meta.get(field) != after_meta.get(field)
    ]
    if mismatches:
        raise ValueError(
            "A/B 运行配置不一致，拒绝比较字段：" + ", ".join(mismatches)
        )
    if before_meta.get("source_sha256") == after_meta.get("source_sha256"):
        print("警告：before/after 的控制器源码 SHA-256 相同", file=sys.stderr)
    print(
        f"before: {before_meta.get('source_ref')} "
        f"sha256={before_meta.get('source_sha256', '')[:16]}, "
        f"window_samples={len(before_rows)}/{before_total}"
    )
    print(
        f"after : {after_meta.get('source_ref')} "
        f"sha256={after_meta.get('source_sha256', '')[:16]}, "
        f"window_samples={len(after_rows)}/{after_total}"
    )
    print(f"策略末端比较窗口: {args.window_seconds:.3f} s")
    print(
        "joint  position_p2p_deg(B/A)  motor_rms_deg_s(B/A)  "
        "motor_p95_deg_s(B/A)  motor_peak_deg_s(B/A)  rms_ratio"
    )
    for joint in range(1, JOINT_COUNT + 1):
        before = _joint_metrics(before_rows, joint)
        after = _joint_metrics(after_rows, joint)
        ratio = _ratio(
            after["motor_velocity_rms_deg_s"],
            before["motor_velocity_rms_deg_s"],
        )
        print(
            f"J{joint:<5}"
            f"{_format_metric(before['position_p2p_deg'])}/"
            f"{_format_metric(after['position_p2p_deg']):<10}  "
            f"{_format_metric(before['motor_velocity_rms_deg_s'])}/"
            f"{_format_metric(after['motor_velocity_rms_deg_s']):<10}  "
            f"{_format_metric(before['motor_velocity_p95_deg_s'])}/"
            f"{_format_metric(after['motor_velocity_p95_deg_s']):<10}  "
            f"{_format_metric(before['motor_velocity_peak_deg_s'])}/"
            f"{_format_metric(after['motor_velocity_peak_deg_s']):<10}  "
            f"{_format_metric(ratio)}x"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record and compare Piper policy-time joint vibration."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record", help="Run one real-arm recording.")
    record.add_argument("--variant", choices=("before", "after"), required=True)
    record.add_argument(
        "--baseline-ref",
        default="HEAD",
        help="Git revision used by --variant before (default: HEAD).",
    )
    record.add_argument("--label", default=None, help="Free-form run label.")
    record.add_argument("--can-name", default="can0")
    record.add_argument("--checkpoint-path", type=Path, required=True)
    record.add_argument("--device", default="cpu")
    record.add_argument(
        "--target-pos-b",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Use the same fixed target for both A/B runs.",
    )
    record.add_argument(
        "--policy-steps", type=int, default=DEFAULT_POLICY_STEPS
    )
    record.add_argument(
        "--sample-hz", type=float, default=DEFAULT_SAMPLE_HZ
    )
    record.add_argument("--output", type=Path, default=None)
    record.add_argument(
        "--include-return",
        action="store_true",
        help="After variant only: also record the new MOVE J return-to-zero phase.",
    )
    record.add_argument(
        "--post-stop-seconds",
        type=float,
        default=0.1,
        help="Feedback duration retained after quick stop (default: 0.1 s).",
    )
    record.add_argument(
        "--run-policy",
        action="store_true",
        help="Required confirmation that commands may be sent to the real arm.",
    )

    compare = subparsers.add_parser(
        "compare", help="Compare the terminal policy window of two CSV files."
    )
    compare.add_argument("--before", type=Path, required=True)
    compare.add_argument("--after", type=Path, required=True)
    compare.add_argument(
        "--window-seconds",
        type=float,
        default=DEFAULT_COMPARE_WINDOW_S,
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.command == "compare":
        args.before = args.before.expanduser().resolve()
        args.after = args.after.expanduser().resolve()
        if not args.before.is_file():
            parser.error(f"--before does not exist: {args.before}")
        if not args.after.is_file():
            parser.error(f"--after does not exist: {args.after}")
        if not math.isfinite(args.window_seconds) or args.window_seconds <= 0.0:
            parser.error("--window-seconds must be finite and positive")
        return

    if not args.run_policy:
        parser.error("record controls the real arm; pass --run-policy to confirm")
    args.checkpoint_path = args.checkpoint_path.expanduser().resolve()
    if not args.checkpoint_path.is_file():
        parser.error(f"--checkpoint-path does not exist: {args.checkpoint_path}")
    if args.policy_steps < 1:
        parser.error("--policy-steps must be positive")
    if not math.isfinite(args.sample_hz) or not 10.0 <= args.sample_hz <= 1000.0:
        parser.error("--sample-hz must be finite and within [10, 1000]")
    if (
        not math.isfinite(args.post_stop_seconds)
        or not 0.0 <= args.post_stop_seconds <= 2.0
    ):
        parser.error("--post-stop-seconds must be within [0, 2]")
    if args.include_return and args.variant != "after":
        parser.error("--include-return is only valid with --variant after")
    if args.output is not None:
        args.output = args.output.expanduser().resolve()
        if args.output.exists():
            parser.error(f"--output already exists: {args.output}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    if args.command == "record":
        _record(args)
    else:
        _compare(args)


if __name__ == "__main__":
    main()
