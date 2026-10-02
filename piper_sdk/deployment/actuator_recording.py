"""Event-based CAN recording and summaries for actuator identification."""

from __future__ import annotations

import csv
import json
import math
import queue
import struct
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np


FRAME_HEADER = [
    "phase", "segment", "pose_index", "step", "direction", "can_id", "dlc",
    "is_extended_id", "is_remote_frame", "is_error_frame", "data_hex",
    "source_timestamp_s", "local_start_monotonic_ns", "local_end_monotonic_ns",
    "result", "error",
]
STEP_SCALARS = [
    "step", "phase", "segment", "pose_index", "joint_index",
    "command_start_monotonic_ns", "observation_end_monotonic_ns",
    "command_period_s", "step_elapsed_s", "joint_feedback_source_s",
    "status_feedback_source_s", "ctrl_mode", "move_mode", "arm_status",
]
STEP_VECTORS = [
    "action_raw", "requested_target_rad", "limited_target_rad", "sent_target_rad",
    "joint_pos_rad", "joint_vel_average_rad_s", "motor_speed_latest_rad_s",
    "motor_current_a", "motor_effort_estimate_nm", "motor_source_s",
]
STEP_HEADER = STEP_SCALARS + [f"{name}_j{j}" for name in STEP_VECTORS for j in range(1, 7)]
COLLECTION_PHASES = {"hold", "sweep", "multijoint", "policy"}


def write_json(path: Path, value: dict) -> None:
    with path.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


class FrameRecorder:
    """Copy frames in the caller, then serialize them on one writer thread."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.context = ("initialization", "initialization", -1, -1)
        self.sent_targets = [None] * 6
        self.error: Exception | None = None
        self.frames_by_direction: Counter = Counter()
        self.frames_by_phase: Counter = Counter()
        self.motor_stats: dict[int, dict] = {}
        self.pending: queue.SimpleQueue = queue.SimpleQueue()
        self.ready = threading.Event()
        self.worker = threading.Thread(target=self._write, name="actuator-data-writer")
        self.worker.start()
        self.ready.wait()
        self.check_error()

    def set_context(self, phase: str, segment: str, pose_index: int = -1, step: int = -1) -> None:
        self.context = (phase, segment, pose_index, step)

    def on_frame(self, direction, message, started_ns, finished_ns, result, error) -> None:
        data = bytes(message.data)
        self.pending.put(("frame", (
            *self.context, direction, message.arbitration_id, message.dlc,
            int(message.is_extended_id), int(message.is_remote_frame), int(message.is_error_frame),
            data.hex(), message.timestamp if direction == "rx" else "",
            started_ns, finished_ns, result, error,
        )))
        if direction == "tx":
            if result != "SEND_MESSAGE_SUCCESS":
                self.error = RuntimeError(f"CAN send 0x{message.arbitration_id:03X}: {result}: {error}")
                raise self.error
            # Decode the transmitted position, including SDK wire quantization.
            can_id = message.arbitration_id
            if 0x155 <= can_id <= 0x157:
                first = (can_id - 0x155) * 2
                self.sent_targets[first:first + 2] = [
                    value * math.pi / 180000.0 for value in struct.unpack(">ii", data)
                ]
            elif 0x15A <= can_id <= 0x15F:
                self.sent_targets[can_id - 0x15A] = int.from_bytes(data[:2], "big") * 25.0 / 65535 - 12.5
        self.check_error()

    def record_step(self, row: list) -> None:
        self.check_error()
        self.pending.put(("step", row))

    def check_error(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"Actuator recording failed: {self.error}") from self.error

    def close(self) -> None:
        self.pending.put(None)
        self.worker.join()

    def _write(self) -> None:
        try:
            with (
                (self.output_dir / "can_frames.csv").open("w", newline="") as frames,
                (self.output_dir / "steps.csv").open("w", newline="") as steps,
            ):
                writers = {"frame": csv.writer(frames), "step": csv.writer(steps)}
                writers["frame"].writerow(FRAME_HEADER)
                writers["step"].writerow(STEP_HEADER)
                self.ready.set()
                next_flush = time.monotonic() + 1.0
                while True:
                    try:
                        item = self.pending.get(timeout=1.0)
                    except queue.Empty:
                        frames.flush()
                        steps.flush()
                        continue
                    if item is None:
                        break
                    kind, row = item
                    writers[kind].writerow(row)
                    if kind == "frame":
                        self._count_frame(row)
                    if time.monotonic() >= next_flush:
                        frames.flush()
                        steps.flush()
                        next_flush = time.monotonic() + 1.0
        except Exception as exc:
            self.error = exc
        finally:
            self.ready.set()

    def _count_frame(self, row: tuple) -> None:
        phase, _, _, _, direction, can_id = row[:6]
        self.frames_by_direction[direction] += 1
        self.frames_by_phase[phase] += 1
        if direction != "rx" or not 0x251 <= can_id <= 0x256:
            return
        # PiPER high-speed frame: signed speed/current and raw motor position.
        speed, current, _ = struct.unpack(">hhi", bytes.fromhex(row[10]))
        speed *= 0.001
        current *= 0.001
        source_s, local_ns = row[11], row[12]
        joint = can_id - 0x250
        if joint not in self.motor_stats:
            self.motor_stats[joint] = {
                "count": 0, "first_source_s": source_s, "last_source_s": source_s,
                "first_local_ns": local_ns, "last_local_ns": local_ns,
                "max_local_gap_s": 0.0,
                "speed_range_rad_s": [speed, speed], "current_range_a": [current, current],
            }
        stats = self.motor_stats[joint]
        stats["max_local_gap_s"] = max(stats["max_local_gap_s"], (local_ns - stats["last_local_ns"]) * 1e-9)
        stats["count"] += 1
        stats["last_source_s"] = source_s
        stats["last_local_ns"] = local_ns
        for key, value in (("speed_range_rad_s", speed), ("current_range_a", current)):
            stats[key][0] = min(stats[key][0], value)
            stats[key][1] = max(stats[key][1], value)

    def summarize(self, outcome: str, failure: str | None) -> dict:
        summary = {
            "outcome": outcome, "failure": failure,
            "frames_by_direction": dict(self.frames_by_direction),
            "frames_by_phase": dict(self.frames_by_phase),
            "motor_feedback_all_phases": {},
        }
        for joint in range(1, 7):
            stats = self.motor_stats.get(joint)
            if stats is None:
                summary["motor_feedback_all_phases"][f"j{joint}"] = {"count": 0}
                continue
            result = dict(stats)
            local_s = (stats["last_local_ns"] - stats["first_local_ns"]) * 1e-9
            source_s = stats["last_source_s"] - stats["first_source_s"]
            result["local_arrival_hz"] = (stats["count"] - 1) / local_s if local_s > 0 else None
            result["source_hz"] = (stats["count"] - 1) / source_s if source_s > 0 else None
            summary["motor_feedback_all_phases"][f"j{joint}"] = result
        with (self.output_dir / "steps.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        summary["recorded_steps"] = len(rows)
        summary["steps_by_phase"] = dict(Counter(row["phase"] for row in rows))
        summary["observed_move_modes"] = dict(Counter(row["move_mode"] for row in rows))
        summary["segments"] = list(dict.fromkeys(row["segment"] for row in rows))
        if rows:
            starts = np.array([int(row["command_start_monotonic_ns"]) for row in rows], dtype=np.int64)
            periods = np.diff(starts) * 1e-9
            summary["command_period_s"] = (
                {"mean": float(periods.mean()), "min": float(periods.min()), "max": float(periods.max())}
                if len(periods) else None
            )
            selected = [row for row in rows if row["phase"] in COLLECTION_PHASES]
            summary["collection_steps"] = len(selected)
            summary["collection_observed_move_modes"] = dict(Counter(row["move_mode"] for row in selected))
            summary["collection_joint_metrics"] = {}
            for joint in range(1, 7):
                q = np.array([float(row[f"joint_pos_rad_j{joint}"]) for row in selected])
                target = np.array([float(row[f"sent_target_rad_j{joint}"]) for row in selected])
                if len(q):
                    summary["collection_joint_metrics"][f"j{joint}"] = {
                        "position_range_rad": [float(q.min()), float(q.max())],
                        "tracking_rmse_rad": float(np.sqrt(np.mean((target - q) ** 2))),
                        "tracking_max_abs_rad": float(np.max(np.abs(target - q))),
                    }
        return summary


def plot_response(output_dir: Path) -> None:
    """Overview of step snapshots; full-rate feedback remains in can_frames.csv."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (output_dir / "steps.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return
    rows = rows[::max(1, len(rows) // 5000)]
    origin = int(rows[0]["command_start_monotonic_ns"])
    command_times = [(int(row["command_start_monotonic_ns"]) - origin) * 1e-9 for row in rows]
    observation_times = [(int(row["observation_end_monotonic_ns"]) - origin) * 1e-9 for row in rows]
    fig, axes = plt.subplots(6, 3, figsize=(17, 16), sharex=True)
    for joint, axes_row in enumerate(axes, start=1):
        for prefix, label in (("requested_target_rad", "requested"), ("sent_target_rad", "sent"), ("joint_pos_rad", "feedback")):
            times = observation_times if prefix == "joint_pos_rad" else command_times
            axes_row[0].plot(times, [float(row[f"{prefix}_j{joint}"]) for row in rows], label=label, linewidth=0.7)
        for prefix, label in (("motor_speed_latest_rad_s", "latest"), ("joint_vel_average_rad_s", "observation average")):
            axes_row[1].plot(observation_times, [float(row[f"{prefix}_j{joint}"]) for row in rows], label=label, linewidth=0.7)
        axes_row[2].plot(observation_times, [float(row[f"motor_current_a_j{joint}"]) for row in rows], linewidth=0.7)
        for ax, units in zip(axes_row, ("position (rad)", "velocity (rad/s)", "current (A)")):
            ax.set_ylabel(f"J{joint} {units}")
            ax.grid(alpha=0.2)
    axes[0, 0].legend()
    axes[0, 1].legend()
    for ax in axes[-1]:
        ax.set_xlabel("Local monotonic time (s)")
    fig.tight_layout()
    fig.savefig(output_dir / "response.png", dpi=120)
    plt.close(fig)
