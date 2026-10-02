"""Prepare causal 200 Hz actuator samples from completed MOVE_J policy batches."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import struct
from collections import Counter
from pathlib import Path

import numpy as np

from piper_sdk.deployment.actuator_recording import FRAME_HEADER, write_json


PERIOD_NS = 5_000_000
INPUT_IDX = (0, 1, 2, 4, 8, 16, 32, 64)
SDK_TORQUE_COEFFICIENTS = np.array([1.18125] * 3 + [0.95844] * 3)
TORQUE_FIRMWARE_SOURCE = "https://github.com/agilexrobotics/piper_sdk/blob/master/asserts/Q%26A.MD#32-sdk-to-obtain-motor-feedback-torque"
MOTION_WINDOW = 20
MOTION_RATE_RAD_S = 0.02
COMMAND_IDS = range(0x155, 0x158)
POSITION_IDS = range(0x2A5, 0x2A8)
MOTOR_IDS = range(0x251, 0x257)


def torque_coefficients_for_firmware(firmware: str) -> np.ndarray:
    version = re.fullmatch(r"S-V(\d+)\.(\d+)-(\d+)", firmware)
    if version is None:
        raise ValueError(f"Unrecognized firmware version for torque conversion: {firmware}")
    coefficients = SDK_TORQUE_COEFFICIENTS.copy()
    if tuple(map(int, version.groups())) <= (1, 8, 2):
        coefficients[:3] *= 4
    return coefficients


def validate_torque_coefficients(manifest: dict) -> None:
    expected = torque_coefficients_for_firmware(manifest["firmware"])
    if not np.array_equal(manifest["torque_coefficients_nm_per_a"], expected):
        raise ValueError("Dataset torque coefficients do not match its firmware; regenerate it from raw CAN records")


def read_policy_frames(path: Path) -> tuple[dict, Counter]:
    # Stream rows are (local time ns, source time s, values...). TX source time
    # is unused: its timestamp is the end of a successful host send call.
    streams = {can_id: [] for can_id in (*COMMAND_IDS, *POSITION_IDS, *MOTOR_IDS)}
    phases = Counter()
    with path.open() as stream:
        reader = csv.reader(stream)
        if next(reader) != FRAME_HEADER:
            raise ValueError(f"Unexpected CAN columns: {path}")
        for row in reader:
            phases[row[0]] += 1
            if row[0] != "policy":
                continue
            can_id = int(row[5])
            if row[4] == "tx" and can_id in COMMAND_IDS:
                if row[14] != "SEND_MESSAGE_SUCCESS":
                    raise ValueError(f"Failed policy send in {path}: {row}")
                values = [v * math.pi / 180000 for v in struct.unpack(">ii", bytes.fromhex(row[10]))]
                streams[can_id].append((int(row[13]), 0.0, *values))
            elif row[4] == "rx" and can_id in POSITION_IDS:
                values = [v * math.pi / 180000 for v in struct.unpack(">ii", bytes.fromhex(row[10]))]
                streams[can_id].append((int(row[12]), float(row[11]), *values))
            elif row[4] == "rx" and can_id in MOTOR_IDS:
                speed, current, _ = struct.unpack(">hhi", bytes.fromhex(row[10]))
                streams[can_id].append((int(row[12]), float(row[11]), speed * .001, current * .001))
    arrays = {}
    for can_id, rows in streams.items():
        if not rows:
            raise ValueError(f"Missing policy CAN stream 0x{can_id:X}: {path}")
        a = np.asarray(rows, dtype=np.float64)
        if np.any(np.diff(a[:, 0]) <= 0):
            raise ValueError(f"Non-increasing local timestamps for 0x{can_id:X}: {path}")
        arrays[can_id] = a
    return arrays, phases


def prepare_trajectory(folder: Path, destination: Path, duration_s: float, torque_coefficients: np.ndarray) -> dict:
    streams, phases = read_policy_frames(folder / "can_frames.csv")
    start = max(int(a[0, 0]) for a in streams.values())
    end = min(int(a[-1, 0]) for a in streams.values())
    times = np.arange(start, end + 1, PERIOD_NS, dtype=np.int64)
    available_grid_samples = len(times)
    times = times[times < start + round(duration_s * 1e9)]
    history = max(INPUT_IDX)
    if len(times) <= history:
        raise ValueError(f"Insufficient common policy history: {folder}")
    sampled = {}
    for can_id, a in streams.items():
        # side=right includes a frame available exactly at the grid timestamp.
        indices = np.searchsorted(a[:, 0].astype(np.int64), times, side="right") - 1
        sampled[can_id] = a[indices]

    command = np.concatenate([sampled[c][:, 2:] for c in COMMAND_IDS], axis=1)
    position = np.concatenate([sampled[c][:, 2:] for c in POSITION_IDS], axis=1)
    velocity = np.column_stack([sampled[c][:, 2] for c in MOTOR_IDS])
    current = np.column_stack([sampled[c][:, 3] for c in MOTOR_IDS])
    error = command - position
    sample_indices = np.arange(history, len(times))
    history_indices = sample_indices[:, None] - np.asarray(INPUT_IDX)[None, :]
    features = np.concatenate((error[history_indices].transpose(0, 2, 1),
                               velocity[history_indices].transpose(0, 2, 1)), axis=2)
    motion = np.maximum(
        np.abs(position[sample_indices] - position[sample_indices - MOTION_WINDOW]),
        np.abs(command[sample_indices] - command[sample_indices - MOTION_WINDOW]),
    ) / (MOTION_WINDOW * PERIOD_NS * 1e-9) > MOTION_RATE_RAD_S

    def frame_times(ids, column, pair=False):
        a = np.column_stack([sampled[c][:, column] for c in ids])
        return np.repeat(a, 2, axis=1) if pair else a

    tx_times = frame_times(COMMAND_IDS, 0, pair=True).astype(np.int64)
    q_times = frame_times(POSITION_IDS, 0, pair=True).astype(np.int64)
    motor_times = frame_times(MOTOR_IDS, 0).astype(np.int64)
    arrays = {
        "time_monotonic_ns": times[history:],
        "inputs": features.astype(np.float32),
        "warmup_time_monotonic_ns": times[:history],
        "warmup_inputs": np.stack((error[:history], velocity[:history]), axis=-1).astype(np.float32),
        "warmup_joint_pos_rad": position[:history].astype(np.float32),
        "sent_target_rad": command[history:].astype(np.float32),
        "joint_pos_rad": position[history:].astype(np.float32),
        "joint_vel_rad_s": velocity[history:].astype(np.float32),
        "current_a": current[history:].astype(np.float32),
        "torque_est_nm": (current[history:] * torque_coefficients).astype(np.float32),
        "motion": motion,
        "command_send_end_monotonic_ns": tx_times[history:],
        "position_receive_monotonic_ns": q_times[history:],
        "motor_receive_monotonic_ns": motor_times[history:],
        "position_source_s": frame_times(POSITION_IDS, 1, pair=True)[history:],
        "motor_source_s": frame_times(MOTOR_IDS, 1)[history:],
    }
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError(f"Non-finite prepared samples: {folder}")
    np.savez_compressed(destination, **arrays)
    return {
        "frames_by_phase": dict(phases),
        "raw_policy_frames_by_id": {f"0x{c:X}": len(a) for c, a in streams.items()},
        "common_start_monotonic_ns": start, "common_end_monotonic_ns": end,
        "discarded_boundary_start_ns": start - min(int(a[0, 0]) for a in streams.values()),
        "discarded_boundary_end_ns": max(int(a[-1, 0]) for a in streams.values()) - end,
        "grid_samples_before_history": len(times), "discarded_history_samples": history,
        "available_grid_samples": available_grid_samples,
        "discarded_tail_samples": available_grid_samples - len(times),
        "retained_grid_end_monotonic_ns": int(times[-1]),
        "warmup_samples": history,
        "samples": len(sample_indices), "motion_samples_by_joint": motion.sum(axis=0).tolist(),
        "hold_samples_by_joint": (~motion).sum(axis=0).tolist(),
        "maximum_command_age_ms": float(((times[:, None] - tx_times) * 1e-6).max()),
        "maximum_position_age_ms": float(((times[:, None] - q_times) * 1e-6).max()),
        "maximum_motor_age_ms": float(((times[:, None] - motor_times) * 1e-6).max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, nargs="+", required=True,
                        help="Completed raw batches; preserve each batch's trajectory splits")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--duration_s", type=float, default=8.0,
                        help="Keep this prefix of each aligned policy trajectory, including the 320 ms warmup")
    args = parser.parse_args()
    if not math.isfinite(args.duration_s) or args.duration_s <= max(INPUT_IDX) * PERIOD_NS * 1e-9:
        parser.error("duration_s must exceed the 320 ms history/warmup interval")
    sources = [path.expanduser().resolve() for path in args.input_dir]
    destination = args.output_dir.expanduser().resolve()
    source_batches, trajectories = [], []
    for source_index, source in enumerate(sources):
        batch = json.loads((source / "batch.json").read_text())
        if batch["outcome"] != "complete" or batch["mode"] != "move_j" or batch["source"] != "policy":
            raise ValueError(f"Expected a completed MOVE_J policy batch: {source}")
        with (source / "targets.csv").open() as stream:
            targets = list(csv.DictReader(stream))
        batch_id = f"batch_{source_index:03d}"
        source_batches.append({"batch_id": batch_id, "source_dir": str(source),
                               "seed": batch["seed"], "policy_steps": batch["policy_steps"],
                               "checkpoint_path": batch["checkpoint_path"],
                               "split_counts": dict(Counter(t["split"] for t in targets))})
        trajectories.extend((batch_id, source, target) for target in targets)
    _, first_source, first_target = trajectories[0]
    first_metadata = json.loads((first_source / first_target["trajectory"] / "metadata.json").read_text())
    torque_coefficients = torque_coefficients_for_firmware(first_metadata["firmware"])
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "trajectories").mkdir()
    manifest = {
        "outcome": "preparing", "source_batches": source_batches, "mode": "move_j", "phase": "policy",
        "sample_period_s": PERIOD_NS * 1e-9, "input_idx": list(INPUT_IDX), "input_order": "pos_vel",
        "lstm_input_names": ["sent_target_minus_position_rad", "joint_velocity_rad_s"],
        "joint_lstm_input_names": [f"{name}_j{joint}" for name in
                                   ("position_error_rad", "joint_position_rad", "joint_velocity_rad_s")
                                   for joint in range(1, 7)],
        "warmup_samples": max(INPUT_IDX),
        "duration_s": args.duration_s,
        "duration_origin": "common_start_monotonic_ns; includes the history/warmup interval; applies to all splits",
        "torque_label": "sdk_current_estimate", "torque_unit": "Nm",
        "torque_coefficients_nm_per_a": torque_coefficients.tolist(),
        "torque_sdk_coefficients_nm_per_a": SDK_TORQUE_COEFFICIENTS.tolist(),
        "torque_firmware_multipliers": (torque_coefficients / SDK_TORQUE_COEFFICIENTS).tolist(),
        "torque_coefficient_source": "SDK cal_effort coefficients with the firmware correction from official Q&A section 3.2",
        "torque_firmware_source": TORQUE_FIRMWARE_SOURCE,
        "firmware": first_metadata["firmware"], "control": first_metadata["control"],
        "setup": first_metadata["setup"],
        "motion_window_s": MOTION_WINDOW * PERIOD_NS * 1e-9, "motion_rate_rad_s": MOTION_RATE_RAD_S,
        "checkpoint_path": source_batches[0]["checkpoint_path"],
        "split_counts": dict(Counter(target["split"] for _, _, target in trajectories)),
        "notes": {
            "time": "Causal zero-order hold on host monotonic time; TX is host send-call end, not motor receipt. Transport effects remain included.",
            "source_clock": "CAN source timestamps are retained, not mixed with the host time axis.",
            "history": "Features contain current-to-past position errors, followed by velocities. Windows stay in one policy trajectory; no padding.",
            "warmup": "The first 64 policy grid errors, velocities, absolute positions and timestamps are retained separately for LSTM state warmup, without labels or scored samples.",
            "splits": "Each raw trajectory keeps its original split. Batch-prefixed names distinguish identical trajectory names across batches.",
            "labels": "SDK current-derived torque with the documented firmware multiplier (J1-J3 x4 for <=1.8-2), including static support; not independently calibrated output torque.",
        },
        "trajectories": [],
    }
    write_json(destination / "manifest.json", manifest)
    for index, (batch_id, source, target) in enumerate(trajectories):
        original_name = target["trajectory"]
        name = f"{batch_id}__{original_name}"
        folder = source / original_name
        metadata = json.loads((folder / "metadata.json").read_text())
        if metadata["outcome"] != "complete" or metadata["mode"] != "move_j" or metadata["source"] != "policy":
            raise ValueError(f"Invalid session: {name}")
        for key in ("control", "checkpoint_path", "setup", "firmware"):
            if metadata[key] != manifest[key]:
                raise ValueError(f"Mixed {key}: {folder}")
        relative = f"trajectories/{name}.npz"
        result = prepare_trajectory(folder, destination / relative, args.duration_s, torque_coefficients)
        manifest["trajectories"].append({"trajectory": name, "source_batch": batch_id,
                                         "source_trajectory": original_name, "target": target,
                                         "split": target["split"], "file": relative, **result})
        if (index + 1) % 20 == 0:
            print(f"Prepared {index+1}/{len(trajectories)} trajectories", flush=True)
    manifest["outcome"] = "complete"
    manifest["samples_by_split"] = {
        split: sum(t["samples"] for t in manifest["trajectories"] if t["split"] == split)
        for split in manifest["split_counts"]
    }
    write_json(destination / "manifest.json", manifest)
    print(f"Prepared dataset: {destination}; samples: {manifest['samples_by_split']}", flush=True)


if __name__ == "__main__":
    main()
