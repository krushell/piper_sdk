"""Deterministic joint references shared by both PiPER collection modes."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrajectoryConfig:
    centers_rad: tuple = (
        (0.0, 0.8, -0.8, 0.0, 0.0, 0.0),
        (0.3, 0.6, -1.0, 0.0, 0.2, 0.0),
        (-0.3, 1.0, -0.6, 0.0, -0.2, 0.0),
    )
    transition_speed_rad_s: float = 0.2
    center_hold_s: float = 5.0
    sweep_amplitudes_rad: tuple = (0.03, 0.10)
    sweep_duration_s: float = 30.0
    sweep_frequency_hz: tuple = (0.1, 0.4)
    between_sweeps_hold_s: float = 2.0
    multijoint_duration_s: float = 40.0
    multijoint_amplitude_rad: float = 0.05
    multijoint_frequencies_hz: tuple = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35)


@dataclass(frozen=True)
class TrajectoryPoint:
    phase: str
    segment: str
    pose_index: int
    joint_index: int  # 1..6 for a single joint, 0 for coordinated motion/holds
    target_rad: tuple[float, ...]


def generate_trajectory(
    cfg: TrajectoryConfig, period_s: float, initial_pos: tuple[float, ...]
) -> list[TrajectoryPoint]:
    if not cfg.centers_rad or any(len(center) != 6 for center in cfg.centers_rad):
        raise ValueError("centers_rad must contain six angles per pose")
    if len(cfg.multijoint_frequencies_hz) != 6:
        raise ValueError("multijoint_frequencies_hz must contain six frequencies")
    if not math.isfinite(cfg.transition_speed_rad_s) or cfg.transition_speed_rad_s <= 0:
        raise ValueError("transition_speed_rad_s must be positive and finite")

    def ticks(duration: float) -> int:
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Trajectory durations must be positive and finite")
        return math.ceil(duration / period_s)

    points: list[TrajectoryPoint] = []
    previous = tuple(initial_pos)
    for pose_index, center in enumerate(cfg.centers_rad):
        center = tuple(float(value) for value in center)
        if not all(math.isfinite(value) for value in center):
            raise ValueError("Trajectory centers must be finite")
        prefix = f"pose_{pose_index}"

        def append(phase: str, name: str, target: tuple, joint: int = 0) -> None:
            points.append(TrajectoryPoint(phase, f"{prefix}/{name}", pose_index, joint, target))

        distance = max(abs(b - a) for a, b in zip(previous, center))
        count = math.ceil(distance / (cfg.transition_speed_rad_s * period_s))
        for step in range(1, count + 1):
            target = tuple(a + (b - a) * step / count for a, b in zip(previous, center))
            append("transition", "transition", target)
        for _ in range(ticks(cfg.center_hold_s)):
            append("hold", "center_hold", center)

        count = ticks(cfg.sweep_duration_s)
        duration = count * period_s
        f0, f1 = cfg.sweep_frequency_hz
        for joint in range(6):
            for amplitude_index, amplitude in enumerate(cfg.sweep_amplitudes_rad):
                name = f"j{joint + 1}_amplitude_{amplitude_index}"
                for step in range(1, count + 1):
                    t = step * period_s
                    phase = 2 * math.pi * (f0 * t + (f1 - f0) * t * t / (2 * duration))
                    target = list(center)
                    target[joint] += amplitude * math.sin(phase)
                    append("sweep", name, tuple(target), joint + 1)
                for _ in range(ticks(cfg.between_sweeps_hold_s)):
                    append("hold", f"{name}_hold", center, joint + 1)

        for step in range(1, ticks(cfg.multijoint_duration_s) + 1):
            t = step * period_s
            target = tuple(
                q + cfg.multijoint_amplitude_rad * math.sin(2 * math.pi * frequency * t)
                for q, frequency in zip(center, cfg.multijoint_frequencies_hz)
            )
            append("multijoint", "multijoint", target)
        previous = points[-1].target_rad
    return points


def trajectory_summary(points: list[TrajectoryPoint], period_s: float) -> dict:
    ranges = [
        [min(p.target_rad[j] for p in points), max(p.target_rad[j] for p in points)]
        for j in range(6)
    ]
    return {
        "steps": len(points),
        "duration_s": len(points) * period_s,
        "joint_target_range_rad": ranges,
        "poses": sorted({p.pose_index for p in points}),
        "segments": list(dict.fromkeys(p.segment for p in points)),
    }


def write_trajectory(path: Path, points: list[TrajectoryPoint], period_s: float) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["step", "nominal_time_s", "phase", "segment", "pose_index", "joint_index"]
            + [f"requested_target_rad_j{j}" for j in range(1, 7)]
        )
        for index, point in enumerate(points):
            writer.writerow(
                [index, index * period_s, point.phase, point.segment, point.pose_index, point.joint_index]
                + list(point.target_rad)
            )
