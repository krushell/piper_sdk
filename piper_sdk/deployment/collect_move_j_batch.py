"""Preview or collect MOVE_J policy rollouts at different nonnegative-z targets.

Each rollout runs collect_actuator_data in a separate process, retaining its
initialization, recording, return-to-zero and failure handling. Without
--preview this command operates the arm. Run long batches in their own tmux.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from piper_sdk.deployment.actuator_recording import write_json
from piper_sdk.deployment.manipulation import (
    PIPER_MAX_TARGET_DELTA_RAD_PER_POLICY_TICK,
    POLICY_CONTROL_PERIOD,
    POSE_COMMAND_RANGES,
    SPEED_PERCENT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--preview", action="store_true", help="Save target positions without opening CAN.")
    parser.add_argument("--num_trajectories", type=int, default=100)
    parser.add_argument("--policy_steps", type=int, default=1000, help="Steps per target; 1000 is 20 seconds.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_path", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--can_name", default="can_piper")
    parser.add_argument("--can_host", default="192.168.123.162")
    parser.add_argument("--can_port", type=int, default=29536)
    args = parser.parse_args()
    if args.num_trajectories <= 0 or args.policy_steps <= 0:
        parser.error("Trajectory and step counts must be positive")
    if not args.preview and args.checkpoint_path is None:
        parser.error("Collection requires --checkpoint_path")
    if args.checkpoint_path is not None:
        args.checkpoint_path = args.checkpoint_path.expanduser().resolve()
        if not args.checkpoint_path.is_file():
            parser.error(f"Checkpoint not found: {args.checkpoint_path}")
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def generate_targets(count: int, seed: int) -> list[dict]:
    # Sobol covers the policy's spherical command range without clustering
    # independent random draws. Restrict pitch to the upper half-space.
    ranges = list(POSE_COMMAND_RANGES)
    ranges[1] = (max(0.0, ranges[1][0]), ranges[1][1])
    samples = torch.quasirandom.SobolEngine(3, scramble=True, seed=seed).draw(count, dtype=torch.float64)
    split_indices = list(range(count))
    random.Random(seed).shuffle(split_indices)
    holdout_count = count // 10
    splits = ["train"] * count
    for index in split_indices[:holdout_count]:
        splits[index] = "validation"
    for index in split_indices[holdout_count:2 * holdout_count]:
        splits[index] = "test"

    targets = []
    for index, sample in enumerate(samples.tolist()):
        radius, pitch, yaw = [lower + value * (upper - lower) for value, (lower, upper) in zip(sample, ranges)]
        targets.append({
            "trajectory": f"policy_{index:03d}", "split": splits[index],
            "radius_m": radius, "pitch_rad": pitch, "yaw_rad": yaw,
            "target_x_b_m": radius * math.cos(pitch) * math.cos(yaw),
            "target_y_b_m": radius * math.cos(pitch) * math.sin(yaw),
            "target_z_b_m": radius * math.sin(pitch),
        })
    return targets


def save_targets(output_dir: Path, targets: list[dict]) -> None:
    with (output_dir / "targets.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(targets[0]))
        writer.writeheader()
        writer.writerows(targets)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")
    for split, color in (("train", "tab:blue"), ("validation", "tab:orange"), ("test", "tab:green")):
        rows = [row for row in targets if row["split"] == split]
        if rows:
            ax.scatter(*[[row[f"target_{axis}_b_m"] for row in rows] for axis in "xyz"],
                       color=color, label=f"{split} ({len(rows)})", s=24)
    ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
           zlim=(0, max(row["target_z_b_m"] for row in targets) * 1.05),
           title=f"{len(targets)} MOVE_J policy targets (command frame)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "targets.png", dpi=150)
    plt.close(fig)


def run_session(command: list[str], log_path: Path) -> None:
    with log_path.open("w") as stream:
        # The parent forwards interruption once, giving the existing collector
        # time to stop the arm and finish its data files before the batch exits.
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = child.wait()
        except BaseException:
            child.terminate()
            child.wait()
            raise
    if returncode != 0:
        raise RuntimeError(f"Collector exited with code {returncode}; see {log_path}")


def collect_batch(args: argparse.Namespace, targets: list[dict], metadata: dict) -> None:
    active = None
    metadata["outcome"] = "running"
    try:
        for index, target in enumerate(targets):
            active = metadata["sessions"][index]
            active.update(outcome="running", started_utc=datetime.now(timezone.utc).isoformat())
            write_json(args.output_dir / "batch.json", metadata)
            command = [
                sys.executable, "-m", "piper_sdk.deployment.collect_actuator_data",
                "--mode", "move_j", "--source", "policy",
                "--checkpoint_path", str(args.checkpoint_path),
                "--policy_steps", str(args.policy_steps), "--device", args.device,
                "--can_name", args.can_name, "--can_host", args.can_host, "--can_port", str(args.can_port),
                "--output_dir", str(args.output_dir / target["trajectory"]),
                "--target_pos_b", *[str(target[f"target_{axis}_b_m"]) for axis in "xyz"],
            ]
            position = [round(target[f"target_{axis}_b_m"], 4) for axis in "xyz"]
            print(f"[{index + 1}/{len(targets)}] {target['trajectory']} ({target['split']}) target={position}", flush=True)
            run_session(command, args.output_dir / f"{target['trajectory']}.log")
            active.update(outcome="complete", finished_utc=datetime.now(timezone.utc).isoformat())
            active = None
        metadata["outcome"] = "complete"
    except BaseException as exc:
        metadata["outcome"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        metadata["failure"] = f"{type(exc).__name__}: {exc}"
        if active is not None:
            active["outcome"] = metadata["outcome"]
        raise
    finally:
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["completed_trajectories"] = sum(row["outcome"] == "complete" for row in metadata["sessions"])
        write_json(args.output_dir / "batch.json", metadata)
        print(f"批次 {metadata['outcome']}: {metadata['completed_trajectories']}/{len(targets)}, {args.output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    targets = generate_targets(args.num_trajectories, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    save_targets(args.output_dir, targets)
    metadata = {
        "mode": "move_j", "source": "policy", "outcome": "preview" if args.preview else "pending",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(args.checkpoint_path) if args.checkpoint_path else None,
        "connection": {"interface": "socketcand", "channel": args.can_name, "host": args.can_host, "port": args.can_port},
        "num_trajectories": args.num_trajectories, "policy_steps": args.policy_steps,
        "policy_duration_s": args.num_trajectories * args.policy_steps * POLICY_CONTROL_PERIOD,
        "period_s": POLICY_CONTROL_PERIOD, "speed_percent": SPEED_PERCENT,
        "max_target_delta_rad_per_tick": PIPER_MAX_TARGET_DELTA_RAD_PER_POLICY_TICK,
        "seed": args.seed, "sampler": "torch.quasirandom.SobolEngine(scramble=True)",
        "pose_command_ranges": POSE_COMMAND_RANGES, "min_target_z_m": 0.0,
        "target_ranges_m": {axis: [min(row[f"target_{axis}_b_m"] for row in targets),
                                   max(row[f"target_{axis}_b_m"] for row in targets)] for axis in "xyz"},
        "split_counts": {split: sum(row["split"] == split for row in targets) for split in ("train", "validation", "test")},
        "notes": {
            "targets": "Uniform coverage in spherical policy-command coordinates, not uniform workspace volume. Explicit position targets use local_yaw=0 in Manipulation.",
            "height": "z is in the policy command frame: gripper z in the arm-base frame minus 0.123 m. Nonnegative target z does not constrain intermediate motion or every link.",
            "duration": "Policy time only; initialization, mode takeover, return and file processing add time per trajectory.",
            "splits": "Keep each trajectory, including all its CAN frames and time windows, in its assigned split. Do not randomly split neighboring samples.",
        },
        "sessions": [{"trajectory": row["trajectory"], "split": row["split"], "outcome": "pending"} for row in targets],
    }
    write_json(args.output_dir / "batch.json", metadata)
    print(f"{len(targets)} 个目标，策略运行共 {metadata['policy_duration_s'] / 60:.2f} 分钟（另加初始化与回位）；z 范围 {metadata['target_ranges_m']['z']} m", flush=True)
    if args.preview:
        print(f"目标预览已保存（未连接 CAN）: {args.output_dir}")
        return

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    collect_batch(args, targets, metadata)


if __name__ == "__main__":
    main()
