"""Train six offline MLPs and export the Isaac Lab ActuatorNetMLP interface."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from piper_sdk.deployment.actuator_recording import write_json
from piper_sdk.deployment.prepare_actuator_dataset import validate_torque_coefficients


class ActuatorMLP(nn.Module):
    """Physical rad/rad/s inputs to SDK-estimated Nm, including normalization."""

    def __init__(self, input_mean, input_std, target_mean, target_std):
        super().__init__()
        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_std", input_std)
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("target_std", target_std)
        self.mlp = nn.Sequential(
            nn.Linear(input_mean.numel(), 64), nn.ELU(),
            nn.Linear(64, 64), nn.ELU(),
            nn.Linear(64, 32), nn.ELU(), nn.Linear(32, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = (inputs - self.input_mean) / self.input_std
        return self.mlp(normalized) * self.target_std + self.target_mean


def load_dataset(root: Path, manifest: dict) -> tuple[dict, list]:
    parts = {split: {key: [] for key in ("inputs", "torque_est_nm", "motion")} for split in ("train", "validation", "test")}
    offsets = {split: 0 for split in parts}
    trajectories = []
    for record in manifest["trajectories"]:
        split = record["split"]
        with np.load(root / record["file"]) as data:
            for key in parts[split]:
                parts[split][key].append(data[key])
            count = len(data["time_monotonic_ns"])
            trajectories.append({"trajectory": record["trajectory"], "split": split,
                                 "start": offsets[split], "stop": offsets[split] + count,
                                 "time_monotonic_ns": data["time_monotonic_ns"]})
            offsets[split] += count
    return {split: {key: np.concatenate(value) for key, value in arrays.items()} for split, arrays in parts.items()}, trajectories


def metrics(target: np.ndarray, predicted: np.ndarray, motion: np.ndarray) -> dict:
    result = {}
    for name, mask in (("overall", np.ones(len(target), dtype=bool)), ("motion", motion), ("hold", ~motion)):
        y, p = target[mask].astype(np.float64), predicted[mask].astype(np.float64)
        # A cropped trajectory can contain no hold samples for an individual joint.
        if not len(y):
            result[name] = {"samples": 0, "mae_nm": None, "rmse_nm": None, "r2": None}
            continue
        error = p - y
        variance_sum = np.sum((y-y.mean())**2)
        result[name] = {"samples": len(y), "mae_nm": float(np.abs(error).mean()),
                        "rmse_nm": float(np.sqrt(np.mean(error**2))),
                        "r2": float(1 - np.sum(error**2) / variance_sum) if variance_sum > 0 else None}
    return result


def fit_baselines(x: np.ndarray, y: np.ndarray, mean, std) -> tuple[float, np.ndarray]:
    design = np.column_stack(((x.astype(np.float64) - mean) / std, np.ones(len(x))))
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return float(y.astype(np.float64).mean()), coefficients


def train_joint(joint: int, data: dict, args, manifest: dict, curve_writer, curve_stream) -> tuple[dict, dict]:
    torch.manual_seed(args.seed + joint)
    device = torch.device(args.device)
    tensors = {
        split: {"x": torch.as_tensor(d["inputs"][:, joint, :], device=device),
                "y": torch.as_tensor(d["torque_est_nm"][:, joint:joint+1], device=device),
                "motion": torch.as_tensor(d["motion"][:, joint], device=device)}
        for split, d in data.items()
    }
    train = tensors["train"]
    input_mean, input_std = train["x"].mean(0), train["x"].std(0, unbiased=False)
    target_mean, target_std = train["y"].mean(), train["y"].std(unbiased=False)
    if bool((input_std <= 0).any()) or float(target_std) <= 0:
        raise ValueError(f"J{joint+1}: constant training feature or label")
    model = ActuatorMLP(input_mean, input_std, target_mean, target_std).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_score, best_epoch, stale = math.inf, 0, 0
    checkpoint_path = args.output_dir / "checkpoints" / f"j{joint+1}_best.pt"
    curve = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = torch.zeros((), device=device)
        for index in torch.randperm(len(train["x"]), device=device).split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            residual = (model(train["x"][index]) - train["y"][index]) / target_std
            loss = residual.square().mean()
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * len(index)
        model.eval()
        with torch.inference_mode():
            validation = tensors["validation"]
            error = (model(validation["x"]) - validation["y"]).squeeze(1).square()
            val_score = float(error.mean())
            train_score = float(loss_sum / len(train["x"])) * float(target_std)**2
        if not math.isfinite(val_score + train_score):
            raise FloatingPointError(f"J{joint+1}: non-finite training or validation loss at epoch {epoch}")
        row = {"joint": joint+1, "epoch": epoch, "train_rmse_nm": math.sqrt(train_score),
               "validation_rmse_nm": math.sqrt(val_score)}
        curve.append(row)
        curve_writer.writerow(row)
        curve_stream.flush()
        if val_score < best_score:
            best_score, best_epoch, stale = val_score, epoch, 0
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "best_epoch": best_epoch, "validation_mse_nm2": best_score,
                        "hidden_dims": [64, 64, 32], "activation": "elu",
                        "input_idx": manifest["input_idx"], "input_order": manifest["input_order"]}, checkpoint_path)
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or stale == 15:
            print(f"J{joint+1} epoch {epoch}: train={math.sqrt(train_score):.5f} val={math.sqrt(val_score):.5f} Nm; best={best_epoch}", flush=True)
        if stale == 15:
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    # The test split is first evaluated after this joint's model selection.
    with torch.inference_mode():
        predictions = {split: model(tensors[split]["x"]).squeeze(1).cpu().numpy() for split in ("validation", "test")}
    mean, std = input_mean.cpu().numpy(), input_std.cpu().numpy()
    constant, coefficients = fit_baselines(data["train"]["inputs"][:, joint, :],
                                           data["train"]["torque_est_nm"][:, joint],
                                           mean, std)
    result = {"best_epoch": best_epoch, "epochs_run": epoch,
              "training_motion_fraction": float(data["train"]["motion"][:, joint].mean()),
              "baselines": {"constant_nm": constant, "linear_normalized_input_coefficients": coefficients.tolist()},
              "normalization": {"input_mean": mean.tolist(), "input_std": std.tolist(),
                                "target_mean_nm": float(target_mean), "target_std_nm": float(target_std)}}
    for split in ("validation", "test"):
        target = data[split]["torque_est_nm"][:, joint]
        motion = data[split]["motion"][:, joint]
        design = np.column_stack(((data[split]["inputs"][:, joint, :].astype(np.float64)-mean)/std, np.ones(len(target))))
        result[split] = {
            "mlp": metrics(target, predictions[split], motion),
            "constant": metrics(target, np.full(len(target), constant), motion),
            "linear": metrics(target, design @ coefficients, motion),
        }
    script_path = args.output_dir / "models" / f"j{joint+1}.pt"
    model.cpu()
    torch.jit.script(model).save(str(script_path))
    loaded = torch.jit.load(str(script_path), map_location=device).eval()
    model.to(device)
    with torch.inference_mode():
        difference = max(float((loaded(tensors[s]["x"]) - model(tensors[s]["x"])).abs().max()) for s in ("validation", "test"))
    if difference > 1e-5:
        raise ValueError(f"J{joint+1}: TorchScript output mismatch {difference} Nm")
    result["torchscript_max_abs_difference_nm"] = difference
    result["torchscript_file"] = str(script_path)
    result["test_better_than_baselines"] = {
        baseline: result["test"]["mlp"]["overall"]["rmse_nm"] < result["test"][baseline]["overall"]["rmse_nm"]
        for baseline in ("constant", "linear")
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["epoch"] for r in curve], [r["train_rmse_nm"] for r in curve], label="train (epoch online)")
    ax.plot([r["epoch"] for r in curve], [r["validation_rmse_nm"] for r in curve], label="validation")
    ax.axvline(best_epoch, color="grey", linestyle="--", label="selected epoch")
    ax.set(xlabel="Epoch", ylabel="RMSE (estimated Nm)", title=f"J{joint+1}")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "plots" / f"j{joint+1}_learning.png", dpi=130)
    plt.close(fig)
    print(f"J{joint+1} exported: test RMSE={result['test']['mlp']['overall']['rmse_nm']:.5f} Nm; export difference={difference:.3g}", flush=True)
    return result, predictions


def save_heldout_predictions(output: Path, data: dict, trajectories: list, predictions: dict) -> None:
    metric_rows = []
    for record in trajectories:
        split = record["split"]
        if split == "train":
            continue
        selection = slice(record["start"], record["stop"])
        target = data[split]["torque_est_nm"][selection]
        motion = data[split]["motion"][selection]
        predicted = predictions[split][selection]
        np.savez_compressed(output / "predictions" / f"{record['trajectory']}.npz",
                            time_monotonic_ns=record["time_monotonic_ns"], torque_est_nm=target,
                            predicted_torque_nm=predicted, motion=motion)
        time_s = (record["time_monotonic_ns"] - record["time_monotonic_ns"][0]) * 1e-9
        fig, axes = plt.subplots(6, 1, figsize=(11, 12), sharex=True)
        for joint, ax in enumerate(axes):
            scores = metrics(target[:, joint], predicted[:, joint], motion[:, joint])
            for regime in ("overall", "motion", "hold"):
                metric_rows.append({"trajectory": record["trajectory"], "split": split, "joint": joint+1,
                                    "regime": regime, **scores[regime]})
            ax.plot(time_s, target[:, joint], alpha=.7, linewidth=.7, label="SDK torque estimate")
            ax.plot(time_s, predicted[:, joint], linewidth=.8, label="MLP prediction")
            ax.set_ylabel(f"J{joint+1} (Nm)")
            ax.grid(alpha=.2)
        axes[0].legend()
        axes[0].set_title(f"{record['trajectory']} ({split}) — estimated torque labels")
        axes[-1].set_xlabel("Time from first usable sample (s)")
        fig.tight_layout()
        fig.savefig(output / "plots" / f"{record['trajectory']}.png", dpi=120)
        plt.close(fig)
    with (output / "trajectory_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2048)
    args = parser.parse_args()
    args.dataset_dir = args.dataset_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch_size must be positive")
    torch.set_num_threads(1)
    manifest = json.loads((args.dataset_dir / "manifest.json").read_text())
    if manifest["outcome"] != "complete":
        raise ValueError("Dataset preparation has not completed")
    validate_torque_coefficients(manifest)
    data, trajectories = load_dataset(args.dataset_dir, manifest)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in ("models", "checkpoints", "plots", "predictions"):
        (args.output_dir / name).mkdir()
    report = {"outcome": "training", "dataset_dir": str(args.dataset_dir),
              "device": args.device, "seed": args.seed, "joint_seeds": list(range(args.seed, args.seed+6)),
              "torch_version": torch.__version__, "training": {"hidden_dims": [64, 64, 32], "activation": "elu",
              "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": args.batch_size,
              "maximum_epochs": args.epochs, "early_stopping_patience": 15,
              "loss": "ordinary MSE; equal weight per sample, no motion/hold class weights",
              "selection": "minimum unweighted validation MSE"},
              "dataset_duration_s": manifest["duration_s"],
              "samples_by_split": manifest["samples_by_split"], "torque_label": manifest["torque_label"],
              "firmware": manifest["firmware"],
              "torque_coefficients_nm_per_a": manifest["torque_coefficients_nm_per_a"],
              "torque_firmware_source": manifest["torque_firmware_source"],
              "notes": "Offline fit to SDK current-derived torque estimates; not calibrated physical torque or validated simulated dynamics.",
              "joints": {}}
    write_json(args.output_dir / "metrics.json", report)
    predictions = {s: [] for s in ("validation", "test")}
    try:
        with (args.output_dir / "learning_curves.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["joint", "epoch", "train_rmse_nm", "validation_rmse_nm"])
            writer.writeheader()
            for joint in range(6):
                result, predicted = train_joint(joint, data, args, manifest, writer, stream)
                report["joints"][f"j{joint+1}"] = result
                for split in predictions:
                    predictions[split].append(predicted[split])
                write_json(args.output_dir / "metrics.json", report)
        stacked = {s: np.column_stack(v) for s, v in predictions.items()}
        save_heldout_predictions(args.output_dir, data, trajectories, stacked)
        network_config = {
            "class": "isaaclab.actuators.ActuatorNetMLPCfg", "required_call_period_s": manifest["sample_period_s"],
            "input_idx": manifest["input_idx"], "input_order": manifest["input_order"],
            "pos_scale": 1.0, "vel_scale": 1.0, "torque_scale": 1.0,
            "reference": "actual sent joint target; MOVE_J speed_percent=5, 0.006 rad maximum change per 20 ms policy tick",
            "normalization": "Embedded in each TorchScript; input rad and rad/s, output estimated Nm",
            "torque_coefficients_nm_per_a": manifest["torque_coefficients_nm_per_a"],
            "firmware": manifest["firmware"], "torque_firmware_source": manifest["torque_firmware_source"],
            "actuators": {f"j{j}": {"joint_names_expr": [f"arm_j{j}"], "network_file": str(args.output_dir / "models" / f"j{j}.pt")} for j in range(1,7)},
            "integration": "Network parameters only. Physical effort/velocity/saturation limits and simulation rollout validation are deferred.",
        }
        write_json(args.output_dir / "actuator_network_config.json", network_config)
        report["outcome"] = "complete"
    except BaseException as exc:
        report["outcome"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(args.output_dir / "metrics.json", report)
    print(f"Training complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
