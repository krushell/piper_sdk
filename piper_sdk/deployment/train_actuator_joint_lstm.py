"""Train one full-arm LSTM: six errors, positions and velocities to six torques."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from piper_sdk.deployment.actuator_recording import write_json
from piper_sdk.deployment.prepare_actuator_dataset import validate_torque_coefficients
from piper_sdk.deployment.train_actuator_net import metrics
from piper_sdk.deployment.train_actuator_lstm import rollout as independent_rollout, write_rows


DEFAULT_LAYERS = 2
DEFAULT_HIDDEN_SIZE = 64
SEQUENCE_LENGTH = 256
DEFAULT_PATIENCE = 15
SPLITS = ("train", "validation", "test")
PREDICTORS = ("joint_lstm", "independent_mlp", "independent_lstm", "full_state_linear", "constant")


class JointActuatorLSTM(nn.Module):
    """One recurrent state per arm; physical units at the model boundary."""

    def __init__(self, input_mean, input_std, target_mean, target_std, num_layers: int, hidden_size: int):
        super().__init__()
        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_std", input_std)
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("target_std", target_std)
        self.lstm = nn.LSTM(18, hidden_size, num_layers=num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, 6)

    def forward(
        self, inputs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        features, next_state = self.lstm((inputs-self.input_mean)/self.input_std, state)
        return self.output(features)*self.target_std+self.target_mean, next_state


def zero_state(model, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (model.lstm.num_layers, inputs.shape[0], model.lstm.hidden_size)
    return inputs.new_zeros(shape), inputs.new_zeros(shape)


@torch.inference_mode()
def rollout(model, inputs: torch.Tensor, state=None):
    if state is None:
        state = zero_state(model, inputs)
    predictions = []
    for start in range(0, inputs.shape[1], SEQUENCE_LENGTH):
        predicted, state = model(inputs[:, start:start+SEQUENCE_LENGTH], state)
        predictions.append(predicted)
    return torch.cat(predictions, dim=1), state


def load_trajectories(args, manifest: dict) -> tuple[dict, dict]:
    reference_reports = {name: json.loads((path / "metrics.json").read_text())
                         for name, path in (("mlp", args.mlp_dir), ("lstm", args.lstm_dir))}
    if any(r["outcome"] != "complete" for r in reference_reports.values()):
        raise ValueError("Reference MLP and LSTM must be complete")
    reference_datasets = {}
    for name, report in reference_reports.items():
        root = Path(report["dataset_dir"])
        reference = json.loads((root / "manifest.json").read_text())
        for key in ("phase", "sample_period_s", "input_idx", "input_order", "control", "setup",
                    "checkpoint_path", "firmware", "torque_coefficients_nm_per_a", "duration_s"):
            if manifest[key] != reference[key]:
                raise ValueError(f"Reference {name} preprocessing differs: {key}")
        reference_datasets[name] = {"dataset_dir": str(root), "split_counts": reference["split_counts"],
                                    "samples_by_split": reference["samples_by_split"]}
    groups = {s: [] for s in SPLITS}
    current = manifest["input_idx"].index(0)
    period_ns = round(manifest["sample_period_s"]*1e9)
    for record in manifest["trajectories"]:
        name, split = record["trajectory"], record["split"]
        with np.load(args.dataset_dir / record["file"]) as data:
            prefix = data["warmup_inputs"]
            warmup = np.concatenate((prefix[:, :, 0], data["warmup_joint_pos_rad"], prefix[:, :, 1]), axis=1)
            scored = np.concatenate((data["inputs"][:, :, current], data["joint_pos_rad"], data["joint_vel_rad_s"]), axis=1)
            times = np.concatenate((data["warmup_time_monotonic_ns"], data["time_monotonic_ns"]))
            if len(warmup) != manifest["warmup_samples"] or np.any(np.diff(times) != period_ns):
                raise ValueError(f"Invalid warmup or grid: {name}")
            item = {"trajectory": name, "split": split, "source_batch": record["source_batch"],
                    "common_start_monotonic_ns": record["common_start_monotonic_ns"],
                    "warmup": len(warmup), "x": np.concatenate((warmup, scored)),
                    "y": data["torque_est_nm"], "motion": data["motion"],
                    "time_monotonic_ns": data["time_monotonic_ns"],
                    "mlp_inputs": data["inputs"], "predictions": {}}
        groups[split].append(item)
    acceptance = {"reference_datasets": reference_datasets, "reference_preprocessing_matches": True,
                  "original_splits_preserved": True, "input_names": manifest["joint_lstm_input_names"],
                  "warmup_position_source": "Raw position feedback decoded with the same causal CAN alignment",
                  "sample_period_s": manifest["sample_period_s"], "duration_s": manifest["duration_s"],
                  "trajectories_by_split": {s: len(rows) for s, rows in groups.items()},
                  "samples_by_split": {s: sum(len(r["y"]) for r in rows) for s, rows in groups.items()},
                  "warmup_samples_by_split": {s: sum(r["warmup"] for r in rows) for s, rows in groups.items()},
                  "hold_fraction_by_split": {s: (~np.concatenate([r["motion"] for r in rows])).mean(0).tolist()
                                             for s, rows in groups.items()}}
    return groups, acceptance


@torch.inference_mode()
def predict_reference_models(records: list, args) -> None:
    """Score frozen independent models on this run's actual held-out trajectories."""
    full = torch.as_tensor(np.stack([r["x"] for r in records]), device=args.device)
    history = torch.as_tensor(np.stack([r["mlp_inputs"] for r in records]), device=args.device)
    independent = np.empty((len(records), history.shape[1], 6), dtype=np.float32)
    mlp = np.empty_like(independent)
    for joint in range(6):
        recurrent_model = torch.jit.load(str(args.lstm_dir / "models" / f"j{joint+1}.pt"),
                                        map_location=args.device).eval()
        own = torch.stack((full[:, :, joint], full[:, :, joint+12]), dim=-1)
        predicted, _ = independent_rollout(recurrent_model, own)
        independent[:, :, joint] = predicted[:, 64:, 0].cpu().numpy()
        feedforward = torch.jit.load(str(args.mlp_dir / "models" / f"j{joint+1}.pt"),
                                    map_location=args.device).eval()
        mlp[:, :, joint] = feedforward(history[:, :, joint]).squeeze(-1).cpu().numpy()
    for index, record in enumerate(records):
        record["predictions"].update(independent_mlp=mlp[index], independent_lstm=independent[index])


def trajectory_tensors(records: list, device: str) -> dict:
    parts = {key: [] for key in ("x", "y", "valid")}
    for record in records:
        warmup = record["warmup"]
        y = np.concatenate((np.zeros((warmup, 6), dtype=np.float32), record["y"]))
        parts["x"].append(torch.as_tensor(record["x"], device=device))
        parts["y"].append(torch.as_tensor(y, device=device))
        parts["valid"].append(torch.arange(len(y), device=device) >= warmup)
    return {key: pad_sequence(value, batch_first=True) for key, value in parts.items()}


def fit_model(groups: dict, args, manifest: dict) -> tuple[JointActuatorLSTM, dict]:
    torch.manual_seed(args.seed)
    train = trajectory_tensors(groups["train"], args.device)
    validation = trajectory_tensors(groups["validation"], args.device)
    scored_x, scored_y = train["x"][train["valid"]], train["y"][train["valid"]]
    mean, std = scored_x.mean(0), scored_x.std(0, unbiased=False)
    target_mean, target_std = scored_y.mean(0), scored_y.std(0, unbiased=False)
    if bool((std <= 0).any()) or bool((target_std <= 0).any()):
        raise ValueError("Constant training feature or label")
    model = JointActuatorLSTM(mean, std, target_mean, target_std,
                             args.num_layers, args.hidden_size).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_score, best_epoch, stale = math.inf, 0, 0
    curve = []
    checkpoint_path = args.output_dir / "checkpoints" / "joint_lstm_best.pt"
    fields = ["epoch", "train_normalized_mse", "validation_normalized_mse"]
    fields += [f"{split}_j{j}_rmse_nm" for split in ("train", "validation") for j in range(1, 7)]
    with (args.output_dir / "learning_curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs+1):
            model.train()
            squared_sum = torch.zeros(6, device=args.device)
            order = torch.randperm(len(groups["train"]), device=args.device)
            for indices in order.split(args.batch_size):
                x, y, valid = train["x"][indices], train["y"][indices], train["valid"][indices]
                state = zero_state(model, x)
                for start in range(0, x.shape[1], SEQUENCE_LENGTH):
                    selection = slice(start, start+SEQUENCE_LENGTH)
                    optimizer.zero_grad(set_to_none=True)
                    predicted, state = model(x[:, selection], state)
                    residual = ((predicted-y[:, selection])/target_std)[valid[:, selection]]
                    loss = residual.square().mean()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    state = tuple(value.detach() for value in state)
                    squared_sum += residual.detach().square().sum(0)
            model.eval()
            predicted, _ = rollout(model, validation["x"])
            val_errors = ((predicted-validation["y"])/target_std)[validation["valid"]]
            val_joint_mse = val_errors.square().mean(0)
            train_joint_mse = squared_sum/train["valid"].sum()
            score = float(val_joint_mse.mean())
            train_score = float(train_joint_mse.mean())
            if not math.isfinite(score+train_score):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            row = {"epoch": epoch, "train_normalized_mse": train_score, "validation_normalized_mse": score}
            for split, values in (("train", train_joint_mse), ("validation", val_joint_mse)):
                row.update({f"{split}_j{j+1}_rmse_nm": value for j, value in enumerate((values.sqrt()*target_std).tolist())})
            curve.append(row)
            writer.writerow(row)
            stream.flush()
            if score < best_score:
                best_score, best_epoch, stale = score, epoch, 0
                torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            "best_epoch": best_epoch, "validation_normalized_mse": best_score,
                            "num_layers": args.num_layers, "hidden_size": args.hidden_size,
                            "input_size": 18, "output_size": 6,
                            "input_names": manifest["joint_lstm_input_names"]}, checkpoint_path)
            else:
                stale += 1
            stop = args.early_stopping_patience > 0 and stale >= args.early_stopping_patience
            if epoch == 1 or epoch % 10 == 0 or stop:
                print(f"epoch {epoch}: train NMSE={train_score:.6f}, val NMSE={score:.6f}, "
                      f"val J2 RMSE={row['validation_j2_rmse_nm']:.5f} Nm; best={best_epoch}", flush=True)
            if stop:
                break
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "epoch": epoch, "validation_normalized_mse": score,
                "num_layers": args.num_layers, "hidden_size": args.hidden_size, "input_size": 18, "output_size": 6,
                "input_names": manifest["joint_lstm_input_names"]},
               args.output_dir / "checkpoints" / "joint_lstm_last.pt")
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    fig, ax = plt.subplots(figsize=(8, 4))
    for split in ("train", "validation"):
        ax.plot([r["epoch"] for r in curve], [r[f"{split}_normalized_mse"] for r in curve], label=split)
    ax.axvline(best_epoch, color="grey", linestyle="--", label="selected epoch")
    ax.set(xlabel="Epoch", ylabel="Mean normalized MSE across six joints")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "plots" / "learning_loss.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for joint, ax in enumerate(axes.flat, 1):
        for split in ("train", "validation"):
            ax.plot([r["epoch"] for r in curve], [r[f"{split}_j{joint}_rmse_nm"] for r in curve], label=split)
        ax.axvline(best_epoch, color="grey", linestyle="--")
        ax.set(title=f"J{joint}", xlabel="Epoch", ylabel="RMSE (estimated Nm)")
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "plots" / "learning_joint_rmse.png", dpi=140)
    plt.close(fig)
    result = {"best_epoch": best_epoch, "epochs_run": epoch, "validation_normalized_mse": best_score,
              "last_validation_normalized_mse": score,
              "early_stopping_patience": args.early_stopping_patience,
              "stopped_early": epoch < args.epochs,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "normalization": {"input_mean": mean.tolist(), "input_std": std.tolist(),
                                "target_mean_nm": target_mean.tolist(), "target_std_nm": target_std.tolist()}}
    return model, result


@torch.inference_mode()
def export_and_predict(model, records: list, args) -> tuple[dict, dict]:
    script_path = args.output_dir / "models" / "joint_lstm.pt"
    model.cpu()
    torch.jit.script(model).save(str(script_path))
    model.to(args.device)
    loaded = torch.jit.load(str(script_path), map_location=args.device).eval()
    data = trajectory_tensors(records, args.device)
    inputs = data["x"]
    # cuDNN selects different FP32 kernels for long sequences and single steps.
    # Compare recurrence and scripting with the same native kernels; training
    # keeps its normal cuDNN execution path.
    with torch.backends.cudnn.flags(enabled=False):
        block, block_state = rollout(model, inputs)
        native_state, script_state = zero_state(model, inputs), zero_state(loaded, inputs)
        scripted = torch.empty_like(block)
        output_difference, state_difference = inputs.new_zeros(()), inputs.new_zeros(())
        for step in range(inputs.shape[1]):
            x = inputs[:, step:step+1]
            native, native_state = model(x, native_state)
            result, script_state = loaded(x, script_state)
            scripted[:, step:step+1] = result
            output_difference = torch.maximum(output_difference, (native-result).abs().max())
            for expected, actual in zip(native_state, script_state):
                state_difference = torch.maximum(state_difference, (expected-actual).abs().max())
        reset_input = inputs[-1:, :len(records[-1]["x"])]
        reset_state = tuple(value[:, :1].clone().zero_() for value in script_state)
        replay, _ = rollout(loaded, reset_input, state=reset_state)
    checks = {"torchscript_max_abs_difference_nm": float(output_difference),
              "torchscript_state_max_abs_difference": float(state_difference),
              "chunk_vs_step_max_abs_difference_nm": float((block-scripted)[data["valid"]].abs().max()),
              "chunk_vs_step_final_state_max_abs_difference": max(float((a-b).abs().max()) for a, b in zip(block_state, script_state)),
              "reset_replay_max_abs_difference_nm": float((replay-scripted[-1:, :reset_input.shape[1]]).abs().max())}
    if max(checks.values()) > 1e-5:
        raise ValueError(f"Recurrent inference mismatch: {checks}")
    predictions = scripted.cpu().numpy()
    checks.update(checked_trajectories=len(records), torchscript_file=str(script_path),
                  verification_device=str(args.device), verification_cudnn_enabled=False)
    return checks, {r["trajectory"]: predictions[i, r["warmup"]:len(r["x"])] for i, r in enumerate(records)}


def fit_linear_baseline(groups: dict, normalization: dict) -> dict:
    x = np.concatenate([r["x"][r["warmup"]:] for r in groups["train"]]).astype(np.float64)
    y = np.concatenate([r["y"] for r in groups["train"]]).astype(np.float64)
    x = (x-normalization["input_mean"])/normalization["input_std"]
    design = np.column_stack((x, np.ones(len(x))))
    standardized_y = (y-normalization["target_mean_nm"])/normalization["target_std_nm"]
    coefficients, _, _, _ = np.linalg.lstsq(design, standardized_y, rcond=None)
    return {"constant_nm": y.mean(0).tolist(), "linear_standardized_coefficients": coefficients.tolist(),
            "linear_inputs": "Current full-arm error, position and velocity; no temporal history"}


def evaluate(groups: dict, predictions: dict, report: dict, output: Path) -> None:
    normalization = report["fit"]["normalization"]
    baseline = report["baselines"]
    coefficients = np.asarray(baseline["linear_standardized_coefficients"])
    records = groups["validation"] + groups["test"]
    trajectory_rows = []
    for record in records:
        values = record["predictions"]
        values["joint_lstm"] = predictions[record["trajectory"]]
        x = (record["x"][record["warmup"]:].astype(np.float64)-normalization["input_mean"])/normalization["input_std"]
        values["full_state_linear"] = (x @ coefficients[:-1]+coefficients[-1])*normalization["target_std_nm"]+normalization["target_mean_nm"]
        values["constant"] = np.broadcast_to(baseline["constant_nm"], record["y"].shape)
        np.savez_compressed(output / "predictions" / f"{record['trajectory']}.npz",
                            time_monotonic_ns=record["time_monotonic_ns"], torque_est_nm=record["y"],
                            motion=record["motion"], **values)
        for joint in range(6):
            for name, predicted in values.items():
                for regime, scores in metrics(record["y"][:, joint], predicted[:, joint], record["motion"][:, joint]).items():
                    trajectory_rows.append({"trajectory": record["trajectory"], "source_batch": record["source_batch"],
                                            "split": record["split"], "joint": joint+1, "model": name,
                                            "regime": regime, **scores})
    write_rows(output / "trajectory_metrics.csv", trajectory_rows)
    cohorts = {"all": records}
    cohorts.update({batch: [r for r in records if r["source_batch"] == batch]
                    for batch in sorted({r["source_batch"] for r in records})})
    rows = []
    report["cohorts"] = {}
    for cohort, cohort_records in cohorts.items():
        result = {"joints": {f"j{j+1}": {} for j in range(6)}, "normalized_mse": {}, "trajectories_by_split": {}}
        for split in ("validation", "test"):
            selected = [r for r in cohort_records if r["split"] == split]
            target = np.concatenate([r["y"] for r in selected])
            motion = np.concatenate([r["motion"] for r in selected])
            result["trajectories_by_split"][split] = len(selected)
            result["normalized_mse"][split] = {}
            for joint in range(6):
                result["joints"][f"j{joint+1}"][split] = {}
            for model in PREDICTORS:
                predicted = np.concatenate([r["predictions"][model] for r in selected])
                result["normalized_mse"][split][model] = float(np.mean(((predicted-target)/normalization["target_std_nm"])**2))
                for joint in range(6):
                    scores = metrics(target[:, joint], predicted[:, joint], motion[:, joint])
                    result["joints"][f"j{joint+1}"][split][model] = scores
                    for regime, values in scores.items():
                        rows.append({"cohort": cohort, "split": split, "joint": joint+1, "model": model, "regime": regime, **values})
        report["cohorts"][cohort] = result
    write_rows(output / "comparison.csv", rows)
    report["joints"] = report["cohorts"]["all"]["joints"]
    for joint, result in report["joints"].items():
        result["test_better_than_reference"] = {
            name: result["test"]["joint_lstm"]["overall"]["rmse_nm"] < result["test"][name]["overall"]["rmse_nm"]
            for name in PREDICTORS if name != "joint_lstm"}
        print(joint, {name: round(result['test'][name]['overall']['rmse_nm'], 6) for name in PREDICTORS}, flush=True)


def plot_predictions(groups: dict, report: dict, output: Path) -> None:
    styles = (("independent_mlp", "Independent MLP", "#72a7d6"),
              ("independent_lstm", "Independent LSTM", "#e79931"),
              ("joint_lstm", "Full-arm LSTM", "#8c2981"))
    for split in ("validation", "test"):
        for record in groups[split]:
            time_s = (record["time_monotonic_ns"]-record["common_start_monotonic_ns"])*1e-9
            fig, axes = plt.subplots(6, 1, figsize=(11, 12), sharex=True)
            for joint, ax in enumerate(axes):
                ax.plot(time_s, record["y"][:, joint], color="grey", alpha=.7, linewidth=.7, label="SDK torque estimate")
                for name, label, color in styles:
                    ax.plot(time_s, record["predictions"][name][:, joint], color=color, linewidth=.8, label=label)
                ax.set_ylabel(f"J{joint+1} (Nm)")
                ax.grid(alpha=.2)
            axes[0].legend(ncol=4, fontsize=8)
            axes[0].set_title(f"{record['trajectory']} ({split}) — identical scored samples")
            axes[-1].set_xlabel("Time from aligned policy start (s)")
            fig.tight_layout()
            fig.savefig(output / "plots" / f"{record['trajectory']}.png", dpi=120)
            plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    names = ("independent_mlp", "independent_lstm", "joint_lstm", "full_state_linear")
    colors = ("#72a7d6", "#e79931", "#8c2981", "#999999")
    for joint, ax in enumerate(axes.flat, 1):
        for index, (name, color) in enumerate(zip(names, colors)):
            values = [report["joints"][f"j{joint}"]["test"][name][regime]["rmse_nm"] for regime in ("overall", "motion", "hold")]
            ax.bar(np.arange(3)+(index-1.5)*.19, values, width=.19, color=color, label=name)
        ax.set_xticks(np.arange(3), ["Overall", "Motion", "Hold"])
        ax.set(title=f"J{joint}", ylabel="RMSE (estimated Nm)")
        ax.grid(axis="y", alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.suptitle(f"Full-arm vs frozen independent networks: same {len(groups['test'])} test trajectories, first 8 seconds", y=.94)
    fig.tight_layout(rect=(0, 0, 1, .9))
    fig.savefig(output / "plots" / "test_comparison.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset_dir", "mlp_dir", "lstm_dir", "output_dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early_stopping_patience", type=int, default=DEFAULT_PATIENCE,
                        help="Epochs without validation improvement before stopping; 0 runs the full epoch budget")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--hidden_size", type=int, default=DEFAULT_HIDDEN_SIZE)
    args = parser.parse_args()
    for name in ("dataset_dir", "mlp_dir", "lstm_dir", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch_size must be positive")
    if args.early_stopping_patience < 0:
        parser.error("early_stopping_patience must be nonnegative")
    if args.num_layers < 1 or args.hidden_size < 1:
        parser.error("num_layers and hidden_size must be positive")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads((args.dataset_dir / "manifest.json").read_text())
    if manifest["outcome"] != "complete":
        raise ValueError("Dataset preparation is incomplete")
    validate_torque_coefficients(manifest)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in ("checkpoints", "models", "plots", "predictions"):
        (args.output_dir / name).mkdir()
    report = {"outcome": "loading", "dataset_dir": str(args.dataset_dir),
              "reference_mlp_dir": str(args.mlp_dir), "reference_lstm_dir": str(args.lstm_dir),
              "device": args.device, "torch_version": torch.__version__, "seed": args.seed,
              "dataset_duration_s": manifest["duration_s"], "samples_by_split": manifest["samples_by_split"],
              "source_batches": manifest["source_batches"], "torque_label": manifest["torque_label"],
              "firmware": manifest["firmware"], "torque_coefficients_nm_per_a": manifest["torque_coefficients_nm_per_a"],
              "torque_firmware_source": manifest["torque_firmware_source"],
              "training": {"num_layers": args.num_layers, "hidden_size": args.hidden_size, "input_size": 18, "output_size": 6,
                           "input_names": manifest["joint_lstm_input_names"], "bidirectional": False,
                           "dropout": 0.0, "batch_first": True, "dtype": "float32", "tf32": False,
                           "sequence_length": SEQUENCE_LENGTH, "trajectory_batch_size": args.batch_size,
                           "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4,
                           "gradient_max_norm": 1.0, "maximum_epochs": args.epochs,
                           "early_stopping_patience": args.early_stopping_patience,
                           "loss": "Ordinary MSE after per-output training-set standardization; no motion/hold weights",
                           "selection": "Minimum validation standardized MSE averaged over valid time points and six outputs",
                           "state": "One state per arm trajectory; zero at start, carry and detach between training chunks",
                           "warmup": "64 prefix inputs update state; warmup and padding excluded from losses and metrics"},
              "notes": {"labels": "SDK current-based torque estimate, not calibrated joint output torque",
                        "comparison": "Independent reference models remain frozen. Their original training dataset is recorded in dataset_acceptance.json; all models are scored on this run's identical held-out trajectories",
                        "interface": "Standalone full-arm model; no Isaac Lab actuator integration"}}
    write_json(args.output_dir / "metrics.json", report)
    started = time.monotonic()
    try:
        groups, acceptance = load_trajectories(args, manifest)
        write_json(args.output_dir / "dataset_acceptance.json", acceptance)
        print(f"Data accepted: {acceptance['samples_by_split']}; reference preprocessing matches", flush=True)
        report["outcome"] = "training"
        write_json(args.output_dir / "metrics.json", report)
        model, report["fit"] = fit_model(groups, args, manifest)
        report["outcome"] = "evaluating"
        write_json(args.output_dir / "metrics.json", report)
        # Test inference and metrics start only after the single best checkpoint is selected.
        report["export_validation"], predictions = export_and_predict(model, groups["validation"]+groups["test"], args)
        predict_reference_models(groups["validation"]+groups["test"], args)
        report["baselines"] = fit_linear_baseline(groups, report["fit"]["normalization"])
        evaluate(groups, predictions, report, args.output_dir)
        write_json(args.output_dir / "metrics.json", report)
        plot_predictions(groups, report, args.output_dir)
        write_json(args.output_dir / "model_config.json", {
            "architecture": "JointActuatorLSTM", "input_names": manifest["joint_lstm_input_names"],
            "output_names": [f"torque_est_nm_j{j}" for j in range(1, 7)],
            "input_shape": ["batch", "time", 18], "output_shape": ["batch", "time", 6],
            "state_shape": [args.num_layers, "batch", args.hidden_size], "sample_period_s": manifest["sample_period_s"],
            "normalization": "Embedded; rad/rad/s input, estimated Nm output",
            "state_reset": "Zero both h and c at the start of each independent arm trajectory",
            "torchscript_file": str(args.output_dir / "models" / "joint_lstm.pt"),
            "checkpoint_file": str(args.output_dir / "checkpoints" / "joint_lstm_best.pt"),
            "last_checkpoint_file": str(args.output_dir / "checkpoints" / "joint_lstm_last.pt"),
            "torque_coefficients_nm_per_a": manifest["torque_coefficients_nm_per_a"],
            "firmware": manifest["firmware"], "torque_firmware_source": manifest["torque_firmware_source"]})
        report["outcome"] = "complete"
    except BaseException as exc:
        report["outcome"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic()-started
        write_json(args.output_dir / "metrics.json", report)
    print(f"Joint LSTM training complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
