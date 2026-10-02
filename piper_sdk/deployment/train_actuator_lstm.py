"""Train six stateful actuator LSTMs and compare them with a fixed MLP baseline."""

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


LAYERS = 2
HIDDEN_SIZE = 32
SEQUENCE_LENGTH = 256
PATIENCE = 15
TOLERANCE = 1e-5
SPLITS = ("train", "validation", "test")


class ActuatorLSTM(nn.Module):
    """Isaac Lab's (input, state) interface, with physical units at the boundary."""

    def __init__(self, input_mean, input_std, target_mean, target_std):
        super().__init__()
        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_std", input_std)
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("target_std", target_std)
        self.lstm = nn.LSTM(2, HIDDEN_SIZE, num_layers=LAYERS, batch_first=True)
        self.output = nn.Linear(HIDDEN_SIZE, 1)

    def forward(
        self, inputs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        features, next_state = self.lstm((inputs - self.input_mean) / self.input_std, state)
        return self.output(features) * self.target_std + self.target_mean, next_state


def zero_state(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (LAYERS, inputs.shape[0], HIDDEN_SIZE)
    return inputs.new_zeros(shape), inputs.new_zeros(shape)


def load_trajectories(root: Path, manifest: dict, mlp_dir: Path, mlp_report: dict) -> tuple[dict, dict]:
    reference_root = Path(mlp_report["dataset_dir"])
    reference = json.loads((reference_root / "manifest.json").read_text())
    for key in ("source_batches", "phase", "sample_period_s", "input_idx", "input_order", "control",
                "torque_coefficients_nm_per_a", "split_counts", "samples_by_split", "duration_s"):
        if manifest[key] != reference[key]:
            raise ValueError(f"LSTM and MLP datasets differ: {key}")
    original_records = {r["trajectory"]: r for r in reference["trajectories"]}
    if {r["trajectory"] for r in manifest["trajectories"]} != set(original_records):
        raise ValueError("LSTM and MLP trajectory lists differ")
    current = manifest["input_idx"].index(0)
    oldest = manifest["input_idx"].index(manifest["warmup_samples"])
    velocity_offset = len(manifest["input_idx"])
    period_ns = round(manifest["sample_period_s"] * 1e9)
    groups = {split: [] for split in SPLITS}
    for record in manifest["trajectories"]:
        name, split = record["trajectory"], record["split"]
        original = original_records[name]
        if split != original["split"]:
            raise ValueError(f"Changed split: {name}")
        with np.load(root / record["file"]) as data, np.load(reference_root / original["file"]) as old:
            for key in ("time_monotonic_ns", "inputs", "torque_est_nm", "motion"):
                if not np.array_equal(data[key], old[key]):
                    raise ValueError(f"Changed MLP comparison samples: {name}/{key}")
            warmup = data["warmup_inputs"]
            warmup_count = len(warmup)
            times = np.concatenate((data["warmup_time_monotonic_ns"], data["time_monotonic_ns"]))
            if warmup_count != manifest["warmup_samples"] or np.any(np.diff(times) != period_ns):
                raise ValueError(f"Invalid warmup boundary or time grid: {name}")
            # The oldest MLP lag in these rows independently identifies the same prefix samples.
            if not np.array_equal(warmup, old["inputs"][:warmup_count, :, [oldest, oldest + velocity_offset]]):
                raise ValueError(f"Warmup differs from recorded MLP history: {name}")
            inputs = data["inputs"][:, :, [current, current + velocity_offset]]
            item = {"trajectory": name, "split": split, "warmup": warmup_count,
                    "x": np.concatenate((warmup, inputs)), "y": data["torque_est_nm"],
                    "motion": data["motion"], "time_monotonic_ns": data["time_monotonic_ns"],
                    "mlp_inputs": data["inputs"]}
        if split != "train":
            with np.load(mlp_dir / "predictions" / f"{name}.npz") as predicted:
                for key, expected in (("time_monotonic_ns", item["time_monotonic_ns"]),
                                      ("torque_est_nm", item["y"]), ("motion", item["motion"])):
                    if not np.array_equal(predicted[key], expected):
                        raise ValueError(f"MLP prediction alignment differs: {name}/{key}")
                item["mlp_prediction"] = predicted["predicted_torque_nm"]
        groups[split].append(item)
    acceptance = {
        "reference_dataset_dir": str(reference_root), "scored_arrays_equal_to_mlp": True,
        "warmup_equal_to_recorded_history": True, "sample_period_s": manifest["sample_period_s"],
        "samples_by_split": {s: sum(len(r["y"]) for r in rows) for s, rows in groups.items()},
        "warmup_samples_by_split": {s: sum(r["warmup"] for r in rows) for s, rows in groups.items()},
        "trajectories_by_split": {s: len(rows) for s, rows in groups.items()},
        "motion_fraction_by_split": {s: np.concatenate([r["motion"] for r in rows]).mean(0).tolist()
                                     for s, rows in groups.items()},
    }
    return groups, acceptance


def joint_tensors(records: list, joint: int, device: str) -> dict:
    parts = {key: [] for key in ("x", "y", "motion", "valid")}
    for record in records:
        warmup = record["warmup"]
        parts["x"].append(torch.as_tensor(record["x"][:, joint, :], device=device))
        y = np.concatenate((np.zeros((warmup, 1), dtype=np.float32), record["y"][:, joint:joint+1]))
        parts["y"].append(torch.as_tensor(y, device=device))
        motion = np.concatenate((np.zeros(warmup, dtype=bool), record["motion"][:, joint]))
        parts["motion"].append(torch.as_tensor(motion, device=device))
        parts["valid"].append(torch.arange(len(y), device=device) >= warmup)
    return {key: pad_sequence(values, batch_first=True) for key, values in parts.items()}


@torch.inference_mode()
def rollout(model, inputs: torch.Tensor, chunk_size: int = SEQUENCE_LENGTH, state=None):
    if state is None:
        state = zero_state(inputs)
    predictions = []
    for start in range(0, inputs.shape[1], chunk_size):
        prediction, state = model(inputs[:, start:start+chunk_size], state)
        predictions.append(prediction)
    return torch.cat(predictions, dim=1), state


def train_joint(joint: int, groups: dict, args, writer, stream) -> dict:
    started = time.monotonic()
    torch.manual_seed(args.seed + joint)
    train = joint_tensors(groups["train"], joint, args.device)
    validation = joint_tensors(groups["validation"], joint, args.device)
    scored_x, scored_y = train["x"][train["valid"]], train["y"][train["valid"]]
    mean, std = scored_x.mean(0), scored_x.std(0, unbiased=False)
    target_mean, target_std = scored_y.mean(), scored_y.std(unbiased=False)
    if bool((std <= 0).any()) or float(target_std) <= 0:
        raise ValueError(f"J{joint+1}: constant training feature or label")
    model = ActuatorLSTM(mean, std, target_mean, target_std).to(args.device)
    motion = train["motion"][train["valid"]].cpu().numpy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_score, best_epoch, stale = math.inf, 0, 0
    checkpoint_path = args.output_dir / "checkpoints" / f"j{joint+1}_best.pt"
    curve = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        squared_sum = torch.zeros((), device=args.device)
        order = torch.randperm(len(groups["train"]), device=args.device)
        for indices in order.split(args.batch_size):
            x, y = train["x"][indices], train["y"][indices]
            valid = train["valid"][indices]
            state = zero_state(x)
            for start in range(0, x.shape[1], SEQUENCE_LENGTH):
                selection = slice(start, start + SEQUENCE_LENGTH)
                optimizer.zero_grad(set_to_none=True)
                prediction, state = model(x[:, selection], state)
                residual = ((prediction - y[:, selection]) / target_std).squeeze(-1)
                batch_squared_sum = residual[valid[:, selection]].square().sum()
                loss = batch_squared_sum / valid[:, selection].sum()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                # Carry numeric memory through the whole trajectory, but not its gradient graph.
                state = tuple(value.detach() for value in state)
                squared_sum += batch_squared_sum.detach()
        model.eval()
        predicted, _ = rollout(model, validation["x"])
        error = (predicted - validation["y"]).squeeze(-1).square()
        val_score = float(error[validation["valid"]].mean())
        train_score = float(squared_sum / train["valid"].sum()) * float(target_std)**2
        if not math.isfinite(val_score + train_score):
            raise FloatingPointError(f"J{joint+1}: non-finite loss at epoch {epoch}")
        row = {"joint": joint+1, "epoch": epoch, "train_rmse_nm": math.sqrt(train_score),
               "validation_rmse_nm": math.sqrt(val_score)}
        curve.append(row)
        writer.writerow(row)
        stream.flush()
        if val_score < best_score:
            best_score, best_epoch, stale = val_score, epoch, 0
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "best_epoch": best_epoch, "validation_mse_nm2": best_score,
                        "num_layers": LAYERS, "hidden_size": HIDDEN_SIZE,
                        "input_names": ["sent_target_minus_position_rad", "joint_velocity_rad_s"]}, checkpoint_path)
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or stale == PATIENCE:
            print(f"J{joint+1} epoch {epoch}: train={math.sqrt(train_score):.5f} val={math.sqrt(val_score):.5f} Nm; best={best_epoch}", flush=True)
        if stale == PATIENCE:
            break
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["epoch"] for r in curve], [r["train_rmse_nm"] for r in curve], label="train (epoch online)")
    ax.plot([r["epoch"] for r in curve], [r["validation_rmse_nm"] for r in curve], label="validation")
    ax.axvline(best_epoch, color="grey", linestyle="--", label="selected epoch")
    ax.set(xlabel="Epoch", ylabel="RMSE (estimated Nm)", title=f"J{joint+1} LSTM")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "plots" / f"j{joint+1}_learning.png", dpi=130)
    plt.close(fig)
    print(f"J{joint+1} selection complete: best epoch={best_epoch}, elapsed={time.monotonic()-started:.1f}s", flush=True)
    return {"best_epoch": best_epoch, "epochs_run": epoch, "validation_mse_nm2": best_score,
            "training_motion_fraction": float(motion.mean()), "training_seconds": time.monotonic()-started,
            "normalization": {"input_mean": mean.cpu().tolist(), "input_std": std.cpu().tolist(),
                              "target_mean_nm": float(target_mean), "target_std_nm": float(target_std)}}


@torch.inference_mode()
def export_and_evaluate(joint: int, groups: dict, args) -> tuple[dict, dict]:
    checkpoint = torch.load(args.output_dir / "checkpoints" / f"j{joint+1}_best.pt", map_location="cpu", weights_only=True)
    state_dict = checkpoint["state_dict"]
    model = ActuatorLSTM(*(state_dict[k] for k in ("input_mean", "input_std", "target_mean", "target_std")))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    script_path = args.output_dir / "models" / f"j{joint+1}.pt"
    torch.jit.script(model).save(str(script_path))
    loaded = torch.jit.load(str(script_path), map_location=args.device).eval()
    model.to(args.device)
    # These are the exact structural queries made by Isaac Lab ActuatorNetLSTM.
    layers = len(loaded.lstm.state_dict()) // 4
    hidden = loaded.lstm.state_dict()["weight_hh_l0"].shape[1]
    if (layers, hidden) != (LAYERS, HIDDEN_SIZE):
        raise ValueError(f"J{joint+1}: incompatible exported LSTM structure")
    records = groups["validation"] + groups["test"]
    tensors = joint_tensors(records, joint, args.device)
    inputs = tensors["x"]
    block_prediction, block_state = rollout(model, inputs)
    native_state, script_state = zero_state(inputs), zero_state(inputs)
    scripted_prediction = torch.empty_like(block_prediction)
    output_difference = inputs.new_zeros(())
    state_difference = inputs.new_zeros(())
    for step in range(inputs.shape[1]):
        x = inputs[:, step:step+1]
        native, native_state = model(x, native_state)
        scripted, script_state = loaded(x, script_state)
        scripted_prediction[:, step:step+1] = scripted
        output_difference = torch.maximum(output_difference, (native-scripted).abs().max())
        for expected, actual in zip(native_state, script_state):
            state_difference = torch.maximum(state_difference, (expected-actual).abs().max())
    block_difference = float((block_prediction-scripted_prediction)[tensors["valid"]].abs().max())
    block_state_difference = max(float((a-b).abs().max()) for a, b in zip(block_state, script_state))
    # Reset a previously used state before replaying another real trajectory independently.
    last_input = inputs[-1:, :len(records[-1]["x"])]
    reset_state = tuple(value[:, :1].clone().zero_() for value in script_state)
    reset_prediction, _ = rollout(loaded, last_input, state=reset_state)
    reset_difference = float((reset_prediction-scripted_prediction[-1:, :last_input.shape[1]]).abs().max())
    checks = {"torchscript_max_abs_difference_nm": float(output_difference),
              "torchscript_state_max_abs_difference": float(state_difference),
              "chunk_vs_step_max_abs_difference_nm": block_difference,
              "chunk_vs_step_final_state_max_abs_difference": block_state_difference,
              "reset_replay_max_abs_difference_nm": reset_difference,
              "checked_trajectories": len(records), "num_layers": layers, "hidden_size": hidden}
    if max(checks[k] for k in ("torchscript_max_abs_difference_nm", "torchscript_state_max_abs_difference",
                              "chunk_vs_step_max_abs_difference_nm", "chunk_vs_step_final_state_max_abs_difference",
                              "reset_replay_max_abs_difference_nm")) > TOLERANCE:
        raise ValueError(f"J{joint+1}: recurrent inference mismatch: {checks}")
    predicted = scripted_prediction.squeeze(-1).cpu().numpy()
    predictions = {record["trajectory"]: predicted[i, record["warmup"]:len(record["x"])]
                   for i, record in enumerate(records)}
    checks["torchscript_file"] = str(script_path)
    print(f"J{joint+1} export checked: script={float(output_difference):.3g}, chunk/step={block_difference:.3g} Nm", flush=True)
    return checks, predictions


def baseline_predictions(record: dict, joint: int, reference: dict) -> dict:
    normalization = reference["normalization"]
    x = (record["mlp_inputs"][:, joint].astype(np.float64) - normalization["input_mean"]) / normalization["input_std"]
    coefficients = np.asarray(reference["baselines"]["linear_normalized_input_coefficients"])
    return {"mlp": record["mlp_prediction"][:, joint],
            "constant": np.full(len(x), reference["baselines"]["constant_nm"]),
            "linear": x @ coefficients[:-1] + coefficients[-1]}


def evaluate_metrics(groups: dict, predictions: dict, mlp_report: dict, report: dict) -> list:
    rows = []
    for joint in range(6):
        reference = mlp_report["joints"][f"j{joint+1}"]
        result = report["joints"][f"j{joint+1}"]
        for split in ("validation", "test"):
            targets = np.concatenate([r["y"][:, joint] for r in groups[split]])
            motion = np.concatenate([r["motion"][:, joint] for r in groups[split]])
            p = {name: [] for name in ("lstm", "mlp", "constant", "linear")}
            for record in groups[split]:
                p["lstm"].append(predictions[record["trajectory"]][:, joint])
                for name, values in baseline_predictions(record, joint, reference).items():
                    p[name].append(values)
            result[split] = {name: metrics(targets, np.concatenate(values), motion) for name, values in p.items()}
            for name, scores in result[split].items():
                for regime in ("overall", "motion", "hold"):
                    rows.append({"split": split, "joint": joint+1, "model": name, "regime": regime,
                                 **scores[regime]})
        result["test_better_than_reference"] = {
            name: result["test"]["lstm"]["overall"]["rmse_nm"] < result["test"][name]["overall"]["rmse_nm"]
            for name in ("mlp", "constant", "linear")}
    return rows


def write_rows(path: Path, rows: list) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_heldout(output: Path, groups: dict, predictions: dict, mlp_report: dict) -> None:
    rows = []
    for split in ("validation", "test"):
        for record in groups[split]:
            name = record["trajectory"]
            predicted = predictions[name]
            np.savez_compressed(output / "predictions" / f"{name}.npz",
                                time_monotonic_ns=record["time_monotonic_ns"], torque_est_nm=record["y"],
                                lstm_predicted_torque_nm=predicted, mlp_predicted_torque_nm=record["mlp_prediction"],
                                motion=record["motion"])
            time_s = (record["time_monotonic_ns"] - record["time_monotonic_ns"][0]) * 1e-9
            fig, axes = plt.subplots(6, 1, figsize=(11, 12), sharex=True)
            for joint, ax in enumerate(axes):
                values = {"lstm": predicted[:, joint], **baseline_predictions(record, joint, mlp_report["joints"][f"j{joint+1}"])}
                for model_name, p in values.items():
                    scores = metrics(record["y"][:, joint], p, record["motion"][:, joint])
                    for regime in ("overall", "motion", "hold"):
                        rows.append({"trajectory": name, "split": split, "joint": joint+1,
                                     "model": model_name, "regime": regime, **scores[regime]})
                ax.plot(time_s, record["y"][:, joint], color="grey", alpha=.7, linewidth=.7, label="SDK torque estimate")
                ax.plot(time_s, record["mlp_prediction"][:, joint], color="tab:orange", alpha=.8, linewidth=.8, label="MLP")
                ax.plot(time_s, predicted[:, joint], color="tab:blue", linewidth=.8, label="LSTM")
                ax.set_ylabel(f"J{joint+1} (Nm)")
                ax.grid(alpha=.2)
            axes[0].legend()
            axes[0].set_title(f"{name} ({split}) — estimated torque labels")
            axes[-1].set_xlabel("Time from first scored sample (s)")
            fig.tight_layout()
            fig.savefig(output / "plots" / f"{name}.png", dpi=120)
            plt.close(fig)
    write_rows(output / "trajectory_metrics.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--mlp_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8, help="Number of whole trajectories per batch")
    args = parser.parse_args()
    for name in ("dataset_dir", "mlp_dir", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch_size must be positive")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads((args.dataset_dir / "manifest.json").read_text())
    mlp_report = json.loads((args.mlp_dir / "metrics.json").read_text())
    if manifest["outcome"] != "complete" or mlp_report["outcome"] != "complete":
        raise ValueError("Dataset and reference MLP must be complete")
    validate_torque_coefficients(manifest)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in ("models", "checkpoints", "plots", "predictions"):
        (args.output_dir / name).mkdir()
    report = {"outcome": "loading", "dataset_dir": str(args.dataset_dir), "reference_mlp_dir": str(args.mlp_dir),
              "device": args.device, "torch_version": torch.__version__, "joint_seeds": list(range(args.seed, args.seed+6)),
              "training": {"num_layers": LAYERS, "hidden_size": HIDDEN_SIZE, "bidirectional": False,
                           "dropout": 0.0, "batch_first": True, "dtype": "float32", "tf32": False,
                           "sequence_length": SEQUENCE_LENGTH, "trajectory_batch_size": args.batch_size,
                           "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4,
                           "gradient_max_norm": 1.0, "maximum_epochs": args.epochs,
                           "early_stopping_patience": PATIENCE,
                           "loss": "ordinary MSE on valid samples; no motion/hold class weights",
                           "selection": "minimum unweighted validation MSE",
                           "state": "zero at each trajectory start; carry and detach between training chunks",
                           "warmup": "prefix inputs update state; prefix and padding excluded from loss and metrics"},
              "dataset_duration_s": manifest["duration_s"],
              "samples_by_split": manifest["samples_by_split"], "torque_label": manifest["torque_label"],
              "torque_coefficients_nm_per_a": manifest["torque_coefficients_nm_per_a"],
              "firmware": manifest["firmware"], "torque_firmware_source": manifest["torque_firmware_source"],
              "notes": "Predictions use recorded joint states. Labels are SDK current-derived estimates, not calibrated output torque or validated simulated dynamics.",
              "joints": {}}
    write_json(args.output_dir / "metrics.json", report)
    started = time.monotonic()
    try:
        groups, acceptance = load_trajectories(args.dataset_dir, manifest, args.mlp_dir, mlp_report)
        write_json(args.output_dir / "dataset_acceptance.json", acceptance)
        print(f"Data accepted: {acceptance['samples_by_split']}; scored arrays match MLP exactly", flush=True)
        report["outcome"] = "training"
        with (args.output_dir / "learning_curves.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["joint", "epoch", "train_rmse_nm", "validation_rmse_nm"])
            writer.writeheader()
            for joint in range(6):
                report["joints"][f"j{joint+1}"] = train_joint(joint, groups, args, writer, stream)
                write_json(args.output_dir / "metrics.json", report)
        # Test evaluation starts only after all six best checkpoints are selected.
        report["outcome"] = "evaluating"
        write_json(args.output_dir / "metrics.json", report)
        predictions = {r["trajectory"]: [] for split in ("validation", "test") for r in groups[split]}
        for joint in range(6):
            checks, values = export_and_evaluate(joint, groups, args)
            report["joints"][f"j{joint+1}"]["export_validation"] = checks
            for name, value in values.items():
                predictions[name].append(value)
            write_json(args.output_dir / "metrics.json", report)
        predictions = {name: np.column_stack(values) for name, values in predictions.items()}
        write_rows(args.output_dir / "comparison.csv", evaluate_metrics(groups, predictions, mlp_report, report))
        write_json(args.output_dir / "metrics.json", report)
        save_heldout(args.output_dir, groups, predictions, mlp_report)
        config = {"class": "isaaclab.actuators.ActuatorNetLSTMCfg", "required_call_period_s": manifest["sample_period_s"],
                  "input_names": manifest["lstm_input_names"], "input_shape": ["batch", 1, 2],
                  "output_shape": ["batch", 1, 1], "state_shape": [LAYERS, "batch", HIDDEN_SIZE],
                  "state_reset": "Set both hidden and cell states to zero at each episode start; retain across 5 ms calls",
                  "normalization": "Embedded in TorchScript; rad and rad/s input, SDK-estimated Nm output; no external scales",
                  "control": manifest["control"], "torque_coefficients_nm_per_a": manifest["torque_coefficients_nm_per_a"],
                  "firmware": manifest["firmware"], "torque_firmware_source": manifest["torque_firmware_source"],
                  "actuators": {f"j{j}": {"joint_names_expr": [f"arm_j{j}"], "network_file": str(args.output_dir / "models" / f"j{j}.pt")}
                                for j in range(1, 7)},
                  "integration": "Network parameters only. Physical effort/velocity/saturation limits and simulation rollout validation are deferred."}
        write_json(args.output_dir / "actuator_network_config.json", config)
        report["outcome"] = "complete"
    except BaseException as exc:
        report["outcome"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(args.output_dir / "metrics.json", report)
    print(f"Training complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
