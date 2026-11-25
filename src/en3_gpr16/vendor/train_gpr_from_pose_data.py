import argparse
import csv
import json
import time
import threading
import warnings
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
AXIS_COUNT = 4


def build_feature_names():
    feature_groups = (
        "q_act_current",
        "q_act_prev",
        "du_k",
        "du_prev",
    )
    return [f"{group}_{axis + 1}" for group in feature_groups for axis in range(AXIS_COUNT)]


FEATURE_NAMES = build_feature_names()


def parse_span_steps(span_steps_arg):
    if isinstance(span_steps_arg, (list, tuple)):
        values = span_steps_arg
    else:
        values = str(span_steps_arg).split(",")

    span_steps = []
    for value in values:
        step = int(str(value).strip())
        if step <= 0:
            raise ValueError(f"Invalid span step: {step}. Span steps must be positive integers.")
        if step not in span_steps:
            span_steps.append(step)

    if not span_steps:
        raise ValueError("At least one span step must be provided.")
    return sorted(span_steps)


def parse_kernel_mode(kernel_mode_arg):
    kernel_mode = str(kernel_mode_arg).strip().lower()
    valid_modes = {"auto", "ard", "shared"}
    if kernel_mode not in valid_modes:
        raise ValueError(f"Invalid kernel mode: {kernel_mode_arg}. Expected one of {sorted(valid_modes)}.")
    return kernel_mode


def parse_error_thresholds(error_thresholds_arg):
    if error_thresholds_arg is None:
        return []

    if isinstance(error_thresholds_arg, (list, tuple)):
        values = error_thresholds_arg
    else:
        values = str(error_thresholds_arg).split(",")

    thresholds = []
    for value in values:
        threshold = float(str(value).strip())
        if threshold < 0:
            raise ValueError(f"Invalid error threshold: {threshold}. Thresholds must be non-negative.")
        if threshold not in thresholds:
            thresholds.append(threshold)
    return sorted(thresholds)


def parse_example_selection_mode(selection_mode_arg):
    selection_mode = str(selection_mode_arg).strip().lower()
    valid_modes = {"best", "worst", "random"}
    if selection_mode not in valid_modes:
        raise ValueError(
            f"Invalid example selection mode: {selection_mode_arg}. Expected one of {sorted(valid_modes)}."
        )
    return selection_mode


def parse_split_mode(split_mode_arg):
    split_mode = str(split_mode_arg).strip().lower()
    valid_modes = {"chronological", "random"}
    if split_mode not in valid_modes:
        raise ValueError(f"Invalid split mode: {split_mode_arg}. Expected one of {sorted(valid_modes)}.")
    return split_mode


def parse_subsample_mode(subsample_mode_arg):
    subsample_mode = str(subsample_mode_arg).strip().lower()
    valid_modes = {"diverse", "uniform"}
    if subsample_mode not in valid_modes:
        raise ValueError(
            f"Invalid subsample mode: {subsample_mode_arg}. Expected one of {sorted(valid_modes)}."
        )
    return subsample_mode


def has_direction_reversal(command_window, reversal_eps):
    if len(command_window) < 3:
        return False

    step_increments = np.diff(command_window, axis=0)
    has_positive = np.any(step_increments > reversal_eps, axis=0)
    has_negative = np.any(step_increments < -reversal_eps, axis=0)
    return bool(np.any(has_positive & has_negative))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def maxae(y_true, y_pred):
    return float(np.max(np.abs(y_true - y_pred)))


def medianae(y_true, y_pred):
    return float(np.median(np.abs(y_true - y_pred)))


def p90ae(y_true, y_pred):
    return float(np.percentile(np.abs(y_true - y_pred), 90))


def normalized_scale(y_true):
    y_true = np.asarray(y_true, dtype=float)
    scale = float(np.max(y_true) - np.min(y_true))
    if scale < 1e-12:
        scale = float(np.std(y_true))
    if scale < 1e-12:
        scale = 1.0
    return scale


def normalized_scales_per_axis(y_true):
    y_true = np.asarray(y_true, dtype=float)
    if y_true.ndim == 1:
        return np.asarray([normalized_scale(y_true)], dtype=float)
    return np.asarray([normalized_scale(y_true[:, axis]) for axis in range(y_true.shape[1])], dtype=float)


def nmae(y_true, y_pred):
    return float(mae(y_true, y_pred) / normalized_scale(y_true))


def nrmse(y_true, y_pred):
    return float(rmse(y_true, y_pred) / normalized_scale(y_true))


def axis_balanced_nmae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    scales = normalized_scales_per_axis(y_true)
    normalized_abs_error = np.abs(y_true - y_pred) / scales
    return float(np.mean(normalized_abs_error))


def axis_balanced_nrmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    scales = normalized_scales_per_axis(y_true)
    normalized_sq_error = ((y_true - y_pred) / scales) ** 2
    return float(np.sqrt(np.mean(normalized_sq_error)))


def r2(y_true, y_pred):
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true, axis=0)) ** 2))
    if sst < 1e-12:
        return 1.0
    return 1.0 - sse / sst


def coverage_rate(y_true, y_pred, y_std, sigma_level):
    if y_std is None:
        return None
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_std = np.asarray(y_std, dtype=float)
    within = np.abs(y_true - y_pred) <= sigma_level * y_std
    return float(np.mean(within))


def threshold_hit_rate(y_true, y_pred, threshold):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    threshold = float(threshold)
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}.")
    within = np.abs(y_true - y_pred) <= threshold
    return float(np.mean(within))


def compute_std_calibration_scale(y_true, y_pred, y_std, min_std=1e-9):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_std = np.asarray(y_std, dtype=float)
    y_std_safe = np.maximum(y_std, min_std)
    normalized_residual = (y_true - y_pred) / y_std_safe
    scale = np.sqrt(np.mean(normalized_residual ** 2, axis=0))
    return np.maximum(scale, 1.0)


def apply_std_calibration(y_std, scale_factors):
    y_std = np.asarray(y_std, dtype=float)
    scale_factors = np.asarray(scale_factors, dtype=float)
    return y_std * scale_factors


def make_metric_block(y_true, y_pred, y_std=None, error_thresholds=None):
    error_thresholds = [] if error_thresholds is None else list(error_thresholds)
    per_axis = []
    for axis in range(y_true.shape[1]):
        axis_std = None if y_std is None else y_std[:, axis]
        metric_row = {
            "axis": axis + 1,
            "mae": mae(y_true[:, axis], y_pred[:, axis]),
            "rmse": rmse(y_true[:, axis], y_pred[:, axis]),
            "maxae": maxae(y_true[:, axis], y_pred[:, axis]),
            "medianae": medianae(y_true[:, axis], y_pred[:, axis]),
            "p90ae": p90ae(y_true[:, axis], y_pred[:, axis]),
            "nmae": nmae(y_true[:, axis], y_pred[:, axis]),
            "nrmse": nrmse(y_true[:, axis], y_pred[:, axis]),
            "r2": r2(y_true[:, axis], y_pred[:, axis]),
        }
        for error_threshold in error_thresholds:
            metric_row[f"within_{float(error_threshold):g}"] = threshold_hit_rate(
                y_true[:, axis],
                y_pred[:, axis],
                error_threshold,
            )
        if axis_std is not None:
            metric_row["coverage_1sigma"] = coverage_rate(y_true[:, axis], y_pred[:, axis], axis_std, 1.0)
            metric_row["coverage_2sigma"] = coverage_rate(y_true[:, axis], y_pred[:, axis], axis_std, 2.0)
        per_axis.append(metric_row)

    overall = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "maxae": maxae(y_true, y_pred),
        "medianae": medianae(y_true, y_pred),
        "p90ae": p90ae(y_true, y_pred),
        "nmae": nmae(y_true, y_pred),
        "nrmse": nrmse(y_true, y_pred),
        "axis_balanced_nmae": axis_balanced_nmae(y_true, y_pred),
        "axis_balanced_nrmse": axis_balanced_nrmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }
    for error_threshold in error_thresholds:
        overall[f"within_{float(error_threshold):g}"] = threshold_hit_rate(y_true, y_pred, error_threshold)
    if y_std is not None:
        overall["coverage_1sigma"] = coverage_rate(y_true, y_pred, y_std, 1.0)
        overall["coverage_2sigma"] = coverage_rate(y_true, y_pred, y_std, 2.0)

    return {
        "overall": overall,
        "per_axis": per_axis,
    }


def split_samples_train_val_test(samples, train_ratio, val_ratio, split_mode="chronological", seed=42):
    total = len(samples)
    if total < 3:
        raise ValueError("At least 3 samples are required for a train/val/test split.")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}.")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"train_ratio + val_ratio must be less than 1, got {train_ratio + val_ratio}."
        )

    split_mode = parse_split_mode(split_mode)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train_end = max(1, min(total - 2, train_end))
    val_end = max(train_end + 1, min(total - 1, val_end))

    if split_mode == "chronological":
        ordered_samples = list(samples)
    else:
        rng = np.random.default_rng(seed)
        shuffled_indices = rng.permutation(total)
        ordered_samples = [samples[index] for index in shuffled_indices]

    train_samples = ordered_samples[:train_end]
    val_samples = ordered_samples[train_end:val_end]
    test_samples = ordered_samples[val_end:]

    if not train_samples or not val_samples or not test_samples:
        raise ValueError(
            "The requested split produced an empty subset. "
            "Please adjust train_ratio/val_ratio or provide more samples."
        )

    return train_samples, val_samples, test_samples


def build_feature_vector(q_act_current, q_act_prev, du_k, du_prev):
    return np.concatenate(
        [
            q_act_current,
            q_act_prev,
            du_k,
            du_prev,
        ]
    )


def make_sample(sample_k, time_k, q_act_current, q_act_prev, q_act_next, du_k, du_prev, realized_increment, delta_d):
    return {
        "sample_k": sample_k,
        "time_k": time_k,
        "span_step": 1,
        "features": build_feature_vector(q_act_current, q_act_prev, du_k, du_prev),
        "target_delta_d": delta_d,
        "q_act_current": q_act_current,
        "q_act_prev": q_act_prev,
        "q_act_next": q_act_next,
        "du_k": du_k,
        "du_prev": du_prev,
        "realized_increment": realized_increment,
    }


def make_multistep_sample(start_idx, end_idx, time_k, span_step, q_act_current, q_act_prev, q_act_next, du_k, du_prev, realized_increment, delta_d):
    sample = make_sample(
        sample_k=start_idx,
        time_k=time_k,
        q_act_current=q_act_current,
        q_act_prev=q_act_prev,
        q_act_next=q_act_next,
        du_k=du_k,
        du_prev=du_prev,
        realized_increment=realized_increment,
        delta_d=delta_d,
    )
    sample["span_step"] = span_step
    sample["sample_end_k"] = end_idx
    return sample


def format_duration(seconds):
    seconds = int(round(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def render_progress_bar(completed, total, width=24):
    if total <= 0:
        return "[" + ("-" * width) + "]"
    filled = int(width * completed / total)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def estimate_training_progress(completed_axes, total_axes, current_axis_elapsed, completed_axis_durations):
    if total_axes <= 0:
        return 0.0, None

    if completed_axis_durations:
        avg_axis_duration = float(np.mean(completed_axis_durations))
        axis_fraction = min(current_axis_elapsed / max(avg_axis_duration, 1e-9), 0.99)
        overall_fraction = min((completed_axes + axis_fraction) / total_axes, 0.99)
        return overall_fraction * 100.0, axis_fraction * 100.0

    overall_fraction = completed_axes / total_axes
    return overall_fraction * 100.0, None


def fit_estimator_with_heartbeat(
    estimator,
    x_train_scaled,
    y_train_axis,
    axis_label,
    heartbeat_seconds,
    completed_axes,
    total_axes,
    completed_axis_durations,
):
    result = {}
    error = {}

    def run_fit():
        try:
            result["estimator"] = estimator.fit(x_train_scaled, y_train_axis)
        except Exception as exc:
            error["exception"] = exc

    worker = threading.Thread(target=run_fit, daemon=True)
    worker.start()

    axis_start = time.perf_counter()
    next_heartbeat = heartbeat_seconds
    while worker.is_alive():
        worker.join(timeout=heartbeat_seconds)
        if worker.is_alive():
            elapsed = time.perf_counter() - axis_start
            overall_pct, axis_pct = estimate_training_progress(
                completed_axes,
                total_axes,
                elapsed,
                completed_axis_durations,
            )
            if axis_pct is None:
                progress_text = f"overall progress >= {overall_pct:.1f}%"
            else:
                progress_text = f"estimated overall {overall_pct:.1f}%, current axis {axis_pct:.1f}%"
            print(
                f"  {axis_label} still training... elapsed {format_duration(elapsed)} ({progress_text})",
                flush=True,
            )
            next_heartbeat += heartbeat_seconds

    if "exception" in error:
        raise error["exception"]

    return result["estimator"], time.perf_counter() - axis_start


def build_samples(csv_path, step_limit=0.8, span_steps=None, filter_reversals=False, reversal_eps=1e-4):
    span_steps = [1] if span_steps is None else parse_span_steps(span_steps)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])

        raw_pose_fields = {"Act_q1", "Act_q2", "Act_q3", "Act_q4", "X1", "Y1", "X2", "Y2", "Time"}
        prebuilt_fields = {
            "sample_k",
            "time_k",
            "q_act_current_1",
            "q_act_prev_1",
            "du_k_1",
            "du_prev_1",
            "realized_increment_1",
            "target_delta_d_1",
            "q_act_next_1",
        }

        if raw_pose_fields.issubset(fieldnames):
            q_act_list = []
            q_cmd_list = []
            time_list = []

            for row in reader:
                q_act_list.append(
                    np.array(
                        [float(row["Act_q1"]), float(row["Act_q2"]), float(row["Act_q3"]), float(row["Act_q4"])],
                        dtype=float,
                    )
                )
                q_cmd_list.append(
                    np.array(
                        [float(row["X1"]), float(row["Y1"]), float(row["X2"]), float(row["Y2"])],
                        dtype=float,
                    )
                )
                time_list.append(float(row["Time"]))

            q_act = np.asarray(q_act_list, dtype=float)
            q_cmd = np.asarray(q_cmd_list, dtype=float)
            time_arr = np.asarray(time_list, dtype=float)

            samples = []
            for span_step in span_steps:
                for start_idx in range(span_step, len(q_cmd) - span_step):
                    end_idx = start_idx + span_step

                    if np.allclose(q_act[end_idx], q_act[start_idx], atol=1e-6):
                        continue

                    if filter_reversals:
                        current_command_window = q_cmd[start_idx : end_idx + 1]
                        prev_command_window = q_cmd[start_idx - span_step : start_idx + 1]
                        if has_direction_reversal(current_command_window, reversal_eps) or has_direction_reversal(
                            prev_command_window, reversal_eps
                        ):
                            continue

                    du_k = q_cmd[end_idx] - q_cmd[start_idx]
                    du_prev = q_cmd[start_idx] - q_cmd[start_idx - span_step]
                    span_step_limit = step_limit * span_step if step_limit > 0 else float("inf")

                    if np.max(np.abs(du_k)) > span_step_limit or np.max(np.abs(du_prev)) > span_step_limit:
                        continue

                    q_act_current = q_act[start_idx]
                    q_act_prev = q_act[start_idx - span_step]
                    realized_increment = q_act[end_idx] - q_act[start_idx]
                    delta_d = du_k - realized_increment

                    samples.append(
                        make_multistep_sample(
                            start_idx=start_idx,
                            end_idx=end_idx,
                            time_k=time_arr[start_idx],
                            span_step=span_step,
                            q_act_current=q_act_current,
                            q_act_prev=q_act_prev,
                            q_act_next=q_act[end_idx],
                            du_k=du_k,
                            du_prev=du_prev,
                            realized_increment=realized_increment,
                            delta_d=delta_d,
                        )
                    )
            samples.sort(key=lambda sample: (sample["sample_k"], sample.get("sample_end_k", sample["sample_k"]), sample.get("span_step", 1)))
            return samples

        if prebuilt_fields.issubset(fieldnames):
            if filter_reversals:
                print(
                    "Reading a prebuilt dataset CSV. Reversal filtering cannot be re-applied because "
                    "the per-step command trajectory is no longer available.",
                    flush=True,
                )
            samples = []
            for row in reader:
                q_act_current = np.array(
                    [float(row[f"q_act_current_{i + 1}"]) for i in range(AXIS_COUNT)],
                    dtype=float,
                )
                q_act_prev = np.array(
                    [float(row[f"q_act_prev_{i + 1}"]) for i in range(AXIS_COUNT)],
                    dtype=float,
                )
                q_act_next = np.array(
                    [float(row[f"q_act_next_{i + 1}"]) for i in range(AXIS_COUNT)],
                    dtype=float,
                )
                du_k = np.array([float(row[f"du_k_{i + 1}"]) for i in range(AXIS_COUNT)], dtype=float)
                du_prev = np.array([float(row[f"du_prev_{i + 1}"]) for i in range(AXIS_COUNT)], dtype=float)
                realized_increment = np.array(
                    [float(row[f"realized_increment_{i + 1}"]) for i in range(AXIS_COUNT)],
                    dtype=float,
                )
                delta_d = np.array(
                    [float(row[f"target_delta_d_{i + 1}"]) for i in range(AXIS_COUNT)],
                    dtype=float,
                )
                samples.append(
                    make_multistep_sample(
                        start_idx=int(row["sample_k"]),
                        end_idx=int(row.get("sample_end_k", row["sample_k"])),
                        time_k=float(row["time_k"]),
                        span_step=int(row.get("span_step", 1)),
                        q_act_current=q_act_current,
                        q_act_prev=q_act_prev,
                        q_act_next=q_act_next,
                        du_k=du_k,
                        du_prev=du_prev,
                        realized_increment=realized_increment,
                        delta_d=delta_d,
                    )
                )
            samples.sort(key=lambda sample: (sample["sample_k"], sample.get("sample_end_k", sample["sample_k"]), sample.get("span_step", 1)))
            return samples

    raise ValueError(
        f"Unsupported CSV format: {csv_path}. "
        "Expected raw pose columns like Act_q1/X1/Time or a prebuilt sample dataset like gpr_dataset_full.csv."
    )


def resolve_csv_path(csv_arg):
    candidate = Path(csv_arg).expanduser()
    if candidate.is_absolute() and candidate.is_file():
        return candidate.resolve()

    search_candidates = []
    for base_dir in (Path.cwd(), SCRIPT_DIR, SCRIPT_DIR.parent):
        resolved = (base_dir / candidate).resolve()
        if resolved not in search_candidates:
            search_candidates.append(resolved)

    for resolved in search_candidates:
        if resolved.is_file():
            return resolved

    exact_name_matches = []
    for base_dir in (Path.cwd(), SCRIPT_DIR, SCRIPT_DIR.parent):
        try:
            for match in base_dir.rglob(candidate.name):
                if match.is_file():
                    resolved = match.resolve()
                    if resolved not in exact_name_matches:
                        exact_name_matches.append(resolved)
        except OSError:
            continue

    if exact_name_matches:
        return exact_name_matches[0]

    fallback_candidates = []
    if candidate.name == "gpr_pose_data.csv":
        for base_dir in (SCRIPT_DIR, SCRIPT_DIR.parent):
            fallback = (base_dir / "gpr_training_outputs" / "gpr_dataset_full.csv").resolve()
            if fallback not in fallback_candidates:
                fallback_candidates.append(fallback)

        for fallback in fallback_candidates:
            if fallback.is_file():
                print(
                    f"Input CSV '{csv_arg}' was not found. Falling back to prebuilt dataset: {fallback}"
                )
                return fallback

    checked = "\n".join(f"  - {path}" for path in search_candidates)
    if fallback_candidates:
        checked += "\n" + "\n".join(f"  - {path}" for path in fallback_candidates)
    raise FileNotFoundError(
        f"Input CSV not found: {csv_arg}\n"
        f"Checked these paths:\n{checked}\n"
        "Pass the correct file with --csv <path-to-gpr_pose_data.csv>."
    )


def write_dataset_csv(samples, path):
    feature_count = len(samples[0]["features"]) if samples else len(FEATURE_NAMES)
    header = (
        ["sample_k", "sample_end_k", "span_step", "time_k"]
        + [f"feature_{i+1}" for i in range(feature_count)]
        + [f"q_act_current_{i+1}" for i in range(AXIS_COUNT)]
        + [f"q_act_prev_{i+1}" for i in range(AXIS_COUNT)]
        + [f"du_k_{i+1}" for i in range(AXIS_COUNT)]
        + [f"du_prev_{i+1}" for i in range(AXIS_COUNT)]
        + [f"realized_increment_{i+1}" for i in range(AXIS_COUNT)]
        + [f"target_delta_d_{i+1}" for i in range(AXIS_COUNT)]
        + [f"q_act_next_{i+1}" for i in range(AXIS_COUNT)]
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for sample in samples:
            row = [
                sample["sample_k"],
                sample.get("sample_end_k", sample["sample_k"]),
                sample.get("span_step", 1),
                f'{sample["time_k"]:.6f}',
            ]
            row += [f"{x:.10f}" for x in sample["features"]]
            row += [f"{x:.10f}" for x in sample["q_act_current"]]
            row += [f"{x:.10f}" for x in sample["q_act_prev"]]
            row += [f"{x:.10f}" for x in sample["du_k"]]
            row += [f"{x:.10f}" for x in sample["du_prev"]]
            row += [f"{x:.10f}" for x in sample["realized_increment"]]
            row += [f"{x:.10f}" for x in sample["target_delta_d"]]
            row += [f"{x:.10f}" for x in sample["q_act_next"]]
            writer.writerow(row)


def build_sklearn_gpr(seed, n_features, n_restarts, kernel_mode):
    kernel_mode = parse_kernel_mode(kernel_mode)
    if kernel_mode == "ard":
        length_scale = np.ones(n_features)
    else:
        length_scale = 1.0

    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * Matern(length_scale=length_scale, length_scale_bounds=(1e-2, 1e3), nu=1.5)
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
    )
    base_model = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        normalize_y=True,
        random_state=seed,
    )
    return base_model


def select_training_subset_indices(x_train, max_train_samples, seed, subsample_mode):
    total = len(x_train)
    if max_train_samples <= 0 or total <= max_train_samples:
        return np.arange(total, dtype=int)

    subsample_mode = parse_subsample_mode(subsample_mode)
    if subsample_mode == "uniform":
        return np.linspace(0, total - 1, max_train_samples, dtype=int)

    # Prioritize coverage of the motion space using current pose and current command increment.
    coverage_features = x_train[:, np.r_[0:AXIS_COUNT, 2 * AXIS_COUNT:3 * AXIS_COUNT]]
    coverage_mean = np.mean(coverage_features, axis=0)
    coverage_std = np.std(coverage_features, axis=0)
    coverage_std = np.where(coverage_std < 1e-12, 1.0, coverage_std)
    coverage_features = (coverage_features - coverage_mean) / coverage_std

    rng = np.random.default_rng(seed)
    selected_indices = np.empty(max_train_samples, dtype=int)
    first_index = int(rng.integers(total))
    selected_indices[0] = first_index

    diff = coverage_features - coverage_features[first_index]
    min_dist_sq = np.sum(diff * diff, axis=1)
    min_dist_sq[first_index] = -1.0

    for i in range(1, max_train_samples):
        next_index = int(np.argmax(min_dist_sq))
        selected_indices[i] = next_index
        diff = coverage_features - coverage_features[next_index]
        dist_sq = np.sum(diff * diff, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)
        min_dist_sq[selected_indices[: i + 1]] = -1.0

    return np.sort(selected_indices)


def train_gpr_model(
    x_train,
    y_train,
    seed,
    n_restarts,
    heartbeat_seconds,
    kernel_mode,
    max_train_samples,
    subsample_mode,
):
    kernel_mode = parse_kernel_mode(kernel_mode)
    x_train_used = x_train
    y_train_used = y_train

    if max_train_samples > 0 and len(x_train) > max_train_samples:
        sample_indices = select_training_subset_indices(
            x_train,
            max_train_samples=max_train_samples,
            seed=seed,
            subsample_mode=subsample_mode,
        )
        x_train_used = x_train[sample_indices]
        y_train_used = y_train[sample_indices]
        print(
            f"Subsampled training set from {len(x_train)} to {len(x_train_used)} samples "
            f"using {parse_subsample_mode(subsample_mode)} coverage sampling to limit GPR memory usage.",
            flush=True,
        )

    effective_kernel_mode = kernel_mode
    if kernel_mode == "auto":
        effective_kernel_mode = "shared" if len(x_train_used) > 2000 else "ard"

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_used)
    base_model = build_sklearn_gpr(seed, x_train_used.shape[1], n_restarts, effective_kernel_mode)
    estimators = []
    completed_axis_durations = []
    total_outputs = y_train_used.shape[1]
    training_start = time.perf_counter()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        print(
            f"Training GPR models for {total_outputs} output axes "
            f"with {x_train_used.shape[0]} samples and {x_train_used.shape[1]} features "
            f"(n_restarts={n_restarts}, kernel_mode={effective_kernel_mode})..."
        )
        for output_axis in range(total_outputs):
            overall_pct = 100.0 * output_axis / max(total_outputs, 1)
            bar = render_progress_bar(output_axis, total_outputs)
            axis_label = f"axis {output_axis + 1}/{total_outputs}"
            print(
                f"{bar} {axis_label} starting "
                f"({overall_pct:.1f}%) "
                f"(elapsed {format_duration(time.perf_counter() - training_start)})",
                flush=True,
            )
            estimator = clone(base_model)
            estimator, axis_elapsed = fit_estimator_with_heartbeat(
                estimator,
                x_train_scaled,
                y_train_used[:, output_axis],
                axis_label=axis_label,
                heartbeat_seconds=heartbeat_seconds,
                completed_axes=output_axis,
                total_axes=total_outputs,
                completed_axis_durations=completed_axis_durations,
            )
            estimators.append(estimator)
            completed_axis_durations.append(axis_elapsed)
            total_elapsed = time.perf_counter() - training_start
            overall_pct = 100.0 * (output_axis + 1) / max(total_outputs, 1)
            bar = render_progress_bar(output_axis + 1, total_outputs)
            print(
                f"{bar} {axis_label} finished "
                f"({overall_pct:.1f}%) "
                f"in {format_duration(axis_elapsed)} "
                f"(total {format_duration(total_elapsed)})",
                flush=True,
            )

    return SimpleNamespace(estimators_=estimators), scaler


def predict_with_uncertainty(model, scaler, x_query):
    x_query_scaled = scaler.transform(x_query)
    n_samples = x_query_scaled.shape[0]
    n_outputs = len(model.estimators_)

    pred_mean = np.zeros((n_samples, n_outputs), dtype=float)
    pred_std = np.zeros((n_samples, n_outputs), dtype=float)

    for output_axis, estimator in enumerate(model.estimators_):
        mean_axis, std_axis = estimator.predict(x_query_scaled, return_std=True)
        pred_mean[:, output_axis] = mean_axis
        pred_std[:, output_axis] = std_axis

    return pred_mean, pred_std


def extract_model_parameters(model, scaler):
    model_params = []
    for output_axis, estimator in enumerate(model.estimators_):
        model_params.append(
            {
                "output_axis": output_axis + 1,
                "kernel": str(estimator.kernel_),
                "log_marginal_likelihood": float(estimator.log_marginal_likelihood_value_),
            }
        )

    scaler_info = {
        "feature_mean": scaler.mean_.tolist(),
        "feature_scale": scaler.scale_.tolist(),
        "feature_names": FEATURE_NAMES,
    }
    return model_params, scaler_info


def write_prediction_csv(test_samples, pred_delta_d, pred_std, output_path):
    header = (
        ["sample_k", "time_k"]
        + [f"true_delta_d_{i+1}" for i in range(4)]
        + [f"pred_delta_d_{i+1}" for i in range(4)]
        + [f"pred_std_delta_d_{i+1}" for i in range(4)]
        + [f"err_delta_d_{i+1}" for i in range(4)]
        + [f"true_realized_increment_{i+1}" for i in range(4)]
        + [f"pred_realized_increment_{i+1}" for i in range(4)]
        + [f"err_realized_increment_{i+1}" for i in range(4)]
        + [f"true_next_q_{i+1}" for i in range(4)]
        + [f"pred_next_q_{i+1}" for i in range(4)]
        + [f"err_next_q_{i+1}" for i in range(4)]
    )
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, sample in enumerate(test_samples):
            true_d = sample["target_delta_d"]
            true_realized = sample["realized_increment"]
            pred_d = pred_delta_d[i]
            pred_realized = sample["du_k"] - pred_d
            err_realized = pred_realized - true_realized
            pred_next_q = sample["q_act_current"] + pred_realized
            err_next_q = pred_next_q - sample["q_act_next"]

            row = [sample["sample_k"], f'{sample["time_k"]:.6f}']
            row += [f"{x:.10f}" for x in true_d]
            row += [f"{x:.10f}" for x in pred_d]
            row += [f"{x:.10f}" for x in pred_std[i]]
            row += [f"{x:.10f}" for x in (pred_d - true_d)]
            row += [f"{x:.10f}" for x in true_realized]
            row += [f"{x:.10f}" for x in pred_realized]
            row += [f"{x:.10f}" for x in err_realized]
            row += [f"{x:.10f}" for x in sample["q_act_next"]]
            row += [f"{x:.10f}" for x in pred_next_q]
            row += [f"{x:.10f}" for x in err_next_q]
            writer.writerow(row)


def load_model_bundle(model_path):
    model_bundle = joblib.load(model_path)
    if not isinstance(model_bundle, (list, tuple)) or len(model_bundle) != 2:
        raise ValueError(f"Unsupported model bundle format in {model_path}. Expected [model, scaler].")
    return model_bundle[0], model_bundle[1]


def format_vector(values, precision=6):
    return "[" + ", ".join(f"{value:.{precision}f}" for value in values) + "]"


def select_prediction_examples(samples, pred_delta_d, limit=0, mode="best", seed=42):
    total = len(samples)
    if limit <= 0 or limit >= total:
        return samples, pred_delta_d

    mode = parse_example_selection_mode(mode)

    scored_indices = []
    for i, sample in enumerate(samples):
        err = pred_delta_d[i] - sample["target_delta_d"]
        score = float(np.linalg.norm(err))
        scored_indices.append((score, i))

    if mode == "best":
        scored_indices.sort(key=lambda item: (item[0], item[1]))
        selected_indices = [index for _, index in scored_indices[:limit]]
    elif mode == "worst":
        scored_indices.sort(key=lambda item: (-item[0], item[1]))
        selected_indices = [index for _, index in scored_indices[:limit]]
    else:
        rng = np.random.default_rng(seed)
        selected_indices = rng.choice(total, size=limit, replace=False).tolist()

    selected_samples = [samples[index] for index in selected_indices]
    selected_pred_delta_d = pred_delta_d[selected_indices]
    return selected_samples, selected_pred_delta_d


def print_prediction_rows(test_samples, pred_delta_d, limit=0, selection_mode="best", seed=42):
    selected_samples, selected_pred_delta_d = select_prediction_examples(
        test_samples,
        pred_delta_d,
        limit=limit,
        mode=selection_mode,
        seed=seed,
    )
    total = len(test_samples)
    rows_to_print = len(selected_samples)
    print(f"Prediction rows shown: {rows_to_print}/{total} ({selection_mode} overall delta_d error)")
    for i in range(rows_to_print):
        sample = selected_samples[i]
        pred_d = selected_pred_delta_d[i]
        true_delta_d = sample["target_delta_d"]
        true_realized = sample["realized_increment"]
        pred_realized = sample["du_k"] - pred_d
        true_next_q = sample["q_act_next"]
        pred_next_q = sample["q_act_current"] + pred_realized

        print(f"Sample {sample['sample_k']} @ time {sample['time_k']:.6f}")
        print(f"  span_step:         {sample.get('span_step', 1)}")
        print(f"  true_delta_d:      {format_vector(true_delta_d)}")
        print(f"  pred_delta_d:      {format_vector(pred_d)}")
        print(f"  true_realized_inc: {format_vector(true_realized)}")
        print(f"  pred_realized_inc: {format_vector(pred_realized)}")
        print(f"  true_next_q:       {format_vector(true_next_q)}")
        print(f"  pred_next_q:       {format_vector(pred_next_q)}")


def save_report(
    report_path,
    metadata,
    model_params,
    metric_delta_d,
    metric_realized,
    metric_next_q,
    pkl_path,
    calibrated_metric_delta_d=None,
):
    lines = [
        "# GPR Training and Test Summary",
        "",
        f"- Source data: `{metadata['source_csv']}`",
        f"- Raw pose rows: {metadata['raw_rows']}",
        f"- Valid time-series samples: {metadata['valid_samples']}",
        f"- Train / val / test split: {metadata['train_size']} / {metadata['val_size']} / {metadata['test_size']}",
        f"- Split ratio (train / val / test): {metadata['train_ratio']:.2f} / {metadata['val_ratio']:.2f} / {metadata['test_ratio']:.2f}",
        f"- Split strategy: {metadata['split_strategy']}",
        f"- Saved model bundle: `{pkl_path}`",
        "",
        "## Trained Kernels",
        "",
    ]
    for item in model_params:
        lines.append(
            f"- Output {item['output_axis']}: kernel={item['kernel']}, "
            f"log_marginal_likelihood={item['log_marginal_likelihood']:.6f}"
        )
    lines += [
        "",
        "## Test Metrics: Delta d",
        "",
        f"- Overall MAE: {metric_delta_d['overall']['mae']:.6f}",
        f"- Overall RMSE: {metric_delta_d['overall']['rmse']:.6f}",
        f"- Overall MaxAE: {metric_delta_d['overall']['maxae']:.6f}",
        f"- Overall MedianAE: {metric_delta_d['overall']['medianae']:.6f}",
        f"- Overall P90AE: {metric_delta_d['overall']['p90ae']:.6f}",
        f"- Overall NMAE: {metric_delta_d['overall']['nmae']:.6f}",
        f"- Overall NRMSE: {metric_delta_d['overall']['nrmse']:.6f}",
        f"- Overall R2: {metric_delta_d['overall']['r2']:.6f}",
        f"- Overall Coverage@1sigma: {metric_delta_d['overall']['coverage_1sigma']:.6f}",
        f"- Overall Coverage@2sigma: {metric_delta_d['overall']['coverage_2sigma']:.6f}",
        "",
    ]
    if calibrated_metric_delta_d is not None:
        lines += [
            "## Test Metrics: Calibrated Delta d",
            "",
            f"- Overall MAE: {calibrated_metric_delta_d['overall']['mae']:.6f}",
            f"- Overall RMSE: {calibrated_metric_delta_d['overall']['rmse']:.6f}",
            f"- Overall MaxAE: {calibrated_metric_delta_d['overall']['maxae']:.6f}",
            f"- Overall MedianAE: {calibrated_metric_delta_d['overall']['medianae']:.6f}",
            f"- Overall P90AE: {calibrated_metric_delta_d['overall']['p90ae']:.6f}",
            f"- Overall NMAE: {calibrated_metric_delta_d['overall']['nmae']:.6f}",
            f"- Overall NRMSE: {calibrated_metric_delta_d['overall']['nrmse']:.6f}",
            f"- Overall R2: {calibrated_metric_delta_d['overall']['r2']:.6f}",
            f"- Overall Coverage@1sigma: {calibrated_metric_delta_d['overall']['coverage_1sigma']:.6f}",
            f"- Overall Coverage@2sigma: {calibrated_metric_delta_d['overall']['coverage_2sigma']:.6f}",
            "",
        ]
    lines += [
        "## Test Metrics: Realized Increment",
        "",
        f"- Overall MAE: {metric_realized['overall']['mae']:.6f}",
        f"- Overall RMSE: {metric_realized['overall']['rmse']:.6f}",
        f"- Overall MaxAE: {metric_realized['overall']['maxae']:.6f}",
        f"- Overall MedianAE: {metric_realized['overall']['medianae']:.6f}",
        f"- Overall P90AE: {metric_realized['overall']['p90ae']:.6f}",
        f"- Overall NMAE: {metric_realized['overall']['nmae']:.6f}",
        f"- Overall NRMSE: {metric_realized['overall']['nrmse']:.6f}",
        f"- Overall R2: {metric_realized['overall']['r2']:.6f}",
        "",
        "## Test Metrics: Next Actual q",
        "",
        f"- Overall MAE: {metric_next_q['overall']['mae']:.6f}",
        f"- Overall RMSE: {metric_next_q['overall']['rmse']:.6f}",
        f"- Overall MaxAE: {metric_next_q['overall']['maxae']:.6f}",
        f"- Overall MedianAE: {metric_next_q['overall']['medianae']:.6f}",
        f"- Overall P90AE: {metric_next_q['overall']['p90ae']:.6f}",
        f"- Overall NMAE: {metric_next_q['overall']['nmae']:.6f}",
        f"- Overall NRMSE: {metric_next_q['overall']['nrmse']:.6f}",
        f"- Overall R2: {metric_next_q['overall']['r2']:.6f}",
        "",
        "Prediction details are saved in `gpr_test_predictions.csv`.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def print_metric_summary(title, metric_block):
    overall = metric_block["overall"]
    print(title)
    print(json.dumps(overall, indent=2))


def print_per_axis_metrics(title, metric_block):
    print(title)
    print(json.dumps(metric_block["per_axis"], indent=2))


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate sklearn GPR from pose data.")
    parser.add_argument("--csv", default="gpr_pose_data.csv", help="Input pose CSV path.")
    parser.add_argument("--outdir", default="gpr_training_outputs", help="Output directory.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Chronological train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Chronological validation split ratio.")
    parser.add_argument(
        "--split-mode",
        default="chronological",
        help="Dataset split mode: chronological or random.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for GPR restarts.")
    parser.add_argument("--step-limit", type=float, default=0.8, help="Reject samples with command increments above this limit.")
    parser.add_argument(
        "--enable-reversal-filter",
        action="store_true",
        help="Enable filtering of multi-step windows whose command trajectory reverses direction within the window.",
    )
    parser.add_argument(
        "--reversal-eps",
        type=float,
        default=1e-4,
        help="Tolerance for detecting command-direction reversals inside a span window.",
    )
    parser.add_argument(
        "--span-steps",
        default="1",
        help="Comma-separated span steps used to build multi-step samples, for example 5,7,9,10,11,15.",
    )
    parser.add_argument(
        "--print-limit",
        type=int,
        default=0,
        help="How many test prediction rows to print to the console. Use 0 to print all rows.",
    )
    parser.add_argument(
        "--print-selection",
        default="best",
        help="How to choose printed examples: best, worst, or random.",
    )
    parser.add_argument(
        "--train-model",
        action="store_true",
        help="Retrain the GPR model before prediction. By default the script loads an existing saved model.",
    )
    parser.add_argument(
        "--gpr-restarts",
        type=int,
        default=10,
        help="Number of optimizer restarts for each GPR output axis. Lower this to train faster.",
    )
    parser.add_argument(
        "--kernel-mode",
        default="auto",
        help="Kernel length-scale mode: auto, ard, or shared. Use shared for large multi-span datasets.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=2000,
        help="If positive, downsample the training set to this many samples before GPR fitting.",
    )
    parser.add_argument(
        "--subsample-mode",
        default="diverse",
        help="Training downsampling mode when max-train-samples is active: diverse or uniform.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=15.0,
        help="How often to print training heartbeat updates while one output axis is fitting.",
    )
    parser.add_argument(
        "--model-name",
        default="gpr_model.pkl",
        help="Saved sklearn model bundle filename. The bundle stores [model, scaler].",
    )
    parser.add_argument(
        "--calibrate-std",
        action="store_true",
        help="Use the validation split to calibrate pred_std, then report calibrated test metrics.",
    )
    parser.add_argument(
        "--error-threshold",
        type=float,
        default=None,
        help="Single absolute error threshold kept for backward compatibility. Prefer --error-thresholds.",
    )
    parser.add_argument(
        "--error-thresholds",
        default="0.1,0.2,0.3",
        help="Comma-separated absolute error thresholds used to report the fraction of predictions within tolerance.",
    )
    args = parser.parse_args()
    error_thresholds = parse_error_thresholds(args.error_thresholds)
    if args.error_threshold is not None and args.error_threshold not in error_thresholds:
        error_thresholds = sorted(error_thresholds + [args.error_threshold])

    csv_path = resolve_csv_path(args.csv)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / args.model_name
    span_steps = parse_span_steps(args.span_steps)
    filter_reversals = args.enable_reversal_filter

    samples = build_samples(
        csv_path,
        step_limit=args.step_limit,
        span_steps=span_steps,
        filter_reversals=filter_reversals,
        reversal_eps=args.reversal_eps,
    )
    if len(samples) < 10:
        raise RuntimeError("Not enough valid samples were constructed from the CSV.")

    train_samples, val_samples, test_samples = split_samples_train_val_test(
        samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        split_mode=args.split_mode,
        seed=args.seed,
    )

    write_dataset_csv(samples, outdir / "gpr_dataset_full.csv")
    write_dataset_csv(train_samples, outdir / "gpr_train_dataset.csv")
    write_dataset_csv(val_samples, outdir / "gpr_val_dataset.csv")
    write_dataset_csv(test_samples, outdir / "gpr_test_dataset.csv")

    x_train = np.asarray([s["features"] for s in train_samples], dtype=float)
    x_val = np.asarray([s["features"] for s in val_samples], dtype=float)
    x_test = np.asarray([s["features"] for s in test_samples], dtype=float)
    y_train = np.asarray([s["target_delta_d"] for s in train_samples], dtype=float)
    y_val = np.asarray([s["target_delta_d"] for s in val_samples], dtype=float)
    y_test = np.asarray([s["target_delta_d"] for s in test_samples], dtype=float)

    raw_train_count = len(train_samples)
    effective_train_count = raw_train_count
    subsample_triggered = args.max_train_samples > 0 and raw_train_count > args.max_train_samples
    if subsample_triggered:
        effective_train_count = args.max_train_samples

    print("Training split summary:")
    print(f"  total valid samples: {len(samples)}")
    print(
        f"  split mode: {parse_split_mode(args.split_mode)} "
        f"(train/val/test = {args.train_ratio:.2f}/{args.val_ratio:.2f}/{1.0 - args.train_ratio - args.val_ratio:.2f})"
    )
    print(f"  train / val / test counts: {len(train_samples)} / {len(val_samples)} / {len(test_samples)}")
    print(f"  max_train_samples: {args.max_train_samples}")
    print(f"  subsample mode: {parse_subsample_mode(args.subsample_mode)}")
    print(f"  subsample triggered: {'yes' if subsample_triggered else 'no'}")
    print(f"  effective training samples used for fitting: {effective_train_count}")

    if args.train_model:
        model, scaler = train_gpr_model(
            x_train,
            y_train,
            seed=args.seed,
            n_restarts=args.gpr_restarts,
            heartbeat_seconds=args.heartbeat_seconds,
            kernel_mode=args.kernel_mode,
            max_train_samples=args.max_train_samples,
            subsample_mode=args.subsample_mode,
        )
        joblib.dump([model, scaler], model_path)
        print(f"Trained and saved model bundle: {model_path}")
    else:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Saved model bundle not found: {model_path}\n"
                "Run once with --train-model to create it."
            )
        model, scaler = load_model_bundle(model_path)
        if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != x_test.shape[1]:
            raise ValueError(
                f"Loaded model expects {scaler.n_features_in_} input features, "
                f"but the current dataset has {x_test.shape[1]}.\n"
                "Run once with --train-model to rebuild the model for the current span-step configuration."
            )
        print(f"Loaded existing model bundle: {model_path}")

    pred_delta_d_val, pred_std_val = predict_with_uncertainty(model, scaler, x_val)
    pred_delta_d, pred_std = predict_with_uncertainty(model, scaler, x_test)

    calibrated_pred_std = None
    calibrated_metric_delta_d = None
    std_calibration_scale = None
    if args.calibrate_std:
        std_calibration_scale = compute_std_calibration_scale(y_val, pred_delta_d_val, pred_std_val)
        calibrated_pred_std = apply_std_calibration(pred_std, std_calibration_scale)
        calibrated_metric_delta_d = make_metric_block(
            y_test,
            pred_delta_d,
            calibrated_pred_std,
            error_thresholds=error_thresholds,
        )

    true_realized = np.asarray([s["realized_increment"] for s in test_samples], dtype=float)
    pred_realized = np.asarray([s["du_k"] for s in test_samples], dtype=float) - pred_delta_d
    true_next_q = np.asarray([s["q_act_next"] for s in test_samples], dtype=float)
    pred_next_q = np.asarray([s["q_act_current"] for s in test_samples], dtype=float) + pred_realized

    metric_delta_d = make_metric_block(y_test, pred_delta_d, pred_std, error_thresholds=error_thresholds)
    metric_realized = make_metric_block(
        true_realized,
        pred_realized,
        error_thresholds=error_thresholds,
    )
    metric_next_q = make_metric_block(true_next_q, pred_next_q, error_thresholds=error_thresholds)

    model_params, scaler_info = extract_model_parameters(model, scaler)

    metrics = {
        "metadata": {
            "source_csv": str(csv_path),
            "raw_rows": sum(1 for _ in csv_path.open("r", encoding="utf-8")) - 1,
            "valid_samples": len(samples),
            "train_size": len(train_samples),
            "val_size": len(val_samples),
            "test_size": len(test_samples),
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
            "split_strategy": parse_split_mode(args.split_mode),
            "seed": args.seed,
            "step_limit": args.step_limit,
            "span_steps": span_steps,
            "filter_reversals": filter_reversals,
            "reversal_eps": args.reversal_eps,
            "model_bundle_path": str(model_path),
            "subsample_mode": parse_subsample_mode(args.subsample_mode),
            "error_thresholds": error_thresholds,
            "std_calibration_enabled": bool(args.calibrate_std),
            "std_calibration_split": "val" if args.calibrate_std else None,
            "std_calibration_scale": None if std_calibration_scale is None else np.asarray(std_calibration_scale).tolist(),
        },
        "scaler": scaler_info,
        "model_parameters": model_params,
        "test_metrics_delta_d": metric_delta_d,
        "test_metrics_realized_increment": metric_realized,
        "test_metrics_next_q": metric_next_q,
    }
    if calibrated_metric_delta_d is not None:
        metrics["test_metrics_calibrated_delta_d"] = calibrated_metric_delta_d

    with (outdir / "gpr_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with (outdir / "gpr_model_parameters.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "scaler": scaler_info,
                "model_parameters": model_params,
            },
            f,
            indent=2,
        )

    write_prediction_csv(test_samples, pred_delta_d, pred_std, outdir / "gpr_test_predictions.csv")
    save_report(
        outdir / "gpr_report.md",
        metrics["metadata"],
        model_params,
        metric_delta_d,
        metric_realized,
        metric_next_q,
        model_path.name,
        calibrated_metric_delta_d=calibrated_metric_delta_d,
    )

    print(f"Source CSV: {csv_path}")
    print(f"Span steps: {span_steps}")
    print(f"Reversal filter: {filter_reversals} (eps={args.reversal_eps})")
    print(f"Valid samples: {len(samples)}")
    print(f"Train / val / test: {len(train_samples)} / {len(val_samples)} / {len(test_samples)}")
    print(f"Saved model bundle: {model_path}")
    print_metric_summary("Test delta d overall metrics:", metric_delta_d)
    print_per_axis_metrics("Test delta d per-axis metrics:", metric_delta_d)
    if calibrated_metric_delta_d is not None:
        print_metric_summary("Test calibrated delta d overall metrics:", calibrated_metric_delta_d)
        print_per_axis_metrics("Test calibrated delta d per-axis metrics:", calibrated_metric_delta_d)
    print_prediction_rows(
        test_samples,
        pred_delta_d,
        limit=args.print_limit,
        selection_mode=args.print_selection,
        seed=args.seed,
    )
    print(f"Outputs saved to: {outdir}")


if __name__ == "__main__":
    main()
