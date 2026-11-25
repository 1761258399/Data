"""Offline adapter around the exact frozen numerical training core.

The validator checks schema-v5 boundary records. Its legacy four-boundary
sample selection is deliberately discarded: EN3 uses three boundaries and
one completed event per target. No GUI, serial, tracker or hardware module
is imported here.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .vendor import train_gpr_from_pose_data as native
from .vendor.validator_module1_learning import EventLearning

AXES = ("X1", "Y1", "X2", "Y2")
STEP_LIMIT_MM = 0.8
MODEL_SCOPE = "optical difference-label prediction; not independently identified physical covariance"
MATCH_KEYS = ("calibration_version", "condition_signature", "measurement_policy")


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def save_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def save_csv(path, rows):
    if not rows:
        raise ValueError("Cannot export an empty table")
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_session(path):
    """Apply frozen boundary validation, then the frozen EN3 3-boundary adapter."""
    path = Path(path)
    checksum = digest(path)
    validator = EventLearning(8, gpr_training_max_increment=STEP_LIMIT_MM)
    _, metadata = validator._build_gpr_samples_from_csvs([str(path)], purpose="evaluation")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    samples = []
    excluded = {"invalid_or_segment": 0, "command_limit": 0}
    for k in range(1, len(rows) - 1):
        window = rows[k - 1:k + 2]
        valid = (
            all(row["MeasurementValid"] == "1" and row.get("EventRole", "sweep") == "sweep"
                and np.isfinite([float(row["Act_q" + str(a + 1)]) for a in range(4)]).all()
                for row in window)
            and len({row["SessionId"] for row in window}) == 1
            and len({row.get("SegmentId", "0") for row in window}) == 1
        )
        if not valid:
            excluded["invalid_or_segment"] += 1
            continue
        q = [np.asarray([float(row["Act_q" + str(a + 1)]) for a in range(4)]) for row in window]
        c = [np.asarray([float(row[axis]) for axis in AXES]) for row in window]
        command, previous = c[2] - c[1], c[1] - c[0]
        if np.max(np.abs(np.r_[command, previous])) > STEP_LIMIT_MM + 1e-12:
            excluded["command_limit"] += 1
            continue
        realized = q[2] - q[1]
        sample = native.make_multistep_sample(
            int(rows[k]["BoundaryIndex"]), int(rows[k + 1]["BoundaryIndex"]),
            float(rows[k]["BoundaryRecordMonoTime"]), 1,
            q[1], q[0], q[2], command, previous, realized, command - realized)
        sample["source_session"] = rows[k]["SessionId"]
        sample["boundary_previous"] = int(rows[k - 1]["BoundaryIndex"])
        samples.append(sample)
    if not samples:
        raise ValueError(str(path) + " has no valid single-event windows")
    features = np.asarray([sample["features"] for sample in samples])
    targets = np.asarray([sample["target_delta_d"] for sample in samples])
    if (features.shape[1:] != (16,) or targets.shape[1:] != (4,)
            or not np.isfinite(features).all() or not np.isfinite(targets).all()):
        raise ValueError("Unexpected or nonfinite EN3 feature/label layout")
    info = {key: metadata[key] for key in ("session_ids",) + MATCH_KEYS}
    info.update(raw_boundaries=len(rows), candidate_windows=len(rows) - 2,
                included=len(samples), excluded=excluded, source_csv=path.name, source_sha256=checksum,
                independent_reference_columns_present=any("Independent" in field or "Ref_q" in field
                                                          for field in rows[0]))
    if digest(path) != checksum:
        raise ValueError("Input CSV changed during preparation: " + str(path))
    return {"samples": samples, "x": features, "y": targets, "meta": info}


def load_pair(train_path, test_path):
    if Path(train_path).resolve() == Path(test_path).resolve():
        raise ValueError("TRAIN and TEST must be different acquisition sessions")
    training = load_session(train_path)
    testing = load_session(test_path)
    if set(training["meta"]["session_ids"]) & set(testing["meta"]["session_ids"]):
        raise ValueError("TRAIN and TEST contain the same acquisition SessionId")
    if training["meta"]["source_sha256"] == testing["meta"]["source_sha256"]:
        raise ValueError("TRAIN and TEST files have the same content")
    for key in MATCH_KEYS:
        if training["meta"][key] != testing["meta"][key]:
            raise ValueError("TRAIN/TEST metadata mismatch: " + key)
    return training, testing


def output_directory(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("--output must be a new or empty directory: " + str(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_dataset(data, output, role):
    native.write_dataset_csv(data["samples"], output / (role + "_dataset.csv"))
    keys = [{"SourceSession": sample["source_session"],
             "BoundaryPrevious": sample["boundary_previous"],
             "BoundaryCurrent": sample["sample_k"], "BoundaryNext": sample["sample_end_k"]}
            for sample in data["samples"]]
    save_csv(output / (role + "_event_keys.csv"), keys)


def environment(threads):
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "joblib": joblib.__version__, "blas_threads": threads}


def request_record(training, testing, args):
    return {"created_at": datetime.now(timezone.utc).isoformat(), "model_name": args.model.stem,
            "model_scope": MODEL_SCOPE, "train": training["meta"], "test": testing["meta"],
            "feature_names": native.FEATURE_NAMES, "step_limit_mm": STEP_LIMIT_MM,
            "span_steps": [1], "use_test_for_fit": False, "no_new_CAL": True,
            "std_calibration_scale": [1.0] * 4, "no_label_magnitude_truncation": True,
            "environment": environment(args.blas_threads)}


def validate_model_training(model, scaler, training):
    """Prove this bundle fits the supplied ALL-valid TRAIN, not the TEST session.

    This also verifies an archived bundle without requiring its machine-specific
    metadata paths. Input order and every normalized training target are checked.
    No model is fitted, cloned, or changed.
    """
    estimators = getattr(model, "estimators_", None)
    if estimators is None or len(estimators) != 4:
        raise ValueError("Expected four independent fitted scalar GPs")
    expected_scaler = StandardScaler().fit(training["x"])
    for key in ("mean_", "scale_", "var_"):
        actual = np.asarray(getattr(scaler, key, []))
        expected = getattr(expected_scaler, key)
        if actual.shape != (16,) or not np.allclose(actual, expected, rtol=1e-12, atol=1e-12):
            raise ValueError("Model scaler does not match ALL supplied TRAIN: " + key)
    scaled = expected_scaler.transform(training["x"])
    kernels = []
    for axis, estimator in enumerate(estimators):
        if estimator.X_train_.shape != scaled.shape or not np.allclose(
                estimator.X_train_, scaled, rtol=1e-12, atol=1e-12):
            raise ValueError("Model's ordered training features differ on axis " + str(axis + 1))
        if not estimator.normalize_y:
            raise ValueError("Expected per-output normalize_y=True")
        target = training["y"][:, axis]
        target_mean = float(np.mean(target))
        target_scale = float(np.std(target))
        if target_scale < 10 * np.finfo(float).eps:
            target_scale = 1.0
        normalized = (target - target_mean) / target_scale
        if (not np.allclose(estimator._y_train_mean, target_mean, rtol=1e-12, atol=1e-12)
                or not np.allclose(estimator._y_train_std, target_scale, rtol=1e-12, atol=1e-12)
                or np.asarray(estimator.y_train_).shape != normalized.shape
                or not np.allclose(estimator.y_train_, normalized, rtol=1e-11, atol=1e-11)):
            raise ValueError("Model's training labels differ on axis " + str(axis + 1))
        kernel = estimator.kernel_
        try:
            shared = np.asarray(kernel.k1.k2.length_scale).size == 1
            correct_nu = float(kernel.k1.k2.nu) == 1.5
            white_noise = float(kernel.k2.noise_level)
            amplitude = float(kernel.k1.k1.constant_value)
        except AttributeError as exc:
            raise ValueError("Unexpected EN3 GP kernel structure") from exc
        if not shared or not correct_nu or not np.isfinite([white_noise, amplitude]).all():
            raise ValueError("Expected shared-length-scale Matérn-1.5 plus WhiteKernel")
        kernels.append(str(kernel))
    return {"passed": True, "train_events": len(training["x"]), "features": 16,
            "ordered_training_features_match": True, "normalized_training_labels_match": True,
            "fit_only_scaler_match": True, "kernels": kernels}


def predict_batches(model, scaler, features, batch_size):
    means, deviations = [], []
    for start in range(0, len(features), batch_size):
        mean, std = native.predict_with_uncertainty(model, scaler, features[start:start + batch_size])
        means.append(mean)
        deviations.append(std)
    mean, std = np.vstack(means), np.vstack(deviations)
    if (mean.shape != (len(features), 4) or std.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 0)):
        raise ValueError("Invalid native prediction mean/standard deviation")
    return mean, std


def export_evaluation(model, scaler, testing, output, batch_size=128):
    prediction, std = predict_batches(model, scaler, testing["x"], batch_size)
    metrics = native.make_metric_block(testing["y"], prediction, std, error_thresholds=[.02, .05, .1, .2])
    metrics["overall"]["p95ae"] = float(np.quantile(np.abs(prediction - testing["y"]), .95))
    native.write_prediction_csv(testing["samples"], prediction, std, output / "test_predictions.csv")
    save_json(output / "test_metrics.json", metrics)
    rows = [dict(Channel="Overall", **metrics["overall"])]
    rows.extend(dict(Channel=AXES[i], **row) for i, row in enumerate(metrics["per_axis"]))
    save_csv(output / "test_metrics.csv", [dict(Channel=row["Channel"], MAE_mm=row["mae"],
        RMSE_mm=row["rmse"], MedAE_mm=row["medianae"], P90AE_mm=row["p90ae"], R2=row["r2"])
        for row in rows])
    baseline = native.make_metric_block(testing["y"], np.zeros_like(testing["y"]))
    save_json(output / "zero_residual_metrics.json", baseline)
    return metrics, prediction, std


def train(args):
    started = time.monotonic()
    if not args.prepare_only and args.model.exists() and not args.overwrite_model:
        raise ValueError("Model already exists; choose a new --model or explicitly use --overwrite-model")
    for source in (args.train, args.test):
        if args.model.resolve() == source.resolve():
            raise ValueError("Model output must not overwrite an input CSV")
    training, testing = load_pair(args.train, args.test)
    output = output_directory(args.output)
    request = request_record(training, testing, args)
    options = dict(seed=args.seed, n_restarts=args.n_restarts, kernel_mode="shared", max_train_samples=0,
                   subsample_mode="uniform", heartbeat_seconds=args.heartbeat_seconds)
    request.update(training_options=options, prepare_only=args.prepare_only)
    save_json(output / "training_request.json", request)
    export_dataset(training, output, "train")
    export_dataset(testing, output, "test")
    np.savez_compressed(output / "features_targets.npz", x_train=training["x"], y_train=training["y"],
                        x_test=testing["x"], y_test=testing["y"])
    prepared = {"stage": "prepared", "fit_performed": False, "model_loaded": False,
                "train_events": len(training["x"]), "test_events": len(testing["x"]), "features": 16,
                "train_test_session_disjoint": True, "condition_calibration_policy_match": True}
    save_json(output / "preparation.json", prepared)
    print(json.dumps(prepared), flush=True)
    if args.prepare_only:
        return
    save_json(output / "status.json", {"stage": "fitting", "training_options": options})
    with threadpool_limits(limits=args.blas_threads):
        model, scaler = native.train_gpr_model(training["x"], training["y"], **options)
        signature = validate_model_training(model, scaler, training)
        args.model.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.model.with_name(args.model.name + ".partial")
        joblib.dump([model, scaler], temporary)
        os.replace(temporary, args.model)
        metrics, _, _ = export_evaluation(model, scaler, testing, output)
    parameters, scaling = native.extract_model_parameters(model, scaler)
    model_info = {"model_file": args.model.name, "model_sha256": digest(args.model), "model_scope": MODEL_SCOPE,
                  "features": 16, "training_events": len(training["x"]), "test_events": len(testing["x"]),
                  "feature_names": native.FEATURE_NAMES, "scaler": scaling, "kernel_parameters": parameters,
                  "training_options": options, "train": training["meta"], "test": testing["meta"],
                  "std_calibration_scale": [1.0] * 4, "training_seconds": time.monotonic() - started}
    save_json(output / "model_info.json", model_info)
    save_json(args.model.with_suffix(".metadata.json"), model_info)
    save_json(output / "training_signature_validation.json", signature)
    save_json(output / "status.json", {"stage": "complete", "fit_performed": True,
        "model_sha256": model_info["model_sha256"], "overall": metrics["overall"]})
    print(json.dumps({"stage": "complete", "model": str(args.model), "overall": metrics["overall"]}), flush=True)


def evaluate(args):
    started = time.monotonic()
    training, testing = load_pair(args.train, args.test)
    output = output_directory(args.output)
    before = digest(args.model)
    # joblib/pickle is executable serialization: load only this repository's
    # trusted artifact or a trusted model produced by the training CLI.
    model, scaler = native.load_model_bundle(args.model)
    signature = validate_model_training(model, scaler, training)
    with threadpool_limits(limits=args.blas_threads):
        metrics, _, _ = export_evaluation(model, scaler, testing, output, args.batch_size)
    export_dataset(testing, output, "test")
    save_json(output / "training_signature_validation.json", signature)
    if digest(args.model) != before:
        raise ValueError("Frozen model changed during evaluation")
    record = request_record(training, testing, args)
    record.update(model_file=args.model.name, model_sha256=before, fit_performed=False,
                  batch_size=args.batch_size, elapsed_seconds=time.monotonic() - started,
                  training_signature_validation=signature, overall=metrics["overall"])
    save_json(output / "evaluation.json", record)
    print(json.dumps({"stage": "evaluated", "test_events": len(testing["x"]),
                      "model_sha256": before, "overall": metrics["overall"]}), flush=True)
