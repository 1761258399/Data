"""Create three EN3 publication figures from local GPR16 evaluation outputs.

No model loading, fitting, inference, production imports, or hardware access.
First run training or evaluation to produce predictions, event keys and metrics.
The raw TEST CSV is read separately. Figures and their summary are written to assets.
Run: python scripts/generate_en3_gpr16_paper_figures.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "results" / "local_runs" / "evaluation"
DEFAULT_ASSETS = ROOT / "results" / "local_runs" / "figures"
DEPENDENCIES = ROOT / "tmp" / "plot_dependencies"
if DEPENDENCIES.is_dir():
    sys.path.insert(0, str(DEPENDENCIES))

import numpy as np


AXES = ("X1", "Y1", "X2", "Y2")
GP_BLUE = "#1764a0"
BASE_GREY = "#7c858d"
ATOL = 1e-9  # Frozen prediction CSVs store decimal values to ten places.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_source(path, sources):
    path = Path(path).resolve()
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            "Source changed during reading: " + str(path))
    sources.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))
    return raw.decode("utf-8-sig")


def read_csv(path, sources):
    reader = csv.DictReader(io.StringIO(read_source(path, sources), newline=""))
    require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)),
            "Missing or duplicate CSV headers: " + str(path))
    rows = list(reader)
    require(rows and all(None not in row and all(v is not None for v in row.values()) for row in rows),
            "Empty or malformed CSV: " + str(path))
    return rows


def array(rows, prefix):
    value = np.asarray([[float(row[f"{prefix}_{j}"]) for j in range(1, 5)] for row in rows])
    require(value.shape == (936, 4) and np.isfinite(value).all(), "Invalid array: " + prefix)
    return value


def statistics(error):
    return dict(mae_mm=float(np.mean(np.abs(error))),
                rmse_mm=float(np.sqrt(np.mean(error ** 2))),
                median_absolute_error_mm=float(np.median(np.abs(error))),
                p90_absolute_error_mm=float(np.quantile(np.abs(error), .90)),
                p95_absolute_error_mm=float(np.quantile(np.abs(error), .95)),
                maximum_absolute_error_mm=float(np.max(np.abs(error))))


def box_statistics(values):
    q1, median, q3 = map(float, np.percentile(values, [25, 50, 75]))
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inliers = values[(values >= low) & (values <= high)]
    fliers = values[(values < low) | (values > high)]
    return dict(n=int(values.size), q1_mm=q1, median_mm=median, q3_mm=q3,
                iqr_mm=iqr, lower_fence_mm=low, upper_fence_mm=high,
                lower_whisker_mm=float(np.min(inliers)), upper_whisker_mm=float(np.max(inliers)),
                outlier_count=int(fliers.size), outlier_values_mm=np.sort(fliers).tolist(),
                whisker_rule="Most extreme observed value within 1.5 IQR of Q1/Q3; all outliers shown")


def load_data(run, test_csv):
    sources = []
    predictions = read_csv(run / "test_predictions.csv", sources)
    event_rows = read_csv(run / "test_event_keys.csv", sources)
    boundaries = read_csv(test_csv, sources)
    metrics = json.loads(read_source(run / "test_metrics.json", sources))
    require(len(predictions) == len(event_rows) == 936, "Expected exactly 936 TEST events")
    boundary_map = {}
    for position, row in enumerate(boundaries):
        key = (row["SessionId"], int(row["BoundaryIndex"]))
        require(key not in boundary_map, "Duplicate raw boundary key")
        boundary_map[key] = (position, row)
    keys, truth_position, initial_position, increments = [], [], [], []
    for prediction, event in zip(predictions, event_rows):
        session = event["SourceSession"]
        previous, current, following = (int(event[name]) for name in (
            "BoundaryPrevious", "BoundaryCurrent", "BoundaryNext"))
        require(previous + 1 == current and following == current + 1,
                "A TEST event does not use consecutive boundaries")
        require(int(prediction["sample_k"]) == current, "Prediction/key order differs")
        group = [boundary_map[(session, index)] for index in (previous, current, following)]
        require(group[0][0] + 1 == group[1][0] and group[1][0] + 1 == group[2][0],
                "Raw boundary records are not adjacent")
        require(all(row["MeasurementValid"] == "1" and row["EventRole"] == "sweep" for _, row in group)
                and len({row["SegmentId"] for _, row in group}) == 1,
                "Invalid boundary or continuity break in TEST event")
        before, after = group[1][1], group[2][1]
        initial_position.append([float(before[f"Act_q{j}"]) for j in range(1, 5)])
        truth_position.append([float(after[f"Act_q{j}"]) for j in range(1, 5)])
        increments.append([float(after[name]) - float(before[name]) for name in AXES])
        keys.append((session, current))
    require(len(set(keys)) == 936, "Duplicate TEST event keys")
    current = np.asarray(initial_position)
    expected_position = np.asarray(truth_position)
    command = np.asarray(increments)
    truth = array(predictions, "true_delta_d")
    predicted = array(predictions, "pred_delta_d")
    std = array(predictions, "pred_std_delta_d")
    require(np.all(std >= 0), "Negative predictive standard deviation")
    gp_error = predicted - truth
    baseline_error = -truth
    expected_label = command - (expected_position - current)
    estimated_position = current + command - predicted
    position_error = estimated_position - expected_position
    require(np.isfinite(current).all() and np.isfinite(expected_position).all() and np.isfinite(command).all(),
            "Nonfinite raw position or command")
    checks = {
        "raw_label_vs_frozen_csv_max_mm": float(np.max(np.abs(expected_label - truth))),
        "stored_residual_error_max_mm": float(np.max(np.abs(array(predictions, "err_delta_d") - gp_error))),
        "stored_next_reference_max_mm": float(np.max(np.abs(array(predictions, "true_next_q") - expected_position))),
        "stored_next_estimate_max_mm": float(np.max(np.abs(array(predictions, "pred_next_q") - estimated_position))),
        "position_plus_residual_error_max_mm": float(np.max(np.abs(position_error + gp_error))),
    }
    require(all(value <= ATOL for value in checks.values()), "Position/residual CSV identities failed")
    gp_overall, base_overall = statistics(gp_error), statistics(baseline_error)
    metric_differences = []
    for name, computed in (("mae", gp_overall["mae_mm"]), ("rmse", gp_overall["rmse_mm"]),
                           ("p90ae", gp_overall["p90_absolute_error_mm"]),
                           ("p95ae", gp_overall["p95_absolute_error_mm"])):
        metric_differences.append(abs(computed - metrics["overall"][name]))
    axis_records = []
    for j, name in enumerate(AXES):
        gp, baseline = statistics(gp_error[:, j]), statistics(baseline_error[:, j])
        native = metrics["per_axis"][j]
        require(native["axis"] == j + 1, "Native metric axis order differs")
        metric_differences.extend([abs(gp["mae_mm"] - native["mae"]), abs(gp["rmse_mm"] - native["rmse"])])
        axis_records.append(dict(axis=name, events=936, baseline=baseline, gpr16=gp,
                                 expected_residual_range_mm=[float(truth[:, j].min()), float(truth[:, j].max())],
                                 expected_next_optical_position_range_mm=[float(expected_position[:, j].min()), float(expected_position[:, j].max())],
                                 signed_position_error_range_mm=[float(position_error[:, j].min()), float(position_error[:, j].max())],
                                 baseline_box=box_statistics(np.abs(baseline_error[:, j])),
                                 gpr16_box=box_statistics(np.abs(gp_error[:, j]))))
    require(max(metric_differences) <= ATOL, "Recomputed metrics differ from native TEST metrics")
    checks["metrics_vs_json_max_absolute_difference"] = max(metric_differences)
    return dict(sources=sources, keys=keys, truth=truth, predicted=predicted, expected_position=expected_position,
                estimated_position=estimated_position, position_error=position_error,
                gp_error=gp_error, baseline_error=baseline_error, axes=axis_records,
                gp_overall=gp_overall, baseline_overall=base_overall, checks=checks)


def style_axis(ax):
    ax.grid(True, color="#dce1e5", linewidth=.45, alpha=.85)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, width=.55, length=3, pad=2)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    for spine in ax.spines.values():
        spine.set_linewidth(.6)
        spine.set_color("#667078")


def limits(*arrays):
    values = np.concatenate([np.ravel(x) for x in arrays])
    low, high = float(values.min()), float(values.max())
    pad = max((high - low) * .065, .005)
    return low - pad, high + pad


def save_figure(fig, name, assets, records, description):
    # A fixed canvas preserves the intended double-column width of seven inches.
    fig.canvas.draw()
    for extension in ("pdf", "png"):
        path = assets / (name + "." + extension)
        fig.savefig(path, dpi=300, facecolor="white")
        records.append(dict(path=str(path), sha256=digest(path), width_inches=float(fig.get_figwidth()),
                            height_inches=float(fig.get_figheight()), dpi=300 if extension == "png" else None,
                            description=description))
    plt.close(fig)


def draw_residual(data, assets, outputs):
    fig, axes = plt.subplots(2, 2, figsize=(7, 3.55))
    fig.subplots_adjust(left=.10, right=.985, bottom=.15, top=.89, wspace=.25, hspace=.67)
    fig.text(.5, .978, "GPR16: all 936 held-out TEST events per axis", ha="center", va="top", fontsize=9)
    for j, ax in enumerate(axes.flat):
        bounds = limits(data["truth"][:, j], data["predicted"][:, j])
        ax.plot(bounds, bounds, "--", color="#49535b", linewidth=.8, zorder=1)
        ax.scatter(data["truth"][:, j], data["predicted"][:, j], s=7, color=GP_BLUE,
                   alpha=.45, edgecolors="none", zorder=2)
        ax.set(xlim=bounds, ylim=bounds)
        item = data["axes"][j]["gpr16"]
        ax.set_title(f"({chr(97+j)}) {AXES[j]}   MAE {item['mae_mm']:.4f}; RMSE {item['rmse_mm']:.4f} mm", fontsize=9, pad=4)
        style_axis(ax)
    fig.supxlabel("Expected residual (mm)", fontsize=9, y=.025)
    fig.supylabel("Estimated residual (mm)", fontsize=9, x=.015)
    save_figure(fig, "gpr16_residual_prediction", assets, outputs,
                "Four-axis residual parity; expected is the optical difference label, not independent physical truth; dashed line is equality; 936 points per axis")


def draw_next_position(data, assets, outputs):
    fig, axes = plt.subplots(2, 4, figsize=(7, 3.2))
    fig.subplots_adjust(left=.115, right=.985, bottom=.18, top=.82, wspace=.39, hspace=.40)
    fig.text(.5, .985, "All 936 events per axis; top: next position; bottom: estimate - reference", ha="center", va="top", fontsize=9)
    for j, name in enumerate(AXES):
        ax, detail = axes[0, j], axes[1, j]
        reference, estimated = data["expected_position"][:, j], data["estimated_position"][:, j]
        bounds = limits(reference, estimated)
        ax.plot(bounds, bounds, "--", color="#49535b", linewidth=.8, zorder=1)
        ax.scatter(reference, estimated, s=5, color=GP_BLUE, alpha=.4, edgecolors="none", zorder=2)
        ax.set(xlim=bounds, ylim=bounds)
        ax.set_title(f"({chr(97+j)}) {name}", fontsize=9, pad=3)
        style_axis(ax)
        ax.tick_params(labelbottom=False)
        error = data["position_error"][:, j]
        extent = max(float(np.max(np.abs(error))) * 1.12, .05)
        detail.axhline(0, color="#49535b", linestyle="--", linewidth=.8, zorder=1)
        detail.scatter(reference, error, s=5, color=GP_BLUE, alpha=.4, edgecolors="none", zorder=2)
        detail.set(xlim=bounds, ylim=(-extent, extent))
        detail.set_title(f"({chr(101+j)}) {name}", fontsize=9, pad=3)
        style_axis(detail)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        detail.xaxis.set_major_locator(MaxNLocator(3))
        detail.yaxis.set_major_locator(MaxNLocator(4))
    axes[0, 0].set_ylabel("Estimated next\nposition (mm)", fontsize=9)
    axes[1, 0].set_ylabel("Position error\n(mm)", fontsize=9)
    fig.supxlabel("Expected next optical position (mm)", fontsize=9, y=.025)
    save_figure(fig, "gpr16_absolute_position", assets, outputs,
                "Upper row: next-position parity; lower row: signed one-step optical-position errors at expanded vertical scale; all 936 observations and all extremes shown; not a free rollout")


def draw_boxes(data, assets, outputs):
    fig, axes = plt.subplots(1, 4, figsize=(7, 2.65))
    fig.subplots_adjust(left=.10, right=.985, bottom=.20, top=.76, wspace=.40)
    fig.text(.5, .985, "Same 936 TEST events; whiskers: 1.5 IQR; all outliers shown", ha="center", va="top", fontsize=9)
    for j, ax in enumerate(axes.flat):
        values = [np.abs(data["baseline_error"][:, j]), np.abs(data["gp_error"][:, j])]
        box = ax.boxplot(values, positions=(1, 2), widths=.53, whis=1.5, showfliers=True,
                         patch_artist=True, medianprops=dict(color="#18252d", linewidth=1.1),
                         whiskerprops=dict(color="#49535b", linewidth=.8), capprops=dict(color="#49535b", linewidth=.8),
                         flierprops=dict(marker="o", markersize=2.6, markeredgewidth=.35, alpha=.7))
        for i, colour in enumerate((BASE_GREY, GP_BLUE)):
            box["boxes"][i].set(facecolor=colour, alpha=.35, edgecolor=colour, linewidth=1)
            box["fliers"][i].set(markerfacecolor=colour, markeredgecolor=colour)
        base_count = data["axes"][j]["baseline_box"]["outlier_count"]
        gp_count = data["axes"][j]["gpr16_box"]["outlier_count"]
        ax.set_xticks([1, 2], ["Zero\nresidual", "GPR16"])
        ax.set_xlim(.4, 2.6)
        ax.set_ylim(0, max(float(np.max(x)) for x in values) * 1.10)
        ax.set_title(f"({chr(97+j)}) {AXES[j]}\nOutliers: {base_count} / {gp_count}", fontsize=9, pad=4)
        ax.grid(axis="y", color="#dce1e5", linewidth=.45, alpha=.85)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9, width=.55, length=3, pad=2)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        for spine in ax.spines.values():
            spine.set_linewidth(.6)
            spine.set_color("#667078")
    fig.supylabel("Absolute error (mm)", fontsize=9, x=.012)
    save_figure(fig, "gpr16_prediction_error_boxplots", assets, outputs,
                "Paired absolute one-step errors (equivalently absolute optical residual prediction errors), 936 observations in each box; zero residual predicts d=0; standard 1.5-IQR whiskers with all fliers")


def main():
    global matplotlib, plt, MaxNLocator
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--test-csv", type=Path, default=ROOT / "data" / "raw" / "test.csv",
                        help="Raw TEST boundary CSV corresponding to the evaluation outputs")
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS,
                        help="Output directory for the six figure files and figure_data_summary.json")
    args = parser.parse_args()
    run = args.run.resolve()
    require(run.is_dir(), "Input run directory does not exist")
    assets = args.assets.resolve()
    assets.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(assets / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    data = load_data(run, args.test_csv.resolve())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.unicode_minus": True,
                         "savefig.facecolor": "white", "axes.titleweight": "semibold"})
    outputs = []
    draw_residual(data, assets, outputs)
    draw_next_position(data, assets, outputs)
    draw_boxes(data, assets, outputs)
    for source in data["sources"]:
        require(digest(source["path"]) == source["sha256"], "Frozen source changed during plotting")
    summary = dict(script=str(Path(__file__).resolve()), script_sha256=digest(__file__),
                   input_run=str(run), source_files=data["sources"],
                   events_per_model=936, scalar_axis_components=3744, axis_order=list(AXES),
                   sampling="None; all 936 TEST events per axis are used in every applicable panel and box",
                   metric_scope="Optical difference-label prediction and measured-start one-step optical-position reconstruction; not physical metrology or closed-loop control",
                   expected_residual="true_delta_d from the frozen CSV; verified against command increment minus next/current optical-position difference",
                   next_position="qhat_next = q_current_measured + (q_command_next-q_command_current) - predicted_residual",
                   paired_baseline="Same TEST event and recorded command, residual prediction set to zero; no separate no-compensation hardware trial",
                   csv_decimal_rounding_tolerance_mm=ATOL, identity_and_metric_checks=data["checks"],
                   overall=dict(zero_residual=data["baseline_overall"], gpr16=data["gp_overall"]),
                   per_axis=data["axes"], predicted_standard_deviations_nonnegative=True,
                   plotting=dict(recommended_width="Double column, 7 inches", pdf="Vector scatter/lines/text with embedded TrueType fonts",
                                 png_dpi=300, box_quantile_method="linear", box_whisker_iqr_multiplier=1.5,
                                 box_outliers="All plotted, counted below each box, and fully listed in this JSON"),
                   outputs=outputs, numpy_version=np.__version__, matplotlib_version=matplotlib.__version__,
                   model_loaded=False, training_or_prediction_performed=False, evidence_problems=[])
    (assets / "figure_data_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(events=936, scalar_axis_components=3744, outputs=len(outputs),
                          baseline_rmse_mm=data["baseline_overall"]["rmse_mm"],
                          gpr16_rmse_mm=data["gp_overall"]["rmse_mm"],
                          summary=str(assets / "figure_data_summary.json")), ensure_ascii=True))


if __name__ == "__main__":
    main()
