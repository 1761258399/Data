"""Plot three historical feedback metrics using four trusted CSV columns only.

No controller-name remapping, prediction, model loading, training or hardware.
Run: python scripts/generate_en3_historical_feedback.py
The defaults read data/raw/historical_feedback.csv and write results/local_runs/figures.
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
DEPENDENCIES = ROOT / "tmp" / "plot_dependencies"
if DEPENDENCIES.is_dir():
    sys.path.insert(0, str(DEPENDENCIES))

import numpy as np


GROUP_ORDER = ("OPEN_LOOP", "LQR", "UKF+GPR", "UKF", "DLS")
FIELDS = ("Feedback_Type", "Pos_Error_mm", "Ang_Error_deg", "Convergence_Time_s")
COLOURS = dict(OPEN_LOOP="#8c9399", **{"LQR": "#875f9d", "UKF+GPR": "#cf8936", "UKF": "#276a9d", "DLS": "#4c9279"})
DISPLAY_NAMES = {"OPEN_LOOP": "OPEN\nLOOP", "LQR": "LQR", "UKF+GPR": "UKF\n+GPR", "UKF": "UKF", "DLS": "DLS"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_csv(path):
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            "CSV changed during reading")
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    header = next(reader)
    require(all(header.count(name) == 1 for name in FIELDS), "Required columns must occur once")
    columns = {name: header.index(name) for name in FIELDS}
    groups = {}
    count = 0
    for line_number, row in enumerate(reader, 2):
        require(len(row) > max(columns.values()), f"Required field missing at CSV line {line_number}")
        group = row[columns["Feedback_Type"]].strip()
        require(group in GROUP_ORDER, "Unexpected raw feedback name: " + group)
        values = [float(row[columns[field]]) for field in FIELDS[1:]]
        require(all(np.isfinite(value) and value >= 0 for value in values), "Invalid trusted metric")
        groups.setdefault(group, []).append(values)
        count += 1
    require(count == 152 and len(groups) == 4 and "OPEN_LOOP" in groups
            and all(len(rows) == 38 for rows in groups.values()),
            "This historical figure requires 38 records in each of four groups, including OPEN_LOOP")
    return {group: np.asarray(groups[group], dtype=float) for group in GROUP_ORDER if group in groups}, {
        "path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
        "rows": count, "trusted_columns": list(FIELDS),
        "ignored_header_columns": [name for name in header if name not in FIELDS],
        "ignored_column_note": "All target/direction and other columns are ignored; their alignment is not assumed valid",
    }


def statistics(values):
    q1, median, q3 = map(float, np.percentile(values, [25, 50, 75]))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = values[(values >= lower) & (values <= upper)]
    fliers = values[(values < lower) | (values > upper)]
    return dict(n=int(values.size), mean=float(np.mean(values)), median=median,
                q1=q1, q3=q3, iqr=iqr, minimum=float(np.min(values)), maximum=float(np.max(values)),
                lower_fence=lower, upper_fence=upper,
                lower_whisker=float(np.min(inside)), upper_whisker=float(np.max(inside)),
                outlier_count=int(fliers.size), outlier_values=np.sort(fliers).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "raw" / "historical_feedback.csv")
    parser.add_argument("--assets", type=Path,
                        default=ROOT / "results" / "local_runs" / "figures")
    args = parser.parse_args()
    csv_path, assets = args.csv.resolve(), args.assets.resolve()
    data, source = load_csv(csv_path)
    groups = tuple(data)
    feedback_groups = tuple(group for group in groups if group != "OPEN_LOOP")
    assets.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(assets / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.unicode_minus": True,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.55))
    fig.subplots_adjust(left=.075, right=.99, bottom=.23, top=.77, wspace=.43)
    specifications = (
        (0, groups, "(a) Position", "Position error (mm)"),
        (1, groups, "(b) Angle", "Angular error (deg)"),
        (2, feedback_groups, "(c) Time", "Convergence time (s)"),
    )
    saved_statistics = {}
    for ax, (column, group_names, title, ylabel) in zip(axes, specifications):
        values = [data[group][:, column] for group in group_names]
        positions = list(range(1, len(group_names) + 1))
        boxes = ax.boxplot(values, positions=positions, widths=.55, whis=1.5,
                           patch_artist=True, showfliers=True,
                           medianprops=dict(color="#172832", linewidth=1),
                           whiskerprops=dict(color="#5a6269", linewidth=.7),
                           capprops=dict(color="#5a6269", linewidth=.7),
                           flierprops=dict(marker="o", markersize=3, markeredgewidth=.4, alpha=.8))
        for i, group in enumerate(group_names):
            colour = COLOURS[group]
            boxes["boxes"][i].set(facecolor=colour, edgecolor=colour, alpha=.35, linewidth=.9)
            boxes["fliers"][i].set(markerfacecolor=colour, markeredgecolor=colour)
        ax.set_xticks(positions, [DISPLAY_NAMES[group] for group in group_names])
        ax.set_xlim(.4, len(group_names) + .6)
        ax.set_ylim(0, max(float(np.max(value)) for value in values) * 1.12)
        ax.set_ylabel(ylabel, fontsize=9, labelpad=3)
        ax.set_title(title + "\n38 records per group", fontsize=9, pad=4, weight="semibold")
        ax.tick_params(labelsize=9, length=3, width=.6, pad=2)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.grid(axis="y", color="#dce1e5", linewidth=.45)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(.6)
            spine.set_color("#667078")
        field = FIELDS[column + 1]
        saved_statistics[field] = {group: statistics(data[group][:, column]) for group in group_names}
    outputs = []
    fig.canvas.draw()
    for extension in ("pdf", "png"):
        path = assets / ("gpr16_historical_feedback." + extension)
        fig.savefig(path, dpi=300, facecolor="white")
        outputs.append(dict(path=str(path), sha256=digest(path), width_inches=7,
                            height_inches=2.55, dpi=300 if extension == "png" else None))
    plt.close(fig)
    require(digest(csv_path) == source["sha256"], "Source CSV changed during plotting")
    summary = dict(script=str(Path(__file__).resolve()), script_sha256=digest(__file__),
                   source=source, groups=list(groups), raw_group_order=list(groups),
                   records_per_group={group: int(data[group].shape[0]) for group in groups},
                   time_comparison_groups=list(feedback_groups), time_excluded_group="OPEN_LOOP",
                   time_exclusion_reason="Open-loop duration is not a feedback confirmed-stop time; its 38 source rows are retained, not deleted",
                   statistics=saved_statistics, quantile_method="linear", whisker_iqr_multiplier=1.5,
                   outliers="All outliers are plotted without truncation and fully listed in statistics",
                   original_group_names_preserved=True, current_gpr16_group_mapping_asserted=False,
                   lqr_group_mapping_asserted=False,
                   provenance_note="Group names come directly from Feedback_Type; no mapping to controllers or the current GPR16 model is inferred",
                   outputs=outputs, matplotlib_version=matplotlib.__version__, numpy_version=np.__version__,
                   model_loaded=False, predictions_or_hardware_runs_performed=False, evidence_problems=[])
    (assets / "historical_feedback_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(outputs=outputs, records_per_group=summary["records_per_group"],
                          groups=list(groups), raw_names_preserved=True), ensure_ascii=True))


if __name__ == "__main__":
    main()
