# Event-Aligned GPR16: Training Code and Data

Repository: [1761258399/Data](https://github.com/1761258399/Data).

Code and data for independent-session, one-step optical displacement prediction of a four-axis stereotactic guide mechanism. This release contains the original acquisition files, portable training and evaluation code, frozen reference predictions, and figure scripts. A pretrained model is not included; the training command below creates a compatible model for evaluation.

## Experiment

The input is `[q_m[i], q_m[i-1], u[i], u[i-1]]` (16 components), and the label is `u[i] - (q_m[i+1] - q_m[i])`. Three consecutive boundaries supply each single-event window. Four independent GPs use Constant times Matérn-1.5 plus WhiteKernel, with one shared length scale per output. Input standardization and output normalization use TRAIN only. Training uses all valid windows, seed 42, and two optimizer restarts per output.

| Item | Reference result |
|---|---:|
| TRAIN boundaries / valid windows | 6001 / 5480 |
| TEST boundaries / valid windows | 1001 / 936 |
| GPR16 MAE / RMSE | 0.057218 / 0.084259 mm |
| Zero-residual MAE / RMSE, same TEST events | 0.084099 / 0.120273 mm |
| RMSE reduction | 29.9% |
| Residual R² | 0.508936 |

These results concern optical difference-label prediction. The historical feedback data retain their original group names and belong to a separate earlier experiment. The entry points reproduce the offline predictor and do not run a UKF controller or connect to hardware.

## Files

| Path | Contents |
|---|---|
| `data/raw/train.csv`, `data/raw/test.csv` | Original boundaries from separate acquisition sessions |
| `data/raw/historical_feedback.csv` | Earlier feedback records, four groups of 38 observations |
| `src/en3_gpr16/` | Portable pipeline, including frozen numerical source under `vendor/` |
| `scripts/train.py`, `scripts/evaluate.py` | Training, preparation and evaluation entry points |
| `scripts/generate_en3_*.py` | Reference-prediction and historical-feedback plotting |
| `results/reference/` | Frozen datasets, event keys, predictions and native metrics |
| `results/reference/input/test.csv` | Exact TEST copy required by the plotting script |
| `docs/` | Original local reproduction reports |
| `MANIFEST_SHA256.json` | File sizes and hashes for this code-and-data release |

The reports in `docs/` record checks performed with the original locally stored model, including 5480/936 windows and a maximum metric difference of approximately `3.6e-12`. They do not indicate that a model file is supplied here. The numerical vendor files preserve the original source hashes recorded in `src/en3_gpr16/vendor/SOURCE.md`.

## Prepare or train

Run commands from the repository root. Use Python 3.9.10 for the original model environment and the versions in `requirements-model.txt`:

```sh
python -m venv .venv-model
# Activate .venv-model, then:
python -m pip install -r requirements-model.txt

# Construct all valid windows without fitting or loading a model:
python -B scripts/train.py --train data/raw/train.csv --test data/raw/test.csv --model models/retrained_GPR16.pkl --output results/local_runs/preparation --prepare-only

# Fit all four GPs and evaluate on the separate TEST session:
python -B scripts/train.py --train data/raw/train.csv --test data/raw/test.csv --model models/retrained_GPR16.pkl --output results/local_runs/retraining
```

Training creates `models/retrained_GPR16.pkl`, its metadata sidecar, and training/evaluation outputs. TRAIN supplies fitting and scaling; TEST labels are used for evaluation. The portable trainer fits axes sequentially using the original numerical configuration. The archived fit used independent axis processes; a newly fitted binary need not be byte-identical to the original artifact.

## Evaluate a model you trained

```sh
python -B scripts/evaluate.py --train data/raw/train.csv --test data/raw/test.csv --model models/retrained_GPR16.pkl --output results/local_runs/evaluation
```

Evaluation verifies ordered TRAIN features, normalized labels and input scaling before predicting TEST. Use a new or empty output directory for each run. Existing model files are preserved unless training is explicitly given `--overwrite-model`. See [CLI_USAGE.md](scripts/CLI_USAGE.md) for outputs and options.

## Reproduce figures from the supplied reference predictions

Use a separate Python 3.10+ environment for the original figure dependencies:

```sh
python -m venv .venv-figures
# Activate .venv-figures, then:
python -m pip install -r requirements-figures.txt
python scripts/generate_en3_gpr16_paper_figures.py --run results/reference --assets results/local_runs/figures
python scripts/generate_en3_historical_feedback.py --csv data/raw/historical_feedback.csv --assets results/local_runs/figures
```

The prediction figures use all 936 TEST events per axis; plotting does not load a model or generate predictions. The historical plot uses all 38 observations in each recorded group.

## Verify the supplied files

```sh
python -B scripts/verify_artifacts.py
```

The verifier uses only the Python standard library and checks release files against `MANIFEST_SHA256.json`; the manifest excludes itself. `.gitattributes` preserves exact source and data bytes. Generated models, environments and local-run outputs are ignored by Git.
