# Offline EN3 GPR16 commands

Run commands from the repository root. This release supplies code, raw data and reference results; it does not contain a pretrained model. Slicer and hardware are not required.

Use Python 3.9.10 and `requirements-model.txt` for the original numerical environment: NumPy 1.26.4, SciPy 1.13.1, scikit-learn 1.6.1, joblib 1.5.2 and threadpoolctl 3.5.0.

## Prepare without loading or fitting a model

```sh
python -B scripts/train.py --train data/raw/train.csv --test data/raw/test.csv --model models/retrained_GPR16.pkl --output results/local_runs/preparation --prepare-only
```

The supplied files produce 5480 TRAIN and 936 TEST windows. Preparation writes `training_request.json`, `preparation.json`, `features_targets.npz`, the two `*_dataset.csv` files and the two `*_event_keys.csv` files. The `--model` path is not opened or written in this mode. `scripts/preparation_smoke_check.json` records the original check, including exact agreement with the frozen feature arrays.

## Train a compatible model

```sh
python -B scripts/train.py --train data/raw/train.csv --test data/raw/test.csv --model models/retrained_GPR16.pkl --output results/local_runs/retraining --blas-threads 4
```

Defaults are all valid TRAIN windows, 16 inputs, four independent scalar GPs, Constant times Matérn-1.5 plus WhiteKernel, one shared length scale per output, seed 42 and two optimizer restarts. Input and output scaling use TRAIN only. TEST labels are not used for fitting or uncertainty calibration. The independent axes run sequentially with the original numerical settings.

Training saves the joblib list `[model, scaler]`; the model is a standard-library `SimpleNamespace` containing four sklearn estimators. It also writes a `<model-stem>.metadata.json` sidecar, `model_info.json`, `status.json`, `training_signature_validation.json` and TEST metrics and predictions. An existing model is preserved unless `--overwrite-model` is explicitly supplied. Newly fitted model bytes and optimization outcomes can depend on the numerical environment.

## Evaluate the model created by training

```sh
python -B scripts/evaluate.py --train data/raw/train.csv --test data/raw/test.csv --model models/retrained_GPR16.pkl --output results/local_runs/evaluation --blas-threads 4
```

The model must exist first. Evaluation verifies its ordered training inputs, normalized targets and input scaler against the supplied TRAIN file. Session IDs must be disjoint; calibration, speed, acceleration, load and measurement policy must match. Only open joblib models from a trusted source because pickle-based loading can execute code.

Outputs include `test_metrics.json`, `test_metrics.csv`, `test_predictions.csv`, `test_dataset.csv`, `test_event_keys.csv`, `zero_residual_metrics.json`, `training_signature_validation.json` and `evaluation.json`. Prediction columns include residual, realized increment and conditional next optical position. CSV prediction values have ten decimal places; metrics use the unrounded arrays. `--batch-size` defaults to 128. Evaluation performs no fit or variance rescaling.

Both entry points require a new or empty `--output` directory. Relative arguments are interpreted from the working directory. The target is the optical difference label `u_i - (q_m[i+1] - q_m[i])`.

## Plot the supplied reference results

Install `requirements-figures.txt` in a separate Python 3.10+ environment, then run:

```sh
python scripts/generate_en3_gpr16_paper_figures.py --run results/reference --assets results/local_runs/figures
python scripts/generate_en3_historical_feedback.py --csv data/raw/historical_feedback.csv --assets results/local_runs/figures
```

The prediction script reads the frozen `test_predictions.csv`, event keys, native metrics and `results/reference/input/test.csv`. All 936 events per axis are used. The historical script retains the four original labels and all 38 observations per group. These commands do not load, fit or query a model.
