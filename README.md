# GPR16 Training Code and Data

Data and code for completed-event optical displacement prediction of a four-axis guide mechanism.

- `data/raw/train.csv`, `test.csv`: separate acquisition sessions with 6001/1001 boundaries and 5480/936 valid event windows.
- `data/raw/historical_feedback.csv`: earlier feedback measurements, four recorded groups of 38 observations.
- `src/en3_gpr16/`: event-window construction and the original numerical training core.
- `scripts/`: training, evaluation and plotting entry points.

The input is `[q_m[i], q_m[i-1], u[i], u[i-1]]` (16 components); the target is `u[i] - (q_m[i+1] - q_m[i])`. Four independent GPs use Constant times Matern-1.5 plus WhiteKernel, one shared length scale per output, TRAIN-only scaling, seed 42 and two optimizer restarts. The entry points run offline without Slicer or a device connection.

Run commands from the repository root. Training creates the model and test predictions locally; use a new or empty output directory for each run.

## Train and evaluate

Use Python 3.9.10 and the numerical versions in `requirements.txt`:

```sh
python -m pip install -r requirements.txt

# Validate and construct all event windows without fitting:
python -B scripts/train.py --train data/raw/train.csv --test data/raw/test.csv --model models/GPR16.pkl --output results/local_runs/preparation --prepare-only

# Fit all valid TRAIN windows and evaluate on TEST:
python -B scripts/train.py --train data/raw/train.csv --test data/raw/test.csv --model models/GPR16.pkl --output results/local_runs/training

# Evaluate the trained model again:
python -B scripts/evaluate.py --train data/raw/train.csv --test data/raw/test.csv --model models/GPR16.pkl --output results/local_runs/evaluation
```

Input/output scaling uses TRAIN only. Evaluation checks the model's ordered training inputs, normalized labels and scaler before predicting TEST. Existing models are preserved unless `--overwrite-model` is explicitly supplied.

## Plot

In a separate Python 3.10+ environment, install `requirements-figures.txt`:

```sh
python -m pip install -r requirements-figures.txt
python scripts/generate_en3_gpr16_paper_figures.py --run results/local_runs/evaluation --test-csv data/raw/test.csv --assets results/local_runs/figures
python scripts/generate_en3_historical_feedback.py --csv data/raw/historical_feedback.csv --assets results/local_runs/figures
```

Prediction plotting reads the outputs produced by training or evaluation. The historical plot reads its raw CSV directly and retains the original group labels. Generated models, predictions, metrics and figures stay in the ignored local output directories.
