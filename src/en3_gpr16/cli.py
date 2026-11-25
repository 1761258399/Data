"""CLI parsing stays free of numerical/GUI imports until thread setup."""
import argparse
import os
from pathlib import Path


def parser_for(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.csv"),
                        help="schema-v5 TRAIN session; also verifies the frozen model's training signature")
    parser.add_argument("--test", type=Path, default=Path("data/raw/test.csv"),
                        help="separate schema-v5 TEST acquisition session")
    parser.add_argument("--model", type=Path, default=Path("models/EN3_GPR16_OPTICAL_FULL.pkl"),
                        help="trusted joblib [model, scaler] bundle")
    parser.add_argument("--output", type=Path, required=True,
                        help="new or empty output directory; input files are read only")
    parser.add_argument("--blas-threads", type=int, default=4)
    return parser


def configure_threads(parser, args):
    if args.blas_threads < 1:
        parser.error("--blas-threads must be positive")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(args.blas_threads)


def main_train(argv=None):
    parser = parser_for("Reproduce EN3's all-valid-event, 16-feature optical residual GP training.")
    parser.add_argument("--prepare-only", action="store_true",
                        help="validate sessions and export features/labels; never fit or load a model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-restarts", type=int, default=2)
    parser.add_argument("--heartbeat-seconds", type=float, default=15.0)
    parser.add_argument("--overwrite-model", action="store_true",
                        help="allow replacing an existing output model (not needed for --prepare-only)")
    args = parser.parse_args(argv)
    configure_threads(parser, args)
    if args.n_restarts < 0 or args.heartbeat_seconds <= 0:
        parser.error("restarts must be nonnegative and heartbeat seconds positive")
    from .pipeline import train
    train(args)


def main_evaluate(argv=None):
    parser = parser_for("Evaluate a trusted frozen EN3 GPR16 bundle without fitting or recalibration.")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="prediction batch size limits temporary TEST-by-TRAIN arrays")
    args = parser.parse_args(argv)
    configure_threads(parser, args)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    from .pipeline import evaluate
    evaluate(args)
