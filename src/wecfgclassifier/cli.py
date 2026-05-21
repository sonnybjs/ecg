from __future__ import annotations

import argparse

from .pipeline import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ECG classifier and task-aware denoiser experiments.")
    parser.add_argument("--days", nargs="+", default=["ALL"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mapping-preset", default="rmxjck_rhythm6_loose")
    parser.add_argument("--classifier-strategy", default="pretrained_cnn_bilstm", choices=["lightweight_pytorch_baseline", "pretrained_cnn_bilstm"])
    parser.add_argument("--classifier-head", default="single", choices=["single", "diagnosis_noise"])
    parser.add_argument("--classifier-input", default="raw", choices=["raw", "frozen_denoiser"])
    parser.add_argument("--classifier-only", action="store_true")
    parser.add_argument("--max-files-per-class", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--classifier-epochs", type=int, default=20)
    parser.add_argument("--denoiser-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-file-batch-sampler", action="store_true")
    parser.add_argument("--log-every-batches", type=int, default=10)
    parser.add_argument("--checkpoint-every-batches", type=int, default=4000)
    parser.add_argument("--validation-max-batches", type=int, default=200)
    parser.add_argument("--test-max-batches", type=int, default=200)
    parser.add_argument("--learning-rate-classifier", type=float, default=1e-3)
    parser.add_argument("--learning-rate-denoiser", type=float, default=2e-5)
    parser.add_argument("--lambda-anchor", type=float, default=0.25)
    parser.add_argument("--lambda-slope", type=float, default=0.05)
    parser.add_argument("--lambda-noise", type=float, default=0.2)
    parser.add_argument("--noise-class-name", default="Noise")
    parser.add_argument("--noise-threshold", type=float, default=0.9)
    parser.add_argument("--diagnosis-confidence-threshold", type=float, default=0.7)
    parser.add_argument("--init-classifier-path")
    parser.add_argument("--init-denoiser-path")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args)
