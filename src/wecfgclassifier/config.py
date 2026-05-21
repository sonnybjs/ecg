from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_DIR = PACKAGE_DIR.parents[1]
RESEARCH_ROOT = PROJECT_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".mplconfig"))

ECGDATA_DIR = Path(os.environ.get("ECFG_ECGDATA_DIR", RESEARCH_ROOT / "Ecgdata")).expanduser()
OUTPUT_ROOT = Path(
    os.environ.get("ECFG_OUTPUT_ROOT", PROJECT_DIR / "experiments" / "task_aware_denoiser_finetune")
).expanduser()
EXTERNAL_MODELS_DIR = Path(os.environ.get("ECFG_EXTERNAL_MODELS_DIR", RESEARCH_ROOT / "DeNoise" / "external_models")).expanduser()
MAPPINGS_DIR = PROJECT_DIR / "mappings"

ASSUMED_INPUT_FS = 250
DENOISER_FS = 360
WINDOW_SEC = 10
WINDOW_SAMPLES = ASSUMED_INPUT_FS * WINDOW_SEC
TARGET_WINDOW_SAMPLES = DENOISER_FS * WINDOW_SEC


@dataclass
class ExperimentConfig:
    days: list[str]
    seed: int
    classifier_strategy: str
    classifier_head: str
    classifier_input: str
    classifier_only: bool
    mapping_preset: str
    mapping_description: str
    class_order: list[str]
    max_files_per_class: int
    val_fraction: float
    test_fraction: float
    classifier_epochs: int
    denoiser_epochs: int
    batch_size: int
    num_workers: int
    file_batch_sampler: bool
    log_every_batches: int
    checkpoint_every_batches: int
    validation_max_batches: int
    test_max_batches: int
    learning_rate_classifier: float
    learning_rate_denoiser: float
    lambda_anchor: float
    lambda_slope: float
    lambda_noise: float
    noise_class_name: str
    noise_threshold: float
    diagnosis_confidence_threshold: float
    window_sec: int
    input_fs: int
    denoiser_fs: int
    init_classifier_path: str | None
    init_denoiser_path: str | None


@dataclass
class FileRecord:
    split: str
    date: str
    raw_type: str
    mapped_label: str
    file_path: str
    file_name: str
    n_samples: int
    n_windows: int


@dataclass
class WindowRecord:
    split: str
    date: str
    raw_type: str
    mapped_label: str
    class_index: int
    file_path: str
    file_name: str
    window_index: int
    start_sample: int
    end_sample: int


@dataclass
class LabelMapping:
    name: str
    description: str
    class_order: list[str]
    class_map: dict[str, str | None]
