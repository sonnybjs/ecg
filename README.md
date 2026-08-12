# Task-Aware ECG Classification and Denoising

This repository contains a PyTorch research pipeline for ECG classification and task-aware denoiser fine-tuning. It treats downstream rhythm recognition as part of the denoising objective while retaining explicit morphology-preservation constraints.

> Research prototype only. This project is not a clinically validated diagnostic system and does not include patient data.

## Goal

The current workflow is:

1. train or adapt a downstream ECG classifier
2. freeze the classifier
3. fine-tune the denoiser using downstream classification loss
4. keep morphology-preservation constraints so the denoiser does not drift into a purely class-hacking front-end

The practical target tasks are:

- `AF`
- `SVT`
- `VT`
- `Pause`
- `Noise`

The current project also supports mapping the raw `Ecgdata` folders into alternate class taxonomies so we can either:

- keep all `19` local folder types as direct classes
- or align the labels to a pretrained open-source classifier family first

## Layout

- [scripts/run_task_aware_pipeline.py](scripts/run_task_aware_pipeline.py)
  Thin launcher for the structured package pipeline.
- [src/wecfgclassifier/pipeline.py](src/wecfgclassifier/pipeline.py)
  Structured experiment orchestration: manifests, loaders, model setup, stage 1 classifier training, stage 2 denoiser fine-tuning, and reports.
- [src/task_aware_finetune.py](src/task_aware_finetune.py)
  Backward-compatible wrapper around the package CLI.
- [docs/PRETRAINED_CLASSIFIER_SCOUTING.md](docs/PRETRAINED_CLASSIFIER_SCOUTING.md)
  Notes on pretrained classifier candidates, their strengths, and why they are or are not directly usable in the current pipeline.
- [docs/LABEL_MAPPING_PRESETS.md](docs/LABEL_MAPPING_PRESETS.md)
  Mapping presets for `raw19`, `rmxjck` rhythm alignment, and beat-branch pseudo-label alignment.
- [mappings](mappings)
  JSON mapping presets used by the training script.
- [experiments](experiments)
  Versioned experiment outputs.

## Current Design Choice

The current prototype now supports two classifier strategies:

- `ECG_Denoiser` as the denoiser under fine-tuning
- `lightweight_pytorch_baseline`
- `pretrained_cnn_bilstm`

The current default is `pretrained_cnn_bilstm`, which reuses local PyTorch weights from:

- `dheerajthuvara/ecg-arrhythmia-detection`

and adapts the final classification layer to the current mapped class set.

This is still a compromise. Several stronger pretrained ECG classifiers exist, but none of the currently available local candidates perfectly match all of:

- single-lead noisy wearable-style ECG
- the current 5-class task mapping
- direct PyTorch gradient flow into the denoiser

So the current strategy is:

- keep the full task-aware denoiser loop gradient-compatible in PyTorch
- start from a stronger local pretrained classifier backbone when available
- keep mapping presets flexible so local folder labels can either stay raw or be aligned to a pretrained taxonomy first

## Running

Example:

```bash
/Users/Sonny/Research/DeNoise/external_models/.venv/bin/python \
  /Users/Sonny/Research/W｜ECFGClassifier/scripts/run_task_aware_pipeline.py \
  --device cpu \
  --classifier-strategy pretrained_cnn_bilstm \
  --mapping-preset rmxjck_rhythm6_loose \
  --days 2025-07-13 2025-07-14 2025-07-15 \
  --max-files-per-class 12 \
  --classifier-epochs 2 \
  --denoiser-epochs 2 \
  --batch-size 8 \
  --checkpoint-every-batches 4000 \
  --skip-plots
```

Progress is written both to terminal and to `train.log` inside the experiment folder.
Each validation pass also logs per-class validation accuracy and writes it to the stage history CSV as `val_<class>_accuracy` plus `val_<class>_support`.

Training uses a file-window batch sampler by default. It shuffles files, then emits windows from the same file near each other so the CSV signal cache is useful. You can disable it with `--no-file-batch-sampler`.

Each experiment saves end-of-epoch stage checkpoints:

- `checkpoints/stage1/best.ckpt`
- `checkpoints/stage1/last.ckpt`
- `checkpoints/stage2/best.ckpt`
- `checkpoints/stage2/last.ckpt`

and plain state dict exports:

- `best_state_dict.pt`
- `last_state_dict.pt`

It also saves interval checkpoints every `--checkpoint-every-batches` optimizer steps:

- `checkpoints/stage1/step_00004000.ckpt`
- `checkpoints/stage1/interval_latest.ckpt`
- `checkpoints/stage2/step_00004000.ckpt`
- `checkpoints/stage2/interval_latest.ckpt`

This makes it easy to continue later with:

- `--init-classifier-path`
- `--init-denoiser-path`

Useful mapping presets right now:

- `raw19_identity`
  - keep all `19` local folder types
- `rmxjck_rhythm6_strict`
  - conservative alignment to `NSR / AFIB / SBR / AB / SVTA / B`
- `rmxjck_rhythm6_loose`
  - broader alignment to the same 6-class rhythm taxonomy so more folders can participate
- `rmxjck_beat3_pseudo`
  - broad alignment to beat-style `N / A / V` buckets for a later ectopy branch
