# Task-Aware Denoiser Fine-Tune Report

## Design

- Stage 1: train a PyTorch rhythm classifier using `raw` input.
- Stage 2: optional task-aware denoiser fine-tune with downstream classification loss.
- Progress is written to `train.log` and echoed to terminal during training.
- Label mapping preset: `raw19_identity`
- Mapping description: Use all 19 Ecgdata folder types as direct training classes.

## Dataset Subset

- Days: 2025-07-13
- Files used: 57
- Windows used: 456
- Train files: 19
- Val files: 19
- Test files: 19
- Max files per mapped class: 3
- Val fraction: 0.2
- Test fraction: 0.2

### File Counts by Mapped Class

- 21AvB: 3 files
- AF: 3 files
- AtCou: 3 files
- AtRun: 3 files
- ComHB: 3 files
- EctAR: 3 files
- EctAt: 3 files
- EctVe: 3 files
- Idiov: 3 files
- JuRhy: 3 files
- Noise: 3 files
- PaRhy: 3 files
- Pause: 3 files
- SArrh: 3 files
- SRhy: 3 files
- SVT: 3 files
- VT: 3 files
- VeCou: 3 files
- Wenck: 3 files

### Raw Folder Types Retained

- 21AvB -> 21AvB: 3 files
- AF -> AF: 3 files
- AtCou -> AtCou: 3 files
- AtRun -> AtRun: 3 files
- ComHB -> ComHB: 3 files
- EctAR -> EctAR: 3 files
- EctAt -> EctAt: 3 files
- EctVe -> EctVe: 3 files
- Idiov -> Idiov: 3 files
- JuRhy -> JuRhy: 3 files
- Noise -> Noise: 3 files
- PaRhy -> PaRhy: 3 files
- Pause -> Pause: 3 files
- SArrh -> SArrh: 3 files
- SRhy -> SRhy: 3 files
- SVT -> SVT: 3 files
- VT -> VT: 3 files
- VeCou -> VeCou: 3 files
- Wenck -> Wenck: 3 files

## Results

### Stage 1 Classifier on `raw` Input

- Val accuracy: 0.0500
- Val macro F1: 0.0075
- Per-class val accuracy: 21AvB=0.0000, AF=0.0000, AtCou=0.0000, AtRun=0.0000, ComHB=0.0000, EctAR=0.0000, EctAt=0.0000, EctVe=0.0000, Idiov=0.0000, JuRhy=0.0000, Noise=0.0000, PaRhy=0.0000, Pause=1.0000, SArrh=0.0000, SRhy=0.0000, SVT=0.0000, VT=0.0000, VeCou=0.0000, Wenck=0.0000

### Held-Out Test Set

- Stage 1 test accuracy: 0.0500
- Stage 1 test macro F1: 0.0066

## Notes

- Classifier strategy: `lightweight_pytorch_baseline`.
- Classifier input mode: `raw`.
- Classifier-only mode: `True`.
- This run uses a mapping-driven label set so we can either keep all raw folder types or align them to a pretrained taxonomy.

### Stage 2 Task-Aware Denoiser Fine-Tune

- Skipped in this classifier-only run.