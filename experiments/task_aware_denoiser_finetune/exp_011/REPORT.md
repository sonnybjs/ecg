# Task-Aware Denoiser Fine-Tune Report

## Design

- Stage 1: train a PyTorch rhythm classifier using `raw` input.
- Stage 2: optional task-aware denoiser fine-tune with downstream classification loss.
- Progress is written to `train.log` and echoed to terminal during training.
- Label mapping preset: `rmxjck_rhythm6_loose`
- Mapping description: Broader mapping to the 6-class rmxjck rhythm taxonomy so more Ecgdata folder types can participate in a first pretrained-aligned experiment. This is intentionally approximate and should be treated as a staging map, not a clinical equivalence map.

## Dataset Subset

- Days: 2025-07-13
- Files used: 18
- Windows used: 144
- Train files: 6
- Val files: 6
- Test files: 6
- Max files per mapped class: 3
- Val fraction: 0.2
- Test fraction: 0.2

### File Counts by Mapped Class

- NSR: 3 files
- AFIB: 3 files
- SBR: 3 files
- AB: 3 files
- SVTA: 3 files
- B: 3 files

### Raw Folder Types Retained

- AF -> AFIB: 3 files
- AtCou -> AB: 2 files
- AtRun -> SVTA: 1 files
- EctAR -> AB: 1 files
- EctVe -> B: 1 files
- JuRhy -> SBR: 3 files
- SRhy -> NSR: 3 files
- SVT -> SVTA: 2 files
- VeCou -> B: 2 files

## Results

### Stage 1 Classifier on `raw` Input

- Val accuracy: 0.1667
- Val macro F1: 0.0485
- Per-class val accuracy: NSR=0.0000, AFIB=0.0000, SBR=0.0000, AB=0.0000, SVTA=0.0000, B=1.0000

### Held-Out Test Set

- Stage 1 test accuracy: 0.1875
- Stage 1 test macro F1: 0.0864

## Notes

- Classifier strategy: `lightweight_pytorch_baseline`.
- Classifier input mode: `raw`.
- Classifier-only mode: `True`.
- This run uses a mapping-driven label set so we can either keep all raw folder types or align them to a pretrained taxonomy.

### Stage 2 Task-Aware Denoiser Fine-Tune

- Skipped in this classifier-only run.