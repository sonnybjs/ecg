# Task-Aware Denoiser Fine-Tune Report

## Design

- Stage 1: train a PyTorch rhythm classifier using `raw` input.
- Stage 2: optional task-aware denoiser fine-tune with downstream classification loss.
- Progress is written to `train.log` and echoed to terminal during training.
- Label mapping preset: `rmxjck_rhythm6_loose`
- Mapping description: Broader mapping to the 6-class rmxjck rhythm taxonomy so more Ecgdata folder types can participate in a first pretrained-aligned experiment. This is intentionally approximate and should be treated as a staging map, not a clinical equivalence map.
- Preservation losses for the denoiser stage are:
  - anchor MSE to frozen denoiser output (`lambda_anchor=0.25`)
  - slope-consistency L1 to frozen denoiser output (`lambda_slope=0.05`)


## Dataset Subset

- Days: 2025-07-13, 2025-07-14, 2025-07-15
- Files used: 72
- Windows used: 579
- Train files: 54
- Val files: 12
- Test files: 6
- Max files per mapped class: 12
- Val fraction: 0.2
- Test fraction: 0.1

### File Counts by Mapped Class

- NSR: 12 files
- AFIB: 12 files
- SBR: 12 files
- AB: 12 files
- SVTA: 12 files
- B: 12 files

### Raw Folder Types Retained

- AF -> AFIB: 12 files
- AtCou -> AB: 5 files
- AtRun -> SVTA: 6 files
- EctAR -> AB: 2 files
- EctAt -> AB: 5 files
- EctVe -> B: 7 files
- Idiov -> B: 1 files
- JuRhy -> SBR: 4 files
- PaRhy -> SBR: 5 files
- SArrh -> NSR: 5 files
- SRhy -> NSR: 7 files
- SVT -> SVTA: 6 files
- VeCou -> B: 4 files
- Wenck -> SBR: 3 files

## Results

### Stage 1 Classifier on `raw` Input

- Val accuracy: 0.2604
- Val macro F1: 0.1984

### Held-Out Test Set

- Stage 1 test accuracy: 0.5417
- Stage 1 test macro F1: 0.5207

- Stage 2 test macro F1: 0.0476
- Stage 2 test accuracy: 0.1667
## Notes

- Classifier strategy: `pretrained_cnn_bilstm`.
- Classifier input mode: `raw`.
- Classifier-only mode: `False`.
- This run uses a mapping-driven label set so we can either keep all raw folder types or align them to a pretrained taxonomy.

### Stage 2 Task-Aware Denoiser Fine-Tune

- Val accuracy: 0.1667
- Val macro F1: 0.0476
- Val classification loss: 2.0264
- Val anchor loss: 0.000008
- Val slope loss: 0.000093