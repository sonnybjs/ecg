# PyTorch Pretrained ECG Classifier Shortlist

This shortlist keeps only candidates that are:

- PyTorch-based
- open-source or accompanied by enough public implementation detail
- available with pretrained weights or released checkpoints

## Best Fit for Immediate Rhythm Fine-Tuning

### 1. `rmxjck/ltaf-ecg-rhythm-classifier`

Reference:

- Hugging Face: https://huggingface.co/rmxjck/ltaf-ecg-rhythm-classifier

Why it is the best immediate rhythm candidate:

- PyTorch
- pretrained checkpoint available
- direct raw-window rhythm classifier
- simpler than the beat-embedding v2 pipeline
- already covers the most relevant rhythm directions for our current subset:
  - `AFIB`
  - `SVTA`
  - sinus / brady / bigeminy patterns

Important mismatch:

- expects `2-lead`, `128 Hz`, `10 s` windows
- trained on LTAF-derived 6-class task
- does not directly cover:
  - `VT`
  - `Pause`
  - `Noise`

Why I still like it:

- among the current pretrained PyTorch rhythm choices, it is the easiest to adapt for a first real fine-tune
- raw-window input makes it much easier to connect to the current denoiser workflow than the v2 beat-sequence model

Recommended use:

- first serious pretrained rhythm backbone to adapt on our labels

## Stronger but Harder Rhythm Candidate

### 2. `rmxjck/ltaf-ecg-rhythm-classifier-v2`

Reference:

- Hugging Face: https://huggingface.co/rmxjck/ltaf-ecg-rhythm-classifier-v2

Why it is strong:

- PyTorch
- pretrained weights available
- best reported AF/SVTA-like rhythm performance among our current local candidates
- beat-sequence modeling is clinically sensible

Why it is harder:

- depends on upstream beat embeddings
- expects beat extraction, RR history, and sequence preparation
- more engineering before it can replace the current lightweight classifier

Recommended use:

- second step after the simpler raw-window rhythm backbone

## Best Beat / Ectopy Candidate

### 3. `rmxjck/ltaf-ecg-beats-classifier-htf`

Reference:

- Hugging Face: https://huggingface.co/rmxjck/ltaf-ecg-beats-classifier-htf

Why it is useful:

- PyTorch
- local source code and checkpoint available
- excellent beat-level `N / A / V` performance
- strong option for:
  - `PAC/PVC`
  - `ectopy`
  - bigeminy / trigeminy support features

Why it is not the first main classifier:

- beat-level, not rhythm-window-level
- expects `2-lead`, `128 Hz`, `2 s` R-peak-centered windows
- does not solve the full `AF / SVT / VT / Pause / Noise` task by itself

Recommended use:

- separate beat/ectopy branch
- possible auxiliary feature extractor for a later rhythm model

## Best Noise / Artifact Candidate

### 4. `moberg-analytics/okekeclean-ecg`

Reference:

- Hugging Face: https://huggingface.co/moberg-analytics/okekeclean-ecg

Why it matters:

- PyTorch checkpoint
- explicitly artifact-focused rather than rhythm-focused
- useful for the pre-denoise quality gate we care about

Why it is not the main rhythm classifier:

- it is an artifact / noise detector, not an arrhythmia classifier

Recommended use:

- noise-quality branch
- artifact veto / quality gating before downstream rhythm classification

## Framework-Only Option

### 5. `DeepPSP/torch_ecg`

Reference:

- GitHub: https://github.com/DeepPSP/torch_ecg

Why it is useful:

- mature PyTorch ECG framework
- useful model implementations and trainers
- good fallback if we decide to build our own stronger classifier instead of adapting a released checkpoint

Limitation:

- framework value is high, but checkpoint value is less direct than the rmxjck models above

## Current Recommendation

If we want the next step to be practical rather than just academically interesting:

1. Use `rmxjck/ltaf-ecg-rhythm-classifier` as the first pretrained PyTorch rhythm backbone to adapt.
2. Keep `rmxjck/ltaf-ecg-rhythm-classifier-v2` as the stronger second-stage upgrade once preprocessing is ready.
3. Use `rmxjck/ltaf-ecg-beats-classifier-htf` for a dedicated ectopy branch, not as the primary rhythm head.
4. Consider `okekeclean-ecg` separately for noise/artifact gating.

## Why I Am Not Recommending HeartKit Here

`HeartKit` may still be a strong external baseline, but it is not in this shortlist because this shortlist is intentionally restricted to PyTorch-based pretrained options.
