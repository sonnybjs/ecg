# Pretrained Classifier Scouting

This note summarizes pretrained classifier candidates that are relevant to the current `classifier -> freeze classifier -> task-aware denoiser fine-tune` workflow.

## Selection Criteria

The best candidate for this project should ideally satisfy all of:

- open-source code
- available pretrained weights
- ECG arrhythmia or rhythm focus
- reasonably close to single-lead or adaptable from 2-lead
- feasible to integrate with the current denoiser pipeline
- preferably PyTorch if it is expected to sit directly in the gradient path

## Candidate 1: HeartKit

Reference:

- GitHub: https://github.com/AmbiqAI/heartkit
- Docs: https://ambiqai.github.io/heartkit/

What it provides:

- pretrained rhythm and beat models
- explicit training docs for arrhythmia models
- ECG task framework with configs and CLI

Why it is attractive:

- the rhythm task is already close to our use case
- strong support for `AFIB/AFL`
- practical wearable-oriented framing

Main limitation for the current task-aware denoiser loop:

- TensorFlow / Keras stack
- not the easiest option for direct gradient flow into the current PyTorch `ECG_Denoiser`

Best use right now:

- baseline rhythm classifier
- reference model for `AFIB/AFL`
- comparison target for later PyTorch classifier upgrades

## Candidate 2: rmxjck LTAF Beat HTF

Reference:

- Hugging Face model card: https://huggingface.co/rmxjck/ltaf-ecg-beats-classifier-htf

What it provides:

- pretrained PyTorch beat classifier
- classes `N / A / V`
- local code and weights are already available

Why it is attractive:

- direct source code
- PyTorch
- excellent ectopy-oriented performance

Main limitation for the current workflow:

- beat-level model, not rhythm-window model
- expects `2-lead`, `128 Hz`, `2 s` R-peak-centered inputs
- our current subset labels are file- or rhythm-level, not beat-level

Best use right now:

- future `PAC/PVC/ectopy` branch
- future auxiliary feature extractor

## Candidate 3: rmxjck LTAF Rhythm Classifier v2

Reference:

- Hugging Face model card: https://huggingface.co/rmxjck/ltaf-ecg-rhythm-classifier-v2

What it provides:

- beat-embedding rhythm classifier
- pretrained PyTorch weights
- classes including `AFIB` and `SVTA`

Why it is attractive:

- strongest current PyTorch rhythm candidate for AF/SVTA-like behavior
- sequence-aware rather than pure raw-window classification

Main limitation for the current workflow:

- built around an upstream HTF beat embedding pipeline
- assumes `2-lead`, `128 Hz`, beat-sequence input logic
- does not directly cover the exact target set `AF / SVT / VT / Pause / Noise`

Best use right now:

- second-stage rhythm model once beat extraction and 2-lead adaptation are stabilized

## Candidate 4: torch_ecg

Reference:

- GitHub: https://github.com/DeepPSP/torch_ecg

What it provides:

- broad PyTorch ECG framework
- implemented models, datasets, trainers, ECG-specific utilities

Why it is attractive:

- PyTorch-native
- easier to adapt to the current task-aware denoiser path than a TensorFlow stack
- useful if we decide to move from a lightweight prototype head to a stronger custom rhythm classifier

Main limitation for the current workflow:

- framework is strong, but the immediate benefit is not as direct as a ready-made local checkpoint matched to our labels
- still requires task-specific adaptation and training work

Best use right now:

- architecture inspiration
- future stronger PyTorch classifier backbone

## Why the Current Prototype Still Uses a Lightweight PyTorch Head

Even after looking at pretrained options, the current first-round fine-tune still uses a simple PyTorch classifier because none of the current candidates satisfy all of:

1. ready-made local pretrained weights
2. direct compatibility with current single-lead `Ecgdata`
3. exact 5-class task mapping
4. clean PyTorch gradient flow into `ECG_Denoiser`

So the current practical strategy is:

1. keep the task-aware denoiser loop simple and stable
2. validate that downstream-loss fine-tuning works end to end
3. use that loop as the scaffold for a stronger classifier swap later

## Current Recommendation

Priority order for next classifier upgrades:

1. keep the current lightweight PyTorch head for loop validation
2. evaluate `HeartKit` as a strong external baseline for AF-oriented rhythm classification
3. adapt `rmxjck` rhythm v2 once beat-sequence preprocessing is ready
4. adapt `rmxjck` HTF beat classifier for a dedicated ectopy branch
