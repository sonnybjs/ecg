# Exp009 Detection MIL: Bitbucket Upload and Resume Notes

This repository should contain only the code needed to train/evaluate the ECG MIL models. Do not commit local data, experiment outputs, logs, or checkpoints.

## Files to commit

- `src/wecfgclassifier/`
- `scripts/train_mil_classifier.py`
- `scripts/detailed_validate_detection_mil_checkpoint.py`
- `scripts/calibrate_detection_mil_thresholds.py`
- `scripts/detailed_validate_mil_checkpoint.py`
- `scripts/threshold_validate_mil_checkpoint.py`
- `mappings/raw19_identity.json`
- `README.md`
- `requirements.txt`
- `.gitignore`
- `docs/BITBUCKET_EXP009_CONTINUE.md`

The current exp009 checkpoint should be transferred separately, for example via Bitbucket Downloads, Git LFS, or a shared storage path. Recommended checkpoint:

```text
experiments/task_aware_denoiser_finetune/mil_experiments/exp_009/checkpoints/best.ckpt
```

## Data layout

By default, the code expects:

```text
<parent-of-repo>/Ecgdata
```

For a different data location, set:

```bash
export ECFG_ECGDATA_DIR=/path/to/Ecgdata
```

For output location:

```bash
export ECFG_OUTPUT_ROOT=/path/to/experiments/task_aware_denoiser_finetune
```

## Continue training from exp009

Use the exp009 checkpoint as `--resume-mil-path`. On a CUDA machine:

```bash
python scripts/train_mil_classifier.py \
  --device cuda \
  --model-type priority_detection_mil \
  --classifier-strategy pretrained_cnn_bilstm \
  --mapping-preset raw19_identity \
  --days ALL \
  --epochs 1 \
  --bag-batch-size 4 \
  --learning-rate 5e-6 \
  --resume-mil-path /path/to/exp_009_best.ckpt \
  --detection-head-type classifier_mlp \
  --event-pooling logsumexp \
  --rhythm-pooling topk_mean \
  --conduction-pooling topk_attention_mix \
  --attention-mix-alpha 0.5 \
  --noise-pooling topk_mean \
  --ce-weight 1.0 \
  --pos-bce-weight 0.3 \
  --group-ce-weight 0.2 \
  --noise-weight 0.2 \
  --global-diagnosis-threshold 0.95 \
  --noise-threshold 0.5 \
  --topk-fraction 0.25 \
  --min-topk 1 \
  --validation-max-bags 1900 \
  --test-max-bags 1900 \
  --validation-every-bags 4000 \
  --checkpoint-every-bags 4000 \
  --log-every-bags 200 \
  --early-stop-metric softmax_macro_f1 \
  --early-stop-patience-validations 8 \
  --early-stop-min-validations 8 \
  --early-stop-min-delta 0.001 \
  --freeze-encoder \
  --freeze-batchnorm \
  --class-balanced-train-sampling \
  --max-grad-norm 1.0
```

For a larger dataset, keep checkpointing/validation every 4000 bags at first. If validation is too slow, increase `--validation-every-bags` to `8000` while keeping `--checkpoint-every-bags 4000`.

## Full validation

```bash
python scripts/detailed_validate_detection_mil_checkpoint.py \
  --checkpoint /path/to/best.ckpt \
  --device cuda \
  --split val \
  --bag-batch-size 4 \
  --max-bags 0 \
  --log-every-batches 250
```
