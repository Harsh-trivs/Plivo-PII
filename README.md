# PII NER Assignment Skeleton

This repo is a skeleton for a token-level NER model that tags PII in STT-style transcripts.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python src\train.py `
    --model_name huawei-noah/TinyBERT_General_4L_312D `
    --train data\train.jsonl `
    --dev data\dev.jsonl `
    --batch_size 16 `
    --epochs 8 `
    --lr 3e-5 `
    --out_dir out

```

## Hyperparameter Tuning

You can perform automated hyperparameter tuning to find the best combination of learning rate, batch size, and epochs. This script will train multiple models and save the best one in the `out` directory.

```bash
python src\hyperparameter_tuning.py
```

## Predict

```bash
python src\predict.py `
    --model_dir out `
    --input data\dev.jsonl `
    --output out\dev_pred.json
 
```

## Evaluate

```bash
python src/eval_span_f1.py \
  --gold data/dev.jsonl \
  --pred out/dev_pred.json
```

## Measure latency

```bash
python src/measure_latency.py \
  --model_dir out \
  --input data/dev.jsonl \
  --runs 50
```

Your task in the assignment is to modify the model and training code to improve entity and PII detection quality while keeping **p95 latency below ~20 ms** per utterance (batch size 1, on a reasonably modern CPU).
