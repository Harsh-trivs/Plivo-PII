import os
import torch
import itertools
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset import PIIDataset, collate_batch
from labels import LABELS
from model import create_model

# ----------------------------
# Define hyperparameter grid
# ----------------------------
HYPERPARAM_GRID = {
    "lr": [3e-5, 5e-5],
    "batch_size": [8, 16],
    "epochs": [3, 8]
}

MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"
TRAIN_FILE = "data/train.jsonl"
DEV_FILE = "data/dev.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 256
OUT_DIR = "out"  # Save best model here
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# Evaluation metric (simple)
# ----------------------------
def evaluate(model, dev_dl):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dev_dl:
            input_ids = torch.tensor(batch["input_ids"], device=DEVICE)
            attention_mask = torch.tensor(batch["attention_mask"], device=DEVICE)
            labels = torch.tensor(batch["labels"], device=DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

    return total_loss / max(1, len(dev_dl))

# ----------------------------
# Load dataset
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_ds = PIIDataset(TRAIN_FILE, tokenizer, LABELS, max_length=MAX_LENGTH, is_train=True)
dev_ds = PIIDataset(DEV_FILE, tokenizer, LABELS, max_length=MAX_LENGTH, is_train=False)

# ----------------------------
# Grid search
# ----------------------------
best_loss = float("inf")
best_params = None

for lr, batch_size, epochs in itertools.product(
    HYPERPARAM_GRID["lr"], HYPERPARAM_GRID["batch_size"], HYPERPARAM_GRID["epochs"]
):
    print(f"\nTesting params: lr={lr}, batch_size={batch_size}, epochs={epochs}")

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id)
    )
    dev_dl = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id)
    )

    model = create_model(MODEL_NAME)
    model.to(DEVICE)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dl) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    for epoch in range(epochs):
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = torch.tensor(batch["input_ids"], device=DEVICE)
            attention_mask = torch.tensor(batch["attention_mask"], device=DEVICE)
            labels = torch.tensor(batch["labels"], device=DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    # Evaluate
    dev_loss = evaluate(model, dev_dl)
    print(f"Validation loss: {dev_loss:.4f}")

    if dev_loss < best_loss:
        best_loss = dev_loss
        best_params = {"lr": lr, "batch_size": batch_size, "epochs": epochs}
        # Save best model + tokenizer in out directory
        model.save_pretrained(OUT_DIR)
        tokenizer.save_pretrained(OUT_DIR)

print("\nBest hyperparameters:")
print(best_params)
print(f"Validation loss: {best_loss:.4f}")
print(f"Best model saved in {OUT_DIR}")
