"""Fine-tune CamemBERT (almanach/camembert-base) on the tweet TEXT -> label, under the same
honest StratifiedGroupKFold(user.created_at) used everywhere else, so its OOF + test
predictions align row-for-row with the metadata models and can be blended + smoothed.

Honest by construction: a fold's val/test rows are never seen during that fold's training;
grouping by created_at keeps every user inside one fold. Outputs camembert_oof_probs.npy /
camembert_test_probs.npy. Run:  python finetune_camembert.py
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")        # use the cached model (flaky network)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
from pandas import json_normalize
import torch
import torch.nn.functional as Fnn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

import config
import features as F

MODEL = "almanach/camembert-base"
MAX_LEN = 80
BATCH = 64
EPOCHS = 2
LR = 2e-5
SEED = config.SEED
HERE = config.PRED_DIR  # save aligned prob arrays to output/model_predictions/

torch.manual_seed(SEED); np.random.seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device {dev} | model {MODEL} | maxlen {MAX_LEN} batch {BATCH} epochs {EPOCHS}")

print("Loading train/test...")
train = json_normalize(pd.read_json(config.TRAIN_PATH, lines=True).to_dict(orient="records"))
test = json_normalize(pd.read_json(config.TEST_PATH, lines=True).to_dict(orient="records"))
y = train["label"].values.astype(np.int64)
groups = train.get("user.created_at", pd.Series("", index=train.index)).fillna("").astype(str).values
tr_text = train.apply(F.get_text, axis=1).fillna("").astype(str).tolist()
te_text = test.apply(F.get_text, axis=1).fillna("").astype(str).tolist()
n, n_te = len(y), len(te_text)
print(f"train {n} | test {n_te}")

tok = AutoTokenizer.from_pretrained(MODEL)


def encode(texts):
    enc = tok(texts, padding="max_length", truncation=True, max_length=MAX_LEN,
              return_tensors="np")
    return enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64)


print("Tokenizing...")
tr_ids, tr_mask = encode(tr_text)
te_ids, te_mask = encode(te_text)
te_ds = TensorDataset(torch.from_numpy(te_ids), torch.from_numpy(te_mask))
te_loader = DataLoader(te_ds, batch_size=256, shuffle=False)


@torch.no_grad()
def predict(model, ids, mask, bs=256):
    model.eval()
    out = np.zeros(len(ids), dtype=np.float64)
    for s in range(0, len(ids), bs):
        xb = torch.from_numpy(ids[s:s+bs]).to(dev)
        mb = torch.from_numpy(mask[s:s+bs]).to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
            logits = model(input_ids=xb, attention_mask=mb).logits
        out[s:s+bs] = Fnn.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
    return out


skf = StratifiedGroupKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(n, dtype=np.float64)
test_probs = np.zeros(n_te, dtype=np.float64)

for fold, (tr, va) in enumerate(skf.split(tr_ids, y, groups), 1):
    print(f"\n=== Fold {fold}/{config.N_FOLDS} | train {len(tr)} val {len(va)} ===")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(dev)
    ds = TensorDataset(torch.from_numpy(tr_ids[tr]), torch.from_numpy(tr_mask[tr]),
                       torch.from_numpy(y[tr]))
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(loader) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(0.06*total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))

    for ep in range(1, EPOCHS+1):
        model.train(); run = 0.0
        for i, (xb, mb, yb) in enumerate(loader):
            xb, mb, yb = xb.to(dev), mb.to(dev), yb.to(dev)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                loss = model(input_ids=xb, attention_mask=mb, labels=yb).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()
            if i % 200 == 0:
                print(f"  ep{ep} step {i}/{len(loader)} loss {run/(i+1):.4f}")
        val = predict(model, tr_ids[va], tr_mask[va])
        print(f"  ep{ep} val acc {accuracy_score(y[va], (val>0.5).astype(int))*100:.2f}%")

    oof[va] = predict(model, tr_ids[va], tr_mask[va])
    test_probs += predict(model, te_ids, te_mask) / config.N_FOLDS
    del model; torch.cuda.empty_cache()

np.save(os.path.join(HERE, "camembert_oof_probs.npy"), oof)
np.save(os.path.join(HERE, "camembert_test_probs.npy"), test_probs)


def smooth(p):
    return pd.Series(p).groupby(groups).transform("mean").values


def best(p):
    return max(accuracy_score(y, (p > t).astype(int)) for t in np.arange(0.40, 0.601, 0.005))


print(f"\n=== CamemBERT fine-tuned (text only) ===")
print(f"OOF text-only: raw {best(oof)*100:.2f}%  | smoothed {best(smooth(oof))*100:.2f}%")
print("Saved camembert_oof_probs.npy / camembert_test_probs.npy for blending.")
