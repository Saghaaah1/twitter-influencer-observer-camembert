"""Fine-tune CamemBERT on (user BIO + tweet TEXT) -> label, as an extra ensemble member.

The social role is often stated in the bio ("journaliste", "officiel", "communication"...),
which generalizes across disjoint users even though the accounts differ. We encode it as a
text PAIR: [CLS] bio [SEP] tweet [SEP]. Same honest StratifiedGroupKFold(user.created_at)
as everything else -> OOF/test align with the metadata models. Outputs
camembertlarge_oof_probs.npy / camembertlarge_test_probs.npy. Run: python finetune_camembert_bio.py
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
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

MODEL = "almanach/camembert-large"
MAX_LEN = 128          # two text fields (bio + tweet)
BATCH = 24
EPOCHS = 2
LR = 1e-5
SEED = config.SEED
HERE = os.path.dirname(os.path.abspath(__file__))

torch.manual_seed(SEED); np.random.seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device {dev} | {MODEL} | maxlen {MAX_LEN} batch {BATCH} epochs {EPOCHS} | bio+tweet (LARGE)")

print("Loading train/test...")
train = json_normalize(pd.read_json(config.TRAIN_PATH, lines=True).to_dict(orient="records"))
test = json_normalize(pd.read_json(config.TEST_PATH, lines=True).to_dict(orient="records"))
y = train["label"].values.astype(np.int64)
groups = train.get("user.created_at", pd.Series("", index=train.index)).fillna("").astype(str).values


def bio_of(df):
    return df.get("user.description", pd.Series("", index=df.index)).fillna("").astype(str).tolist()


tr_bio, te_bio = bio_of(train), bio_of(test)
tr_tweet = train.apply(F.get_text, axis=1).fillna("").astype(str).tolist()
te_tweet = test.apply(F.get_text, axis=1).fillna("").astype(str).tolist()
n, n_te = len(y), len(te_tweet)
print(f"train {n} | test {n_te}")

tok = AutoTokenizer.from_pretrained(MODEL)


def encode(bio, tweet):
    enc = tok(bio, tweet, padding="max_length", truncation="longest_first",
              max_length=MAX_LEN, return_tensors="np")
    return enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64)


print("Tokenizing bio+tweet (LARGE) pairs...")
tr_ids, tr_mask = encode(tr_bio, tr_tweet)
te_ids, te_mask = encode(te_bio, te_tweet)


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
    total = len(loader) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(0.06*total), total)
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
                print(f"  ep{ep} step {i}/{len(loader)} loss {run/(i+1):.4f}", flush=True)
        val = predict(model, tr_ids[va], tr_mask[va])
        print(f"  ep{ep} val acc {accuracy_score(y[va], (val>0.5).astype(int))*100:.2f}%", flush=True)

    oof[va] = predict(model, tr_ids[va], tr_mask[va])
    test_probs += predict(model, te_ids, te_mask) / config.N_FOLDS
    del model; torch.cuda.empty_cache()

np.save(os.path.join(HERE, "camembertlarge_oof_probs.npy"), oof)
np.save(os.path.join(HERE, "camembertlarge_test_probs.npy"), test_probs)


def smooth(p):
    return pd.Series(p).groupby(groups).transform("mean").values


def best(p):
    return max(accuracy_score(y, (p > t).astype(int)) for t in np.arange(0.40, 0.601, 0.005))


print(f"\n=== CamemBERT fine-tuned (bio + tweet) ===")
print(f"OOF: raw {best(oof)*100:.2f}%  | smoothed {best(smooth(oof))*100:.2f}%")
print("Saved camembertlarge_oof_probs.npy / camembertlarge_test_probs.npy for blending.")
