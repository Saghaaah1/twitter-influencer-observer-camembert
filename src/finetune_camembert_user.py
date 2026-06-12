"""Fine-tune CamemBERT at the USER level -> label, as an extra ensemble member.

WHY THIS IS DIFFERENT (and why it should help):
  The per-tweet CamemBERT models (camembert / camembertbio) classify ONE tweet at a
  time and are weak alone (~70-75%). But the label is USER-CONSTANT and each user has
  ~5 tweets, so the natural unit of decision is the USER, not the tweet. Here we build
  ONE document per user = bio + ALL of that user's tweets concatenated, fine-tune to
  predict the user's label, then BROADCAST each user's predicted probability back to all
  of that user's tweet rows. This gives the transformer ~5x the textual evidence per
  decision -- the same logic that made per-user smoothing the biggest post-feature gain,
  but applied INSIDE the model instead of after it.

ALIGNMENT (so make_submission.py can blend it):
  - Same StratifiedGroupKFold(user.created_at, n_splits=5, seed) as every other model.
  - A user is wholly inside one fold, so we train on the fold's TRAIN users, predict each
    VAL user once, and write that prob to every val ROW of that user -> oof[] is per-row
    and aligned. test_probs[] likewise per-row (each test user's prob broadcast to rows).
  Outputs camembertuser_oof_probs.npy / camembertuser_test_probs.npy.
  Run: python finetune_camembert_user.py
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

MODEL = "almanach/camembert-base"
MAX_LEN = 256          # one document per user (bio + several tweets) -> needs more room
BATCH = 24             # longer sequences -> smaller batch to fit GPU memory
EPOCHS = 3             # fewer examples than per-tweet (one per user) -> afford an extra epoch
LR = 2e-5
MAX_TWEETS = 8         # cap tweets per user document (users have <=15; 8 covers the mass)
SEED = config.SEED
HERE = config.PRED_DIR

torch.manual_seed(SEED); np.random.seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device {dev} | {MODEL} | maxlen {MAX_LEN} batch {BATCH} epochs {EPOCHS} | USER-LEVEL")

print("Loading train/test...")
train = json_normalize(pd.read_json(config.TRAIN_PATH, lines=True).to_dict(orient="records"))
test = json_normalize(pd.read_json(config.TEST_PATH, lines=True).to_dict(orient="records"))
y_row = train["label"].values.astype(np.int64)
g_train = train.get("user.created_at", pd.Series("", index=train.index)).fillna("").astype(str).values
g_test = test.get("user.created_at", pd.Series("", index=test.index)).fillna("").astype(str).values
n_row, n_te_row = len(y_row), len(test)
print(f"train rows {n_row} | test rows {n_te_row}")

# ---------------------------------------------------------------------------
# Build ONE document per user: bio + up to MAX_TWEETS of the user's tweets.
# Returns, for a dataframe: (user_keys[], user_docs[], row_to_user_index[])
# so a user-level prediction can be scattered back to every row of that user.
# ---------------------------------------------------------------------------
def build_user_docs(df, groups):
    bio = df.get("user.description", pd.Series("", index=df.index)).fillna("").astype(str).values
    tweet = df.apply(F.get_text, axis=1).fillna("").astype(str).values
    # stable first-seen order of users, and a row -> user-index map
    user_keys = []
    key_to_uidx = {}
    row_to_uidx = np.empty(len(df), dtype=np.int64)
    per_user_tweets = []
    per_user_bio = []
    for i, k in enumerate(groups):
        if k not in key_to_uidx:
            key_to_uidx[k] = len(user_keys)
            user_keys.append(k)
            per_user_tweets.append([])
            per_user_bio.append(bio[i])   # bio is user-constant; first occurrence is fine
        ui = key_to_uidx[k]
        row_to_uidx[i] = ui
        if len(per_user_tweets[ui]) < MAX_TWEETS:
            t = tweet[i].strip().replace("\n", " ")
            if t:
                per_user_tweets[ui].append(t)
    # assemble: bio [SEP] tweet1 [SEP] tweet2 ...  (use the tokenizer sep token later)
    docs = []
    for ui in range(len(user_keys)):
        parts = []
        if per_user_bio[ui].strip():
            parts.append(per_user_bio[ui].strip().replace("\n", " "))
        parts.extend(per_user_tweets[ui])
        docs.append(" </s> ".join(parts) if parts else "")
    return user_keys, docs, row_to_uidx


print("Aggregating tweets into per-user documents...")
tr_keys, tr_docs, tr_row2u = build_user_docs(train, g_train)
te_keys, te_docs, te_row2u = build_user_docs(test, g_test)
n_user, n_te_user = len(tr_keys), len(te_keys)
# user-level label (constant within a user) and grouping key (the user itself)
y_user = np.array([y_row[np.where(tr_row2u == ui)[0][0]] for ui in range(n_user)], dtype=np.int64)
g_user = np.array(tr_keys)  # each user is its own group -> GroupKFold == per-user split
print(f"users: train {n_user} (mean {n_row/n_user:.2f} tweets) | test {n_te_user}")

tok = AutoTokenizer.from_pretrained(MODEL)


def encode(docs):
    enc = tok(list(docs), padding="max_length", truncation=True,
              max_length=MAX_LEN, return_tensors="np")
    return enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64)


print("Tokenizing user documents...")
tr_ids, tr_mask = encode(tr_docs)
te_ids, te_mask = encode(te_docs)


@torch.no_grad()
def predict(model, ids, mask, bs=128):
    model.eval()
    out = np.zeros(len(ids), dtype=np.float64)
    for s in range(0, len(ids), bs):
        xb = torch.from_numpy(ids[s:s+bs]).to(dev)
        mb = torch.from_numpy(mask[s:s+bs]).to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
            logits = model(input_ids=xb, attention_mask=mb).logits
        out[s:s+bs] = Fnn.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
    return out


# user-level CV: split USERS (StratifiedGroupKFold with each user its own group is just a
# stratified split over users; we keep the same class as the rest of the repo for parity).
skf = StratifiedGroupKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=SEED)
oof_user = np.zeros(n_user, dtype=np.float64)
test_user = np.zeros(n_te_user, dtype=np.float64)

for fold, (tr, va) in enumerate(skf.split(tr_ids, y_user, g_user), 1):
    print(f"\n=== Fold {fold}/{config.N_FOLDS} | train users {len(tr)} val users {len(va)} ===", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(dev)
    ds = TensorDataset(torch.from_numpy(tr_ids[tr]), torch.from_numpy(tr_mask[tr]),
                       torch.from_numpy(y_user[tr]))
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
            if i % 100 == 0:
                print(f"  ep{ep} step {i}/{len(loader)} loss {run/(i+1):.4f}", flush=True)
        val = predict(model, tr_ids[va], tr_mask[va])
        print(f"  ep{ep} USER val acc {accuracy_score(y_user[va], (val>0.5).astype(int))*100:.2f}%", flush=True)

    oof_user[va] = predict(model, tr_ids[va], tr_mask[va])
    test_user += predict(model, te_ids, te_mask) / config.N_FOLDS
    del model; torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Broadcast user-level probabilities back to per-ROW arrays (aligned with all
# other models, so make_submission.py blends them by row).
# ---------------------------------------------------------------------------
oof_row = oof_user[tr_row2u]
test_row = test_user[te_row2u]
assert len(oof_row) == n_row and len(test_row) == n_te_row

np.save(os.path.join(HERE, "camembertuser_oof_probs.npy"), oof_row)
np.save(os.path.join(HERE, "camembertuser_test_probs.npy"), test_row)


def best(p, yy):
    return max(accuracy_score(yy, (p > t).astype(int)) for t in np.arange(0.40, 0.601, 0.005))


print(f"\n=== CamemBERT fine-tuned (USER-level) ===")
print(f"USER-level OOF acc      : {best(oof_user, y_user)*100:.2f}%")
print(f"Broadcast to rows (==smoothed): {best(oof_row, y_row)*100:.2f}%")
print("Saved camembertuser_oof_probs.npy / camembertuser_test_probs.npy for blending.")
print("Next: run  python make_submission.py  to fold it into the ensemble.")

