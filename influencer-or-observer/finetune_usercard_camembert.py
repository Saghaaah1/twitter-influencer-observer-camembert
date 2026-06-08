"""Fine-tune CamemBERT on aggregated per-user "cards".

The existing finetune_camembert.py trains on one tweet at a time, then smooths
row predictions back to the user. This script makes the user the training unit:
it concatenates profile text plus a small sample of that user's tweets into one
card, trains under user-level CV, then maps user probabilities back to rows.

Outputs row-aligned arrays:
    usercard_camembert_oof_probs.npy
    usercard_camembert_test_probs.npy

Then run:
    python make_submission.py
"""
import argparse
import os
import random

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
from pandas import json_normalize
import torch
import torch.nn.functional as Fnn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

import config
import features as FE


MODEL = "almanach/camembert-base"
MAX_LEN = 256
BATCH = 16
EPOCHS = 3
LR = 2e-5
MAX_TWEETS = 8
SEED = config.SEED
HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = "usercard_camembert"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load(path, nrows=None):
    df = pd.read_json(path, lines=True, nrows=nrows)
    return json_normalize(df.to_dict(orient="records"))


def user_key(df):
    created = df.get("user.created_at", pd.Series("", index=df.index)).fillna("").astype(str)
    name = df.get("user.name", pd.Series("", index=df.index)).fillna("").astype(str)
    return (created + "||" + name).values


def clean_text(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return " ".join(str(x).replace("\n", " ").replace("\r", " ").split())


def source_name(x):
    return FE.parse_source(x)


def build_cards(df, max_tweets=MAX_TWEETS):
    keys = user_key(df)
    work = pd.DataFrame({
        "key": keys,
        "name": df.get("user.name", pd.Series("", index=df.index)).map(clean_text),
        "bio": df.get("user.description", pd.Series("", index=df.index)).map(clean_text),
        "loc": df.get("user.location", pd.Series("", index=df.index)).map(clean_text),
        "source": df.get("source", pd.Series("", index=df.index)).map(source_name).map(clean_text),
        "text": df.apply(FE.get_text, axis=1).map(clean_text),
    })
    if "created_at" in df.columns:
        work["created_at"] = df["created_at"]
        work = work.sort_values(["key", "created_at"], kind="mergesort")

    cards = {}
    for key, g in work.groupby("key", sort=False):
        first = g.iloc[0]
        sources = g["source"].value_counts().head(4).index.tolist()
        tweets = [t for t in g["text"].tolist() if t][:max_tweets]
        parts = [
            f"nom: {first['name']}",
            f"bio: {first['bio']}",
            f"lieu: {first['loc']}",
            "sources: " + " ; ".join(sources),
            "tweets: " + " </s> ".join(tweets),
        ]
        cards[key] = " </s> ".join(p for p in parts if p.strip())
    return keys, cards


def encode(tokenizer, texts, max_len):
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="np",
    )
    return enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64)


@torch.no_grad()
def predict(model, ids, mask, device, bs=128):
    model.eval()
    out = np.zeros(len(ids), dtype=np.float64)
    for start in range(0, len(ids), bs):
        xb = torch.from_numpy(ids[start:start + bs]).to(device)
        mb = torch.from_numpy(mask[start:start + bs]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(input_ids=xb, attention_mask=mb).logits
        out[start:start + bs] = Fnn.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
    return out


def best_acc(y, p):
    best_t, best_a = 0.5, 0.0
    for t in np.arange(0.40, 0.601, 0.005):
        a = accuracy_score(y, (p > t).astype(int))
        if a > best_a:
            best_t, best_a = float(t), a
    return best_t, best_a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="train a tiny 2-fold run")
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--max-tweets", type=int, default=MAX_TWEETS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nrows = config.SMOKE_NROWS if args.smoke else None
    n_folds = config.SMOKE_FOLDS if args.smoke else config.N_FOLDS
    epochs = 1 if args.smoke else args.epochs
    print(f"device={device} model={MODEL} folds={n_folds} max_len={args.max_len}")

    print("Loading train/test...")
    train = load(config.TRAIN_PATH, nrows=nrows)
    test = load(config.TEST_PATH, nrows=nrows)
    y_row = train["label"].values.astype(np.int64)
    tr_keys, tr_cards_map = build_cards(train, args.max_tweets)
    te_keys, te_cards_map = build_cards(test, args.max_tweets)

    tr_users = pd.DataFrame({"key": tr_keys, "label": y_row}).groupby("key", sort=False)["label"].first()
    user_keys = tr_users.index.to_numpy()
    y_user = tr_users.values.astype(np.int64)
    tr_card_texts = [tr_cards_map[k] for k in user_keys]
    te_user_keys = np.array(list(te_cards_map.keys()))
    te_card_texts = [te_cards_map[k] for k in te_user_keys]
    print(f"train rows={len(train)} users={len(user_keys)} test users={len(te_user_keys)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tr_ids, tr_mask = encode(tokenizer, tr_card_texts, args.max_len)
    te_ids, te_mask = encode(tokenizer, te_card_texts, args.max_len)

    oof_user = np.zeros(len(user_keys), dtype=np.float64)
    test_user = np.zeros(len(te_user_keys), dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    for fold, (tr, va) in enumerate(skf.split(tr_ids, y_user), 1):
        print(f"\n=== Fold {fold}/{n_folds} train_users={len(tr)} val_users={len(va)} ===")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(device)
        ds = TensorDataset(
            torch.from_numpy(tr_ids[tr]),
            torch.from_numpy(tr_mask[tr]),
            torch.from_numpy(y_user[tr]),
        )
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=False)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = max(1, len(loader) * epochs)
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

        for ep in range(1, epochs + 1):
            model.train()
            run = 0.0
            for i, (xb, mb, yb) in enumerate(loader):
                xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                    loss = model(input_ids=xb, attention_mask=mb, labels=yb).loss
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                run += loss.item()
                if i % 100 == 0:
                    print(f"  ep{ep} step {i}/{len(loader)} loss {run / (i + 1):.4f}")
            val = predict(model, tr_ids[va], tr_mask[va], device)
            print(f"  ep{ep} val_user_acc {accuracy_score(y_user[va], (val > 0.5).astype(int))*100:.2f}%")

        oof_user[va] = predict(model, tr_ids[va], tr_mask[va], device)
        test_user += predict(model, te_ids, te_mask, device) / n_folds
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    user_oof_map = dict(zip(user_keys, oof_user))
    user_test_map = dict(zip(te_user_keys, test_user))
    oof_rows = np.array([user_oof_map[k] for k in tr_keys], dtype=np.float64)
    test_rows = np.array([user_test_map[k] for k in te_keys], dtype=np.float64)

    thr, acc = best_acc(y_row, oof_rows)
    suffix = "_smoke" if args.smoke else ""
    np.save(os.path.join(HERE, f"{PREFIX}_oof_probs{suffix}.npy"), oof_rows)
    np.save(os.path.join(HERE, f"{PREFIX}_test_probs{suffix}.npy"), test_rows)
    print(f"\nUser-card CamemBERT row OOF: {acc*100:.2f}% @thr {thr:.3f}")
    print(f"Saved {PREFIX}_oof_probs{suffix}.npy / {PREFIX}_test_probs{suffix}.npy")


if __name__ == "__main__":
    main()
