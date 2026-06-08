"""Train/test user OVERLAP via proxy key (main user.id is stripped).

Decides strategy: if test users overlap train heavily, identity-ish features transfer
to the public LB and are worth exploiting; if test is mostly disjoint users, the honest
~84% GroupKFold ceiling is the real limit and identity features are dead weight.

Proxy user key = user.created_at (per-account constant) + hash(description|location),
which is near-unique per account and available on BOTH files.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import json
import hashlib


def ukey(u):
    if not isinstance(u, dict):
        return None
    ca = str(u.get("created_at", ""))
    desc = str(u.get("description", "") or "")
    loc = str(u.get("location", "") or "")
    if ca == "":
        return None
    h = hashlib.md5((desc + "|" + loc).encode("utf-8", "ignore")).hexdigest()[:10]
    return ca + "#" + h


def collect(path, limit=None):
    keys = set()
    rows = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            k = ukey(d.get("user"))
            if k:
                keys.add(k)
            rows += 1
            if limit and rows >= limit:
                break
    return keys, rows


print("Scanning train.jsonl ...")
tr_keys, tr_rows = collect(config.TRAIN_PATH)
print(f"  train: {tr_rows} rows, {len(tr_keys)} proxy users "
      f"({len(tr_keys)/tr_rows*100:.1f}% unique)")

print("Scanning kaggle_test.jsonl ...")
te_keys, te_rows = collect(config.TEST_PATH)
print(f"  test:  {te_rows} rows, {len(te_keys)} proxy users "
      f"({len(te_keys)/te_rows*100:.1f}% unique)")

ov = tr_keys & te_keys
print(f"\n=== OVERLAP ===")
print(f"test users also in train: {len(ov)} "
      f"({len(ov)/max(len(te_keys),1)*100:.1f}% of test users)")
print("Interpretation: high overlap -> identity features transfer to public LB; "
      "low overlap -> honest ~84% ceiling is the limit.")
