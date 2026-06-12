"""Final ensemble -> submission CSV.

Auto-discovers every base model's aligned OOF/test probability arrays
(<name>_oof_probs.npy + <name>_test_probs.npy), keeps only the FULL-SIZE ones
(guards against leftover *_smoke arrays), then:
  - per-user smooths each model once (linear -> exact; key = user.created_at),
  - greedy Caruana ensemble selection on the smoothed OOF (honest: OOF labels only),
  - applies the chosen weights to the smoothed test probs,
  - thresholds (tuned on smoothed OOF) and writes the submission.

Run after the base models exist:
    python train.py --group --blend     # mlp_, lgb_
    python stack_models.py              # hgb_, et_ (+ cat_/xgb_ if installed)
    python finetune_camembert.py        # camembert_
    python make_submission.py           # -> submission_final.csv
"""
import os
import glob
import numpy as np
import pandas as pd
from pandas import json_normalize
from sklearn.metrics import accuracy_score
import config

HERE = config.PRED_DIR  # aligned prob arrays live in output/model_predictions/
N_TRAIN, N_TEST = 154914, 103380


def ukey(df):
    # Proxy per-user key for smoothing. Direct ids (incl. user.name) are stripped;
    # user.created_at alone is constant per account and near-unique -> reliable key.
    return df.get("user.created_at", pd.Series("", index=df.index)).fillna("").astype(str).values


print("Loading train labels/keys and test ids/keys...")
train = json_normalize(pd.read_json(config.TRAIN_PATH, lines=True).to_dict(orient="records"))
test = json_normalize(pd.read_json(config.TEST_PATH, lines=True).to_dict(orient="records"))
y = train["label"].values.astype(int)
tr_keys = ukey(train)
te_keys = ukey(test)
assert len(y) == N_TRAIN and len(test) == N_TEST

# discover aligned, FULL-SIZE base predictions
models = {}
for oof_path in sorted(glob.glob(os.path.join(HERE, "*_oof_probs.npy"))):
    name = os.path.basename(oof_path)[:-len("_oof_probs.npy")]
    test_path = os.path.join(HERE, f"{name}_test_probs.npy")
    if not os.path.exists(test_path):
        continue
    oof, tst = np.load(oof_path), np.load(test_path)
    if len(oof) != N_TRAIN or len(tst) != N_TEST:
        print(f"  skip {name}: wrong size (oof {len(oof)}, test {len(tst)}) — likely smoke")
        continue
    models[name] = (oof, tst)
print(f"base models: {list(models.keys())}")
assert models, "no full-size base prediction arrays found"


def smooth(p, keys):
    return pd.Series(p).groupby(np.asarray(keys)).transform("mean").values


def best_thr_acc(p):
    bt, ba = 0.5, 0.0
    for t in np.arange(0.40, 0.601, 0.005):
        a = accuracy_score(y, (p > t).astype(int))
        if a > ba:
            ba, bt = a, float(t)
    return bt, ba


# smooth every model once (linear, exact)
soof = {k: smooth(v[0], tr_keys) for k, v in models.items()}
stest = {k: smooth(v[1], te_keys) for k, v in models.items()}

print("\nsolo smoothed-OOF accuracy:")
for k in models:
    print(f"  {k:12s}: {best_thr_acc(soof[k])[1]*100:.2f}%")

# greedy Caruana ensemble selection (with replacement) on smoothed OOF
names = list(models.keys())
ens, cur, best_acc = [], np.zeros(N_TRAIN), 0.0
for _ in range(20):
    pick, pick_acc, pick_vec = None, best_acc, None
    for k in names:
        cand = (cur * len(ens) + soof[k]) / (len(ens) + 1)
        a = best_thr_acc(cand)[1]
        if a > pick_acc:
            pick_acc, pick, pick_vec = a, k, cand
    if pick is None:
        break
    ens.append(pick); cur = pick_vec; best_acc = pick_acc
    print(f"  + {pick:12s} -> {best_acc*100:.2f}%")

from collections import Counter
w = Counter(ens)
thr, acc = best_thr_acc(cur)
print(f"\nENSEMBLE weights: {dict(w)}")
print(f"ENSEMBLE smoothed-OOF: {acc*100:.2f}% @thr {thr:.3f}")

# apply to test
tot = sum(w.values())
test_blend = sum(w[k] * stest[k] for k in w) / tot
preds = (test_blend > thr).astype(int)
sub = pd.DataFrame({"ID": test["challenge_id"], "Prediction": preds})
assert list(sub.columns) == ["ID", "Prediction"]
assert len(sub) == N_TEST and not sub["ID"].duplicated().any()
out = os.path.join(config.OUTPUT_DIR, "submission_final.csv")
sub.to_csv(out, index=False)
print(f"\nSaved {out}  ({len(sub)} rows, pred mean {preds.mean():.4f})")
