"""Multi-model honest stack to push past the smoothed MLP+LGBM blend (84.78% OOF).

Trains diverse GBDTs (CatBoost, XGBoost, sklearn HistGBM, ExtraTrees) under the SAME
StratifiedGroupKFold folds as train.py (folds depend only on y + groups, so the saved
mlp_/lgb_ OOF arrays align by row). Combines them with the saved MLP+LGBM via greedy
Caruana ensemble selection on the PER-USER-SMOOTHED OOF (smoothing is linear = group mean,
so smoothing each model once and working in smoothed space is exact). Writes submission.

Honest by construction: same fold-safe source_te + median-impute as train.py; user-constant
created_at only as grouping/smoothing key; weights selected on OOF only.
"""
import os
import numpy as np
import pandas as pd
from pandas import json_normalize
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
import config
import features as F

HERE = config.PRED_DIR  # aligned prob arrays live in output/model_predictions/
SEED = config.SEED
N = config.N_FOLDS


def load(path):
    df = pd.read_json(path, lines=True)
    return json_normalize(df.to_dict(orient="records"))


def user_key(df):
    # Proxy per-user key for grouping/smoothing. Direct ids (incl. user.name) are stripped;
    # user.created_at alone is constant per account and near-unique -> reliable key.
    return df.get("user.created_at", pd.Series("", index=df.index)).fillna("").astype(str).values


def assemble(X_base, parsed_src, df, te_map, gmean):
    X = F.add_static_source_features(parsed_src, df, X_base).copy()
    X["source_te"] = F.apply_source_te(parsed_src, te_map, gmean)
    nonq = [c for c in X.columns if c not in F.QUOTED_IMPUTE_COLS]
    X[nonq] = X[nonq].replace([np.inf, -np.inf], np.nan).fillna(-1)
    X[F.QUOTED_IMPUTE_COLS] = X[F.QUOTED_IMPUTE_COLS].replace([np.inf, -np.inf], np.nan)
    return X


print("Loading full train/test...")
train = load(config.TRAIN_PATH)
test = load(config.TEST_PATH)
y = train["label"].values.astype(int)
groups = user_key(train)
tr_keys = groups
te_keys = user_key(test)

print("Base features...")
Xtb = F.build_features(train)
Xeb = F.build_features(test)
ps_tr = F.parse_source_series(train)
ps_te = F.parse_source_series(test)
# prune still-constant (mirror train.py)
guard = [c for c in Xtb.columns if c not in F.QUOTED_IMPUTE_COLS]
const = [c for c in guard if Xtb[c].nunique(dropna=False) <= 1]
if const:
    Xtb = Xtb.drop(columns=const); Xeb = Xeb.drop(columns=const)

skf = StratifiedGroupKFold(n_splits=N, shuffle=True, random_state=SEED)

# model factories (fresh per fold). CatBoost/XGBoost are optional — included only if
# importable, so the stack runs on sklearn-only when they are not installed.
try:
    import catboost as cbt
    _HAS_CAT = True
except Exception:
    _HAS_CAT = False
try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
print(f"optional models: catboost={_HAS_CAT} xgboost={_HAS_XGB}")


def make_models():
    m = {
        "hgb": HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05,
                                              max_leaf_nodes=63, l2_regularization=1.0,
                                              random_state=SEED),
        "et": ExtraTreesClassifier(n_estimators=600, max_features="sqrt", n_jobs=-1,
                                   random_state=SEED),
    }
    if _HAS_CAT:
        m["cat"] = cbt.CatBoostClassifier(iterations=700, depth=7, learning_rate=0.04,
                                          l2_leaf_reg=5.0, random_seed=SEED, verbose=0,
                                          allow_writing_files=False)
    if _HAS_XGB:
        m["xgb"] = xgb.XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.04,
                                     subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                                     tree_method="hist", random_state=SEED, n_jobs=-1,
                                     eval_metric="logloss")
    return m


names = list(make_models().keys())
oofs = {k: np.zeros(len(y)) for k in names}
tests = {k: np.zeros(len(test)) for k in names}

for fold, (tr, va) in enumerate(skf.split(Xtb, y, groups), 1):
    te_map, gm = F.compute_source_te(ps_tr.iloc[tr].values, y[tr])
    Xtr = assemble(Xtb, ps_tr, train, te_map, gm)
    Xte = assemble(Xeb, ps_te, test, te_map, gm)
    # fold-safe median impute quoted cols (mirror train.py), then -1 fill remaining NaN
    iq = [c for c in F.QUOTED_IMPUTE_COLS if c in Xtr.columns]
    med = {c: Xtr.iloc[tr][c].median(skipna=True) for c in iq}
    med = {c: (v if np.isfinite(v) else -1.0) for c, v in med.items()}
    for c, v in med.items():
        Xtr[c] = Xtr[c].fillna(v); Xte[c] = Xte[c].fillna(v)
    Atr = Xtr.values.astype(np.float32)
    Ate = Xte.values.astype(np.float32)
    for k, m in make_models().items():
        m.fit(Atr[tr], y[tr])
        oofs[k][va] = m.predict_proba(Atr[va])[:, 1]
        tests[k] += m.predict_proba(Ate)[:, 1] / N
    print(f"  fold {fold}/{N} done")

# save raw oof/test
for k in names:
    np.save(os.path.join(HERE, f"{k}_oof_probs.npy"), oofs[k])
    np.save(os.path.join(HERE, f"{k}_test_probs.npy"), tests[k])

# bring in saved MLP + LGBM (same folds)
oofs["mlp"] = np.load(os.path.join(HERE, "mlp_oof_probs.npy"))
tests["mlp"] = np.load(os.path.join(HERE, "mlp_test_probs.npy"))
oofs["lgb"] = np.load(os.path.join(HERE, "lgb_oof_probs.npy"))
tests["lgb"] = np.load(os.path.join(HERE, "lgb_test_probs.npy"))
allnames = names + ["mlp", "lgb"]


def smooth(p, keys):
    return pd.Series(p).groupby(np.asarray(keys)).transform("mean").values


def best_thr_acc(p, yy):
    bt, ba = 0.5, 0.0
    for t in np.arange(0.40, 0.601, 0.005):
        a = accuracy_score(yy, (p > t).astype(int))
        if a > ba:
            ba, bt = a, float(t)
    return bt, ba


# smooth every model's oof/test ONCE (linear -> exact)
soof = {k: smooth(oofs[k], tr_keys) for k in allnames}
stest = {k: smooth(tests[k], te_keys) for k in allnames}

print("\n=== solo smoothed-OOF accuracy ===")
for k in allnames:
    _, a = best_thr_acc(soof[k], y)
    print(f"  {k:4s}: {a*100:.2f}%")

# greedy Caruana ensemble selection in smoothed space
ens = []
cur = np.zeros(len(y))
best_acc = 0.0
for step in range(12):
    best_k, best_step_acc, best_vec = None, best_acc, None
    for k in allnames:
        cand = (cur * len(ens) + soof[k]) / (len(ens) + 1)
        _, a = best_thr_acc(cand, y)
        if a > best_step_acc:
            best_step_acc, best_k, best_vec = a, k, cand
    if best_k is None:
        break
    ens.append(best_k); cur = best_vec; best_acc = best_step_acc
    print(f"  + {best_k:4s} -> {best_acc*100:.2f}% (ens={ens})")

from collections import Counter
w = Counter(ens)
print(f"\nweights: {dict(w)}")
thr, acc = best_thr_acc(cur, y)
print(f"ENSEMBLE smoothed-OOF: {acc*100:.2f}% @thr {thr:.3f}  "
      f"(smoothed MLP+LGBM blend was 84.78%)")

# build test ensemble with same weights, smoothed, threshold from oof
tot = sum(w.values())
test_ens = sum(w[k] * stest[k] for k in w) / tot
preds = (test_ens > thr).astype(int)
sub = pd.DataFrame({"ID": test["challenge_id"], "Prediction": preds})
assert len(sub) == 103380 and not sub["ID"].duplicated().any()
out = os.path.join(config.OUTPUT_DIR, "submission_stack.csv")
sub.to_csv(out, index=False)
print(f"\nSaved {out}  (pred mean {preds.mean():.4f})")
