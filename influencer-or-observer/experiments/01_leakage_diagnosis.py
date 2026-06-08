"""
Diagnostic: is the v2 feature upgrade leaking user identity under row-wise CV?

Hypothesis: the new user-CONSTANT profile-text features (desc_len, loc_len, etc.)
inflate StratifiedKFold OOF (same user appears across folds) but don't generalize to
a disjoint-user public test set — explaining "OOF up, public down".

Test: compare OOF under StratifiedKFold vs GroupKFold (grouped by a proxy user key),
for the feature set WITH vs WITHOUT the suspect features. If the suspects help under
StratifiedKFold but NOT under GroupKFold, identity leakage is confirmed.

Uses a fast LightGBM (leakage is model-agnostic; we measure features, not the MLP).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import numpy as np
import pandas as pd
from pandas import json_normalize
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score
import features as F

NROWS = 80000
SUSPECTS = ["desc_len", "log_desc_len", "desc_has_url", "desc_has_mention",
            "loc_len", "profile_completeness", "user_bg_tile"]

print(f"Loading {NROWS} rows...")
df = pd.read_json(config.TRAIN_PATH, lines=True, nrows=NROWS)
df = json_normalize(df.to_dict(orient="records"))
y = df["label"].values.astype(int)

# Proxy user key: account creation time is EXACTLY constant per user and near-unique
# across users. Combine with user.name for extra safety.
created = df.get("user.created_at", pd.Series("", index=df.index)).fillna("").astype(str)
name = df.get("user.name", pd.Series("", index=df.index)).fillna("").astype(str)
user_key = (created + "||" + name).values

n_unique = len(set(user_key))
vc = pd.Series(user_key).value_counts()
repeated_rows = int((pd.Series(user_key).map(vc) > 1).sum())
print(f"Proxy users: {n_unique} unique keys over {len(y)} rows "
      f"({n_unique/len(y)*100:.1f}% unique)")
print(f"Rows whose user appears >1 time: {repeated_rows} ({repeated_rows/len(y)*100:.1f}%)")
print(f"Max tweets from one user: {vc.max()}; mean tweets/user: {len(y)/n_unique:.2f}")

# Build features (v2), fill -1 for the tree (trees handle sentinels fine).
X = F.build_features(df).replace([np.inf, -np.inf], np.nan).fillna(-1)
X_full = X
X_noSus = X.drop(columns=[c for c in SUSPECTS if c in X.columns])
print(f"\nFeature sets: full={X_full.shape[1]} cols, "
      f"without-suspects={X_noSus.shape[1]} cols (dropped {len(SUSPECTS)})")

params = dict(objective="binary", num_leaves=63, learning_rate=0.05, n_estimators=300,
              subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)


def cv_acc(Xm, splitter, groups=None):
    oof = np.zeros(len(y))
    it = splitter.split(Xm, y, groups) if groups is not None else splitter.split(Xm, y)
    for tr, va in it:
        m = lgb.LGBMClassifier(**params)
        m.fit(Xm.iloc[tr], y[tr])
        oof[va] = m.predict(Xm.iloc[va])
    return accuracy_score(y, oof)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gkf = GroupKFold(n_splits=5)

print("\n=== OOF accuracy (LightGBM) ===")
print(f"{'feature set':<22}{'StratifiedKFold':>18}{'GroupKFold(user)':>18}")
res = {}
for name_, Xm in [("WITH suspects", X_full), ("WITHOUT suspects", X_noSus)]:
    a_strat = cv_acc(Xm, skf)
    a_group = cv_acc(Xm, gkf, groups=user_key)
    res[name_] = (a_strat, a_group)
    print(f"{name_:<22}{a_strat*100:>17.2f}%{a_group*100:>17.2f}%")

ws, wg = res["WITH suspects"]
ns, ng = res["WITHOUT suspects"]
print("\n=== Interpretation ===")
print(f"Suspects' effect under StratifiedKFold: {(ws-ns)*100:+.2f}%  (row-wise, can leak users)")
print(f"Suspects' effect under GroupKFold(user): {(wg-ng)*100:+.2f}%  (disjoint users ~ public LB)")
print(f"Strat-vs-Group gap, WITH suspects:    {(ws-wg)*100:+.2f}%")
print(f"Strat-vs-Group gap, WITHOUT suspects: {(ns-ng)*100:+.2f}%")
if (ws - ns) > (wg - ng) + 0.003:
    print(">>> CONFIRMED: suspects help row-wise CV but NOT disjoint-user CV = identity leakage.")
else:
    print(">>> NOT confirmed: suspects are not primarily identity proxies; look elsewhere.")
