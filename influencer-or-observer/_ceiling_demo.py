"""Why CV accuracy can look ~100% but public is capped ~0.856.
Three numbers from the SAME features/model:
  (1) memorize-train: fit on all train, predict train  -> can be ~99% (overfit)
  (2) row-wise CV (StratifiedKFold): same user spans folds -> inflated
  (3) honest CV (StratifiedGroupKFold by user): disjoint users ~ the public LB
The gap between (1)/(2) and (3) is identity memorization that does NOT transfer to
strangers. The public test is 100% strangers, so only (3) predicts it.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from pandas import json_normalize
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score
import config, features as F

df = json_normalize(pd.read_json(config.TRAIN_PATH, lines=True).to_dict(orient="records"))
y = df["label"].values.astype(int)
groups = df.get("user.created_at", pd.Series("", index=df.index)).fillna("").astype(str).values
X = F.build_features(df).replace([np.inf, -np.inf], np.nan)
ps = F.parse_source_series(df)
X = F.add_static_source_features(ps, df, X)
X["source_te"] = F.apply_source_te(ps, *F.compute_source_te(ps.values, y))  # simple TE for demo
X = X.fillna(-1)

# (1) memorize: high-capacity, low-reg model fit and scored on the SAME rows
mem = lgb.LGBMClassifier(objective="binary", num_leaves=255, n_estimators=800,
                         learning_rate=0.1, min_child_samples=1, reg_lambda=0,
                         random_state=42, n_jobs=-1, verbose=-1).fit(X, y)
print(f"(1) memorize-train accuracy        : {accuracy_score(y, mem.predict(X))*100:.2f}%")

params = dict(objective="binary", num_leaves=63, n_estimators=400, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=42,
              n_jobs=-1, verbose=-1)


def oof(splitter, g=None):
    o = np.zeros(len(y))
    it = splitter.split(X, y, g) if g is not None else splitter.split(X, y)
    for tr, va in it:
        m = lgb.LGBMClassifier(**params).fit(X.iloc[tr], y[tr])
        o[va] = m.predict(X.iloc[va])
    return accuracy_score(y, o)


a_row = oof(StratifiedKFold(5, shuffle=True, random_state=42))
a_grp = oof(StratifiedGroupKFold(5, shuffle=True, random_state=42), groups)
print(f"(2) row-wise CV (same user in folds): {a_row*100:.2f}%   <- inflated, like sub21's 97%")
print(f"(3) honest CV (disjoint users)      : {a_grp*100:.2f}%   <- this predicts the public LB")
print(f"\nGap (1)->(3) = {(accuracy_score(y, mem.predict(X))-a_grp)*100:.1f} pts of identity memorization")
print("Public test users are 100% strangers -> you score (3), not (1). That's the ceiling.")
