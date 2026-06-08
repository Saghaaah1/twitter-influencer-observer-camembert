"""Label-FREE within-dataset user aggregation (keyed by user.created_at proxy).

This is NOT the leaky 'user target encoding' of sub10/sub21 (those used the LABEL).
Here we aggregate only label-free behavioral features across a user's own tweets, which
are all present at inference (test has ~3.2 tweets/user too). Under GroupKFold a user is
wholly inside one fold, so a val user's aggregates use only that user's val rows -> no
train->val leak. Question: does denoising per-tweet signals to per-user lift the ceiling?
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import numpy as np
import pandas as pd
from pandas import json_normalize
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score
import features as F

NROWS = 120000
print(f"Loading {NROWS} rows...")
df = pd.read_json(config.TRAIN_PATH, lines=True, nrows=NROWS)
df = json_normalize(df.to_dict(orient="records"))
y = df["label"].values.astype(int)
ukey = df.get("user.created_at", pd.Series("", index=df.index)).fillna("").astype(str)
groups = ukey.values

Xb = F.build_features(df).replace([np.inf, -np.inf], np.nan)
parsed_src = F.parse_source_series(df)
Xb = F.add_static_source_features(parsed_src, df, Xb)
txt = df.apply(F.get_text, axis=1).fillna("").astype(str)

# Per-tweet raw signals to aggregate
tw = pd.DataFrame(index=df.index)
tw["caps"] = txt.apply(lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1))
tw["tlen"] = txt.str.len().fillna(0)
tw["is_rt"] = txt.str.startswith("RT ").astype(int)
tw["has_url"] = (txt.str.count(r"http") > 0).astype(int)
tw["is_quote"] = df.get("is_quote_status", pd.Series(False, index=df.index)).fillna(False).astype(int)
tw["src"] = parsed_src.values
tw["uk"] = ukey.values

# Label-free user aggregates (computed on the WHOLE df; each user wholly in one fold).
g = tw.groupby("uk")
agg = pd.DataFrame(index=df.index)
agg["u_tweet_count"] = g["uk"].transform("size")
agg["u_caps_mean"] = g["caps"].transform("mean")
agg["u_caps_std"] = g["caps"].transform("std").fillna(0)
agg["u_tlen_mean"] = g["tlen"].transform("mean")
agg["u_tlen_std"] = g["tlen"].transform("std").fillna(0)
agg["u_rt_frac"] = g["is_rt"].transform("mean")
agg["u_url_frac"] = g["has_url"].transform("mean")
agg["u_quote_frac"] = g["is_quote"].transform("mean")
agg["u_n_sources"] = g["src"].transform("nunique")
# deviation of this tweet from the user's own mean (style consistency)
agg["d_caps"] = tw["caps"].values - agg["u_caps_mean"].values
agg["d_tlen"] = tw["tlen"].values - agg["u_tlen_mean"].values

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
params = dict(objective="binary", num_leaves=63, learning_rate=0.05, n_estimators=300,
              subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)


def cv(extra):
    oof = np.zeros(len(y))
    for tr, va in sgkf.split(Xb, y, groups):
        smap, sgm = F.compute_source_te(parsed_src.iloc[tr].values, y[tr])
        X = Xb.copy(); X["source_te"] = F.apply_source_te(parsed_src, smap, sgm)
        if extra is not None:
            X = pd.concat([X, extra], axis=1)
        X = X.fillna(-1)
        m = lgb.LGBMClassifier(**params); m.fit(X.iloc[tr], y[tr]); oof[va] = m.predict(X.iloc[va])
    return accuracy_score(y, oof)


base = cv(None)
full = cv(agg)
print(f"\nbaseline                    : {base*100:.2f}%")
print(f"+ user aggregates (label-free): {full*100:.2f}%  ({(full-base)*100:+.2f})")
print(f"\nmean tweets/user in sample: {agg['u_tweet_count'].mean():.2f}, "
      f"max: {agg['u_tweet_count'].max():.0f}")
