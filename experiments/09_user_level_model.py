"""USER-LEVEL modeling. The label is 100% user-constant (0 mixed groups), so this is a
user-classification task: aggregate each user's ~5 tweets into ONE feature row, classify
users, broadcast the prediction to all their tweets. Aggregation denoises per-tweet noise
and adds cross-tweet stats (n_tweets, source diversity, RT fraction) the tweet-level model
can't see. Honest: grouping key (user.created_at) is available at inference; no labels used
in aggregation. One row per user -> plain StratifiedKFold over users is leak-free.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import config
import numpy as np
import pandas as pd
from pandas import json_normalize
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import features as F

NROWS = 154914
print(f"Loading {NROWS} rows...")
df = pd.read_json(config.TRAIN_PATH, lines=True, nrows=NROWS)
df = json_normalize(df.to_dict(orient="records"))
y_row = df["label"].values.astype(int)
ukey = df.get("user.created_at", pd.Series("", index=df.index)).fillna("").astype(str).values

print("Building tweet-level honest features...")
X = F.build_features(df).replace([np.inf, -np.inf], np.nan)
parsed_src = F.parse_source_series(df)
X = F.add_static_source_features(parsed_src, df, X)
X["_uk"] = ukey
X["_src"] = parsed_src.values

# Columns that are USER-CONSTANT vs per-tweet VARYING (decide aggregation).
VARYING = ["caps_ratio", "text_len", "word_count", "hashtag_count", "mention_count",
           "url_count", "exclaim_count", "question_count", "avg_word_len", "unique_words",
           "type_token_ratio", "is_rt", "is_quote", "is_reply", "hour_sin", "hour_cos",
           "ent_hashtags", "ent_mentions", "ent_urls", "ent_media", "has_quoted"]
feat_cols = [c for c in X.columns if c not in ("_uk", "_src")]
const_cols = [c for c in feat_cols if c not in VARYING]
varying_present = [c for c in VARYING if c in feat_cols]

print("Aggregating to user level...")
g = X.groupby("_uk", sort=False)
# constant features: first value (they're identical within user)
agg = g[const_cols].first()
# varying features: mean + std
vmean = g[varying_present].mean().add_suffix("_mean")
vstd = g[varying_present].std().fillna(0).add_suffix("_std")
# cross-tweet stats
extra = pd.DataFrame(index=agg.index)
extra["u_n_tweets"] = g.size()
extra["u_n_sources"] = g["_src"].nunique()
extra["u_frac_rt"] = g["is_rt"].mean() if "is_rt" in X.columns else 0
extra["u_frac_quote"] = g["is_quote"].mean() if "is_quote" in X.columns else 0
extra["u_n_hours"] = g["hour_sin"].nunique() if "hour_sin" in X.columns else 0
Xu = pd.concat([agg, vmean, vstd, extra], axis=1)

# user label (constant) — first label per user, aligned to Xu.index order
ylabel = pd.Series(y_row, index=ukey).groupby(level=0).first()
yu = ylabel.reindex(Xu.index).values.astype(int)
user_src = g["_src"].agg(lambda s: s.value_counts().index[0])  # modal source per user
user_src = user_src.reindex(Xu.index)
print(f"Users: {len(Xu)}  feature dim: {Xu.shape[1]}  pos rate {yu.mean():.3f}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
params = dict(objective="binary", num_leaves=63, learning_rate=0.05, n_estimators=400,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
              random_state=42, n_jobs=-1, verbose=-1)

oof = np.zeros(len(Xu))
src_arr = user_src.values
for tr, va in skf.split(Xu, yu):
    smap, sgm = F.compute_source_te(src_arr[tr], yu[tr])
    Xt = Xu.copy()
    Xt["u_source_te"] = F.apply_source_te(src_arr, smap, sgm)
    Xt = Xt.fillna(-1)
    m = lgb.LGBMClassifier(**params)
    m.fit(Xt.iloc[tr], yu[tr])
    oof[va] = m.predict_proba(Xt.iloc[va])[:, 1]

# user-level accuracy (best thr) and broadcast row-level accuracy
def bestacc(p, yy):
    b = 0.0
    for t in np.arange(0.40, 0.601, 0.005):
        b = max(b, accuracy_score(yy, (p > t).astype(int)))
    return b

user_acc = bestacc(oof, yu)
# broadcast to rows
uidx = {k: i for i, k in enumerate(Xu.index)}
row_p = np.array([oof[uidx[k]] for k in ukey])
row_acc = bestacc(row_p, y_row)
print(f"\n=== USER-LEVEL LightGBM (StratifiedKFold over users) ===")
print(f"user-level OOF acc : {user_acc*100:.2f}%")
print(f"broadcast row acc  : {row_acc*100:.2f}%   (tweet-level blend+smooth was 84.78%)")
