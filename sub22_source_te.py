"""
Submission 22: sub19 honest features + fold-safe target encoding for source only.

Source app (tweet-level, not user-constant) is the one legitimate TE candidate.
Profile colors were sub21's mistake — they're user-constant and leak identity.
Source is tweet-level: the same user can tweet from iPhone AND TweetDeck.
"""
import re
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd
from pandas import json_normalize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold

from sub16_honest_features import build_features


def parse_source(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "unknown"
    match = re.search(r">([^<]+)<", str(s))
    return match.group(1).strip() if match else str(s)[:80]


def make_source_keys(df):
    return df.get("source", pd.Series("", index=df.index)).fillna("").apply(parse_source).astype(str).values


def add_source_base_features(df, X, top_sources):
    X = X.copy()
    sources = pd.Series(make_source_keys(df), index=df.index)
    source_map = {src: i for i, src in enumerate(top_sources)}

    pro_tools = {
        "TweetDeck", "Hootsuite Inc.", "dlvr.it", "Buffer", "Echobox",
        "AgoraPulse Manager", "Twitter Media Studio", "Sprout Social",
        "Zapier.com", "IFTTT", "WordPress.com", "SocialFlow",
    }

    X["source_enc"] = sources.map(source_map).fillna(999).astype(int)
    X["source_is_pro"] = sources.isin(pro_tools).astype(int)
    X["source_len"] = sources.str.len().astype(int)

    if "timestamp_ms" in df.columns:
        ts = pd.to_numeric(df["timestamp_ms"], errors="coerce") / 1000
        dt = pd.to_datetime(ts, unit="s", utc=True, errors="coerce")
        X["hour_of_day"] = dt.dt.hour.fillna(-1).astype(int)
        X["day_of_week"] = dt.dt.dayofweek.fillna(-1).astype(int)
        X["is_weekend"] = (dt.dt.dayofweek >= 5).fillna(False).astype(int)

    return X


def _stats(keys, y, prior, smoothing=20.0):
    sums = defaultdict(float)
    counts = defaultdict(int)
    for key, label in zip(keys, y):
        sums[key] += float(label)
        counts[key] += 1
    means = {k: (sums[k] + prior * smoothing) / (counts[k] + smoothing) for k in counts}
    return means, counts


def _transform(keys, means, counts, prior):
    encoded = np.array([means.get(k, prior) for k in keys], dtype=float)
    count_vals = np.array([counts.get(k, 0) for k in keys], dtype=float)
    return encoded, count_vals


def make_oof(keys, y, splits=5, smoothing=20.0):
    prior = float(np.mean(y))
    out = np.full(len(y), prior, dtype=float)
    kfold = KFold(n_splits=splits, shuffle=True, random_state=42)
    for tr_idx, val_idx in kfold.split(y):
        means, counts = _stats(keys[tr_idx], y[tr_idx], prior, smoothing)
        out[val_idx] = _transform(keys[val_idx], means, counts, prior)[0]
    return out


def add_fold_source_te(X_fit, X_val, X_test, y_fit, fit_keys, val_keys, test_keys):
    X_fit = X_fit.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    prior = float(np.mean(y_fit))

    X_fit["source_te"] = make_oof(fit_keys, y_fit)
    fit_counts = pd.Series(fit_keys).map(pd.Series(fit_keys).value_counts()).fillna(0).values
    X_fit["source_te_count"] = fit_counts

    means, counts = _stats(fit_keys, y_fit, prior)
    X_val["source_te"], X_val["source_te_count"] = _transform(val_keys, means, counts, prior)
    X_test["source_te"], X_test["source_te_count"] = _transform(test_keys, means, counts, prior)

    return X_fit, X_val, X_test


def main():
    print("Loading data...")
    train = pd.read_json("train.jsonl", lines=True)
    train = json_normalize(train.to_dict(orient="records"))
    test = pd.read_json("kaggle_test.jsonl", lines=True)
    test = json_normalize(test.to_dict(orient="records"))
    y = train["label"].values
    print(f"Train: {train.shape}, Test: {test.shape}, Labels: {np.bincount(y)}")

    print("Building features...")
    X_tr = build_features(train)
    X_te = build_features(test)

    train_source_keys = make_source_keys(train)
    test_source_keys = make_source_keys(test)
    top_sources = pd.Series(train_source_keys).value_counts().head(50).index

    X_tr = add_source_base_features(train, X_tr, top_sources)
    X_te = add_source_base_features(test, X_te, top_sources)
    print(f"Base features: {X_tr.shape}")

    params = {
        "objective": "binary",
        "metric": "binary_error",
        "num_leaves": 95,
        "learning_rate": 0.04,
        "n_estimators": 4500,
        "min_child_samples": 25,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.15,
        "reg_lambda": 0.2,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y))
    test_probs = np.zeros(len(X_te))

    for fold, (tr_idx, val_idx) in enumerate(kfold.split(X_tr, y), start=1):
        X_fit, X_val, X_test_fold = add_fold_source_te(
            X_tr.iloc[tr_idx], X_tr.iloc[val_idx], X_te, y[tr_idx],
            train_source_keys[tr_idx], train_source_keys[val_idx], test_source_keys,
        )

        X_fit = X_fit.replace([np.inf, -np.inf], np.nan).fillna(-1)
        X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(-1)
        X_test_fold = X_test_fold.replace([np.inf, -np.inf], np.nan).fillna(-1)

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_fit, y[tr_idx],
            eval_set=[(X_val, y[val_idx])],
            callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(500)],
        )
        oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]
        test_probs += model.predict_proba(X_test_fold)[:, 1] / 5
        acc = accuracy_score(y[val_idx], (oof_probs[val_idx] > 0.5).astype(int))
        print(f"  Fold {fold}: {acc * 100:.2f}%")

    cv_acc = accuracy_score(y, (oof_probs > 0.5).astype(int))
    print(f"OOF CV Accuracy: {cv_acc * 100:.2f}%")

    output = pd.DataFrame({
        "ID": test["challenge_id"],
        "Prediction": (test_probs > 0.5).astype(int),
    })
    output.to_csv("sub22_source_te.csv", index=False)
    np.save("sub22_test_probs.npy", test_probs)
    np.save("sub22_oof_probs.npy", oof_probs)
    print("Saved sub22_source_te.csv")

    importance = pd.Series(model.feature_importances_, index=X_fit.columns).sort_values(ascending=False)
    print("\nTop 20 features:")
    print(importance.head(20).to_string())


if __name__ == "__main__":
    main()
