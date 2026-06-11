"""
Submission 19: All honest features + source app encoding

Source (the app used to tweet) is extremely predictive:
- TweetDeck/Hootsuite/Buffer/Echobox = 87-100% influencer
- Twitter for Android = 32% influencer
- Twitter for Mac / unknown = 14-41% influencer

This is legitimate signal: influencers use professional tools.
"""
import numpy as np
import pandas as pd
from pandas import json_normalize
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from datetime import datetime
import re, sys
sys.path.insert(0, '.')
from sub16_honest_features import build_features

REF_DATE = datetime(2021, 6, 1)

def parse_source(s):
    if not s or (isinstance(s, float) and np.isnan(s)):
        return 'unknown'
    m = re.search(r'>([^<]+)<', str(s))
    return m.group(1).strip() if m else str(s)[:50]

# Source influencer rates from training data analysis
SOURCE_INFLUENCER_RATE = {
    'Twitter for iPhone': 0.537,
    'Twitter Web App': 0.458,
    'Twitter for Android': 0.322,
    'Twitter for iPad': 0.291,
    'TweetDeck': 0.872,
    'Hootsuite Inc.': 0.910,
    'dlvr.it': 0.901,
    'IFTTT': 0.698,
    'WordPress.com': 0.767,
    'Buffer': 0.876,
    'Echobox': 1.000,
    'unknown': 0.137,
    'AgoraPulse Manager': 0.922,
    'Twitter Media Studio': 1.000,
    'Paper.li': 0.368,
    'Twitter for Mac': 0.415,
    'Sprout Social': 1.000,
    'Zapier.com': 0.784,
}

def add_source_features(df, X):
    sources = df.get('source', pd.Series('', index=df.index)).fillna('')
    parsed = sources.apply(parse_source)

    # Source influencer rate (pre-computed from training, safe to use on test)
    global_mean = 0.467
    X = X.copy()
    X['source_influencer_rate'] = parsed.map(SOURCE_INFLUENCER_RATE).fillna(global_mean)

    # Is professional tool?
    pro_tools = {'TweetDeck', 'Hootsuite Inc.', 'dlvr.it', 'Buffer', 'Echobox',
                 'AgoraPulse Manager', 'Twitter Media Studio', 'Sprout Social',
                 'Zapier.com', 'IFTTT', 'WordPress.com'}
    X['source_is_pro'] = parsed.isin(pro_tools).astype(int)

    # Main apps encoding
    source_map = {
        'Twitter for iPhone': 0, 'Twitter Web App': 1, 'Twitter for Android': 2,
        'Twitter for iPad': 3, 'TweetDeck': 4, 'Hootsuite Inc.': 5,
        'dlvr.it': 6, 'unknown': 7
    }
    X['source_enc'] = parsed.map(source_map).fillna(99)

    # Timestamp features
    if 'timestamp_ms' in df.columns:
        ts = pd.to_numeric(df['timestamp_ms'], errors='coerce') / 1000
        dt = pd.to_datetime(ts, unit='s', utc=True, errors='coerce')
        X['hour_of_day'] = dt.dt.hour.fillna(-1)
        X['day_of_week'] = dt.dt.dayofweek.fillna(-1)
        X['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)

    return X

def main():
    print("Loading data...")
    train = pd.read_json('train.jsonl', lines=True)
    train = json_normalize(train.to_dict(orient='records'))
    test = pd.read_json('kaggle_test.jsonl', lines=True)
    test = json_normalize(test.to_dict(orient='records'))

    y = train['label'].values
    print(f"Labels: {np.bincount(y)}")

    print("Building features...")
    X_tr = build_features(train)
    X_te = build_features(test)

    print("Adding source features...")
    X_tr = add_source_features(train, X_tr)
    X_te = add_source_features(test, X_te)

    X_tr = X_tr.replace([np.inf, -np.inf], np.nan).fillna(-1)
    X_te = X_te.replace([np.inf, -np.inf], np.nan).fillna(-1)
    print(f"Feature matrix: {X_tr.shape}")

    params = {
        'objective': 'binary',
        'metric': 'binary_error',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'n_estimators': 2000,
        'min_child_samples': 30,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    test_probs = np.zeros(len(X_te))

    for fold, (tr_idx, val_idx) in enumerate(kfold.split(X_tr, y)):
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr.iloc[tr_idx], y[tr_idx],
                  eval_set=[(X_tr.iloc[val_idx], y[val_idx])],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])
        oof_preds[val_idx] = model.predict(X_tr.iloc[val_idx])
        test_probs += model.predict_proba(X_te)[:, 1] / 5
        print(f"  Fold {fold+1}: {accuracy_score(y[val_idx], oof_preds[val_idx])*100:.2f}%")

    cv_acc = accuracy_score(y, oof_preds)
    print(f"OOF CV Accuracy: {cv_acc*100:.2f}%")

    final_preds = (test_probs > 0.5).astype(int)
    output = pd.DataFrame({'ID': test['challenge_id'], 'Prediction': final_preds})
    output.to_csv('sub19_with_source.csv', index=False)
    print("Saved sub19_with_source.csv")
    np.save('sub19_test_probs.npy', test_probs)

    importance = pd.Series(model.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
    print("\nTop 15 features:")
    print(importance.head(15).to_string())

if __name__ == '__main__':
    main()
