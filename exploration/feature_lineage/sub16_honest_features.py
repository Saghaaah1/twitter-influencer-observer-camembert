"""
Submission 16: Honest features only — no fake user aggregations

Since user.id/screen_name are stripped, we cannot identify users across tweets.
We use only features that are legitimately available per-tweet:

1. User profile metadata (constant per user within one tweet):
   - account_age, statuses_count, listed_count, favourites_count
   - derived ratios: tweets/day, listed/tweet, fav/tweet

2. Tweet engagement: retweet_count, favorite_count, quote/reply counts

3. Tweet metadata: is_quote, is_reply, truncated, lang

4. Text surface features: length, hashtag/mention/url count, caps_ratio

5. Quoted tweet features (quoted user has full stats including followers/friends!)

6. Profile image / banner presence (profile completeness signals)
"""
import numpy as np
import pandas as pd
from pandas import json_normalize
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from datetime import datetime

REF_DATE = datetime(2021, 6, 1)

def parse_date(s):
    if pd.isna(s) or s is None:
        return None
    try:
        return datetime.strptime(str(s), '%a %b %d %H:%M:%S +0000 %Y')
    except:
        return None

def safe_len(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0
    if isinstance(x, list): return len(x)
    return 0

def get_text(row):
    text = row.get('text', '') or ''
    ext = row.get('extended_tweet.full_text')
    if ext and not (isinstance(ext, float) and np.isnan(ext)):
        text = ext
    return str(text)

def build_features(df):
    f = pd.DataFrame(index=df.index)

    # === USER PROFILE (same across all tweets from same user) ===
    f['user_statuses_count'] = df.get('user.statuses_count', pd.Series(0, index=df.index)).fillna(0)
    f['user_listed_count'] = df.get('user.listed_count', pd.Series(0, index=df.index)).fillna(0)
    f['user_favourites_count'] = df.get('user.favourites_count', pd.Series(0, index=df.index)).fillna(0)
    f['user_geo_enabled'] = df.get('user.geo_enabled', pd.Series(False, index=df.index)).fillna(False).astype(int)
    f['user_default_profile'] = df.get('user.default_profile', pd.Series(True, index=df.index)).fillna(True).astype(int)
    f['user_default_profile_image'] = df.get('user.default_profile_image', pd.Series(True, index=df.index)).fillna(True).astype(int)
    f['user_has_url'] = df.get('user.url', pd.Series(np.nan, index=df.index)).notna().astype(int)
    f['user_has_banner'] = df.get('user.profile_banner_url', pd.Series(np.nan, index=df.index)).notna().astype(int)
    f['user_has_desc'] = df.get('user.description', pd.Series('', index=df.index)).apply(
        lambda x: 0 if (x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip() == '') else 1)
    f['user_protected'] = df.get('user.protected', pd.Series(False, index=df.index)).fillna(False).astype(int)
    f['user_contributors_enabled'] = df.get('user.contributors_enabled', pd.Series(False, index=df.index)).fillna(False).astype(int)
    f['user_geo_enabled'] = df.get('user.geo_enabled', pd.Series(False, index=df.index)).fillna(False).astype(int)
    f['user_is_translator'] = df.get('user.is_translator', pd.Series(False, index=df.index)).fillna(False).astype(int)
    f['user_profile_use_bg_image'] = df.get('user.profile_use_background_image', pd.Series(True, index=df.index)).fillna(True).astype(int)
    f['user_has_location'] = df.get('user.location', pd.Series(np.nan, index=df.index)).apply(
        lambda x: 0 if (x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip() == '') else 1)

    # Account age
    user_created = df.get('user.created_at', pd.Series(np.nan, index=df.index))
    f['user_account_age_days'] = user_created.apply(
        lambda s: max(0, (REF_DATE - parse_date(s)).days) if parse_date(s) else 0)

    # Log transforms
    f['log_age'] = np.log1p(f['user_account_age_days'])
    f['log_statuses'] = np.log1p(f['user_statuses_count'])
    f['log_listed'] = np.log1p(f['user_listed_count'])
    f['log_favs'] = np.log1p(f['user_favourites_count'])

    # Behavioral ratios
    f['tweets_per_day'] = f['user_statuses_count'] / (f['user_account_age_days'] + 1)
    f['listed_per_tweet'] = f['user_listed_count'] / (f['user_statuses_count'] + 1)
    f['fav_per_tweet'] = f['user_favourites_count'] / (f['user_statuses_count'] + 1)
    f['listed_per_day'] = f['user_listed_count'] / (f['user_account_age_days'] + 1)
    f['log_tweets_per_day'] = np.log1p(f['tweets_per_day'])
    f['log_listed_per_tweet'] = np.log1p(f['listed_per_tweet'])
    f['log_fav_per_tweet'] = np.log1p(f['fav_per_tweet'])

    # === TWEET ENGAGEMENT ===
    f['retweet_count'] = df.get('retweet_count', pd.Series(0, index=df.index)).fillna(0)
    f['favorite_count'] = df.get('favorite_count', pd.Series(0, index=df.index)).fillna(0)
    f['quote_count'] = df.get('quote_count', pd.Series(0, index=df.index)).fillna(0)
    f['reply_count'] = df.get('reply_count', pd.Series(0, index=df.index)).fillna(0)
    f['engagement'] = f['retweet_count'] + f['favorite_count'] + f['quote_count'] + f['reply_count']
    f['log_rt'] = np.log1p(f['retweet_count'])
    f['log_fav_tweet'] = np.log1p(f['favorite_count'])
    f['log_engagement'] = np.log1p(f['engagement'])

    # === TWEET METADATA ===
    f['is_quote'] = df.get('is_quote_status', pd.Series(False, index=df.index)).fillna(False).astype(int)
    f['is_reply'] = df.get('in_reply_to_status_id', pd.Series(np.nan, index=df.index)).notna().astype(int)
    f['truncated'] = df.get('truncated', pd.Series(False, index=df.index)).fillna(False).astype(int)

    lang_map = {'fr': 0, 'en': 1, 'es': 2, 'ar': 3, 'de': 4, 'pt': 5, 'it': 6, 'und': 7, 'ja': 8, 'tr': 9}
    f['lang_enc'] = df.get('lang', pd.Series('fr', index=df.index)).fillna('fr').map(lambda x: lang_map.get(x, 99))

    # === TEXT SURFACE FEATURES ===
    texts = df.apply(get_text, axis=1)
    f['text_len'] = texts.str.len().fillna(0)
    f['word_count'] = texts.str.split().str.len().fillna(0)
    f['hashtag_count'] = texts.str.count(r'#\w+')
    f['mention_count'] = texts.str.count(r'@\w+')
    f['url_count'] = texts.str.count(r'http')
    f['caps_ratio'] = texts.apply(lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1))
    f['is_rt'] = texts.str.startswith('RT ').astype(int)
    f['exclaim_count'] = texts.str.count('!')
    f['question_count'] = texts.str.count(r'\?')
    f['avg_word_len'] = texts.apply(lambda t: np.mean([len(w) for w in t.split()]) if t.split() else 0)
    f['unique_words'] = texts.apply(lambda t: len(set(t.lower().split())) if t.split() else 0)
    f['type_token_ratio'] = f['unique_words'] / (f['word_count'] + 1)

    # === ENTITIES ===
    if 'entities.hashtags' in df.columns:
        f['ent_hashtags'] = df['entities.hashtags'].apply(safe_len)
        f['ent_mentions'] = df['entities.user_mentions'].apply(safe_len) if 'entities.user_mentions' in df.columns else pd.Series(0, index=df.index)
        f['ent_urls'] = df['entities.urls'].apply(safe_len) if 'entities.urls' in df.columns else pd.Series(0, index=df.index)
        f['ent_media'] = df['entities.media'].apply(safe_len) if 'entities.media' in df.columns else pd.Series(0, index=df.index)

    # === QUOTED TWEET FEATURES (quoted user HAS followers/friends!) ===
    if 'quoted_status.user.followers_count' in df.columns:
        q_fol = df['quoted_status.user.followers_count'].fillna(-1)
        q_fri = df['quoted_status.user.friends_count'].fillna(-1)
        f['quoted_followers'] = q_fol
        f['quoted_friends'] = q_fri
        f['quoted_listed'] = df.get('quoted_status.user.listed_count', pd.Series(-1, index=df.index)).fillna(-1)
        f['quoted_statuses'] = df.get('quoted_status.user.statuses_count', pd.Series(-1, index=df.index)).fillna(-1)
        f['quoted_verified'] = df.get('quoted_status.user.verified', pd.Series(False, index=df.index)).fillna(False).astype(int)
        f['quoted_default_profile'] = df.get('quoted_status.user.default_profile', pd.Series(True, index=df.index)).fillna(True).astype(int)
        f['quoted_fol_ratio'] = q_fol / (q_fri + 1)
        f['log_q_fol'] = np.log1p(q_fol.clip(0))
        f['log_q_fri'] = np.log1p(q_fri.clip(0))
        f['quoted_rt'] = df.get('quoted_status.retweet_count', pd.Series(-1, index=df.index)).fillna(-1)
        f['quoted_fav'] = df.get('quoted_status.favorite_count', pd.Series(-1, index=df.index)).fillna(-1)
        f['quoted_reply'] = df.get('quoted_status.reply_count', pd.Series(-1, index=df.index)).fillna(-1)
        f['log_q_rt'] = np.log1p(f['quoted_rt'].clip(0))
        f['log_q_fav'] = np.log1p(f['quoted_fav'].clip(0))
        f['quoted_engagement'] = (f['quoted_rt'] + f['quoted_fav'] + f['quoted_reply']).clip(-1)
        f['log_q_engagement'] = np.log1p(f['quoted_engagement'].clip(0))

        # Quoted user account age
        q_created = df.get('quoted_status.user.created_at', pd.Series(np.nan, index=df.index))
        f['quoted_user_age_days'] = q_created.apply(
            lambda s: max(0, (REF_DATE - parse_date(s)).days) if parse_date(s) else -1)
        q_statuses = df.get('quoted_status.user.statuses_count', pd.Series(-1, index=df.index)).fillna(-1)
        f['quoted_tweets_per_day'] = q_statuses / (f['quoted_user_age_days'] + 1)
        f['quoted_listed_per_tweet'] = f['quoted_listed'] / (q_statuses + 1)

        # Ratio: author stats vs quoted user stats
        f['author_vs_quoted_fol'] = f['log_statuses'] / (f['log_q_fol'] + 1)

    return f.fillna(-1).replace([np.inf, -np.inf], -1)


def main():
    print("Loading data...")
    train = pd.read_json('train.jsonl', lines=True)
    train = json_normalize(train.to_dict(orient='records'))
    test = pd.read_json('kaggle_test.jsonl', lines=True)
    test = json_normalize(test.to_dict(orient='records'))

    y = train['label'].values
    print(f"Train: {train.shape}, Labels: {np.bincount(y)}")

    print("Building features...")
    X_train = build_features(train)
    X_test = build_features(test)
    print(f"Feature matrix: {X_train.shape}")
    print("Features:", list(X_train.columns))

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
    test_probs = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(kfold.split(X_train, y)):
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train.iloc[tr_idx], y[tr_idx],
                  eval_set=[(X_train.iloc[val_idx], y[val_idx])],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])
        oof_preds[val_idx] = model.predict(X_train.iloc[val_idx])
        test_probs += model.predict_proba(X_test)[:, 1] / 5
        print(f"  Fold {fold+1}: {accuracy_score(y[val_idx], oof_preds[val_idx])*100:.2f}%")

    cv_acc = accuracy_score(y, oof_preds)
    print(f"OOF CV Accuracy: {cv_acc*100:.2f}%")

    final_preds = (test_probs > 0.5).astype(int)
    output = pd.DataFrame({'ID': test['challenge_id'], 'Prediction': final_preds})
    output.to_csv('sub16_honest_features.csv', index=False)
    print("Saved sub16_honest_features.csv")

    np.save('sub16_test_probs.npy', test_probs)

    importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\nTop 20 features:")
    print(importance.head(20).to_string())

if __name__ == '__main__':
    main()
