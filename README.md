# Influencer or Observer: Predicting Social Roles

## Team

## Setup

```bash
conda activate sara
# or pip install pandas scikit-learn lightgbm nltk scipy
```

## Data

Place `train.jsonl` and `kaggle_test.jsonl` in the same directory as the scripts.

## Running the best model

```bash
python sub19_with_source.py       # Best honest unsubmitted candidate
```

## Model progression (CV accuracy)

| Script | Approach | CV Acc |
|--------|----------|--------|
| sub1_tfidf_lr.py | TF-IDF + Logistic Regression | 67.1% |
| sub2_tfidf_lgbm.py | TF-IDF + LightGBM | 63.5% |
| sub3_features_lgbm.py | Metadata features + LightGBM | 87.47% |
| sub4_combined_lgbm.py | SVD text + metadata | 85.3% |
| sub5_user_agg.py | User aggregations + LightGBM | 91.14% |
| sub6_user_label_enc.py | User label enc + SVD | 85.3% |
| sub7_rich_features.py | Rich features + OOF user target enc | 93.18% |
| sub9_best_features.py | Rich features v2 | 92.93% |
| sub10_user_agg_v2.py | Maximal user aggregations + LightGBM | **94.40%** |
| sub12_final_best.py | All features + user agg | 93.99% |
| sub13_ensemble_best.py | Ensemble of sub7+sub9+sub10+sub12 | ~94.5% |
| sub15_final_push.py | Multi-seed LightGBM | ~94.5%+ |
| sub19_with_source.py | Honest base features + source app signal | ~90.5% CV |
| sub20_nested_source_fingerprint.py | Profile fingerprint TE; CV likely identity-proxy inflated | 93.40% |
| sub21_rich_profile_source.py | Profile color/source categorical TE; CV inflated, public 0.839 | 97.48% CV / 0.839 public |

## Key Findings

### What works (and why)

1. **User profile metadata is the strongest signal** (87% from metadata alone):
   - `user_account_age_days`: Older accounts are more likely influencers
   - `user_statuses_count`: More tweets = influencer signal
   - `user_favourites_count` / tweet = behavioural ratio
   - `tweets_per_day`: Activity level

2. **Important correction: direct user IDs are stripped**:
   - `user.id`, `user.id_str`, and `user.screen_name` are not present.
   - The top-level `id_str` is the tweet id, not a user id.
   - Earlier "user aggregation" scripts therefore do not truly aggregate by user.
   - Profile fingerprints are mostly unique in train, so they add less than expected.

3. **Source app is the safest new signal**:
   - Tweet source app has real signal: TweetDeck/Hootsuite/Buffer-style tools are more common for influencers.
   - Profile color/background fields are user-constant and can act as identity proxies under row-wise CV.
   - `sub21_rich_profile_source.py` reached 97.48% CV but only 0.839 public, so treat that CV as invalid.

4. **Text features add little (~63% alone)**:
   - The label is about WHO tweets, not WHAT they tweet
   - SVD-compressed text actually *hurts* when combined with metadata (85% vs 87%)
   - Key text signal: caps_ratio, text_len, hashtag/mention counts

5. **Quoted tweet features provide indirect follower signals**:
   - `quoted_status.user.followers_count`: Present! Direct followers/friends 
     of main user are stripped, but quoted user's stats are available
   - High-follower quoted users indicate influencer behavior

### Feature Engineering Details

```python
# Most important features (from sub10 LightGBM):
user_account_age_days   # How long the user has been on Twitter
fav_per_tweet           # Favourites per tweet ratio
user_favourites_count   # Total favourites
tweets_per_day          # Activity level
user_statuses_count     # Total tweets ever
listed_per_tweet        # Listed count per tweet
caps_ratio              # Writing style signal
d_caps_ratio            # Deviation from user's own mean caps ratio
```

### Model Architecture

- **LightGBM** with 255 leaves, early stopping
- **5-fold stratified cross-validation** with out-of-fold predictions
- **Ensemble**: weighted average by CV score
