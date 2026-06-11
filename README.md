# Influencer or Observer — Predicting Social Roles

Binary classification of French Twitter accounts (**influencer** vs. **observer**) from a
single tweet's metadata and text. **Final result: public leaderboard 0.856 — tied for #1.**

The whole approach is built around one principle: **don't fool yourself with identity
leakage.** Direct user IDs are stripped from the data and the public test set is made of
users never seen in training, so only signal that generalizes across *different* users
counts. Every design choice keeps the cross-validation estimate honest and aligned with the
public leaderboard.

---

## Repository layout

```
.
├── src/                          # production pipeline (run these)
│   ├── config.py                 #   single source of truth: paths, seed, hyperparameters
│   ├── features.py               #   honest feature engineering + fold-safe source target encoding
│   ├── model.py                  #   TabularMLP (256 -> 128 -> 64 -> 1, BatchNorm + GELU + Dropout)
│   ├── train.py                  #   tabular pipeline: CV -> MLP + LightGBM blend -> smooth -> submission
│   ├── stack_models.py           #   adds CatBoost / XGBoost / HistGBM / ExtraTrees to the ensemble
│   ├── finetune_camembert.py     #   fine-tune CamemBERT on tweet text
│   ├── finetune_camembert_bio.py #   fine-tune CamemBERT on bio + tweet
│   └── make_submission.py        #   blend all model predictions + per-user smoothing -> final CSV
├── experiments/                  # 12 numbered probes — the evidence behind every decision
├── exploration/                  # initial data-understanding scripts + feature_lineage/
├── data/                         # train.jsonl / kaggle_test.jsonl (place here; gitignored)
└── output/                       # submissions + model_predictions/ (prob arrays); gitignored
```

---

## Setup

```bash
conda activate sara     # or: pip install pandas scikit-learn lightgbm torch transformers catboost xgboost
# place train.jsonl and kaggle_test.jsonl in data/
```

## Reproduce the 0.856 submission

```bash
cd src
python train.py --group --blend --smooth   # MLP + LightGBM, honest CV, per-user smoothing
python stack_models.py                      # add CatBoost / XGBoost / HistGBM / ExtraTrees
python finetune_camembert.py                # fine-tune CamemBERT on tweet text    (GPU, ~1 hr)
python finetune_camembert_bio.py            # fine-tune CamemBERT on bio + tweet    (GPU, ~1 hr)
python make_submission.py                   # blend everything -> output/submission_final.csv
```

Each model writes aligned `<name>_oof_probs.npy` / `<name>_test_probs.npy` into
`output/model_predictions/`. `make_submission.py` auto-discovers them, selects ensemble
weights by greedy search on the per-user-smoothed out-of-fold (OOF) predictions, and writes
the final submission. Quick sanity check: `python train.py --smoke`.

---

## How it got to 0.856

| Stage | honest OOF (GroupKFold) | public LB |
|-------|-------------------------|-----------|
| MLP only | 83.94% | — |
| + LightGBM blend (α = 0.55) | 84.31% | — |
| + per-user smoothing (label is user-constant) | 84.78% | 0.848 |
| + CatBoost / HistGBM / ExtraTrees stack | 84.84% | 0.850 |
| + fine-tuned CamemBERT (tweet) | 85.14% | 0.852 |
| **+ CamemBERT (bio + tweet)** | **85.40%** | **0.856** |

0.856 is the data's honest ceiling — the whole leaderboard plateaus there.

---

## The five key ideas

1. **Honest cross-validation.** `--group` uses `StratifiedGroupKFold` keyed by a proxy user
   id (`user.created_at`, constant per account). Every user stays inside one fold, so
   user-constant features cannot be memorized as identity. The gap between default row-wise
   CV and `--group` measures the inflation (it is large — see below).
2. **Only honest features.** Per-tweet metadata, behavioral ratios, text-surface stats,
   quoted-user stats, cyclic time, and a *fold-safe* source-app target encoding. Identity-
   proxy columns (profile colors / background image) are hard-banned via an assertion.
3. **The task is user-level.** The label is 100% user-constant, so `--smooth` averages each
   user's tweet probabilities before thresholding — the single biggest post-feature gain.
4. **Diversity blend.** `--blend` trains LightGBM on the same folds; `stack_models.py` adds
   CatBoost / XGBoost / HistGBM / ExtraTrees. Weights and threshold are tuned on aligned OOF.
5. **French text via CamemBERT.** Metadata saturates at ~0.850; the remaining signal is in
   the *text*. Fine-tuned `almanach/camembert-base` (tweet, and bio + tweet) under the same
   honest folds adds the complementary signal that carried the ensemble to **0.856**.

---

## Evidence (the `experiments/` folder)

Each probe is model-agnostic (mostly LightGBM under honest `StratifiedGroupKFold`) and
answers one question. Run from the repo root, e.g. `python experiments/02_train_test_overlap.py`
(they import `config`/`features` from `src/` automatically).

| # | Script | Question | Conclusion |
|---|--------|----------|------------|
| 01 | `01_leakage_diagnosis.py` | Do user-constant "profile" features leak identity under row-wise CV? | Yes — they help `StratifiedKFold` but not `GroupKFold`. Row-wise OOF is inflated by identity memorization. |
| 02 | `02_train_test_overlap.py` | How many test users appear in train? | **0%** (0 of 32,117 test proxy-users). The public LB is 100% disjoint-user → identity features are dead weight. |
| 03 | `03_feature_url_domain.py` | Does a fold-safe TE of the tweet's URL domain add signal? | **−0.07%** — automation shorteners (bit.ly/dlvr.it) are already captured by `source_te`. No lift. |
| 04 | `04_feature_text.py` | Does bio / tweet text (TF-IDF→SVD) add signal? | Bio alone 66.9%, tweet 64.2%; on top of metadata only **+0.12% / +0.05%**. Shallow text is near-useless. |
| 05 | `05_feature_extra.py` | Bio role-words, text-style/emoji/digit ratios, richer quoted features? | All groups **≤ +0.08%** (noise). |
| 06 | `06_feature_user_aggregation.py` | Label-free within-dataset user aggregation? | **+0.04%** (noise). |
| 07 | `07_label_user_consistency.py` | Is the label consistent within a user? | **100% user-constant** (0 mixed-label groups / 30,696 users) → it is a *user*-classification task. |
| 08 | `08_user_smoothing.py` | Does averaging each user's tweet predictions help? | **+0.47%** (84.31% → 84.78%). The biggest post-feature gain. `created_at` is the right key. |
| 09 | `09_user_level_model.py` | Does an explicit user-level aggregated model beat smoothing? | 84.48% — slightly below the smoothed blend; aggregation ≈ smoothing, diversity matters more. |
| 10 | `10_text_camembert_probe.py` | Does a frozen CamemBERT tweet embedding add signal over metadata? | **+0.28%** (84.54% → 84.82%) — first real lift beyond metadata; justified fine-tuning CamemBERT. |
| 11 | `11_hashtag_mention_te.py` | Does TE of *which* hashtags / mentioned accounts a user uses help? | **Negative** (84.54% → 84.37%) — overfits fold-specific tags, doesn't generalize. |
| 12 | `12_memorize_vs_honest.py` | Why can CV look ~100% but public cap at ~0.856? | memorize-train **100%** → row-wise CV 87.6% → honest CV **84.1%**. The 16-pt gap is identity memorization that does not transfer to the all-strangers public test. |

**Why CamemBERT and not TwHIN-BERT?** The tweets are French — CamemBERT is French-specific,
TwHIN-BERT is multilingual (shallower French). In testing, TwHIN-BERT added **+0.00%** to
the ensemble and CamemBERT-large added **+0.04%** (noise), so the production pipeline keeps
only fine-tuned **CamemBERT-base** (tweet, and bio + tweet).

**Cautionary history:** an earlier submission reached 97.48% CV but only 0.839 public — pure
identity memorization under row-wise CV. That lesson is the reason for every honest-CV choice
above (see probes 01 and 12).

---

## Key data facts

- **Direct user IDs are stripped** (`user.id`, `user.id_str`, `user.screen_name` absent; the
  top-level `id_str` is the *tweet* id). So any cross-row "user aggregation" silently
  aggregates across the whole dataset = leakage.
- **Train and test users are 100% disjoint** — the model must generalize to strangers.
- **User profile metadata is the strongest honest signal** (account age, statuses count,
  favourites/tweet ratio, activity rate, listed count), plus the tweet **source app**
  (TweetDeck/Hootsuite/Buffer-style tools skew influencer) via a fold-safe target encoding.
- **Quoted-tweet stats** (`quoted_status.user.followers_count`, …) are present even though the
  main user's follower counts are stripped — an indirect influence signal.
