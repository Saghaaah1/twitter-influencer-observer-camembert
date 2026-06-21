# Influencer or Observer — Predicting Social Roles

Binary classification of French Twitter accounts (**influencer** vs. **observer**) from a
single tweet's metadata and text. **Final result: public leaderboard 0.857.**

The whole approach is built around one principle: **don't fool yourself with identity
leakage.** Direct user IDs are stripped from the data and the public test set is made of
users never seen in training, so only signal that generalizes across *different* users
counts. Every design choice keeps the cross-validation estimate honest and aligned with the
public leaderboard.

---

## Repository layout

```
.
├── main.py                       # one-command pipeline runner (python main.py [--no-text|--smoke])
├── src/                          # production pipeline (run these)
│   ├── config.py                 #   single source of truth: paths, seed, hyperparameters
│   ├── features.py               #   honest feature engineering + fold-safe source target encoding
│   ├── model.py                  #   TabularMLP (256 -> 128 -> 64 -> 1, BatchNorm + GELU + Dropout)
│   ├── train.py                  #   tabular pipeline: CV -> MLP + LightGBM blend -> smooth -> submission
│   ├── stack_models.py           #   adds CatBoost / XGBoost / HistGBM / ExtraTrees to the ensemble
│   ├── finetune_camembert.py     #   fine-tune CamemBERT on tweet text
│   ├── finetune_camembert_bio.py #   fine-tune CamemBERT on bio + tweet
│   ├── finetune_camembert_user.py#   fine-tune CamemBERT at the USER level (bio + ALL the user's tweets)
│   └── make_submission.py        #   blend all model predictions + per-user smoothing -> final CSV
├── experiments/                  # 12 numbered probes — the evidence behind every decision
├── exploration/                  # initial data-understanding scripts + eda.py + feature_lineage/
├── data/                         # train.jsonl / kaggle_test.jsonl (place here; gitignored)
└── output/                       # submissions + model_predictions/ (prob arrays); gitignored
```

---

## Setup

```bash
conda activate sara     # or: pip install -r requirements.txt
# place train.jsonl and kaggle_test.jsonl in data/
```

## Reproduce the 0.857 submission

One command runs the whole pipeline in order:

```bash
python main.py                # full pipeline (GPU needed for the 3 CamemBERT stages)
python main.py --no-text      # tabular models only (skips the ~3 h of GPU fine-tunes)
python main.py --smoke        # ~10 s sanity check that the pipeline runs
python main.py --list         # show the stages
```

Or run each stage by hand (the order `main.py` uses):

```bash
cd src
python train.py --group --blend --smooth   # MLP + LightGBM, honest CV, per-user smoothing
python stack_models.py                      # add CatBoost / XGBoost / HistGBM / ExtraTrees
python finetune_camembert.py                # fine-tune CamemBERT on tweet text    (GPU, ~1 hr)
python finetune_camembert_bio.py            # fine-tune CamemBERT on bio + tweet    (GPU, ~1 hr)
python finetune_camembert_user.py           # user-level CamemBERT (bio + all tweets) (GPU, ~1 hr)
python make_submission.py                   # blend everything -> output/submission_final.csv
```

Each model writes aligned `<name>_oof_probs.npy` / `<name>_test_probs.npy` into
`output/model_predictions/`. `make_submission.py` auto-discovers them, selects ensemble
weights by greedy search on the per-user-smoothed out-of-fold (OOF) predictions, and writes
the final submission. Quick sanity check: `python main.py --smoke`.

---

## How it got to 0.857

| Stage | honest OOF (GroupKFold) | public LB |
|-------|-------------------------|-----------|
| MLP only | 83.94% | — |
| + LightGBM blend (α = 0.55) | 84.31% | — |
| + per-user smoothing (label is user-constant) | 84.78% | 0.848 |
| + CatBoost / HistGBM / ExtraTrees stack | 84.84% | 0.850 |
| + fine-tuned CamemBERT (tweet) | 85.14% | 0.852 |
| + CamemBERT (bio + tweet) | 85.40% | 0.856 |
| **+ user-level CamemBERT (bio + all tweets)** | **85.63%** | **0.857** |

OOF tracks the public leaderboard closely throughout — the payoff of honest validation.
Metadata signal saturates around 0.850; every gain past it comes from **richer text
modelling**, culminating in the user-level CamemBERT. (Top of the leaderboard at submission
time was 0.865; 0.857 placed **4th**.)

---

## The six key ideas

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
6. **User-level text — the final lift to 0.857.** Since the label is user-constant and each
   account has ~5 tweets, the natural unit is the *user*, not the tweet. `finetune_camembert_user.py`
   builds **one document per account** (bio + all of its tweets, max 256 tokens), fine-tunes
   CamemBERT to predict the *account's* label, and broadcasts that prediction to all the
   account's tweet rows. Reading ~5× the text per decision makes it the strongest single
   text member (**76.9%** smoothed, vs ~74.6% for the per-tweet text models) and lifts the
   ensemble to **85.63% OOF / 0.857 public**. It applies the smoothing logic *inside* the
   model instead of after it.

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
| 12 | `12_memorize_vs_honest.py` | Why can CV look ~100% but public stay ~0.85? | memorize-train **100%** → row-wise CV 87.6% → honest CV **84.1%**. The 16-pt gap is identity memorization that does not transfer to the all-strangers public test. |

**Why CamemBERT and not TwHIN-BERT?** The tweets are French — CamemBERT is French-specific,
TwHIN-BERT is multilingual (shallower French). In testing, TwHIN-BERT added **+0.00%** to
the ensemble and CamemBERT-large added **+0.04%** (noise), so the production pipeline keeps
only fine-tuned **CamemBERT-base** (tweet, bio + tweet, and user-level).

**The post-experiment win.** Probe 09 found an *explicit user-level tabular model* roughly
ties smoothing. The breakthrough was doing the user-level aggregation on **text** instead:
`finetune_camembert_user.py` (key idea 6) gives CamemBERT one document per account, which
added the +0.0023 OOF that moved the public score 0.856 → **0.857**. Things that did *not*
help afterward, tested honestly: pseudo-labeling confident test users into the tabular
models (no lift — metadata is saturated) and a second/larger backbone (noise).

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
