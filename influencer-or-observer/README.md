# Influencer or Observer — honest social-role prediction

A self-contained PyTorch + LightGBM pipeline for the Kaggle task *"Influencer or Observer:
Predicting Social Roles"* — a binary classification of Twitter accounts from a single
tweet's metadata (French-language corpus). The entire design is organized around **one
principle: do not fool yourself with identity leakage.**

> Cautionary history: earlier submissions scored 94–97% in cross-validation but only 0.839
> on the public leaderboard, because some features act as per-user *identity proxies* under
> row-wise CV. The public test set is **100% disjoint-user** (see
> [`experiments/02`](experiments/02_train_test_overlap.py)), so any signal that does not
> generalize across *different* users is worthless. Every choice here keeps the OOF estimate
> honest and aligned with the public LB.

## Result

| Stage | honest OOF (GroupKFold) | public LB |
|-------|-------------------------|-----------|
| MLP only | 83.94% | — |
| + LightGBM blend (α=0.55) | 84.31% | — |
| + per-user smoothing | 84.78% | 0.848 |
| + CatBoost / HistGBM / ExtraTrees stack | 84.84% | 0.850 |
| + fine-tuned CamemBERT (tweet) | 85.14% | 0.852 |
| **+ CamemBERT (bio + tweet)** | **85.40%** | **0.856 — tied #1** |

The honest information ceiling of this data is ~0.856 — the whole leaderboard plateaus there
(see [`experiments/12`](experiments/12_memorize_vs_honest.py)). A larger model
(CamemBERT-large) and a different architecture (TwHIN-BERT) were both tried and added nothing.

## Layout

```
.
├── config.py                 # single source of truth: paths, seed, hyperparams, banned columns
├── features.py               # "honest" feature engineering + fold-safe source target encoding
├── model.py                  # TabularMLP (256→128→64→1, BatchNorm+GELU+Dropout)
├── train.py                  # tabular pipeline: load → CV → MLP+LGBM blend → smooth → submission
├── stack_models.py           # adds CatBoost / XGBoost / HistGBM / ExtraTrees to the ensemble
├── finetune_camembert.py     # fine-tune CamemBERT on tweet text   (text model #1)
├── finetune_camembert_bio.py # fine-tune CamemBERT on bio + tweet  (text model #2)
├── make_submission.py        # blend ALL model predictions + per-user smooth → submission_final.csv
├── experiments/              # the evidence behind every design decision (see its README)
└── PIPELINE.zh.md            # detailed pipeline walkthrough (Chinese)
```

## Setup

```bash
conda activate sara        # or: pip install pandas scikit-learn lightgbm torch transformers
# place train.jsonl and kaggle_test.jsonl in the PARENT directory of this repo
```

## Run (reproduce the 0.856 submission)

```bash
python train.py --group --blend --smooth   # MLP + LightGBM, honest CV, per-user smoothing
python stack_models.py                      # add CatBoost / XGBoost / HistGBM / ExtraTrees
python finetune_camembert.py                # fine-tune CamemBERT on tweet text   (GPU, ~1 hr)
python finetune_camembert_bio.py            # fine-tune CamemBERT on bio + tweet   (GPU, ~1 hr)
python make_submission.py                   # blend everything + smooth -> submission_final.csv
```

Each base model writes aligned `<name>_oof_probs.npy` / `<name>_test_probs.npy`;
`make_submission.py` auto-discovers them, picks weights by greedy ensemble selection on the
smoothed OOF, and writes the final CSV. Quick sanity check: `python train.py --smoke`.

## Key ideas

1. **Honest cross-validation.** `--group` uses `StratifiedGroupKFold` keyed by a proxy user
   id (`user.created_at`, which is constant per account). This keeps every user inside one
   fold so user-constant features cannot be memorized as identity. The gap between default
   and `--group` OOF measures the row-wise inflation.
2. **Only honest features.** Per-tweet metadata, behavioral ratios, text surface stats,
   quoted-user stats, cyclic time, and a *fold-safe* source-app target encoding. Identity-
   proxy columns (profile colors / background image) are hard-banned via an assertion.
3. **The task is user-level.** The label is 100% user-constant
   ([`experiments/07`](experiments/07_label_user_consistency.py)), so `--smooth` averages
   each user's tweet probabilities before thresholding — the largest post-feature gain.
4. **Diversity blend.** `--blend` trains a LightGBM on the same folds and ensembles it with
   the MLP; weight and threshold are tuned on aligned OOF only.
5. **French text via CamemBERT.** Metadata saturates at ~0.850; the remaining signal is in
   the *text*. Fine-tuned `almanach/camembert-base` (tweet, and bio + tweet) under the same
   honest folds adds the complementary signal that carried the ensemble to **0.856** — the
   one lever that beat the metadata ceiling. (CamemBERT-large / TwHIN-BERT added nothing;
   those attempts are archived.)

See [`experiments/README.md`](experiments/README.md) for the full evidence trail.
