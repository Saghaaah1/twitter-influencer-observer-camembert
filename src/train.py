"""
Main training pipeline: load -> honest features -> 5-fold CV (fold-safe source TE
+ fold-safe scaling) -> deep MLP -> OOF + averaged test probs -> submission CSV.

Usage:
    python train.py                  # full run, row-wise StratifiedKFold, MLP only
    python train.py --group          # honest CV: StratifiedGroupKFold by proxy user key
    python train.py --group --blend --smooth   # recommended: honest CV + ensemble +
                                               # per-user prediction smoothing
    python train.py --smoke          # tiny smoke test (5k rows, 2 epochs, 2 folds)

Anti-leakage design (mirrors the approved plan):
- Only honest features (see features.py); identity-proxy columns are never read.
- The source target encoding and the StandardScaler are BOTH fit on each fold's
  training rows only, then applied to that fold's val rows and the full test set.
- Expect OOF ~89-91% under default StratifiedKFold. If OOF jumps to ~97%, something
  leaked -> stop and inspect (that is the sub21 cautionary tale).
- --group keeps every user's rows inside ONE fold (disjoint users across folds), so
  user-CONSTANT features cannot be memorized as identity proxies. Its OOF is LOWER but
  HONEST: it tracks the disjoint-user public LB. The gap between default and --group OOF
  quantifies how much row-wise CV is inflated by identity memorization.
"""
import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
from pandas import json_normalize
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import config
import features as F
from model import build_model


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(path, nrows=None):
    df = pd.read_json(path, lines=True, nrows=nrows)
    df = json_normalize(df.to_dict(orient="records"))
    return df


def build_user_group_key(df):
    """Proxy per-user key for GroupKFold — used ONLY to split folds, never as a feature.

    Direct user ids (user.id / id_str / screen_name / name) are stripped from this data,
    but `user.created_at` is EXACTLY constant per account and near-unique across accounts,
    so it alone is a reliable proxy user key (verified: 0 mixed-label groups, and adding
    other user-constant fields does not change the grouping). Grouping by this key keeps
    every user's rows inside a single fold, so user-CONSTANT features (tweets_per_day,
    log_age, desc_len, source_te, ...) can no longer be memorized as identity proxies and
    inflate the OOF. The resulting OOF tracks a disjoint-user public test set (the sub21
    cautionary tale). The key is discarded after splitting; it does not enter X.
    """
    return df.get("user.created_at", pd.Series("", index=df.index)).fillna("").astype(str).values


@torch.no_grad()
def predict_probs(model, X, device, batch_size=8192):
    """Return sigmoid probabilities for a feature matrix (numpy float32)."""
    model.eval()
    probs = np.empty(len(X), dtype=np.float64)
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[start:start + batch_size]).to(device)
        logits = model(xb)
        probs[start:start + batch_size] = torch.sigmoid(logits).cpu().numpy()
    return probs


def train_one_fold(X_tr, y_tr, X_val, y_val, device, epochs, patience, writer=None, fold=1):
    """Train an MLP on one fold; return best model (by val acc) + best val acc + best epoch.

    If `writer` is given, logs per-epoch train loss / val accuracy / learning rate to
    TensorBoard under tags like 'fold1/train_loss'. Logging is observation-only and
    does NOT change any hyperparameter or the training trajectory.
    """
    in_dim = X_tr.shape[1]
    model = build_model(in_dim).to(device)

    ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.float32)))
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.MAX_LR, epochs=epochs,
        steps_per_epoch=len(loader), pct_start=0.3,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = -1.0
    best_epoch = -1
    best_state = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * len(xb)
            n_seen += len(xb)

        train_loss = running_loss / max(n_seen, 1)
        val_probs = predict_probs(model, X_val, device)
        val_acc = accuracy_score(y_val, (val_probs > 0.5).astype(int))
        cur_lr = optimizer.param_groups[0]["lr"]

        if writer is not None:
            writer.add_scalar(f"fold{fold}/train_loss", train_loss, epoch)
            writer.add_scalar(f"fold{fold}/val_acc", val_acc, epoch)
            writer.add_scalar(f"fold{fold}/lr", cur_lr, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_acc, best_epoch


def assemble_features(X_base, parsed_src, df, te_map, global_mean):
    """Combine honest base features + static source/time + fold-safe source TE.

    Leaves QUOTED_IMPUTE_COLS as NaN — they are median-imputed fold-safe in the loop.
    """
    X = F.add_static_source_features(parsed_src, df, X_base)
    X = X.copy()
    X["source_te"] = F.apply_source_te(parsed_src, te_map, global_mean)
    # Clean source_te/time NaN/inf only; do NOT -1-fill the quoted cols (kept NaN).
    non_quoted = [c for c in X.columns if c not in F.QUOTED_IMPUTE_COLS]
    X[non_quoted] = X[non_quoted].replace([np.inf, -np.inf], np.nan).fillna(-1)
    X[F.QUOTED_IMPUTE_COLS] = X[F.QUOTED_IMPUTE_COLS].replace([np.inf, -np.inf], np.nan)
    return X


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny smoke test")
    parser.add_argument("--group", action="store_true",
                        help="honest CV: StratifiedGroupKFold by proxy user key "
                             "(disjoint users across folds, mirrors the public LB). "
                             "Default is row-wise StratifiedKFold (optimistic OOF).")
    parser.add_argument("--blend", action="store_true",
                        help="also train an honest LightGBM per fold and ensemble it with "
                             "the MLP; blend weight + threshold tuned on aligned OOF.")
    parser.add_argument("--smooth", action="store_true",
                        help="per-user prediction smoothing: average each user's tweet "
                             "probabilities (grouped by proxy user key) before thresholding. "
                             "The label is user-constant, so this denoises per-tweet errors.")
    args = parser.parse_args()

    smoke = args.smoke
    n_rows = config.SMOKE_NROWS if smoke else None
    epochs = config.SMOKE_EPOCHS if smoke else config.EPOCHS
    n_folds = config.SMOKE_FOLDS if smoke else config.N_FOLDS
    patience = epochs if smoke else config.PATIENCE  # don't early-stop during smoke

    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({gpu_name}) | smoke={smoke}")

    # TensorBoard writer. One subfolder per launch; smoke runs go to runs/smoke.
    from datetime import datetime
    run_name = "smoke" if smoke else datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config.LOG_DIR, run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs -> {log_dir}")

    # === Load ===
    print("Loading data...")
    train = load_data(config.TRAIN_PATH, nrows=n_rows)
    test = load_data(config.TEST_PATH, nrows=n_rows)
    y = train["label"].values.astype(np.int64)
    print(f"Train: {train.shape} | Test: {test.shape} | Labels: {np.bincount(y)} "
          f"(pos rate {y.mean():.4f})")

    # === Honest base features (built once) ===
    print("Building honest base features...")
    X_tr_base = F.build_features(train)
    X_te_base = F.build_features(test)

    # === Parse source once per dataset (the slow part) ===
    print("Parsing source apps...")
    parsed_src_tr = F.parse_source_series(train)
    parsed_src_te = F.parse_source_series(test)

    # Variance guard (authoritative dead-col prune): drop any still-constant column.
    # Decided on TRAIN nunique (target-independent), same list applied to test (alignment).
    # Excludes the quoted cols (deliberately NaN here). build_features already drops the
    # known dead set; this self-heals if a future data refresh makes another col constant.
    guard_cols = [c for c in X_tr_base.columns if c not in F.QUOTED_IMPUTE_COLS]
    const_cols = [c for c in guard_cols if X_tr_base[c].nunique(dropna=False) <= 1]
    if const_cols:
        print("Auto-pruning still-constant cols:", const_cols)
        X_tr_base = X_tr_base.drop(columns=const_cols)
        X_te_base = X_te_base.drop(columns=const_cols)

    # Leakage guard: none of the banned identity-proxy columns may appear.
    leaked = [c for c in config.BANNED_COLUMNS if c in X_tr_base.columns]
    assert not leaked, f"Identity-proxy columns leaked into features: {leaked}"

    # === 5-fold CV ===
    # --group → StratifiedGroupKFold (users disjoint across folds → honest, public-aligned
    # OOF). Default → row-wise StratifiedKFold (same user spans folds → optimistic OOF).
    if args.group:
        groups = build_user_group_key(train)
        n_users = len(np.unique(groups))
        print(f"CV: StratifiedGroupKFold by proxy user key "
              f"({n_users} users over {len(y)} rows, {n_users/len(y)*100:.1f}% unique)")
        skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=config.SEED)
    else:
        groups = None
        print("CV: StratifiedKFold (row-wise; OOF may be optimistic - use --group for honest CV)")
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.SEED)
    oof_probs = np.zeros(len(y), dtype=np.float64)
    test_probs = np.zeros(len(test), dtype=np.float64)
    # Companion GBDT accumulators (filled only when --blend).
    lgb_oof = np.zeros(len(y), dtype=np.float64)
    lgb_test = np.zeros(len(test), dtype=np.float64)
    if args.blend:
        import lightgbm as lgb
    fold_accs = []
    feat_names = None

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr_base, y, groups), start=1):
        # Fold-safe source target encoding (training rows of THIS fold only).
        te_map, gmean = F.compute_source_te(parsed_src_tr.iloc[tr_idx].values, y[tr_idx])

        X_tr_full = assemble_features(X_tr_base, parsed_src_tr, train, te_map, gmean)
        X_te_full = assemble_features(X_te_base, parsed_src_te, test, te_map, gmean)

        # Fold-safe present-rows median imputation of the quoted cols.
        # static_no_y: reads NO label; fit on THIS fold's train rows only (over present
        # rows), applied to train/val/test. Median (heavy-tailed). Runs BEFORE the scaler
        # so scaler stats see imputed, sentinel-free data. assemble_features returns fresh
        # frames each fold, so the in-place fill does not leak across folds.
        impute_cols = [c for c in F.QUOTED_IMPUTE_COLS if c in X_tr_full.columns]
        tr_block = X_tr_full.iloc[tr_idx]
        medians = {c: tr_block[c].median(skipna=True) for c in impute_cols}
        medians = {c: (v if np.isfinite(v) else -1.0) for c, v in medians.items()}
        for c, v in medians.items():
            X_tr_full[c] = X_tr_full[c].fillna(v)
            X_te_full[c] = X_te_full[c].fillna(v)

        if feat_names is None:
            feat_names = list(X_tr_full.columns)

        # Raw (unscaled) fold arrays — used as-is by the scale-invariant GBDT.
        X_tr_raw = X_tr_full.iloc[tr_idx].values.astype(np.float32)
        X_val_raw = X_tr_full.iloc[val_idx].values.astype(np.float32)
        X_test_raw = X_te_full.values.astype(np.float32)

        # Fold-safe standardization (fit on this fold's training rows only) for the MLP.
        scaler = StandardScaler()
        X_tr_fold = scaler.fit_transform(X_tr_raw).astype(np.float32)
        X_val_fold = scaler.transform(X_val_raw).astype(np.float32)
        X_test_scaled = scaler.transform(X_test_raw).astype(np.float32)

        model, best_val_acc, best_epoch = train_one_fold(
            X_tr_fold, y[tr_idx], X_val_fold, y[val_idx], device, epochs, patience,
            writer=writer, fold=fold)

        oof_probs[val_idx] = predict_probs(model, X_val_fold, device)
        test_probs += predict_probs(model, X_test_scaled, device) / n_folds

        # Companion honest GBDT on the SAME fold (unscaled features, fold-safe source_te).
        if args.blend:
            lgbm = lgb.LGBMClassifier(**config.LGB_PARAMS)
            lgbm.fit(X_tr_raw, y[tr_idx])
            lgb_oof[val_idx] = lgbm.predict_proba(X_val_raw)[:, 1]
            lgb_test += lgbm.predict_proba(X_test_raw)[:, 1] / n_folds

        fold_accs.append(best_val_acc)
        writer.add_scalar("summary/fold_best_val_acc", best_val_acc, fold)
        print(f"  Fold {fold}/{n_folds}: best val acc {best_val_acc*100:.2f}% @ epoch {best_epoch}")

    print(f"\nFinal feature count: {len(feat_names)}")
    print("Features:", feat_names)

    # === OOF accuracy + threshold tuning (tuned only on OOF -> not leakage) ===
    def best_threshold(probs):
        """Return (best_thr, best_acc) over a 0.40..0.60 grid on the OOF labels."""
        bt, ba = 0.5, accuracy_score(y, (probs > 0.5).astype(int))
        for thr in np.arange(0.40, 0.601, 0.01):
            acc = accuracy_score(y, (probs > thr).astype(int))
            if acc > ba:
                ba, bt = acc, float(thr)
        return bt, ba

    oof_acc_05 = accuracy_score(y, (oof_probs > 0.5).astype(int))
    best_thr, best_thr_acc = best_threshold(oof_probs)
    print(f"\nOOF accuracy @0.50: {oof_acc_05*100:.2f}%")
    print(f"Best OOF threshold: {best_thr:.2f} -> {best_thr_acc*100:.2f}%")

    # === Blend with the companion GBDT (weight + threshold tuned jointly on OOF) ===
    # Default submission source = MLP only; overridden below when --blend wins on OOF.
    # sub_oof is the OOF counterpart of sub_probs (same model mix) -> used for honest
    # threshold tuning and to validate the per-user smoothing lift.
    sub_probs, sub_oof, sub_thr = test_probs, oof_probs, best_thr
    if args.blend:
        lgb_thr, lgb_acc = best_threshold(lgb_oof)
        print(f"LGBM-only OOF: best_thr {lgb_thr:.2f} -> {lgb_acc*100:.2f}%")
        best_alpha, best_blend_thr, best_blend_acc = 1.0, best_thr, best_thr_acc
        for alpha in np.linspace(0.0, 1.0, 21):          # alpha*MLP + (1-alpha)*LGBM
            blend_oof = alpha * oof_probs + (1.0 - alpha) * lgb_oof
            thr, acc = best_threshold(blend_oof)
            if acc > best_blend_acc:
                best_blend_acc, best_alpha, best_blend_thr = acc, float(alpha), thr
        print(f"Best blend: alpha(MLP)={best_alpha:.2f} thr={best_blend_thr:.2f} "
              f"-> {best_blend_acc*100:.2f}%  (MLP {best_thr_acc*100:.2f}% | "
              f"LGBM {lgb_acc*100:.2f}%)")
        writer.add_scalar("summary/blend_alpha", best_alpha, 0)
        writer.add_scalar("summary/blend_oof_acc", best_blend_acc, 0)
        sub_probs = best_alpha * test_probs + (1.0 - best_alpha) * lgb_test
        sub_oof = best_alpha * oof_probs + (1.0 - best_alpha) * lgb_oof
        sub_thr = best_blend_thr

    # === Per-user prediction smoothing (label is user-constant -> denoise per-tweet) ===
    if args.smooth:
        def smooth_by_user(probs, keys):
            return pd.Series(probs).groupby(np.asarray(keys)).transform("mean").values
        tr_keys = build_user_group_key(train)
        te_keys = build_user_group_key(test)
        sm_oof = smooth_by_user(sub_oof, tr_keys)
        sub_probs = smooth_by_user(sub_probs, te_keys)
        sm_thr, sm_acc = best_threshold(sm_oof)
        pre_acc = accuracy_score(y, (sub_oof > sub_thr).astype(int))
        print(f"Per-user smoothing: OOF {pre_acc*100:.2f}% -> {sm_acc*100:.2f}% "
              f"(+{(sm_acc-pre_acc)*100:.2f}), thr {sm_thr:.2f}")
        writer.add_scalar("summary/smooth_oof_acc", sm_acc, 0)
        sub_thr = sm_thr

    writer.add_scalar("summary/oof_acc_0.50", oof_acc_05, 0)
    writer.add_scalar("summary/oof_acc_best_thr", best_thr_acc, 0)
    writer.add_text("summary/result",
                    f"OOF@0.50={oof_acc_05*100:.2f}% | best_thr={best_thr:.2f} "
                    f"-> {best_thr_acc*100:.2f}% | features={len(feat_names)}", 0)

    if not smoke and oof_acc_05 > 0.95:
        print("WARNING: OOF > 95% — possible leakage. Inspect features before trusting this.")

    # === Submission (blended probs + threshold when --blend, else MLP-only) ===
    final_preds = (sub_probs > sub_thr).astype(int)
    submission = pd.DataFrame({"ID": test["challenge_id"], "Prediction": final_preds})

    # Submission sanity checks.
    assert list(submission.columns) == ["ID", "Prediction"]
    assert set(np.unique(final_preds)).issubset({0, 1})
    assert submission["ID"].notna().all() and not submission["ID"].duplicated().any()
    if not smoke:
        assert len(submission) == 103380, f"Expected 103380 rows, got {len(submission)}"

    # Smoke runs write to *_smoke paths so a sanity check can NEVER overwrite the real
    # submission / probability arrays produced by a full run.
    def out_path(path):
        if not smoke:
            return path
        base, ext = os.path.splitext(path)
        return base + "_smoke" + ext

    sub_path = out_path(config.SUBMISSION_PATH)
    submission.to_csv(sub_path, index=False)
    np.save(out_path(config.OOF_PATH), oof_probs)
    np.save(out_path(config.TEST_PROBS_PATH), test_probs)
    if args.blend:
        _here = os.path.dirname(config.OOF_PATH)
        np.save(out_path(os.path.join(_here, "lgb_oof_probs.npy")), lgb_oof)
        np.save(out_path(os.path.join(_here, "lgb_test_probs.npy")), lgb_test)

    print(f"\nSaved submission: {sub_path} "
          f"({len(submission)} rows, pred mean {final_preds.mean():.4f})")
    print(submission.head().to_string(index=False))

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
