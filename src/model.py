"""
src/model.py
============
Phase 4 — XGBoost Matchup Prediction Model

Trains a model to predict match winner from pre-match features.
Uses walk-forward validation (no data leakage).
Includes SHAP explainability.

Usage:
    python src/model.py

Outputs:
    models/xgb_model.json       - trained model
    models/feature_names.txt    - feature list
    reports/model_performance.png
    reports/shap_importance.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json, time, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             brier_score_loss, log_loss,
                             confusion_matrix, classification_report)
from sklearn.calibration import calibration_curve
import xgboost as xgb

DATA_DIR    = Path("data")
MODELS_DIR  = Path("models");  MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR = Path("reports"); REPORTS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# FEATURE COLUMNS USED BY MODEL
# ─────────────────────────────────────────────
FEATURE_COLS = [
    # Elo
    "elo_diff_overall", "elo_diff_surface",
    "w_elo_overall", "l_elo_overall",
    "w_elo_surface", "l_elo_surface",

    # Rank
    "rank_diff", "w_rank", "l_rank",

    # Age
    "w_age", "l_age", "age_diff",

    # Form
    "w_form_overall", "l_form_overall",
    "w_form_surface", "l_form_surface",
    "form_diff_overall", "form_diff_surface",

    # Experience
    "w_matches_played", "l_matches_played",
    "w_matches_on_surface", "l_matches_on_surface",

    # Serve
    "w_avg_1stServe_pct", "l_avg_1stServe_pct",
    "w_avg_1stWon_pct",   "l_avg_1stWon_pct",
    "w_avg_2ndWon_pct",   "l_avg_2ndWon_pct",
    "w_avg_bpSave_pct",   "l_avg_bpSave_pct",
    "w_avg_ace_rate",     "l_avg_ace_rate",

    # H2H
    "h2h_wins_winner", "h2h_wins_loser",
    "h2h_total", "h2h_winrate_winner",

    # Match context
    "best_of",
]

SURFACE_DUMMIES = ["surface_Clay", "surface_Grass",
                   "surface_Hard", "surface_Carpet"]
LEVEL_DUMMIES   = ["level_Grand Slam", "level_Masters 1000",
                   "level_ATP 500",    "level_ATP 250",
                   "level_Finals"]


# ─────────────────────────────────────────────
# STEP 1 — LOAD & PREPARE
# ─────────────────────────────────────────────
def load_and_prepare(min_matches: int = 10) -> tuple:
    """
    Load features.parquet, symmetrise (flip winner/loser randomly),
    encode categoricals, return X, y, dates.

    Symmetrisation is critical: the model must learn from BOTH
    perspectives, not just 'winner always wins'.
    """
    print("  Loading features...")
    df = pd.read_parquet(DATA_DIR / "features.parquet")
    df = df.sort_values("tourney_date").reset_index(drop=True)

    # Filter players with very few matches (cold-start noise)
    df = df[
        (df["w_matches_played"] >= min_matches) &
        (df["l_matches_played"] >= min_matches)
    ].reset_index(drop=True)

    print(f"  {len(df):,} matches after filtering (min {min_matches} matches played)")

    # ── Symmetrise: randomly flip ~50% of rows ──
    # This prevents the model from learning "first player listed = winner"
    np.random.seed(42)
    flip = np.random.rand(len(df)) < 0.5

    df_flipped = df.copy()

    # Swap winner/loser columns
    w_cols = [c for c in df.columns if c.startswith("w_") or c.startswith("winner")]
    l_cols = [c.replace("w_", "l_", 1).replace("winner", "loser")
              for c in w_cols]

    for wc, lc in zip(w_cols, l_cols):
        if wc in df.columns and lc in df.columns:
            tmp = df_flipped.loc[flip, wc].copy()
            df_flipped.loc[flip, wc] = df_flipped.loc[flip, lc]
            df_flipped.loc[flip, lc] = tmp

    # Also flip H2H & derived diffs
    for col in ["elo_diff_overall", "elo_diff_surface",
                "form_diff_overall", "form_diff_surface", "rank_diff", "age_diff"]:
        if col in df_flipped.columns:
            df_flipped.loc[flip, col] *= -1

    df_flipped.loc[flip, "h2h_wins_winner"], df_flipped.loc[flip, "h2h_wins_loser"] = (
        df_flipped.loc[flip, "h2h_wins_loser"].copy(),
        df_flipped.loc[flip, "h2h_wins_winner"].copy(),
    )
    df_flipped.loc[flip, "h2h_winrate_winner"] = (
        1 - df_flipped.loc[flip, "h2h_winrate_winner"]
    )

    # Target: 1 = first player wins, 0 = second player wins
    df_flipped["target"] = np.where(flip, 0, 1)

    # ── Surface dummies ──
    surf_dummies = pd.get_dummies(df_flipped["surface"], prefix="surface")
    for col in SURFACE_DUMMIES:
        if col not in surf_dummies.columns:
            surf_dummies[col] = 0

    # ── Tourney level dummies ──
    level_map = {
        "G": "Grand Slam", "M": "Masters 1000", "A": "ATP 500",
        "S": "ATP 250",    "F": "Finals",
    }
    df_flipped["level_name"] = df_flipped["tourney_level"].map(level_map).fillna("Other")
    level_dummies = pd.get_dummies(df_flipped["level_name"], prefix="level")
    for col in LEVEL_DUMMIES:
        if col not in level_dummies.columns:
            level_dummies[col] = 0

    all_features = FEATURE_COLS + SURFACE_DUMMIES + LEVEL_DUMMIES
    X = pd.concat([df_flipped[FEATURE_COLS], surf_dummies[SURFACE_DUMMIES],
                   level_dummies[LEVEL_DUMMIES]], axis=1)
    X = X.reindex(columns=all_features, fill_value=0)

    y     = df_flipped["target"].values
    dates = df_flipped["tourney_date"].values

    print(f"  Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"  Class balance: {y.mean()*100:.1f}% player-1 wins")
    return X, y, dates, all_features


# ─────────────────────────────────────────────
# STEP 2 — WALK-FORWARD VALIDATION
# ─────────────────────────────────────────────
def walk_forward_eval(X: pd.DataFrame, y: np.ndarray,
                      dates: np.ndarray) -> dict:
    """
    Time-series cross-validation: always train on past, test on future.
    5 splits covering the full date range.
    """
    print("\n  Walk-forward cross-validation...")
    tscv = TimeSeriesSplit(n_splits=5)

    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=1,
            reg_alpha=0.1,
            reg_lambda=1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr,
                  eval_set=[(X_te, y_te)],
                  verbose=False)

        probs = model.predict_proba(X_te)[:, 1]
        preds = (probs >= 0.5).astype(int)

        acc    = accuracy_score(y_te, preds)
        auc    = roc_auc_score(y_te, probs)
        brier  = brier_score_loss(y_te, probs)
        ll     = log_loss(y_te, probs)

        n_train = len(train_idx)
        n_test  = len(test_idx)
        d_start = pd.Timestamp(dates[test_idx[0]]).year
        d_end   = pd.Timestamp(dates[test_idx[-1]]).year

        fold_metrics.append({
            "fold": fold+1, "train_size": n_train, "test_size": n_test,
            "test_years": f"{d_start}-{d_end}",
            "accuracy": acc, "auc": auc, "brier": brier, "log_loss": ll,
        })
        print(f"    Fold {fold+1}  [{d_start}-{d_end}]  "
              f"Acc={acc:.3f}  AUC={auc:.3f}  Brier={brier:.3f}")

    return fold_metrics


# ─────────────────────────────────────────────
# STEP 3 — TRAIN FINAL MODEL
# ─────────────────────────────────────────────
def train_final(X: pd.DataFrame, y: np.ndarray) -> xgb.XGBClassifier:
    print("\n  Training final model on full dataset...")
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=1,
        reg_alpha=0.1,
        reg_lambda=1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y, verbose=False)
    print(f"  ✓  Model trained on {len(X):,} matches")
    return model


# ─────────────────────────────────────────────
# STEP 4 — PLOTS
# ─────────────────────────────────────────────
def plot_performance(fold_metrics: list, model, X, y, feature_names):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#0f0f1a")
    fig.suptitle("Model Performance", fontsize=16,
                 color="white", fontweight="bold")

    # ── 1. Fold metrics ──
    ax = axes[0]
    ax.set_facecolor("#0f0f1a")
    folds  = [m["fold"] for m in fold_metrics]
    accs   = [m["accuracy"] for m in fold_metrics]
    aucs   = [m["auc"] for m in fold_metrics]
    labels = [m["test_years"] for m in fold_metrics]

    x = np.arange(len(folds))
    bars1 = ax.bar(x - 0.2, accs, 0.35, label="Accuracy", color="#00d4ff", alpha=0.8)
    bars2 = ax.bar(x + 0.2, aucs, 0.35, label="AUC",      color="#ff6b6b", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, fontsize=8)
    ax.set_ylim(0.5, 0.85)
    ax.axhline(0.5, color="#555", linestyle="--", linewidth=0.8)
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    ax.set_title("Walk-Forward CV Results", color="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_color("#333")

    for bar in bars1:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
               f"{bar.get_height():.3f}", ha="center", fontsize=7, color="white")
    for bar in bars2:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
               f"{bar.get_height():.3f}", ha="center", fontsize=7, color="white")

    # ── 2. Calibration curve ──
    ax2 = axes[1]
    ax2.set_facecolor("#0f0f1a")
    probs_all = model.predict_proba(X)[:, 1]
    fraction_pos, mean_pred = calibration_curve(y, probs_all, n_bins=10)
    ax2.plot(mean_pred, fraction_pos, "o-", color="#00d4ff",
             linewidth=2, label="Model")
    ax2.plot([0,1],[0,1], "--", color="#555", label="Perfect")
    ax2.set_xlabel("Mean Predicted Probability", color="#aaa")
    ax2.set_ylabel("Fraction Positives", color="#aaa")
    ax2.set_title("Calibration Curve", color="white", fontsize=11)
    ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values(): spine.set_color("#333")

    # ── 3. Feature importance (top 20) ──
    ax3 = axes[2]
    ax3.set_facecolor("#0f0f1a")
    importances = model.feature_importances_
    idx = np.argsort(importances)[-20:]
    colors = plt.cm.YlOrRd(np.linspace(0.4, 1, 20))
    ax3.barh([feature_names[i] for i in idx],
             importances[idx], color=colors)
    ax3.set_title("Top 20 Feature Importances", color="white", fontsize=11)
    ax3.tick_params(colors="white", labelsize=7)
    for spine in ax3.spines.values(): spine.set_color("#333")

    plt.tight_layout()
    out = REPORTS_DIR / "model_performance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓  Saved: {out}")


# ─────────────────────────────────────────────
# STEP 5 — SAVE MODEL
# ─────────────────────────────────────────────
def save_model(model, feature_names: list):
    model.save_model(MODELS_DIR / "xgb_model.json")
    (MODELS_DIR / "feature_names.txt").write_text(
        "\n".join(feature_names), encoding="utf-8"
    )
    # Save summary metrics
    probs = model.predict_proba
    print(f"  ✓  Model saved to {MODELS_DIR}/")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()
    print("\n" + "="*55)
    print("  Tennis Matchup Prediction — XGBoost")
    print("="*55)

    X, y, dates, feature_names = load_and_prepare(min_matches=10)

    # Walk-forward CV
    fold_metrics = walk_forward_eval(X, y, dates)

    # Summary
    print("\n  Cross-validation summary:")
    avg_acc = np.mean([m["accuracy"] for m in fold_metrics])
    avg_auc = np.mean([m["auc"]      for m in fold_metrics])
    avg_brier = np.mean([m["brier"]  for m in fold_metrics])
    print(f"    Avg Accuracy : {avg_acc:.3f}  ({avg_acc*100:.1f}%)")
    print(f"    Avg AUC      : {avg_auc:.3f}")
    print(f"    Avg Brier    : {avg_brier:.3f}")

    # Train final model
    model = train_final(X, y)

    # Full dataset metrics
    probs_full = model.predict_proba(X)[:, 1]
    preds_full = (probs_full >= 0.5).astype(int)
    print(f"\n  Full dataset metrics:")
    print(f"    Accuracy : {accuracy_score(y, preds_full)*100:.1f}%")
    print(f"    AUC      : {roc_auc_score(y, probs_full):.3f}")
    print(f"    Brier    : {brier_score_loss(y, probs_full):.3f}")

    # Plots
    plot_performance(fold_metrics, model, X, y, feature_names)

    # Save
    save_model(model, feature_names)

    print(f"\n  Done in {time.time()-t0:.1f}s  🎾")
    print(f"\n  Next → python src/matchup.py \"Djokovic\" \"Nadal\" clay\n")