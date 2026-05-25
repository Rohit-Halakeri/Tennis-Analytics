"""
src/matchup.py
==============
Phase 4b — Head-to-Head Matchup Engine

Given two player names and a surface, predicts:
  - Win probability for each player
  - Key advantages / disadvantages
  - Historical H2H record
  - Surface-specific breakdown

Usage:
    python src/matchup.py "Novak Djokovic" "Rafael Nadal" Clay
    python src/matchup.py "Carlos Alcaraz" "Jannik Sinner" Hard

Outputs:
    reports/<player1>_vs_<player2>_matchup.png
    reports/<player1>_vs_<player2>_matchup.txt
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

DATA_DIR    = Path("data")
MODELS_DIR  = Path("models")
REPORTS_DIR = Path("reports"); REPORTS_DIR.mkdir(exist_ok=True)

SURFACE_COLORS = {
    "Hard": "#4A90D9", "Clay": "#C0622D",
    "Grass": "#2E8B57", "Carpet": "#8B6914",
}


# ─────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────
def load_artifacts():
    model = xgb.XGBClassifier()
    model.load_model(MODELS_DIR / "xgb_model.json")
    feature_names = (MODELS_DIR / "feature_names.txt").read_text(
        encoding="utf-8").strip().split("\n")
    matches  = pd.read_parquet(DATA_DIR / "matches_with_elo.parquet")
    elo_df   = pd.read_parquet(DATA_DIR / "elo_ratings.parquet")
    features = pd.read_parquet(DATA_DIR / "features.parquet")
    return model, feature_names, matches, elo_df, features


# ─────────────────────────────────────────────
# FIND PLAYER
# ─────────────────────────────────────────────
def find_player(matches, elo_df, name: str):
    mask = elo_df["player_name"].str.lower().str.contains(name.lower(), na=False)
    candidates = elo_df[mask]
    if candidates.empty:
        print(f"  ✗  '{name}' not found. Try a different spelling.")
        return None
    if len(candidates) > 1:
        # Pick highest overall Elo
        candidates = candidates.sort_values("elo_overall", ascending=False)
        print(f"  Multiple found for '{name}', using: {candidates.iloc[0]['player_name']}")
    return candidates.iloc[0]


# ─────────────────────────────────────────────
# BUILD FEATURE VECTOR
# ─────────────────────────────────────────────
def build_feature_vector(p1_row, p2_row, surface: str,
                         features: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Build a single-row feature vector for p1 vs p2 on surface.
    Uses the most recent feature snapshot for each player.
    """
    surf = surface.capitalize()
    surf_col = f"elo_{surf.lower()}"

    def last_features(pid, as_winner: bool):
        """Get the most recent pre-match feature snapshot for a player."""
        col_id = "winner_id" if as_winner else "loser_id"
        pf = features[features[col_id] == pid].sort_values("tourney_date")
        if pf.empty:
            return {}
        row = pf.iloc[-1]
        prefix = "w_" if as_winner else "l_"
        return {
            "form_overall":      row.get(f"{prefix}form_overall"),
            "form_surface":      row.get(f"{prefix}form_surface"),
            "matches_played":    row.get(f"{prefix}matches_played", 0),
            "matches_on_surface":row.get(f"{prefix}matches_on_surface", 0),
            "avg_1stServe_pct":  row.get(f"{prefix}avg_1stServe_pct"),
            "avg_1stWon_pct":    row.get(f"{prefix}avg_1stWon_pct"),
            "avg_2ndWon_pct":    row.get(f"{prefix}avg_2ndWon_pct"),
            "avg_bpSave_pct":    row.get(f"{prefix}avg_bpSave_pct"),
            "avg_ace_rate":      row.get(f"{prefix}avg_ace_rate"),
        }

    p1_id = p1_row["player_id"]
    p2_id = p2_row["player_id"]

    # Try getting features from winner perspective first, then loser
    f1 = last_features(p1_id, True)
    if not f1:
        f1 = last_features(p1_id, False)
    f2 = last_features(p2_id, True)
    if not f2:
        f2 = last_features(p2_id, False)

    # H2H between the two players
    h2h_matches = features[
        ((features["winner_id"] == p1_id) & (features["loser_id"] == p2_id)) |
        ((features["winner_id"] == p2_id) & (features["loser_id"] == p1_id))
    ]
    p1_h2h_wins = (h2h_matches["winner_id"] == p1_id).sum()
    p2_h2h_wins = (h2h_matches["winner_id"] == p2_id).sum()
    h2h_total   = len(h2h_matches)
    h2h_winrate = p1_h2h_wins / h2h_total if h2h_total > 0 else 0.5

    p1_elo_ov   = p1_row.get("elo_overall", 1500)
    p2_elo_ov   = p2_row.get("elo_overall", 1500)
    p1_elo_surf = p1_row.get(surf_col, p1_elo_ov)
    p2_elo_surf = p2_row.get(surf_col, p2_elo_ov)

    vec = {
        # Elo
        "elo_diff_overall":  p1_elo_ov - p2_elo_ov,
        "elo_diff_surface":  p1_elo_surf - p2_elo_surf,
        "w_elo_overall":     p1_elo_ov,
        "l_elo_overall":     p2_elo_ov,
        "w_elo_surface":     p1_elo_surf,
        "l_elo_surface":     p2_elo_surf,

        # Rank (use Elo rank as proxy if rank unavailable)
        "rank_diff":         0,
        "w_rank":            p1_row.get("rank_overall", 50),
        "l_rank":            p2_row.get("rank_overall", 50),

        # Age
        "w_age":             np.nan,
        "l_age":             np.nan,
        "age_diff":          0,

        # Form
        "w_form_overall":    f1.get("form_overall"),
        "l_form_overall":    f2.get("form_overall"),
        "w_form_surface":    f1.get("form_surface"),
        "l_form_surface":    f2.get("form_surface"),
        "form_diff_overall": (f1.get("form_overall") or 0.5) - (f2.get("form_overall") or 0.5),
        "form_diff_surface": (f1.get("form_surface") or 0.5) - (f2.get("form_surface") or 0.5),

        # Experience
        "w_matches_played":     f1.get("matches_played", 0),
        "l_matches_played":     f2.get("matches_played", 0),
        "w_matches_on_surface": f1.get("matches_on_surface", 0),
        "l_matches_on_surface": f2.get("matches_on_surface", 0),

        # Serve
        "w_avg_1stServe_pct": f1.get("avg_1stServe_pct"),
        "l_avg_1stServe_pct": f2.get("avg_1stServe_pct"),
        "w_avg_1stWon_pct":   f1.get("avg_1stWon_pct"),
        "l_avg_1stWon_pct":   f2.get("avg_1stWon_pct"),
        "w_avg_2ndWon_pct":   f1.get("avg_2ndWon_pct"),
        "l_avg_2ndWon_pct":   f2.get("avg_2ndWon_pct"),
        "w_avg_bpSave_pct":   f1.get("avg_bpSave_pct"),
        "l_avg_bpSave_pct":   f2.get("avg_bpSave_pct"),
        "w_avg_ace_rate":     f1.get("avg_ace_rate"),
        "l_avg_ace_rate":     f2.get("avg_ace_rate"),

        # H2H
        "h2h_wins_winner":    p1_h2h_wins,
        "h2h_wins_loser":     p2_h2h_wins,
        "h2h_total":          h2h_total,
        "h2h_winrate_winner": h2h_winrate,

        # Match context
        "best_of": 3,
    }

    # Surface dummies
    for s in ["Hard", "Clay", "Grass", "Carpet"]:
        vec[f"surface_{s}"] = 1 if s == surf else 0

    # Level dummies (assume Masters 1000 for hypothetical)
    for lv in ["Grand Slam", "Masters 1000", "ATP 500", "ATP 250", "Finals"]:
        vec[f"level_{lv}"] = 1 if lv == "Masters 1000" else 0

    row_df = pd.DataFrame([vec])
    row_df = row_df.reindex(columns=feature_names, fill_value=0)
    return row_df, {
        "p1_h2h_wins": int(p1_h2h_wins), "p2_h2h_wins": int(p2_h2h_wins),
        "h2h_total": int(h2h_total),
        "p1_elo_surf": p1_elo_surf, "p2_elo_surf": p2_elo_surf,
        "p1_elo_ov": p1_elo_ov, "p2_elo_ov": p2_elo_ov,
        "f1": f1, "f2": f2,
    }


# ─────────────────────────────────────────────
# MATCHUP VISUALIZATION
# ─────────────────────────────────────────────
def plot_matchup(p1_name, p2_name, surface, prob_p1, meta, p1_row, p2_row):
    prob_p2 = 1 - prob_p1
    surf_color = SURFACE_COLORS.get(surface.capitalize(), "#4A90D9")

    fig = plt.figure(figsize=(16, 9), facecolor="#0f0f1a")
    fig.suptitle(f"{p1_name}  vs  {p2_name}",
                 fontsize=24, fontweight="bold", color="white", y=0.97)
    fig.text(0.5, 0.92, f"Surface: {surface.capitalize()}  |  Hypothetical Matchup Analysis",
             ha="center", fontsize=12, color="#aaa")

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35,
                          left=0.05, right=0.97, top=0.88, bottom=0.06)

    # ── 1. Win probability gauge ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#0f0f1a")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.axis("off")

    # Background bar
    ax1.barh(0.5, 1, height=0.18, color="#1a1a2e", left=0)
    # P1 bar
    ax1.barh(0.5, prob_p1, height=0.18,
             color="#00d4ff" if prob_p1 >= 0.5 else "#ff6b6b", left=0)
    # P2 bar
    ax1.barh(0.5, prob_p2, height=0.18,
             color="#ff6b6b" if prob_p1 >= 0.5 else "#00d4ff",
             left=prob_p1, alpha=0.7)

    ax1.text(prob_p1/2, 0.5, f"{prob_p1*100:.1f}%",
             ha="center", va="center", fontsize=14,
             fontweight="bold", color="white")
    ax1.text(prob_p1 + prob_p2/2, 0.5, f"{prob_p2*100:.1f}%",
             ha="center", va="center", fontsize=14,
             fontweight="bold", color="white")
    ax1.text(0.5, 0.75, "Win Probability", ha="center",
             fontsize=11, color="white")
    ax1.text(0.02, 0.32, p1_name.split()[-1], fontsize=9, color="#00d4ff")
    ax1.text(0.98, 0.32, p2_name.split()[-1], fontsize=9,
             color="#ff6b6b", ha="right")

    winner = p1_name if prob_p1 >= 0.5 else p2_name
    ax1.text(0.5, 0.15,
             f"{'🏆 ' + winner.split()[-1] + ' favoured'}",
             ha="center", fontsize=11, color="#FFD700", fontweight="bold")
    ax1.set_title("Prediction", color="white", fontsize=11, pad=10)

    # ── 2. Elo comparison ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#0f0f1a")
    cats = ["Overall\nElo", f"{surface}\nElo"]
    p1_vals = [meta["p1_elo_ov"], meta["p1_elo_surf"]]
    p2_vals = [meta["p2_elo_ov"], meta["p2_elo_surf"]]
    x = np.arange(len(cats))
    b1 = ax2.bar(x-0.2, p1_vals, 0.35, color="#00d4ff",
                 label=p1_name.split()[-1], alpha=0.85)
    b2 = ax2.bar(x+0.2, p2_vals, 0.35, color="#ff6b6b",
                 label=p2_name.split()[-1], alpha=0.85)
    for bar, val in zip(list(b1)+list(b2), p1_vals+p2_vals):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                f"{val:.0f}", ha="center", fontsize=9,
                color="white", fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(cats, color="white")
    ax2.set_title("Elo Ratings", color="white", fontsize=11)
    ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values(): spine.set_color("#333")
    ymin = min(p1_vals+p2_vals) - 100
    ymax = max(p1_vals+p2_vals) + 80
    ax2.set_ylim(ymin, ymax)

    # ── 3. H2H record ──
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#0f0f1a"); ax3.axis("off")
    h2h_data = [
        ("H2H Matches", str(meta["h2h_total"])),
        (f"{p1_name.split()[-1]} H2H Wins", str(meta["p1_h2h_wins"])),
        (f"{p2_name.split()[-1]} H2H Wins", str(meta["p2_h2h_wins"])),
    ]
    y_pos = 0.85
    ax3.text(0.5, 1.0, "Head-to-Head Record",
             ha="center", fontsize=11, color="white", fontweight="bold",
             transform=ax3.transAxes)
    for label, val in h2h_data:
        ax3.text(0.1, y_pos, label, fontsize=10, color="#aaa",
                 transform=ax3.transAxes)
        ax3.text(0.85, y_pos, val, fontsize=12, color="white",
                 fontweight="bold", ha="right", transform=ax3.transAxes)
        y_pos -= 0.2

    if meta["h2h_total"] > 0:
        p1_pct = meta["p1_h2h_wins"] / meta["h2h_total"]
        ax3.barh(0.1, p1_pct, height=0.08, color="#00d4ff",
                 transform=ax3.transAxes)
        ax3.barh(0.1, 1-p1_pct, height=0.08, left=p1_pct,
                 color="#ff6b6b", transform=ax3.transAxes, alpha=0.7)

    # ── 4. Serve comparison ──
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor("#0f0f1a")
    f1 = meta["f1"]; f2 = meta["f2"]
    serve_metrics = ["avg_1stServe_pct", "avg_1stWon_pct",
                     "avg_2ndWon_pct",   "avg_bpSave_pct"]
    serve_labels  = ["1st Serve%", "1st Won%", "2nd Won%", "BP Save%"]

    p1_serve = [f1.get(m, np.nan) for m in serve_metrics]
    p2_serve = [f2.get(m, np.nan) for m in serve_metrics]

    x = np.arange(len(serve_labels))
    ax4.bar(x-0.2, [v if not np.isnan(v) else 0 for v in p1_serve],
            0.35, color="#00d4ff", label=p1_name.split()[-1], alpha=0.85)
    ax4.bar(x+0.2, [v if not np.isnan(v) else 0 for v in p2_serve],
            0.35, color="#ff6b6b", label=p2_name.split()[-1], alpha=0.85)
    ax4.set_xticks(x); ax4.set_xticklabels(serve_labels, color="white", fontsize=8)
    ax4.set_title("Serve Comparison", color="white", fontsize=11)
    ax4.set_ylim(0, 1)
    ax4.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    ax4.tick_params(colors="white")
    for spine in ax4.spines.values(): spine.set_color("#333")

    # ── 5. Surface Elo across all surfaces ──
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor("#0f0f1a")
    surfs = ["overall", "hard", "clay", "grass"]
    p1_elos = [p1_row.get(f"elo_{s}", 1500) for s in surfs]
    p2_elos = [p2_row.get(f"elo_{s}", 1500) for s in surfs]
    x = np.arange(len(surfs))
    ax5.plot(x, p1_elos, "o-", color="#00d4ff",
             linewidth=2, label=p1_name.split()[-1])
    ax5.plot(x, p2_elos, "s-", color="#ff6b6b",
             linewidth=2, label=p2_name.split()[-1])
    ax5.fill_between(x, p1_elos, p2_elos,
                     where=[p1>p2 for p1,p2 in zip(p1_elos,p2_elos)],
                     alpha=0.15, color="#00d4ff")
    ax5.fill_between(x, p1_elos, p2_elos,
                     where=[p1<p2 for p1,p2 in zip(p1_elos,p2_elos)],
                     alpha=0.15, color="#ff6b6b")
    ax5.set_xticks(x)
    ax5.set_xticklabels([s.capitalize() for s in surfs], color="white")
    ax5.set_title("Elo Across Surfaces", color="white", fontsize=11)
    ax5.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    ax5.tick_params(colors="white")
    for spine in ax5.spines.values(): spine.set_color("#333")

    # ── 6. Key advantages ──
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor("#0f0f1a"); ax6.axis("off")
    ax6.text(0.5, 1.02, "Key Advantages", ha="center",
             fontsize=11, color="white", fontweight="bold",
             transform=ax6.transAxes)

    advantages = []
    surf_lower = surface.lower()
    if p1_row.get(f"elo_{surf_lower}", 1500) > p2_row.get(f"elo_{surf_lower}", 1500) + 50:
        advantages.append((p1_name.split()[-1], f"Better {surface} Elo", "#00d4ff"))
    elif p2_row.get(f"elo_{surf_lower}", 1500) > p1_row.get(f"elo_{surf_lower}", 1500) + 50:
        advantages.append((p2_name.split()[-1], f"Better {surface} Elo", "#ff6b6b"))

    if meta["p1_h2h_wins"] > meta["p2_h2h_wins"]:
        advantages.append((p1_name.split()[-1], "H2H record lead", "#00d4ff"))
    elif meta["p2_h2h_wins"] > meta["p1_h2h_wins"]:
        advantages.append((p2_name.split()[-1], "H2H record lead", "#ff6b6b"))

    def cmp_serve(key, label, p1_better="higher"):
        v1 = f1.get(key, np.nan); v2 = f2.get(key, np.nan)
        if np.isnan(v1) or np.isnan(v2): return
        if abs(v1-v2) > 0.03:
            winner_name = p1_name.split()[-1] if v1>v2 else p2_name.split()[-1]
            col = "#00d4ff" if v1>v2 else "#ff6b6b"
            advantages.append((winner_name, label, col))

    cmp_serve("avg_bpSave_pct",    "Better BP saving")
    cmp_serve("avg_1stServe_pct",  "Better 1st serve %")
    cmp_serve("avg_1stWon_pct",    "Better 1st serve points")

    y_pos = 0.88
    for player, adv, col in advantages[:6]:
        ax6.text(0.05, y_pos, f"✓ {player}:", fontsize=9,
                 color=col, fontweight="bold", transform=ax6.transAxes)
        ax6.text(0.42, y_pos, adv, fontsize=9,
                 color="#ddd", transform=ax6.transAxes)
        y_pos -= 0.16

    if not advantages:
        ax6.text(0.5, 0.5, "Very evenly matched!",
                 ha="center", va="center", fontsize=11,
                 color="#FFD700", transform=ax6.transAxes)

    # Save
    p1_slug = p1_name.replace(" ", "_")
    p2_slug = p2_name.replace(" ", "_")
    fname = REPORTS_DIR / f"{p1_slug}_vs_{p2_slug}_{surface}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓  Saved: {fname}")
    return fname


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage: python src/matchup.py \"Player One\" \"Player Two\" [Surface]")
        print("       Surface: Hard | Clay | Grass  (default: Hard)")
        sys.exit(1)

    # Parse — last arg is surface if it matches known surfaces
    known_surfaces = {"hard", "clay", "grass", "carpet"}
    if args[-1].lower() in known_surfaces:
        surface  = args[-1].capitalize()
        players  = args[:-1]
    else:
        surface = "Hard"
        players = args

    if len(players) < 2:
        print("Please provide two player names.")
        sys.exit(1)

    p1_name_query = players[0]
    p2_name_query = players[1]

    print(f"\n{'='*55}")
    print(f"  Matchup: {p1_name_query}  vs  {p2_name_query}")
    print(f"  Surface: {surface}")
    print(f"{'='*55}\n")

    model, feature_names, matches, elo_df, features = load_artifacts()

    p1_row = find_player(matches, elo_df, p1_name_query)
    p2_row = find_player(matches, elo_df, p2_name_query)

    if p1_row is None or p2_row is None:
        sys.exit(1)

    p1_name = p1_row["player_name"]
    p2_name = p2_row["player_name"]
    print(f"  {p1_name}  vs  {p2_name}  on {surface}\n")

    # Build feature vector
    X_pred, meta = build_feature_vector(
        p1_row, p2_row, surface, features, feature_names)

    # Predict
    prob_p1 = model.predict_proba(X_pred)[0][1]
    prob_p2 = 1 - prob_p1

    print(f"  ── Prediction ──")
    print(f"  {p1_name:<30} {prob_p1*100:.1f}%")
    print(f"  {p2_name:<30} {prob_p2*100:.1f}%")
    print(f"  → Favoured: {'  ' + p1_name if prob_p1 >= 0.5 else p2_name}")

    print(f"\n  ── H2H Record ──")
    print(f"  {p1_name} leads: {meta['p1_h2h_wins']}–{meta['p2_h2h_wins']}"
          f" ({meta['h2h_total']} matches)")

    print(f"\n  ── Elo on {surface} ──")
    print(f"  {p1_name:<30} {meta['p1_elo_surf']:.0f}")
    print(f"  {p2_name:<30} {meta['p2_elo_surf']:.0f}")

    # Plot
    plot_matchup(p1_name, p2_name, surface, prob_p1, meta, p1_row, p2_row)
    print(f"\n  Done! Check reports/ folder 🎾\n")