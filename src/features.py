"""
src/features.py
===============
Builds the full feature matrix for matchup prediction.

For every match it computes rolling/historical stats for both players:
  - Recent form (last 20 matches win rate)
  - Surface-specific win rates
  - Serve dominance metrics
  - Head-to-head (H2H) record
  - Elo differentials (from elo.py output)
  - Rank differential

Usage:
    python src/features.py

Outputs:
    data/features.parquet   - one row per match, ready for ML model
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR   = Path("data")
FORM_WINDOW = 20    # rolling matches for recent form


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def safe_div(a, b):
    return np.where(b > 0, a / b, np.nan)


# ─────────────────────────────────────────────
# STEP 1 — ROLLING PLAYER STATS
# ─────────────────────────────────────────────
def build_player_history(df: pd.DataFrame) -> dict:
    """
    Walk chronologically. For each match, record the player's
    rolling stats BEFORE the match (no data leakage).

    Returns a dict: match_index -> {player_id -> stats_dict}
    """
    print("  Building player rolling histories...")

    # We'll store each player's running list of match results
    # Format per entry: {won, surface, w_1stServe_pct, w_bpSave_pct, ...}
    history = {}   # player_id -> list of result dicts

    def get_stats(pid, surface, n=FORM_WINDOW):
        """Compute rolling stats from the last n matches for player pid."""
        h = history.get(pid, [])
        if not h:
            return {}

        recent      = h[-n:]
        surface_all = [x for x in h if x["surface"] == surface]
        recent_surf = surface_all[-n:]

        def win_rate(matches):
            if not matches:
                return np.nan
            return sum(m["won"] for m in matches) / len(matches)

        def avg(matches, key):
            vals = [m[key] for m in matches if m.get(key) is not None]
            return np.mean(vals) if vals else np.nan

        return {
            "form_overall":      win_rate(recent),
            "form_surface":      win_rate(recent_surf),
            "matches_played":    len(h),
            "matches_on_surface": len(surface_all),
            "avg_1stServe_pct":  avg(recent, "sv1_pct"),
            "avg_1stWon_pct":    avg(recent, "sv1won_pct"),
            "avg_2ndWon_pct":    avg(recent, "sv2won_pct"),
            "avg_bpSave_pct":    avg(recent, "bp_save_pct"),
            "avg_ace_rate":      avg(recent, "ace_rate"),
            "avg_df_rate":       avg(recent, "df_rate"),
            "avg_1stServe_surf": avg(recent_surf, "sv1_pct"),
            "avg_bpSave_surf":   avg(recent_surf, "bp_save_pct"),
        }

    # Storage: index -> {winner_stats, loser_stats}
    prematch = {}

    for i, row in df.iterrows():
        wid  = row["winner_id"]
        lid  = row["loser_id"]
        surf = row["surface"]

        # Capture pre-match stats
        prematch[i] = {
            "w": get_stats(wid, surf),
            "l": get_stats(lid, surf),
        }

        # Update winner history
        history.setdefault(wid, []).append({
            "won": 1, "surface": surf,
            "sv1_pct":     row.get("w_1stServe_pct"),
            "sv1won_pct":  row.get("w_1stWon_pct"),
            "sv2won_pct":  row.get("w_2ndWon_pct"),
            "bp_save_pct": row.get("w_bpSave_pct"),
            "ace_rate":    row["w_ace"] / row["w_svpt"] if pd.notna(row.get("w_svpt")) and row.get("w_svpt", 0) > 0 else None,
            "df_rate":     row["w_df"]  / row["w_svpt"] if pd.notna(row.get("w_svpt")) and row.get("w_svpt", 0) > 0 else None,
        })

        # Update loser history
        history.setdefault(lid, []).append({
            "won": 0, "surface": surf,
            "sv1_pct":     row.get("l_1stServe_pct"),
            "sv1won_pct":  row.get("l_1stWon_pct"),
            "sv2won_pct":  row.get("l_2ndWon_pct"),
            "bp_save_pct": row.get("l_bpSave_pct"),
            "ace_rate":    row["l_ace"] / row["l_svpt"] if pd.notna(row.get("l_svpt")) and row.get("l_svpt", 0) > 0 else None,
            "df_rate":     row["l_df"]  / row["l_svpt"] if pd.notna(row.get("l_svpt")) and row.get("l_svpt", 0) > 0 else None,
        })

        if i % 10000 == 0:
            print(f"    {100 * i // len(df)}%  ({i:,} matches processed)")

    print("    100%  done ✓")
    return prematch


# ─────────────────────────────────────────────
# STEP 2 — HEAD-TO-HEAD
# ─────────────────────────────────────────────
def build_h2h(df: pd.DataFrame) -> dict:
    """
    Walk chronologically. For each match, capture H2H record
    between the two players BEFORE this match.

    Returns dict: (player_a_id, player_b_id) -> {wins_a, wins_b, total}
    """
    print("  Building H2H records...")
    h2h = {}   # frozenset({wid, lid}) -> {wid: wins, lid: wins}
    prematch_h2h = {}

    for i, row in df.iterrows():
        wid = row["winner_id"]
        lid = row["loser_id"]
        key = frozenset({wid, lid})

        record = h2h.get(key, {wid: 0, lid: 0})
        prematch_h2h[i] = {
            "h2h_wins_winner":  record.get(wid, 0),
            "h2h_wins_loser":   record.get(lid, 0),
            "h2h_total":        record.get(wid, 0) + record.get(lid, 0),
            "h2h_winrate_winner": (
                record.get(wid, 0) / (record.get(wid, 0) + record.get(lid, 0))
                if (record.get(wid, 0) + record.get(lid, 0)) > 0 else 0.5
            ),
        }

        # Update after recording pre-match
        h2h.setdefault(key, {wid: 0, lid: 0})
        h2h[key][wid] = h2h[key].get(wid, 0) + 1

    print("    done ✓")
    return prematch_h2h


# ─────────────────────────────────────────────
# STEP 3 — ASSEMBLE FEATURE MATRIX
# ─────────────────────────────────────────────
def assemble_features(df: pd.DataFrame,
                      prematch: dict,
                      h2h_records: dict) -> pd.DataFrame:
    print("  Assembling feature matrix...")

    rows = []
    for i, row in df.iterrows():
        pm = prematch[i]
        h2h = h2h_records[i]
        ws = pm["w"]
        ls = pm["l"]

        feat = {
            # ── Match identifiers ──
            "match_index":      i,
            "tourney_date":     row["tourney_date"],
            "tourney_name":     row.get("tourney_name"),
            "tourney_level":    row.get("tourney_level"),
            "surface":          row.get("surface"),
            "round":            row.get("round"),
            "best_of":          row.get("best_of"),

            # ── Players ──
            "winner_id":   row["winner_id"],
            "winner_name": row["winner_name"],
            "loser_id":    row["loser_id"],
            "loser_name":  row["loser_name"],

            # ── Elo (from elo.py) ──
            "w_elo_overall":    row.get("w_elo_overall"),
            "l_elo_overall":    row.get("l_elo_overall"),
            "w_elo_surface":    row.get("w_elo_surface"),
            "l_elo_surface":    row.get("l_elo_surface"),
            "elo_diff_overall": row.get("elo_diff_overall"),
            "elo_diff_surface": row.get("elo_diff_surface"),

            # ── Rank ──
            "w_rank":       row.get("winner_rank"),
            "l_rank":       row.get("loser_rank"),
            "rank_diff":    row.get("rank_diff"),
            "upset":        row.get("upset"),

            # ── Age ──
            "w_age":        row.get("winner_age"),
            "l_age":        row.get("loser_age"),
            "age_diff":     (row.get("winner_age", np.nan) or np.nan) - (row.get("loser_age", np.nan) or np.nan),

            # ── Form ──
            "w_form_overall":   ws.get("form_overall"),
            "l_form_overall":   ls.get("form_overall"),
            "w_form_surface":   ws.get("form_surface"),
            "l_form_surface":   ls.get("form_surface"),
            "form_diff_overall": (ws.get("form_overall") or np.nan) - (ls.get("form_overall") or np.nan),
            "form_diff_surface": (ws.get("form_surface") or np.nan) - (ls.get("form_surface") or np.nan),

            # ── Experience ──
            "w_matches_played":    ws.get("matches_played", 0),
            "l_matches_played":    ls.get("matches_played", 0),
            "w_matches_on_surface": ws.get("matches_on_surface", 0),
            "l_matches_on_surface": ls.get("matches_on_surface", 0),

            # ── Serve stats (rolling) ──
            "w_avg_1stServe_pct":  ws.get("avg_1stServe_pct"),
            "l_avg_1stServe_pct":  ls.get("avg_1stServe_pct"),
            "w_avg_1stWon_pct":    ws.get("avg_1stWon_pct"),
            "l_avg_1stWon_pct":    ls.get("avg_1stWon_pct"),
            "w_avg_2ndWon_pct":    ws.get("avg_2ndWon_pct"),
            "l_avg_2ndWon_pct":    ls.get("avg_2ndWon_pct"),
            "w_avg_bpSave_pct":    ws.get("avg_bpSave_pct"),
            "l_avg_bpSave_pct":    ls.get("avg_bpSave_pct"),
            "w_avg_ace_rate":      ws.get("avg_ace_rate"),
            "l_avg_ace_rate":      ls.get("avg_ace_rate"),
            "w_avg_df_rate":       ws.get("avg_df_rate"),
            "l_avg_df_rate":       ls.get("avg_df_rate"),

            # ── Surface-specific serve ──
            "w_avg_1stServe_surf": ws.get("avg_1stServe_surf"),
            "l_avg_1stServe_surf": ls.get("avg_1stServe_surf"),
            "w_avg_bpSave_surf":   ws.get("avg_bpSave_surf"),
            "l_avg_bpSave_surf":   ls.get("avg_bpSave_surf"),

            # ── H2H ──
            "h2h_wins_winner":    h2h["h2h_wins_winner"],
            "h2h_wins_loser":     h2h["h2h_wins_loser"],
            "h2h_total":          h2h["h2h_total"],
            "h2h_winrate_winner": h2h["h2h_winrate_winner"],

            # ── Target ──
            "target": 1,  # winner always wins; we'll flip randomly during training
        }
        rows.append(feat)

    features = pd.DataFrame(rows)
    print(f"  ✓  Feature matrix: {features.shape[0]:,} rows × {features.shape[1]} cols")
    return features


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import time
    t0 = time.time()

    print("\n" + "="*50)
    print("  Tennis Feature Engineering")
    print("="*50)

    # Load Elo-enriched matches
    elo_path = DATA_DIR / "matches_with_elo.parquet"
    if not elo_path.exists():
        print("  ⚠  matches_with_elo.parquet not found.")
        print("  Run: python src/elo.py  first!\n")
        exit(1)

    df = pd.read_parquet(elo_path)
    df = df.sort_values("tourney_date").reset_index(drop=True)
    print(f"  Loaded {len(df):,} matches with Elo\n")

    prematch    = build_player_history(df)
    h2h_records = build_h2h(df)
    features    = assemble_features(df, prematch, h2h_records)

    # Save
    out = DATA_DIR / "features.parquet"
    features.to_parquet(out, index=False)
    print(f"\n  Saved → {out}")

    # Quick summary
    print(f"\n  Feature columns ({features.shape[1]}):")
    for col in features.columns:
        missing = features[col].isna().mean() * 100
        print(f"    {col:<35} {missing:5.1f}% missing")

    print(f"\n  Done in {time.time() - t0:.1f}s  🎾\n")