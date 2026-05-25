"""
src/elo.py
==========
Computes surface-specific Elo ratings for every player across all matches.

Elo logic:
  - Separate rating per surface (Hard, Clay, Grass, Carpet, Overall)
  - K-factor decays as player plays more matches (more certain = smaller updates)
  - Ratings update chronologically match-by-match

Usage (standalone):
    python src/elo.py

Outputs:
    data/elo_ratings.parquet   - final Elo per player per surface
    data/matches_with_elo.parquet - original match data + Elo columns appended
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR       = Path("data")
INITIAL_ELO    = 1500
K_BASE         = 32        # starting K-factor
K_MIN          = 10        # minimum K-factor (experienced players)
K_DECAY_RATE   = 0.05      # how fast K decays with match count
SURFACES       = ["Hard", "Clay", "Grass", "Carpet", "Overall"]


# ─────────────────────────────────────────────
# CORE ELO FUNCTIONS
# ─────────────────────────────────────────────
def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that player A beats player B."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def k_factor(match_count: int) -> float:
    """Dynamic K: high for new players, stabilises for veterans."""
    return max(K_MIN, K_BASE * np.exp(-K_DECAY_RATE * match_count))


def update_elo(rating_winner: float, rating_loser: float,
               count_winner: int, count_loser: int):
    """Return updated (winner_elo, loser_elo) after one match."""
    exp_w = expected_score(rating_winner, rating_loser)
    exp_l = 1 - exp_w

    kw = k_factor(count_winner)
    kl = k_factor(count_loser)

    new_winner = rating_winner + kw * (1 - exp_w)
    new_loser  = rating_loser  + kl * (0 - exp_l)
    return new_winner, new_loser


# ─────────────────────────────────────────────
# MAIN COMPUTATION
# ─────────────────────────────────────────────
def compute_elo(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk through every match chronologically and update Elo ratings.

    Returns:
        matches_elo : original df with pre-match Elo columns added
        final_elo   : final Elo rating per player per surface
    """
    # Sort chronologically
    df = df.sort_values("tourney_date").reset_index(drop=True)

    # Dictionaries: player_id -> {surface -> elo}
    ratings = {}   # player_id -> {surface: float}
    counts  = {}   # player_id -> {surface: int}   (match count per surface)

    def get_rating(pid, surface):
        return ratings.setdefault(pid, {}).get(surface, INITIAL_ELO)

    def get_count(pid, surface):
        return counts.setdefault(pid, {}).get(surface, 0)

    def set_rating(pid, surface, val):
        ratings[pid][surface] = val

    def inc_count(pid, surface):
        counts[pid][surface] = counts[pid].get(surface, 0) + 1

    # Storage for pre-match Elo (what we append to the dataframe)
    records = {
        "w_elo_overall": [], "l_elo_overall": [],
        "w_elo_surface": [], "l_elo_surface": [],
        "elo_diff_overall": [], "elo_diff_surface": [],
    }

    print("  Computing Elo ratings match-by-match...")
    total = len(df)
    milestone = total // 10

    for i, row in df.iterrows():
        if i % milestone == 0:
            print(f"    {100 * i // total}%  ({i:,} / {total:,} matches)")

        wid = row["winner_id"]
        lid = row["loser_id"]
        surf = row["surface"] if row["surface"] in SURFACES else "Overall"

        # Pre-match ratings (what we record)
        w_elo_ov   = get_rating(wid, "Overall")
        l_elo_ov   = get_rating(lid, "Overall")
        w_elo_surf = get_rating(wid, surf)
        l_elo_surf = get_rating(lid, surf)

        records["w_elo_overall"].append(round(w_elo_ov, 2))
        records["l_elo_overall"].append(round(l_elo_ov, 2))
        records["w_elo_surface"].append(round(w_elo_surf, 2))
        records["l_elo_surface"].append(round(l_elo_surf, 2))
        records["elo_diff_overall"].append(round(w_elo_ov - l_elo_ov, 2))
        records["elo_diff_surface"].append(round(w_elo_surf - l_elo_surf, 2))

        # Update overall Elo
        new_w_ov, new_l_ov = update_elo(
            w_elo_ov, l_elo_ov, get_count(wid, "Overall"), get_count(lid, "Overall")
        )
        set_rating(wid, "Overall", new_w_ov)
        set_rating(lid, "Overall", new_l_ov)
        inc_count(wid, "Overall")
        inc_count(lid, "Overall")

        # Update surface Elo (if known surface)
        if surf != "Overall":
            new_w_s, new_l_s = update_elo(
                w_elo_surf, l_elo_surf, get_count(wid, surf), get_count(lid, surf)
            )
            set_rating(wid, surf, new_w_s)
            set_rating(lid, surf, new_l_s)
            inc_count(wid, surf)
            inc_count(lid, surf)

    print("    100%  done ✓")

    # Append Elo columns to match dataframe
    for col, vals in records.items():
        df[col] = vals

    # ── Build final ratings table ──
    all_players = set(df["winner_id"].unique()) | set(df["loser_id"].unique())

    # Build name lookup
    name_lookup = (
        pd.concat([
            df[["winner_id", "winner_name"]].rename(columns={"winner_id": "player_id", "winner_name": "player_name"}),
            df[["loser_id",  "loser_name" ]].rename(columns={"loser_id":  "player_id", "loser_name":  "player_name"}),
        ])
        .drop_duplicates("player_id")
        .set_index("player_id")["player_name"]
    )

    rows = []
    for pid in all_players:
        row = {"player_id": pid, "player_name": name_lookup.get(pid, "Unknown")}
        for surf in SURFACES:
            row[f"elo_{surf.lower()}"] = round(get_rating(pid, surf), 2)
            row[f"matches_{surf.lower()}"] = get_count(pid, surf)
        rows.append(row)

    final_elo = pd.DataFrame(rows).sort_values("elo_overall", ascending=False).reset_index(drop=True)
    final_elo["rank_overall"] = final_elo["elo_overall"].rank(ascending=False).astype(int)

    return df, final_elo


# ─────────────────────────────────────────────
# QUICK PREVIEW HELPERS
# ─────────────────────────────────────────────
def top_players(final_elo: pd.DataFrame, surface: str = "overall", n: int = 20):
    col = f"elo_{surface.lower()}"
    return (
        final_elo[["player_name", col, f"matches_{surface.lower()}"]]
        .sort_values(col, ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def player_profile(final_elo: pd.DataFrame, name: str):
    mask = final_elo["player_name"].str.lower().str.contains(name.lower())
    row  = final_elo[mask]
    if row.empty:
        print(f"Player '{name}' not found.")
        return None
    return row[[
        "player_name",
        "elo_overall", "elo_hard", "elo_clay", "elo_grass", "elo_carpet",
        "matches_overall", "matches_hard", "matches_clay", "matches_grass",
    ]].iloc[0]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import time
    t0 = time.time()

    print("\n" + "="*50)
    print("  Tennis Elo Rating Engine")
    print("="*50)

    df = pd.read_parquet(DATA_DIR / "matches_clean.parquet")
    print(f"  Loaded {len(df):,} matches\n")

    matches_elo, final_elo = compute_elo(df)

    # Save
    matches_elo.to_parquet(DATA_DIR / "matches_with_elo.parquet", index=False)
    final_elo.to_parquet(DATA_DIR / "elo_ratings.parquet", index=False)
    print(f"\n  Saved matches_with_elo.parquet & elo_ratings.parquet")

    # Preview
    print("\n  Top 15 players by Overall Elo:")
    print(top_players(final_elo, "overall", 15).to_string(index=False))

    print("\n  Top 10 Clay Elo:")
    print(top_players(final_elo, "clay", 10).to_string(index=False))

    print("\n  Top 10 Hard Elo:")
    print(top_players(final_elo, "hard", 10).to_string(index=False))

    print(f"\n  Done in {time.time() - t0:.1f}s  🎾\n")