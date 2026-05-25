"""
tennis_pipeline.py
==================
One-shot pipeline to download, merge, and clean all Jeff Sackmann ATP match CSVs.

Usage:
    pip install pandas requests tqdm
    python tennis_pipeline.py

Output files (in ./data/):
    raw/        - original downloaded CSVs (kept for reproducibility)
    matches_raw.parquet     - merged, unmodified
    matches_clean.parquet   - fully cleaned, ready for feature engineering
    pipeline_report.txt     - summary stats and data quality report
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from io import StringIO

# ─────────────────────────────────────────────
# CONFIG — tweak these if needed
# ─────────────────────────────────────────────
BASE_URL   = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
YEAR_START = 1990       # change to 1968 for full history
YEAR_END   = 2024
SAVE_RAW   = True       # keep individual year CSVs
DATA_DIR   = Path("data")
RAW_DIR    = DATA_DIR / "raw"

# ─────────────────────────────────────────────
# COLUMN SCHEMA
# ─────────────────────────────────────────────
KEEP_COLS = [
    # Match metadata
    "tourney_id", "tourney_name", "surface", "draw_size",
    "tourney_level", "tourney_date", "match_num",
    "best_of", "round",

    # Winner
    "winner_id", "winner_name", "winner_hand", "winner_ht",
    "winner_ioc", "winner_age", "winner_rank", "winner_rank_points",

    # Loser
    "loser_id", "loser_name", "loser_hand", "loser_ht",
    "loser_ioc", "loser_age", "loser_rank", "loser_rank_points",

    # Score & stats
    "score", "minutes",

    # Winner serve stats
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon",
    "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced",

    # Loser serve stats
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon",
    "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced",
]

NUMERIC_COLS = [
    "winner_ht", "winner_age", "winner_rank", "winner_rank_points",
    "loser_ht", "loser_age", "loser_rank", "loser_rank_points",
    "minutes",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon",
    "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon",
    "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced",
]


# ─────────────────────────────────────────────
# STEP 1 — DOWNLOAD
# ─────────────────────────────────────────────
def download_all(year_start: int, year_end: int) -> list[pd.DataFrame]:
    """Download all yearly match CSVs from Sackmann's GitHub."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    years  = list(range(year_start, year_end + 1))

    print(f"\n{'='*55}")
    print(f"  Downloading ATP matches {year_start}–{year_end}")
    print(f"{'='*55}")

    for year in tqdm(years, desc="Downloading"):
        fname = f"atp_matches_{year}.csv"
        fpath = RAW_DIR / fname
        url   = f"{BASE_URL}/{fname}"

        # Use cached file if available
        if fpath.exists():
            df = pd.read_csv(fpath, low_memory=False)
        else:
            try:
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                if SAVE_RAW:
                    fpath.write_text(resp.text, encoding="utf-8")
                df = pd.read_csv(StringIO(resp.text), low_memory=False)
                time.sleep(0.15)  # be polite to GitHub
            except requests.HTTPError:
                tqdm.write(f"  ⚠  {fname} not found — skipping")
                continue
            except Exception as e:
                tqdm.write(f"  ✗  {fname} error: {e}")
                continue

        df["source_year"] = year
        frames.append(df)

    print(f"\n  ✓  Downloaded {len(frames)} files ({year_end - year_start + 1} years requested)")
    return frames


# ─────────────────────────────────────────────
# STEP 2 — MERGE
# ─────────────────────────────────────────────
def merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate all year DataFrames into one."""
    print("\n  Merging all years...")
    raw = pd.concat(frames, ignore_index=True)
    print(f"  ✓  Raw shape: {raw.shape[0]:,} rows × {raw.shape[1]} cols")
    return raw


# ─────────────────────────────────────────────
# STEP 3 — CLEAN
# ─────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    print("\n  Cleaning...")
    original_rows = len(df)
    report_lines  = []

    # ── 3a. Keep only wanted columns (add any that exist) ──
    existing = [c for c in KEEP_COLS if c in df.columns]
    missing  = [c for c in KEEP_COLS if c not in df.columns]
    df = df[existing + ["source_year"]].copy()
    if missing:
        report_lines.append(f"Columns absent in source: {missing}")

    # ── 3b. Parse dates ──
    df["tourney_date"] = pd.to_datetime(
        df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    df["year"]  = df["tourney_date"].dt.year
    df["month"] = df["tourney_date"].dt.month

    # ── 3c. Drop walkovers / retirements with no stats ──
    retired_mask = df["score"].str.contains(
        r"RET|W/O|DEF|BYE|ABN", case=False, na=False
    )
    n_retired = retired_mask.sum()
    df = df[~retired_mask].reset_index(drop=True)
    report_lines.append(f"Removed {n_retired:,} retirements/walkovers")

    # ── 3d. Numeric coercion ──
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 3e. Surface normalisation ──
    surface_map = {
        "hard": "Hard", "clay": "Clay", "grass": "Grass",
        "carpet": "Carpet", "indoor hard": "Hard",
    }
    df["surface"] = (
        df["surface"]
        .str.strip()
        .str.lower()
        .map(surface_map)
        .fillna("Unknown")
    )

    # ── 3f. Hand normalisation ──
    for col in ("winner_hand", "loser_hand"):
        if col in df.columns:
            df[col] = df[col].str.upper().replace({"U": np.nan, "": np.nan})

    # ── 3g. Drop rows missing both players' names / IDs ──
    essential = ["winner_id", "loser_id", "winner_name", "loser_name"]
    essential = [c for c in essential if c in df.columns]
    before    = len(df)
    df        = df.dropna(subset=essential)
    report_lines.append(f"Dropped {before - len(df):,} rows missing player identity")

    # ── 3h. Remove impossible values ──
    if "minutes" in df.columns:
        df.loc[df["minutes"] < 10,   "minutes"] = np.nan   # sub-10 min match impossible
        df.loc[df["minutes"] > 480,  "minutes"] = np.nan   # 8 hr+ match impossible
    if "winner_age" in df.columns:
        df.loc[df["winner_age"] < 14, "winner_age"] = np.nan
        df.loc[df["winner_age"] > 55, "winner_age"] = np.nan
    if "loser_age" in df.columns:
        df.loc[df["loser_age"]  < 14, "loser_age"]  = np.nan
        df.loc[df["loser_age"]  > 55, "loser_age"]  = np.nan

    # ── 3i. Derived serve metrics ──
    # 1st serve percentage
    df["w_1stServe_pct"] = np.where(
        df["w_svpt"] > 0, df["w_1stIn"] / df["w_svpt"], np.nan
    )
    df["l_1stServe_pct"] = np.where(
        df["l_svpt"] > 0, df["l_1stIn"] / df["l_svpt"], np.nan
    )

    # 1st serve win %
    df["w_1stWon_pct"] = np.where(
        df["w_1stIn"] > 0, df["w_1stWon"] / df["w_1stIn"], np.nan
    )
    df["l_1stWon_pct"] = np.where(
        df["l_1stIn"] > 0, df["l_1stWon"] / df["l_1stIn"], np.nan
    )

    # 2nd serve win %
    df["w_2ndWon_pct"] = np.where(
        (df["w_svpt"] - df["w_1stIn"]) > 0,
        df["w_2ndWon"] / (df["w_svpt"] - df["w_1stIn"]), np.nan
    )
    df["l_2ndWon_pct"] = np.where(
        (df["l_svpt"] - df["l_1stIn"]) > 0,
        df["l_2ndWon"] / (df["l_svpt"] - df["l_1stIn"]), np.nan
    )

    # Break point save %
    df["w_bpSave_pct"] = np.where(
        df["w_bpFaced"] > 0, df["w_bpSaved"] / df["w_bpFaced"], np.nan
    )
    df["l_bpSave_pct"] = np.where(
        df["l_bpFaced"] > 0, df["l_bpSaved"] / df["l_bpFaced"], np.nan
    )

    # ── 3j. Rank differential ──
    if "winner_rank" in df.columns and "loser_rank" in df.columns:
        df["rank_diff"] = df["loser_rank"] - df["winner_rank"]
        df["upset"]     = (df["rank_diff"] < 0).astype(int)   # 1 = lower-ranked player won

    # ── 3k. Tourney level labels ──
    level_map = {
        "G": "Grand Slam", "M": "Masters 1000", "A": "ATP 500",
        "D": "Davis Cup",   "F": "Finals",       "C": "Challenger",
        "S": "ATP 250",     "O": "Olympics",
    }
    if "tourney_level" in df.columns:
        df["tourney_level_name"] = df["tourney_level"].map(level_map).fillna("Other")

    final_rows = len(df)
    pct_kept   = 100 * final_rows / original_rows
    report_lines.insert(0, f"Original rows : {original_rows:,}")
    report_lines.append(f"Final rows     : {final_rows:,}  ({pct_kept:.1f}% kept)")
    report_lines.append(f"Columns        : {df.shape[1]}")
    report_lines.append(f"Date range     : {df['tourney_date'].min().date()} → {df['tourney_date'].max().date()}")
    report_lines.append(f"Surfaces       : {df['surface'].value_counts().to_dict()}")
    report_lines.append(f"Missing rate   :\n{(df[NUMERIC_COLS].isna().mean() * 100).round(1).to_string()}")

    print(f"  ✓  Clean shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df, report_lines


# ─────────────────────────────────────────────
# STEP 4 — SAVE
# ─────────────────────────────────────────────
def save(raw: pd.DataFrame, clean_df: pd.DataFrame, report: list[str]):
    DATA_DIR.mkdir(exist_ok=True)

    raw_path   = DATA_DIR / "matches_raw.parquet"
    clean_path = DATA_DIR / "matches_clean.parquet"
    report_path = DATA_DIR / "pipeline_report.txt"

    raw.to_parquet(raw_path, index=False)
    clean_df.to_parquet(clean_path, index=False)

    report_txt = "\n".join(report)
    report_path.write_text(report_txt, encoding="utf-8")


    print(f"\n  Saved:")
    print(f"    {raw_path}   ({raw_path.stat().st_size / 1e6:.1f} MB)")
    print(f"    {clean_path} ({clean_path.stat().st_size / 1e6:.1f} MB)")
    print(f"    {report_path}")
    print(f"\n  Pipeline report:\n{'─'*45}")
    print(report_txt)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    t0     = time.time()
    frames = download_all(YEAR_START, YEAR_END)
    raw    = merge_frames(frames)
    raw.to_parquet(DATA_DIR / "matches_raw.parquet", index=False)

    clean_df, report = clean(raw)
    save(raw, clean_df, report)

    mins = (time.time() - t0) / 60
    print(f"\n  Done in {mins:.1f} min  🎾\n")