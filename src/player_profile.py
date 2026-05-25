"""
src/player_profile.py
=====================
Phase 3 — Player Profiling & Analytics

For any player, computes:
  - Serve dominance metrics
  - Surface win rates
  - Strength / weakness radar profile
  - Tournament level performance
  - Career timeline (Elo over time)

Usage:
    python src/player_profile.py "Novak Djokovic"
    python src/player_profile.py "Rafael Nadal"

Outputs (in reports/):
    <player>_profile.png      - radar chart + surface bar chart
    <player>_career.png       - Elo over time
    <player>_summary.txt      - text report
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

DATA_DIR    = Path("data")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

SURFACE_COLORS = {
    "Hard":   "#4A90D9",
    "Clay":   "#C0622D",
    "Grass":  "#2E8B57",
    "Carpet": "#8B6914",
    "Overall":"#7B2D8B",
}

LEVEL_ORDER = ["Grand Slam", "Masters 1000", "ATP 500", "ATP 250",
               "Finals", "Olympics", "Davis Cup", "Challenger", "Other"]


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_data():
    matches = pd.read_parquet(DATA_DIR / "matches_with_elo.parquet")
    elo_df  = pd.read_parquet(DATA_DIR / "elo_ratings.parquet")
    return matches, elo_df


# ─────────────────────────────────────────────
# PLAYER LOOKUP
# ─────────────────────────────────────────────
def find_player(matches: pd.DataFrame, name: str):
    """Return player_id and canonical name."""
    mask_w = matches["winner_name"].str.lower().str.contains(name.lower(), na=False)
    mask_l = matches["loser_name"].str.lower().str.contains(name.lower(), na=False)
    candidates = pd.concat([
        matches[mask_w][["winner_id", "winner_name"]].rename(
            columns={"winner_id": "pid", "winner_name": "pname"}),
        matches[mask_l][["loser_id",  "loser_name" ]].rename(
            columns={"loser_id":  "pid", "loser_name":  "pname"}),
    ]).drop_duplicates("pid")

    if candidates.empty:
        print(f"  ✗  Player '{name}' not found.")
        return None, None

    if len(candidates) > 1:
        print(f"  Multiple matches found for '{name}':")
        for _, r in candidates.iterrows():
            print(f"    {r['pname']}  (id={r['pid']})")
        # Pick the one with most matches
        counts = []
        for _, r in candidates.iterrows():
            c = ((matches["winner_id"] == r["pid"]) | (matches["loser_id"] == r["pid"])).sum()
            counts.append((c, r["pid"], r["pname"]))
        counts.sort(reverse=True)
        pid, pname = counts[0][1], counts[0][2]
        print(f"  → Using: {pname}")
    else:
        pid   = candidates.iloc[0]["pid"]
        pname = candidates.iloc[0]["pname"]

    return pid, pname


# ─────────────────────────────────────────────
# COMPUTE PLAYER STATS
# ─────────────────────────────────────────────
def compute_stats(matches: pd.DataFrame, elo_df: pd.DataFrame,
                  pid, pname: str) -> dict:
    """Build full stats dict for a player."""

    w = matches[matches["winner_id"] == pid].copy()
    l = matches[matches["loser_id"]  == pid].copy()
    all_m = len(w) + len(l)

    # ── Overall ──
    win_rate = len(w) / all_m if all_m > 0 else 0

    # ── Surface breakdown ──
    surface_stats = {}
    for surf in ["Hard", "Clay", "Grass", "Carpet"]:
        ww = w[w["surface"] == surf]
        ll = l[l["surface"] == surf]
        total = len(ww) + len(ll)
        surface_stats[surf] = {
            "wins":     len(ww),
            "losses":   len(ll),
            "total":    total,
            "win_rate": len(ww) / total if total > 0 else np.nan,
        }

    # ── Serve metrics (as winner) ──
    def safe_mean(series):
        return series.dropna().mean() if len(series.dropna()) > 0 else np.nan

    serve = {
        "first_serve_pct": safe_mean(w["w_1stServe_pct"]),
        "first_won_pct":   safe_mean(w["w_1stWon_pct"]),
        "second_won_pct":  safe_mean(w["w_2ndWon_pct"]),
        "bp_save_pct":     safe_mean(w["w_bpSave_pct"]),
        "ace_rate":        safe_mean(w["w_ace"] / w["w_svpt"].replace(0, np.nan)),
        "df_rate":         safe_mean(w["w_df"]  / w["w_svpt"].replace(0, np.nan)),
    }

    # Also get serve stats when losing (combined for better estimate)
    serve_l = {
        "first_serve_pct": safe_mean(l["l_1stServe_pct"]),
        "first_won_pct":   safe_mean(l["l_1stWon_pct"]),
        "second_won_pct":  safe_mean(l["l_2ndWon_pct"]),
        "bp_save_pct":     safe_mean(l["l_bpSave_pct"]),
    }

    # Average win/loss serve stats for truer picture
    def avg_wl(key):
        a, b = serve.get(key, np.nan), serve_l.get(key, np.nan)
        vals = [x for x in [a, b] if not np.isnan(x)]
        return np.mean(vals) if vals else np.nan

    combined_serve = {k: avg_wl(k) for k in serve_l}
    combined_serve["ace_rate"] = serve["ace_rate"]
    combined_serve["df_rate"]  = serve["df_rate"]

    # ── Tournament level ──
    level_map = {
        "G": "Grand Slam", "M": "Masters 1000", "A": "ATP 500",
        "D": "Davis Cup",   "F": "Finals",       "C": "Challenger",
        "S": "ATP 250",     "O": "Olympics",
    }
    w2 = w.copy(); l2 = l.copy()
    w2["level_name"] = w2["tourney_level"].map(level_map).fillna("Other")
    l2["level_name"] = l2["tourney_level"].map(level_map).fillna("Other")

    level_stats = {}
    for lvl in LEVEL_ORDER:
        wc = (w2["level_name"] == lvl).sum()
        lc = (l2["level_name"] == lvl).sum()
        tot = wc + lc
        level_stats[lvl] = {
            "wins": wc, "losses": lc, "total": tot,
            "win_rate": wc / tot if tot > 0 else np.nan,
        }

    # ── Elo ratings ──
    elo_row = elo_df[elo_df["player_name"] == pname]
    elo_vals = {}
    if not elo_row.empty:
        r = elo_row.iloc[0]
        for s in ["overall", "hard", "clay", "grass", "carpet"]:
            elo_vals[s] = r.get(f"elo_{s}", INITIAL_ELO := 1500)

    # ── Career Elo timeline ──
    timeline_w = w[["tourney_date", "w_elo_overall"]].rename(
        columns={"w_elo_overall": "elo"})
    timeline_l = l[["tourney_date", "l_elo_overall"]].rename(
        columns={"l_elo_overall": "elo"})
    timeline = (pd.concat([timeline_w, timeline_l])
                  .sort_values("tourney_date")
                  .dropna())

    # ── Grand Slam record ──
    gs_w = w2[w2["level_name"] == "Grand Slam"]
    gs_l = l2[l2["level_name"] == "Grand Slam"]
    gs_titles = len(gs_w[gs_w["round"] == "F"]) if "round" in gs_w.columns else 0

    return {
        "name":          pname,
        "pid":           pid,
        "total_matches": all_m,
        "wins":          len(w),
        "losses":        len(l),
        "win_rate":      win_rate,
        "surface":       surface_stats,
        "serve":         combined_serve,
        "levels":        level_stats,
        "elo":           elo_vals,
        "timeline":      timeline,
        "gs_titles":     gs_titles,
        "career_start":  matches[
            (matches["winner_id"] == pid) | (matches["loser_id"] == pid)
        ]["tourney_date"].min(),
        "career_end":    matches[
            (matches["winner_id"] == pid) | (matches["loser_id"] == pid)
        ]["tourney_date"].max(),
    }


# ─────────────────────────────────────────────
# RADAR CHART
# ─────────────────────────────────────────────
def radar_chart(ax, stats: dict, color: str):
    """Draw a radar chart of player profile."""
    serve = stats["serve"]
    surf  = stats["surface"]

    labels = [
        "Overall\nWin Rate",
        "1st Serve\n%",
        "1st Serve\nWon %",
        "2nd Serve\nWon %",
        "BP Save\n%",
        "Clay\nWin Rate",
        "Grass\nWin Rate",
        "Hard\nWin Rate",
    ]

    def pct(val, default=0.5):
        return val if not np.isnan(val) else default

    values = [
        pct(stats["win_rate"]),
        pct(serve.get("first_serve_pct", np.nan), 0.6),
        pct(serve.get("first_won_pct",   np.nan), 0.7),
        pct(serve.get("second_won_pct",  np.nan), 0.5),
        pct(serve.get("bp_save_pct",     np.nan), 0.6),
        pct(surf["Clay"]["win_rate"],    0.5),
        pct(surf["Grass"]["win_rate"],   0.5),
        pct(surf["Hard"]["win_rate"],    0.5),
    ]

    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values += values[:1]

    ax.set_facecolor("#0f0f1a")
    ax.plot(angles, values, "o-", linewidth=2, color=color)
    ax.fill(angles, values, alpha=0.25, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8, color="white")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], size=6, color="#888")
    ax.grid(color="#333", linestyle="--", linewidth=0.5)
    ax.spines["polar"].set_color("#444")


# ─────────────────────────────────────────────
# FULL PROFILE PLOT
# ─────────────────────────────────────────────
def plot_profile(stats: dict, color: str = "#00d4ff"):
    fig = plt.figure(figsize=(16, 10), facecolor="#0f0f1a")
    fig.suptitle(
        f"  {stats['name']}  —  Player Profile",
        fontsize=22, fontweight="bold", color="white", x=0.05, ha="left"
    )

    # Sub-title stats line
    years = ""
    if pd.notna(stats["career_start"]) and pd.notna(stats["career_end"]):
        y1 = stats["career_start"].year
        y2 = stats["career_end"].year
        years = f"{y1} – {y2}"

    fig.text(0.05, 0.92,
        f"W/L: {stats['wins']}/{stats['losses']}  |  "
        f"Win Rate: {stats['win_rate']*100:.1f}%  |  "
        f"GS Titles: {stats['gs_titles']}  |  "
        f"Overall Elo: {stats['elo'].get('overall', 0):.0f}  |  "
        f"Career: {years}",
        fontsize=11, color="#aaa"
    )

    # ── Layout ──
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35,
                          left=0.05, right=0.97, top=0.88, bottom=0.07)

    # 1. Radar
    ax_radar = fig.add_subplot(gs[0, 0], polar=True)
    radar_chart(ax_radar, stats, color)
    ax_radar.set_title("Player Radar", color="white", pad=15, fontsize=11)

    # 2. Surface win rates
    ax_surf = fig.add_subplot(gs[0, 1])
    ax_surf.set_facecolor("#0f0f1a")
    surfaces = ["Hard", "Clay", "Grass", "Carpet"]
    wr   = [stats["surface"][s]["win_rate"] for s in surfaces]
    tots = [stats["surface"][s]["total"]    for s in surfaces]
    cols = [SURFACE_COLORS[s] for s in surfaces]
    bars = ax_surf.bar(surfaces, [w if not np.isnan(w) else 0 for w in wr],
                       color=cols, edgecolor="#222", linewidth=0.8)
    ax_surf.axhline(0.5, color="#555", linestyle="--", linewidth=0.8)
    for bar, w, t in zip(bars, wr, tots):
        if not np.isnan(w):
            ax_surf.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{w*100:.0f}%\n({t})", ha="center", va="bottom",
                        fontsize=8, color="white")
    ax_surf.set_ylim(0, 1.05)
    ax_surf.set_title("Win Rate by Surface", color="white", fontsize=11)
    ax_surf.set_ylabel("Win Rate", color="#aaa")
    ax_surf.tick_params(colors="white")
    ax_surf.spines[:].set_color("#333")
    for spine in ax_surf.spines.values(): spine.set_color("#333")
    ax_surf.set_facecolor("#0f0f1a")
    ax_surf.yaxis.label.set_color("#aaa")
    ax_surf.tick_params(axis="x", colors="white")
    ax_surf.tick_params(axis="y", colors="#aaa")

    # 3. Elo by surface
    ax_elo = fig.add_subplot(gs[0, 2])
    ax_elo.set_facecolor("#0f0f1a")
    elo_surfs = ["overall", "hard", "clay", "grass"]
    elo_vals  = [stats["elo"].get(s, 1500) for s in elo_surfs]
    elo_cols  = [SURFACE_COLORS.get(s.capitalize(), "#7B2D8B") for s in elo_surfs]
    bars2 = ax_elo.bar([s.capitalize() for s in elo_surfs], elo_vals,
                       color=elo_cols, edgecolor="#222")
    ax_elo.axhline(1500, color="#555", linestyle="--", linewidth=0.8, label="Average (1500)")
    for bar, val in zip(bars2, elo_vals):
        ax_elo.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f"{val:.0f}", ha="center", fontsize=9, color="white", fontweight="bold")
    ax_elo.set_title("Elo Rating by Surface", color="white", fontsize=11)
    ax_elo.tick_params(colors="white")
    for spine in ax_elo.spines.values(): spine.set_color("#333")
    ax_elo.set_facecolor("#0f0f1a")
    ax_elo.tick_params(axis="x", colors="white")
    ax_elo.tick_params(axis="y", colors="#aaa")
    ax_elo.set_ylim(min(elo_vals) - 100, max(elo_vals) + 80)

    # 4. Career Elo timeline
    ax_time = fig.add_subplot(gs[1, :2])
    ax_time.set_facecolor("#0f0f1a")
    tl = stats["timeline"]
    if len(tl) > 10:
        ax_time.plot(tl["tourney_date"], tl["elo"], color=color,
                    linewidth=1.5, alpha=0.9)
        ax_time.fill_between(tl["tourney_date"], tl["elo"].min(),
                             tl["elo"], alpha=0.1, color=color)
        ax_time.axhline(1500, color="#555", linestyle="--", linewidth=0.8)
        peak_idx = tl["elo"].idxmax()
        ax_time.scatter(tl.loc[peak_idx, "tourney_date"],
                       tl.loc[peak_idx, "elo"],
                       color="#FFD700", s=80, zorder=5, label=f"Peak: {tl.loc[peak_idx,'elo']:.0f}")
        ax_time.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=9)
    ax_time.set_title("Career Elo Timeline", color="white", fontsize=11)
    ax_time.tick_params(colors="white")
    for spine in ax_time.spines.values(): spine.set_color("#333")
    ax_time.set_facecolor("#0f0f1a")
    ax_time.tick_params(axis="x", colors="#aaa")
    ax_time.tick_params(axis="y", colors="#aaa")

    # 5. Tournament level performance
    ax_lvl = fig.add_subplot(gs[1, 2])
    ax_lvl.set_facecolor("#0f0f1a")
    lvl_data = [(lvl, stats["levels"][lvl]) for lvl in LEVEL_ORDER
                if stats["levels"][lvl]["total"] > 0]
    if lvl_data:
        lvl_names = [d[0].replace(" ", "\n") for d in lvl_data]
        lvl_wr    = [d[1]["win_rate"] for d in lvl_data]
        lvl_tot   = [d[1]["total"]    for d in lvl_data]
        colors_lvl = plt.cm.RdYlGn([w if not np.isnan(w) else 0.5 for w in lvl_wr])
        bars3 = ax_lvl.barh(lvl_names,
                            [w if not np.isnan(w) else 0 for w in lvl_wr],
                            color=colors_lvl, edgecolor="#222")
        ax_lvl.axvline(0.5, color="#555", linestyle="--", linewidth=0.8)
        for bar, wr2, tot in zip(bars3, lvl_wr, lvl_tot):
            if not np.isnan(wr2):
                ax_lvl.text(min(wr2 + 0.02, 0.95), bar.get_y() + bar.get_height()/2,
                           f"{wr2*100:.0f}% ({tot})",
                           va="center", fontsize=7, color="white")
        ax_lvl.set_xlim(0, 1.1)
    ax_lvl.set_title("Win Rate by Level", color="white", fontsize=11)
    ax_lvl.tick_params(colors="white")
    for spine in ax_lvl.spines.values(): spine.set_color("#333")
    ax_lvl.set_facecolor("#0f0f1a")
    ax_lvl.tick_params(axis="x", colors="#aaa")
    ax_lvl.tick_params(axis="y", colors="white", labelsize=7)

    fname = REPORTS_DIR / f"{stats['name'].replace(' ','_')}_profile.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓  Saved: {fname}")
    return fname


# ─────────────────────────────────────────────
# TEXT SUMMARY
# ─────────────────────────────────────────────
def text_summary(stats: dict) -> str:
    s = stats
    lines = [
        f"{'='*50}",
        f"  PLAYER PROFILE: {s['name']}",
        f"{'='*50}",
        f"  Career        : {s['career_start'].date() if pd.notna(s['career_start']) else 'N/A'}"
        f" → {s['career_end'].date() if pd.notna(s['career_end']) else 'N/A'}",
        f"  Total matches : {s['total_matches']}",
        f"  Win / Loss    : {s['wins']} / {s['losses']}",
        f"  Win rate      : {s['win_rate']*100:.1f}%",
        f"  GS Titles     : {s['gs_titles']}",
        f"",
        f"  Elo Ratings:",
    ]
    for surf, val in s["elo"].items():
        lines.append(f"    {surf.capitalize():<10}: {val:.0f}")

    lines += ["", "  Surface Record:"]
    for surf, d in s["surface"].items():
        if d["total"] > 0:
            wr = d["win_rate"] * 100 if not np.isnan(d["win_rate"]) else 0
            lines.append(f"    {surf:<8}: {d['wins']}W / {d['losses']}L  ({wr:.1f}%)")

    lines += ["", "  Serve Profile:"]
    for k, v in s["serve"].items():
        if not np.isnan(v):
            lines.append(f"    {k:<22}: {v*100:.1f}%")

    lines += ["", "  Strengths:"]
    if s["serve"].get("bp_save_pct", 0) > 0.65:
        lines.append("    ✓ Elite break point saving")
    if s["surface"]["Clay"]["win_rate"] > 0.7 if not np.isnan(s["surface"]["Clay"]["win_rate"]) else False:
        lines.append("    ✓ Dominant clay court player")
    if s["surface"]["Hard"]["win_rate"] > 0.7 if not np.isnan(s["surface"]["Hard"]["win_rate"]) else False:
        lines.append("    ✓ Dominant hard court player")
    if s["win_rate"] > 0.75:
        lines.append("    ✓ All-time elite win rate (>75%)")
    if s["elo"].get("overall", 0) > 1900:
        lines.append("    ✓ All-time great Elo rating (>1900)")

    lines += ["", "  Weaknesses:"]
    if s["serve"].get("df_rate", 0) > 0.06:
        lines.append("    ✗ High double fault rate")
    if s["serve"].get("second_won_pct", 1) < 0.48:
        lines.append("    ✗ Vulnerable 2nd serve")
    if s["surface"]["Grass"]["win_rate"] < 0.6 if not np.isnan(s["surface"]["Grass"]["win_rate"]) else False:
        lines.append("    ✗ Below average grass court record")

    lines.append(f"\n{'='*50}\n")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    player_name = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Novak Djokovic"

    print(f"\n  Building profile for: {player_name}")
    matches, elo_df = load_data()

    pid, pname = find_player(matches, player_name)
    if pid is None:
        sys.exit(1)

    print(f"  Found: {pname}  (id={pid})")
    stats = compute_stats(matches, elo_df, pid, pname)

    # Print text summary
    summary = text_summary(stats)
    print(summary)

    # Save text report
    txt_path = REPORTS_DIR / f"{pname.replace(' ','_')}_summary.txt"
    txt_path.write_text(summary, encoding="utf-8")

    # Save visual profile
    plot_profile(stats, color="#00d4ff")

    print(f"  Reports saved to: {REPORTS_DIR}/\n")