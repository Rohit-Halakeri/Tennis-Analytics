"""
app/dashboard.py
================
Phase 5 — Interactive Streamlit Dashboard

Run:
    streamlit run app/dashboard.py

Features:
    - Player search & profile viewer
    - Head-to-head matchup predictor
    - Surface analysis
    - Leaderboard (top players by Elo)
"""

import streamlit as st
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
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Tennis Analytics",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f0f1a; color: white; }
    .main-header {
        font-size: 2.8rem; font-weight: 900;
        background: linear-gradient(90deg, #00d4ff, #7B2D8B);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 1rem 0;
    }
    .sub-header {
        text-align: center; color: #aaa;
        font-size: 1rem; margin-bottom: 2rem;
    }
    .metric-card {
        background: #1a1a2e; border-radius: 12px;
        padding: 1.2rem; border: 1px solid #333;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem; font-weight: 700; color: #00d4ff;
    }
    .metric-label {
        font-size: 0.85rem; color: #aaa; margin-top: 0.3rem;
    }
    .player-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 16px; padding: 1.5rem;
        border: 1px solid #333; margin: 0.5rem 0;
    }
    .win-bar-container {
        background: #1a1a2e; border-radius: 50px;
        height: 40px; overflow: hidden; margin: 1rem 0;
        display: flex; align-items: center;
    }
    .advantage-item {
        background: #1a1a2e; border-radius: 8px;
        padding: 0.5rem 1rem; margin: 0.3rem 0;
        border-left: 3px solid #00d4ff;
    }
    div[data-testid="stSidebar"] {
        background-color: #0d0d1a;
        border-right: 1px solid #333;
    }
    .stSelectbox > div { background: #1a1a2e; }
    .stButton > button {
        background: linear-gradient(90deg, #00d4ff, #7B2D8B);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 2rem; font-weight: 700;
        font-size: 1rem; width: 100%;
        transition: transform 0.2s;
    }
    .stButton > button:hover { transform: scale(1.02); }
    h1, h2, h3 { color: white !important; }
    .stTabs [data-baseweb="tab"] { color: #aaa; }
    .stTabs [aria-selected="true"] { color: #00d4ff !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING (cached)
# ─────────────────────────────────────────────
DATA_DIR   = Path("data")
MODELS_DIR = Path("models")
SURFACE_COLORS = {
    "Hard": "#4A90D9", "Clay": "#C0622D",
    "Grass": "#2E8B57", "Carpet": "#8B6914",
}

@st.cache_resource
def load_all():
    matches  = pd.read_parquet(DATA_DIR / "matches_with_elo.parquet")
    elo_df   = pd.read_parquet(DATA_DIR / "elo_ratings.parquet")
    features = pd.read_parquet(DATA_DIR / "features.parquet")
    model    = xgb.XGBClassifier()
    model.load_model(MODELS_DIR / "xgb_model.json")
    feat_names = (MODELS_DIR / "feature_names.txt").read_text(
        encoding="utf-8").strip().split("\n")
    return matches, elo_df, features, model, feat_names

matches, elo_df, features, model, feat_names = load_all()

# Player list for dropdowns
player_list = sorted(elo_df[elo_df["matches_overall"] >= 20]["player_name"].tolist())


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def get_player_row(name):
    return elo_df[elo_df["player_name"] == name].iloc[0]

def get_player_stats(name):
    row  = get_player_row(name)
    pid  = row["player_id"]
    w    = matches[matches["winner_id"] == pid]
    l    = matches[matches["loser_id"]  == pid]
    total = len(w) + len(l)

    surf_stats = {}
    for s in ["Hard", "Clay", "Grass", "Carpet"]:
        ww = w[w["surface"] == s]
        ll = l[l["surface"] == s]
        tot = len(ww) + len(ll)
        surf_stats[s] = {
            "wins": len(ww), "losses": len(ll), "total": tot,
            "win_rate": len(ww)/tot if tot > 0 else np.nan
        }

    def sm(s): return s.dropna().mean() if len(s.dropna()) > 0 else np.nan

    timeline_w = w[["tourney_date","w_elo_overall"]].rename(columns={"w_elo_overall":"elo"})
    timeline_l = l[["tourney_date","l_elo_overall"]].rename(columns={"l_elo_overall":"elo"})
    timeline = pd.concat([timeline_w, timeline_l]).sort_values("tourney_date").dropna()

    level_map = {"G":"Grand Slam","M":"Masters 1000","A":"ATP 500",
                 "S":"ATP 250","F":"Finals","D":"Davis Cup","O":"Olympics"}
    w2 = w.copy(); l2 = l.copy()
    w2["lvl"] = w2["tourney_level"].map(level_map).fillna("Other")
    l2["lvl"] = l2["tourney_level"].map(level_map).fillna("Other")
    gs_titles = len(w2[(w2["lvl"]=="Grand Slam") & (w2["round"]=="F")]) if "round" in w2.columns else 0

    return {
        "name": name, "pid": pid,
        "wins": len(w), "losses": len(l), "total": total,
        "win_rate": len(w)/total if total > 0 else 0,
        "elo": dict(row),
        "surface": surf_stats,
        "timeline": timeline,
        "gs_titles": gs_titles,
        "serve": {
            "first_serve_pct": sm(pd.concat([w["w_1stServe_pct"], l["l_1stServe_pct"]])),
            "first_won_pct":   sm(pd.concat([w["w_1stWon_pct"],   l["l_1stWon_pct"]])),
            "second_won_pct":  sm(pd.concat([w["w_2ndWon_pct"],   l["l_2ndWon_pct"]])),
            "bp_save_pct":     sm(pd.concat([w["w_bpSave_pct"],   l["l_bpSave_pct"]])),
        }
    }

def predict_matchup(p1_name, p2_name, surface):
    p1 = get_player_row(p1_name)
    p2 = get_player_row(p2_name)
    p1_id = p1["player_id"]; p2_id = p2["player_id"]
    surf  = surface.capitalize()
    surf_col = f"elo_{surf.lower()}"

    def last_feat(pid, as_w):
        col = "winner_id" if as_w else "loser_id"
        pf  = features[features[col]==pid].sort_values("tourney_date")
        if pf.empty: return {}
        r = pf.iloc[-1]; px = "w_" if as_w else "l_"
        return {k: r.get(f"{px}{k}") for k in [
            "form_overall","form_surface","matches_played","matches_on_surface",
            "avg_1stServe_pct","avg_1stWon_pct","avg_2ndWon_pct",
            "avg_bpSave_pct","avg_ace_rate"]}

    f1 = last_feat(p1_id, True) or last_feat(p1_id, False)
    f2 = last_feat(p2_id, True) or last_feat(p2_id, False)

    h2h_m = features[
        ((features["winner_id"]==p1_id)&(features["loser_id"]==p2_id))|
        ((features["winner_id"]==p2_id)&(features["loser_id"]==p1_id))
    ]
    p1_h2h = (h2h_m["winner_id"]==p1_id).sum()
    p2_h2h = (h2h_m["winner_id"]==p2_id).sum()
    h2h_tot = len(h2h_m)
    h2h_wr  = p1_h2h/h2h_tot if h2h_tot > 0 else 0.5

    p1_elo_ov=p1.get("elo_overall",1500); p2_elo_ov=p2.get("elo_overall",1500)
    p1_elo_s =p1.get(surf_col,p1_elo_ov); p2_elo_s =p2.get(surf_col,p2_elo_ov)

    vec = {
        "elo_diff_overall": p1_elo_ov-p2_elo_ov,
        "elo_diff_surface": p1_elo_s-p2_elo_s,
        "w_elo_overall": p1_elo_ov, "l_elo_overall": p2_elo_ov,
        "w_elo_surface": p1_elo_s,  "l_elo_surface": p2_elo_s,
        "rank_diff":0, "w_rank":p1.get("rank_overall",50), "l_rank":p2.get("rank_overall",50),
        "w_age":np.nan,"l_age":np.nan,"age_diff":0,
        "w_form_overall":f1.get("form_overall"),"l_form_overall":f2.get("form_overall"),
        "w_form_surface":f1.get("form_surface"),"l_form_surface":f2.get("form_surface"),
        "form_diff_overall":(f1.get("form_overall") or .5)-(f2.get("form_overall") or .5),
        "form_diff_surface":(f1.get("form_surface") or .5)-(f2.get("form_surface") or .5),
        "w_matches_played":f1.get("matches_played",0),"l_matches_played":f2.get("matches_played",0),
        "w_matches_on_surface":f1.get("matches_on_surface",0),
        "l_matches_on_surface":f2.get("matches_on_surface",0),
        "w_avg_1stServe_pct":f1.get("avg_1stServe_pct"),"l_avg_1stServe_pct":f2.get("avg_1stServe_pct"),
        "w_avg_1stWon_pct":f1.get("avg_1stWon_pct"),    "l_avg_1stWon_pct":f2.get("avg_1stWon_pct"),
        "w_avg_2ndWon_pct":f1.get("avg_2ndWon_pct"),    "l_avg_2ndWon_pct":f2.get("avg_2ndWon_pct"),
        "w_avg_bpSave_pct":f1.get("avg_bpSave_pct"),    "l_avg_bpSave_pct":f2.get("avg_bpSave_pct"),
        "w_avg_ace_rate":f1.get("avg_ace_rate"),         "l_avg_ace_rate":f2.get("avg_ace_rate"),
        "h2h_wins_winner":p1_h2h,"h2h_wins_loser":p2_h2h,
        "h2h_total":h2h_tot,"h2h_winrate_winner":h2h_wr,"best_of":3,
    }
    for s in ["Hard","Clay","Grass","Carpet"]:
        vec[f"surface_{s}"] = 1 if s==surf else 0
    for lv in ["Grand Slam","Masters 1000","ATP 500","ATP 250","Finals"]:
        vec[f"level_{lv}"] = 1 if lv=="Masters 1000" else 0

    X = pd.DataFrame([vec]).reindex(columns=feat_names, fill_value=0)
    prob = model.predict_proba(X)[0][1]
    return prob, p1_h2h, p2_h2h, h2h_tot, p1_elo_s, p2_elo_s, f1, f2


def make_radar(ax, stats, color, name):
    labels = ["Win\nRate","1st Serve\n%","1st Won\n%",
              "2nd Won\n%","BP Save\n%","Clay\nWR","Grass\nWR","Hard\nWR"]
    def pv(v, d=0.5): return v if v and not np.isnan(v) else d
    serve = stats["serve"]; surf = stats["surface"]
    values = [
        pv(stats["win_rate"]),
        pv(serve.get("first_serve_pct"), 0.6),
        pv(serve.get("first_won_pct"),   0.7),
        pv(serve.get("second_won_pct"),  0.5),
        pv(serve.get("bp_save_pct"),     0.6),
        pv(surf["Clay"]["win_rate"]),
        pv(surf["Grass"]["win_rate"]),
        pv(surf["Hard"]["win_rate"]),
    ]
    N = len(labels)
    angles = [n/N*2*np.pi for n in range(N)]; angles += angles[:1]
    values += values[:1]
    ax.set_facecolor("#0f0f1a")
    ax.plot(angles, values, "o-", lw=2, color=color)
    ax.fill(angles, values, alpha=0.2, color=color)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, size=7, color="white")
    ax.set_ylim(0,1); ax.set_yticks([.25,.5,.75,1])
    ax.set_yticklabels(["25%","50%","75%","100%"], size=5, color="#666")
    ax.grid(color="#333", ls="--", lw=0.5)
    ax.spines["polar"].set_color("#444")
    ax.set_title(name.split()[-1], color=color, fontsize=10, pad=12)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎾 Tennis Analytics")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠  Home",
        "👤  Player Profile",
        "⚔️   Matchup Predictor",
        "🏆  Leaderboard",
    ])
    st.markdown("---")
    st.markdown("**Data:** ATP 1990–2024")
    st.markdown("**Matches:** 108,573")
    st.markdown("**Players:** ~3,000+")
    st.markdown("**Model:** XGBoost")
    st.markdown("---")
    st.markdown("<small style='color:#555'>Built with ❤️ using<br>Pandas · XGBoost · Streamlit</small>",
                unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────
if "Home" in page:
    st.markdown('<div class="main-header">🎾 Tennis Analytics Dashboard</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">35 years of ATP data · Surface-specific Elo · XGBoost matchup prediction</div>',
                unsafe_allow_html=True)

    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-value">108K</div><div class="metric-label">Total Matches</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-value">35</div><div class="metric-label">Years of Data</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-value">3000+</div><div class="metric-label">Players Profiled</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-value">~67%</div><div class="metric-label">Model Accuracy</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Quick matchup from home
    st.markdown("### ⚡ Quick Matchup")
    qc1, qc2, qc3, qc4 = st.columns([3,3,2,1])
    with qc1:
        qp1 = st.selectbox("Player 1", player_list,
                            index=player_list.index("Novak Djokovic") if "Novak Djokovic" in player_list else 0,
                            key="home_p1")
    with qc2:
        qp2 = st.selectbox("Player 2", player_list,
                            index=player_list.index("Rafael Nadal") if "Rafael Nadal" in player_list else 1,
                            key="home_p2")
    with qc3:
        qsurf = st.selectbox("Surface", ["Hard","Clay","Grass","Carpet"], key="home_surf")
    with qc4:
        st.markdown("<br>", unsafe_allow_html=True)
        go = st.button("Predict 🎾")

    if go and qp1 != qp2:
        prob, p1h, p2h, h2h_tot, p1es, p2es, f1, f2 = predict_matchup(qp1, qp2, qsurf)
        prob2 = 1 - prob
        winner = qp1 if prob >= 0.5 else qp2
        wp     = max(prob, prob2) * 100

        st.markdown(f"### 🏆 {winner} favoured — {wp:.1f}% win probability")
        p1_pct = int(prob * 100)
        p2_pct = 100 - p1_pct
        bar_html = f"""
        <div style="display:flex;height:44px;border-radius:22px;overflow:hidden;margin:1rem 0;">
            <div style="width:{p1_pct}%;background:#00d4ff;display:flex;align-items:center;
                        justify-content:center;color:white;font-weight:700;font-size:1.1rem;">
                {qp1.split()[-1]} {p1_pct}%
            </div>
            <div style="width:{p2_pct}%;background:#ff6b6b;display:flex;align-items:center;
                        justify-content:center;color:white;font-weight:700;font-size:1.1rem;">
                {p2_pct}% {qp2.split()[-1]}
            </div>
        </div>"""
        st.markdown(bar_html, unsafe_allow_html=True)

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{p1es:.0f}</div><div class="metric-label">{qp1.split()[-1]} {qsurf} Elo</div></div>', unsafe_allow_html=True)
        with mc2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{h2h_tot}</div><div class="metric-label">H2H Matches</div></div>', unsafe_allow_html=True)
        with mc3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{p2es:.0f}</div><div class="metric-label">{qp2.split()[-1]} {qsurf} Elo</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🌍 Big 3 Quick Comparison")
    big3 = ["Novak Djokovic","Rafael Nadal","Roger Federer"]
    bc = st.columns(3)
    for i, name in enumerate(big3):
        if name in player_list:
            r = get_player_row(name)
            with bc[i]:
                wr = elo_df[elo_df["player_name"]==name].iloc[0]
                w_ = matches[matches["winner_id"]==r["player_id"]]
                l_ = matches[matches["loser_id"]==r["player_id"]]
                tot = len(w_)+len(l_)
                win_pct = len(w_)/tot*100 if tot>0 else 0
                st.markdown(f"""
                <div class="player-card">
                    <h3 style="color:#00d4ff;margin:0">{name}</h3>
                    <p style="color:#FFD700;margin:0.3rem 0">Elo: {r['elo_overall']:.0f}</p>
                    <p style="color:#aaa;margin:0">Win Rate: {win_pct:.1f}%</p>
                    <p style="color:#aaa;margin:0">Clay: {r['elo_clay']:.0f} | Hard: {r['elo_hard']:.0f} | Grass: {r['elo_grass']:.0f}</p>
                </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: PLAYER PROFILE
# ─────────────────────────────────────────────
elif "Player Profile" in page:
    st.markdown("## 👤 Player Profile")
    selected = st.selectbox("Search player", player_list,
                             index=player_list.index("Novak Djokovic") if "Novak Djokovic" in player_list else 0)

    with st.spinner(f"Loading {selected}..."):
        stats = get_player_stats(selected)

    row = get_player_row(selected)

    # Header metrics
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-value">{stats["wins"]}</div><div class="metric-label">Wins</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="metric-value">{stats["losses"]}</div><div class="metric-label">Losses</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div class="metric-value">{stats["win_rate"]*100:.1f}%</div><div class="metric-label">Win Rate</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><div class="metric-value">{stats["gs_titles"]}</div><div class="metric-label">GS Titles</div></div>', unsafe_allow_html=True)
    with c5: st.markdown(f'<div class="metric-card"><div class="metric-value">{row["elo_overall"]:.0f}</div><div class="metric-label">Overall Elo</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Charts row 1
    col1, col2, col3 = st.columns([1,1,1])

    with col1:
        st.markdown("#### 🕸️ Player Radar")
        fig, ax = plt.subplots(1,1, figsize=(4,4), subplot_kw={"projection":"polar"},
                               facecolor="#0f0f1a")
        make_radar(ax, stats, "#00d4ff", selected)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("#### 📊 Win Rate by Surface")
        fig, ax = plt.subplots(figsize=(4,4), facecolor="#0f0f1a")
        ax.set_facecolor("#0f0f1a")
        surfs = ["Hard","Clay","Grass","Carpet"]
        wrs   = [stats["surface"][s]["win_rate"] for s in surfs]
        tots  = [stats["surface"][s]["total"]    for s in surfs]
        cols  = [SURFACE_COLORS[s] for s in surfs]
        bars  = ax.bar(surfs, [w if w and not np.isnan(w) else 0 for w in wrs],
                       color=cols, edgecolor="#222")
        ax.axhline(0.5, color="#555", ls="--", lw=0.8)
        for bar, wr, tot in zip(bars, wrs, tots):
            if wr and not np.isnan(wr):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                       f"{wr*100:.0f}%\n({tot})", ha="center",
                       fontsize=8, color="white")
        ax.set_ylim(0,1.1); ax.tick_params(colors="white")
        for spine in ax.spines.values(): spine.set_color("#333")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col3:
        st.markdown("#### 🏆 Elo by Surface")
        fig, ax = plt.subplots(figsize=(4,4), facecolor="#0f0f1a")
        ax.set_facecolor("#0f0f1a")
        elo_surfs = ["overall","hard","clay","grass"]
        elo_vals  = [row.get(f"elo_{s}",1500) for s in elo_surfs]
        elo_cols  = ["#7B2D8B","#4A90D9","#C0622D","#2E8B57"]
        bars2 = ax.bar([s.capitalize() for s in elo_surfs], elo_vals,
                       color=elo_cols, edgecolor="#222")
        ax.axhline(1500, color="#555", ls="--", lw=0.8)
        for bar, val in zip(bars2, elo_vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                   f"{val:.0f}", ha="center", fontsize=9,
                   color="white", fontweight="bold")
        ax.set_ylim(min(elo_vals)-100, max(elo_vals)+80)
        ax.tick_params(colors="white")
        for spine in ax.spines.values(): spine.set_color("#333")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Career timeline
    st.markdown("#### 📈 Career Elo Timeline")
    tl = stats["timeline"]
    if len(tl) > 10:
        fig, ax = plt.subplots(figsize=(12,3), facecolor="#0f0f1a")
        ax.set_facecolor("#0f0f1a")
        ax.plot(tl["tourney_date"], tl["elo"], color="#00d4ff", lw=1.5)
        ax.fill_between(tl["tourney_date"], tl["elo"].min(), tl["elo"],
                        alpha=0.1, color="#00d4ff")
        ax.axhline(1500, color="#555", ls="--", lw=0.8)
        peak_idx = tl["elo"].idxmax()
        ax.scatter(tl.loc[peak_idx,"tourney_date"], tl.loc[peak_idx,"elo"],
                  color="#FFD700", s=80, zorder=5,
                  label=f"Peak: {tl.loc[peak_idx,'elo']:.0f}")
        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
        ax.tick_params(colors="white")
        for spine in ax.spines.values(): spine.set_color("#333")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Strengths & weaknesses
    st.markdown("---")
    sw1, sw2 = st.columns(2)
    with sw1:
        st.markdown("#### ✅ Strengths")
        strengths = []
        if stats["win_rate"] > 0.75: strengths.append("Elite overall win rate (>75%)")
        if stats["surface"]["Clay"]["win_rate"] and stats["surface"]["Clay"]["win_rate"] > 0.7:
            strengths.append("Dominant clay court player")
        if stats["surface"]["Hard"]["win_rate"] and stats["surface"]["Hard"]["win_rate"] > 0.75:
            strengths.append("Dominant hard court player")
        if stats["surface"]["Grass"]["win_rate"] and stats["surface"]["Grass"]["win_rate"] > 0.8:
            strengths.append("Exceptional grass court player")
        if row.get("elo_overall",0) > 1900: strengths.append("All-time great Elo (>1900)")
        if stats["serve"].get("bp_save_pct",0) and stats["serve"]["bp_save_pct"] > 0.65:
            strengths.append("Elite break point saving")
        for s in strengths:
            st.markdown(f'<div class="advantage-item">✅ {s}</div>', unsafe_allow_html=True)
        if not strengths:
            st.markdown("*Not enough data to determine strengths*")

    with sw2:
        st.markdown("#### ⚠️ Weaknesses")
        weaknesses = []
        if stats["surface"]["Clay"]["win_rate"] and stats["surface"]["Clay"]["win_rate"] < 0.6:
            weaknesses.append("Below average on clay")
        if stats["surface"]["Grass"]["win_rate"] and stats["surface"]["Grass"]["win_rate"] < 0.6:
            weaknesses.append("Below average on grass")
        if stats["serve"].get("second_won_pct") and stats["serve"]["second_won_pct"] < 0.48:
            weaknesses.append("Vulnerable second serve")
        if stats["total"] < 100: weaknesses.append("Limited match data available")
        for w in weaknesses:
            st.markdown(f'<div class="advantage-item" style="border-left-color:#ff6b6b">⚠️ {w}</div>',
                       unsafe_allow_html=True)
        if not weaknesses:
            st.markdown("*No significant weaknesses detected*")


# ─────────────────────────────────────────────
# PAGE: MATCHUP PREDICTOR
# ─────────────────────────────────────────────
elif "Matchup" in page:
    st.markdown("## ⚔️ Matchup Predictor")
    st.markdown("Select two players and a surface to predict the winner.")

    c1, c2, c3 = st.columns([3,3,2])
    with c1:
        p1 = st.selectbox("Player 1", player_list,
                           index=player_list.index("Novak Djokovic") if "Novak Djokovic" in player_list else 0)
    with c2:
        p2 = st.selectbox("Player 2", player_list,
                           index=player_list.index("Rafael Nadal") if "Rafael Nadal" in player_list else 1)
    with c3:
        surf = st.selectbox("Surface", ["Hard","Clay","Grass","Carpet"])

    if p1 == p2:
        st.warning("Please select two different players!")
    else:
        if st.button("🎾 Predict Matchup"):
            with st.spinner("Running prediction..."):
                prob, p1h, p2h, h2h_tot, p1es, p2es, f1, f2 = predict_matchup(p1, p2, surf)
            prob2 = 1 - prob
            winner = p1 if prob >= 0.5 else p2
            wp = max(prob, prob2)*100

            st.markdown("---")
            st.markdown(f"### 🏆 Prediction: {winner} wins ({wp:.1f}%)")

            # Win probability bar
            p1_pct = int(prob*100); p2_pct = 100-p1_pct
            bar = f"""
            <div style="display:flex;height:50px;border-radius:25px;overflow:hidden;margin:1rem 0;box-shadow:0 4px 15px rgba(0,212,255,0.2);">
                <div style="width:{p1_pct}%;background:linear-gradient(90deg,#00d4ff,#0099bb);
                            display:flex;align-items:center;justify-content:center;
                            color:white;font-weight:800;font-size:1.1rem;">
                    {p1.split()[-1]} {p1_pct}%
                </div>
                <div style="width:{p2_pct}%;background:linear-gradient(90deg,#cc4444,#ff6b6b);
                            display:flex;align-items:center;justify-content:center;
                            color:white;font-weight:800;font-size:1.1rem;">
                    {p2_pct}% {p2.split()[-1]}
                </div>
            </div>"""
            st.markdown(bar, unsafe_allow_html=True)

            # Metrics row
            m1,m2,m3,m4,m5 = st.columns(5)
            with m1: st.markdown(f'<div class="metric-card"><div class="metric-value">{p1es:.0f}</div><div class="metric-label">{p1.split()[-1]} {surf} Elo</div></div>', unsafe_allow_html=True)
            with m2: st.markdown(f'<div class="metric-card"><div class="metric-value">{p1h}–{p2h}</div><div class="metric-label">H2H Record</div></div>', unsafe_allow_html=True)
            with m3: st.markdown(f'<div class="metric-card"><div class="metric-value">{h2h_tot}</div><div class="metric-label">Total H2H</div></div>', unsafe_allow_html=True)
            with m4: st.markdown(f'<div class="metric-card"><div class="metric-value">{p2es:.0f}</div><div class="metric-label">{p2.split()[-1]} {surf} Elo</div></div>', unsafe_allow_html=True)
            with m5: st.markdown(f'<div class="metric-card"><div class="metric-value">{surf}</div><div class="metric-label">Surface</div></div>', unsafe_allow_html=True)

            st.markdown("---")

            # Side by side radar
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown(f"#### {p1} Profile")
                s1 = get_player_stats(p1)
                fig, ax = plt.subplots(figsize=(5,5), subplot_kw={"projection":"polar"},
                                       facecolor="#0f0f1a")
                make_radar(ax, s1, "#00d4ff", p1)
                st.pyplot(fig, use_container_width=True); plt.close()

            with rc2:
                st.markdown(f"#### {p2} Profile")
                s2 = get_player_stats(p2)
                fig, ax = plt.subplots(figsize=(5,5), subplot_kw={"projection":"polar"},
                                       facecolor="#0f0f1a")
                make_radar(ax, s2, "#ff6b6b", p2)
                st.pyplot(fig, use_container_width=True); plt.close()

            # Key advantages
            st.markdown("#### 🔑 Key Advantages")
            adv = []
            r1 = get_player_row(p1); r2 = get_player_row(p2)
            surf_col = f"elo_{surf.lower()}"
            if r1.get(surf_col,1500) > r2.get(surf_col,1500)+50:
                adv.append((p1.split()[-1], f"Higher {surf} Elo ({r1.get(surf_col,1500):.0f} vs {r2.get(surf_col,1500):.0f})", "#00d4ff"))
            elif r2.get(surf_col,1500) > r1.get(surf_col,1500)+50:
                adv.append((p2.split()[-1], f"Higher {surf} Elo ({r2.get(surf_col,1500):.0f} vs {r1.get(surf_col,1500):.0f})", "#ff6b6b"))
            if p1h > p2h:
                adv.append((p1.split()[-1], f"H2H lead ({p1h}–{p2h})", "#00d4ff"))
            elif p2h > p1h:
                adv.append((p2.split()[-1], f"H2H lead ({p2h}–{p1h})", "#ff6b6b"))

            ac = st.columns(len(adv)) if adv else st.columns(1)
            for i, (player, advantage, col) in enumerate(adv):
                with ac[i]:
                    st.markdown(f'<div class="advantage-item" style="border-left-color:{col}">🏅 <b style="color:{col}">{player}</b>: {advantage}</div>',
                               unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: LEADERBOARD
# ─────────────────────────────────────────────
elif "Leaderboard" in page:
    st.markdown("## 🏆 All-Time Leaderboard")

    surf_filter = st.selectbox("Rank by", ["Overall","Hard","Clay","Grass"])
    col_name    = f"elo_{surf_filter.lower()}"
    match_col   = f"matches_{surf_filter.lower()}"
    min_matches = st.slider("Minimum matches on surface", 20, 200, 50)

    lb = (elo_df[elo_df[match_col] >= min_matches]
          [["player_name", "elo_overall","elo_hard","elo_clay","elo_grass", match_col]]
          .sort_values(col_name, ascending=False)
          .head(30)
          .reset_index(drop=True))
    lb.index += 1
    lb.columns = ["Player","Overall Elo","Hard Elo","Clay Elo","Grass Elo","Matches"]
    lb = lb.round(1)

    # Color the selected surface column
    def color_elo(val):
        if val > 1900: return "color: #FFD700; font-weight: bold"
        if val > 1800: return "color: #00d4ff"
        if val > 1700: return "color: #aaffaa"
        return "color: white"

    elo_col = f"{surf_filter} Elo"
    st.dataframe(
        lb.style.applymap(color_elo, subset=["Overall Elo","Hard Elo","Clay Elo","Grass Elo"]),
        use_container_width=True, height=600
    )

    # Bar chart of top 15
    st.markdown(f"#### Top 15 by {surf_filter} Elo")
    top15 = lb.head(15)
    fig, ax = plt.subplots(figsize=(12,5), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    colors = plt.cm.YlOrRd(np.linspace(0.4,1,15))[::-1]
    bars = ax.barh(top15["Player"][::-1],
                   top15[f"{surf_filter} Elo"][::-1],
                   color=colors)
    ax.axvline(1500, color="#555", ls="--", lw=0.8, label="Average")
    for bar, val in zip(bars, top15[f"{surf_filter} Elo"][::-1]):
        ax.text(bar.get_width()+5, bar.get_y()+bar.get_height()/2,
               f"{val:.0f}", va="center", fontsize=9, color="white")
    ax.set_xlim(min(top15[f"{surf_filter} Elo"])-100,
                max(top15[f"{surf_filter} Elo"])+100)
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values(): spine.set_color("#333")
    st.pyplot(fig, use_container_width=True)
    plt.close()