# 🎾 Tennis Analytics — Matchup Prediction & Player Profiling

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33-red?style=for-the-badge&logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-2.2-150458?style=for-the-badge&logo=pandas)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Data](https://img.shields.io/badge/Matches-108%2C573-brightgreen?style=for-the-badge)

**35 years of ATP match data · Surface-specific Elo ratings · XGBoost matchup prediction · Interactive Streamlit dashboard**

[Features](#-features) · [Demo](#-demo) · [Quick Start](#-quick-start) · [How It Works](#-how-it-works) · [Results](#-results) · [Future Work](#-future-work)

</div>

---

## 🎯 Project Overview

A complete end-to-end data analytics project that ingests 35 years of ATP tennis data (1990–2024), engineers a rich feature set, trains a machine learning model to predict match outcomes, and serves results through an interactive dashboard.

Given any two players and a surface, the system:
- Predicts win probability using a trained XGBoost model
- Shows surface-specific Elo ratings for both players
- Displays head-to-head historical record
- Highlights key advantages and weaknesses
- Profiles each player with radar charts and career timelines

---

## ✨ Features

### 📊 Data Pipeline
- Auto-downloads 35 years of ATP match CSVs from Jeff Sackmann's dataset
- Cleans, merges, and stores 108,573 matches as fast Parquet files
- Handles retirements, walkovers, and encoding issues automatically
- Fully reproducible — anyone can clone and regenerate data in ~10 minutes

### ⚡ Elo Rating Engine
- Computes **surface-specific Elo** (Hard / Clay / Grass / Carpet / Overall) for every player
- Dynamic K-factor that decays with match experience
- Processes 108,573 matches in **under 10 seconds**
- Validated against known tennis reality (Nadal #1 clay, Djokovic #1 hard)

### 🧠 Feature Engineering (50+ features)
- Rolling 20-match form windows (overall + per surface)
- Head-to-head records (no data leakage — only past matches)
- Serve dominance metrics: 1st serve %, break point save %, ace rate
- Elo differentials per surface
- Tournament level and surface encoding

### 🤖 XGBoost Prediction Model
- Walk-forward cross-validation (5 folds) — no future data leakage
- ~67% accuracy on held-out test sets
- Calibrated win probabilities (not just binary predictions)
- SHAP-ready feature importance analysis

### 👤 Player Profiling
- 6-panel dark-themed profile card per player
- Radar chart across 8 dimensions
- Career Elo timeline with peak marker
- Win rate by surface and tournament level
- Auto-detected strengths and weaknesses

### 🖥️ Interactive Streamlit Dashboard
- **Home** — quick matchup predictor + Big 3 comparison
- **Player Profile** — search any player, full visual breakdown
- **Matchup Predictor** — pick 2 players + surface → win probability
- **Leaderboard** — top players by any surface Elo

---

## 📸 Demo

### Player Profile — Novak Djokovic
![Djokovic Profile](screenshots/djokovic_profile.png)

### Player Profile — Rafael Nadal
![Nadal Profile](screenshots/nadal_profile.png)

### Player Profile — Roger Federer
![Federer Profile](screenshots/federer_profile.png)

### Matchup Predictor Dashboard
![Matchup](screenshots/matchup_dashboard.png)

### Leaderboard
![Leaderboard](screenshots/leaderboard.png)

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/Rohit-Halakeri/Tennis-Analytics.git
cd Tennis-Analytics
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download & clean data (~10 min, runs once)
```bash
python src/tennis_pipeline.py
```

### 5. Build Elo ratings (~10 seconds)
```bash
python src/elo.py
```

### 6. Engineer features (~35 seconds)
```bash
python src/features.py
```

### 7. Train prediction model (~2 min)
```bash
python src/model.py
```

### 8. Launch dashboard 🎉
```bash
streamlit run app/dashboard.py
```

Open `http://localhost:8501` in your browser.

---

## 🔬 How It Works

### Data Flow
```
Jeff Sackmann GitHub (60+ CSVs)
        ↓
tennis_pipeline.py  →  matches_clean.parquet (108k rows)
        ↓
elo.py              →  elo_ratings.parquet + matches_with_elo.parquet
        ↓
features.py         →  features.parquet (50+ features, no leakage)
        ↓
model.py            →  xgb_model.json (~67% accuracy)
        ↓
dashboard.py        →  Streamlit app (localhost:8501)
```

### Elo Rating System
Each player has 5 separate Elo ratings (Overall, Hard, Clay, Grass, Carpet).
Ratings update after every match using:

```
E(A) = 1 / (1 + 10^((R_B - R_A) / 400))
K    = max(10, 32 × e^(-0.05 × matches_played))
R_A' = R_A + K × (1 - E(A))
```

The dynamic K-factor means new players update faster, experienced players update slower — reflecting real-world uncertainty.

### Walk-Forward Validation
To prevent data leakage, the model is evaluated using time-series cross-validation:
- Always train on past matches, test on future matches
- 5 folds covering the full 1990–2024 range
- Final model trained on all data

### Feature Engineering (no leakage)
Every feature is computed from matches **before** the current one:
- Rolling form = win rate over last 20 matches **prior** to this match
- H2H record = head-to-head results **before** this meeting
- Serve stats = rolling averages from **previous** matches only

---

## 📈 Results

### Model Performance
| Metric | Score |
|--------|-------|
| Accuracy (CV avg) | ~67% |
| AUC (CV avg) | ~0.73 |
| Brier Score | ~0.22 |

### Elo Validation (matches tennis reality ✅)
| Surface | #1 Player | Elo |
|---------|-----------|-----|
| Overall | Novak Djokovic | 2017 |
| Hard | Novak Djokovic | 1999 |
| Clay | Rafael Nadal | 1967 |
| Grass | Roger Federer | 1829 |

### Sample Predictions
| Matchup | Surface | Prediction |
|---------|---------|------------|
| Djokovic vs Nadal | Clay | Djokovic 72% |
| Djokovic vs Nadal | Hard | Djokovic ~70% |
| Federer vs Nadal | Grass | Federer 87% |
| Sinner vs Alcaraz | Hard | Sinner 59% |

---

## 📁 Project Structure

```
Tennis-Analytics/
├── data/                          # Data files (gitignored, regeneratable)
│   ├── raw/                       # Individual year CSVs
│   ├── matches_clean.parquet      # 108k cleaned matches
│   ├── matches_with_elo.parquet   # + Elo columns
│   ├── elo_ratings.parquet        # Final Elo per player
│   ├── features.parquet           # 50+ engineered features
│   └── pipeline_report.txt        # Data quality report
├── src/                           # Core pipeline scripts
│   ├── tennis_pipeline.py         # Download + clean data
│   ├── elo.py                     # Elo rating engine
│   ├── features.py                # Feature engineering
│   ├── player_profile.py          # Player profiling + charts
│   ├── model.py                   # XGBoost training + evaluation
│   └── matchup.py                 # CLI matchup predictor
├── app/
│   └── dashboard.py               # Streamlit dashboard
├── models/                        # Trained model artifacts
│   ├── xgb_model.json             # Trained XGBoost model
│   └── feature_names.txt          # Feature list
├── reports/                       # Generated visualizations
│   ├── *_profile.png              # Player profile cards
│   ├── *_vs_*_matchup.png         # Matchup cards
│   └── model_performance.png      # CV results + feature importance
├── screenshots/                   # Dashboard screenshots for README
├── notebooks/                     # EDA and analysis notebooks
├── tests/                         # Unit tests
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

---

## 🔮 Future Improvements

### Data
- [ ] Add WTA (women's) data for cross-gender analysis
- [ ] Include player physical attributes (height, reach, nationality)
- [ ] Scrape live ATP rankings and recent match results
- [ ] Add tournament draw data (bracket position, travel fatigue)
- [ ] Include weather/conditions data for outdoor matches

### Modelling
- [ ] Add neural network model (TabNet or MLP) for comparison
- [ ] Build set-level and game-level prediction (not just match)
- [ ] Train separate models per surface for better specialization
- [ ] Add confidence intervals to win probability predictions
- [ ] Implement proper Bayesian Elo with uncertainty estimation
- [ ] Add SHAP waterfall plots per prediction for full explainability

### Features
- [ ] Fatigue modelling (days since last match, travel distance)
- [ ] Tournament pressure index (early round vs final)
- [ ] Clutch score (performance in tiebreaks, 5th sets)
- [ ] Momentum metric (winning streak, recent form trend)
- [ ] Serve/return dominance index (combined metric)
- [ ] Playing style clustering (serve-and-volley vs baseliner)

### Dashboard
- [ ] Deploy to Streamlit Cloud (free hosting, shareable link)
- [ ] Add tournament bracket simulator
- [ ] Add Grand Slam draw prediction tool
- [ ] Player comparison side-by-side (not just matchup)
- [ ] Historical match search and filter
- [ ] Mobile-responsive layout improvements

### Engineering
- [ ] Add unit tests for Elo engine and feature pipeline
- [ ] Automate weekly data refresh with GitHub Actions
- [ ] Add DVC for full data version control
- [ ] Package as a proper Python library (pip installable)
- [ ] Add API endpoint (FastAPI) for programmatic predictions

---

## 📚 Data Source

All match data sourced from **Jeff Sackmann's ATP Tennis Dataset**:
- Repository: [github.com/JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp)
- Coverage: ATP matches 1968–2024 (this project uses 1990–2024)
- ~108,573 matches after cleaning

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| Pandas + PyArrow | Data processing + Parquet storage |
| XGBoost | Match prediction model |
| Scikit-learn | Model evaluation + cross-validation |
| Matplotlib | Player profile + matchup visualizations |
| Streamlit | Interactive web dashboard |
| Git + GitHub | Version control |

---

## 👨‍💻 Author

**Rohit Halakeri**
- GitHub: [@Rohit-Halakeri](https://github.com/Rohit-Halakeri)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Built with ❤️ and way too much tennis data 🎾
<br>
If you found this useful, please ⭐ the repo!
</div>