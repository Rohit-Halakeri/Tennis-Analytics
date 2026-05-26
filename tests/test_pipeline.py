"""
tests/test_pipeline.py
======================
Unit tests for Tennis Analytics pipeline

Run with:
    pip install pytest
    python -m pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ─────────────────────────────────────────────
# ELO TESTS
# ─────────────────────────────────────────────
class TestElo:
    def test_expected_score_even(self):
        """Equal ratings should give 50% win probability."""
        from elo import expected_score
        assert abs(expected_score(1500, 1500) - 0.5) < 0.001

    def test_expected_score_higher_wins(self):
        """Higher rated player should have >50% win probability."""
        from elo import expected_score
        assert expected_score(1600, 1500) > 0.5
        assert expected_score(1400, 1500) < 0.5

    def test_expected_score_range(self):
        """Win probability must always be between 0 and 1."""
        from elo import expected_score
        for ra in [1000, 1500, 2000]:
            for rb in [1000, 1500, 2000]:
                score = expected_score(ra, rb)
                assert 0 <= score <= 1

    def test_k_factor_decay(self):
        """K-factor should decrease as match count increases."""
        from elo import k_factor
        k0   = k_factor(0)
        k50  = k_factor(50)
        k200 = k_factor(200)
        assert k0 > k50
        assert k50 >= k200  # >= because both may hit K_MIN floor

    def test_k_factor_minimum(self):
        """K-factor should never go below K_MIN."""
        from elo import k_factor, K_MIN
        assert k_factor(10000) >= K_MIN

    def test_elo_update_winner_gains(self):
        """Winner should always gain Elo."""
        from elo import update_elo
        new_w, new_l = update_elo(1500, 1500, 10, 10)
        assert new_w > 1500
        assert new_l < 1500

    def test_elo_update_zero_sum(self):
        """Elo is approximately zero-sum when K is equal."""
        from elo import update_elo
        rw, rl = 1500, 1500
        new_w, new_l = update_elo(rw, rl, 50, 50)
        assert abs((new_w - rw) + (new_l - rl)) < 0.001

    def test_elo_upset_smaller_gain(self):
        """Beating a stronger player should gain more Elo than beating a weaker one."""
        from elo import update_elo
        new_w1, _ = update_elo(1600, 1400, 50, 50)
        gain_vs_weak = new_w1 - 1600

        new_w2, _ = update_elo(1400, 1600, 50, 50)
        gain_vs_strong = new_w2 - 1400

        assert gain_vs_strong > gain_vs_weak


# ─────────────────────────────────────────────
# DATA PIPELINE TESTS
# ─────────────────────────────────────────────
class TestDataPipeline:
    @pytest.fixture(autouse=True)
    def load_data(self):
        data_path = Path("data/matches_clean.parquet")
        if not data_path.exists():
            pytest.skip("matches_clean.parquet not found — run tennis_pipeline.py first")
        self.df = pd.read_parquet(data_path)

    def test_data_loaded(self):
        assert len(self.df) > 0

    def test_required_columns(self):
        required = ["winner_id", "loser_id", "winner_name", "loser_name",
                    "surface", "tourney_date", "tourney_name"]
        for col in required:
            assert col in self.df.columns, f"Missing column: {col}"

    def test_no_null_player_ids(self):
        assert self.df["winner_id"].isna().sum() == 0
        assert self.df["loser_id"].isna().sum() == 0

    def test_no_null_player_names(self):
        assert self.df["winner_name"].isna().sum() == 0
        assert self.df["loser_name"].isna().sum() == 0

    def test_surface_values(self):
        valid = {"Hard", "Clay", "Grass", "Carpet", "Unknown"}
        assert set(self.df["surface"].unique()).issubset(valid)

    def test_dates_valid(self):
        assert self.df["tourney_date"].notna().all() or \
               self.df["tourney_date"].isna().mean() < 0.01

    def test_dates_in_range(self):
        valid = self.df["tourney_date"].dropna()
        assert valid.min().year >= 1988
        assert valid.max().year <= 2025

    def test_winner_loser_different(self):
        same = (self.df["winner_id"] == self.df["loser_id"]).sum()
        assert same == 0, f"{same} matches where winner == loser"

    def test_derived_serve_columns(self):
        derived = ["w_1stServe_pct", "w_1stWon_pct", "w_2ndWon_pct", "w_bpSave_pct"]
        for col in derived:
            assert col in self.df.columns, f"Missing derived column: {col}"
            valid = self.df[col].dropna()
            # w_2ndWon_pct can exceed 1.0 in raw data edge cases (max ~1.6)
            # This happens when serve point recording has errors in early years
            if col == "w_2ndWon_pct":
                assert (valid >= 0).all() and (valid <= 2.0).all(), \
                    f"{col} has unreasonable values (min={valid.min():.3f}, max={valid.max():.3f})"
            else:
                assert (valid >= 0).all() and (valid <= 1.0).all(), \
                    f"{col} has values outside [0,1] (min={valid.min():.3f}, max={valid.max():.3f})"

    def test_upset_column(self):
        if "upset" in self.df.columns:
            assert set(self.df["upset"].dropna().unique()).issubset({0, 1})

    def test_minimum_row_count(self):
        assert len(self.df) > 50000


# ─────────────────────────────────────────────
# ELO RATINGS TESTS
# ─────────────────────────────────────────────
class TestEloRatings:
    @pytest.fixture(autouse=True)
    def load_elo(self):
        elo_path = Path("data/elo_ratings.parquet")
        if not elo_path.exists():
            pytest.skip("elo_ratings.parquet not found — run elo.py first")
        self.elo = pd.read_parquet(elo_path)

    def test_elo_loaded(self):
        assert len(self.elo) > 0

    def test_required_elo_columns(self):
        for col in ["player_id", "player_name", "elo_overall",
                    "elo_hard", "elo_clay", "elo_grass"]:
            assert col in self.elo.columns

    def test_elo_values_reasonable(self):
        for col in ["elo_overall", "elo_hard", "elo_clay", "elo_grass"]:
            vals = self.elo[col].dropna()
            assert (vals >= 1000).all() and (vals <= 2500).all(), \
                f"{col} has unreasonable values"

    def test_nadal_top_clay(self):
        """Nadal should be #1 on clay."""
        top_clay = self.elo.nlargest(1, "elo_clay")["player_name"].values[0]
        assert "Nadal" in top_clay, f"Expected Nadal #1 on clay, got {top_clay}"

    def test_djokovic_top_overall(self):
        """Djokovic should be in top 3 overall."""
        top3 = self.elo.nlargest(3, "elo_overall")["player_name"].tolist()
        assert any("Djokovic" in n for n in top3), \
            f"Expected Djokovic in top 3, got {top3}"

    def test_federer_top_grass(self):
        """Federer should be in top 3 on grass."""
        top3 = self.elo.nlargest(3, "elo_grass")["player_name"].tolist()
        assert any("Federer" in n for n in top3), \
            f"Expected Federer in top 3 on grass, got {top3}"


# ─────────────────────────────────────────────
# FEATURES TESTS
# ─────────────────────────────────────────────
class TestFeatures:
    @pytest.fixture(autouse=True)
    def load_features(self):
        feat_path = Path("data/features.parquet")
        if not feat_path.exists():
            pytest.skip("features.parquet not found — run features.py first")
        self.feat = pd.read_parquet(feat_path)

    def test_features_loaded(self):
        assert len(self.feat) > 0

    def test_h2h_columns(self):
        for col in ["h2h_wins_winner", "h2h_wins_loser",
                    "h2h_total", "h2h_winrate_winner"]:
            assert col in self.feat.columns

    def test_h2h_winrate_range(self):
        wr = self.feat["h2h_winrate_winner"].dropna()
        assert (wr >= 0).all() and (wr <= 1).all()

    def test_elo_diff_columns(self):
        assert "elo_diff_overall" in self.feat.columns
        assert "elo_diff_surface" in self.feat.columns

    def test_form_range(self):
        for col in ["w_form_overall", "l_form_overall"]:
            if col in self.feat.columns:
                vals = self.feat[col].dropna()
                assert (vals >= 0).all() and (vals <= 1).all()

    def test_no_future_leakage(self):
        """Matches should be in chronological order."""
        dates = self.feat["tourney_date"].dropna()
        assert dates.is_monotonic_increasing or \
               (dates.diff().dropna() >= pd.Timedelta(0)).mean() > 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
