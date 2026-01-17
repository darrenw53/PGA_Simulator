from __future__ import annotations

import numpy as np
import pandas as pd


def make_course_fit_weights() -> dict:
    # Sensible defaults (operator can override in UI)
    return {
        "sg_total": 1.00,
        "sg_t2g": 0.75,
        "sg_putt_proxy": 0.35,
        "birdies_per_round": 0.40,
        "gir_pct": 0.20,
        "drive_avg": 0.15,
        "drive_acc": 0.10,
        "scrambling_pct": 0.20,
    }


def build_model_table(
    fanduel: pd.DataFrame,
    stats: pd.DataFrame,
    wgr: pd.DataFrame,
) -> pd.DataFrame:
    """
    Joins:
      FanDuel field (names + salary + FPPG)
      SportsRadar player stats (player_id + stats)
      WGR (player_id + rank)
    Matching strategy:
      1) exact join on name (fd_name vs first+last from stats)
      2) if you later add a master mapping file, we can do ID matching.
    """

    fd = fanduel.copy()
    st = stats.copy()
    wg = wgr.copy()

    st["name"] = (st["first_name"].fillna("").astype(str).str.strip() + " " + st["last_name"].fillna("").astype(str).str.strip()).str.strip()

    # Name join
    merged = fd.merge(
        st,
        left_on="fd_name",
        right_on="name",
        how="left",
        suffixes=("", "_stats"),
    )

    # Some FanDuel entries might not match; keep only matched rows
    merged = merged.dropna(subset=["player_id"]).copy()

    merged = merged.merge(wg[["player_id", "wgr_rank"]], on="player_id", how="left")

    # Clean numeric columns
    for c in [
        "Salary", "FPPG", "wgr_rank",
        "scoring_avg", "birdies_per_round", "gir_pct", "scrambling_pct",
        "drive_avg", "drive_acc",
        "strokes_gained", "strokes_gained_total", "strokes_gained_tee_green"
    ]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    # Fallbacks
    merged["wgr_rank"] = merged["wgr_rank"].fillna(999.0)
    merged["FPPG"] = merged["FPPG"].fillna(0.0)

    # If scoring_avg missing, approximate from field average
    if merged["scoring_avg"].notna().any():
        avg = float(merged["scoring_avg"].dropna().mean())
        merged["scoring_avg"] = merged["scoring_avg"].fillna(avg)
    else:
        merged["scoring_avg"] = 71.5

    return merged


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return s * 0
    return (s - mu) / sd
