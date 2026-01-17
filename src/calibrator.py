import numpy as np
import pandas as pd

def calibrate_global(sim_df: pd.DataFrame, actual_df: pd.DataFrame) -> dict:
    """
    actual_df expects columns: player_id, actual_finish (1=winner)
    sim_df expects: player_id, avg_finish, win_%
    """
    m = sim_df.merge(actual_df, on="player_id", how="inner")
    if m.empty:
        return {"note": "No overlap between sim and actual."}

    # Did we systematically under/over-rate favorites?
    # Simple diagnostic: correlation between predicted avg_finish and actual_finish
    corr = float(m["avg_finish"].corr(m["actual_finish"]))
    # Smaller is better for avg_finish; so corr should be positive
    # We'll also compute mean error
    err = float((m["avg_finish"] - m["actual_finish"]).mean())

    # Return calibration suggestions (you can turn these into automatic tweaks)
    return {
        "overlap_n": int(len(m)),
        "pred_vs_actual_finish_corr": corr,
        "mean_finish_error": err
    }

