import numpy as np
import pandas as pd

def simulate_tournament(
    players: pd.DataFrame,
    n_sims: int,
    tour_mean_round: float,
    base_sd_round: float,
    cut_top_n: int,
    hot_cold_corr: float,
    course_difficulty: float,
    field_strength_mult: float,
    volatility_mult: float,
    rng_seed: int | None = None
) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)

    # Player mean adjustment (better players score lower)
    strength = players["base_strength"].to_numpy()
    # Normalize strength so it doesn't explode
    strength = (strength - np.nanmean(strength)) / (np.nanstd(strength) + 1e-9)

    # Convert strength into strokes-gained-like effect
    # Positive strength => better => lower mean score
    player_mu = tour_mean_round - field_strength_mult * 0.6 * strength

    # Course difficulty shifts everyone
    player_mu = player_mu + course_difficulty

    sd = base_sd_round * volatility_mult

    ids = players["id"].fillna(players.index.astype(str)).to_numpy()
    names = players.get("name", pd.Series(ids)).fillna(ids).to_numpy()

    wins = np.zeros(len(players), dtype=int)
    top5 = np.zeros(len(players), dtype=int)
    made_cut = np.zeros(len(players), dtype=int)
    avg_finish = np.zeros(len(players), dtype=float)

    for _ in range(n_sims):
        # Round shocks with correlation (simple AR(1))
        shocks = np.zeros((len(players), 4))
        shocks[:, 0] = rng.normal(0, sd, size=len(players))
        for r in range(1, 4):
            shocks[:, r] = hot_cold_corr * shocks[:, r-1] + rng.normal(0, sd*np.sqrt(1-hot_cold_corr**2), size=len(players))

        scores = player_mu[:, None] + shocks  # (N,4)
        total2 = scores[:, :2].sum(axis=1)

        # Cut after 2 rounds
        cut_line_idx = np.argsort(total2)[:cut_top_n]
        in_cut = np.zeros(len(players), dtype=bool)
        in_cut[cut_line_idx] = True

        # Missed cut: set remaining rounds to NaN and treat total as big number
        total4 = np.full(len(players), 9999.0)
        total4[in_cut] = scores[in_cut].sum(axis=1)

        order = np.argsort(total4)  # winner first
        finish_pos = np.empty(len(players), dtype=int)
        finish_pos[order] = np.arange(1, len(players)+1)

        wins[order[0]] += 1
        top5[order[:5]] += 1
        made_cut[in_cut] += 1
        avg_finish += finish_pos

    out = pd.DataFrame({
        "player_id": ids,
        "player_name": names,
        "win_%": wins / n_sims * 100.0,
        "top5_%": top5 / n_sims * 100.0,
        "make_cut_%": made_cut / n_sims * 100.0,
        "avg_finish": avg_finish / n_sims
    }).sort_values(["win_%","top5_%","make_cut_%"], ascending=False)

    return out

