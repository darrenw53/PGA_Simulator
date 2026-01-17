from __future__ import annotations

import itertools
from bisect import bisect_right
from dataclasses import dataclass
from typing import Optional, Tuple, Set, List

import numpy as np
import pandas as pd


@dataclass
class LineupMeta:
    total_salary: int
    total_points: float
    n_candidates: int
    remaining_slots: int


def _prep_pool(sim_results: pd.DataFrame) -> pd.DataFrame:
    df = sim_results.copy()

    # Ensure required cols
    if "name" not in df.columns or "Salary" not in df.columns or "proj_fd_points" not in df.columns:
        raise ValueError("sim_results must include columns: name, Salary, proj_fd_points")

    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
    df["proj_fd_points"] = pd.to_numeric(df["proj_fd_points"], errors="coerce")

    # Drop bad rows
    df = df.dropna(subset=["Salary", "proj_fd_points"]).copy()

    # Salary must be positive integer-ish
    df = df[df["Salary"] > 0].copy()
    df["Salary"] = df["Salary"].astype(int)

    # Use player_id if present, else fallback to name for uniqueness
    key = "player_id" if "player_id" in df.columns else "name"
    df = df.drop_duplicates(subset=[key]).copy()

    return df.reset_index(drop=True)


def _best_under_cap_mim(
    pool: pd.DataFrame,
    cap: int,
    k: int,
) -> Optional[Tuple[List[int], int, float]]:
    """
    Meet-in-the-middle optimizer:
    choose k players from pool maximizing points under salary cap.
    Returns (indices, total_salary, total_points) or None.
    """
    if k == 0:
        return ([], 0, 0.0)

    n = len(pool)
    if n < k:
        return None

    salaries = pool["Salary"].to_numpy(dtype=int)
    points = pool["proj_fd_points"].to_numpy(dtype=float)

    # Split k into a + b
    a = k // 2
    b = k - a

    idxs = list(range(n))

    # Precompute all combos of size a
    combos_a = []
    for comb in itertools.combinations(idxs, a):
        s = int(salaries[list(comb)].sum())
        if s <= cap:
            p = float(points[list(comb)].sum())
            combos_a.append((s, p, comb))

    # Precompute all combos of size b
    combos_b = []
    for comb in itertools.combinations(idxs, b):
        s = int(salaries[list(comb)].sum())
        if s <= cap:
            p = float(points[list(comb)].sum())
            combos_b.append((s, p, comb))

    if not combos_a or not combos_b:
        return None

    # Sort combos_b by salary; for each salary keep best points so far (prefix max)
    combos_b.sort(key=lambda x: x[0])
    b_salaries = [c[0] for c in combos_b]
    b_best_points = []
    b_best_combo = []

    best_p = -1e18
    best_c = None
    for s, p, comb in combos_b:
        if p > best_p:
            best_p = p
            best_c = (s, p, comb)
        b_best_points.append(best_p)
        b_best_combo.append(best_c)

    # Try each a-combo, pick best b-combo under remaining cap,
    # while ensuring no duplicate players (disjoint sets).
    best_total_p = -1e18
    best_total_s = None
    best_total_idxs = None

    for s_a, p_a, comb_a in combos_a:
        rem = cap - s_a
        j = bisect_right(b_salaries, rem) - 1
        if j < 0:
            continue

        # candidate best b under salary rem (not necessarily disjoint)
        # We may need to step back to find a disjoint one.
        # In practice, conflicts are rare; a small backward scan is fast.
        set_a = set(comb_a)

        jj = j
        while jj >= 0:
            s_b, p_b, comb_b = b_best_combo[jj]
            if comb_b is None:
                jj -= 1
                continue
            if set_a.isdisjoint(comb_b):
                total_s = s_a + s_b
                total_p = p_a + p_b
                if total_p > best_total_p:
                    best_total_p = total_p
                    best_total_s = total_s
                    best_total_idxs = list(comb_a) + list(comb_b)
                break
            jj -= 1

    if best_total_idxs is None:
        return None

    return (best_total_idxs, int(best_total_s), float(best_total_p))


def optimize_fanduel_lineup(
    sim_results: pd.DataFrame,
    salary_cap: int = 60000,
    lineup_size: int = 6,
    candidate_pool: int = 40,
    lock_names: Optional[Set[str]] = None,
    exclude_names: Optional[Set[str]] = None,
):
    """
    Fast lineup optimizer:
      - Build candidate pool by proj_fd_points + value
      - Apply lock/exclude
      - Solve remaining slots using meet-in-the-middle

    Returns: (lineup_df, meta_dict) or (None, None)
    """
    lock_names = lock_names or set()
    exclude_names = exclude_names or set()

    df = _prep_pool(sim_results)

    # Apply excludes
    df = df[~df["name"].isin(exclude_names)].copy()
    if df.empty:
        return None, None

    # Locked
    locked = df[df["name"].isin(lock_names)].copy()
    if len(locked) > lineup_size:
        return None, None

    locked_salary = int(locked["Salary"].sum()) if not locked.empty else 0
    locked_points = float(locked["proj_fd_points"].sum()) if not locked.empty else 0.0

    if locked_salary > salary_cap:
        return None, None

    remaining_slots = lineup_size - len(locked)
    remaining_cap = salary_cap - locked_salary

    # Build candidate pool (points + value)
    df["value"] = df["proj_fd_points"] / (df["Salary"].clip(lower=1) / 1000.0)

    top_points = df.sort_values("proj_fd_points", ascending=False).head(candidate_pool)
    top_value = df.sort_values("value", ascending=False).head(candidate_pool)

    candidates = pd.concat([top_points, top_value, locked], ignore_index=True).drop_duplicates(
        subset=["player_id"] if "player_id" in df.columns else ["name"]
    )

    # Remove locked from the pool we choose from
    choose_pool = candidates[~candidates["name"].isin(lock_names)].copy()
    choose_pool = choose_pool.sort_values("proj_fd_points", ascending=False).reset_index(drop=True)

    if remaining_slots == 0:
        lineup = locked.copy().sort_values("proj_fd_points", ascending=False).reset_index(drop=True)
        meta = LineupMeta(
            total_salary=locked_salary,
            total_points=locked_points,
            n_candidates=len(candidates),
            remaining_slots=0,
        )
        return lineup, meta.__dict__

    # Quick impossibility checks:
    # cheapest possible remaining slots salary must fit
    cheapest = choose_pool["Salary"].nsmallest(remaining_slots).sum()
    if int(cheapest) > remaining_cap:
        return None, None

    # Solve remaining slots under remaining cap
    sol = _best_under_cap_mim(choose_pool, remaining_cap, remaining_slots)
    if sol is None:
        return None, None

    idxs, s_sel, p_sel = sol
    picked = choose_pool.iloc[idxs].copy()

    lineup = pd.concat([locked, picked], ignore_index=True)
    lineup = lineup.sort_values("proj_fd_points", ascending=False).reset_index(drop=True)

    meta = LineupMeta(
        total_salary=int(lineup["Salary"].sum()),
        total_points=float(lineup["proj_fd_points"].sum()),
        n_candidates=len(candidates),
        remaining_slots=remaining_slots,
    )
    return lineup, meta.__dict__
