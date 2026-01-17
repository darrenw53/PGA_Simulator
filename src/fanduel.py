from __future__ import annotations

import itertools
import pandas as pd


def load_fanduel_csv(path_or_df) -> pd.DataFrame:
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        df = pd.read_csv(path_or_df)

    # Expect at least: Id, First Name, Last Name, Salary, FPPG
    needed = {"Id", "First Name", "Last Name", "Salary", "FPPG"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing FanDuel columns: {sorted(missing)}")

    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
    df["FPPG"] = pd.to_numeric(df["FPPG"], errors="coerce")
    df = df.dropna(subset=["Salary"]).copy()
    return df


def optimize_fanduel_lineup(
    sim_results: pd.DataFrame,
    salary_cap: int = 60000,
    lineup_size: int = 6,
    candidate_pool: int = 30,
    lock_names: set[str] | None = None,
    exclude_names: set[str] | None = None,
):
    """
    Brute-force over a reduced candidate pool:
      - Always include top by proj_fd_points and top by value (points per salary)
      - Then search combinations for the best projected points under salary cap
    """
    lock_names = lock_names or set()
    exclude_names = exclude_names or set()

    df = sim_results.copy()
    df = df[~df["name"].isin(exclude_names)].copy()

    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
    df["proj_fd_points"] = pd.to_numeric(df["proj_fd_points"], errors="coerce")

    df = df.dropna(subset=["Salary", "proj_fd_points"]).copy()
    df["value"] = df["proj_fd_points"] / (df["Salary"].clip(lower=1) / 1000.0)

    # Locked players must exist
    locked = df[df["name"].isin(lock_names)].copy()
    if len(locked) > lineup_size:
        return None, None

    salary_locked = int(locked["Salary"].sum()) if not locked.empty else 0
    if salary_locked > salary_cap:
        return None, None

    remaining_slots = lineup_size - len(locked)
    remaining_cap = salary_cap - salary_locked

    # Candidate base:
    top_points = df.sort_values("proj_fd_points", ascending=False).head(candidate_pool)
    top_value = df.sort_values("value", ascending=False).head(candidate_pool)
    candidates = pd.concat([top_points, top_value]).drop_duplicates(subset=["player_id"]).copy()

    # Ensure locked included
    candidates = pd.concat([candidates, locked]).drop_duplicates(subset=["player_id"]).copy()

    # Remove locked from the "choose" pool so we don't double-count
    choose_pool = candidates[~candidates["name"].isin(lock_names)].copy()

    # If small, just use all
    choose_pool = choose_pool.sort_values("proj_fd_points", ascending=False).head(max(candidate_pool, 18))

    best_points = -1e9
    best_lineup = None

    choose_rows = choose_pool.to_dict("records")

    # Brute force on remaining slots
    for combo in itertools.combinations(choose_rows, remaining_slots):
        combo_df = pd.DataFrame(combo)
        total_salary = int(combo_df["Salary"].sum()) + salary_locked
        if total_salary > salary_cap:
            continue
        total_points = float(combo_df["proj_fd_points"].sum()) + float(locked["proj_fd_points"].sum() if not locked.empty else 0.0)
        if total_points > best_points:
            best_points = total_points
            best_lineup = pd.concat([locked, combo_df], ignore_index=True)

    if best_lineup is None:
        return None, None

    best_lineup = best_lineup.sort_values("proj_fd_points", ascending=False).reset_index(drop=True)
    meta = {
        "total_salary": int(best_lineup["Salary"].sum()),
        "total_points": float(best_lineup["proj_fd_points"].sum()),
    }
    return best_lineup, meta

