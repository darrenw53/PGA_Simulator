import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

def _is_player_like(d: Dict[str, Any]) -> bool:
    if not isinstance(d, dict):
        return False
    has_id = any(k in d for k in ("id", "player_id", "competitor_id", "uid")) or (
        isinstance(d.get("player"), dict) and any(k in d["player"] for k in ("id", "player_id", "uid"))
    )
    has_name = any(k in d for k in ("name", "full_name", "display_name")) or any(k in d for k in ("first_name", "last_name"))
    if isinstance(d.get("player"), dict):
        p = d["player"]
        has_name = has_name or any(k in p for k in ("name", "full_name", "display_name")) or any(k in p for k in ("first_name", "last_name"))
    return has_id and has_name

def _find_player_records(obj: Any, max_records: int = 5000) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []

    def walk(x: Any):
        nonlocal found
        if len(found) >= max_records:
            return
        if isinstance(x, dict):
            if _is_player_like(x):
                found.append(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for it in x:
                walk(it)

    walk(obj)
    return found

def _extract_id(rec: Dict[str, Any]) -> Optional[str]:
    for k in ("id", "player_id", "competitor_id", "uid"):
        if rec.get(k):
            return str(rec.get(k))
    if isinstance(rec.get("player"), dict):
        p = rec["player"]
        for k in ("id", "player_id", "competitor_id", "uid"):
            if p.get(k):
                return str(p.get(k))
    return None

def _extract_wgr_rank(rec: Dict[str, Any]) -> Optional[float]:
    for k in ("rank", "position", "world_rank", "wgr_rank"):
        if rec.get(k) is not None:
            try:
                return float(rec.get(k))
            except Exception:
                pass
    if isinstance(rec.get("ranking"), dict):
        r = rec["ranking"]
        for k in ("rank", "position", "world_rank"):
            if r.get(k) is not None:
                try:
                    return float(r.get(k))
                except Exception:
                    pass
    return None

def _extract_scoring_avg(rec: Dict[str, Any]) -> Optional[float]:
    for k in ("scoring_average", "scoring_avg", "avg_score", "score_avg"):
        if rec.get(k) is not None:
            try:
                return float(rec.get(k))
            except Exception:
                pass
    stats_list = rec.get("statistics") or rec.get("stats") or rec.get("categories")
    if isinstance(stats_list, list):
        for it in stats_list:
            if not isinstance(it, dict):
                continue
            label = str(it.get("name") or it.get("type") or it.get("category") or "").lower()
            if "scoring" in label and "average" in label:
                val = it.get("value") or it.get("avg") or it.get("number")
                try:
                    return float(val)
                except Exception:
                    continue
    return None

def build_player_table(stats_json: dict, wgr_json: dict, player_master: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Returns DataFrame with stable identity:
      id, name, wgr_rank, wgr_score, scoring_avg, base_strength

    If player_master is provided, it becomes the authoritative source for name/id mapping.
    """

    # --- Parse stats payload (id + scoring avg) ---
    stat_recs = _find_player_records(stats_json, max_records=8000)
    stats_rows: List[Dict[str, Any]] = []
    seen = set()

    for rec in stat_recs:
        pid = _extract_id(rec)
        if not pid or pid in seen:
            continue
        seen.add(pid)
        stats_rows.append({"id": pid, "scoring_avg": _extract_scoring_avg(rec)})

    stats_df = pd.DataFrame(stats_rows)
    if not stats_df.empty:
        stats_df["id"] = stats_df["id"].astype(str)
        stats_df["scoring_avg"] = pd.to_numeric(stats_df["scoring_avg"], errors="coerce")

    # --- Parse WGR payload (id + rank) ---
    wgr_recs = _find_player_records(wgr_json, max_records=8000)
    wgr_rows: List[Dict[str, Any]] = []
    seen = set()

    for rec in wgr_recs:
        pid = _extract_id(rec)
        if not pid or pid in seen:
            continue
        seen.add(pid)
        wgr_rows.append({"id": pid, "wgr_rank": _extract_wgr_rank(rec)})

    wgr_df = pd.DataFrame(wgr_rows)
    if not wgr_df.empty:
        wgr_df["id"] = wgr_df["id"].astype(str)
        wgr_df["wgr_rank"] = pd.to_numeric(wgr_df["wgr_rank"], errors="coerce")

    # --- Merge core performance tables ---
    if stats_df.empty and wgr_df.empty:
        base = pd.DataFrame(columns=["id", "scoring_avg", "wgr_rank"])
    elif stats_df.empty:
        base = wgr_df.copy()
        base["scoring_avg"] = pd.NA
    elif wgr_df.empty:
        base = stats_df.copy()
        base["wgr_rank"] = pd.NA
    else:
        base = stats_df.merge(wgr_df, on="id", how="outer")

    # --- Attach Player Master identity (authoritative names) ---
    if player_master is not None and not player_master.empty:
        pm = player_master.copy()
        pm["id"] = pm["id"].astype(str)
        if "name" not in pm.columns:
            pm["name"] = pm["id"]

        base = base.merge(pm[["id", "name"]], on="id", how="left")
    else:
        base["name"] = base["id"]

    # Ensure name is never blank
    base["name"] = (
        base["name"]
        .replace("", pd.NA)
        .fillna(base["id"])
        .astype(str)
    )

    # --- Derived features ---
    base["wgr_score"] = base["wgr_rank"].fillna(999).rpow(-0.25)

    if base["scoring_avg"].isna().all():
        base["scoring_avg"] = 70.5
    else:
        base["scoring_avg"] = base["scoring_avg"].fillna(base["scoring_avg"].median())

    base["base_strength"] = (70.5 - base["scoring_avg"]) + 2.0 * base["wgr_score"]

    base = base.sort_values("base_strength", ascending=False).reset_index(drop=True)
    return base
