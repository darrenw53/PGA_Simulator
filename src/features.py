import pandas as pd
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Helpers: robust JSON searching
# -----------------------------
def _is_player_like(d: Dict[str, Any]) -> bool:
    """Heuristic: dict contains an id AND at least one name-ish field OR has nested 'player' with id."""
    if not isinstance(d, dict):
        return False

    # direct id
    has_id = any(k in d for k in ("id", "player_id", "competitor_id", "uid"))
    # nested player object
    has_player_obj = isinstance(d.get("player"), dict) and any(k in d["player"] for k in ("id", "player_id", "uid"))

    # name-ish
    has_name = any(k in d for k in ("name", "full_name", "display_name")) or (
        any(k in d for k in ("first_name", "last_name"))
    )
    if isinstance(d.get("player"), dict):
        p = d["player"]
        has_name = has_name or any(k in p for k in ("name", "full_name", "display_name")) or (
            any(k in p for k in ("first_name", "last_name"))
        )

    return (has_id or has_player_obj) and has_name


def _find_player_records(obj: Any, max_records: int = 5000) -> List[Dict[str, Any]]:
    """
    Recursively walk JSON and return a list of dict records that look like player rows.
    We stop early if we find a large enough list to avoid deep scans on huge payloads.
    """
    found: List[Dict[str, Any]] = []

    def walk(x: Any):
        nonlocal found
        if len(found) >= max_records:
            return

        if isinstance(x, dict):
            # If this dict itself is player-like, include it
            if _is_player_like(x):
                found.append(x)

            # Walk values
            for v in x.values():
                walk(v)

        elif isinstance(x, list):
            # If list items are dicts, check each
            for it in x:
                walk(it)

    walk(obj)
    return found


def _extract_id_and_name(rec: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Try multiple locations/field names for id + name."""
    pid = None
    name = None

    # direct id
    for k in ("id", "player_id", "competitor_id", "uid"):
        if rec.get(k):
            pid = str(rec.get(k))
            break

    # nested player id
    if pid is None and isinstance(rec.get("player"), dict):
        p = rec["player"]
        for k in ("id", "player_id", "competitor_id", "uid"):
            if p.get(k):
                pid = str(p.get(k))
                break

    # name priority
    for k in ("name", "full_name", "display_name"):
        if rec.get(k):
            name = str(rec.get(k))
            break

    # build from first/last
    if name is None:
        fn = rec.get("first_name")
        ln = rec.get("last_name")
        if fn or ln:
            name = f"{fn or ''} {ln or ''}".strip()

    # nested player name
    if name is None and isinstance(rec.get("player"), dict):
        p = rec["player"]
        for k in ("name", "full_name", "display_name"):
            if p.get(k):
                name = str(p.get(k))
                break
        if name is None:
            fn = p.get("first_name")
            ln = p.get("last_name")
            if fn or ln:
                name = f"{fn or ''} {ln or ''}".strip()

    return pid, name


def _extract_wgr_rank(rec: Dict[str, Any]) -> Optional[float]:
    """Find rank fields in common shapes."""
    # direct rank-like fields
    for k in ("rank", "position", "world_rank", "wgr_rank"):
        if rec.get(k) is not None:
            try:
                return float(rec.get(k))
            except Exception:
                pass

    # nested
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
    """
    Try to find scoring average in multiple possible representations.
    SportsRadar stats payloads can vary.
    """
    # direct common fields
    for k in ("scoring_average", "scoring_avg", "avg_score", "score_avg"):
        if rec.get(k) is not None:
            try:
                return float(rec.get(k))
            except Exception:
                pass

    # Sometimes stats are a list of key/value pairs
    stats_list = rec.get("statistics") or rec.get("stats") or rec.get("categories")
    if isinstance(stats_list, list):
        # look for scoring avg-ish entries
        for it in stats_list:
            if not isinstance(it, dict):
                continue
            # fields might be: {"name":"Scoring Average","value":"70.12"} etc
            label = str(it.get("name") or it.get("type") or it.get("category") or "").lower()
            if "scoring" in label and "average" in label:
                val = it.get("value") or it.get("avg") or it.get("number")
                try:
                    return float(val)
                except Exception:
                    continue

    return None


# -----------------------------
# Main: build player table
# -----------------------------
def build_player_table(stats_json: dict, wgr_json: dict) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      id, name, wgr_rank, wgr_score, scoring_avg, base_strength
    Robust to different JSON shapes by recursively searching for player-like dicts.
    """

    # 1) Pull candidates
    stat_recs = _find_player_records(stats_json, max_records=5000)
    wgr_recs = _find_player_records(wgr_json, max_records=5000)

    # 2) Build stats df (id/name/scoring)
    stats_rows: List[Dict[str, Any]] = []
    seen_stats_ids = set()

    for rec in stat_recs:
        pid, name = _extract_id_and_name(rec)
        if not pid:
            continue
        # dedupe by id
        if pid in seen_stats_ids:
            continue
        seen_stats_ids.add(pid)

        scoring = _extract_scoring_avg(rec)
        stats_rows.append(
            {
                "id": pid,
                "name": name,
                "scoring_avg": scoring,
            }
        )

    stats_df = pd.DataFrame(stats_rows)

    # 3) Build WGR df (id/wgr_rank)
    wgr_rows: List[Dict[str, Any]] = []
    seen_wgr_ids = set()

    for rec in wgr_recs:
        pid, _ = _extract_id_and_name(rec)
        if not pid:
            continue
        if pid in seen_wgr_ids:
            continue
        seen_wgr_ids.add(pid)

        rk = _extract_wgr_rank(rec)
        wgr_rows.append({"id": pid, "wgr_rank": rk})

    wgr_df = pd.DataFrame(wgr_rows)

    # 4) Merge
    if stats_df.empty and not wgr_df.empty:
        # at least return ids/names if we can (names might be missing from WGR recs though)
        merged = wgr_df.copy()
        merged["name"] = merged.get("name", None)
        merged["scoring_avg"] = None
    elif stats_df.empty and wgr_df.empty:
        return pd.DataFrame(columns=["id", "name", "wgr_rank", "wgr_score", "scoring_avg", "base_strength"])
    else:
        merged = stats_df.merge(wgr_df, on="id", how="left") if not wgr_df.empty else stats_df.copy()

    # 5) Clean / defaults
    if "name" not in merged.columns:
        merged["name"] = merged["id"]

    merged["name"] = merged["name"].fillna(merged["id"]).astype(str)

    merged["wgr_rank"] = pd.to_numeric(merged.get("wgr_rank"), errors="coerce")

    # Gentle curve: lower rank => higher score
    merged["wgr_score"] = merged["wgr_rank"].fillna(999).rpow(-0.25)

    merged["scoring_avg"] = pd.to_numeric(merged.get("scoring_avg"), errors="coerce")
    if merged["scoring_avg"].isna().all():
        # fallback if we couldn't find scoring averages
        merged["scoring_avg"] = 70.5
    else:
        merged["scoring_avg"] = merged["scoring_avg"].fillna(merged["scoring_avg"].median())

    # 6) Base strength (better = higher)
    merged["base_strength"] = (70.5 - merged["scoring_avg"]) + 2.0 * merged["wgr_score"]

    # Keep a stable ordering
    merged = merged.sort_values("base_strength", ascending=False).reset_index(drop=True)

    return merged
