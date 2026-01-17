import pandas as pd

def build_player_table(stats_json: dict, wgr_json: dict) -> pd.DataFrame:
    players = stats_json.get("players", []) or stats_json.get("statistics", []) or []
    df = pd.json_normalize(players)

    wgr_players = wgr_json.get("players", []) or wgr_json.get("rankings", []) or []
    wgr = pd.json_normalize(wgr_players)

    if "id" not in df.columns:
        df["id"] = None
    if "name" not in df.columns:
        df["name"] = None

    # WGR join attempt
    if "player.id" in wgr.columns and "id" in df.columns:
        wgr = wgr.rename(columns={"player.id": "id"})
    if "rank" not in wgr.columns:
        wgr["rank"] = wgr.get("position", None)

    if "id" in wgr.columns and "rank" in wgr.columns:
        merged = df.merge(wgr[["id", "rank"]], on="id", how="left")
    else:
        merged = df.copy()
        merged["rank"] = None

    merged["wgr_rank"] = pd.to_numeric(merged.get("rank"), errors="coerce")
    merged["wgr_score"] = merged["wgr_rank"].fillna(999).rpow(-0.25)

    merged["scoring_avg"] = pd.to_numeric(merged.get("scoring_average"), errors="coerce")
    if merged["scoring_avg"].isna().all():
        merged["scoring_avg"] = 70.5

    merged["base_strength"] = (70.5 - merged["scoring_avg"]) + 2.0 * merged["wgr_score"]
    return merged
