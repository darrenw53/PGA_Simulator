import pandas as pd

def build_player_table(stats_json: dict, wgr_json: dict) -> pd.DataFrame:
    # NOTE: SportsRadar field names can vary. This is a tolerant “best effort” parser.
    players = stats_json.get("players", []) or stats_json.get("statistics", []) or []
    df = pd.json_normalize(players)

    wgr_players = wgr_json.get("players", []) or wgr_json.get("rankings", []) or []
    wgr = pd.json_normalize(wgr_players)

    # Try to locate common keys
    # You may need to adjust once you see real payload structure.
    # We'll create safe columns if missing.
    for col in ["id", "name"]:
        if col not in df.columns:
            df[col] = None

    # WGR join attempt
    if "player.id" in wgr.columns and "id" in df.columns:
        wgr = wgr.rename(columns={"player.id": "id"})
    if "rank" not in wgr.columns:
        # sometimes 'position' or similar
        wgr["rank"] = wgr.get("position", None)

    merged = df.merge(wgr[["id","rank"]] if "id" in wgr.columns else wgr, on="id", how="left")

    # Create a baseline “strength” score from WGR (lower rank => stronger)
    merged["wgr_rank"] = pd.to_numeric(merged.get("rank"), errors="coerce")
    merged["wgr_score"] = merged["wgr_rank"].fillna(999).rpow(-0.25)  # gentle curve

    # Placeholder stat columns
    # You’ll map to real fields like sg_total, scoring_avg, etc. once you inspect payload.
    merged["scoring_avg"] = pd.to_numeric(merged.get("scoring_average"), errors="coerce")
    if merged["scoring_avg"].isna().all():
        merged["scoring_avg"] = 70.5  # fallback

    # Strength: lower scoring_avg => better; plus WGR stabilizer
    merged["base_strength"] = (70.5 - merged["scoring_avg"]) + 2.0 * merged["wgr_score"]

    return merged

