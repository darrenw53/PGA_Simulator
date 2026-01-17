import streamlit as st
import hmac

# ============================================================
# PASSWORD GATE (Option A - plain password in Secrets)
# ============================================================
# Streamlit Secrets should contain:
#
# SPORTSRADAR_API_KEY="YOUR_KEY"
#
# [auth]
# password_hash="signalai123"
#
# (We keep the key name "password_hash" for compatibility with what you already entered.)

def verify_password(password: str) -> bool:
    auth = st.secrets.get("auth", {})
    stored = auth.get("password_hash", "")
    return hmac.compare_digest(str(password), str(stored))

def login_gate():
    if st.session_state.get("auth_ok"):
        return

    st.set_page_config(page_title="PGA Simulator", layout="wide")
    st.title("🔒 SignalAI Login")

    with st.form("login_form", clear_on_submit=False):
        password = st.text_input("Password", type="password")
        ok = st.form_submit_button("Sign in")

    if ok:
        if verify_password(password):
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")

    st.stop()

login_gate()

with st.sidebar:
    if st.button("Logout"):
        st.session_state.clear()
        st.rerun()


# ============================================================
# APP
# ============================================================
import pandas as pd

from src.sr_api import (
    pga_schedule,
    pga_player_stats,
    wgr_rankings,
    tournament_scores_round,
)
from src.features import build_player_table
from src.simulator import simulate_tournament
from src.data_store import save_json

st.title("🏌️ PGA Tournament Simulator (v1)")

with st.sidebar:
    st.header("Data")
    year = st.number_input("Season Year", min_value=2020, max_value=2030, value=2026, step=1)
    load_btn = st.button("Load Schedule + Player Data", type="primary")

@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_data(year: int):
    sched = pga_schedule(year)
    stats = pga_player_stats(year)
    wgr = wgr_rankings(year)
    return sched, stats, wgr

if load_btn:
    try:
        sched, stats, wgr = load_data(int(year))
        st.session_state["sched"] = sched
        st.session_state["stats"] = stats
        st.session_state["wgr"] = wgr
        st.success("Loaded data from SportsRadar.")
    except Exception as e:
        st.error(f"Failed to load: {e}")

sched = st.session_state.get("sched")
stats = st.session_state.get("stats")
wgr = st.session_state.get("wgr")

if not (sched and stats and wgr):
    st.info("Use the sidebar to load data.")
    st.stop()

tournaments = sched.get("tournaments", []) or []
if not tournaments:
    st.warning("No tournaments found in schedule response.")
    st.stop()

tourn_df = pd.json_normalize(tournaments)
tourn_df["name"] = tourn_df.get("name", tourn_df.get("tournament.name", "Tournament"))
tourn_df["id"] = tourn_df.get("id", tourn_df.get("tournament.id", None))
tourn_df["start_date"] = tourn_df.get("start_date", tourn_df.get("start_date_time", ""))

sel = st.selectbox(
    "Select Tournament",
    options=tourn_df.index.tolist(),
    format_func=lambda i: f"{tourn_df.loc[i,'start_date']} — {tourn_df.loc[i,'name']}",
)
tournament_id = str(tourn_df.loc[sel, "id"])
tournament_name = str(tourn_df.loc[sel, "name"])

st.subheader(f"Simulation Controls — {tournament_name}")

c1, c2, c3 = st.columns(3)
with c1:
    n_sims = st.slider("Simulations", 500, 50000, 10000, step=500)
    rng_seed = st.number_input("Random Seed (optional)", value=12345, step=1)
with c2:
    tour_mean = st.slider("Tour Mean Round Score", 66.0, 74.0, 70.5, 0.1)
    base_sd = st.slider("Base Round SD", 1.5, 5.0, 2.8, 0.1)
with c3:
    cut_n = st.slider("Cut (Top N after R2)", 50, 90, 65, 1)
    hot_cold = st.slider("Hot/Cold Correlation", 0.0, 0.8, 0.25, 0.01)

st.markdown("### Adjustments")
a1, a2, a3, a4 = st.columns(4)
with a1:
    course_diff = st.slider("Course Difficulty (strokes / round)", -2.0, 2.0, 0.0, 0.1)
with a2:
    field_mult = st.slider("Field Strength Weight", 0.0, 2.5, 1.0, 0.05)
with a3:
    vol_mult = st.slider("Volatility Multiplier", 0.6, 2.0, 1.0, 0.05)
with a4:
    show_top = st.slider("Show Top N Players", 10, 200, 50, 5)

players = build_player_table(stats, wgr)

run = st.button("Run Simulation", type="primary")
if run:
    with st.spinner("Simulating..."):
        res = simulate_tournament(
            players=players,
            n_sims=int(n_sims),
            tour_mean_round=float(tour_mean),
            base_sd_round=float(base_sd),
            cut_top_n=int(cut_n),
            hot_cold_corr=float(hot_cold),
            course_difficulty=float(course_diff),
            field_strength_mult=float(field_mult),
            volatility_mult=float(vol_mult),
            rng_seed=int(rng_seed) if rng_seed else None,
        )

    st.success("Done.")
    st.dataframe(res.head(int(show_top)), use_container_width=True)

    st.download_button(
        "Download Results CSV",
        data=res.to_csv(index=False).encode("utf-8"),
        file_name=f"sim_results_{year}_{tournament_id}.csv",
        mime="text/csv",
    )

    save_json(
        {
            "year": int(year),
            "tournament_id": tournament_id,
            "tournament_name": tournament_name,
            "controls": {
                "n_sims": int(n_sims),
                "tour_mean": float(tour_mean),
                "base_sd": float(base_sd),
                "cut_n": int(cut_n),
                "hot_cold": float(hot_cold),
                "course_diff": float(course_diff),
                "field_mult": float(field_mult),
                "vol_mult": float(vol_mult),
                "seed": int(rng_seed),
            },
            "results_preview": res.head(200).to_dict(orient="records"),
        },
        name=f"projection_{year}_{tournament_id}",
    )

st.markdown("---")
st.subheader("Weekly Learning (v1)")
round_no = st.selectbox("Round to pull actual scores from (if available)", ["01", "02", "03", "04"], index=3)

if st.button("Fetch Actual Round Scores (for diagnostics)"):
    try:
        scores = tournament_scores_round(int(year), tournament_id, round_no=round_no)
        save_json(scores, name=f"actual_scores_{year}_{tournament_id}_r{round_no}")
        st.success("Saved actual score payload to history.")
        st.json(scores, expanded=False)
    except Exception as e:
        st.error(f"Failed to fetch scores: {e}")
