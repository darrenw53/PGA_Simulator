from __future__ import annotations

from pathlib import Path
import streamlit as st

from src.file_loader import WeeklyData, load_weekly_data, list_week_folders, list_fanduel_csvs
from src.features import build_model_table, make_course_fit_weights
from src.simulator import SimConfig, simulate_tournament
from src.fanduel import optimize_fanduel_lineup
from src.run_store import list_runs, load_predictions, load_settings, save_run, save_actuals_csv


APP_TITLE = "SignalAI • PGA Simulator"
PASSWORD = "signalai123"


def password_gate() -> bool:
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False

    if st.session_state.auth_ok:
        return True

    st.title(APP_TITLE)
    st.subheader("Login")
    pwd = st.text_input("Password", type="password")
    if st.button("Enter", use_container_width=True):
        if pwd == PASSWORD:
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


def _safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def _format_money(x):
    try:
        return f"${int(x):,}"
    except Exception:
        return str(x)


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    if not password_gate():
        return

    # persistent slots
    if "sim_results" not in st.session_state:
        st.session_state.sim_results = None
    if "last_tournament_id" not in st.session_state:
        st.session_state.last_tournament_id = None
    if "last_tournament_name" not in st.session_state:
        st.session_state.last_tournament_name = None
    if "last_run_record" not in st.session_state:
        st.session_state.last_run_record = None

    st.title(APP_TITLE)
    st.caption("File-driven weekly simulator + FanDuel lineup builder + saved runs (no API calls).")

    repo_root = Path(__file__).parent

    # =========================
    # RUN HISTORY (LEFT SIDEBAR)
    # =========================
    st.sidebar.header("Run History")
    runs = list_runs(repo_root)

    if runs:
        run_labels = [
            f"{r.tournament_name} • {r.run_id} • {r.created_utc}"
            for r in runs
        ]
        sel_idx = st.sidebar.selectbox(
            "Load a past run",
            options=list(range(len(runs))),
            format_func=lambda i: run_labels[i],
            index=0,
        )
        sel_run = runs[sel_idx]

        colh1, colh2 = st.sidebar.columns(2)
        if colh1.button("Load run", use_container_width=True):
            st.session_state.sim_results = load_predictions(sel_run)
            st.session_state.last_tournament_id = sel_run.tournament_id
            st.session_state.last_tournament_name = sel_run.tournament_name
            st.session_state.last_run_record = sel_run
            st.rerun()

        # downloads from sidebar
        with open(sel_run.predictions_path, "rb") as f:
            st.sidebar.download_button("Download predictions.csv", f, file_name=f"{sel_run.tournament_id}_{sel_run.run_id}_predictions.csv", use_container_width=True)
        with open(sel_run.settings_path, "rb") as f:
            st.sidebar.download_button("Download settings.json", f, file_name=f"{sel_run.tournament_id}_{sel_run.run_id}_settings.json", use_container_width=True)

        if sel_run.actuals_path and sel_run.actuals_path.exists():
            with open(sel_run.actuals_path, "rb") as f:
                st.sidebar.download_button("Download actuals.csv", f, file_name=f"{sel_run.tournament_id}_{sel_run.run_id}_actuals.csv", use_container_width=True)
    else:
        st.sidebar.caption("No saved runs yet. Run a simulation with Auto-save enabled.")

    st.sidebar.divider()

    # =========================
    # WEEKLY DATA LOADING
    # =========================
    st.sidebar.header("Weekly Data")
    weekly_root = repo_root / "data" / "weekly"
    weekly_root.mkdir(parents=True, exist_ok=True)

    week_folders = list_week_folders(weekly_root)

    data_mode = st.sidebar.radio(
        "How to load weekly data?",
        ["Use data/weekly folder", "Upload files (one-off)"],
        index=0,
    )

    weekly_data: WeeklyData | None = None
    week_label = None

    if data_mode == "Use data/weekly folder":
        if not week_folders:
            st.sidebar.warning("No week folders found in data/weekly yet.")
            st.info("Create data/weekly/<week_name>/ and add schedule.json, player_statistics.json, wgr_rankings.json, plus your FanDuel CSV.")
            st.stop()

        week_label = st.sidebar.selectbox("Select week folder", options=week_folders, index=0)
        folder_path = weekly_root / week_label

        csv_choices = list_fanduel_csvs(folder_path)
        if not csv_choices:
            st.sidebar.error("No CSV found in that week folder. Put your FanDuel CSV in it (any name).")
            st.stop()

        fd_choice = st.sidebar.selectbox("FanDuel CSV in folder", csv_choices, index=0)
        weekly_data = load_weekly_data(folder_path, fanduel_filename=fd_choice)

    else:
        st.sidebar.info("Upload the four files for a one-off run.")
        up_sched = st.sidebar.file_uploader("schedule.json", type=["json"])
        up_stats = st.sidebar.file_uploader("player_statistics.json", type=["json"])
        up_wgr = st.sidebar.file_uploader("wgr_rankings.json", type=["json"])
        up_fd = st.sidebar.file_uploader("FanDuel players CSV", type=["csv"])

        if up_sched and up_stats and up_wgr and up_fd:
            weekly_data = load_weekly_data(
                folder=None,
                schedule_bytes=up_sched.getvalue(),
                stats_bytes=up_stats.getvalue(),
                wgr_bytes=up_wgr.getvalue(),
                fanduel_bytes=up_fd.getvalue(),
            )
            week_label = "Uploaded files"

    if weekly_data is None:
        st.info("Load a week folder (data/weekly/...) or upload the files to begin.")
        st.stop()

    # =========================
    # TOURNAMENT SELECTION
    # =========================
    st.sidebar.header("Tournament")
    tourney_df = weekly_data.schedule_tournaments.copy()
    tourney_df["label"] = tourney_df["name"].astype(str) + " (" + tourney_df["start_date"].astype(str) + ")"
    sel_label = st.sidebar.selectbox("Select tournament", tourney_df["label"].tolist(), index=0)
    sel_row = tourney_df.loc[tourney_df["label"] == sel_label].iloc[0]
    tournament_id = str(sel_row["id"])
    tournament_name = str(sel_row["name"])
    course_meta = weekly_data.get_course_meta(tournament_id)

    # =========================
    # FIELD + MODEL TABLE
    # =========================
    fd_players = weekly_data.fanduel_players.copy()
    st.sidebar.header("Field")
    st.sidebar.caption(f"FanDuel rows: {len(fd_players):,}")
    min_salary = st.sidebar.slider("Min salary filter", 0, 15000, 0, step=100)
    max_salary = st.sidebar.slider("Max salary filter", 0, 15000, 15000, step=100)
    fd_players = fd_players[(fd_players["Salary"] >= min_salary) & (fd_players["Salary"] <= max_salary)].copy()

    model_table = build_model_table(
        fanduel=fd_players,
        stats=weekly_data.player_stats,
        wgr=weekly_data.wgr_players,
    )
    if model_table.empty:
        st.error("No players matched between FanDuel CSV and your stats/WGR files.")
        st.stop()

    # =========================
    # SLIDERS
    # =========================
    st.sidebar.header("Course-fit sliders")
    defaults = make_course_fit_weights()

    w_sg_total = st.sidebar.slider("Weight: SG Total", -3.0, 3.0, defaults["sg_total"], 0.05)
    w_sg_t2g = st.sidebar.slider("Weight: SG Tee-to-Green", -3.0, 3.0, defaults["sg_t2g"], 0.05)
    w_putt = st.sidebar.slider("Weight: Putting proxy (strokes_gained)", -3.0, 3.0, defaults["sg_putt_proxy"], 0.05)
    w_birdies = st.sidebar.slider("Weight: Birdies/round", -3.0, 3.0, defaults["birdies_per_round"], 0.05)
    w_gir = st.sidebar.slider("Weight: GIR%", -3.0, 3.0, defaults["gir_pct"], 0.05)
    w_drive = st.sidebar.slider("Weight: Driving distance", -3.0, 3.0, defaults["drive_avg"], 0.05)
    w_acc = st.sidebar.slider("Weight: Driving accuracy", -3.0, 3.0, defaults["drive_acc"], 0.05)
    w_scramble = st.sidebar.slider("Weight: Scrambling%", -3.0, 3.0, defaults["scrambling_pct"], 0.05)
    wgr_weight = st.sidebar.slider("WGR impact (rank → strength)", 0.0, 3.0, 1.0, 0.05)

    st.sidebar.header("Simulation")
    n_sims = st.sidebar.slider("Simulations", 100, 50000, 5000, step=100)
    rng_seed = st.sidebar.text_input("RNG seed (optional)", value="")
    cut_line = st.sidebar.slider("Cut size (after R2)", 50, 80, 65, step=1)
    round_sd = st.sidebar.slider("Round score volatility (stdev)", 1.0, 4.0, 2.3, 0.05)
    course_difficulty = st.sidebar.slider("Course difficulty shift (strokes)", -2.0, 2.0, 0.0, 0.05)

    st.sidebar.header("Run Saving")
    auto_save = st.sidebar.checkbox("Auto-save run outputs", value=True)
    run_note = st.sidebar.text_input("Run note (optional)", value="")

    # =========================
    # HEADER
    # =========================
    colA, colB, colC = st.columns([2.2, 1.2, 1.2])
    with colA:
        st.subheader(tournament_name)
        st.write(f"Selected data: **{week_label}**")
        if course_meta:
            st.caption(
                f"Course: {course_meta.get('course_name', '—')} • "
                f"Par {course_meta.get('par', '—')} • "
                f"Yardage {course_meta.get('yardage', '—')}"
            )
    with colB:
        st.metric("Field size", f"{len(model_table):,}")
    with colC:
        st.metric("Salary cap", _format_money(60000))

    st.markdown("### Field (merged)")
    preview_cols = [
        "player_id", "name", "Salary", "FPPG",
        "wgr_rank", "scoring_avg",
        "strokes_gained_total", "strokes_gained_tee_green", "strokes_gained",
        "birdies_per_round",
    ]
    show_cols = [c for c in preview_cols if c in model_table.columns]
    st.dataframe(model_table[show_cols], use_container_width=True)

    # =========================
    # RUN SIMULATION
    # =========================
    st.markdown("## Tournament simulation")

    if st.button("Run simulation", type="primary", use_container_width=True):
        weights = {
            "sg_total": w_sg_total,
            "sg_t2g": w_sg_t2g,
            "sg_putt_proxy": w_putt,
            "birdies_per_round": w_birdies,
            "gir_pct": w_gir,
            "drive_avg": w_drive,
            "drive_acc": w_acc,
            "scrambling_pct": w_scramble,
        }

        cfg = SimConfig(
            n_sims=int(n_sims),
            rng_seed=_safe_int(rng_seed, default=None) if rng_seed.strip() else None,
            cut_size=int(cut_line),
            round_sd=float(round_sd),
            course_difficulty=float(course_difficulty),
            wgr_weight=float(wgr_weight),
            course_fit_weights=weights,
        )

        with st.spinner("Simulating..."):
            results = simulate_tournament(model_table, cfg)

        st.session_state.sim_results = results
        st.session_state.last_tournament_id = tournament_id
        st.session_state.last_tournament_name = tournament_name

        st.success("Simulation complete.")

        if auto_save:
            settings_payload = {
                "week_label": week_label,
                "tournament_id": tournament_id,
                "tournament_name": tournament_name,
                "run_note": run_note,
                "sim": {
                    "n_sims": int(n_sims),
                    "rng_seed": _safe_int(rng_seed, default=None) if rng_seed.strip() else None,
                    "cut_size": int(cut_line),
                    "round_sd": float(round_sd),
                    "course_difficulty": float(course_difficulty),
                    "wgr_weight": float(wgr_weight),
                },
                "course_fit_weights": weights,
                "field_filters": {
                    "min_salary": int(min_salary),
                    "max_salary": int(max_salary),
                },
            }

            rec = save_run(
                repo_root=repo_root,
                tournament_id=tournament_id,
                tournament_name=tournament_name,
                settings=settings_payload,
                predictions=results,
            )
            st.session_state.last_run_record = rec
            st.info(f"Saved run: {rec.run_id}")

    # =========================
    # SHOW LATEST RESULTS + DOWNLOADS + ACTUALS PLACEHOLDER
    # =========================
    results = st.session_state.sim_results
    if results is not None and not results.empty:
        st.markdown("### Latest simulation results")
        summ_cols = ["name", "Salary", "FPPG", "win_pct", "top10_pct", "make_cut_pct", "avg_finish", "proj_fd_points"]
        summ_cols = [c for c in summ_cols if c in results.columns]
        st.dataframe(results[summ_cols].head(50), use_container_width=True)

        st.download_button(
            "Download current predictions CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name=f"predictions_{st.session_state.last_tournament_id}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # If last run was saved, show one-click downloads for that run
        rec = st.session_state.last_run_record
        if rec is not None:
            st.markdown("### Saved run files")
            c1, c2, c3 = st.columns(3)
            with open(rec.predictions_path, "rb") as f:
                c1.download_button("Download saved predictions.csv", f, file_name=f"{rec.tournament_id}_{rec.run_id}_predictions.csv", use_container_width=True)
            with open(rec.settings_path, "rb") as f:
                c2.download_button("Download saved settings.json", f, file_name=f"{rec.tournament_id}_{rec.run_id}_settings.json", use_container_width=True)

            # Placeholder: upload actuals later
            st.markdown("### Import actual results (placeholder)")
            st.caption("After the tournament ends, upload a CSV here and it will be saved next to this run as actuals.csv.")
            up_actuals = st.file_uploader("Upload actuals.csv for this run", type=["csv"], key="actuals_uploader")
            if up_actuals is not None:
                path = save_actuals_csv(rec, up_actuals.getvalue())
                st.success(f"Saved actuals to: {path.as_posix()}")

                with open(path, "rb") as f:
                    c3.download_button("Download actuals.csv", f, file_name=f"{rec.tournament_id}_{rec.run_id}_actuals.csv", use_container_width=True)

    # =========================
    # LINEUP BUILDER
    # =========================
    st.markdown("## FanDuel lineup builder (6 golfers, ≤ $60,000)")

    if results is None or results.empty:
        st.info("Run a simulation first (or load a past run from Run History).")
        st.stop()

    col1, col2, col3 = st.columns([1.2, 1.2, 1.6])
    with col1:
        candidate_k = st.number_input("Candidate pool size (search space)", 12, 120, 40, 1)
    with col2:
        lock_names = st.multiselect("Lock players (optional)", results["name"].tolist(), default=[])
    with col3:
        exclude_names = st.multiselect("Exclude players (optional)", results["name"].tolist(), default=[])

    blend_alpha = st.slider(
        "Lineup scoring weight (Simulation vs FanDuel FPPG)",
        0.0, 1.0, 0.75, 0.05,
        help="0.0 = only FanDuel FPPG, 1.0 = only sim projection. Middle values blend both."
    )

    if st.button("Build best lineup", use_container_width=True):
        with st.spinner("Searching best lineup under $60,000..."):
            lineup, meta = optimize_fanduel_lineup(
                sim_results=results,
                salary_cap=60000,
                lineup_size=6,
                candidate_pool=int(candidate_k),
                lock_names=set(lock_names),
                exclude_names=set(exclude_names),
                blend_alpha=float(blend_alpha),
            )

        if lineup is None or lineup.empty:
            st.error("No valid lineup found. Try increasing candidate pool or removing locks.")
        else:
            st.success("Lineup found.")
            cols = ["name", "Salary", "FPPG", "proj_fd_points"]
            if "blend_points" in lineup.columns:
                cols.append("blend_points")
            cols += ["win_pct", "top10_pct", "make_cut_pct"]
            cols = [c for c in cols if c in lineup.columns]
            st.dataframe(lineup[cols], use_container_width=True)

            st.metric("Total salary", _format_money(meta.get("total_salary", lineup["Salary"].sum())))
            st.metric("Projected lineup score", f"{meta.get('total_points', 0.0):.2f}")

            st.download_button(
                "Download lineup CSV",
                data=lineup.to_csv(index=False).encode("utf-8"),
                file_name=f"fanduel_lineup_{st.session_state.last_tournament_id}.csv",
                mime="text/csv",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
