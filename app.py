"""FixFirst app.  Run:  streamlit run app.py   (after python pipeline.py)"""
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="FixFirst", layout="wide")


def render_traffic_page() -> None:
    st.title("Traffic: peak-hour handling")
    st.caption("Upload local traffic counts to identify busy hours and view a rule-based handling plan.")
    uploaded_file = st.file_uploader("Traffic counts CSV", type="csv")
    if uploaded_file is None:
        st.info("Upload a CSV with a timestamp column and a vehicle-count column to begin.")
        return

    try:
        source = pd.read_csv(uploaded_file)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Could not read this CSV: {error}")
        return
    if source.empty or len(source.columns) == 0:
        st.warning("The uploaded CSV has no data rows or columns.")
        return

    columns = source.columns.tolist()
    time_default = columns.index("timestamp") if "timestamp" in columns else 0
    count_default = columns.index("vehicle_count") if "vehicle_count" in columns else min(1, len(columns) - 1)
    time_col, count_col, zone_col = st.columns(3)
    with time_col:
        selected_time = st.selectbox("Datetime column", columns, index=time_default)
    with count_col:
        selected_count = st.selectbox("Vehicle-count column", columns, index=count_default)
    with zone_col:
        zone_options = ["No zone column"] + columns
        default_zone = columns.index("zone") + 1 if "zone" in columns else 0
        selected_zone = st.selectbox("Zone / junction column", zone_options, index=default_zone)

    data = source.copy()
    data[selected_time] = pd.to_datetime(data[selected_time], errors="coerce")
    data[selected_count] = pd.to_numeric(data[selected_count], errors="coerce")
    data = data.dropna(subset=[selected_time, selected_count]).copy()
    if data.empty:
        st.warning("No rows have both a valid timestamp and a numeric vehicle count.")
        return

    if selected_zone != "No zone column":
        zones = sorted(data[selected_zone].dropna().unique(), key=str)
        chosen_zones = st.multiselect("Filter zones / junctions", zones, default=zones)
        data = data[data[selected_zone].isin(chosen_zones)]
        if data.empty:
            st.info("No rows match the selected zones.")
            return

    data["hour"] = data[selected_time].dt.hour
    data["weekday"] = data[selected_time].dt.day_name()
    observed_hourly = data.groupby("hour")[selected_count].mean()
    threshold = observed_hourly.quantile(0.85)
    peak_hours = observed_hourly[observed_hourly >= threshold].index.tolist()
    by_hour = observed_hourly.reindex(range(24), fill_value=0)
    off_peak = observed_hourly.drop(index=peak_hours, errors="ignore")
    peak_to_off_peak = (
        f"{observed_hourly[peak_hours].mean() / max(off_peak.mean(), 1):.1f}x"
        if not off_peak.empty else "n/a"
    )

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Peak hours", ", ".join(f"{hour:02d}:00" for hour in peak_hours) or "none")
    metric2.metric("Busiest hour avg", f"{int(observed_hourly.max()):,} vehicles")
    metric3.metric("Peak vs off-peak", peak_to_off_peak)

    st.subheader("Average volume by hour")
    st.bar_chart(by_hour)

    st.subheader("Weekday x hour heatmap")
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap = (data.pivot_table(index="weekday", columns="hour", values=selected_count, aggfunc="mean")
               .reindex(weekday_order).reindex(columns=range(24)))
    st.dataframe(heatmap.style.background_gradient(axis=None, cmap="OrRd").format("{:.0f}"),
                 use_container_width=True)

    st.subheader("Suggested handling at peaks")
    plan = pd.DataFrame({
        "hour": [f"{hour:02d}:00" for hour in peak_hours],
        "avg_volume": [int(observed_hourly[hour]) for hour in peak_hours],
        "ratio_to_threshold": [round(observed_hourly[hour] / threshold, 2) if threshold else 0
                               for hour in peak_hours],
    })
    plan["action"] = plan["ratio_to_threshold"].apply(
        lambda ratio: "Extend green phase + deploy marshals" if ratio >= 1.15
        else "Extend green phase" if ratio >= 1.05 else "Monitor"
    )
    st.dataframe(plan, hide_index=True, use_container_width=True)
    st.caption(
        f"Peak = observed hours at or above the 85th percentile of average hourly volume "
        f"({int(threshold):,} vehicles). Actions are rule-based thresholds, not a model."
    )


page = st.sidebar.radio("Dashboard", ["FixFirst", "Traffic"], horizontal=True)
if page == "Traffic":
    render_traffic_page()
    st.stop()

df = pd.read_csv("scores.csv")
rep = json.load(open("report.json"))
L = "location_id"
ct = rep["chance_tests"]

st.title("FixFirst: which locations should be reviewed first?")
st.caption("Synthetic educational data. Decision support for municipal operations teams, not an automated decision.")

k = st.sidebar.slider("Queue size", 5, 50, 15)
sig_pick = st.sidebar.multiselect("Backlog level", ["Well above city typical", "Above typical", "Typical"],
                                  default=["Well above city typical", "Above typical", "Typical"])
view = df[df["signal"].isin(sig_pick)].head(k)

tab1, tab2, tab3 = st.tabs(["Fix-first queue", "Can we trust it? (validation)", "What's missing"])

with tab1:
    st.write("Queue = **current unresolved workload** per location: open incidents weighted by severity, injuries on "
             "open incidents, and unresolved complaints. It describes what is waiting now, not a prediction.")
    left, right = st.columns([3, 2])
    with left:
        table = view[["rank", L, "backlog_score", "signal", "open_serious", "open_incidents", "open_complaints"]].copy()
        table["backlog_score"] = table["backlog_score"].round(0)
        st.dataframe(table
                     .rename(columns={"backlog_score": "backlog score (0-100)", "signal": "vs city typical",
                                      "open_serious": "open High/Critical", "open_incidents": "open incidents",
                                      "open_complaints": "unresolved complaints"}),
                     use_container_width=True, hide_index=True)
        st.download_button("Download current queue as CSV", view.to_csv(index=False),
                           file_name="fixfirst_queue.csv", mime="text/csv")
    with right:
        pts = view.dropna(subset=["lat", "lon"])[["lat", "lon"]] if {"lat", "lon"} <= set(view.columns) else None
        if pts is not None and len(pts):
            st.map(pts)
    if len(view):
        pick = st.selectbox("Open a location", view[L].tolist())
        r = df[df[L] == pick].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Rank", int(r["rank"]))
        c2.metric("Backlog score", f"{r['backlog_score']:.0f} / 100")
        c3.markdown(f"**vs city typical**  \n{r['signal']}")
        st.subheader("Why this ranked")
        st.write(r["backlog_reason"])
        if rep["forecast_validated"]:
            st.write(f"Forecast rank range (30 resampled models): {int(r['forecast_rank_lo'])} to "
                     f"{int(r['forecast_rank_hi'])}. {r['forecast_drivers']}")
        else:
            st.info("A next-period forecast is not shown as evidence: on held-out data it did not beat simple "
                    "baselines (see the validation tab).")
    else:
        st.info("No locations match the current filters.")

with tab2:
    st.write("**Question: do locations differ in ways that persist, so a forecast can say where problems will be?**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Incident count variance / mean", f"{ct['incident_count_variance_to_mean']:.2f}",
              help="About 1.0 means location counts vary only as much as pure chance would produce.")
    c2.metric("First half vs second half (corr)", f"{ct['corr_first_half_vs_second_half_incidents']:.2f}",
              help="Near 0 means busy locations in one half are not busier in the next.")
    c3.metric("Incidents vs complaints (corr)", f"{ct['corr_incidents_vs_complaints_per_location']:.2f}",
              help="Near 0 means complaints say nothing about incidents.")
    c4.metric("Complaint count variance / mean", f"{ct['complaint_count_variance_to_mean']:.2f}")
    st.write(f"Held-out test: models trained on windows {rep['train_windows']}, tested on {rep['test_windows']} "
             f"({rep['window_days']}-day windows). Share of next-period serious incidents captured in the top "
             f"{rep['k']} locations (higher is better):")
    cap = pd.Series(rep["capture_at_k"]).rename("capture@k")
    st.bar_chart(cap)
    st.dataframe(pd.Series(rep["mean_abs_error"]).rename("MAE (lower is better)").to_frame(), use_container_width=True)
    st.caption("glm = Poisson regression trained from scratch (numpy). gbm = gradient boosting challenger. "
               "base_cum / base_last = rank by total / latest serious incident count.")
    if not rep["forecast_validated"]:
        st.warning("Result: no model beat the baselines or random ranking by a meaningful margin. In this dataset, "
                   "location differences look like random variation, so FixFirst ranks the current backlog and "
                   "does not claim to predict where incidents will happen.")
    if not rep["silence_flag_supported"]:
        st.warning("Under-reporting ('silent hotspot') flags are switched off: complaints are unrelated to incidents "
                   "here, so a low complaint count carries no information.")

with tab3:
    st.write("Data checks and how they were handled:")
    st.dataframe(pd.Series({k_: str(v) for k_, v in rep["data_issues"].items()}).rename("found").to_frame(),
                 use_container_width=True)
    st.write("Not in the data, so the queue cannot use it:")
    st.markdown("- Crew size, cost and budget, so no cost-benefit ranking\n"
                "- Real coordinates and traffic or population exposure per incident\n"
                "- Any persistent location effect (see validation tab)\n"
                "- Causes of incidents, so rankings are not evidence an intervention would work")
