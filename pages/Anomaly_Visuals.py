# pages/Anomaly_Visuals.py
"""
Anomaly Visuals
Detailed analysis views extracted from the Anomaly Detection dashboard.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from navigation import apply_sidebar_style, render_sidebar, ensure_session, require_roles, render_page_header
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

try:
    from anomaly_detection import create_ml_detector
except Exception:
    create_ml_detector = None

st.set_page_config(
    page_title="Anomaly Visuals",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_sidebar_style()
ensure_session(timeout_minutes=None)
render_sidebar("📈 Anomaly Visuals")
require_roles(("admin", "manager"))

st.markdown(
    """
<style>
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    .tab-content {
        padding: 1.5rem;
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        margin-top: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

render_page_header("📈 Anomaly Visuals")
st.markdown('<div class="section-header" style="color: #6078ea;">Detailed Analysis</div>', unsafe_allow_html=True)

if not st.session_state.get("anomaly_data_loaded"):
    st.info("Run Anomaly Detection to load data and generate results for this page.")

daily_summaries = st.session_state.get("daily_summaries")
rule_anomalies = st.session_state.get("rule_anomalies")
statistical_anomalies = st.session_state.get("statistical_anomalies")
ml_anomalies = st.session_state.get("ml_anomalies")
ensemble_results = st.session_state.get("ensemble_results")

lateness_threshold = st.session_state.get("lateness_threshold", 30)
short_day_threshold = st.session_state.get("short_day_threshold", 4)
max_locations = st.session_state.get("max_locations", 2)
contamination_rate = st.session_state.get("contamination_rate", 0.1)
z_threshold = st.session_state.get("z_threshold", 3.0)
ml_threshold = st.session_state.get("ml_threshold", 0.7)
enable_adaptive = st.session_state.get("enable_adaptive", True)

analysis_tabs = st.tabs(
    [
        "Rule-Based Analysis",
        "Statistical Analysis",
        "ML Analysis",
        "Employee View",
        "Trends and Patterns",
        "Evaluation",
    ]
)

with analysis_tabs[0]:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    if rule_anomalies is not None and not rule_anomalies.empty:
        col1, col2 = st.columns([2, 1])
        summary_data = []
        summary_df = None

        with col1:
            st.markdown("#### Rule Violations")

            for _, row in rule_anomalies.iterrows():
                for anomaly in row["anomalies"]:
                    summary_data.append(
                        {
                            "Employee": row["employee_name"],
                            "Date": row["date"],
                            "Anomaly Type": anomaly.get("subtype", "Unknown"),
                            "Description": anomaly.get("description", ""),
                            "Severity": anomaly.get("severity", "Unknown"),
                        }
                    )

            if summary_data:
                summary_df = pd.DataFrame(summary_data)

                severity_colors = {
                    "High": "#f44336",
                    "Medium": "#ff9800",
                    "Low": "#4CAF50",
                }

                type_counts = summary_df["Anomaly Type"].value_counts().reset_index()
                type_counts.columns = ["Anomaly Type", "Count"]

                fig = px.bar(
                    type_counts,
                    x="Anomaly Type",
                    y="Count",
                    color="Anomaly Type",
                    title="Rule Violations by Type",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                )

                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Count of rule violations by type across the selected period.")

                with st.expander("View Detailed Violations"):
                    st.dataframe(summary_df, use_container_width=True)
            else:
                st.info("No rule violations found")

        with col2:
            st.markdown("#### Rule Configuration")

            st.metric("Lateness Threshold", f"{lateness_threshold} min")
            st.metric("Short Day Threshold", f"{short_day_threshold} hours")
            st.metric("Max Locations", max_locations)

            st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)
            st.markdown("#### Severity Distribution")

            if summary_df is not None and not summary_df.empty:
                severity_counts = summary_df["Severity"].value_counts()

                fig = px.pie(
                    values=severity_counts.values,
                    names=severity_counts.index,
                    title="Severity Distribution",
                    color=severity_counts.index,
                    color_discrete_map=severity_colors,
                )

                st.plotly_chart(fig, use_container_width=True)
                st.caption("Share of rule violations by severity level.")
    else:
        st.info("Run detection to see rule-based analysis")

    st.markdown("</div>", unsafe_allow_html=True)

with analysis_tabs[1]:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    if statistical_anomalies is not None and not statistical_anomalies.empty:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#### Statistical Outliers")

            fig = px.histogram(
                statistical_anomalies,
                x="ensemble_score",
                nbins=20,
                title="Anomaly Score Distribution",
                labels={"ensemble_score": "Anomaly Score"},
                color_discrete_sequence=["#636EFA"],
            )

            fig.add_vline(
                x=0.5,
                line_dash="dash",
                line_color="red",
                annotation_text="Threshold",
                annotation_position="top",
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption("Distribution of statistical anomaly scores; dashed line shows the threshold.")

            top_anomalies = statistical_anomalies.nlargest(10, "ensemble_score")
            st.dataframe(
                top_anomalies[
                    ["employee_name", "date", "ensemble_score", "ensemble_anomaly"]
                ],
                use_container_width=True,
            )

        with col2:
            st.markdown("#### Statistical Metrics")

            avg_score = statistical_anomalies["ensemble_score"].mean()
            anomaly_rate = statistical_anomalies["ensemble_anomaly"].mean() * 100

            st.metric("Average Score", f"{avg_score:.3f}")
            st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
            st.metric("Contamination", f"{contamination_rate * 100:.1f}%")

            st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)
            st.markdown("#### Configuration")
            st.info(f"Z-score threshold: {z_threshold}")
            st.info("Using robust scaling")
    else:
        st.info("Run detection to see statistical analysis")

    st.markdown("</div>", unsafe_allow_html=True)

with analysis_tabs[2]:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    if ml_anomalies is not None and not ml_anomalies.empty:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#### ML Predictions")

            fig = px.scatter(
                ml_anomalies,
                x="date",
                y="ml_anomaly_score",
                color="ml_severity",
                hover_data=["employee_name", "department"],
                title="ML Anomaly Scores Over Time",
                color_discrete_map={
                    "Critical": "#f44336",
                    "High": "#ff9800",
                    "Medium": "#2196F3",
                    "Low": "#4CAF50",
                    "Normal": "#9e9e9e",
                },
            )

            fig.add_hline(
                y=ml_threshold,
                line_dash="dash",
                line_color="red",
                annotation_text="Threshold",
                annotation_position="top right",
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption("ML anomaly scores over time; color indicates severity.")

        with col2:
            st.markdown("#### ML Performance")

            total_predictions = len(ml_anomalies)
            anomalies_detected = ml_anomalies["ml_anomaly_flag"].sum()
            avg_confidence = ml_anomalies["confidence_score"].mean()

            st.metric("Total Predictions", total_predictions)
            st.metric("Anomalies Detected", anomalies_detected)
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")

            st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)
            st.markdown("#### ML Settings")
            st.info(f"Probability threshold: {ml_threshold}")
            st.info(
                f"Adaptive learning: {'Enabled' if enable_adaptive else 'Disabled'}"
            )

            can_retrain = (
                create_ml_detector is not None
                and daily_summaries is not None
                and not daily_summaries.empty
            )
            if st.button("Retrain ML Model", use_container_width=True, disabled=not can_retrain):
                with st.spinner("Retraining ML model..."):
                    ml_detector = create_ml_detector()
                    ml_detector.train_isolation_forest(daily_summaries)
                    st.success("ML model retrained!")
            elif not can_retrain:
                st.info("Load data and run detection to enable ML retraining.")
    else:
        st.info("Run detection to see ML analysis")

    st.markdown("</div>", unsafe_allow_html=True)

with analysis_tabs[3]:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    if daily_summaries is not None and not daily_summaries.empty:
        if "employee_name" in daily_summaries.columns:
            employees = daily_summaries["employee_name"].unique()
        else:
            employees = []

        selected_employee = (
            st.selectbox("Select Employee", employees) if len(employees) > 0 else None
        )

        if selected_employee:
            emp_id = daily_summaries[
                daily_summaries["employee_name"] == selected_employee
            ]["employee_id"].iloc[0]

            emp_daily = daily_summaries[
                daily_summaries["employee_name"] == selected_employee
            ]
            emp_rule = None
            emp_stat = None
            emp_ml = None

            if rule_anomalies is not None:
                emp_rule = rule_anomalies[
                    rule_anomalies["employee_name"] == selected_employee
                ]

            if statistical_anomalies is not None:
                emp_stat = statistical_anomalies[
                    statistical_anomalies["employee_name"] == selected_employee
                ]

            if ml_anomalies is not None:
                emp_ml = ml_anomalies[ml_anomalies["employee_name"] == selected_employee]

            col1, col2, col3 = st.columns(3)

            with col1:
                dept = (
                    emp_daily["department"].iloc[0]
                    if "department" in emp_daily.columns
                    else "Unknown"
                )
                st.markdown(f"**Department:** {dept}")
                st.markdown(f"**Employee ID:** {emp_id}")

            with col2:
                avg_hours = (
                    emp_daily["work_duration_hours"].mean()
                    if "work_duration_hours" in emp_daily.columns
                    else 0
                )
                total_days = len(emp_daily)
                st.markdown(f"**Avg Hours:** {avg_hours:.1f}")
                st.markdown(f"**Days Tracked:** {total_days}")

            with col3:
                anomaly_days = 0
                if ensemble_results is not None:
                    emp_ensemble = ensemble_results[
                        ensemble_results["employee_name"] == selected_employee
                    ]
                    anomaly_days = (emp_ensemble["total_methods_flagged"] > 0).sum()

                anomaly_rate = (anomaly_days / total_days * 100) if total_days > 0 else 0
                st.markdown(f"**Anomaly Days:** {anomaly_days}")
                st.markdown(f"**Anomaly Rate:** {anomaly_rate:.1f}%")

            st.markdown("#### Attendance Timeline")

            timeline_data = emp_daily[["date", "work_duration_hours"]].copy()

            if emp_rule is not None and not emp_rule.empty:
                rule_dates = set(emp_rule["date"])
                timeline_data["rule_anomaly"] = timeline_data["date"].isin(rule_dates)

            if emp_stat is not None and not emp_stat.empty:
                stat_dates = set(emp_stat["date"])
                timeline_data["stat_anomaly"] = timeline_data["date"].isin(stat_dates)

            if emp_ml is not None and not emp_ml.empty:
                ml_dates = set(emp_ml[emp_ml["ml_anomaly_flag"] == 1]["date"])
                timeline_data["ml_anomaly"] = timeline_data["date"].isin(ml_dates)

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=timeline_data["date"],
                    y=timeline_data["work_duration_hours"],
                    name="Work Hours",
                    marker_color="#4CAF50",
                    opacity=0.7,
                )
            )

            colors = {
                "rule_anomaly": "#f44336",
                "stat_anomaly": "#2196F3",
                "ml_anomaly": "#ff9800",
            }
            names = {
                "rule_anomaly": "Rule",
                "stat_anomaly": "Statistical",
                "ml_anomaly": "ML",
            }

            for col, color in colors.items():
                if col in timeline_data.columns:
                    anomaly_dates = timeline_data[timeline_data[col]]["date"]
                    if len(anomaly_dates) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=anomaly_dates,
                                y=[timeline_data["work_duration_hours"].max()]
                                * len(anomaly_dates),
                                mode="markers",
                                name=f"{names[col]} Anomaly",
                                marker=dict(color=color, size=10, symbol="diamond"),
                                hovertemplate="%{x}<br>%{text}",
                                text=[f"{names[col]} Anomaly"] * len(anomaly_dates),
                            )
                        )

            fig.update_layout(
                title=f"Attendance Timeline for {selected_employee}",
                xaxis_title="Date",
                yaxis_title="Work Hours",
                showlegend=True,
                hovermode="x unified",
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption("Employee work hours over time with anomalies highlighted by method.")

            st.markdown("#### Detected Anomalies")

            anomaly_details = []
            if emp_rule is not None and not emp_rule.empty:
                for _, row in emp_rule.iterrows():
                    for anomaly in row["anomalies"]:
                        anomaly_details.append(
                            {
                                "Date": row["date"],
                                "Method": "Rule-Based",
                                "Type": anomaly.get("subtype", "Unknown"),
                                "Description": anomaly.get("description", ""),
                                "Severity": anomaly.get("severity", "Unknown"),
                            }
                        )

            if emp_stat is not None and not emp_stat.empty:
                for _, row in emp_stat[emp_stat["ensemble_anomaly"] == 1].iterrows():
                    anomaly_details.append(
                        {
                            "Date": row["date"],
                            "Method": "Statistical",
                            "Type": "Statistical Outlier",
                            "Description": f"Score: {row['ensemble_score']:.3f}",
                            "Severity": "Medium" if row["ensemble_score"] > 0.7 else "Low",
                        }
                    )

            if emp_ml is not None and not emp_ml.empty:
                for _, row in emp_ml[emp_ml["ml_anomaly_flag"] == 1].iterrows():
                    anomaly_details.append(
                        {
                            "Date": row["date"],
                            "Method": "ML",
                            "Type": "ML Anomaly",
                            "Description": f"Score: {row['ml_anomaly_score']:.3f}",
                            "Severity": row["ml_severity"],
                        }
                    )

            if anomaly_details:
                details_df = pd.DataFrame(anomaly_details)
                st.dataframe(details_df, use_container_width=True)
            else:
                st.success(f"No anomalies detected for {selected_employee}")
        else:
            st.info("Select an employee to view their anomaly profile")
    else:
        st.info("Run detection to load employee data")

    st.markdown("</div>", unsafe_allow_html=True)

with analysis_tabs[4]:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    if ensemble_results is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Anomaly Trends Over Time")

            daily_trend = ensemble_results.groupby("date").agg(
                {"total_methods_flagged": "sum", "ensemble_score": "mean"}
            ).reset_index()

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_trace(
                go.Bar(
                    x=daily_trend["date"],
                    y=daily_trend["total_methods_flagged"],
                    name="Anomaly Count",
                    marker_color="#f44336",
                    opacity=0.7,
                ),
                secondary_y=False,
            )

            fig.add_trace(
                go.Scatter(
                    x=daily_trend["date"],
                    y=daily_trend["ensemble_score"],
                    name="Avg Score",
                    line=dict(color="#2196F3", width=2),
                    mode="lines+markers",
                ),
                secondary_y=True,
            )

            fig.update_layout(
                title="Daily Anomaly Trends",
                xaxis_title="Date",
                showlegend=True,
            )

            fig.update_yaxes(title_text="Anomaly Count", secondary_y=False)
            fig.update_yaxes(title_text="Average Score", secondary_y=True)

            st.plotly_chart(fig, use_container_width=True)
            st.caption("Daily anomaly counts and average ensemble score over time.")

        with col2:
            st.markdown("#### Department Comparison")

            if "department" in ensemble_results.columns:
                dept_stats = ensemble_results.groupby("department").agg(
                    {
                        "employee_id": "nunique",
                        "total_methods_flagged": lambda x: (x > 0).sum(),
                    }
                ).reset_index()

                dept_stats.columns = ["Department", "Employees", "Anomaly Days"]
                dept_stats["Anomaly Rate"] = (
                    dept_stats["Anomaly Days"] / dept_stats["Employees"]
                )

                fig = px.bar(
                    dept_stats,
                    x="Department",
                    y="Anomaly Rate",
                    color="Department",
                    title="Anomaly Rate by Department",
                    text=dept_stats["Anomaly Rate"].apply(lambda x: f"{x:.1%}"),
                )

                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Anomaly rate by department based on ensemble results.")
            else:
                st.info("Department data not available")

        st.markdown("#### Day of Week Analysis")

        if "date" in ensemble_results.columns:
            ensemble_with_dow = ensemble_results.copy()
            ensemble_with_dow["day_of_week"] = ensemble_with_dow["date"].dt.day_name()
            ensemble_with_dow["has_anomaly"] = (
                ensemble_with_dow["total_methods_flagged"] > 0
            )

            heatmap_data = ensemble_with_dow.groupby("day_of_week").agg(
                {"has_anomaly": "mean", "total_methods_flagged": "count"}
            ).reset_index()

            day_order = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            heatmap_data["day_of_week"] = pd.Categorical(
                heatmap_data["day_of_week"], categories=day_order, ordered=True
            )
            heatmap_data = heatmap_data.sort_values("day_of_week")

            fig = px.bar(
                heatmap_data,
                x="day_of_week",
                y="has_anomaly",
                color="has_anomaly",
                title="Anomaly Rate by Day of Week",
                color_continuous_scale="RdYlGn_r",
                text=heatmap_data["has_anomaly"].apply(lambda x: f"{x:.1%}"),
            )

            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Anomaly rate by day of week to highlight recurring patterns.")
    else:
        st.info("Run detection to see trends and patterns")

    st.markdown("</div>", unsafe_allow_html=True)

with analysis_tabs[5]:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("#### 🧪 Evaluation (Precision / Recall / F1)")

    prepared_df = st.session_state.get("prepared_data")
    if prepared_df is None or prepared_df.empty or "is_anomaly_true" not in prepared_df.columns:
        st.info("Ground-truth labels not found. Generate/import synthetic data with 'is_anomaly_true'.")
    elif ml_anomalies is None or ml_anomalies.empty:
        st.info("Run ML detection to populate predictions.")
    else:
        truth_df = prepared_df.copy()
        truth_df["employee_id_key"] = truth_df["employee_id"].astype(str).str.strip()
        truth_df["is_anomaly_true"] = pd.to_numeric(
            truth_df["is_anomaly_true"], errors="coerce"
        ).fillna(0).astype(int)
        truth_df["date"] = pd.to_datetime(
            truth_df.get("date"), errors="coerce", dayfirst=True, format="mixed"
        )
        truth_df = truth_df.dropna(subset=["date"])
        truth_df["date_key"] = truth_df["date"].dt.normalize().dt.strftime("%Y-%m-%d")
        truth_daily = (
            truth_df.groupby(["employee_id_key", "date_key"], as_index=False)["is_anomaly_true"]
            .max()
        )
        total_truth_anomalies = int(truth_daily["is_anomaly_true"].sum())

        eval_df = ml_anomalies.copy()
        eval_df["employee_id_key"] = eval_df["employee_id"].astype(str).str.strip()
        eval_df["date_key"] = pd.to_datetime(
            eval_df["date"], errors="coerce", dayfirst=True, format="mixed"
        ).dt.normalize().dt.strftime("%Y-%m-%d")
        eval_df = eval_df.merge(
            truth_daily, on=["employee_id_key", "date_key"], how="left", indicator=True
        )
        matched_rows = int((eval_df["_merge"] == "both").sum())
        eval_df = eval_df[eval_df["_merge"] == "both"].copy()

        if eval_df.empty:
            st.warning(
                "No overlap between ML predictions and ground-truth labels after key normalization "
                "(employee_id + date)."
            )
        else:
            eval_df["is_anomaly_true"] = eval_df["is_anomaly_true"].fillna(0).astype(int)

            y_true = eval_df["is_anomaly_true"].astype(int)
            y_pred = eval_df["ml_anomaly_flag"].astype(int)

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0, 0, 0, 0))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Precision", f"{precision:.3f}")
            m2.metric("Recall", f"{recall:.3f}")
            m3.metric("F1 Score", f"{f1:.3f}")
            m4.metric("Support (Anomalies)", int(y_true.sum()))
            st.caption(
                f"Truth anomaly days: {total_truth_anomalies} | "
                f"Matched prediction rows: {matched_rows}"
            )

            st.caption(f"Confusion Matrix — TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

            # Heatmap for confusion matrix
            fig = go.Figure(
                data=go.Heatmap(
                    z=[[tn, fp], [fn, tp]],
                    x=["Pred: Normal", "Pred: Anomaly"],
                    y=["True: Normal", "True: Anomaly"],
                    colorscale="Blues",
                    text=[[tn, fp], [fn, tp]],
                    texttemplate="%{text}",
                    hovertemplate="Value: %{z}<extra></extra>",
                )
            )
            fig.update_layout(
                title="Confusion Matrix",
                margin=dict(l=0, r=0, t=40, b=0),
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Sample Evaluation Rows"):
                st.dataframe(
                    eval_df[
                        [
                            "employee_id",
                            "employee_name",
                            "date",
                            "ml_anomaly_score",
                            "ml_anomaly_flag",
                            "is_anomaly_true",
                        ]
                    ].head(20),
                    use_container_width=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)
 
# Footer
st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
    <p>AI-Powered Attendance System</p>
    <p>Anomaly Visuals Module</p>
    <p>Trends • Patterns • Anomalies • Insights</p>
    <p style="font-size: 0.8em;">Version 2.1 | Developed by: Itoro Udonyah (NOU234244897) | <a href="https://github.com/itoroudonyah" target="_blank">GitHub</a></p>
</div>
""", unsafe_allow_html=True)
