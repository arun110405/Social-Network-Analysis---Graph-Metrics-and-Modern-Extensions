"""
=========================================================
Performance Evaluation Dashboard
=========================================================
"""

import streamlit as st

from modules.performance import PerformanceManager


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Performance Evaluation",
    layout="wide"
)

st.title("⚡ Performance Evaluation")

st.write(
    "This page evaluates first-run processing performance "
    "and persistent disk-cache performance."
)


# =====================================================
# LOAD PERFORMANCE DATA
# =====================================================

performance_df = PerformanceManager.load()

if performance_df.empty:

    st.warning(
        "No performance measurements are available."
    )

    st.info(
        "Run each dataset from the Home page to generate "
        "performance records."
    )

    st.stop()


# =====================================================
# FILTER CONTROLS
# =====================================================

available_datasets = sorted(
    performance_df["Dataset"]
    .dropna()
    .unique()
    .tolist()
)

selected_datasets = st.multiselect(
    "Select Datasets",
    options=available_datasets,
    default=available_datasets
)

if not selected_datasets:

    st.warning(
        "Select at least one dataset."
    )

    st.stop()


filtered_df = performance_df[
    performance_df["Dataset"].isin(
        selected_datasets
    )
].copy()


# =====================================================
# OVERVIEW METRICS
# =====================================================

st.subheader("📊 Benchmark Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Runs",
    len(filtered_df)
)

col2.metric(
    "First Runs",
    int(
        (
            filtered_df["Run Type"]
            == "First Run"
        ).sum()
    )
)

col3.metric(
    "Cached Runs",
    int(
        (
            filtered_df["Run Type"]
            == "Cached Run"
        ).sum()
    )
)

col4.metric(
    "Average Total Time",
    f"{filtered_df['Total Time'].mean():.4f} s"
)


# =====================================================
# CACHE IMPROVEMENT
# =====================================================

st.subheader("🚀 Cache Performance Improvement")

improvement_df = (
    PerformanceManager.cache_improvement(
        filtered_df
    )
)

if improvement_df.empty:

    st.info(
        "Both a first run and a cached run are required "
        "for each dataset before improvement can be calculated."
    )

else:

    st.dataframe(
        improvement_df,
        width="stretch"
    )

    chart_df = (
        improvement_df[
            [
                "Dataset",
                "First Run Average (s)",
                "Cached Run Average (s)",
            ]
        ]
        .set_index("Dataset")
    )

    st.bar_chart(
        chart_df,
        width="stretch"
    )


# =====================================================
# AVERAGE STAGE TIMES
# =====================================================

st.subheader("⏱ Average Stage Times")

average_df = (
    PerformanceManager.average_by_run_type(
        filtered_df
    )
)

st.dataframe(
    average_df,
    width="stretch"
)

if not average_df.empty:

    stage_chart_df = average_df.copy()

    stage_chart_df["Label"] = (
        stage_chart_df["Dataset"]
        + " — "
        + stage_chart_df["Run Type"]
    )

    stage_columns = [
        "Raw Graph Load Time",
        "Graph Cache Load Time",
        "Preprocessing Time",
        "Statistics Time",
        "Centrality Time",
        "Correlation Time",
        "Analysis Cache Load Time",
    ]

    stage_chart_df = (
        stage_chart_df[
            ["Label", *stage_columns]
        ]
        .set_index("Label")
    )

    st.bar_chart(
        stage_chart_df,
        width="stretch"
    )


# =====================================================
# TOTAL TIME HISTORY
# =====================================================

st.subheader("📈 Processing-Time History")

history_df = filtered_df.copy()

history_df["Run Number"] = (
    history_df.groupby(
        "Dataset"
    ).cumcount() + 1
)

history_chart = history_df.pivot_table(
    index="Run Number",
    columns="Dataset",
    values="Total Time",
    aggfunc="last"
)

st.line_chart(
    history_chart,
    width="stretch"
)


# =====================================================
# LATEST RUNS
# =====================================================

st.subheader("🕒 Latest Recorded Run")

latest_rows = []

for dataset in selected_datasets:

    latest = PerformanceManager.latest_run(
        filtered_df,
        dataset
    )

    if latest is not None:

        latest_rows.append(
            latest.to_dict()
        )

if latest_rows:

    import pandas as pd

    latest_df = pd.DataFrame(
        latest_rows
    )

    st.dataframe(
        latest_df,
        width="stretch"
    )


# =====================================================
# RAW PERFORMANCE HISTORY
# =====================================================

with st.expander(
    "View Full Benchmark History"
):

    st.dataframe(
        filtered_df,
        width="stretch"
    )


# =====================================================
# INTERPRETATION
# =====================================================

st.subheader("🧠 Performance Interpretation")

if not improvement_df.empty:

    for _, row in improvement_df.iterrows():

        dataset = row["Dataset"]
        improvement = row["Improvement (%)"]
        speedup = row["Speed-up"]

        if improvement >= 90:

            interpretation = (
                "The persistent cache provides a very large "
                "performance improvement."
            )

        elif improvement >= 70:

            interpretation = (
                "The persistent cache provides a substantial "
                "performance improvement."
            )

        elif improvement >= 40:

            interpretation = (
                "The persistent cache provides a moderate "
                "performance improvement."
            )

        else:

            interpretation = (
                "The performance improvement is limited, "
                "and further optimisation may be useful."
            )

        st.write(
            f"**{dataset}:** {improvement:.2f}% faster, "
            f"with an approximately {speedup:.2f}× speed-up. "
            f"{interpretation}"
        )


# =====================================================
# CLEAR HISTORY
# =====================================================

st.markdown("---")

with st.expander(
    "Performance Data Management"
):

    st.warning(
        "Clearing the performance history does not delete "
        "graph, analysis, community, or visualisation caches."
    )

    confirm_clear = st.checkbox(
        "I understand that benchmark history will be deleted"
    )

    if st.button(
        "Clear Performance History",
        disabled=not confirm_clear
    ):

        if PerformanceManager.clear():

            st.success(
                "Performance history cleared."
            )

            st.rerun()

        else:

            st.info(
                "No performance history was found."
            )


# =====================================================
# CACHE INFORMATION
# =====================================================

st.markdown("---")

st.caption(
    f"Performance records are stored in: "
    f"{PerformanceManager.PERFORMANCE_FILE}"
)