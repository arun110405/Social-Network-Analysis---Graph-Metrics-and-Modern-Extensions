"""
=========================================================
Visual Analytics Dashboard
=========================================================
"""

import pandas as pd
import streamlit as st

from modules.cache_manager import CacheManager
from modules.visual_analytics import VisualAnalytics


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Visual Analytics",
    layout="wide"
)

st.title("📈 Network Visual Analytics")

st.write(
    "This page displays cached centrality distributions, "
    "correlation patterns and influential-node comparisons."
)


# =====================================================
# DATASET SELECTION
# =====================================================

dataset_option = st.selectbox(
    "Select Dataset",
    [
        "Karate Club",
        "Email-Eu-Core",
        "Web-Google"
    ]
)

dataset_keys = {
    "Karate Club": "karate",
    "Email-Eu-Core": "email",
    "Web-Google": "web_google"
}

dataset_key = dataset_keys[
    dataset_option
]


# =====================================================
# CACHE VALIDATION
# =====================================================

required_files = [
    "centrality.csv",
    "correlation.json",
]

missing_files = [
    filename
    for filename in required_files
    if not CacheManager.exists(
        dataset_key,
        filename
    )
]

if missing_files:

    st.warning(
        "Cached analysis is missing. Run this dataset "
        "once from the Home page first."
    )

    st.write(
        "Missing files:",
        ", ".join(missing_files)
    )

    st.stop()


# =====================================================
# LOAD CACHED DATA
# =====================================================

try:

    centrality_df = CacheManager.load_csv(
        dataset_key,
        "centrality.csv"
    )

    correlation_df = pd.DataFrame(
        CacheManager.load_json(
            dataset_key,
            "correlation.json"
        )
    )

    VisualAnalytics.validate_centrality(
        centrality_df
    )

except Exception as error:

    st.error(
        f"Unable to load cached analytics: {error}"
    )

    st.stop()


st.success(
    "⚡ Visual analytics loaded from cached results."
)


# =====================================================
# TABS
# =====================================================

(
    heatmap_tab,
    distributions_tab,
    relationships_tab,
    ranking_tab,
    statistics_tab
) = st.tabs(
    [
        "Correlation Heatmap",
        "Distributions",
        "Metric Relationships",
        "Top Nodes",
        "Descriptive Statistics",
    ]
)


# =====================================================
# CORRELATION HEATMAP
# =====================================================

with heatmap_tab:

    st.subheader(
        "🔗 Spearman Correlation Heatmap"
    )

    st.write(
        "Values close to 1 indicate strong positive agreement. "
        "Values close to -1 indicate inverse relationships."
    )

    try:

        heatmap_figure = (
            VisualAnalytics.correlation_heatmap(
                correlation_df
            )
        )

        st.pyplot(
            heatmap_figure,
            width="stretch"
        )

        st.subheader(
            "Correlation Matrix"
        )

        st.dataframe(
            correlation_df,
            width="stretch"
        )

    except Exception as error:

        st.error(
            f"Unable to generate heatmap: {error}"
        )


# =====================================================
# CENTRALITY DISTRIBUTIONS
# =====================================================

with distributions_tab:

    st.subheader(
        "📊 Centrality Score Distributions"
    )

    selected_metric = st.selectbox(
        "Choose Centrality Metric",
        VisualAnalytics.CENTRALITY_METRICS,
        key="distribution_metric"
    )

    bins = st.slider(
        "Histogram Bins",
        min_value=10,
        max_value=60,
        value=30,
        step=5
    )

    try:

        distribution_figure = (
            VisualAnalytics.centrality_distribution(
                centrality_df,
                selected_metric,
                bins
            )
        )

        st.pyplot(
            distribution_figure,
            width="stretch"
        )

        values = pd.to_numeric(
            centrality_df[
                selected_metric
            ],
            errors="coerce"
        ).dropna()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Mean",
            f"{values.mean():.6f}"
        )

        col2.metric(
            "Median",
            f"{values.median():.6f}"
        )

        col3.metric(
            "Maximum",
            f"{values.max():.6f}"
        )

        col4.metric(
            "Standard Deviation",
            f"{values.std():.6f}"
        )

    except Exception as error:

        st.error(
            f"Unable to generate distribution: {error}"
        )


# =====================================================
# METRIC RELATIONSHIPS
# =====================================================

with relationships_tab:

    st.subheader(
        "🔍 Degree Centrality versus PageRank"
    )

    st.write(
        "This scatter plot shows whether nodes with many "
        "direct connections also receive high PageRank scores."
    )

    try:

        scatter_figure = (
            VisualAnalytics.degree_pagerank_scatter(
                centrality_df
            )
        )

        st.pyplot(
            scatter_figure,
            width="stretch"
        )

        relationship = (
            centrality_df[
                [
                    "Degree Centrality",
                    "PageRank",
                ]
            ]
            .corr(
                method="spearman"
            )
            .iloc[0, 1]
        )

        st.metric(
            "Spearman Relationship",
            f"{relationship:.4f}"
        )

        if relationship >= 0.8:

            st.success(
                "Degree Centrality and PageRank show a "
                "very strong positive relationship."
            )

        elif relationship >= 0.5:

            st.info(
                "Degree Centrality and PageRank show a "
                "moderate-to-strong positive relationship."
            )

        elif relationship >= 0.2:

            st.warning(
                "Degree Centrality and PageRank show only "
                "a weak positive relationship."
            )

        else:

            st.warning(
                "Degree Centrality and PageRank show little "
                "agreement in this network."
            )

    except Exception as error:

        st.error(
            f"Unable to create relationship plot: {error}"
        )


# =====================================================
# TOP NODE RANKINGS
# =====================================================

with ranking_tab:

    st.subheader(
        "🏆 Top Influential Nodes"
    )

    ranking_metric = st.selectbox(
        "Select Ranking Metric",
        VisualAnalytics.CENTRALITY_METRICS,
        key="ranking_metric"
    )

    top_n = st.slider(
        "Number of Nodes",
        min_value=5,
        max_value=30,
        value=10,
        step=5
    )

    try:

        ranking_figure = (
            VisualAnalytics.top_nodes_chart(
                centrality_df,
                ranking_metric,
                top_n
            )
        )

        st.pyplot(
            ranking_figure,
            width="stretch"
        )

        ranking_table = (
            centrality_df[
                ["Node", ranking_metric]
            ]
            .sort_values(
                ranking_metric,
                ascending=False
            )
            .head(top_n)
            .reset_index(drop=True)
        )

        ranking_table.insert(
            0,
            "Rank",
            range(
                1,
                len(ranking_table) + 1
            )
        )

        st.dataframe(
            ranking_table,
            width="stretch"
        )

    except Exception as error:

        st.error(
            f"Unable to generate ranking chart: {error}"
        )


# =====================================================
# DESCRIPTIVE STATISTICS
# =====================================================

with statistics_tab:

    st.subheader(
        "📋 Centrality Descriptive Statistics"
    )

    try:

        summary_df = (
            VisualAnalytics.descriptive_statistics(
                centrality_df
            )
        )

        st.dataframe(
            summary_df,
            width="stretch"
        )

        st.caption(
            "Count, mean, standard deviation, quartiles and "
            "maximum values are calculated from the cached "
            "centrality results."
        )

    except Exception as error:

        st.error(
            f"Unable to produce descriptive statistics: {error}"
        )


# =====================================================
# CACHE INFORMATION
# =====================================================

st.markdown("---")

st.caption(
    "No graph loading or centrality recomputation is performed. "
    "All figures use cached centrality and correlation results."
)