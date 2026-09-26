"""
=========================================================
Network Explainability Dashboard
=========================================================
"""

import pandas as pd
import streamlit as st

from modules.cache_manager import CacheManager
from modules.explainability import ExplainabilityAnalyzer


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Network Explainability",
    layout="wide"
)

st.title("🧠 Explainable Network Analysis")

st.write(
    "This page explains why nodes are considered influential "
    "based on cached centrality results."
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
# CACHE CHECK
# =====================================================

if not CacheManager.exists(
    dataset_key,
    "centrality.csv"
):

    st.warning(
        "Centrality cache is missing. Run this dataset "
        "once from the Home page first."
    )

    st.stop()


# =====================================================
# LOAD CACHED CENTRALITY
# =====================================================

try:

    centrality_df = CacheManager.load_csv(
        dataset_key,
        "centrality.csv"
    )

    ExplainabilityAnalyzer.validate_dataframe(
        centrality_df
    )

except Exception as error:

    st.error(
        f"Unable to load centrality results: {error}"
    )

    st.stop()


st.success(
    "⚡ Centrality results loaded from disk cache."
)


# =====================================================
# TABS
# =====================================================

node_tab, ranking_tab, dataset_tab = st.tabs(
    [
        "Node Explanation",
        "Top Node Explanations",
        "Dataset Summary",
    ]
)


# =====================================================
# NODE EXPLANATION TAB
# =====================================================

with node_tab:

    st.subheader("🔎 Explain an Individual Node")

    node_options = (
        centrality_df["Node"]
        .astype(str)
        .tolist()
    )

    selected_node = st.selectbox(
        "Choose Node",
        node_options
    )

    try:

        explanation = (
            ExplainabilityAnalyzer.explain_node(
                centrality_df,
                selected_node
            )
        )

        st.subheader(
            f"Explanation for Node {selected_node}"
        )

        st.info(
            explanation[
                "Overall Explanation"
            ]
        )

        metric_values = explanation[
            "Metric Values"
        ]

        percentiles = explanation[
            "Percentiles"
        ]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Degree Centrality",
            f"{metric_values['Degree Centrality']:.6f}",
            f"{percentiles['Degree Centrality']:.2f} percentile"
        )

        col2.metric(
            "Betweenness",
            f"{metric_values['Betweenness Centrality']:.6f}",
            f"{percentiles['Betweenness Centrality']:.2f} percentile"
        )

        col3.metric(
            "Closeness",
            f"{metric_values['Closeness Centrality']:.6f}",
            f"{percentiles['Closeness Centrality']:.2f} percentile"
        )

        col4.metric(
            "PageRank",
            f"{metric_values['PageRank']:.8f}",
            f"{percentiles['PageRank']:.2f} percentile"
        )

        st.subheader(
            "Detailed Metric Explanations"
        )

        explanations = explanation[
            "Explanations"
        ]

        with st.expander(
            "Degree Centrality Explanation",
            expanded=True
        ):
            st.write(
                explanations[
                    "Degree Centrality"
                ]
            )

        with st.expander(
            "Betweenness Centrality Explanation"
        ):
            st.write(
                explanations[
                    "Betweenness Centrality"
                ]
            )

        with st.expander(
            "Closeness Centrality Explanation"
        ):
            st.write(
                explanations[
                    "Closeness Centrality"
                ]
            )

        with st.expander(
            "PageRank Explanation"
        ):
            st.write(
                explanations[
                    "PageRank"
                ]
            )

        st.subheader(
            "Influence Profile"
        )

        profile_df = pd.DataFrame({
            "Metric": list(
                percentiles.keys()
            ),
            "Percentile": list(
                percentiles.values()
            )
        })

        st.bar_chart(
            profile_df.set_index(
                "Metric"
            ),
            width="stretch"
        )

    except Exception as error:

        st.error(
            f"Unable to explain node: {error}"
        )


# =====================================================
# TOP NODE RANKING TAB
# =====================================================

with ranking_tab:

    st.subheader(
        "🏆 Explain Top Influential Nodes"
    )

    selected_metric = st.selectbox(
        "Select Centrality Metric",
        ExplainabilityAnalyzer.METRICS,
        key="ranking_metric"
    )

    top_n = st.slider(
        "Number of Nodes",
        min_value=5,
        max_value=25,
        value=10,
        step=5
    )

    try:

        ranking_df = (
            ExplainabilityAnalyzer.explain_top_nodes(
                centrality_df,
                selected_metric,
                top_n
            )
        )

        st.dataframe(
            ranking_df,
            width="stretch"
        )

        top_node = ranking_df.iloc[0]

        st.success(
            f"Node {top_node['Node']} is ranked first by "
            f"{selected_metric}, with a score of "
            f"{float(top_node[selected_metric]):.6f}. "
            f"This places it at approximately the "
            f"{float(top_node['Percentile']):.2f} percentile."
        )

    except Exception as error:

        st.error(
            f"Unable to generate ranking: {error}"
        )


# =====================================================
# DATASET SUMMARY TAB
# =====================================================

with dataset_tab:

    st.subheader(
        "📋 Dataset-Level Explanation"
    )

    try:

        findings = (
            ExplainabilityAnalyzer.dataset_summary(
                centrality_df,
                dataset_option
            )
        )

        for finding in findings:
            st.write(
                f"• {finding}"
            )

        st.subheader(
            "Centrality Descriptive Statistics"
        )

        summary_df = (
            centrality_df[
                ExplainabilityAnalyzer.METRICS
            ]
            .describe()
            .transpose()
            .reset_index()
            .rename(
                columns={
                    "index": "Metric"
                }
            )
        )

        st.dataframe(
            summary_df,
            width="stretch"
        )

        st.caption(
            "All explanations are generated using deterministic "
            "rules and cached centrality values. No graph "
            "recomputation is performed on this page."
        )

    except Exception as error:

        st.error(
            f"Unable to generate dataset summary: {error}"
        )