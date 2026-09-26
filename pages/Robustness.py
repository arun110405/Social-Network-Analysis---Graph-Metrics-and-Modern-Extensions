"""
=========================================================
Graph Robustness Analysis Dashboard
=========================================================
"""

import hashlib
import json
import time

import pandas as pd
import streamlit as st

from modules.cache_manager import CacheManager
from modules.robustness import RobustnessAnalyzer


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Robustness Analysis",
    layout="wide"
)

st.title("🛡️ Graph Robustness Analysis")

st.write(
    "This page evaluates how stable centrality rankings remain "
    "after nodes or edges are removed from the network."
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
# CACHE REQUIREMENTS
# =====================================================

required_files = [
    "graph.pkl",
    "centrality.csv",
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
        "Required cached files are missing. "
        "Run this dataset once from the Home page first."
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

    original_graph = CacheManager.load_graph(
        dataset_key
    )

    original_centrality_df = (
        CacheManager.load_csv(
            dataset_key,
            "centrality.csv"
        )
    )

    if original_graph is None:
        raise ValueError(
            "The processed graph cache could not be loaded"
        )

except Exception as error:

    st.error(
        f"Unable to load cached graph data: {error}"
    )

    st.stop()


st.success(
    "⚡ Graph and centrality results loaded from disk cache."
)


# =====================================================
# EXPERIMENT SETTINGS
# =====================================================

st.subheader("⚙️ Experiment Configuration")

col1, col2 = st.columns(2)

with col1:

    perturbation_type = st.selectbox(
        "Structural Change",
        [
            "Node Removal",
            "Edge Removal"
        ]
    )

with col2:

    metrics = st.multiselect(
        "Centrality Measures",
        RobustnessAnalyzer.SUPPORTED_METRICS,
        default=[
            "Degree Centrality",
            "PageRank"
        ]
    )

if not metrics:

    st.warning(
        "Select at least one centrality measure."
    )

    st.stop()


percentage_mode = st.radio(
    "Experiment Mode",
    [
        "Single Removal Level",
        "Multiple Removal Levels"
    ],
    horizontal=True
)


if percentage_mode == "Single Removal Level":

    removal_percentages = [
        st.slider(
            "Removal Percentage",
            min_value=1,
            max_value=40,
            value=10,
            step=1
        )
    ]

else:

    selected_levels = st.multiselect(
        "Removal Percentages",
        options=[
            1,
            5,
            10,
            15,
            20,
            25,
            30
        ],
        default=[
            5,
            10,
            20
        ]
    )

    removal_percentages = sorted(
        selected_levels
    )


top_k = st.slider(
    "Top-K Ranking Overlap",
    min_value=5,
    max_value=30,
    value=10,
    step=5
)


random_seed = st.number_input(
    "Random Seed",
    min_value=1,
    max_value=100000,
    value=42,
    step=1
)


# =====================================================
# EXPERIMENT CACHE NAME
# =====================================================

def build_experiment_filename():
    """
    Build a deterministic cache filename from settings.
    """

    settings = {
        "dataset": dataset_key,
        "perturbation": perturbation_type,
        "percentages": removal_percentages,
        "metrics": metrics,
        "top_k": top_k,
        "seed": int(random_seed),
    }

    settings_text = json.dumps(
        settings,
        sort_keys=True
    )

    identifier = hashlib.sha256(
        settings_text.encode("utf-8")
    ).hexdigest()[:12]

    return (
        f"robustness_{identifier}.csv"
    )


experiment_filename = (
    build_experiment_filename()
)


# =====================================================
# RUN EXPERIMENT
# =====================================================

run_experiment = st.button(
    "Run Robustness Experiment",
    type="primary"
)


if run_experiment:

    start_time = time.perf_counter()

    try:

        if CacheManager.exists(
            dataset_key,
            experiment_filename
        ):

            results_df = CacheManager.load_csv(
                dataset_key,
                experiment_filename
            )

            loaded_from_cache = True

        else:

            with st.spinner(
                "Applying structural changes and "
                "recomputing selected centrality measures..."
            ):

                results_df = (
                    RobustnessAnalyzer.run_series(
                        original_graph=
                            original_graph,
                        original_centrality_df=
                            original_centrality_df,
                        perturbation_type=
                            perturbation_type,
                        percentages=
                            removal_percentages,
                        metrics=
                            metrics,
                        top_k=
                            top_k,
                        seed=
                            int(random_seed)
                    )
                )

            CacheManager.save_csv(
                dataset_key,
                experiment_filename,
                results_df
            )

            loaded_from_cache = False

        elapsed_time = (
            time.perf_counter()
            - start_time
        )

        if loaded_from_cache:

            st.success(
                "⚡ Robustness results loaded from disk cache."
            )

        else:

            st.success(
                "✅ Robustness experiment completed and cached."
            )

        st.info(
            f"⏱ Processing time: {elapsed_time:.4f} seconds"
        )

        # =================================================
        # RESULTS TABLE
        # =================================================

        st.subheader("📋 Robustness Results")

        display_df = results_df.copy()

        display_df["Interpretation"] = (
            display_df[
                "Spearman Stability"
            ].apply(
                RobustnessAnalyzer.interpret_stability
            )
        )

        st.dataframe(
            display_df,
            width="stretch"
        )

        # =================================================
        # STABILITY CHART
        # =================================================

        st.subheader(
            "📈 Ranking Stability by Removal Level"
        )

        stability_chart = (
            results_df.pivot_table(
                index="Removal Percentage",
                columns="Metric",
                values="Spearman Stability",
                aggfunc="mean"
            )
        )

        st.line_chart(
            stability_chart,
            width="stretch"
        )

        # =================================================
        # TOP-K OVERLAP CHART
        # =================================================

        overlap_column = (
            f"Top-{top_k} Overlap (%)"
        )

        if overlap_column in results_df.columns:

            st.subheader(
                f"🏆 Top-{top_k} Ranking Overlap"
            )

            overlap_chart = (
                results_df.pivot_table(
                    index="Removal Percentage",
                    columns="Metric",
                    values=overlap_column,
                    aggfunc="mean"
                )
            )

            st.line_chart(
                overlap_chart,
                width="stretch"
            )

        # =================================================
        # STRUCTURAL CHANGE
        # =================================================

        st.subheader(
            "🧱 Structural Change Summary"
        )

        structural_columns = [
            "Removal Percentage",
            "Original Nodes",
            "Perturbed Nodes",
            "Original Edges",
            "Perturbed Edges",
            "Original Density",
            "Perturbed Density",
            "Perturbed Components",
        ]

        structural_df = (
            results_df[
                structural_columns
            ]
            .drop_duplicates(
                subset=[
                    "Removal Percentage"
                ]
            )
            .reset_index(drop=True)
        )

        st.dataframe(
            structural_df,
            width="stretch"
        )

        # =================================================
        # METRIC SUMMARY
        # =================================================

        st.subheader(
            "🧠 Stability Interpretation"
        )

        metric_summary = (
            results_df.groupby(
                "Metric",
                as_index=False
            )
            .agg({
                "Spearman Stability": "mean",
                overlap_column: "mean"
            })
        )

        metric_summary[
            "Spearman Stability"
        ] = metric_summary[
            "Spearman Stability"
        ].round(4)

        metric_summary[
            overlap_column
        ] = metric_summary[
            overlap_column
        ].round(2)

        metric_summary[
            "Interpretation"
        ] = metric_summary[
            "Spearman Stability"
        ].apply(
            RobustnessAnalyzer.interpret_stability
        )

        st.dataframe(
            metric_summary,
            width="stretch"
        )

        most_stable = metric_summary.loc[
            metric_summary[
                "Spearman Stability"
            ].idxmax()
        ]

        least_stable = metric_summary.loc[
            metric_summary[
                "Spearman Stability"
            ].idxmin()
        ]

        st.success(
            f"The most stable measure is "
            f"{most_stable['Metric']} with an average "
            f"Spearman stability of "
            f"{most_stable['Spearman Stability']:.4f}."
        )

        st.warning(
            f"The least stable measure is "
            f"{least_stable['Metric']} with an average "
            f"Spearman stability of "
            f"{least_stable['Spearman Stability']:.4f}."
        )

        # =================================================
        # CACHE INFORMATION
        # =================================================

        st.subheader(
            "💾 Cached Experiment"
        )

        st.write(
            CacheManager.get_path(
                dataset_key
            )
            / experiment_filename
        )

    except Exception as error:

        st.error(
            f"Robustness experiment failed: {error}"
        )


else:

    st.info(
        "Configure the experiment and click "
        "'Run Robustness Experiment'."
    )


# =====================================================
# METHODOLOGICAL NOTE
# =====================================================

st.markdown("---")

st.caption(
    "For node-removal experiments, stability is measured "
    "only across nodes remaining in both the original and "
    "perturbed graphs. The random seed makes experiments "
    "reproducible."
)