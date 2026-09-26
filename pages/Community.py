"""
=========================================================
Community Detection and Comparison Page
=========================================================

This page supports:

1. Detailed analysis of one algorithm on one dataset.
2. Comparison of multiple algorithms on one dataset.
3. Comparison of one algorithm across multiple datasets.
=========================================================
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.cache_manager import CacheManager
from modules.community_detection import CommunityAnalyzer


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Community Detection",
    layout="wide",
)

st.title("👥 Community Detection and Comparison")

st.write(
    "Detect network communities, compare different algorithms "
    "on the same dataset, and compare the same algorithm across "
    "multiple datasets."
)


# =====================================================
# DATASET CONFIGURATION
# =====================================================

DATASETS = {
    "Karate Club": "karate",
    "Email-Eu-Core": "email",
    "Web-Google": "web_google",
}

ALGORITHMS = {
    "Greedy Modularity": "greedy",
    "Louvain": "louvain",
    "Label Propagation": "label_propagation",
}


# =====================================================
# USER CONTROLS
# =====================================================

control_col_1, control_col_2 = st.columns(2)

with control_col_1:

    dataset_option = st.selectbox(
        "Select Dataset",
        list(DATASETS.keys()),
    )

with control_col_2:

    selected_algorithm_name = st.selectbox(
        "Select Community Detection Algorithm",
        list(ALGORITHMS.keys()),
    )

dataset_key = DATASETS[dataset_option]

selected_algorithm = ALGORITHMS[
    selected_algorithm_name
]


# =====================================================
# ALGORITHM SETTINGS
# =====================================================

setting_col_1, setting_col_2, setting_col_3 = st.columns(3)

with setting_col_1:

    seed = st.number_input(
        "Random Seed",
        min_value=0,
        value=42,
        step=1,
        help=(
            "Used to make Louvain and Label Propagation "
            "results reproducible."
        ),
    )

with setting_col_2:

    resolution = st.slider(
        "Louvain Resolution",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help=(
            "Higher values generally produce more communities. "
            "This setting only affects Louvain."
        ),
    )

with setting_col_3:

    centrality_measure = st.selectbox(
        "Node Size in Visualisation",
        [
            "PageRank",
            "Degree",
        ],
    )

    current_centrality_measure = centrality_measure

# =====================================================
# ADVANCED EVALUATION SETTINGS
# =====================================================

st.subheader("Advanced Evaluation")

evaluation_col_1, evaluation_col_2 = st.columns(2)

with evaluation_col_1:
    run_stability_testing = st.checkbox(
        "Run repeated-seed stability analysis",
        value=True,
        help=(
            "Runs each algorithm several times using different "
            "random seeds and measures result consistency."
        ),
    )

with evaluation_col_2:
    stability_runs = st.number_input(
        "Number of Stability Runs",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
        disabled=not run_stability_testing,
        help=(
            "Three runs are suitable for initial testing. "
            "Five runs provide a stronger evaluation."
        ),
    )

# Convert Streamlit values to normal Python values
run_stability_testing = bool(
    run_stability_testing
)

stability_runs = int(
    stability_runs
)

# =====================================================
# ANALYSIS MODE BUTTONS
# =====================================================

st.subheader("Select Analysis Type")

button_col_1, button_col_2, button_col_3 = st.columns(3)

with button_col_1:

    run_single_button = st.button(
        "Run Selected Algorithm",
        type="primary",
        width="stretch",
    )

with button_col_2:

    compare_algorithms_button = st.button(
        "Compare Algorithms on Dataset",
        width="stretch",
    )

with button_col_3:

    compare_datasets_button = st.button(
        "Compare Algorithm Across Datasets",
        width="stretch",
    )


# =====================================================
# CACHE FILE HELPERS
# =====================================================

def algorithm_cache_files(
    algorithm_key: str,
) -> Dict[str, str]:
    """
    Return algorithm-specific cache filenames.
    """

    prefix = f"community_{algorithm_key}"

    return {
        "communities": f"{prefix}_communities.json",
        "summary": f"{prefix}_summary.csv",
        "membership": f"{prefix}_membership.csv",
        "stats": f"{prefix}_stats.json",
    }


def algorithm_cache_ready(
    current_dataset_key: str,
    algorithm_key: str,
) -> bool:
    """
    Check whether all required cache files exist for one
    dataset and one algorithm.
    """

    cache_files = algorithm_cache_files(
        algorithm_key
    )

    return all(
        CacheManager.exists(
            current_dataset_key,
            filename,
        )
        for filename in cache_files.values()
    )


# =====================================================
# GRAPH LOADING
# =====================================================

def load_processed_graph(
    current_dataset_key: str,
    current_dataset_name: str,
) -> nx.Graph | None:
    """
    Load a processed graph from the persistent graph cache.
    """

    graph = CacheManager.load_graph(
        current_dataset_key
    )

    if graph is None:

        st.warning(
            f"Processed graph cache for {current_dataset_name} "
            "was not found. Run this dataset from the Home page first."
        )

        return None

    return graph


# =====================================================
# GRAPH STRUCTURAL STATISTICS
# =====================================================

def graph_structure_record(
    graph: nx.Graph,
) -> Dict:
    """
    Return graph characteristics useful for explaining
    differences in community-detection results.
    """

    prepared_graph = CommunityAnalyzer.prepare_graph(
        graph
    )

    return {
        "Nodes": prepared_graph.number_of_nodes(),
        "Edges": prepared_graph.number_of_edges(),
        "Density": nx.density(prepared_graph),
        "Average Degree": (
            sum(
                dict(
                    prepared_graph.degree()
                ).values()
            )
            / prepared_graph.number_of_nodes()
        ),
        "Clustering Coefficient": (
            nx.average_clustering(
                prepared_graph
            )
        ),
    }


# =====================================================
# CENTRALITY VALUES FOR VISUALISATION
# =====================================================

def load_node_importance(
    graph: nx.Graph,
    current_dataset_key: str,
    measure: str,
) -> Dict:
    """
    Load PageRank or Degree values used to control node size
    in the community network visualisation.
    """

    if measure == "Degree":

        return dict(
            graph.degree()
        )

    possible_centrality_files = [
        "centrality.csv",
        "centrality_results.csv",
    ]

    for filename in possible_centrality_files:

        if not CacheManager.exists(
            current_dataset_key,
            filename,
        ):
            continue

        centrality_df = CacheManager.load_csv(
            current_dataset_key,
            filename,
        )

        if (
            centrality_df is None
            or centrality_df.empty
        ):
            continue

        possible_node_columns = [
            "Node",
            "node",
        ]

        possible_pagerank_columns = [
            "PageRank",
            "pagerank",
            "page_rank",
        ]

        node_column = next(
            (
                column
                for column in possible_node_columns
                if column in centrality_df.columns
            ),
            None,
        )

        pagerank_column = next(
            (
                column
                for column in possible_pagerank_columns
                if column in centrality_df.columns
            ),
            None,
        )

        if (
            node_column is not None
            and pagerank_column is not None
        ):

            return dict(
                zip(
                    centrality_df[node_column],
                    centrality_df[pagerank_column],
                )
            )

    prepared_graph = (
        CommunityAnalyzer.prepare_graph(
            graph
        )
    )

    return nx.pagerank(
        prepared_graph
    )


# =====================================================
# LOAD OR COMPUTE ALGORITHM
# =====================================================

def load_or_compute_algorithm(
    graph: nx.Graph,
    current_dataset_key: str,
    current_dataset_name: str,
    algorithm_key: str,
) -> Tuple:
    """
    Load cached community-detection results or compute and
    save new results.

    Old cache files are automatically upgraded when graph
    statistics are missing.
    """

    cache_files = algorithm_cache_files(
        algorithm_key
    )

    # =================================================
    # LOAD EXISTING CACHE
    # =================================================

    if algorithm_cache_ready(
        current_dataset_key,
        algorithm_key,
    ):

        cached_communities = CacheManager.load_json(
            current_dataset_key,
            cache_files["communities"],
        )

        communities = (
            CommunityAnalyzer.communities_from_dict(
                cached_communities
            )
        )

        summary_df = CacheManager.load_csv(
            current_dataset_key,
            cache_files["summary"],
        )

        membership_df = CacheManager.load_csv(
            current_dataset_key,
            cache_files["membership"],
        )

        community_stats = CacheManager.load_json(
            current_dataset_key,
            cache_files["stats"],
        )

        # ---------------------------------------------
        # UPGRADE OLD CACHE FILES
        # ---------------------------------------------

        required_structure_keys = {
            "Nodes",
            "Edges",
            "Density",
            "Average Degree",
            "Clustering Coefficient",
        }

        missing_structure_keys = (
            required_structure_keys
            - set(community_stats.keys())
        )

        if missing_structure_keys:

            structure_stats = graph_structure_record(
                graph
            )

            community_stats.update(
                {
                    "Nodes": structure_stats["Nodes"],
                    "Edges": structure_stats["Edges"],
                    "Density": round(
                        structure_stats["Density"],
                        8,
                    ),
                    "Average Degree": round(
                        structure_stats["Average Degree"],
                        6,
                    ),
                    "Clustering Coefficient": round(
                        structure_stats[
                            "Clustering Coefficient"
                        ],
                        6,
                    ),
                }
            )

            # Save upgraded cache
            CacheManager.save_json(
                current_dataset_key,
                cache_files["stats"],
                community_stats,
            )

        return (
            communities,
            summary_df,
            membership_df,
            community_stats,
            True,
        )

    # =================================================
    # COMPUTE NEW RESULTS
    # =================================================

    result = CommunityAnalyzer.run_algorithm(
        graph=graph,
        algorithm=algorithm_key,
        seed=int(seed),
        resolution=float(resolution),
    )

    communities = result["communities"]

    summary_df = (
        CommunityAnalyzer.summarize_communities(
            communities
        )
    )

    membership_df = (
        CommunityAnalyzer.membership_dataframe(
            communities
        )
    )

    structure_stats = graph_structure_record(
        graph
    )

    community_stats = {
        "Dataset": current_dataset_name,
        "Algorithm": result["algorithm_name"],
        "Algorithm Key": algorithm_key,

        "Number of Communities": result[
            "number_of_communities"
        ],

        "Modularity": round(
            result["modularity"],
            6,
        ),

        "Runtime Seconds": round(
            result["runtime_seconds"],
            6,
        ),

        "Largest Community": result[
            "largest_community"
        ],

        "Smallest Community": result[
            "smallest_community"
        ],

        "Average Community Size": round(
            result["average_community_size"],
            3,
        ),

        "Nodes": structure_stats["Nodes"],

        "Edges": structure_stats["Edges"],

        "Density": round(
            structure_stats["Density"],
            8,
        ),

        "Average Degree": round(
            structure_stats["Average Degree"],
            6,
        ),

        "Clustering Coefficient": round(
            structure_stats[
                "Clustering Coefficient"
            ],
            6,
        ),

        "Seed": int(seed),

        "Resolution": float(resolution),
    }

    # =================================================
    # SAVE CACHE
    # =================================================

    CacheManager.save_json(
        current_dataset_key,
        cache_files["communities"],
        CommunityAnalyzer.communities_to_dict(
            communities
        ),
    )

    CacheManager.save_csv(
        current_dataset_key,
        cache_files["summary"],
        summary_df,
    )

    CacheManager.save_csv(
        current_dataset_key,
        cache_files["membership"],
        membership_df,
    )

    CacheManager.save_json(
        current_dataset_key,
        cache_files["stats"],
        community_stats,
    )

    return (
        communities,
        summary_df,
        membership_df,
        community_stats,
        False,
    )


# =====================================================
# NETWORK VISUALISATION
# =====================================================

def create_community_figure(
    graph: nx.Graph,
    communities: List[set],
    importance_scores: Dict,
    algorithm_name: str,
    current_dataset_name: str,
    current_centrality_measure: str,
    max_visual_nodes: int = 300,
    positions: Dict | None = None,
) -> go.Figure:
    """
    Create an interactive community network graph.

    Node colour represents community membership.
    Node size represents PageRank or Degree.

    When positions are provided, the same node layout is used
    across multiple algorithms for a fair visual comparison.
    """

    prepared_graph = (
        CommunityAnalyzer.prepare_graph(
            graph
        )
    )

    membership = (
        CommunityAnalyzer.membership_mapping(
            communities
        )
    )

    visual_graph = prepared_graph
    sampled_visualisation = False

    # =================================================
    # SAMPLE LARGE NETWORKS
    # =================================================

    if (
        prepared_graph.number_of_nodes()
        > max_visual_nodes
    ):

        sampled_visualisation = True

        selected_nodes = sorted(
            prepared_graph.degree(),
            key=lambda item: item[1],
            reverse=True,
        )[:max_visual_nodes]

        selected_node_ids = [
            node
            for node, _ in selected_nodes
        ]

        visual_graph = prepared_graph.subgraph(
            selected_node_ids
        ).copy()

        if positions is not None:

            positions = {
                node: positions[node]
                for node in visual_graph.nodes()
                if node in positions
            }

    # =================================================
    # SHARED OR NEW NODE POSITIONS
    # =================================================

    if positions is None:

        positions = nx.spring_layout(
            visual_graph,
            seed=42,
            iterations=50,
        )

    else:

        # Keep only positions belonging to visible nodes
        positions = {
            node: positions[node]
            for node in visual_graph.nodes()
            if node in positions
        }

        # Calculate positions for any missing visible nodes
        missing_nodes = [
            node
            for node in visual_graph.nodes()
            if node not in positions
        ]

        if missing_nodes:

            calculated_positions = nx.spring_layout(
                visual_graph,
                seed=42,
                iterations=50,
            )

            for node in missing_nodes:
                positions[node] = (
                    calculated_positions[node]
                )

    # =================================================
    # EDGE TRACE
    # =================================================

    edge_x = []
    edge_y = []

    for source, target in visual_graph.edges():

        source_x, source_y = positions[source]
        target_x, target_y = positions[target]

        edge_x.extend(
            [
                source_x,
                target_x,
                None,
            ]
        )

        edge_y.extend(
            [
                source_y,
                target_y,
                None,
            ]
        )

    edge_trace = go.Scattergl(
        x=edge_x,
        y=edge_y,
        mode="lines",
        hoverinfo="none",
        line={
            "width": 0.5,
            "color": "rgba(130,130,130,0.35)",
        },
        name="Edges",
    )

    # =================================================
    # NODE IMPORTANCE SCORES
    # =================================================

    graph_scores = {}

    for node in visual_graph.nodes():

        direct_score = importance_scores.get(
            node
        )

        string_score = importance_scores.get(
            str(node)
        )

        score = (
            direct_score
            if direct_score is not None
            else string_score
        )

        graph_scores[node] = float(
            score
            if score is not None
            else 0.0
        )

    score_values = list(
        graph_scores.values()
    )

    minimum_score = (
        min(score_values)
        if score_values
        else 0.0
    )

    maximum_score = (
        max(score_values)
        if score_values
        else 1.0
    )

    score_range = (
        maximum_score - minimum_score
    )

    # =================================================
    # NODE TRACE
    # =================================================

    node_x = []
    node_y = []
    node_sizes = []
    node_colours = []
    node_hover_text = []

    for node in visual_graph.nodes():

        x_position, y_position = positions[node]

        node_x.append(
            x_position
        )

        node_y.append(
            y_position
        )

        score = graph_scores[node]

        if score_range > 0:

            normalised_score = (
                (score - minimum_score)
                / score_range
            )

        else:

            normalised_score = 0.5

        node_sizes.append(
            10 + normalised_score * 30
        )

        community_index = membership.get(
            node,
            -1,
        )

        community_number = (
            community_index + 1
            if community_index >= 0
            else 0
        )

        node_colours.append(
            community_index
        )

        node_colours.append(
            community_number
        )


        node_hover_text.append(
            f"Node: {node}<br>"
            f"Community: {community_number}<br>"
            f"{current_centrality_measure}: "
            f"{score:.6f}<br>"
            f"Degree: {visual_graph.degree(node)}"
        )

    node_trace = go.Scattergl(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        hovertext=node_hover_text,
        marker={
            "size": node_sizes,
            "color": node_colours,
            "colorscale": "Turbo",
            "showscale": True,
            "colorbar": {
                "title": "Community",
            },
            "line": {
                "width": 0.5,
                "color": "white",
            },
            "opacity": 0.9,
        },
        name="Nodes",
    )

    # =================================================
    # FIGURE
    # =================================================

    figure_title = (
        f"{current_dataset_name}: "
        f"{algorithm_name} Communities"
    )

    if sampled_visualisation:

        figure_title += (
            f" — Top {max_visual_nodes} "
            "Connected Nodes"
        )

    figure = go.Figure(
        data=[
            edge_trace,
            node_trace,
        ]
    )

    figure.update_layout(
        title=figure_title,
        showlegend=False,
        hovermode="closest",
        margin={
            "l": 10,
            "r": 10,
            "t": 60,
            "b": 10,
        },
        xaxis={
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
        },
        yaxis={
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
        },
        height=700,
    )

    return figure

# =====================================================
# SINGLE ALGORITHM ANALYSIS
# =====================================================

def display_single_algorithm(
    graph: nx.Graph,
    current_dataset_key: str,
    current_dataset_name: str,
    algorithm_key: str,
) -> None:
    """
    Display detailed community-detection results for one
    selected algorithm and dataset.
    """

    start_time = time.perf_counter()

    (
        communities,
        summary_df,
        membership_df,
        community_stats,
        loaded_from_cache,
    ) = load_or_compute_algorithm(
        graph=graph,
        current_dataset_key=current_dataset_key,
        current_dataset_name=current_dataset_name,
        algorithm_key=algorithm_key,
    )

    page_elapsed = (
        time.perf_counter()
        - start_time
    )

    if loaded_from_cache:

        st.success(
            "⚡ Community results loaded from disk cache."
        )

    else:

        st.success(
            "✅ Community detection completed and cached."
        )

    st.caption(
        f"Page processing time: "
        f"{page_elapsed:.3f} seconds"
    )

    st.subheader("📊 Community Overview")

    metric_col_1, metric_col_2, metric_col_3, \
        metric_col_4, metric_col_5 = st.columns(5)

    metric_col_1.metric(
        "Algorithm",
        community_stats["Algorithm"],
    )

    metric_col_2.metric(
        "Communities",
        community_stats[
            "Number of Communities"
        ],
    )

    metric_col_3.metric(
        "Modularity",
        community_stats["Modularity"],
    )

    metric_col_4.metric(
        "Largest Community",
        community_stats[
            "Largest Community"
        ],
    )

    metric_col_5.metric(
        "Runtime",
        (
            f'{community_stats["Runtime Seconds"]:.4f}s'
        ),
    )

    st.caption(
        "Modularity measures how strongly nodes are grouped "
        "within communities compared with the connections "
        "between communities."
    )

    # -------------------------------------------------
    # COMMUNITY GRAPH
    # -------------------------------------------------

    st.subheader(
        "🌐 Community Structure and Central Nodes"
    )

    importance_scores = load_node_importance(
        graph=graph,
        current_dataset_key=current_dataset_key,
        measure=centrality_measure,
    )

    community_figure = create_community_figure(
        graph=graph,
        communities=communities,
        importance_scores=importance_scores,
        algorithm_name=community_stats[
            "Algorithm"
        ],
        current_dataset_name=current_dataset_name,
        current_centrality_measure=centrality_measure,
    )

    st.plotly_chart(
        community_figure,
        width="stretch",
    )

    st.caption(
        "Node colour represents community membership. "
        f"Node size represents {centrality_measure}. "
        "The larger nodes are more influential under the "
        "selected centrality measurement."
    )

    # -------------------------------------------------
    # COMMUNITY SIZE ANALYSIS
    # -------------------------------------------------

    st.subheader("📋 Community Size Analysis")

    summary_col_1, summary_col_2 = st.columns(
        [1, 1.4]
    )

    with summary_col_1:

        st.dataframe(
            summary_df,
            width="stretch",
            hide_index=True,
        )

    with summary_col_2:

        size_figure = px.bar(
            summary_df,
            x="Community",
            y="Size",
            title="Community Size Distribution",
            hover_data=["Percentage"],
        )

        st.plotly_chart(
            size_figure,
            width="stretch",
        )

    # -------------------------------------------------
    # NODE MEMBERSHIP
    # -------------------------------------------------

    st.subheader("🔎 Node Community Membership")

    selected_community = st.selectbox(
        "Filter by Community",
        [
            "All"
        ] + summary_df[
            "Community"
        ].tolist(),
        key=(
            f"membership_filter_"
            f"{current_dataset_key}_"
            f"{algorithm_key}"
        ),
    )

    if selected_community == "All":

        filtered_membership = (
            membership_df
        )

    else:

        filtered_membership = (
            membership_df[
                membership_df[
                    "Community"
                ] == selected_community
            ]
        )

    st.dataframe(
        filtered_membership,
        width="stretch",
        hide_index=True,
    )

    # -------------------------------------------------
    # CACHE INFORMATION
    # -------------------------------------------------

    with st.expander(
        "💾 View Cached Community Files"
    ):

        cache_files = algorithm_cache_files(
            algorithm_key
        )

        cache_directory = CacheManager.get_path(
            current_dataset_key
        )

        for label, filename in cache_files.items():

            st.write(
                f"{label.replace('_', ' ').title()}:",
                Path(cache_directory) / filename,
            )


def display_algorithm_comparison(
    graph: nx.Graph,
    current_dataset_key: str,
    current_dataset_name: str,
    current_centrality_measure: str,
) -> None:
    """
    Compare all algorithms on one dataset using quality,
    efficiency, stability, balance and external validity.
    """

    full_records = []
    partition_results = {}

    stability_detail = {}

    total_start = time.perf_counter()

    for algorithm_name, algorithm_key in ALGORITHMS.items():

        (
            communities,
            summary_df,
            membership_df,
            stats,
            loaded_from_cache,
        ) = load_or_compute_algorithm(
            graph=graph,
            current_dataset_key=current_dataset_key,
            current_dataset_name=current_dataset_name,
            algorithm_key=algorithm_key,
        )

        partition_results[
            algorithm_key
        ] = {
            "algorithm_name": stats[
                "Algorithm"
            ],
            "communities": communities,
        }

        balance_metrics = (
            CommunityAnalyzer.community_balance_metrics(
                communities
            )
        )

        stability_nmi = None
        stability_ari = None
        modularity_variation = None
        count_variation = None

        if run_stability_testing:

            with st.spinner(
                f"Testing {algorithm_name} stability..."
            ):

                stability_result = (
                    CommunityAnalyzer.stability_analysis(
                        graph=graph,
                        algorithm=algorithm_key,
                        number_of_runs=int(
                            stability_runs
                        ),
                        base_seed=int(seed),
                        resolution=float(
                            resolution
                        ),
                    )
                )

            stability_detail[
                algorithm_name
            ] = stability_result

            stability_nmi = stability_result[
                "Mean Pairwise NMI"
            ]

            stability_ari = stability_result[
                "Mean Pairwise ARI"
            ]

            modularity_variation = stability_result[
                "Modularity Standard Deviation"
            ]

            count_variation = stability_result[
                "Community Count Standard Deviation"
            ]

        ground_truth_nmi = None
        ground_truth_ari = None

        if current_dataset_key == "karate":

            ground_truth_result = (
                CommunityAnalyzer.evaluate_karate_ground_truth(
                    graph=graph,
                    communities=communities,
                )
            )

            if ground_truth_result is not None:

                ground_truth_nmi = (
                    ground_truth_result[
                        "Ground Truth NMI"
                    ]
                )

                ground_truth_ari = (
                    ground_truth_result[
                        "Ground Truth ARI"
                    ]
                )

        full_records.append(
            {
                "Dataset": current_dataset_name,
                "Algorithm": stats["Algorithm"],

                "Communities": stats[
                    "Number of Communities"
                ],

                "Modularity": stats[
                    "Modularity"
                ],

                "Runtime (seconds)": stats[
                    "Runtime Seconds"
                ],

                "Largest Community": stats[
                    "Largest Community"
                ],

                "Smallest Community": stats[
                    "Smallest Community"
                ],

                "Average Community Size": stats[
                    "Average Community Size"
                ],

                "Community Size CV": (
                    balance_metrics[
                        "Community Size CV"
                    ]
                ),

                "Size Balance Entropy": (
                    balance_metrics[
                        "Normalised Size Entropy"
                    ]
                ),

                "Stability NMI": stability_nmi,

                "Stability ARI": stability_ari,

                "Modularity Variation": (
                    modularity_variation
                ),

                "Community Count Variation": (
                    count_variation
                ),

                "Ground Truth NMI": (
                    ground_truth_nmi
                ),

                "Ground Truth ARI": (
                    ground_truth_ari
                ),
            }
        )

    comparison_df = pd.DataFrame(
        full_records
    )

    agreement_df = (
        CommunityAnalyzer.agreement_dataframe(
            partition_results
        )
    )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    st.success(
        f"✅ Algorithms compared on "
        f"{current_dataset_name}."
    )

    st.caption(
        f"Total processing time: "
        f"{total_elapsed:.3f} seconds"
    )

    # =================================================
    # DEFINITION OF BETTER
    # =================================================

    st.subheader("🎯 What Does Better Mean?")

    criteria_df = pd.DataFrame(
        [
            {
                "Criterion": "Community separation",
                "Measurement": "Modularity",
                "Better Result": "Higher",
                "Meaning": (
                    "Stronger internal connections and "
                    "weaker connections between communities"
                ),
            },
            {
                "Criterion": "Runtime efficiency",
                "Measurement": "Runtime",
                "Better Result": "Lower",
                "Meaning": (
                    "Less computational time"
                ),
            },
            {
                "Criterion": "Stability",
                "Measurement": "Repeated-run NMI and ARI",
                "Better Result": "Higher",
                "Meaning": (
                    "More consistent community assignments"
                ),
            },
            {
                "Criterion": "Size balance",
                "Measurement": "Community-size CV",
                "Better Result": "Lower",
                "Meaning": (
                    "Less inequality between community sizes"
                ),
            },
            {
                "Criterion": "External validity",
                "Measurement": "Ground-truth NMI and ARI",
                "Better Result": "Higher",
                "Meaning": (
                    "Closer agreement with known groups"
                ),
            },
        ]
    )

    st.dataframe(
        criteria_df,
        width="stretch",
        hide_index=True,
    )

    # =================================================
    # FULL COMPARISON
    # =================================================

    st.subheader(
        f"📊 Full Comparison: "
        f"{current_dataset_name}"
    )

    st.dataframe(
        comparison_df,
        width="stretch",
        hide_index=True,
    )
    ####################################################
    # =====================================================
# VISUAL COMPARISON OF ALGORITHM PARTITIONS
# =====================================================

    st.subheader(
        "🕸️ Visual Comparison of Community Partitions"
    )

    st.caption(
        "All algorithms use the same node positions. "
        "Therefore, differences in node colour represent "
        "differences in community assignment rather than "
        "changes in the network layout."
    )

    # Use an undirected graph for a clearer community view
    comparison_visual_graph = (
        graph.to_undirected()
        if graph.is_directed()
        else graph.copy()
    )

    # Limit the graph before calculating shared positions
    comparison_max_nodes = 300

    if (
        comparison_visual_graph.number_of_nodes()
        > comparison_max_nodes
    ):

        ranked_nodes = sorted(
            comparison_visual_graph.degree(),
            key=lambda item: item[1],
            reverse=True,
        )

        comparison_nodes = [
            node
            for node, degree in ranked_nodes[
                :comparison_max_nodes
            ]
        ]

        comparison_visual_graph = (
            comparison_visual_graph.subgraph(
                comparison_nodes
            ).copy()
        )

        st.warning(
            f"The complete network is too large for a clear "
            f"side-by-side view. The "
            f"{comparison_max_nodes} highest-degree nodes "
            f"are displayed."
        )

    # Calculate one layout and reuse it for every algorithm
    shared_positions = nx.spring_layout(
        comparison_visual_graph,
        seed=42,
        iterations=50,
    )

    # Load node importance once
        # Calculate node importance directly for the displayed graph.
    # This avoids cache-path and argument-order problems.
    visual_importance_scores = nx.degree_centrality(
        comparison_visual_graph
    )

    visual_columns = st.columns(
        len(partition_results)
    )

    visible_nodes = set(
        comparison_visual_graph.nodes()
    )

    for column, (
        algorithm_key,
        algorithm_result,
    ) in zip(
        visual_columns,
        partition_results.items(),
    ):

        filtered_communities = []

        for community in algorithm_result[
            "communities"
        ]:

            visible_community_nodes = (
                set(community)
                & visible_nodes
            )

            if visible_community_nodes:

                filtered_communities.append(
                    visible_community_nodes
                )

        with column:

            st.markdown(
                f"#### "
                f"{algorithm_result['algorithm_name']}"
            )

            figure = create_community_figure(
                graph=comparison_visual_graph,
                communities=filtered_communities,
                importance_scores=(
                    visual_importance_scores
                ),
                algorithm_name=(
                    algorithm_result[
                        "algorithm_name"
                    ]
                ),
                current_dataset_name=(
                    current_dataset_name
                ),
                current_centrality_measure=(
                    "Degree Centrality"
                ),
                max_visual_nodes=(
                    comparison_max_nodes
                ),
                positions=shared_positions,
            )

            figure.update_layout(
                height=500,
                title={
                    "text": (
                        algorithm_result[
                            "algorithm_name"
                        ]
                    ),
                    "font": {
                        "size": 16,
                    },
                },
                margin={
                    "l": 5,
                    "r": 5,
                    "t": 45,
                    "b": 5,
                },
            )

            st.plotly_chart(
                figure,
                width="stretch",
                key=(
                    "partition_visual_"
                    f"{current_dataset_key}_"
                    f"{algorithm_key}"
                ),
            )

        ###################################3333
    # =================================================
    # WINNERS BY CRITERION
    # =================================================

    quality_winner = comparison_df.sort_values(
        by="Modularity",
        ascending=False,
    ).iloc[0]

    speed_winner = comparison_df.sort_values(
        by="Runtime (seconds)",
        ascending=True,
    ).iloc[0]

    balance_winner = comparison_df.sort_values(
        by="Community Size CV",
        ascending=True,
    ).iloc[0]

    winner_col_1, winner_col_2, winner_col_3 = (
        st.columns(3)
    )

    winner_col_1.metric(
        "Best Community Separation",
        quality_winner["Algorithm"],
        f'{quality_winner["Modularity"]:.4f}',
    )

    winner_col_2.metric(
        "Fastest",
        speed_winner["Algorithm"],
        (
            f'{speed_winner["Runtime (seconds)"]:.4f}s'
        ),
    )

    winner_col_3.metric(
        "Most Balanced Sizes",
        balance_winner["Algorithm"],
        (
            f'CV {balance_winner["Community Size CV"]:.4f}'
        ),
    )

    if (
        run_stability_testing
        and comparison_df[
            "Stability NMI"
        ].notna().any()
    ):

        stability_winner = comparison_df.sort_values(
            by="Stability NMI",
            ascending=False,
        ).iloc[0]

        st.metric(
            "Most Stable Algorithm",
            stability_winner["Algorithm"],
            (
                f'NMI '
                f'{stability_winner["Stability NMI"]:.4f}'
            ),
        )

    if (
        current_dataset_key == "karate"
        and comparison_df[
            "Ground Truth NMI"
        ].notna().any()
    ):

        truth_winner = comparison_df.sort_values(
            by="Ground Truth NMI",
            ascending=False,
        ).iloc[0]

        st.metric(
            "Best Agreement With Actual Karate Split",
            truth_winner["Algorithm"],
            (
                f'NMI '
                f'{truth_winner["Ground Truth NMI"]:.4f}'
            ),
        )

    # =================================================
    # QUALITY AND EFFICIENCY CHARTS
    # =================================================

    chart_col_1, chart_col_2 = st.columns(2)

    with chart_col_1:

        modularity_figure = px.bar(
            comparison_df,
            x="Algorithm",
            y="Modularity",
            title="Community Separation Quality",
            text_auto=".4f",
        )

        st.plotly_chart(
            modularity_figure,
            width="stretch",
        )

    with chart_col_2:

        runtime_figure = px.bar(
            comparison_df,
            x="Algorithm",
            y="Runtime (seconds)",
            title="Runtime Efficiency",
            text_auto=".4f",
        )

        st.plotly_chart(
            runtime_figure,
            width="stretch",
        )

    # =================================================
    # STABILITY CHARTS
    # =================================================

    if run_stability_testing:

        st.subheader(
            "🔁 Repeated-Run Stability"
        )

        stability_chart_df = comparison_df[
            [
                "Algorithm",
                "Stability NMI",
                "Stability ARI",
                "Modularity Variation",
                "Community Count Variation",
            ]
        ]

        st.dataframe(
            stability_chart_df,
            width="stretch",
            hide_index=True,
        )

        stability_figure = px.bar(
            stability_chart_df,
            x="Algorithm",
            y=[
                "Stability NMI",
                "Stability ARI",
            ],
            barmode="group",
            title=(
                "Consistency of Community Assignments "
                "Across Seeds"
            ),
        )

        st.plotly_chart(
            stability_figure,
            width="stretch",
        )

        with st.expander(
            "View Individual Stability Runs"
        ):

            for (
                algorithm_name,
                stability_result,
            ) in stability_detail.items():

                st.markdown(
                    f"#### {algorithm_name}"
                )

                st.dataframe(
                    stability_result[
                        "Run Results"
                    ],
                    width="stretch",
                    hide_index=True,
                )

    

    # =================================================
    # GROUND-TRUTH RESULTS
    # =================================================

    if (
        current_dataset_key == "karate"
        and comparison_df[
            "Ground Truth NMI"
        ].notna().any()
    ):

        st.subheader(
            "🥋 Agreement With Observed Karate Club Split"
        )

        ground_truth_df = comparison_df[
            [
                "Algorithm",
                "Ground Truth NMI",
                "Ground Truth ARI",
                "Communities",
            ]
        ]

        st.dataframe(
            ground_truth_df,
            width="stretch",
            hide_index=True,
        )

        ground_truth_figure = px.bar(
            ground_truth_df,
            x="Algorithm",
            y=[
                "Ground Truth NMI",
                "Ground Truth ARI",
            ],
            barmode="group",
            title=(
                "Agreement With Known Club Membership"
            ),
        )

        st.plotly_chart(
            ground_truth_figure,
            width="stretch",
        )

    # =================================================
    # PARTITION AGREEMENT
    # =================================================

    st.subheader(
        "🤝 Agreement Between Different Algorithms"
    )

    st.dataframe(
        agreement_df,
        width="stretch",
        hide_index=True,
    )

    # =================================================
    # WHY ANALYSIS
    # =================================================

    st.subheader(
        "🧠 Why Did One Algorithm Perform Better?"
    )

    interpretation = (
        generate_algorithm_interpretation(
            comparison_df=comparison_df,
            dataset_name=current_dataset_name,
        )
    )

    for paragraph in interpretation:

        st.write(paragraph)

    # =================================================
    # FINAL CONCLUSION
    # =================================================

    st.subheader(
        "✅ Evidence-Based Conclusion"
    )

    st.info(
        "There is no universal winner. The preferred algorithm "
        "depends on the study objective. Use modularity when the "
        "priority is community separation, runtime for scalability, "
        "repeated-run NMI and ARI for stability, and ground-truth "
        "agreement when known labels are available."
    )

# =====================================================
# ONE ALGORITHM ACROSS MULTIPLE DATASETS
# =====================================================

def display_dataset_comparison(
    algorithm_key: str,
    algorithm_name: str,
) -> None:
    """
    Apply the same community-detection algorithm to all
    available datasets and compare its behaviour.

    Comparison criteria include:
    - Modularity
    - Runtime
    - Number of communities
    - Community-size balance
    - Repeated-run stability
    - Graph structural characteristics
    """

    # =================================================
    # ADVANCED EVALUATION SETTINGS
    # These controls must appear before they are used
    # =================================================

    st.subheader("Advanced Evaluation")

    evaluation_col_1, evaluation_col_2 = st.columns(2)

    with evaluation_col_1:

        run_stability_testing = st.checkbox(
            "Run repeated-seed stability analysis",
            value=True,
            key=(
                f"cross_dataset_stability_"
                f"{algorithm_key}"
            ),
            help=(
                "Runs the selected algorithm several times "
                "using different seeds and measures whether "
                "it returns similar communities."
            ),
        )

    with evaluation_col_2:

        stability_runs = st.number_input(
            "Number of Stability Runs",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
            key=(
                f"cross_dataset_stability_runs_"
                f"{algorithm_key}"
            ),
            disabled=not run_stability_testing,
            help=(
                "Use three runs initially for larger datasets. "
                "Five runs provide a stronger stability analysis."
            ),
        )

    # comparison_records must be a list because one result
    # dictionary will be appended for each dataset.
    comparison_records = []

    unavailable_datasets = []

    stability_details = {}

    total_start = time.perf_counter()

    # =================================================
    # PROCESS EACH DATASET
    # =================================================

    for (
        current_dataset_name,
        current_dataset_key,
    ) in DATASETS.items():

        graph = load_processed_graph(
            current_dataset_key=current_dataset_key,
            current_dataset_name=current_dataset_name,
        )

        if graph is None:

            unavailable_datasets.append(
                current_dataset_name
            )

            continue

        # ---------------------------------------------
        # LOAD OR COMPUTE COMMUNITY RESULTS
        # ---------------------------------------------

        (
            communities,
            summary_df,
            membership_df,
            stats,
            loaded_from_cache,
        ) = load_or_compute_algorithm(
            graph=graph,
            current_dataset_key=current_dataset_key,
            current_dataset_name=current_dataset_name,
            algorithm_key=algorithm_key,
        )

        # ---------------------------------------------
        # PREPARE GRAPH FOR CONSISTENT MEASUREMENTS
        # ---------------------------------------------

        prepared_graph = (
            CommunityAnalyzer.prepare_graph(
                graph
            )
        )

        # ---------------------------------------------
        # COMMUNITY-SIZE BALANCE
        # Always calculate before using
        # ---------------------------------------------

        balance_metrics = (
            CommunityAnalyzer.community_balance_metrics(
                communities
            )
        )

        # ---------------------------------------------
        # DEFAULT STABILITY VALUES
        # These must exist even when testing is disabled
        # ---------------------------------------------

        stability_nmi = None
        stability_ari = None
        modularity_variation = None
        community_count_variation = None
        mean_stability_runtime = None

        # ---------------------------------------------
        # REPEATED-RUN STABILITY ANALYSIS
        # ---------------------------------------------

        if run_stability_testing:

            with st.spinner(
                f"Testing {algorithm_name} stability on "
                f"{current_dataset_name}..."
            ):

                stability_result = (
                    CommunityAnalyzer.stability_analysis(
                        graph=graph,
                        algorithm=algorithm_key,
                        number_of_runs=int(
                            stability_runs
                        ),
                        base_seed=int(seed),
                        resolution=float(
                            resolution
                        ),
                    )
                )

            stability_nmi = stability_result.get(
                "Mean Pairwise NMI"
            )

            stability_ari = stability_result.get(
                "Mean Pairwise ARI"
            )

            modularity_variation = (
                stability_result.get(
                    "Modularity Standard Deviation"
                )
            )

            community_count_variation = (
                stability_result.get(
                    "Community Count Standard Deviation"
                )
            )

            mean_stability_runtime = (
                stability_result.get(
                    "Mean Runtime Seconds"
                )
            )

            stability_details[
                current_dataset_name
            ] = stability_result

        # ---------------------------------------------
        # SAFE GRAPH STATISTICS
        # ---------------------------------------------

        number_of_nodes = (
            prepared_graph.number_of_nodes()
        )

        number_of_edges = (
            prepared_graph.number_of_edges()
        )

        density = (
            nx.density(prepared_graph)
            if number_of_nodes > 1
            else 0.0
        )

        average_degree = (
            sum(
                dict(
                    prepared_graph.degree()
                ).values()
            )
            / number_of_nodes
            if number_of_nodes > 0
            else 0.0
        )

        undirected_graph = (
            prepared_graph.to_undirected()
            if prepared_graph.is_directed()
            else prepared_graph
        )

        clustering_coefficient = (
            nx.average_clustering(
                undirected_graph
            )
            if number_of_nodes > 0
            else 0.0
        )

        # ---------------------------------------------
        # SAFE COMMUNITY STATISTICS
        # ---------------------------------------------

        community_sizes = [
            len(community)
            for community in communities
        ]

        number_of_communities = len(
            communities
        )

        largest_community = (
            max(community_sizes)
            if community_sizes
            else 0
        )

        smallest_community = (
            min(community_sizes)
            if community_sizes
            else 0
        )

        average_community_size = (
            sum(community_sizes)
            / number_of_communities
            if number_of_communities > 0
            else 0.0
        )

        modularity_value = stats.get(
            "Modularity"
        )

        if modularity_value is None:

            modularity_value = (
                CommunityAnalyzer.calculate_modularity(
                    graph,
                    communities,
                )
            )

        # ---------------------------------------------
        # ADD ONE RECORD FOR THIS DATASET
        # ---------------------------------------------

        comparison_records.append(
            {
                "Dataset": current_dataset_name,

                "Algorithm": stats.get(
                    "Algorithm",
                    algorithm_name,
                ),

                "Nodes": stats.get(
                    "Nodes",
                    number_of_nodes,
                ),

                "Edges": stats.get(
                    "Edges",
                    number_of_edges,
                ),

                "Density": stats.get(
                    "Density",
                    round(
                        density,
                        8,
                    ),
                ),

                "Average Degree": stats.get(
                    "Average Degree",
                    round(
                        average_degree,
                        6,
                    ),
                ),

                "Clustering Coefficient": stats.get(
                    "Clustering Coefficient",
                    round(
                        clustering_coefficient,
                        6,
                    ),
                ),

                "Communities": stats.get(
                    "Number of Communities",
                    number_of_communities,
                ),

                "Modularity": round(
                    float(
                        modularity_value
                    ),
                    6,
                ),

                "Runtime (seconds)": stats.get(
                    "Runtime Seconds",
                    0.0,
                ),

                "Largest Community": stats.get(
                    "Largest Community",
                    largest_community,
                ),

                "Smallest Community": stats.get(
                    "Smallest Community",
                    smallest_community,
                ),

                "Average Community Size": stats.get(
                    "Average Community Size",
                    round(
                        average_community_size,
                        3,
                    ),
                ),

                "Community Size CV": (
                    balance_metrics.get(
                        "Community Size CV",
                        0.0,
                    )
                ),

                "Size Balance Entropy": (
                    balance_metrics.get(
                        "Normalised Size Entropy",
                        0.0,
                    )
                ),

                "Community Size Standard Deviation": (
                    balance_metrics.get(
                        "Community Size Standard Deviation",
                        0.0,
                    )
                ),

                "Stability NMI": stability_nmi,

                "Stability ARI": stability_ari,

                "Modularity Variation": (
                    modularity_variation
                ),

                "Community Count Variation": (
                    community_count_variation
                ),

                "Mean Stability Runtime": (
                    mean_stability_runtime
                ),

                "Loaded From Cache": (
                    loaded_from_cache
                ),
            }
        )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    # =================================================
    # CHECK WHETHER DATA EXISTS
    # =================================================

    if not comparison_records:

        st.error(
            "No processed graph caches were found. "
            "Run the datasets from the Home page first."
        )

        return

    comparison_df = pd.DataFrame(
        comparison_records
    )

    # Convert numeric columns safely
    numeric_columns = [
        "Nodes",
        "Edges",
        "Density",
        "Average Degree",
        "Clustering Coefficient",
        "Communities",
        "Modularity",
        "Runtime (seconds)",
        "Largest Community",
        "Smallest Community",
        "Average Community Size",
        "Community Size CV",
        "Size Balance Entropy",
        "Community Size Standard Deviation",
        "Stability NMI",
        "Stability ARI",
        "Modularity Variation",
        "Community Count Variation",
        "Mean Stability Runtime",
    ]

    for column in numeric_columns:

        if column in comparison_df.columns:

            comparison_df[column] = (
                pd.to_numeric(
                    comparison_df[column],
                    errors="coerce",
                )
            )

    # =================================================
    # STATUS
    # =================================================

    st.success(
        f"✅ {algorithm_name} compared across "
        f"{len(comparison_df)} datasets."
    )

    st.caption(
        f"Total processing time: "
        f"{total_elapsed:.3f} seconds"
    )

    if unavailable_datasets:

        st.warning(
            "The following datasets were skipped because "
            "their processed graph caches were unavailable: "
            + ", ".join(
                unavailable_datasets
            )
        )

    # =================================================
    # DEFINE WHAT 'BETTER' MEANS
    # =================================================

    st.subheader(
        "🎯 What Does Better Mean?"
    )

    criteria_records = [
        {
            "Criterion": "Community separation",
            "Measure": "Modularity",
            "Preferred Result": "Higher",
            "Meaning": (
                "More edges inside communities and fewer "
                "edges between different communities."
            ),
        },
        {
            "Criterion": "Efficiency",
            "Measure": "Runtime",
            "Preferred Result": "Lower",
            "Meaning": (
                "The algorithm completes the analysis faster."
            ),
        },
        {
            "Criterion": "Stability",
            "Measure": "Repeated-run NMI and ARI",
            "Preferred Result": "Higher",
            "Meaning": (
                "Different runs produce similar community "
                "assignments."
            ),
        },
        {
            "Criterion": "Size balance",
            "Measure": "Community Size CV",
            "Preferred Result": "Lower",
            "Meaning": (
                "Community sizes are more evenly distributed."
            ),
        },
    ]

    st.dataframe(
        pd.DataFrame(
            criteria_records
        ),
        width="stretch",
        hide_index=True,
    )

    # =================================================
    # FULL COMPARISON TABLE
    # =================================================

    st.subheader(
        f"📊 {algorithm_name} Across Datasets"
    )

    display_columns = [
        "Dataset",
        "Nodes",
        "Edges",
        "Density",
        "Average Degree",
        "Clustering Coefficient",
        "Communities",
        "Modularity",
        "Runtime (seconds)",
        "Largest Community",
        "Average Community Size",
        "Community Size CV",
        "Stability NMI",
        "Stability ARI",
        "Modularity Variation",
    ]

    available_display_columns = [
        column
        for column in display_columns
        if column in comparison_df.columns
    ]

    st.dataframe(
        comparison_df[
            available_display_columns
        ],
        width="stretch",
        hide_index=True,
    )

    # =================================================
    # BEST RESULTS
    # =================================================

    highest_modularity = (
        comparison_df.dropna(
            subset=["Modularity"]
        )
        .sort_values(
            by="Modularity",
            ascending=False,
        )
        .iloc[0]
    )

    fastest_dataset = (
        comparison_df.dropna(
            subset=["Runtime (seconds)"]
        )
        .sort_values(
            by="Runtime (seconds)",
            ascending=True,
        )
        .iloc[0]
    )

    most_communities = (
        comparison_df.dropna(
            subset=["Communities"]
        )
        .sort_values(
            by="Communities",
            ascending=False,
        )
        .iloc[0]
    )

    most_balanced = (
        comparison_df.dropna(
            subset=["Community Size CV"]
        )
        .sort_values(
            by="Community Size CV",
            ascending=True,
        )
        .iloc[0]
    )

    metric_col_1, metric_col_2 = st.columns(
        2
    )

    metric_col_3, metric_col_4 = st.columns(
        2
    )

    metric_col_1.metric(
        "Highest Modularity Dataset",
        highest_modularity["Dataset"],
        (
            f'{highest_modularity["Modularity"]:.4f}'
        ),
    )

    metric_col_2.metric(
        "Fastest Dataset",
        fastest_dataset["Dataset"],
        (
            f'{fastest_dataset["Runtime (seconds)"]:.4f}s'
        ),
    )

    metric_col_3.metric(
        "Most Communities",
        most_communities["Dataset"],
        int(
            most_communities["Communities"]
        ),
    )

    metric_col_4.metric(
        "Most Balanced Community Sizes",
        most_balanced["Dataset"],
        (
            f'CV '
            f'{most_balanced["Community Size CV"]:.4f}'
        ),
    )

    if (
        run_stability_testing
        and comparison_df[
            "Stability NMI"
        ].notna().any()
    ):

        most_stable = (
            comparison_df.dropna(
                subset=["Stability NMI"]
            )
            .sort_values(
                by="Stability NMI",
                ascending=False,
            )
            .iloc[0]
        )

        st.metric(
            "Most Stable Dataset",
            most_stable["Dataset"],
            (
                f'NMI '
                f'{most_stable["Stability NMI"]:.4f}'
            ),
        )

    # =================================================
    # MODULARITY AND RUNTIME
    # =================================================

    chart_col_1, chart_col_2 = st.columns(2)

    with chart_col_1:

        modularity_figure = px.bar(
            comparison_df,
            x="Dataset",
            y="Modularity",
            title=(
                f"{algorithm_name}: "
                "Modularity Across Datasets"
            ),
            text_auto=".4f",
        )

        st.plotly_chart(
            modularity_figure,
            width="stretch",
        )

    with chart_col_2:

        runtime_figure = px.bar(
            comparison_df,
            x="Dataset",
            y="Runtime (seconds)",
            title=(
                f"{algorithm_name}: "
                "Runtime Across Datasets"
            ),
            text_auto=".4f",
        )

        st.plotly_chart(
            runtime_figure,
            width="stretch",
        )

    # =================================================
    # COMMUNITY COUNT AND BALANCE
    # =================================================

    chart_col_3, chart_col_4 = st.columns(2)

    with chart_col_3:

        community_count_figure = px.bar(
            comparison_df,
            x="Dataset",
            y="Communities",
            title=(
                f"{algorithm_name}: "
                "Communities Detected"
            ),
            text_auto=True,
        )

        st.plotly_chart(
            community_count_figure,
            width="stretch",
        )

    with chart_col_4:

        balance_figure = px.bar(
            comparison_df,
            x="Dataset",
            y="Community Size CV",
            title=(
                f"{algorithm_name}: "
                "Community-Size Inequality"
            ),
            text_auto=".4f",
        )

        st.plotly_chart(
            balance_figure,
            width="stretch",
        )

    st.caption(
        "A lower Community Size CV indicates more similarly "
        "sized communities. This does not automatically mean "
        "the communities are more accurate because real networks "
        "may naturally contain unequal groups."
    )

    # =================================================
    # STRUCTURAL CHARACTERISTICS
    # =================================================

    st.subheader(
        "🔗 Dataset Structure and Algorithm Behaviour"
    )

    structure_columns = [
        "Dataset",
        "Nodes",
        "Edges",
        "Density",
        "Average Degree",
        "Clustering Coefficient",
        "Communities",
        "Modularity",
        "Runtime (seconds)",
    ]

    st.dataframe(
        comparison_df[
            structure_columns
        ],
        width="stretch",
        hide_index=True,
    )

    density_modularity_figure = px.scatter(
        comparison_df,
        x="Density",
        y="Modularity",
        size="Nodes",
        text="Dataset",
        hover_data=[
            "Edges",
            "Communities",
            "Average Degree",
            "Clustering Coefficient",
        ],
        title=(
            "Relationship Between Graph Density "
            "and Modularity"
        ),
    )

    density_modularity_figure.update_traces(
        textposition="top center"
    )

    st.plotly_chart(
        density_modularity_figure,
        width="stretch",
    )

    # =================================================
    # STABILITY ACROSS DATASETS
    # =================================================

    if (
        run_stability_testing
        and comparison_df[
            "Stability NMI"
        ].notna().any()
    ):

        st.subheader(
            "🔁 Stability Across Datasets"
        )

        stability_dataset_df = (
            comparison_df[
                [
                    "Dataset",
                    "Stability NMI",
                    "Stability ARI",
                    "Modularity Variation",
                    "Community Count Variation",
                ]
            ]
        )

        st.dataframe(
            stability_dataset_df,
            width="stretch",
            hide_index=True,
        )

        stability_dataset_figure = px.bar(
            stability_dataset_df,
            x="Dataset",
            y=[
                "Stability NMI",
                "Stability ARI",
            ],
            barmode="group",
            title=(
                f"{algorithm_name}: "
                "Stability Across Datasets"
            ),
        )

        st.plotly_chart(
            stability_dataset_figure,
            width="stretch",
        )

        with st.expander(
            "View Individual Stability Runs"
        ):

            for (
                current_dataset_name,
                stability_result,
            ) in stability_details.items():

                st.markdown(
                    f"#### {current_dataset_name}"
                )

                run_results_df = (
                    stability_result.get(
                        "Run Results"
                    )
                )

                if run_results_df is not None:

                    st.dataframe(
                        run_results_df,
                        width="stretch",
                        hide_index=True,
                    )

    # =================================================
    # WHY ANALYSIS
    # =================================================

    st.subheader(
        "🧠 Why Did the Algorithm Perform "
        "Differently Across Datasets?"
    )

    dataset_interpretation = (
        generate_dataset_interpretation(
            comparison_df=comparison_df,
            algorithm_name=algorithm_name,
        )
    )

    for paragraph in dataset_interpretation:

        st.write(paragraph)

    # Additional evidence-based explanation

    highest_modularity_dataset = (
        highest_modularity["Dataset"]
    )

    lowest_modularity = (
        comparison_df.dropna(
            subset=["Modularity"]
        )
        .sort_values(
            by="Modularity",
            ascending=True,
        )
        .iloc[0]
    )

    st.write(
        f"Under the community-separation criterion, "
        f"**{highest_modularity_dataset}** produced the best "
        f"result because it had the highest modularity score. "
        f"This means {algorithm_name} found relatively stronger "
        "connections within communities and weaker connections "
        "between communities on this dataset."
    )

    st.write(
        f"By comparison, **{lowest_modularity['Dataset']}** "
        f"had the lowest modularity score. This may indicate "
        "weaker community boundaries, more connections between "
        "groups, or a network structure that is less suited to "
        f"the optimisation strategy used by {algorithm_name}."
    )

    st.write(
        f"**{fastest_dataset['Dataset']}** was fastest, but this "
        "does not mean it had the best community structure. "
        "Runtime is strongly influenced by the number of nodes, "
        "number of edges and graph density."
    )

    if (
        run_stability_testing
        and comparison_df[
            "Stability NMI"
        ].notna().any()
    ):

        st.write(
            f"**{most_stable['Dataset']}** produced the most "
            "consistent repeated results. Its higher stability "
            "NMI indicates that changing the random seed caused "
            "less variation in node-community assignments."
        )

    # =================================================
    # CONCLUSION
    # =================================================

    st.subheader(
        "✅ Cross-Dataset Conclusion"
    )

    st.info(
        "There is no single universal meaning of better. "
        "Higher modularity means stronger community separation, "
        "lower runtime means better computational efficiency, "
        "higher repeated-run NMI and ARI mean greater stability, "
        "and lower community-size CV means more evenly sized "
        "communities. The final judgement should consider these "
        "criteria together with graph size, density, clustering "
        "and the research objective."
    )
# =====================================================
# ALGORITHM CHARACTERISTICS
# =====================================================

def algorithm_reason(
    algorithm_name: str,
) -> str:
    """
    Explain the important behavioural characteristics of
    each community-detection algorithm.
    """

    explanations = {
        "Greedy Modularity": (
            "Greedy Modularity repeatedly selects the merge "
            "that gives the greatest immediate modularity "
            "increase. It is deterministic and reproducible, "
            "but an early merge cannot later be reversed. "
            "This can prevent it from finding a stronger final "
            "partition on networks with complex or hierarchical "
            "community structure."
        ),

        "Louvain": (
            "Louvain performs multilevel optimisation. It first "
            "moves individual nodes between communities and then "
            "compresses communities into higher-level nodes. "
            "This repeated refinement can identify stronger "
            "multilevel partitions, although the output may "
            "change slightly with the random seed and resolution."
        ),

        "Label Propagation": (
            "Label Propagation allows nodes to adopt labels that "
            "are common among their neighbours. It avoids expensive "
            "global optimisation and is therefore usually efficient. "
            "However, its result can depend on node-processing order "
            "and may be less stable when community boundaries are weak."
        ),
    }

    return explanations.get(
        algorithm_name,
        "No algorithm explanation is available.",
    )


# =====================================================
# EVIDENCE-BASED ALGORITHM INTERPRETATION
# =====================================================

def generate_algorithm_interpretation(
    comparison_df: pd.DataFrame,
    dataset_name: str,
) -> List[str]:
    """
    Explain what performed better and why using the measured
    results rather than modularity alone.
    """

    explanations = []

    highest_modularity = comparison_df.sort_values(
        by="Modularity",
        ascending=False,
    ).iloc[0]

    fastest = comparison_df.sort_values(
        by="Runtime (seconds)",
        ascending=True,
    ).iloc[0]

    most_stable = None

    if (
        "Stability NMI" in comparison_df.columns
        and comparison_df[
            "Stability NMI"
        ].notna().any()
    ):
        most_stable = comparison_df.sort_values(
            by="Stability NMI",
            ascending=False,
        ).iloc[0]

    most_balanced = comparison_df.sort_values(
        by="Community Size CV",
        ascending=True,
    ).iloc[0]

    explanations.append(
        f"On {dataset_name}, "
        f"{highest_modularity['Algorithm']} produced the "
        f"highest modularity score of "
        f"{highest_modularity['Modularity']:.4f}. "
        "It therefore performed better under the community-"
        "separation criterion: its partition contained relatively "
        "more within-community connections and fewer connections "
        "between communities."
    )

    explanations.append(
        algorithm_reason(
            highest_modularity["Algorithm"]
        )
    )

    explanations.append(
        f"{fastest['Algorithm']} was the fastest algorithm, "
        f"requiring approximately "
        f"{fastest['Runtime (seconds)']:.4f} seconds. "
        "It performed better under the computational-efficiency "
        "criterion, but this does not prove that it produced the "
        "most meaningful communities."
    )

    if most_stable is not None:

        explanations.append(
            f"{most_stable['Algorithm']} was the most stable "
            f"across repeated runs, with a mean pairwise NMI of "
            f"{most_stable['Stability NMI']:.4f}. "
            "This means its repeated runs produced the most "
            "similar node assignments."
        )

    explanations.append(
        f"{most_balanced['Algorithm']} produced the most evenly "
        f"sized partition, with the lowest community-size "
        f"coefficient of variation of "
        f"{most_balanced['Community Size CV']:.4f}. "
        "This indicates lower size inequality, although evenly "
        "sized communities are not automatically more correct."
    )

    if (
        "Ground Truth NMI"
        in comparison_df.columns
        and comparison_df[
            "Ground Truth NMI"
        ].notna().any()
    ):

        best_ground_truth = comparison_df.sort_values(
            by="Ground Truth NMI",
            ascending=False,
        ).iloc[0]

        explanations.append(
            f"For Karate Club, "
            f"{best_ground_truth['Algorithm']} showed the "
            f"strongest agreement with the observed club split, "
            f"with NMI "
            f"{best_ground_truth['Ground Truth NMI']:.4f} "
            f"and ARI "
            f"{best_ground_truth['Ground Truth ARI']:.4f}. "
            "It therefore performed better under the external-"
            "validity criterion."
        )

    return explanations


# =====================================================
# CROSS-DATASET INTERPRETATION
# =====================================================

def generate_dataset_interpretation(
    comparison_df: pd.DataFrame,
    algorithm_name: str,
) -> List[str]:
    """
    Explain why the selected algorithm behaves differently
    across datasets.
    """

    explanations = []

    highest_modularity = comparison_df.sort_values(
        by="Modularity",
        ascending=False,
    ).iloc[0]

    lowest_modularity = comparison_df.sort_values(
        by="Modularity",
        ascending=True,
    ).iloc[0]

    fastest = comparison_df.sort_values(
        by="Runtime (seconds)",
        ascending=True,
    ).iloc[0]

    slowest = comparison_df.sort_values(
        by="Runtime (seconds)",
        ascending=False,
    ).iloc[0]

    explanations.append(
        f"{algorithm_name} achieved its highest modularity on "
        f"{highest_modularity['Dataset']} "
        f"({highest_modularity['Modularity']:.4f}) and its "
        f"lowest modularity on "
        f"{lowest_modularity['Dataset']} "
        f"({lowest_modularity['Modularity']:.4f}). "
        "This suggests that the first dataset contains a clearer "
        "community structure under this algorithm's optimisation "
        "strategy."
    )

    explanations.append(
        f"The algorithm ran fastest on "
        f"{fastest['Dataset']} and slowest on "
        f"{slowest['Dataset']}. The slowest dataset contained "
        f"{int(slowest['Nodes'])} nodes and "
        f"{int(slowest['Edges'])} edges, so the algorithm had "
        "more network structure to examine."
    )

    density_winner = float(
        highest_modularity["Density"]
    )

    density_loser = float(
        lowest_modularity["Density"]
    )

    if density_winner > density_loser:

        density_reason = (
            "The dataset with higher modularity was also denser. "
            "Its greater concentration of edges may have provided "
            "stronger internal connectivity within the detected groups."
        )

    elif density_winner < density_loser:

        density_reason = (
            "The dataset with higher modularity was less dense. "
            "This indicates that overall density alone did not create "
            "the better partition; the placement of edges and the "
            "strength of community boundaries were more important."
        )

    else:

        density_reason = (
            "The compared datasets had similar density, so the "
            "difference is more likely connected to edge placement, "
            "degree distribution and clustering structure."
        )

    explanations.append(
        density_reason
    )

    explanations.append(
        algorithm_reason(
            algorithm_name
        )
    )

    return explanations

    current_centrality_measure = st.selectbox(
    "Node Size Measure",
    options=[
        "Degree",
        "PageRank",
        "Betweenness",
        "Closeness",
    ],
    index=0,
    help=(
        "Controls node size in the community network "
        "visualisation."
    ),
)


# =====================================================
# PAGE EXECUTION
# =====================================================

if run_single_button:

    selected_graph = load_processed_graph(
        current_dataset_key=dataset_key,
        current_dataset_name=dataset_option,
    )

    if selected_graph is not None:

        display_single_algorithm(
            graph=selected_graph,
            current_dataset_key=dataset_key,
            current_dataset_name=dataset_option,
            algorithm_key=selected_algorithm,
        )


elif compare_algorithms_button:

    selected_graph = load_processed_graph(
        current_dataset_key=dataset_key,
        current_dataset_name=dataset_option,
    )

    if selected_graph is not None:

        display_algorithm_comparison(
            graph=selected_graph,
            current_dataset_key=dataset_key,
            current_dataset_name=dataset_option,
            current_centrality_measure=current_centrality_measure,
        )


elif compare_datasets_button:

    display_dataset_comparison(
        algorithm_key=selected_algorithm,
        algorithm_name=selected_algorithm_name,
    )


else:

    st.info(
        "Choose one of the three analysis options above."
    )

    st.markdown(
        """
### Available comparisons

**Run Selected Algorithm**

Runs one selected algorithm on one selected dataset and displays
its communities, modularity score, runtime, community sizes and
network visualisation.

**Compare Algorithms on Dataset**

Runs Greedy Modularity, Louvain and Label Propagation on the
selected dataset. This answers:

> Which community-detection algorithm works better for this dataset?

**Compare Algorithm Across Datasets**

Runs the selected algorithm on Karate Club, Email-Eu-Core and
Web-Google. This answers:

> On which type of dataset does this algorithm perform better?
        """
    )