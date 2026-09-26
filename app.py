"""
=========================================================
Social Network Analysis Dashboard
=========================================================
"""

import time
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from modules.export import ExportManager
from modules.loader import GraphLoader
from modules.preprocessing import GraphPreprocessor
from modules.graph_statistics import GraphStatistics
from modules.centrality import CentralityAnalyzer
from modules.correlation import CorrelationAnalyzer
from modules.visualization import GraphVisualizer
from modules.comparison import DatasetComparator
from modules.cache_manager import CacheManager
from modules.performance import PerformanceManager


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Social Network Analysis Dashboard",
    layout="wide"
)

st.title("📊 Social Network Analysis Dashboard")


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Controls")

dataset_option = st.sidebar.selectbox(
    "Select Dataset",
    (
        "Karate Club (Built-in)",
        "Email-Eu-Core",
        "Web-Google (Sampled)"
    )
)

run_button = st.sidebar.button("Run Analysis")

show_comparison = st.sidebar.checkbox(
    "Run Dataset Comparison"
)


# =====================================================
# DATASET KEYS
# =====================================================

DATASET_KEYS = {

    "Karate Club (Built-in)": "karate",

    "Email-Eu-Core": "email",

    "Web-Google (Sampled)": "web_google"

}

dataset_key = DATASET_KEYS[dataset_option]


# =====================================================
# GRAPH LOADER
# =====================================================

# =====================================================
# GRAPH LOADER WITH BENCHMARK TIMINGS
# =====================================================

def load_graph_with_timings(option):
    """
    Load a processed graph while recording timings for:
    - processed graph cache loading
    - raw graph loading
    - preprocessing
    """

    dataset_keys = {
        "Karate Club (Built-in)": "karate",
        "Email-Eu-Core": "email",
        "Web-Google (Sampled)": "web_google"
    }

    if option not in dataset_keys:
        raise ValueError("Invalid dataset.")

    key = dataset_keys[option]

    timings = {
        "raw_graph_load_time": 0.0,
        "graph_cache_load_time": 0.0,
        "preprocessing_time": 0.0,
    }

    # -------------------------------------------------
    # TRY PROCESSED GRAPH CACHE
    # -------------------------------------------------

    graph_cache_start = time.perf_counter()

    cached_graph = CacheManager.load_graph(key)

    timings["graph_cache_load_time"] = (
        time.perf_counter() - graph_cache_start
    )

    if cached_graph is not None:
        return cached_graph, timings, True

    # -------------------------------------------------
    # LOAD RAW DATASET
    # -------------------------------------------------

    raw_load_start = time.perf_counter()

    if option == "Karate Club (Built-in)":

        graph = GraphLoader.load_karate_club()

    elif option == "Email-Eu-Core":

        graph = GraphLoader.load_graph(
            "data/raw/email_eu/email-Eu-Core.txt"
        )

    elif option == "Web-Google (Sampled)":

        graph = GraphLoader.load_web_google_sample(
            max_nodes=1500
        )

    else:

        raise ValueError("Invalid dataset.")

    timings["raw_graph_load_time"] = (
        time.perf_counter() - raw_load_start
    )

    # -------------------------------------------------
    # PREPROCESS
    # -------------------------------------------------

    preprocessing_start = time.perf_counter()

    graph = GraphPreprocessor.preprocess(graph)

    timings["preprocessing_time"] = (
        time.perf_counter() - preprocessing_start
    )

    # -------------------------------------------------
    # SAVE PROCESSED GRAPH
    # -------------------------------------------------

    CacheManager.save_graph(
        key,
        graph
    )

    return graph, timings, False


def load_graph(option):
    """
    Standard graph loader used by comparison and visualisation code.
    """

    graph, _, _ = load_graph_with_timings(option)

    return graph


def ensure_graph(graph, dataset_option):
    """
    Return the existing graph or load it when required.
    """

    if graph is None:
        graph = load_graph(dataset_option)

    if graph is None:
        raise RuntimeError(
            f"Unable to load graph for dataset: {dataset_option}"
        )

    return graph


# =====================================================
# MAIN EXECUTION
# =====================================================
if run_button:

    total_start_time = time.perf_counter()

    # =================================================
    # CHECK ANALYSIS CACHE
    # =================================================

    analysis_cache_exists = (
        CacheManager.exists(
            dataset_key,
            "stats.json"
        )
        and
        CacheManager.exists(
            dataset_key,
            "centrality.csv"
        )
        and
        CacheManager.exists(
            dataset_key,
            "correlation.json"
        )
    )

    # =================================================
    # LOAD GRAPH WITH STAGE TIMINGS
    # =================================================

    (
        graph,
        graph_timings,
        graph_loaded_from_cache
    ) = load_graph_with_timings(
        dataset_option
    )

    raw_graph_load_time = graph_timings[
        "raw_graph_load_time"
    ]

    graph_cache_load_time = graph_timings[
        "graph_cache_load_time"
    ]

    preprocessing_time = graph_timings[
        "preprocessing_time"
    ]

    statistics_time = 0.0
    centrality_time = 0.0
    correlation_time = 0.0
    analysis_cache_load_time = 0.0

    # =================================================
    # LOAD ANALYSIS CACHE
    # =================================================

    if analysis_cache_exists:

        cache_start = time.perf_counter()

        stats = CacheManager.load_json(
            dataset_key,
            "stats.json"
        )

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

        analysis_cache_load_time = (
            time.perf_counter() - cache_start
        )

        run_type = "Cached Run"

        st.success(
            "⚡ Loaded analysis from disk cache."
        )

    # =================================================
    # COMPUTE ANALYSIS
    # =================================================

    else:

        st.warning(
            "First execution. Computing analysis..."
        )

        statistics_start = time.perf_counter()

        stats = GraphStatistics.compute(
            graph
        )

        statistics_time = (
            time.perf_counter() - statistics_start
        )

        centrality_start = time.perf_counter()

        centrality_df = (
            CentralityAnalyzer.compute_all(
                graph
            )
        )

        centrality_time = (
            time.perf_counter() - centrality_start
        )

        correlation_start = time.perf_counter()

        correlation_df = (
            CorrelationAnalyzer.spearman_pairwise(
                centrality_df
            )
        )

        correlation_time = (
            time.perf_counter() - correlation_start
        )

        CacheManager.save_json(
            dataset_key,
            "stats.json",
            stats
        )

        CacheManager.save_csv(
            dataset_key,
            "centrality.csv",
            centrality_df
        )

        CacheManager.save_json(
            dataset_key,
            "correlation.json",
            correlation_df.to_dict()
        )

        run_type = "First Run"

        st.success(
            "✅ Analysis computed and cached."
        )

    # =================================================
    # TOTAL TIME
    # =================================================

    total_time = (
        time.perf_counter() - total_start_time
    )

    # =================================================
    # RECORD BENCHMARK
    # =================================================

    PerformanceManager.record(
        dataset=dataset_option,
        run_type=run_type,
        raw_graph_load_time=raw_graph_load_time,
        graph_cache_load_time=graph_cache_load_time,
        preprocessing_time=preprocessing_time,
        statistics_time=statistics_time,
        centrality_time=centrality_time,
        correlation_time=correlation_time,
        analysis_cache_load_time=analysis_cache_load_time,
        total_time=total_time,
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
    )

    st.info(
        f"⏱ Total processing time: "
        f"{total_time:.4f} seconds"
    )

    timing_df = pd.DataFrame({
        "Stage": [
            "Raw Graph Loading",
            "Processed Graph Cache Loading",
            "Preprocessing",
            "Graph Statistics",
            "Centrality Computation",
            "Correlation Computation",
            "Analysis Cache Loading",
            "Total",
        ],
        "Time (seconds)": [
            raw_graph_load_time,
            graph_cache_load_time,
            preprocessing_time,
            statistics_time,
            centrality_time,
            correlation_time,
            analysis_cache_load_time,
            total_time,
        ]
    })

    st.subheader("⏱ Current Run Benchmark")

    st.dataframe(
        timing_df,
        width="stretch"
    )

    # =====================================================
    # GRAPH STATISTICS
    # =====================================================

    st.subheader("📌 Graph Statistics")

    st.dataframe(
        GraphStatistics.to_dataframe(stats),
        use_container_width=True
    )


    # =====================================================
    # CENTRALITY
    # =====================================================

    st.subheader("📈 Centrality Measures")

    st.dataframe(
        centrality_df,
        use_container_width=True
    )


    st.subheader("🏆 Top Influential Nodes")

    top_nodes = CentralityAnalyzer.top_k(
        centrality_df,
        "Degree Centrality",
        k=10
    )

    st.dataframe(
        top_nodes,
        width="stretch"
    )
        # Save Top Nodes to cache
    CacheManager.save_csv(
        dataset_key,
        "top10.csv",
        top_nodes
    )


    # =====================================================
    # CORRELATION
    # =====================================================

    st.subheader("🔗 Correlation Matrix")

    st.dataframe(
        correlation_df,
        use_container_width=True
    )
    # =====================================================
    # NETWORK VISUALIZATION
    # =====================================================

    st.subheader("🌐 Network Visualization")

    network_plot_name = f"{dataset_key}_network"
    network_plot_path = (
        CacheManager.BASE_DIR
        / "plots"
        / f"{network_plot_name}.html"
    )

    if ExportManager.plot_exists(network_plot_name):

        with open(
            network_plot_path,
            "r",
            encoding="utf-8"
        ) as file:
            components.html(
                file.read(),
                height=700,
                scrolling=True
            )

    else:

        # The HTML plot is missing, so a graph is required.
        graph = ensure_graph(
            graph,
            dataset_option
        )

        fig = GraphVisualizer.plot_graph(
            graph,
            title="Network Graph"
        )

        ExportManager.export_plotly(
            fig,
            network_plot_name
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )
    # =====================================================
    # PAGERANK VISUALIZATION
    # =====================================================

    st.subheader("🔥 PageRank Influence")

    pagerank_plot_name = f"{dataset_key}_pagerank"
    pagerank_plot_path = (
        CacheManager.BASE_DIR
        / "plots"
        / f"{pagerank_plot_name}.html"
    )

    if ExportManager.plot_exists(pagerank_plot_name):

        with open(
            pagerank_plot_path,
            "r",
            encoding="utf-8"
        ) as file:
            components.html(
                file.read(),
                height=700,
                scrolling=True
            )

    else:

        # The HTML plot is missing, so a graph is required.
        graph = ensure_graph(
            graph,
            dataset_option
        )

        # Reuse PageRank from the cached centrality table.
        pagerank_scores = dict(
            zip(
                centrality_df["Node"],
                centrality_df["PageRank"]
            )
        )

        fig2 = GraphVisualizer.plot_by_centrality(
            graph,
            pagerank_scores,
            title="PageRank Influence Map"
        )

        ExportManager.export_plotly(
            fig2,
            pagerank_plot_name
        )

        st.plotly_chart(
            fig2,
            width="stretch"
        )


    # =====================================================
    # EXPORT INFO (NO RECOMPUTE)
    # =====================================================

    st.subheader("💾 Cached Output Locations")

    st.write(
        "📁 Stats:",
        CacheManager.get_path(dataset_key) / "stats.json"
    )

    st.write(
        "📁 Centrality:",
        CacheManager.get_path(dataset_key) / "centrality.csv"
    )

    st.write(
        "📁 Correlation:",
        CacheManager.get_path(dataset_key) / "correlation.json"
    )


# =====================================================
# SMALL PERFORMANCE INFO PANEL
# =====================================================

st.sidebar.markdown("---")
st.sidebar.caption("⚡ Cached system enabled")
st.sidebar.caption("Large datasets load only once")

# =====================================================
# DATASET COMPARISON (FULLY CACHED + OPTIMIZED)
# =====================================================

@st.cache_data(show_spinner=False)
def load_all_datasets():

    karate = load_graph("Karate Club (Built-in)")

    email = load_graph("Email-Eu-Core")

    web = load_graph("Web-Google (Sampled)")

    return {
        "Karate": karate,
        "Email": email,
        "Web-Google": web
    }


def get_comparison_table():

    try:

        if CacheManager.comparison_exists():

            return CacheManager.load_comparison()

    except Exception:

        pass

    graphs = load_all_datasets()

    comparison = DatasetComparator.compare_graphs(graphs)

    CacheManager.save_comparison(comparison)

    return comparison

# =====================================================
# SHOW COMPARISON UI
# =====================================================

if show_comparison:

    st.subheader("📊 Dataset Comparison Dashboard")

    with st.spinner("Computing dataset comparison..."):

        comparison_df = get_comparison_table()

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    st.success("Comparison loaded (cached for future runs)")