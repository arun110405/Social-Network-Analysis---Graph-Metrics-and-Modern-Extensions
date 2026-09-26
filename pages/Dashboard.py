"""
=========================================================
Network Analytics Dashboard
=========================================================
Overview page for the Social Network Analysis platform.
=========================================================
"""

import streamlit as st

from modules.cache_manager import CacheManager


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Network Analytics Dashboard",
    layout="wide",
)

st.title("📊 Network Analytics Dashboard")

st.markdown(
    """
This interactive platform provides comparative **Social Network
Analysis** across multiple real-world graph datasets.

The system focuses on identifying influential nodes, comparing
centrality measures, analysing community structure, testing
ranking robustness and evaluating computational performance.
"""
)


# =====================================================
# PROJECT OVERVIEW
# =====================================================

st.subheader("🔍 Project Overview")

overview_col_1, overview_col_2, overview_col_3 = st.columns(3)

with overview_col_1:

    st.metric(
        "Datasets",
        "3",
    )

    st.caption(
        "Karate Club, Email-Eu-Core and sampled Web-Google"
    )

with overview_col_2:

    st.metric(
        "Centrality Measures",
        "4",
    )

    st.caption(
        "Degree, Betweenness, Closeness and PageRank"
    )

with overview_col_3:

    st.metric(
        "Community Algorithms",
        "3",
    )

    st.caption(
        "Greedy Modularity, Louvain and Label Propagation"
    )


# =====================================================
# ANALYTICAL WORKFLOW
# =====================================================

st.subheader("⚙️ Analytical Workflow")

st.markdown(
    """
**1. Dataset Loading →**
Load benchmark social, communication and web networks.

**2. Preprocessing →**
Clean graph structures and prepare connected graphs for analysis.

**3. Graph Statistics →**
Measure nodes, edges, density, degree, clustering and connectivity.

**4. Centrality Analysis →**
Calculate Degree, Betweenness, Closeness and PageRank.

**5. Ranking Comparison →**
Compare influential-node rankings using Spearman correlation.

**6. Community Detection →**
Detect and compare network communities using multiple algorithms.

**7. Visual Analytics →**
Explore heatmaps, centrality distributions, influential nodes and
community structures.

**8. Robustness Evaluation →**
Test whether influential-node rankings remain stable after node
and edge removal.

**9. Performance Evaluation →**
Compare first-run computation with persistent cached execution.
"""
)


# =====================================================
# DATASET CACHE STATUS
# =====================================================

st.subheader("💾 Dataset Analysis Status")

DATASETS = {
    "Karate Club": "karate",
    "Email-Eu-Core": "email",
    "Web-Google": "web_google",
}

status_records = []

for dataset_name, dataset_key in DATASETS.items():

    graph_ready = CacheManager.graph_exists(
        dataset_key
    )

    centrality_ready = CacheManager.exists(
        dataset_key,
        "centrality.csv",
    )

    correlation_ready = CacheManager.exists(
        dataset_key,
        "correlation.json",
    )

    stats_ready = CacheManager.exists(
        dataset_key,
        "stats.json",
    )

    analysis_ready = (
        graph_ready
        and centrality_ready
        and correlation_ready
        and stats_ready
    )

    status_records.append(
        {
            "Dataset": dataset_name,
            "Graph Cache": (
                "✅ Ready"
                if graph_ready
                else "❌ Missing"
            ),
            "Centrality": (
                "✅ Ready"
                if centrality_ready
                else "❌ Missing"
            ),
            "Correlation": (
                "✅ Ready"
                if correlation_ready
                else "❌ Missing"
            ),
            "Statistics": (
                "✅ Ready"
                if stats_ready
                else "❌ Missing"
            ),
            "Overall": (
                "✅ Analysis Ready"
                if analysis_ready
                else "⚠️ Run Home Analysis"
            ),
        }
    )

st.dataframe(
    status_records,
    width="stretch",
    hide_index=True,
)


# =====================================================
# CORE ANALYSIS
# =====================================================

st.subheader("📈 Core Network Analysis")

core_col_1, core_col_2 = st.columns(2)

with core_col_1:

    st.markdown(
        """
### 🎯 Centrality Analysis

Identify influential nodes from different structural perspectives.

- **Degree Centrality** — direct connectivity
- **Betweenness Centrality** — bridge importance
- **Closeness Centrality** — network reachability
- **PageRank** — recursive global influence

The system compares these measures rather than assuming that one
definition of influence is sufficient.
"""
    )

with core_col_2:

    st.markdown(
        """
### 🔗 Ranking Correlation

Centrality rankings are compared using **Spearman rank
correlation**.

This identifies:

- where centrality measures agree;
- where rankings diverge;
- whether agreement changes across sparse and dense networks.

These results support the first two research questions.
"""
    )


# =====================================================
# COMMUNITY ANALYSIS
# =====================================================

st.subheader("👥 Community Detection")

community_col_1, community_col_2 = st.columns(2)

with community_col_1:

    st.markdown(
        """
### Algorithms

The system evaluates three community-detection methods:

- Greedy Modularity
- Louvain
- Label Propagation

Each algorithm is applied to the same datasets so their behaviour
can be compared fairly.
"""
    )

with community_col_2:

    st.markdown(
        """
### Evaluation

Community algorithms are compared using:

- Modularity
- Runtime
- Number of communities
- Community-size balance
- Repeated-run stability
- NMI and ARI
- Karate Club ground-truth agreement

The visual comparison uses consistent node positions so differences
in colour represent genuine differences in community assignment.
"""
    )


# =====================================================
# ROBUSTNESS
# =====================================================

st.subheader("🛡️ Robustness Evaluation")

st.markdown(
    """
The robustness module tests whether influential nodes remain
important after structural changes to the network.

Experiments include:

- **5% node/edge removal**
- **10% node/edge removal**
- **20% node/edge removal**
- fixed random seed for reproducibility;
- Spearman ranking stability;
- Top-K influential-node overlap.

This directly evaluates the stability of centrality measures under
network perturbation.
"""
)


# =====================================================
# PERFORMANCE
# =====================================================

st.subheader("⚡ Performance and Caching")

performance_col_1, performance_col_2 = st.columns(2)

with performance_col_1:

    st.markdown(
        """
### First Run

The initial execution may include:

- raw dataset loading;
- graph preprocessing;
- graph statistics;
- centrality calculation;
- correlation calculation.

These stages are benchmarked separately.
"""
    )

with performance_col_2:

    st.markdown(
        """
### Cached Run

Processed graphs and analytical results are stored persistently.

Cached outputs include:

- `graph.pkl`
- `stats.json`
- `centrality.csv`
- `correlation.json`
- community results
- visualisations

This avoids unnecessary recomputation after restarting Streamlit.
"""
    )


# =====================================================
# VISUAL ANALYTICS
# =====================================================

st.subheader("📊 Visual Analytics")

st.markdown(
    """
The platform provides interactive analytical visualisations including:

- network structure;
- PageRank influence maps;
- Spearman correlation heatmaps;
- centrality distributions;
- influential-node rankings;
- community-coloured network graphs;
- algorithm comparison charts;
- stability plots;
- performance benchmarks.

Plotly is used to provide interactive zooming, hovering and
exploration.
"""
)


# =====================================================
# RESEARCH QUESTIONS
# =====================================================

st.subheader("🎓 Research Questions")

st.markdown(
    """
**RQ1 — How consistently do different centrality measures identify
influential nodes across different types of networks?**

Supported through centrality rankings, Top-K nodes and Spearman
correlation.

---

**RQ2 — To what extent do centrality rankings correlate across
sparse and dense graph structures?**

Supported through cross-dataset graph statistics, correlation
heatmaps and comparative analysis.

---

**RQ3 — Which centrality measures are most stable under structural
changes in the network?**

Supported through node-removal and edge-removal robustness
experiments.
"""
)


# =====================================================
# AVAILABLE PAGES
# =====================================================

st.subheader("🧭 Analysis Pages")

page_col_1, page_col_2, page_col_3 = st.columns(3)

with page_col_1:

    st.markdown(
        """
### 📊 Comparison
Compare graph statistics, centrality rankings and correlations.

### 👥 Community
Run and compare community-detection algorithms.
"""
    )

with page_col_2:

    st.markdown(
        """
### 📈 Visual Analytics
Explore distributions, heatmaps and metric relationships.

### 🛡️ Robustness
Evaluate ranking stability under structural perturbations.
"""
    )

with page_col_3:

    st.markdown(
        """
### ⚡ Performance
Analyse first-run and cached execution times.

### 💡 Explainability
Interpret node influence using metric-based explanations.
"""
    )


# =====================================================
# SYSTEM STATUS
# =====================================================

st.markdown("---")

ready_count = sum(
    1
    for record in status_records
    if record["Overall"] == "✅ Analysis Ready"
)

if ready_count == len(DATASETS):

    st.success(
        "✅ All datasets have cached analytical results. "
        "The platform is ready for comparative analysis."
    )

else:

    st.warning(
        f"⚠️ {ready_count}/{len(DATASETS)} datasets currently "
        "have complete cached analysis results. Run missing "
        "datasets from the Home page before using all comparison "
        "features."
    )
