# Social Network Analysis – Graph Metrics and Modern Extensions

**M.Sc. Advanced Data Science and Artificial Intelligence Dissertation**  
**University of Liverpool**  
**Student:** Arun Prasath Velkutty  
**Academic Year:** 2025–2026

---

1. Project Overview

This dissertation presents an interactive Social Network Analysis (SNA)
platform for analysing graph-structured datasets and evaluating node
importance using established graph-theoretic measures.

The system computes and compares Degree Centrality, Betweenness
Centrality, Closeness Centrality and PageRank across multiple network
datasets. It also evaluates ranking agreement, robustness under structural
changes, PageRank parameter sensitivity and community-detection
performance.

The platform provides interactive visualisation, deterministic
node-level explanations, performance analysis, persistent caching and
result export functionality. The implementation is designed as a
reproducible analytical workflow rather than as a new centrality or
community-detection algorithm.

---

2. Research Aim

To develop and evaluate an interactive Social Network Analysis platform
for comparing node importance and structural behaviour across multiple
network datasets.

---

3. Research Objectives

The objectives of this dissertation are to:

1. Load and preprocess multiple graph datasets.
2. Compute Degree, Betweenness, Closeness and PageRank centrality.
3. Compare centrality rankings using Spearman Rank Correlation.
4. Evaluate centrality robustness under node and edge removal.
5. Assess PageRank ranking sensitivity to different damping factors.
6. Compare Greedy Modularity, Louvain and Label Propagation community
   detection using multiple evaluation criteria.
7. Provide interactive visual analytics and deterministic explanations
   of node importance.
8. Evaluate execution performance and persistent caching.
9. Validate the implementation using automated tests.

---

4. Research Questions

**RQ1:** How consistently do different centrality measures identify
influential nodes across different types of networks?

**RQ2:** To what extent do centrality rankings correlate across different
network structures?

**RQ3:** How stable are Degree and PageRank rankings under structural
changes in the network?

---

5. Main Features

### Centrality Analysis

- Degree Centrality
- Betweenness Centrality
- Closeness Centrality
- PageRank
- Centrality ranking comparison
- Spearman Rank Correlation

### Robustness and Sensitivity

- Node-removal experiments
- Edge-removal experiments
- 5%, 10% and 20% structural perturbations
- Spearman ranking stability
- Top-10 ranking overlap
- PageRank damping-factor sensitivity

### Community Detection

- Greedy Modularity
- Louvain
- Label Propagation
- Modularity evaluation
- Repeated-run stability
- NMI
- ARI
- Ground-truth comparison where available
- Community-size balance
- Runtime comparison

### Visual Analytics

- Interactive network visualisation
- Centrality rankings
- Correlation analysis
- Community visualisation
- Graph statistics
- Interactive analytical dashboards

### Explainability

- Deterministic node-level explanations
- Percentile-based interpretation
- Metric-specific explanations
- Overall node importance summary
- Strongest and weakest metric identification

### Performance and Reproducibility

- Execution-time measurement
- Persistent disk-based caching
- Cached graph and analysis results
- Automated testing
- CSV export
- JSON export
- HTML graph export

---

6. Technologies Used

| Technology | Purpose |
|------------|---------|
| Python 3.12 | Programming language |
| NetworkX | Graph construction and network analysis |
| Streamlit | Interactive application and dashboard |
| Plotly | Interactive charts and visualisation |
| Pandas | Data processing and tabular analysis |
| NumPy | Numerical computing |
| SciPy | Statistical and correlation analysis |
| scikit-learn | Supporting analytical and preprocessing functionality |
| Python-Louvain | Louvain community detection |
| PyVis | Interactive network visualisation |
| pytest | Automated testing |

---

## 7. Project Structure

```text
Social-Network-Analysis/
│
├── README.md
├── LICENSE
├── requirements.txt
├── app.py
├── .gitignore
│
├── pages/
│   ├── Dashboard.py
│   ├── Comparison.py
│   ├── Community.py
│   ├── Robustness.py
│   ├── Performance.py
│   ├── Visual_Analytics.py
│   └── Explainability.py
│
├── modules/
│   ├── __init__.py
│   ├── cache_manager.py
│   ├── centrality.py
│   ├── community_detection.py
│   ├── comparison.py
│   ├── correlation.py
│   ├── explainability.py
│   ├── export.py
│   ├── graph_statistics.py
│   ├── insights.py
│   ├── loader.py
│   ├── performance.py
│   ├── preprocessing.py
│   ├── robustness.py
│   ├── visual_analytics.py
│   └── visualization.py
│
├── utils/
│   ├── __init__.py
│   ├── config.py
│   ├── constants.py
│   └── logger.py
│
├── data/
│   └── raw/
│       ├── email_eu/
│       │   └── email-Eu-Core.txt
│       └── web_google/
│           └── web-Google.txt
│
├── evaluation/
│   └── pagerank_sensitivity.py
│
└── tests/
    ├── conftest.py
    ├── test_cache_manager.py
    ├── test_centrality.py
    ├── test_community_detection.py
    ├── test_correlation.py
    ├── test_explainability.py
    ├── test_loader.py
    ├── test_pagerank_sensitivity.py
    └── test_preprocessing.py
```

The project is organised into separate components for the Streamlit interface, network-analysis modules, utility functions, datasets, evaluation scripts and automated tests.


8. Datasets

The evaluated system uses three network datasets:

1. Zachary's Karate Club

A small social network containing 34 nodes and 78 edges. The dataset is
provided directly through NetworkX and does not require a separate local
dataset file.

2. Email-Eu-Core

An email communication network used to evaluate the analytical workflow
on a larger graph. The processed network used in the experiments contains
986 nodes and 16,064 edges.

3. Web-Google

A large web graph originally containing 875,713 nodes and 5,105,039
directed edges.

Due to its size, the dissertation analyses a 1,500-node sample. The
sample is generated using a highest-degree seed followed by
breadth-first search (BFS), with neighbours explored according to
descending degree.

The Web-Google sample is therefore not a random sample of the complete
network.

---

9. Installation

    1. Create a virtual environment

    Windows:
    python -m venv .venv

    2. Activate the virtual environment
    Windows:
    .venv\Scripts\activate


    3. Install dependencies
    pip install -r requirements.txt


10. Dataset Setup

Place the required raw datasets in the following locations:
```text
data/
└── raw/
    ├── email_eu_core/
    │   └── email-Eu-Core.txt
    │
    └── web_google/
        └── web-Google.txt
```
Zachary's Karate Club does not need to be stored locally because it is
loaded directly from NetworkX.

11. Running the Application

From the project root directory, run:

streamlit run app.py

The Streamlit application provides the following pages:

Dashboard – overview of graph statistics and centrality results
Comparison – comparison of centrality rankings across datasets
Community – community-detection algorithms and evaluation
Robustness – node and edge removal and ranking stability
Performance – execution-time and caching analysis
Visual Analytics – additional interactive network analysis
Explainability – deterministic node-level explanations


12. Running the Tests

The automated test suite can be executed using:

pytest

The final implementation was validated using 58 automated tests, all of
which passed.

58 passed

The tests include validation of analytical functions, centrality
calculations, ranking behaviour and other implementation components.

13. Export

The system supports export of analytical results in:

CSV
JSON
HTML for graph visualisations

Generated results can be recreated through the application and therefore
do not need to be included in the source-code archive.

14. Caching

The system uses persistent disk-based caching to avoid repeating
expensive graph processing and analytical calculations.

Cached results are stored locally during execution and can be recreated
when required.

Caching is used to reduce repeated computation time; it does not change
the underlying algorithmic complexity of the network-analysis methods.

15. Explainability

The final implementation uses deterministic, metric-grounded
explanations rather than a generative Large Language Model.

Explanations are generated from calculated centrality values and their
percentile rankings. The system provides metric-specific descriptions
and an overall interpretation of node importance.

No OpenAI API key or external LLM service is required to run the final
implementation.

16. Reproducibility

The analytical workflow is implemented using fixed procedures for
dataset loading, preprocessing, centrality calculation, ranking
comparison, robustness analysis, PageRank sensitivity and community
evaluation.

The same datasets, configuration and procedures can therefore be used
to reproduce the reported analyses.

17. Software Engineering Principles

The implementation follows:

Modular architecture
Separation of concerns
Reusable components
Object-oriented design where appropriate
Type hinting
PEP 8 coding conventions
Unit and functional testing
Reproducible research practices
Persistent result caching


18. Future Enhancements

Possible future extensions include:

Graph embeddings such as Node2Vec
Graph Neural Networks
GraphSAGE
Temporal network analysis
Dynamic graph visualisation
Graph Transformer models
Extended robustness analysis for additional centrality measures
Targeted node and edge perturbation experiments


19. Author

Arun Prasath Velkutty

M.Sc. Advanced Data Science and Artificial Intelligence
University of Liverpool
2025–2026
