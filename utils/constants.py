"""
=========================================================
Constants Module
=========================================================

Project:
    Social Network Analysis – Graph Metrics and Modern Extensions

Author:
    Arun Prasath Velkutty

University:
    University of Liverpool
    MSc Computer Science Dissertation (2025/26)

Description:
    This module stores application-wide constant values
    that remain unchanged during program execution.

=========================================================
"""

# =========================================================
# Supported Dataset Names
# =========================================================

SUPPORTED_DATASETS = [
    "Karate Club",
    "Cora Citation Network",
    "Email-Eu-Core"
]

# =========================================================
# Supported File Extensions
# =========================================================

SUPPORTED_FILE_TYPES = [
    ".gml",
    ".csv",
    ".txt",
    ".edges",
    ".edgelist"
]

# =========================================================
# Centrality Metrics
# =========================================================

CENTRALITY_METRICS = [
    "Degree Centrality",
    "Betweenness Centrality",
    "Closeness Centrality",
    "PageRank"
]

# =========================================================
# Graph Types
# =========================================================

GRAPH_TYPES = [
    "Undirected",
    "Directed"
]

# =========================================================
# Community Detection Algorithms
# =========================================================

COMMUNITY_ALGORITHMS = [
    "Louvain",
    "Greedy Modularity"
]

# =========================================================
# Correlation Methods
# =========================================================

CORRELATION_METHODS = [
    "Spearman"
]

# =========================================================
# Export Formats
# =========================================================

EXPORT_FORMATS = [
    "CSV",
    "JSON",
    "PNG",
    "PDF"
]

# =========================================================
# Default DataFrame Column Names
# =========================================================

CENTRALITY_COLUMNS = [
    "Node",
    "Degree Centrality",
    "Betweenness Centrality",
    "Closeness Centrality",
    "PageRank"
]

# =========================================================
# Dashboard Pages
# =========================================================

PAGES = [
    "Dashboard",
    "Comparison",
    "Community Detection",
    "Explainability"
]