"""
=========================================================
Centrality Metrics Module
=========================================================
Description:
    Computes key centrality measures on graphs:

    - Degree Centrality
    - Betweenness Centrality
    - Closeness Centrality
    - PageRank

Also provides unified ranking outputs for comparison
and correlation analysis.

=========================================================
"""

from typing import Dict

import networkx as nx
import pandas as pd

from utils.logger import logger
from utils.constants import CENTRALITY_COLUMNS
from utils.config import Config


class CentralityAnalyzer:
    """
    Computes and manages centrality metrics.
    """

    # =====================================================
    # Degree Centrality
    # =====================================================

    @staticmethod
    def degree_centrality(graph: nx.Graph) -> Dict:
        """
        Compute Degree Centrality.
        """
        logger.info("Computing Degree Centrality")
        return nx.degree_centrality(graph)

    # =====================================================
    # Betweenness Centrality
    # =====================================================

    @staticmethod
    def betweenness_centrality(graph: nx.Graph) -> Dict:
        """
        Compute Betweenness Centrality.

        Uses approximate computation for large graphs
        to greatly reduce execution time.
        """

        logger.info("Computing Betweenness Centrality")

        if graph.number_of_nodes() > 1000:
            return nx.betweenness_centrality(
                graph,
                k=min(300, graph.number_of_nodes()),
                normalized=True,
                seed=42
            )

        return nx.betweenness_centrality(
            graph,
            normalized=True
        )

    # =====================================================
    # Closeness Centrality
    # =====================================================

    @staticmethod
    def closeness_centrality(graph: nx.Graph) -> Dict:
        """
        Compute Closeness Centrality.
        """
        logger.info("Computing Closeness Centrality")
        return nx.closeness_centrality(graph)

    # =====================================================
    # PageRank
    # =====================================================

    @staticmethod
    def pagerank(graph: nx.Graph) -> Dict:
        """
        Compute PageRank.
        """
        logger.info("Computing PageRank")

        return nx.pagerank(
            graph,
            alpha=Config.PAGERANK_DAMPING_FACTOR,
            max_iter=Config.PAGERANK_MAX_ITERATIONS,
            tol=Config.PAGERANK_TOLERANCE,
        )

    # =====================================================
    # Compute All Centralities
    # =====================================================

    @staticmethod
    def compute_all(graph: nx.Graph) -> pd.DataFrame:
        """
        Compute all centrality measures and return DataFrame.
        """

        logger.info("Computing all centrality measures")

        degree = CentralityAnalyzer.degree_centrality(graph)
        betweenness = CentralityAnalyzer.betweenness_centrality(graph)
        closeness = CentralityAnalyzer.closeness_centrality(graph)
        pagerank = CentralityAnalyzer.pagerank(graph)

        df = pd.DataFrame({
            "Node": list(graph.nodes()),
            "Degree Centrality": [degree[n] for n in graph.nodes()],
            "Betweenness Centrality": [betweenness[n] for n in graph.nodes()],
            "Closeness Centrality": [closeness[n] for n in graph.nodes()],
            "PageRank": [pagerank[n] for n in graph.nodes()],
        })

        logger.info("Centrality computation complete")

        return df

    # =====================================================
    # Rank Nodes
    # =====================================================

    @staticmethod
    def rank_nodes(df: pd.DataFrame, metric: str) -> pd.DataFrame:
        """
        Rank nodes based on a centrality metric.
        """

        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found")

        ranked = df.sort_values(by=metric, ascending=False).reset_index(drop=True)

        ranked["Rank"] = range(1, len(ranked) + 1)

        logger.info("Nodes ranked by %s", metric)

        return ranked

    # =====================================================
    # Top-K Nodes
    # =====================================================

    @staticmethod
    def top_k(df: pd.DataFrame, metric: str, k: int = 5) -> pd.DataFrame:
        """
        Return top-k most important nodes.
        """

        ranked = CentralityAnalyzer.rank_nodes(df, metric)

        return ranked.head(k)