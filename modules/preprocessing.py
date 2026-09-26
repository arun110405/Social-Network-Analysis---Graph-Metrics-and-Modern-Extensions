"""
=========================================================
Graph Preprocessing Module
=========================================================

Description:
    This module provides preprocessing utilities for graph
    cleaning and normalization before applying centrality
    analysis.

Key functions:
    - Remove self-loops
    - Handle isolated nodes
    - Extract largest connected component
    - Normalize graphs for fair comparison

=========================================================
"""

from typing import Tuple
import networkx as nx

from utils.logger import logger


class GraphPreprocessor:
    """
    Preprocessing pipeline for graph cleaning and normalization.
    """

    # =====================================================
    # Remove Self-Loops
    # =====================================================

    @staticmethod
    def remove_self_loops(graph: nx.Graph) -> nx.Graph:
        """
        Remove self-loops from the graph.

        Parameters
        ----------
        graph : nx.Graph

        Returns
        -------
        nx.Graph
        """

        initial_loops = nx.number_of_selfloops(graph)

        graph = graph.copy()
        graph.remove_edges_from(nx.selfloop_edges(graph))

        logger.info("Removed %d self-loops", initial_loops)

        return graph

    # =====================================================
    # Remove Isolated Nodes
    # =====================================================

    @staticmethod
    def remove_isolates(graph: nx.Graph) -> nx.Graph:
        """
        Remove isolated nodes from the graph.
        """

        graph = graph.copy()

        isolates = list(nx.isolates(graph))
        graph.remove_nodes_from(isolates)

        logger.info("Removed %d isolated nodes", len(isolates))

        return graph

    # =====================================================
    # Largest Connected Component
    # =====================================================

    @staticmethod
    def largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """
        Extract largest connected component.

        Ensures meaningful centrality comparison.
        """

        graph = graph.copy()

        if graph.is_directed():
            components = nx.weakly_connected_components(graph)
        else:
            components = nx.connected_components(graph)

        largest_cc = max(components, key=len)

        subgraph = graph.subgraph(largest_cc).copy()

        logger.info(
            "Extracted largest connected component: %d nodes",
            subgraph.number_of_nodes(),
        )

        return subgraph

    # =====================================================
    # Full Preprocessing Pipeline
    # =====================================================

    @staticmethod
    def preprocess(graph: nx.Graph) -> nx.Graph:
        """
        Apply full preprocessing pipeline.

        Steps:
            1. Remove self-loops
            2. Remove isolates
            3. Extract largest connected component
        """

        logger.info("Starting graph preprocessing pipeline")

        graph = GraphPreprocessor.remove_self_loops(graph)
        graph = GraphPreprocessor.remove_isolates(graph)
        graph = GraphPreprocessor.largest_connected_component(graph)

        logger.info(
            "Preprocessing complete: %d nodes, %d edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )

        return graph

    # =====================================================
    # Graph Comparison Helper
    # =====================================================

    @staticmethod
    def compare_graphs(g1: nx.Graph, g2: nx.Graph) -> Tuple[int, int]:
        """
        Compare two graphs (nodes, edges difference).
        """

        node_diff = abs(g1.number_of_nodes() - g2.number_of_nodes())
        edge_diff = abs(g1.number_of_edges() - g2.number_of_edges())

        logger.info(
            "Graph comparison -> Node diff: %d, Edge diff: %d",
            node_diff,
            edge_diff,
        )

        return node_diff, edge_diff