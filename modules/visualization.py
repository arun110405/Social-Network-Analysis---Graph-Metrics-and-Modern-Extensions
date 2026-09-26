"""
=========================================================
Graph Visualization Module
=========================================================

Description:
    This module provides interactive graph visualisation
    using NetworkX and Plotly.

    Features:
    - Graph structure visualization
    - Node sizing based on centrality
    - Color mapping for metrics
    - Interactive hover information

=========================================================
"""

import networkx as nx
import plotly.graph_objects as go

from utils.logger import logger
from utils.config import Config


class GraphVisualizer:
    """
    Handles interactive graph visualisation.
    """

    # =====================================================
    # Base Graph Plot
    # =====================================================

    @staticmethod
    def plot_graph(graph: nx.Graph, title: str = "Graph") -> go.Figure:
        """
        Visualise graph structure using Plotly.
        """

        logger.info("Generating graph visualization")

        pos = nx.spring_layout(graph, seed=Config.RANDOM_SEED)

        edge_x = []
        edge_y = []

        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color="gray"),
            hoverinfo="none",
            mode="lines",
        )

        node_x = []
        node_y = []

        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=list(graph.nodes()),
            textposition="top center",
            hoverinfo="text",
            marker=dict(
                size=10,
                color="skyblue",
                line=dict(width=1, color="black"),
            ),
        )

        fig = go.Figure(data=[edge_trace, node_trace])

        fig.update_layout(
            title=title,
            showlegend=False,
            width=Config.FIGURE_WIDTH,
            height=Config.FIGURE_HEIGHT,
            margin=dict(l=20, r=20, t=40, b=20),
        )

        logger.info("Graph visualization created successfully")

        return fig

    # =====================================================
    # Visualise by Centrality
    # =====================================================

    @staticmethod
    def plot_by_centrality(
        graph: nx.Graph,
        centrality: dict,
        title: str = "Centrality Visualization",
    ) -> go.Figure:
        """
        Visualise graph with node size based on centrality.
        """

        logger.info("Generating centrality-based visualization")

        pos = nx.kamada_kawai_layout(graph)

        node_x = []
        node_y = []
        node_size = []
        node_text = []

        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            value = centrality.get(node, 0)
            node_size.append(value * 1000 + 10)

            node_text.append(f"Node: {node}<br>Score: {value:.4f}")

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            text=node_text,
            hoverinfo="text",
            marker=dict(
                size=node_size,
                color=node_size,
                colorscale="Viridis",
                showscale=True,
                line=dict(width=1, color="black"),
            ),
        )

        edge_x = []
        edge_y = []

        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1, color="lightgray"),
            hoverinfo="none",
        )

        fig = go.Figure(data=[edge_trace, node_trace])

        fig.update_layout(
            title=title,
            showlegend=False,
            width=Config.FIGURE_WIDTH,
            height=Config.FIGURE_HEIGHT,
        )

        logger.info("Centrality visualization created")

        return fig