"""
=========================================================
Graph Statistics Module
=========================================================
Description:
Computes descriptive statistics for graph datasets.

=========================================================
"""

import networkx as nx
import pandas as pd


class GraphStatistics:

    """
    Computes structural graph statistics.
    """

    @staticmethod
    def compute(graph):

        if graph.is_directed():
            G = graph.to_undirected()
        else:
            G = graph.copy()

        stats = {}

        # ------------------------------------------------
        # Basic Information
        # ------------------------------------------------

        stats["Nodes"] = G.number_of_nodes()
        stats["Edges"] = G.number_of_edges()

        stats["Directed"] = graph.is_directed()

        stats["Density"] = round(nx.density(G), 6)

        # ------------------------------------------------
        # Degree Statistics
        # ------------------------------------------------

        degrees = [d for _, d in G.degree()]

        stats["Average Degree"] = round(sum(degrees) / len(degrees), 3)

        stats["Maximum Degree"] = max(degrees)

        stats["Minimum Degree"] = min(degrees)

        # ------------------------------------------------
        # Connectivity
        # ------------------------------------------------

        stats["Connected Components"] = nx.number_connected_components(G)

        largest_cc = max(nx.connected_components(G), key=len)

        largest_graph = G.subgraph(largest_cc)

        stats["Largest Component Size"] = largest_graph.number_of_nodes()

        # ------------------------------------------------
        # Clustering
        # ------------------------------------------------

        stats["Average Clustering"] = round(
            nx.average_clustering(G),
            5
        )

        # ------------------------------------------------
        # Diameter
        # ------------------------------------------------

        try:

            stats["Diameter"] = nx.diameter(largest_graph)

        except:

            stats["Diameter"] = "N/A"

        # ------------------------------------------------
        # Average Path Length
        # ------------------------------------------------

        try:

            stats["Average Path Length"] = round(
                nx.average_shortest_path_length(
                    largest_graph
                ),
                4
            )

        except:

            stats["Average Path Length"] = "N/A"

        # ------------------------------------------------

        stats["Self Loops"] = nx.number_of_selfloops(G)

        stats["Isolated Nodes"] = len(
            list(nx.isolates(G))
        )

        return stats

    @staticmethod
    def to_dataframe(stats):

        df = pd.DataFrame({
        "Metric": list(stats.keys()),
        "Value": [str(v) for v in stats.values()]
        })

        return df