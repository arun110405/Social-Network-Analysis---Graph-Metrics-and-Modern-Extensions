import pandas as pd
import networkx as nx


class DatasetComparator:

    @staticmethod
    def compare_graphs(graphs: dict):
        """
        graphs = {
            "karate": G1,
            "email": G2,
            "web": G3
        }
        """

        results = []

        for name, G in graphs.items():

            results.append({
                "dataset": name,
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "density": nx.density(G),
                "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
                "components": nx.number_connected_components(G.to_undirected())
                if not G.is_directed()
                else 1
            })

        return pd.DataFrame(results)