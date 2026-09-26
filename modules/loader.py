from pathlib import Path
from typing import Union, Dict

import networkx as nx
import pandas as pd

from utils.logger import logger


class GraphLoader:

    @staticmethod
    def load_karate_club() -> nx.Graph:
        logger.info("Loading Karate Club dataset")
        return nx.karate_club_graph()

    @staticmethod
    def load_graph(filepath: Union[str, Path]):

        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")

        filename = filepath.name.lower()

        if filename == "email-eu-core.txt":
            return GraphLoader._load_email_dataset(filepath)

        elif filename == "web-google.txt":
            return GraphLoader._load_web_google(filepath)

        elif filepath.suffix.lower() == ".csv":
            return GraphLoader._load_csv(filepath)

        elif filepath.suffix.lower() == ".gml":
            return nx.read_gml(filepath)

        else:
            return GraphLoader._load_edgelist(filepath)

    @staticmethod
    def load_web_google_sample(max_nodes: int = 1500) -> nx.Graph:
        """
        Load a connected sample from the Web-Google dataset.

        Sampling strategy:
        1. Load the original directed Web-Google graph.
        2. Convert it to an undirected graph.
        3. Select the highest-degree node as the seed.
        4. Use breadth-first search to collect a connected sample.
        5. Return an induced subgraph containing up to max_nodes nodes.

        This prevents preprocessing from reducing the sample to a
        very small largest connected component.
        """

        filepath = Path(
            "data/raw/web_google/web-Google.txt"
        )

        if not filepath.exists():
            raise FileNotFoundError(
                f"Web-Google dataset not found: {filepath}"
            )

        logger.info(
            "Loading Web-Google dataset for connected sampling"
        )

        directed_graph = nx.read_edgelist(
            filepath,
            delimiter="\t",
            comments="#",
            nodetype=int,
            create_using=nx.DiGraph()
        )

        if directed_graph.number_of_nodes() == 0:
            raise ValueError(
                "Web-Google dataset contains no nodes"
            )

        # Convert to undirected for connected sampling.
        graph = directed_graph.to_undirected()

        # Select the most connected node as the BFS seed.
        seed_node = max(
            graph.degree,
            key=lambda item: item[1]
        )[0]

        logger.info(
            "Selected Web-Google BFS seed node: %s",
            seed_node
        )

        # Build a connected sample using BFS.
        selected_nodes = []
        visited = {seed_node}
        queue = [seed_node]

        while queue and len(selected_nodes) < max_nodes:

            current_node = queue.pop(0)
            selected_nodes.append(current_node)

            neighbours = sorted(
                graph.neighbors(current_node),
                key=lambda node: graph.degree(node),
                reverse=True
            )

            for neighbour in neighbours:

                if neighbour not in visited:

                    visited.add(neighbour)
                    queue.append(neighbour)

                if (
                    len(selected_nodes)
                    + len(queue)
                    >= max_nodes
                ):
                    break

        sampled_graph = graph.subgraph(
            selected_nodes[:max_nodes]
        ).copy()

        # Remove self-loops if present.
        sampled_graph.remove_edges_from(
            nx.selfloop_edges(sampled_graph)
        )

        if sampled_graph.number_of_nodes() == 0:
            raise ValueError(
                "Web-Google sampling produced an empty graph"
            )

        logger.info(
            "Connected Web-Google sample created: "
            "%d nodes, %d edges",
            sampled_graph.number_of_nodes(),
            sampled_graph.number_of_edges()
        )

        return sampled_graph


    @staticmethod
    def get_ego_source(graph):
        """
        Always safe seed node (NO FAILURES EVER)
        """

        degrees = dict(graph.degree())

        if not degrees:
            return None

        return max(degrees, key=degrees.get)
    @staticmethod
    def _load_csv(filepath: Path):

        df = pd.read_csv(filepath)

        return nx.from_pandas_edgelist(
            df,
            "source",
            "target"
        )

    @staticmethod
    def _load_edgelist(filepath: Path):

        return nx.read_edgelist(
            filepath,
            comments="#",
            nodetype=int
        )

    @staticmethod
    def _load_email_dataset(filepath: Path):

        return nx.read_edgelist(
            filepath,
            nodetype=int,
            create_using=nx.Graph()
        )

    @staticmethod
    def _load_web_google(filepath: Path):

        return nx.read_edgelist(
            filepath,
            delimiter="\t",
            comments="#",
            nodetype=int,
            create_using=nx.DiGraph()
        )

    @staticmethod
    def graph_summary(graph):

        return {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "directed": graph.is_directed(),
            "density": round(nx.density(graph), 6),
            "self_loops": nx.number_of_selfloops(graph),
            "isolates": len(list(nx.isolates(graph))),
        }

    @staticmethod
    def validate(graph):

        if graph.number_of_nodes() == 0:
            raise ValueError("Empty graph")

        if graph.number_of_edges() == 0:
            raise ValueError("Graph has no edges")

        return True
