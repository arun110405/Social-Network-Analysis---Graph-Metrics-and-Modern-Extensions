"""
=========================================================
Community Detection Module
=========================================================
Algorithms:

1. Greedy Modularity Maximisation
2. Louvain Community Detection
3. Asynchronous Label Propagation

The module also provides community summaries, modularity
scores, runtime measurements and agreement analysis.
=========================================================
"""

from __future__ import annotations
from itertools import combinations
from typing import Dict, List, Mapping, Set, Tuple


import numpy as np
import time
import networkx as nx
import pandas as pd

try:
    import community as community_louvain
except ImportError:
    community_louvain = None

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from utils.logger import logger


class CommunityAnalyzer:
    """
    Detects, compares and summarises network communities.
    """

    SUPPORTED_ALGORITHMS = {
        "greedy": "Greedy Modularity",
        "louvain": "Louvain",
        "label_propagation": "Label Propagation",
    }

    # =====================================================
    # PREPARE GRAPH
    # =====================================================

    @staticmethod
    def prepare_graph(graph: nx.Graph) -> nx.Graph:
        """
        Return a clean undirected graph suitable for community
        detection and modularity evaluation.
        """

        if graph is None:
            raise ValueError("Graph cannot be None")

        if graph.number_of_nodes() == 0:
            raise ValueError("Graph has no nodes")

        prepared_graph = (
            graph.to_undirected()
            if graph.is_directed()
            else graph.copy()
        )

        prepared_graph.remove_edges_from(
            nx.selfloop_edges(prepared_graph)
        )

        prepared_graph.remove_nodes_from(
            list(nx.isolates(prepared_graph))
        )

        if prepared_graph.number_of_nodes() == 0:
            raise ValueError(
                "Graph contains no connected nodes after preprocessing"
            )

        if prepared_graph.number_of_edges() == 0:
            raise ValueError(
                "Graph contains no edges after preprocessing"
            )

        return prepared_graph

    # =====================================================
    # ALGORITHM VALIDATION
    # =====================================================

    @staticmethod
    def validate_algorithm(algorithm: str) -> str:
        """
        Validate and normalise an algorithm key.
        """

        algorithm_key = algorithm.strip().lower()

        if algorithm_key not in CommunityAnalyzer.SUPPORTED_ALGORITHMS:
            supported = ", ".join(
                CommunityAnalyzer.SUPPORTED_ALGORITHMS.keys()
            )

            raise ValueError(
                f"Unsupported community algorithm: {algorithm}. "
                f"Supported algorithms are: {supported}"
            )

        return algorithm_key

    # =====================================================
    # COMMUNITY DETECTION
    # =====================================================

    @staticmethod
    def detect_communities(
        graph: nx.Graph,
        algorithm: str = "greedy",
        seed: int = 42,
        resolution: float = 1.0,
    ) -> List[Set]:
        """
        Detect communities using the selected algorithm.

        Parameters
        ----------
        graph:
            NetworkX graph.

        algorithm:
            greedy, louvain or label_propagation.

        seed:
            Random seed used for reproducible stochastic methods.

        resolution:
            Louvain resolution parameter. Values above 1 usually
            produce more communities, while values below 1 usually
            produce fewer communities.
        """

        algorithm_key = CommunityAnalyzer.validate_algorithm(
            algorithm
        )

        prepared_graph = CommunityAnalyzer.prepare_graph(graph)

        logger.info(
            "Starting %s community detection",
            CommunityAnalyzer.SUPPORTED_ALGORITHMS[
                algorithm_key
            ],
        )

        if algorithm_key == "greedy":

            communities = list(
                nx.algorithms.community.greedy_modularity_communities(
                    prepared_graph
                )
            )

        elif algorithm_key == "louvain":

            if community_louvain is None:
                raise ImportError(
                    "The python-louvain package is required. "
                    "Install it using: pip install python-louvain"
                )

            partition = community_louvain.best_partition(
                prepared_graph,
                random_state=seed,
                resolution=resolution,
            )

            grouped_communities: Dict[int, Set] = {}

            for node, community_id in partition.items():
                grouped_communities.setdefault(
                    community_id,
                    set(),
                ).add(node)

            communities = list(
                grouped_communities.values()
            )

        elif algorithm_key == "label_propagation":

            communities = list(
                nx.algorithms.community.asyn_lpa_communities(
                    prepared_graph,
                    seed=seed,
                )
            )

        else:
            raise ValueError(
                f"Unsupported algorithm: {algorithm_key}"
            )

        communities = [
            set(community)
            for community in communities
            if len(community) > 0
        ]

        communities = sorted(
            communities,
            key=len,
            reverse=True,
        )

        if not communities:
            raise ValueError(
                "The selected algorithm did not return any communities"
            )

        logger.info(
            "%s completed: %d communities detected",
            CommunityAnalyzer.SUPPORTED_ALGORITHMS[
                algorithm_key
            ],
            len(communities),
        )

        return communities

    # =====================================================
    # RUN ALGORITHM WITH TIMING
    # =====================================================

    @staticmethod
    def run_algorithm(
        graph: nx.Graph,
        algorithm: str,
        seed: int = 42,
        resolution: float = 1.0,
    ) -> Dict:
        """
        Run one community algorithm and return its results,
        modularity score and execution time.
        """

        algorithm_key = CommunityAnalyzer.validate_algorithm(
            algorithm
        )

        start_time = time.perf_counter()

        communities = CommunityAnalyzer.detect_communities(
            graph=graph,
            algorithm=algorithm_key,
            seed=seed,
            resolution=resolution,
        )

        runtime = time.perf_counter() - start_time

        modularity = CommunityAnalyzer.calculate_modularity(
            graph,
            communities,
        )

        sizes = [
            len(community)
            for community in communities
        ]

        return {
            "algorithm_key": algorithm_key,
            "algorithm_name": (
                CommunityAnalyzer.SUPPORTED_ALGORITHMS[
                    algorithm_key
                ]
            ),
            "communities": communities,
            "modularity": float(modularity),
            "runtime_seconds": float(runtime),
            "number_of_communities": len(communities),
            "largest_community": max(sizes) if sizes else 0,
            "smallest_community": min(sizes) if sizes else 0,
            "average_community_size": (
                sum(sizes) / len(sizes)
                if sizes
                else 0.0
            ),
        }

    # =====================================================
    # MODULARITY
    # =====================================================

    @staticmethod
    def calculate_modularity(
        graph: nx.Graph,
        communities: List[Set],
    ) -> float:
        """
        Calculate the modularity score of a partition.
        """

        if not communities:
            raise ValueError(
                "Communities cannot be empty"
            )

        prepared_graph = CommunityAnalyzer.prepare_graph(
            graph
        )

        prepared_nodes = set(
            prepared_graph.nodes()
        )

        community_nodes = set().union(
            *communities
        )

        if prepared_nodes != community_nodes:
            missing_nodes = prepared_nodes - community_nodes
            extra_nodes = community_nodes - prepared_nodes

            raise ValueError(
                "Community partition does not match graph nodes. "
                f"Missing nodes: {len(missing_nodes)}, "
                f"extra nodes: {len(extra_nodes)}"
            )

        score = nx.algorithms.community.modularity(
            prepared_graph,
            communities,
        )

        return float(score)

    # =====================================================
    # SUMMARY TABLE
    # =====================================================

    @staticmethod
    def summarize_communities(
        communities: List[Set],
    ) -> pd.DataFrame:
        """
        Return a summary describing each community.
        """

        records = []

        total_nodes = sum(
            len(community)
            for community in communities
        )

        for index, community in enumerate(
            communities,
            start=1,
        ):
            size = len(community)

            percentage = (
                (size / total_nodes) * 100
                if total_nodes > 0
                else 0
            )

            records.append(
                {
                    "Community": index,
                    "Size": size,
                    "Percentage": round(
                        percentage,
                        2,
                    ),
                }
            )

        return pd.DataFrame(records)

    # =====================================================
    # NODE MEMBERSHIP
    # =====================================================

    @staticmethod
    def membership_mapping(
        communities: List[Set],
    ) -> Dict:
        """
        Return a node-to-community mapping.
        """

        mapping = {}

        for community_number, community in enumerate(
            communities,
            start=1,
        ):
            for node in community:
                mapping[node] = community_number

        return mapping

    @staticmethod
    def membership_dataframe(
        communities: List[Set],
    ) -> pd.DataFrame:
        """
        Return every node and its assigned community.
        """

        records = []

        mapping = CommunityAnalyzer.membership_mapping(
            communities
        )

        for node, community_number in mapping.items():
            records.append(
                {
                    "Node": node,
                    "Community": community_number,
                }
            )

        dataframe = pd.DataFrame(records)

        if not dataframe.empty:
            dataframe = dataframe.sort_values(
                by=["Community", "Node"],
            ).reset_index(drop=True)

        return dataframe

    # =====================================================
    # MULTI-ALGORITHM COMPARISON
    # =====================================================

    @staticmethod
    def compare_algorithms(
        graph: nx.Graph,
        algorithms: List[str] | None = None,
        seed: int = 42,
        resolution: float = 1.0,
    ) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
        """
        Run multiple algorithms on the same graph and return
        a comparison table and full result dictionary.
        """

        if algorithms is None:
            algorithms = list(
                CommunityAnalyzer.SUPPORTED_ALGORITHMS.keys()
            )

        comparison_records = []
        results = {}

        for algorithm in algorithms:

            algorithm_key = (
                CommunityAnalyzer.validate_algorithm(
                    algorithm
                )
            )

            result = CommunityAnalyzer.run_algorithm(
                graph=graph,
                algorithm=algorithm_key,
                seed=seed,
                resolution=resolution,
            )

            results[algorithm_key] = result

            comparison_records.append(
                {
                    "Algorithm": result["algorithm_name"],
                    "Algorithm Key": algorithm_key,
                    "Communities": result[
                        "number_of_communities"
                    ],
                    "Modularity": round(
                        result["modularity"],
                        6,
                    ),
                    "Runtime (seconds)": round(
                        result["runtime_seconds"],
                        6,
                    ),
                    "Largest Community": result[
                        "largest_community"
                    ],
                    "Smallest Community": result[
                        "smallest_community"
                    ],
                    "Average Community Size": round(
                        result[
                            "average_community_size"
                        ],
                        3,
                    ),
                }
            )

        comparison_df = pd.DataFrame(
            comparison_records
        )

        if not comparison_df.empty:
            comparison_df = comparison_df.sort_values(
                by="Modularity",
                ascending=False,
            ).reset_index(drop=True)

        return comparison_df, results

    # =====================================================
    # ALGORITHM AGREEMENT
    # =====================================================

    @staticmethod
    def compare_partitions(
        communities_a: List[Set],
        communities_b: List[Set],
    ) -> Dict[str, float]:
        """
        Compare two community partitions using:

        - Normalised Mutual Information
        - Adjusted Rand Index
        """

        mapping_a = CommunityAnalyzer.membership_mapping(
            communities_a
        )

        mapping_b = CommunityAnalyzer.membership_mapping(
            communities_b
        )

        common_nodes = sorted(
            set(mapping_a).intersection(mapping_b),
            key=str,
        )

        if len(common_nodes) < 2:
            return {
                "NMI": 0.0,
                "ARI": 0.0,
                "Common Nodes": len(common_nodes),
            }

        labels_a = [
            mapping_a[node]
            for node in common_nodes
        ]

        labels_b = [
            mapping_b[node]
            for node in common_nodes
        ]

        nmi = normalized_mutual_info_score(
            labels_a,
            labels_b,
        )

        ari = adjusted_rand_score(
            labels_a,
            labels_b,
        )

        return {
            "NMI": float(nmi),
            "ARI": float(ari),
            "Common Nodes": len(common_nodes),
        }

    @staticmethod
    def agreement_dataframe(
        algorithm_results: Mapping[str, Dict],
    ) -> pd.DataFrame:
        """
        Produce pairwise agreement results for all algorithms.
        """

        algorithm_keys = list(
            algorithm_results.keys()
        )

        records = []

        for index, algorithm_a in enumerate(
            algorithm_keys
        ):
            for algorithm_b in algorithm_keys[
                index + 1:
            ]:
                result_a = algorithm_results[
                    algorithm_a
                ]

                result_b = algorithm_results[
                    algorithm_b
                ]

                agreement = (
                    CommunityAnalyzer.compare_partitions(
                        result_a["communities"],
                        result_b["communities"],
                    )
                )

                records.append(
                    {
                        "Algorithm A": result_a[
                            "algorithm_name"
                        ],
                        "Algorithm B": result_b[
                            "algorithm_name"
                        ],
                        "NMI": round(
                            agreement["NMI"],
                            6,
                        ),
                        "ARI": round(
                            agreement["ARI"],
                            6,
                        ),
                        "Common Nodes": agreement[
                            "Common Nodes"
                        ],
                    }
                )

        return pd.DataFrame(records)
        # =====================================================
    # COMMUNITY BALANCE
    # =====================================================

    @staticmethod
    def community_balance_metrics(
        communities: List[Set],
    ) -> Dict[str, float]:
        """
        Measure how evenly nodes are distributed across the
        detected communities.

        The coefficient of variation is:

            standard deviation / mean community size

        Lower values indicate more similarly sized communities.
        Higher values indicate greater size inequality.

        A low value is not automatically better because real
        networks may naturally contain unequal communities.
        """

        if not communities:
            return {
                "Community Size Standard Deviation": 0.0,
                "Community Size CV": 0.0,
                "Normalised Size Entropy": 0.0,
            }

        community_sizes = np.array(
            [
                len(community)
                for community in communities
            ],
            dtype=float,
        )

        mean_size = float(
            np.mean(community_sizes)
        )

        standard_deviation = float(
            np.std(
                community_sizes,
                ddof=0,
            )
        )

        coefficient_of_variation = (
            standard_deviation / mean_size
            if mean_size > 0
            else 0.0
        )

        total_nodes = float(
            np.sum(community_sizes)
        )

        probabilities = (
            community_sizes / total_nodes
            if total_nodes > 0
            else np.array([])
        )

        if (
            len(probabilities) > 1
            and np.all(probabilities > 0)
        ):
            entropy = float(
                -np.sum(
                    probabilities
                    * np.log(probabilities)
                )
            )

            maximum_entropy = float(
                np.log(
                    len(probabilities)
                )
            )

            normalised_entropy = (
                entropy / maximum_entropy
                if maximum_entropy > 0
                else 1.0
            )

        else:
            normalised_entropy = 1.0

        return {
            "Community Size Standard Deviation": round(
                standard_deviation,
                6,
            ),
            "Community Size CV": round(
                coefficient_of_variation,
                6,
            ),
            "Normalised Size Entropy": round(
                normalised_entropy,
                6,
            ),
        }

    # =====================================================
    # GROUND-TRUTH EVALUATION
    # =====================================================

    @staticmethod
    def evaluate_karate_ground_truth(
        graph: nx.Graph,
        communities: List[Set],
    ) -> Dict[str, float] | None:
        """
        Compare detected Karate Club communities with the
        observed Mr. Hi and Officer club split.

        Returns NMI and ARI scores. Returns None when suitable
        club labels are unavailable.
        """

        prepared_graph = (
            CommunityAnalyzer.prepare_graph(
                graph
            )
        )

        membership = (
            CommunityAnalyzer.membership_mapping(
                communities
            )
        )

        reference_graph = (
            nx.karate_club_graph()
        )

        true_labels = []
        predicted_labels = []

        for node in sorted(
            prepared_graph.nodes(),
            key=str,
        ):

            club_label = prepared_graph.nodes[
                node
            ].get("club")

            if (
                club_label is None
                and node in reference_graph
            ):
                club_label = reference_graph.nodes[
                    node
                ].get("club")

            if (
                club_label is None
                or node not in membership
            ):
                return None

            true_labels.append(
                0
                if club_label == "Mr. Hi"
                else 1
            )

            predicted_labels.append(
                membership[node]
            )

        if len(true_labels) < 2:
            return None

        nmi = normalized_mutual_info_score(
            true_labels,
            predicted_labels,
        )

        ari = adjusted_rand_score(
            true_labels,
            predicted_labels,
        )

        return {
            "Ground Truth NMI": round(
                float(nmi),
                6,
            ),
            "Ground Truth ARI": round(
                float(ari),
                6,
            ),
        }

    # =====================================================
    # REPEATED-RUN STABILITY
    # =====================================================

    @staticmethod
    def stability_analysis(
        graph: nx.Graph,
        algorithm: str,
        number_of_runs: int = 5,
        base_seed: int = 42,
        resolution: float = 1.0,
    ) -> Dict:
        """
        Run an algorithm multiple times using different seeds.

        Stability is measured using pairwise NMI and ARI
        between the resulting community assignments.

        Higher mean NMI and ARI indicate that repeated runs
        produce more similar partitions.
        """

        algorithm_key = (
            CommunityAnalyzer.validate_algorithm(
                algorithm
            )
        )

        number_of_runs = max(
            int(number_of_runs),
            2,
        )

        run_results = []

        for run_number in range(
            number_of_runs
        ):

            current_seed = (
                int(base_seed)
                + run_number
            )

            start_time = (
                time.perf_counter()
            )

            communities = (
                CommunityAnalyzer.detect_communities(
                    graph=graph,
                    algorithm=algorithm_key,
                    seed=current_seed,
                    resolution=resolution,
                )
            )

            runtime = (
                time.perf_counter()
                - start_time
            )

            modularity = (
                CommunityAnalyzer.calculate_modularity(
                    graph,
                    communities,
                )
            )

            balance = (
                CommunityAnalyzer.community_balance_metrics(
                    communities
                )
            )

            run_results.append(
                {
                    "Run": run_number + 1,
                    "Seed": current_seed,
                    "Communities": communities,
                    "Number of Communities": len(
                        communities
                    ),
                    "Modularity": float(
                        modularity
                    ),
                    "Runtime Seconds": float(
                        runtime
                    ),
                    "Community Size CV": balance[
                        "Community Size CV"
                    ],
                }
            )

        pairwise_nmi_scores = []
        pairwise_ari_scores = []

        for result_a, result_b in combinations(
            run_results,
            2,
        ):

            agreement = (
                CommunityAnalyzer.compare_partitions(
                    result_a["Communities"],
                    result_b["Communities"],
                )
            )

            pairwise_nmi_scores.append(
                agreement["NMI"]
            )

            pairwise_ari_scores.append(
                agreement["ARI"]
            )

        modularity_values = np.array(
            [
                result["Modularity"]
                for result in run_results
            ],
            dtype=float,
        )

        community_count_values = np.array(
            [
                result[
                    "Number of Communities"
                ]
                for result in run_results
            ],
            dtype=float,
        )

        runtime_values = np.array(
            [
                result["Runtime Seconds"]
                for result in run_results
            ],
            dtype=float,
        )

        nmi_values = np.array(
            pairwise_nmi_scores,
            dtype=float,
        )

        ari_values = np.array(
            pairwise_ari_scores,
            dtype=float,
        )

        run_records = []

        for result in run_results:
            run_records.append(
                {
                    "Run": result["Run"],
                    "Seed": result["Seed"],
                    "Communities": result[
                        "Number of Communities"
                    ],
                    "Modularity": round(
                        result["Modularity"],
                        6,
                    ),
                    "Runtime (seconds)": round(
                        result[
                            "Runtime Seconds"
                        ],
                        6,
                    ),
                    "Community Size CV": round(
                        result[
                            "Community Size CV"
                        ],
                        6,
                    ),
                }
            )

        return {
            "Algorithm": (
                CommunityAnalyzer.SUPPORTED_ALGORITHMS[
                    algorithm_key
                ]
            ),
            "Runs": number_of_runs,

            "Mean Modularity": round(
                float(
                    np.mean(
                        modularity_values
                    )
                ),
                6,
            ),

            "Modularity Standard Deviation": round(
                float(
                    np.std(
                        modularity_values,
                        ddof=0,
                    )
                ),
                6,
            ),

            "Mean Communities": round(
                float(
                    np.mean(
                        community_count_values
                    )
                ),
                3,
            ),

            "Community Count Standard Deviation": round(
                float(
                    np.std(
                        community_count_values,
                        ddof=0,
                    )
                ),
                6,
            ),

            "Mean Runtime Seconds": round(
                float(
                    np.mean(
                        runtime_values
                    )
                ),
                6,
            ),

            "Mean Pairwise NMI": round(
                float(
                    np.mean(
                        nmi_values
                    )
                )
                if len(nmi_values) > 0
                else 1.0,
                6,
            ),

            "Mean Pairwise ARI": round(
                float(
                    np.mean(
                        ari_values
                    )
                )
                if len(ari_values) > 0
                else 1.0,
                6,
            ),

            "Run Results": pd.DataFrame(
                run_records
            ),
        }

    # =====================================================
    # SERIALISATION
    # =====================================================

    @staticmethod
    def communities_to_dict(
        communities: List[Set],
    ) -> Dict:
        """
        Convert communities into JSON-compatible data.
        """

        return {
            "communities": [
                list(community)
                for community in communities
            ]
        }

    @staticmethod
    def communities_from_dict(
        data: Dict,
    ) -> List[Set]:
        """
        Restore communities from cached JSON data.
        """

        if not isinstance(data, dict):
            raise ValueError(
                "Cached community data must be a dictionary"
            )

        if "communities" not in data:
            raise ValueError(
                "Invalid cached community data"
            )

        communities = [
            set(community)
            for community in data["communities"]
        ]

        return sorted(
            communities,
            key=len,
            reverse=True,
        )


    @staticmethod
    def community_separation_metrics(
        graph: nx.Graph,
        communities: List[Set],
    ) -> Dict[str, float]:
        """
        Calculate how many edges remain inside communities
        and how many connect different communities.
        """

        membership = (
            CommunityAnalyzer.membership_mapping(
                communities
            )
        )

        internal_edges = 0
        external_edges = 0

        for source, target in graph.edges():

            source_community = membership.get(
                source
            )

            target_community = membership.get(
                target
            )

            if (
                source_community is not None
                and source_community
                == target_community
            ):
                internal_edges += 1

            else:
                external_edges += 1

        total_edges = (
            internal_edges
            + external_edges
        )

        internal_percentage = (
            internal_edges
            / total_edges
            * 100
            if total_edges > 0
            else 0.0
        )

        external_percentage = (
            external_edges
            / total_edges
            * 100
            if total_edges > 0
            else 0.0
        )

        return {
            "Internal Community Edges": (
                internal_edges
            ),
            "Between Community Edges": (
                external_edges
            ),
            "Internal Edge Percentage": round(
                internal_percentage,
                3,
            ),
            "Between Edge Percentage": round(
                external_percentage,
                3,
            ),
        }