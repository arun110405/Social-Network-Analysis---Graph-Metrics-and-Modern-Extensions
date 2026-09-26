"""
=========================================================
Graph Robustness Analysis Module
=========================================================
Description:
Evaluates the stability of centrality measures under
controlled structural changes to a graph.

Supported perturbations:
- Random node removal
- Random edge removal

Evaluation measures:
- Spearman rank correlation
- Top-k node overlap
- Graph size and density change
=========================================================
"""

from typing import Dict, List, Optional, Tuple
import random

import networkx as nx
import pandas as pd
from scipy.stats import spearmanr

from utils.config import Config


class RobustnessAnalyzer:
    """
    Performs graph perturbation and centrality-stability analysis.
    """

    SUPPORTED_METRICS = [
        "Degree Centrality",
        "Betweenness Centrality",
        "Closeness Centrality",
        "PageRank",
    ]

    # =====================================================
    # GRAPH VALIDATION
    # =====================================================

    @staticmethod
    def validate_graph(graph: nx.Graph) -> None:
        """
        Validate that the graph can be analysed.
        """

        if graph is None:
            raise ValueError("Graph cannot be None")

        if graph.number_of_nodes() == 0:
            raise ValueError("Graph contains no nodes")

        if graph.number_of_edges() == 0:
            raise ValueError("Graph contains no edges")

    # =====================================================
    # GRAPH PREPARATION
    # =====================================================

    @staticmethod
    def prepare_graph(graph: nx.Graph) -> nx.Graph:
        """
        Return a clean copy suitable for robustness analysis.
        """

        RobustnessAnalyzer.validate_graph(graph)

        prepared = graph.copy()

        prepared.remove_edges_from(
            nx.selfloop_edges(prepared)
        )

        isolates = list(nx.isolates(prepared))
        prepared.remove_nodes_from(isolates)

        if prepared.number_of_nodes() == 0:
            raise ValueError(
                "Graph became empty during preparation"
            )

        return prepared

    # =====================================================
    # RANDOM NODE REMOVAL
    # =====================================================

    @staticmethod
    def remove_random_nodes(
        graph: nx.Graph,
        removal_percentage: float,
        seed: int = Config.RANDOM_SEED
    ) -> nx.Graph:
        """
        Remove a percentage of nodes randomly.
        """

        prepared = RobustnessAnalyzer.prepare_graph(graph)

        if not 0 < removal_percentage < 100:
            raise ValueError(
                "Removal percentage must be between 0 and 100"
            )

        nodes = list(prepared.nodes())

        removal_count = max(
            1,
            int(
                len(nodes)
                * removal_percentage
                / 100
            )
        )

        removal_count = min(
            removal_count,
            len(nodes) - 1
        )

        random_generator = random.Random(seed)

        selected_nodes = random_generator.sample(
            nodes,
            removal_count
        )

        perturbed = prepared.copy()
        perturbed.remove_nodes_from(selected_nodes)

        isolates = list(nx.isolates(perturbed))
        perturbed.remove_nodes_from(isolates)

        if perturbed.number_of_nodes() == 0:
            raise ValueError(
                "Node removal produced an empty graph"
            )

        return perturbed

    # =====================================================
    # RANDOM EDGE REMOVAL
    # =====================================================

    @staticmethod
    def remove_random_edges(
        graph: nx.Graph,
        removal_percentage: float,
        seed: int = Config.RANDOM_SEED
    ) -> nx.Graph:
        """
        Remove a percentage of edges randomly.
        """

        prepared = RobustnessAnalyzer.prepare_graph(graph)

        if not 0 < removal_percentage < 100:
            raise ValueError(
                "Removal percentage must be between 0 and 100"
            )

        edges = list(prepared.edges())

        removal_count = max(
            1,
            int(
                len(edges)
                * removal_percentage
                / 100
            )
        )

        removal_count = min(
            removal_count,
            len(edges) - 1
        )

        random_generator = random.Random(seed)

        selected_edges = random_generator.sample(
            edges,
            removal_count
        )

        perturbed = prepared.copy()
        perturbed.remove_edges_from(selected_edges)

        isolates = list(nx.isolates(perturbed))
        perturbed.remove_nodes_from(isolates)

        if perturbed.number_of_edges() == 0:
            raise ValueError(
                "Edge removal produced a graph with no edges"
            )

        return perturbed

    # =====================================================
    # CENTRALITY COMPUTATION
    # =====================================================

    @staticmethod
    def compute_metric(
        graph: nx.Graph,
        metric: str,
        betweenness_samples: int = 200
    ) -> Dict:
        """
        Compute one selected centrality metric.
        """

        if metric not in RobustnessAnalyzer.SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported metric: {metric}"
            )

        if metric == "Degree Centrality":

            return nx.degree_centrality(graph)

        if metric == "Betweenness Centrality":

            node_count = graph.number_of_nodes()

            if node_count > 500:

                return nx.betweenness_centrality(
                    graph,
                    k=min(
                        betweenness_samples,
                        node_count
                    ),
                    normalized=True,
                    seed=Config.RANDOM_SEED
                )

            return nx.betweenness_centrality(
                graph,
                normalized=True
            )

        if metric == "Closeness Centrality":

            return nx.closeness_centrality(graph)

        if metric == "PageRank":

            return nx.pagerank(
                graph,
                alpha=Config.PAGERANK_DAMPING_FACTOR,
                max_iter=Config.PAGERANK_MAX_ITERATIONS,
                tol=Config.PAGERANK_TOLERANCE
            )

        raise ValueError(
            f"Metric could not be computed: {metric}"
        )

    # =====================================================
    # CACHED ORIGINAL SCORES
    # =====================================================

    @staticmethod
    def scores_from_dataframe(
        centrality_df: pd.DataFrame,
        metric: str
    ) -> Dict:
        """
        Convert cached centrality results into a node-score map.
        """

        if "Node" not in centrality_df.columns:
            raise ValueError(
                "Centrality dataframe has no Node column"
            )

        if metric not in centrality_df.columns:
            raise ValueError(
                f"Centrality dataframe has no {metric} column"
            )

        return {
            str(node): float(score)
            for node, score in zip(
                centrality_df["Node"],
                centrality_df[metric]
            )
        }

    # =====================================================
    # SPEARMAN STABILITY
    # =====================================================

    @staticmethod
    def spearman_stability(
        original_scores: Dict,
        perturbed_scores: Dict
    ) -> Tuple[float, int]:
        """
        Calculate Spearman correlation on nodes present in both graphs.
        """

        original_string_keys = {
            str(node): float(value)
            for node, value in original_scores.items()
        }

        perturbed_string_keys = {
            str(node): float(value)
            for node, value in perturbed_scores.items()
        }

        common_nodes = sorted(
            set(original_string_keys)
            & set(perturbed_string_keys)
        )

        if len(common_nodes) < 2:
            return 0.0, len(common_nodes)

        original_values = [
            original_string_keys[node]
            for node in common_nodes
        ]

        perturbed_values = [
            perturbed_string_keys[node]
            for node in common_nodes
        ]

        correlation, _ = spearmanr(
            original_values,
            perturbed_values
        )

        if pd.isna(correlation):
            correlation = 0.0

        return float(correlation), len(common_nodes)

    # =====================================================
    # TOP-K OVERLAP
    # =====================================================

    @staticmethod
    def top_k_overlap(
        original_scores: Dict,
        perturbed_scores: Dict,
        k: int = 10
    ) -> float:
        """
        Calculate percentage overlap between top-k ranked nodes.
        """

        original_ranked = sorted(
            original_scores,
            key=original_scores.get,
            reverse=True
        )[:k]

        perturbed_ranked = sorted(
            perturbed_scores,
            key=perturbed_scores.get,
            reverse=True
        )[:k]

        original_top = {
            str(node)
            for node in original_ranked
        }

        perturbed_top = {
            str(node)
            for node in perturbed_ranked
        }

        if not original_top:
            return 0.0

        overlap = (
            len(original_top & perturbed_top)
            / len(original_top)
        ) * 100

        return round(float(overlap), 2)

    # =====================================================
    # STRUCTURAL SUMMARY
    # =====================================================

    @staticmethod
    def graph_summary(graph: nx.Graph) -> Dict:
        """
        Return structural values used in the experiment.
        """

        return {
            "Nodes": graph.number_of_nodes(),
            "Edges": graph.number_of_edges(),
            "Density": round(
                nx.density(graph),
                8
            ),
            "Components": (
                nx.number_weakly_connected_components(graph)
                if graph.is_directed()
                else nx.number_connected_components(graph)
            ),
        }

    # =====================================================
    # SINGLE EXPERIMENT
    # =====================================================

    @staticmethod
    def run_experiment(
        original_graph: nx.Graph,
        original_centrality_df: pd.DataFrame,
        perturbation_type: str,
        removal_percentage: float,
        metrics: Optional[List[str]] = None,
        top_k: int = 10,
        seed: int = Config.RANDOM_SEED
    ) -> Tuple[pd.DataFrame, nx.Graph]:
        """
        Run one robustness experiment.
        """

        if metrics is None:
            metrics = [
                "Degree Centrality",
                "PageRank",
            ]

        prepared_graph = (
            RobustnessAnalyzer.prepare_graph(
                original_graph
            )
        )

        if perturbation_type == "Node Removal":

            perturbed_graph = (
                RobustnessAnalyzer.remove_random_nodes(
                    prepared_graph,
                    removal_percentage,
                    seed
                )
            )

        elif perturbation_type == "Edge Removal":

            perturbed_graph = (
                RobustnessAnalyzer.remove_random_edges(
                    prepared_graph,
                    removal_percentage,
                    seed
                )
            )

        else:

            raise ValueError(
                "Perturbation type must be "
                "'Node Removal' or 'Edge Removal'"
            )

        original_summary = (
            RobustnessAnalyzer.graph_summary(
                prepared_graph
            )
        )

        perturbed_summary = (
            RobustnessAnalyzer.graph_summary(
                perturbed_graph
            )
        )

        results = []

        for metric in metrics:

            original_scores = (
                RobustnessAnalyzer.scores_from_dataframe(
                    original_centrality_df,
                    metric
                )
            )

            perturbed_scores = (
                RobustnessAnalyzer.compute_metric(
                    perturbed_graph,
                    metric
                )
            )

            correlation, common_nodes = (
                RobustnessAnalyzer.spearman_stability(
                    original_scores,
                    perturbed_scores
                )
            )

            overlap = (
                RobustnessAnalyzer.top_k_overlap(
                    original_scores,
                    perturbed_scores,
                    top_k
                )
            )

            results.append({
                "Perturbation": perturbation_type,
                "Removal Percentage": removal_percentage,
                "Metric": metric,
                "Spearman Stability": round(
                    correlation,
                    6
                ),
                f"Top-{top_k} Overlap (%)": overlap,
                "Common Nodes": common_nodes,
                "Original Nodes": original_summary["Nodes"],
                "Perturbed Nodes": perturbed_summary["Nodes"],
                "Original Edges": original_summary["Edges"],
                "Perturbed Edges": perturbed_summary["Edges"],
                "Original Density": original_summary["Density"],
                "Perturbed Density": perturbed_summary["Density"],
                "Perturbed Components": perturbed_summary["Components"],
            })

        return pd.DataFrame(results), perturbed_graph

    # =====================================================
    # MULTIPLE PERCENTAGE EXPERIMENT
    # =====================================================

    @staticmethod
    def run_series(
        original_graph: nx.Graph,
        original_centrality_df: pd.DataFrame,
        perturbation_type: str,
        percentages: List[float],
        metrics: List[str],
        top_k: int = 10,
        seed: int = Config.RANDOM_SEED
    ) -> pd.DataFrame:
        """
        Run robustness experiments for several removal levels.
        """

        experiment_frames = []

        for percentage in percentages:

            result_df, _ = (
                RobustnessAnalyzer.run_experiment(
                    original_graph=original_graph,
                    original_centrality_df=original_centrality_df,
                    perturbation_type=perturbation_type,
                    removal_percentage=percentage,
                    metrics=metrics,
                    top_k=top_k,
                    seed=seed
                )
            )

            experiment_frames.append(
                result_df
            )

        if not experiment_frames:
            return pd.DataFrame()

        return pd.concat(
            experiment_frames,
            ignore_index=True
        )

    # =====================================================
    # STABILITY INTERPRETATION
    # =====================================================

    @staticmethod
    def interpret_stability(
        correlation: float
    ) -> str:
        """
        Convert Spearman stability to a readable interpretation.
        """

        if correlation >= 0.90:
            return "Very high stability"

        if correlation >= 0.75:
            return "High stability"

        if correlation >= 0.50:
            return "Moderate stability"

        if correlation >= 0.25:
            return "Low stability"

        return "Very low stability"