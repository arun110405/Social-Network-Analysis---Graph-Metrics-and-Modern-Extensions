"""
=========================================================
Network Explainability Module
=========================================================
Description:
Provides rule-based, human-readable explanations for:
- Degree Centrality
- Betweenness Centrality
- Closeness Centrality
- PageRank
- Overall node influence
- Dataset-level centrality patterns
=========================================================
"""

from typing import Dict, List

import pandas as pd


class ExplainabilityAnalyzer:
    """
    Creates human-readable explanations from cached
    network centrality results.
    """

    METRICS = [
        "Degree Centrality",
        "Betweenness Centrality",
        "Closeness Centrality",
        "PageRank",
    ]

    # =====================================================
    # VALIDATION
    # =====================================================

    @staticmethod
    def validate_dataframe(centrality_df: pd.DataFrame) -> None:
        """
        Validate that all required centrality columns exist.
        """

        required_columns = [
            "Node",
            *ExplainabilityAnalyzer.METRICS,
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in centrality_df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Centrality data is missing required columns: "
                + ", ".join(missing_columns)
            )

        if centrality_df.empty:
            raise ValueError(
                "Centrality data is empty"
            )

    # =====================================================
    # PERCENTILE POSITION
    # =====================================================

    @staticmethod
    def percentile_rank(
        centrality_df: pd.DataFrame,
        metric: str,
        value: float
    ) -> float:
        """
        Return the percentage of nodes whose value is less
        than or equal to the selected node's value.
        """

        if metric not in centrality_df.columns:
            raise ValueError(
                f"Metric not found: {metric}"
            )

        numeric_values = pd.to_numeric(
            centrality_df[metric],
            errors="coerce"
        ).dropna()

        if numeric_values.empty:
            return 0.0

        percentile = (
            numeric_values.le(value).mean() * 100
        )

        return round(float(percentile), 2)

    # =====================================================
    # CLASSIFICATION
    # =====================================================

    @staticmethod
    def classify_percentile(
        percentile: float
    ) -> str:
        """
        Convert a percentile value to an interpretation.
        """

        if percentile >= 95:
            return "exceptionally high"

        if percentile >= 80:
            return "high"

        if percentile >= 60:
            return "above average"

        if percentile >= 40:
            return "moderate"

        if percentile >= 20:
            return "below average"

        return "low"

    # =====================================================
    # METRIC EXPLANATIONS
    # =====================================================

    @staticmethod
    def explain_degree(
        value: float,
        percentile: float
    ) -> str:

        category = (
            ExplainabilityAnalyzer.classify_percentile(
                percentile
            )
        )

        if percentile >= 80:
            implication = (
                "The node has many direct connections and may "
                "act as a local hub for communication or information flow."
            )

        elif percentile >= 40:
            implication = (
                "The node has a moderate number of direct relationships "
                "and participates actively in the local network."
            )

        else:
            implication = (
                "The node has relatively few direct connections and is "
                "more peripheral in the network."
            )

        return (
            f"The degree centrality is {value:.6f}, which is "
            f"{category} and places the node at approximately the "
            f"{percentile:.2f} percentile. {implication}"
        )

    @staticmethod
    def explain_betweenness(
        value: float,
        percentile: float
    ) -> str:

        category = (
            ExplainabilityAnalyzer.classify_percentile(
                percentile
            )
        )

        if percentile >= 80:
            implication = (
                "The node frequently lies on shortest paths between "
                "other nodes and may act as a broker, bridge, or "
                "potential bottleneck."
            )

        elif percentile >= 40:
            implication = (
                "The node sometimes connects different areas of the "
                "network but is not one of the strongest bridging nodes."
            )

        else:
            implication = (
                "The node rarely connects otherwise separate regions "
                "and has limited control over shortest-path communication."
            )

        return (
            f"The betweenness centrality is {value:.6f}, which is "
            f"{category} at approximately the {percentile:.2f} "
            f"percentile. {implication}"
        )

    @staticmethod
    def explain_closeness(
        value: float,
        percentile: float
    ) -> str:

        category = (
            ExplainabilityAnalyzer.classify_percentile(
                percentile
            )
        )

        if percentile >= 80:
            implication = (
                "The node can reach other nodes through relatively short "
                "paths and may spread information efficiently."
            )

        elif percentile >= 40:
            implication = (
                "The node has moderate access to the rest of the network."
            )

        else:
            implication = (
                "The node is relatively distant from many other nodes, "
                "so information may require more steps to travel from it."
            )

        return (
            f"The closeness centrality is {value:.6f}, which is "
            f"{category} at approximately the {percentile:.2f} "
            f"percentile. {implication}"
        )

    @staticmethod
    def explain_pagerank(
        value: float,
        percentile: float
    ) -> str:

        category = (
            ExplainabilityAnalyzer.classify_percentile(
                percentile
            )
        )

        if percentile >= 80:
            implication = (
                "The node is connected to other important nodes and has "
                "strong global influence within the network."
            )

        elif percentile >= 40:
            implication = (
                "The node has moderate global importance and receives "
                "influence from reasonably important neighbours."
            )

        else:
            implication = (
                "The node receives limited influence from the network and "
                "is less globally prominent."
            )

        return (
            f"The PageRank score is {value:.8f}, which is "
            f"{category} at approximately the {percentile:.2f} "
            f"percentile. {implication}"
        )

    # =====================================================
    # NODE EXPLANATION
    # =====================================================

    @staticmethod
    def explain_node(
        centrality_df: pd.DataFrame,
        node
    ) -> Dict:
        """
        Generate a complete explanation for one node.
        """

        ExplainabilityAnalyzer.validate_dataframe(
            centrality_df
        )

        node_matches = centrality_df[
            centrality_df["Node"].astype(str)
            == str(node)
        ]

        if node_matches.empty:
            raise ValueError(
                f"Node not found: {node}"
            )

        row = node_matches.iloc[0]

        metric_values = {}
        percentiles = {}

        for metric in ExplainabilityAnalyzer.METRICS:

            value = float(row[metric])

            percentile = (
                ExplainabilityAnalyzer.percentile_rank(
                    centrality_df,
                    metric,
                    value
                )
            )

            metric_values[metric] = value
            percentiles[metric] = percentile

        explanations = {
            "Degree Centrality":
                ExplainabilityAnalyzer.explain_degree(
                    metric_values["Degree Centrality"],
                    percentiles["Degree Centrality"]
                ),

            "Betweenness Centrality":
                ExplainabilityAnalyzer.explain_betweenness(
                    metric_values["Betweenness Centrality"],
                    percentiles["Betweenness Centrality"]
                ),

            "Closeness Centrality":
                ExplainabilityAnalyzer.explain_closeness(
                    metric_values["Closeness Centrality"],
                    percentiles["Closeness Centrality"]
                ),

            "PageRank":
                ExplainabilityAnalyzer.explain_pagerank(
                    metric_values["PageRank"],
                    percentiles["PageRank"]
                ),
        }

        strongest_metric = max(
            percentiles,
            key=percentiles.get
        )

        weakest_metric = min(
            percentiles,
            key=percentiles.get
        )

        overall_score = sum(
            percentiles.values()
        ) / len(percentiles)

        if overall_score >= 80:
            overall_category = (
                "a highly influential node"
            )

        elif overall_score >= 60:
            overall_category = (
                "an above-average influential node"
            )

        elif overall_score >= 40:
            overall_category = (
                "a moderately influential node"
            )

        elif overall_score >= 20:
            overall_category = (
                "a below-average influential node"
            )

        else:
            overall_category = (
                "a peripheral node"
            )

        overall_explanation = (
            f"Node {node} is classified as {overall_category}. "
            f"Its strongest characteristic is {strongest_metric}, "
            f"where it is at approximately the "
            f"{percentiles[strongest_metric]:.2f} percentile. "
            f"Its weakest characteristic is {weakest_metric}, "
            f"at approximately the "
            f"{percentiles[weakest_metric]:.2f} percentile."
        )

        return {
            "Node": node,
            "Metric Values": metric_values,
            "Percentiles": percentiles,
            "Explanations": explanations,
            "Overall Score": round(
                overall_score,
                2
            ),
            "Overall Explanation":
                overall_explanation,
            "Strongest Metric":
                strongest_metric,
            "Weakest Metric":
                weakest_metric,
        }

    # =====================================================
    # TOP-N EXPLANATIONS
    # =====================================================

    @staticmethod
    def explain_top_nodes(
        centrality_df: pd.DataFrame,
        metric: str,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Return a ranked explanatory table for top nodes.
        """

        ExplainabilityAnalyzer.validate_dataframe(
            centrality_df
        )

        if metric not in ExplainabilityAnalyzer.METRICS:
            raise ValueError(
                f"Unsupported metric: {metric}"
            )

        ranked_df = (
            centrality_df
            .sort_values(
                metric,
                ascending=False
            )
            .head(top_n)
            .copy()
        )

        ranked_df["Rank"] = range(
            1,
            len(ranked_df) + 1
        )

        ranked_df["Percentile"] = ranked_df[
            metric
        ].apply(
            lambda value:
            ExplainabilityAnalyzer.percentile_rank(
                centrality_df,
                metric,
                float(value)
            )
        )

        ranked_df["Interpretation"] = ranked_df[
            "Percentile"
        ].apply(
            ExplainabilityAnalyzer.classify_percentile
        )

        return ranked_df[
            [
                "Rank",
                "Node",
                metric,
                "Percentile",
                "Interpretation",
            ]
        ]

    # =====================================================
    # DATASET SUMMARY
    # =====================================================

    @staticmethod
    def dataset_summary(
        centrality_df: pd.DataFrame,
        dataset_name: str
    ) -> List[str]:
        """
        Create dataset-level explanatory statements.
        """

        ExplainabilityAnalyzer.validate_dataframe(
            centrality_df
        )

        findings = []

        for metric in ExplainabilityAnalyzer.METRICS:

            highest_row = centrality_df.loc[
                centrality_df[metric].idxmax()
            ]

            mean_value = float(
                centrality_df[metric].mean()
            )

            median_value = float(
                centrality_df[metric].median()
            )

            findings.append(
                f"In {dataset_name}, Node "
                f"{highest_row['Node']} has the highest "
                f"{metric} score of "
                f"{float(highest_row[metric]):.6f}. "
                f"The network-wide mean is "
                f"{mean_value:.6f}, while the median is "
                f"{median_value:.6f}."
            )

        return findings