"""
=========================================================
Visual Analytics Module
=========================================================
Description:
Creates statistical visualisations from cached centrality
and correlation results.

Features:
- Spearman correlation heatmap
- Centrality distributions
- Degree versus PageRank scatter plot
- Top-node comparison chart
=========================================================
"""

from typing import List

import matplotlib.pyplot as plt
import pandas as pd


class VisualAnalytics:
    """
    Creates analytical figures from cached result tables.
    """

    CENTRALITY_METRICS: List[str] = [
        "Degree Centrality",
        "Betweenness Centrality",
        "Closeness Centrality",
        "PageRank",
    ]

    # =====================================================
    # VALIDATION
    # =====================================================

    @staticmethod
    def validate_centrality(
        centrality_df: pd.DataFrame
    ) -> None:
        """
        Validate the cached centrality dataframe.
        """

        required_columns = [
            "Node",
            *VisualAnalytics.CENTRALITY_METRICS,
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in centrality_df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing centrality columns: "
                + ", ".join(missing_columns)
            )

        if centrality_df.empty:
            raise ValueError(
                "Centrality dataframe is empty"
            )

    # =====================================================
    # CORRELATION HEATMAP
    # =====================================================

    @staticmethod
    def correlation_heatmap(
        correlation_df: pd.DataFrame
    ):
        """
        Create a Spearman correlation heatmap.
        """

        if correlation_df.empty:
            raise ValueError(
                "Correlation dataframe is empty"
            )

        numeric_df = correlation_df.apply(
            pd.to_numeric,
            errors="coerce"
        )

        figure, axis = plt.subplots(
            figsize=(8, 6)
        )

        image = axis.imshow(
            numeric_df.values,
            vmin=-1,
            vmax=1,
            aspect="auto"
        )

        axis.set_xticks(
            range(len(numeric_df.columns))
        )

        axis.set_xticklabels(
            numeric_df.columns,
            rotation=40,
            ha="right"
        )

        axis.set_yticks(
            range(len(numeric_df.index))
        )

        axis.set_yticklabels(
            numeric_df.index
        )

        for row_index in range(
            len(numeric_df.index)
        ):

            for column_index in range(
                len(numeric_df.columns)
            ):

                value = numeric_df.iloc[
                    row_index,
                    column_index
                ]

                if pd.notna(value):

                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center"
                    )

        axis.set_title(
            "Spearman Correlation Heatmap"
        )

        figure.colorbar(
            image,
            ax=axis,
            label="Correlation"
        )

        figure.tight_layout()

        return figure

    # =====================================================
    # CENTRALITY DISTRIBUTION
    # =====================================================

    @staticmethod
    def centrality_distribution(
        centrality_df: pd.DataFrame,
        metric: str,
        bins: int = 30
    ):
        """
        Create a histogram for one centrality metric.
        """

        VisualAnalytics.validate_centrality(
            centrality_df
        )

        if metric not in (
            VisualAnalytics.CENTRALITY_METRICS
        ):
            raise ValueError(
                f"Unsupported metric: {metric}"
            )

        values = pd.to_numeric(
            centrality_df[metric],
            errors="coerce"
        ).dropna()

        figure, axis = plt.subplots(
            figsize=(9, 5)
        )

        axis.hist(
            values,
            bins=bins,
            edgecolor="black"
        )

        axis.set_title(
            f"{metric} Distribution"
        )

        axis.set_xlabel(metric)
        axis.set_ylabel("Number of Nodes")

        axis.grid(
            axis="y",
            alpha=0.3
        )

        figure.tight_layout()

        return figure

    # =====================================================
    # DEGREE VS PAGERANK
    # =====================================================

    @staticmethod
    def degree_pagerank_scatter(
        centrality_df: pd.DataFrame
    ):
        """
        Create a scatter plot comparing degree centrality
        with PageRank.
        """

        VisualAnalytics.validate_centrality(
            centrality_df
        )

        x_values = pd.to_numeric(
            centrality_df[
                "Degree Centrality"
            ],
            errors="coerce"
        )

        y_values = pd.to_numeric(
            centrality_df["PageRank"],
            errors="coerce"
        )

        valid_rows = (
            x_values.notna()
            & y_values.notna()
        )

        figure, axis = plt.subplots(
            figsize=(8, 6)
        )

        axis.scatter(
            x_values[valid_rows],
            y_values[valid_rows],
            alpha=0.65
        )

        axis.set_title(
            "Degree Centrality versus PageRank"
        )

        axis.set_xlabel(
            "Degree Centrality"
        )

        axis.set_ylabel(
            "PageRank"
        )

        axis.grid(
            alpha=0.3
        )

        figure.tight_layout()

        return figure

    # =====================================================
    # TOP NODE COMPARISON
    # =====================================================

    @staticmethod
    def top_nodes_chart(
        centrality_df: pd.DataFrame,
        metric: str,
        top_n: int = 10
    ):
        """
        Create a horizontal bar chart for the top nodes.
        """

        VisualAnalytics.validate_centrality(
            centrality_df
        )

        if metric not in (
            VisualAnalytics.CENTRALITY_METRICS
        ):
            raise ValueError(
                f"Unsupported metric: {metric}"
            )

        top_nodes = (
            centrality_df[
                ["Node", metric]
            ]
            .sort_values(
                metric,
                ascending=False
            )
            .head(top_n)
            .sort_values(
                metric,
                ascending=True
            )
        )

        labels = (
            top_nodes["Node"]
            .astype(str)
        )

        values = pd.to_numeric(
            top_nodes[metric],
            errors="coerce"
        )

        figure, axis = plt.subplots(
            figsize=(9, 6)
        )

        axis.barh(
            labels,
            values
        )

        axis.set_title(
            f"Top {top_n} Nodes by {metric}"
        )

        axis.set_xlabel(metric)
        axis.set_ylabel("Node")

        axis.grid(
            axis="x",
            alpha=0.3
        )

        figure.tight_layout()

        return figure

    # =====================================================
    # DESCRIPTIVE STATISTICS
    # =====================================================

    @staticmethod
    def descriptive_statistics(
        centrality_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Return descriptive statistics for centrality metrics.
        """

        VisualAnalytics.validate_centrality(
            centrality_df
        )

        summary = (
            centrality_df[
                VisualAnalytics.CENTRALITY_METRICS
            ]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
            .describe()
            .transpose()
            .reset_index()
            .rename(
                columns={
                    "index": "Metric"
                }
            )
        )

        return summary