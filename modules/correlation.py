"""
=========================================================
Correlation Analysis Module
=========================================================

Description:
    This module performs statistical correlation analysis
    between centrality measures.

    Key functionality:
    - Spearman rank correlation
    - Correlation matrix generation
    - Agreement analysis between metrics

This directly supports research questions on:
    RQ1: Consistency of centrality measures
    RQ2: Agreement across network structures

=========================================================
"""

import pandas as pd
from scipy.stats import spearmanr

from utils.logger import logger


class CorrelationAnalyzer:
    """
    Performs correlation analysis between centrality metrics.
    """

    # =====================================================
    # Spearman Correlation (Pairwise)
    # =====================================================

    @staticmethod
    def spearman_pairwise(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute pairwise Spearman correlation between
        centrality measures.
        """

        logger.info("Computing Spearman correlation matrix")

        metrics = [
            "Degree Centrality",
            "Betweenness Centrality",
            "Closeness Centrality",
            "PageRank",
        ]

        correlation_matrix = pd.DataFrame(index=metrics, columns=metrics)

        for m1 in metrics:
            for m2 in metrics:

                corr, _ = spearmanr(df[m1], df[m2])

                correlation_matrix.loc[m1, m2] = corr

        correlation_matrix = correlation_matrix.astype(float)

        logger.info("Spearman correlation computation complete")

        return correlation_matrix

    # =====================================================
    # Overall Agreement Score
    # =====================================================

    @staticmethod
    def overall_agreement(corr_matrix: pd.DataFrame) -> float:
        """
        Compute overall agreement score across metrics.
        """

        import numpy as np

        # remove diagonal (self-correlation)
        values = corr_matrix.values
        mask = ~np.eye(len(values), dtype=bool)

        avg_corr = values[mask].mean()

        logger.info("Overall agreement score computed: %.4f", avg_corr)

        return float(avg_corr)

    # =====================================================
    # Rank Correlation Between Two Metrics
    # =====================================================

    @staticmethod
    def compare_metrics(df: pd.DataFrame, m1: str, m2: str) -> float:
        """
        Compare two centrality metrics using Spearman correlation.
        """

        if m1 not in df.columns or m2 not in df.columns:
            raise ValueError("Invalid metric names provided")

        corr, _ = spearmanr(df[m1], df[m2])

        logger.info("Correlation between %s and %s = %.4f", m1, m2, corr)

        return float(corr)