"""
=========================================================
Tests for Correlation Analysis
=========================================================
"""

import pandas as pd
import pytest

from modules.correlation import CorrelationAnalyzer


def create_test_centrality_dataframe():
    """
    Create deterministic centrality data for testing.
    """

    return pd.DataFrame(
        {
            "Node": [
                1,
                2,
                3,
                4,
                5,
            ],

            "Degree Centrality": [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
            ],

            "Betweenness Centrality": [
                0.5,
                0.4,
                0.3,
                0.2,
                0.1,
            ],

            "Closeness Centrality": [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
            ],

            "PageRank": [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
        }
    )


def test_spearman_returns_dataframe():
    """
    Spearman analysis should return a Pandas DataFrame.
    """

    dataframe = (
        create_test_centrality_dataframe()
    )

    correlation = (
        CorrelationAnalyzer.spearman_pairwise(
            dataframe
        )
    )

    assert isinstance(
        correlation,
        pd.DataFrame
    )


def test_identical_rankings_have_perfect_correlation():
    """
    Identical rankings should have Spearman correlation 1.
    """

    dataframe = (
        create_test_centrality_dataframe()
    )

    correlation = (
        CorrelationAnalyzer.spearman_pairwise(
            dataframe
        )
    )

    value = correlation.loc[
        "Degree Centrality",
        "PageRank"
    ]

    assert value == pytest.approx(
        1.0,
        abs=1e-9,
    )


def test_reverse_rankings_have_negative_correlation():
    """
    Completely reversed rankings should have correlation -1.
    """

    dataframe = (
        create_test_centrality_dataframe()
    )

    correlation = (
        CorrelationAnalyzer.spearman_pairwise(
            dataframe
        )
    )

    value = correlation.loc[
        "Degree Centrality",
        "Betweenness Centrality"
    ]

    assert value == pytest.approx(
        -1.0,
        abs=1e-9,
    )


def test_correlation_diagonal_equals_one():
    """
    Every metric should correlate perfectly with itself.
    """

    dataframe = (
        create_test_centrality_dataframe()
    )

    correlation = (
        CorrelationAnalyzer.spearman_pairwise(
            dataframe
        )
    )

    for column in correlation.columns:

        assert correlation.loc[
            column,
            column
        ] == pytest.approx(
            1.0,
            abs=1e-9,
        )