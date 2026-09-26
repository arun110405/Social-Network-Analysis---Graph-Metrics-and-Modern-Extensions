"""
=========================================================
Tests for Centrality Analysis
=========================================================
"""

import networkx as nx
import pytest

from modules.centrality import CentralityAnalyzer


def test_compute_all_returns_required_columns():
    """
    CentralityAnalyzer.compute_all() should return all
    required centrality measures.
    """

    graph = nx.path_graph(5)

    results = (
        CentralityAnalyzer.compute_all(
            graph
        )
    )

    required_columns = {
        "Node",
        "Degree Centrality",
        "Betweenness Centrality",
        "Closeness Centrality",
        "PageRank",
    }

    assert required_columns.issubset(
        set(results.columns)
    )


def test_compute_all_returns_every_node():
    """
    Every graph node should appear in the centrality table.
    """

    graph = nx.path_graph(5)

    results = (
        CentralityAnalyzer.compute_all(
            graph
        )
    )

    assert len(results) == 5

    assert set(
        results["Node"]
    ) == set(
        graph.nodes()
    )


def test_degree_centrality_matches_networkx():
    """
    Degree Centrality calculated by the project should match
    NetworkX on a deterministic test graph.
    """

    graph = nx.path_graph(5)

    results = (
        CentralityAnalyzer.compute_all(
            graph
        )
    )

    expected = nx.degree_centrality(
        graph
    )

    for node, expected_value in expected.items():

        actual_value = results.loc[
            results["Node"] == node,
            "Degree Centrality"
        ].iloc[0]

        assert actual_value == pytest.approx(
            expected_value,
            rel=1e-6,
            abs=1e-9,
        )


def test_closeness_centrality_matches_networkx():
    """
    Closeness Centrality should match NetworkX.
    """

    graph = nx.path_graph(5)

    results = (
        CentralityAnalyzer.compute_all(
            graph
        )
    )

    expected = nx.closeness_centrality(
        graph
    )

    for node, expected_value in expected.items():

        actual_value = results.loc[
            results["Node"] == node,
            "Closeness Centrality"
        ].iloc[0]

        assert actual_value == pytest.approx(
            expected_value,
            rel=1e-6,
            abs=1e-9,
        )


def test_pagerank_values_sum_to_one():
    """
    PageRank values should approximately form a probability
    distribution and sum to one.
    """

    graph = nx.path_graph(5)

    results = (
        CentralityAnalyzer.compute_all(
            graph
        )
    )

    pagerank_sum = results[
        "PageRank"
    ].sum()

    assert pagerank_sum == pytest.approx(
        1.0,
        rel=1e-5,
        abs=1e-6,
    )


def test_middle_node_has_highest_betweenness():
    """
    For a 5-node path, the middle node should have the
    highest Betweenness Centrality.
    """

    graph = nx.path_graph(5)

    results = (
        CentralityAnalyzer.compute_all(
            graph
        )
    )

    top_node = results.sort_values(
        by="Betweenness Centrality",
        ascending=False,
    ).iloc[0]["Node"]

    assert top_node == 2

def test_betweenness_centrality_matches_networkx():
    """
    Betweenness Centrality calculated by the project should
    match NetworkX on a deterministic test graph.
    """

    graph = nx.path_graph(5)

    results = (
        CentralityAnalyzer.compute_all(
            graph
        )
    )

    expected = nx.betweenness_centrality(
        graph
    )

    for node, expected_value in expected.items():

        actual_value = results.loc[
            results["Node"] == node,
            "Betweenness Centrality"
        ].iloc[0]

        assert actual_value == pytest.approx(
            expected_value,
            rel=1e-6,
            abs=1e-9,
        )


def test_pagerank_matches_networkx():
    """
    PageRank calculated by the project should match
    NetworkX on a deterministic test graph.
    """

    graph = nx.path_graph(5)

    results = (
        CentralityAnalyzer.compute_all(
            graph
        )
    )

    expected = nx.pagerank(
        graph
    )

    for node, expected_value in expected.items():

        actual_value = results.loc[
            results["Node"] == node,
            "PageRank"
        ].iloc[0]

        assert actual_value == pytest.approx(
            expected_value,
            rel=1e-6,
            abs=1e-9,
        )