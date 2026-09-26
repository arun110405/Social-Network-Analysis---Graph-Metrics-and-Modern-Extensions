"""
=========================================================
Tests for Graph Loader
=========================================================
"""

import networkx as nx

from modules.loader import GraphLoader


def test_karate_club_loads_successfully():
    """
    The built-in Karate Club dataset should load as a
    NetworkX graph with the expected number of nodes
    and edges.
    """

    graph = GraphLoader.load_karate_club()

    assert graph is not None

    assert isinstance(
        graph,
        nx.Graph
    )

    assert graph.number_of_nodes() == 34

    assert graph.number_of_edges() == 78


def test_karate_club_is_not_empty():
    """
    Loaded graph should contain both nodes and edges.
    """

    graph = GraphLoader.load_karate_club()

    assert graph.number_of_nodes() > 0

    assert graph.number_of_edges() > 0


def test_karate_club_contains_expected_nodes():
    """
    NetworkX Karate Club nodes are numbered from 0 to 33.
    """

    graph = GraphLoader.load_karate_club()

    assert 0 in graph.nodes

    assert 33 in graph.nodes