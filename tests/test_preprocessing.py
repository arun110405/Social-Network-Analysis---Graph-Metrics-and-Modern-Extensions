"""
=========================================================
Tests for Graph Preprocessing
=========================================================
"""

import networkx as nx

from modules.preprocessing import GraphPreprocessor


def test_preprocessing_removes_self_loops():
    """
    Self-loops should be removed during preprocessing.
    """

    graph = nx.Graph()

    graph.add_edges_from(
        [
            (1, 1),
            (1, 2),
            (2, 3),
            (3, 4),
        ]
    )

    processed_graph = (
        GraphPreprocessor.preprocess(
            graph
        )
    )

    assert (
        nx.number_of_selfloops(
            processed_graph
        )
        == 0
    )


def test_preprocessing_keeps_connected_graph():
    """
    A graph that is already connected should remain
    connected after preprocessing.
    """

    graph = nx.path_graph(5)

    processed_graph = (
        GraphPreprocessor.preprocess(
            graph
        )
    )

    assert nx.is_connected(
        processed_graph
    )

    assert (
        processed_graph.number_of_nodes()
        == 5
    )


def test_preprocessing_keeps_largest_component():
    """
    When multiple components exist, preprocessing should
    retain the largest connected component.
    """

    graph = nx.Graph()

    # Large component: 4 nodes
    graph.add_edges_from(
        [
            (1, 2),
            (2, 3),
            (3, 4),
        ]
    )

    # Smaller component: 2 nodes
    graph.add_edge(
        10,
        11
    )

    processed_graph = (
        GraphPreprocessor.preprocess(
            graph
        )
    )

    assert (
        processed_graph.number_of_nodes()
        == 4
    )

    assert set(
        processed_graph.nodes()
    ) == {
        1,
        2,
        3,
        4,
    }


def test_preprocessing_handles_directed_graph():
    """
    Directed graphs should remain valid after preprocessing.
    """

    graph = nx.DiGraph()

    graph.add_edges_from(
        [
            (1, 2),
            (2, 3),
            (3, 1),
        ]
    )

    processed_graph = (
        GraphPreprocessor.preprocess(
            graph
        )
    )

    assert processed_graph is not None

    assert (
        processed_graph.number_of_nodes()
        == 3
    )