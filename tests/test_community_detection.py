"""
=========================================================
Tests for Community Detection
=========================================================
"""

import networkx as nx

from modules.community_detection import CommunityAnalyzer


def test_greedy_detects_communities():
    """
    Greedy Modularity should detect at least one community.
    """

    graph = nx.karate_club_graph()

    communities = (
        CommunityAnalyzer.detect_communities(
            graph,
            algorithm="greedy",
        )
    )

    assert communities is not None

    assert len(communities) > 0


def test_louvain_detects_communities():
    """
    Louvain should detect at least one community.
    """

    graph = nx.karate_club_graph()

    communities = (
        CommunityAnalyzer.detect_communities(
            graph,
            algorithm="louvain",
            seed=42,
        )
    )

    assert communities is not None

    assert len(communities) > 0


def test_label_propagation_detects_communities():
    """
    Label Propagation should detect at least one community.
    """

    graph = nx.karate_club_graph()

    communities = (
        CommunityAnalyzer.detect_communities(
            graph,
            algorithm="label_propagation",
            seed=42,
        )
    )

    assert communities is not None

    assert len(communities) > 0


def test_communities_cover_all_nodes():
    """
    Every node should belong to exactly one detected community.
    """

    graph = nx.karate_club_graph()

    communities = (
        CommunityAnalyzer.detect_communities(
            graph,
            algorithm="greedy",
        )
    )

    detected_nodes = set().union(
        *communities
    )

    assert detected_nodes == set(
        graph.nodes()
    )


def test_communities_do_not_overlap():
    """
    Community partitions should not assign the same node
    to multiple communities.
    """

    graph = nx.karate_club_graph()

    communities = (
        CommunityAnalyzer.detect_communities(
            graph,
            algorithm="greedy",
        )
    )

    all_nodes = []

    for community in communities:

        all_nodes.extend(
            community
        )

    assert len(all_nodes) == len(
        set(all_nodes)
    )


def test_modularity_is_valid():
    """
    Modularity should lie within its valid theoretical range.
    """

    graph = nx.karate_club_graph()

    communities = (
        CommunityAnalyzer.detect_communities(
            graph,
            algorithm="greedy",
        )
    )

    modularity = (
        CommunityAnalyzer.calculate_modularity(
            graph,
            communities,
        )
    )

    assert -1.0 <= modularity <= 1.0


def test_membership_dataframe_contains_every_node():
    """
    Membership table should contain one row for every node.
    """

    graph = nx.karate_club_graph()

    communities = (
        CommunityAnalyzer.detect_communities(
            graph,
            algorithm="greedy",
        )
    )

    membership = (
        CommunityAnalyzer.membership_dataframe(
            communities
        )
    )

    assert len(membership) == (
        graph.number_of_nodes()
    )

    assert set(
        membership["Node"]
    ) == set(
        graph.nodes()
    )


def test_community_balance_metrics_are_returned():
    """
    Community balance evaluation should return the expected
    metrics.
    """

    graph = nx.karate_club_graph()

    communities = (
        CommunityAnalyzer.detect_communities(
            graph,
            algorithm="greedy",
        )
    )

    metrics = (
        CommunityAnalyzer.community_balance_metrics(
            communities
        )
    )

    assert (
        "Community Size CV"
        in metrics
    )

    assert (
        "Normalised Size Entropy"
        in metrics
    )

    assert (
        metrics["Community Size CV"]
        >= 0
    )


def test_karate_ground_truth_evaluation():
    """
    Karate Club should support NMI and ARI comparison against
    the known club split.
    """

    graph = nx.karate_club_graph()

    communities = (
        CommunityAnalyzer.detect_communities(
            graph,
            algorithm="greedy",
        )
    )

    result = (
        CommunityAnalyzer.evaluate_karate_ground_truth(
            graph,
            communities,
        )
    )

    assert result is not None

    assert "Ground Truth NMI" in result

    assert "Ground Truth ARI" in result

    assert (
        0.0
        <= result["Ground Truth NMI"]
        <= 1.0
    )