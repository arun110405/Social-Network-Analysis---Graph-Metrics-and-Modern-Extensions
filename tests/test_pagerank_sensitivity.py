import networkx as nx

from evaluation.pagerank_sensitivity import (
    compute_pagerank,
    ranking_spearman,
    top_k_overlap,
)


# =====================================================
# PAGERANK OUTPUT
# =====================================================

def test_pagerank_contains_all_nodes():

    graph = nx.karate_club_graph()

    scores = compute_pagerank(
        graph,
        alpha=0.85,
    )

    assert len(scores) == graph.number_of_nodes()

    assert set(scores.keys()) == set(
        graph.nodes()
    )


# =====================================================
# PAGERANK SUM
# =====================================================

def test_pagerank_scores_sum_to_one():

    graph = nx.karate_club_graph()

    scores = compute_pagerank(
        graph,
        alpha=0.85,
    )

    assert abs(
        sum(scores.values()) - 1.0
    ) < 1e-6


# =====================================================
# IDENTICAL RANKING
# =====================================================

def test_identical_pagerank_has_perfect_spearman():

    graph = nx.karate_club_graph()

    scores = compute_pagerank(
        graph,
        alpha=0.85,
    )

    correlation = ranking_spearman(
        scores,
        scores,
    )

    assert abs(
        correlation - 1.0
    ) < 1e-9


# =====================================================
# IDENTICAL TOP-K
# =====================================================

def test_identical_pagerank_has_full_top10_overlap():

    graph = nx.karate_club_graph()

    scores = compute_pagerank(
        graph,
        alpha=0.85,
    )

    overlap = top_k_overlap(
        scores,
        scores,
        k=10,
    )

    assert overlap == 100.0


# =====================================================
# DIFFERENT ALPHA VALUES
# =====================================================

def test_different_alpha_values_are_valid():

    graph = nx.karate_club_graph()

    for alpha in [
        0.70,
        0.80,
        0.85,
        0.90,
        0.95,
    ]:

        scores = compute_pagerank(
            graph,
            alpha=alpha,
        )

        assert len(scores) == 34

        assert all(
            score >= 0
            for score in scores.values()
        )

        assert abs(
            sum(scores.values()) - 1.0
        ) < 1e-6