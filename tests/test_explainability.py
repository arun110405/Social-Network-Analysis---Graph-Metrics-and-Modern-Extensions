"""
=========================================================
Tests for Explainability Analysis
=========================================================

Validates:
- required output structure
- consistency with computed centrality metrics
- percentile calculations
- strongest/weakest metric identification
- top-node ranking
- input validation
=========================================================
"""

import networkx as nx
import pandas as pd
import pytest

from modules.centrality import CentralityAnalyzer
from modules.explainability import ExplainabilityAnalyzer


# =====================================================
# TEST DATA
# =====================================================

@pytest.fixture
def centrality_df():
    """
    Generate deterministic centrality results from a small
    NetworkX path graph.
    """

    graph = nx.path_graph(5)

    return CentralityAnalyzer.compute_all(
        graph
    )


# =====================================================
# OUTPUT STRUCTURE
# =====================================================

def test_explain_node_returns_required_structure(
    centrality_df
):
    """
    explain_node() should return all required fields.
    """

    result = ExplainabilityAnalyzer.explain_node(
        centrality_df,
        node=2
    )

    required_keys = {
        "Node",
        "Metric Values",
        "Percentiles",
        "Explanations",
        "Overall Score",
        "Overall Explanation",
        "Strongest Metric",
        "Weakest Metric",
    }

    assert required_keys.issubset(
        result.keys()
    )


# =====================================================
# METRIC STRUCTURE
# =====================================================

def test_explanation_contains_all_metrics(
    centrality_df
):
    """
    All four centrality measures should appear in the
    metric values, percentile values and explanations.
    """

    result = ExplainabilityAnalyzer.explain_node(
        centrality_df,
        node=2
    )

    expected_metrics = set(
        ExplainabilityAnalyzer.METRICS
    )

    assert set(
        result["Metric Values"].keys()
    ) == expected_metrics

    assert set(
        result["Percentiles"].keys()
    ) == expected_metrics

    assert set(
        result["Explanations"].keys()
    ) == expected_metrics


# =====================================================
# COMPUTED METRIC CONSISTENCY
# =====================================================

def test_explanation_metric_values_match_centrality(
    centrality_df
):
    """
    Values used in the explanation must exactly match the
    computed centrality results for the selected node.
    """

    selected_node = 2

    result = ExplainabilityAnalyzer.explain_node(
        centrality_df,
        node=selected_node
    )

    row = centrality_df[
        centrality_df["Node"] == selected_node
    ].iloc[0]

    for metric in ExplainabilityAnalyzer.METRICS:

        expected_value = float(
            row[metric]
        )

        explanation_value = result[
            "Metric Values"
        ][metric]

        assert explanation_value == pytest.approx(
            expected_value,
            rel=1e-9,
            abs=1e-12,
        )


# =====================================================
# PERCENTILE CONSISTENCY
# =====================================================

def test_percentiles_match_manual_calculation(
    centrality_df
):
    """
    Percentiles reported by the explanation should match
    an independently calculated percentile.
    """

    selected_node = 2

    result = ExplainabilityAnalyzer.explain_node(
        centrality_df,
        node=selected_node
    )

    row = centrality_df[
        centrality_df["Node"] == selected_node
    ].iloc[0]

    for metric in ExplainabilityAnalyzer.METRICS:

        value = float(
            row[metric]
        )

        expected_percentile = round(
            (
                centrality_df[metric]
                .le(value)
                .mean()
                * 100
            ),
            2,
        )

        actual_percentile = result[
            "Percentiles"
        ][metric]

        assert actual_percentile == pytest.approx(
            expected_percentile
        )


# =====================================================
# PERCENTILE RANGE
# =====================================================

def test_percentiles_are_within_valid_range(
    centrality_df
):
    """
    Percentile values must remain between 0 and 100.
    """

    result = ExplainabilityAnalyzer.explain_node(
        centrality_df,
        node=2
    )

    for percentile in result[
        "Percentiles"
    ].values():

        assert 0 <= percentile <= 100


# =====================================================
# OVERALL SCORE CONSISTENCY
# =====================================================

def test_overall_score_matches_mean_percentile(
    centrality_df
):
    """
    Overall influence score should equal the mean of the
    four metric percentiles.
    """

    result = ExplainabilityAnalyzer.explain_node(
        centrality_df,
        node=2
    )

    expected_score = round(
        sum(
            result["Percentiles"].values()
        )
        / len(
            result["Percentiles"]
        ),
        2,
    )

    assert result[
        "Overall Score"
    ] == pytest.approx(
        expected_score
    )


# =====================================================
# STRONGEST METRIC
# =====================================================

def test_strongest_metric_is_correct(
    centrality_df
):
    """
    The strongest metric should correspond to the highest
    percentile value.
    """

    result = ExplainabilityAnalyzer.explain_node(
        centrality_df,
        node=2
    )

    expected = max(
        result["Percentiles"],
        key=result["Percentiles"].get
    )

    assert result[
        "Strongest Metric"
    ] == expected


# =====================================================
# WEAKEST METRIC
# =====================================================

def test_weakest_metric_is_correct(
    centrality_df
):
    """
    The weakest metric should correspond to the lowest
    percentile value.
    """

    result = ExplainabilityAnalyzer.explain_node(
        centrality_df,
        node=2
    )

    expected = min(
        result["Percentiles"],
        key=result["Percentiles"].get
    )

    assert result[
        "Weakest Metric"
    ] == expected


# =====================================================
# EXPLANATION TEXT CONTAINS COMPUTED VALUE
# =====================================================

def test_explanation_contains_metric_value(
    centrality_df
):
    """
    Human-readable explanations should include the
    calculated metric values used to generate them.
    """

    result = ExplainabilityAnalyzer.explain_node(
        centrality_df,
        node=2
    )

    degree_value = result[
        "Metric Values"
    ][
        "Degree Centrality"
    ]

    degree_explanation = result[
        "Explanations"
    ][
        "Degree Centrality"
    ]

    assert (
        f"{degree_value:.6f}"
        in degree_explanation
    )


# =====================================================
# CLASSIFICATION BOUNDARIES
# =====================================================

@pytest.mark.parametrize(
    "percentile,expected",
    [
        (96, "exceptionally high"),
        (85, "high"),
        (65, "above average"),
        (50, "moderate"),
        (25, "below average"),
        (10, "low"),
    ],
)
def test_percentile_classification(
    percentile,
    expected
):
    """
    Percentile classifications should follow the defined
    deterministic thresholds.
    """

    result = (
        ExplainabilityAnalyzer.classify_percentile(
            percentile
        )
    )

    assert result == expected


# =====================================================
# TOP NODE RANKING
# =====================================================

def test_top_nodes_are_sorted_correctly(
    centrality_df
):
    """
    explain_top_nodes() should rank nodes in descending
    order of the selected metric.
    """

    result = (
        ExplainabilityAnalyzer.explain_top_nodes(
            centrality_df,
            metric="Degree Centrality",
            top_n=3
        )
    )

    scores = result[
        "Degree Centrality"
    ].tolist()

    assert scores == sorted(
        scores,
        reverse=True
    )

    assert result[
        "Rank"
    ].tolist() == [
        1,
        2,
        3
    ]


# =====================================================
# DATASET SUMMARY
# =====================================================

def test_dataset_summary_returns_one_result_per_metric(
    centrality_df
):
    """
    Dataset summary should generate one explanatory
    statement for each centrality measure.
    """

    results = (
        ExplainabilityAnalyzer.dataset_summary(
            centrality_df,
            "Test Network"
        )
    )

    assert len(results) == len(
        ExplainabilityAnalyzer.METRICS
    )

    assert all(
        isinstance(
            statement,
            str
        )
        and len(statement) > 0
        for statement in results
    )


# =====================================================
# MISSING NODE VALIDATION
# =====================================================

def test_missing_node_raises_error(
    centrality_df
):
    """
    Requesting a node that does not exist should fail
    clearly.
    """

    with pytest.raises(
        ValueError,
        match="Node not found"
    ):

        ExplainabilityAnalyzer.explain_node(
            centrality_df,
            node=999
        )


# =====================================================
# MISSING COLUMN VALIDATION
# =====================================================

def test_missing_centrality_column_raises_error():
    """
    Missing required centrality columns should be detected.
    """

    invalid_df = pd.DataFrame(
        {
            "Node": [1, 2],
            "Degree Centrality": [
                0.5,
                0.5
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing required columns"
    ):

        ExplainabilityAnalyzer.validate_dataframe(
            invalid_df
        )


# =====================================================
# EMPTY DATA VALIDATION
# =====================================================

def test_empty_dataframe_raises_error():
    """
    Empty centrality data should not generate explanations.
    """

    empty_df = pd.DataFrame(
        columns=[
            "Node",
            "Degree Centrality",
            "Betweenness Centrality",
            "Closeness Centrality",
            "PageRank",
        ]
    )

    with pytest.raises(
        ValueError,
        match="Centrality data is empty"
    ):

        ExplainabilityAnalyzer.validate_dataframe(
            empty_df
        )