"""
=========================================================
PageRank Damping Factor Sensitivity Evaluation
=========================================================

Evaluates how PageRank node rankings change when the
damping factor (alpha) is varied.

Baseline alpha = 0.85

Outputs:
- Spearman ranking correlation against baseline
- Top-10 overlap against baseline
- CSV results for dissertation evaluation
=========================================================
"""

from pathlib import Path
import sys

import networkx as nx
import pandas as pd
from scipy.stats import spearmanr


# =====================================================
# PROJECT ROOT
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from modules.cache_manager import CacheManager


# =====================================================
# CONFIGURATION
# =====================================================

DATASETS = {
    "Karate Club": "karate",
    "Email-Eu-Core": "email",
    "Web-Google": "web_google",
}

ALPHA_VALUES = [
    0.70,
    0.80,
    0.85,
    0.90,
    0.95,
]

BASELINE_ALPHA = 0.85

TOP_K = 10

OUTPUT_DIR = (
    CacheManager.BASE_DIR
    / "evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================
# COMPUTE PAGERANK
# =====================================================

def compute_pagerank(
    graph: nx.Graph,
    alpha: float,
) -> dict:
    """
    Compute PageRank using the selected damping factor.
    """

    return nx.pagerank(
        graph,
        alpha=alpha,
        max_iter=200,
        tol=1.0e-8,
    )


# =====================================================
# SPEARMAN RANKING STABILITY
# =====================================================

def ranking_spearman(
    baseline_scores: dict,
    comparison_scores: dict,
) -> float:
    """
    Compare two PageRank rankings using Spearman correlation.
    """

    common_nodes = sorted(
        set(baseline_scores.keys())
        & set(comparison_scores.keys()),
        key=str,
    )

    if len(common_nodes) < 2:
        return 0.0

    baseline_values = [
        baseline_scores[node]
        for node in common_nodes
    ]

    comparison_values = [
        comparison_scores[node]
        for node in common_nodes
    ]

    correlation, _ = spearmanr(
        baseline_values,
        comparison_values,
    )

    if pd.isna(correlation):
        return 0.0

    return float(correlation)


# =====================================================
# TOP-K OVERLAP
# =====================================================

def top_k_overlap(
    baseline_scores: dict,
    comparison_scores: dict,
    k: int = 10,
) -> float:
    """
    Calculate percentage overlap between the two Top-K sets.
    """

    baseline_top = {
        node
        for node, score in sorted(
            baseline_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:k]
    }

    comparison_top = {
        node
        for node, score in sorted(
            comparison_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:k]
    }

    overlap = len(
        baseline_top
        & comparison_top
    )

    return (
        overlap / k
    ) * 100


# =====================================================
# RUN ONE DATASET
# =====================================================

def evaluate_dataset(
    dataset_name: str,
    dataset_key: str,
) -> list:
    """
    Run PageRank sensitivity analysis for one dataset.
    """

    graph = CacheManager.load_graph(
        dataset_key
    )

    if graph is None:
        print(
            f"[SKIPPED] {dataset_name}: "
            "graph.pkl not found."
        )
        return []

    print(
        f"\nEvaluating {dataset_name}"
    )

    print(
        f"Nodes: {graph.number_of_nodes()}, "
        f"Edges: {graph.number_of_edges()}"
    )

    # ---------------------------------------------
    # BASELINE
    # ---------------------------------------------

    baseline_scores = compute_pagerank(
        graph,
        BASELINE_ALPHA,
    )

    records = []

    # ---------------------------------------------
    # TEST DIFFERENT ALPHA VALUES
    # ---------------------------------------------

    for alpha in ALPHA_VALUES:

        scores = compute_pagerank(
            graph,
            alpha,
        )

        spearman = ranking_spearman(
            baseline_scores,
            scores,
        )

        overlap = top_k_overlap(
            baseline_scores,
            scores,
            TOP_K,
        )

        record = {
            "Dataset": dataset_name,
            "Alpha": alpha,
            "Baseline Alpha": BASELINE_ALPHA,
            "Spearman Stability": round(
                spearman,
                6,
            ),
            "Top-10 Overlap (%)": round(
                overlap,
                2,
            ),
            "Nodes": graph.number_of_nodes(),
            "Edges": graph.number_of_edges(),
        }

        records.append(
            record
        )

        print(
            f"Alpha={alpha:.2f} | "
            f"Spearman={spearman:.6f} | "
            f"Top-10={overlap:.2f}%"
        )

    return records


# =====================================================
# MAIN
# =====================================================

def main():

    all_results = []

    print(
        "=" * 60
    )

    print(
        "PAGERANK DAMPING FACTOR SENSITIVITY"
    )

    print(
        f"Baseline alpha = {BASELINE_ALPHA}"
    )

    print(
        "=" * 60
    )

    for dataset_name, dataset_key in DATASETS.items():

        results = evaluate_dataset(
            dataset_name,
            dataset_key,
        )

        all_results.extend(
            results
        )

    if not all_results:

        print(
            "\nNo datasets were available."
        )
        return

    results_df = pd.DataFrame(
        all_results
    )

    output_file = (
        OUTPUT_DIR
        / "pagerank_sensitivity.csv"
    )

    results_df.to_csv(
        output_file,
        index=False,
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        "\nSaved to:"
    )

    print(
        output_file
    )


if __name__ == "__main__":
    main()