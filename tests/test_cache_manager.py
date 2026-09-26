"""
=========================================================
Tests for Cache Manager
=========================================================
"""

import networkx as nx
import pandas as pd

from modules.cache_manager import CacheManager


def test_csv_cache_round_trip(
    tmp_path,
    monkeypatch,
):
    """
    Saving and loading a CSV should reproduce the same
    DataFrame.
    """

    monkeypatch.setattr(
        CacheManager,
        "BASE_DIR",
        tmp_path,
    )

    dataframe = pd.DataFrame(
        {
            "Node": [
                1,
                2,
                3,
            ],

            "Score": [
                0.2,
                0.5,
                0.9,
            ],
        }
    )

    CacheManager.save_csv(
        "test_dataset",
        "test.csv",
        dataframe,
    )

    loaded = CacheManager.load_csv(
        "test_dataset",
        "test.csv",
    )

    pd.testing.assert_frame_equal(
        dataframe,
        loaded,
    )


def test_json_cache_round_trip(
    tmp_path,
    monkeypatch,
):
    """
    JSON data should be saved and restored correctly.
    """

    monkeypatch.setattr(
        CacheManager,
        "BASE_DIR",
        tmp_path,
    )

    data = {
        "nodes": 34,
        "edges": 78,
        "density": 0.1,
    }

    CacheManager.save_json(
        "test_dataset",
        "stats.json",
        data,
    )

    loaded = CacheManager.load_json(
        "test_dataset",
        "stats.json",
    )

    assert loaded == data


def test_graph_cache_round_trip(
    tmp_path,
    monkeypatch,
):
    """
    A NetworkX graph should survive pickle save/load.
    """

    monkeypatch.setattr(
        CacheManager,
        "BASE_DIR",
        tmp_path,
    )

    original_graph = nx.path_graph(
        5
    )

    CacheManager.save_graph(
        "test_dataset",
        original_graph,
    )

    loaded_graph = CacheManager.load_graph(
        "test_dataset"
    )

    assert loaded_graph is not None

    assert set(
        loaded_graph.nodes()
    ) == set(
        original_graph.nodes()
    )

    assert set(
        loaded_graph.edges()
    ) == set(
        original_graph.edges()
    )


def test_exists_detects_saved_file(
    tmp_path,
    monkeypatch,
):
    """
    CacheManager.exists() should detect non-empty saved files.
    """

    monkeypatch.setattr(
        CacheManager,
        "BASE_DIR",
        tmp_path,
    )

    dataframe = pd.DataFrame(
        {
            "A": [
                1,
                2,
            ]
        }
    )

    CacheManager.save_csv(
        "test_dataset",
        "file.csv",
        dataframe,
    )

    assert CacheManager.exists(
        "test_dataset",
        "file.csv",
    )


def test_missing_graph_returns_none(
    tmp_path,
    monkeypatch,
):
    """
    Loading a graph that has not been cached should safely
    return None.
    """

    monkeypatch.setattr(
        CacheManager,
        "BASE_DIR",
        tmp_path,
    )

    graph = CacheManager.load_graph(
        "missing_dataset"
    )

    assert graph is None