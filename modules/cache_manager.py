import json
import pickle
from pathlib import Path

import pandas as pd
from utils.config import Config

class CacheManager:

    BASE_DIR = Config.RESULTS_DIR

    # =====================================================
    # DATASET DIRECTORY
    # =====================================================

    @staticmethod
    def get_path(dataset_name: str) -> Path:
        path = CacheManager.BASE_DIR / dataset_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    # =====================================================
    # FILE CHECK
    # =====================================================

    @staticmethod
    def exists(dataset_name: str, filename: str) -> bool:
        file_path = CacheManager.get_path(dataset_name) / filename
        return file_path.exists() and file_path.stat().st_size > 0

    # =====================================================
    # CSV
    # =====================================================

    @staticmethod
    def load_csv(dataset_name: str, filename: str) -> pd.DataFrame:
        file_path = CacheManager.get_path(dataset_name) / filename
        return pd.read_csv(file_path)

    @staticmethod
    def save_csv(
        dataset_name: str,
        filename: str,
        dataframe: pd.DataFrame
    ) -> Path:

        file_path = CacheManager.get_path(dataset_name) / filename
        dataframe.to_csv(file_path, index=False)
        return file_path

    # =====================================================
    # JSON
    # =====================================================

    @staticmethod
    def load_json(dataset_name: str, filename: str):
        file_path = CacheManager.get_path(dataset_name) / filename

        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def save_json(dataset_name: str, filename: str, data) -> Path:
        file_path = CacheManager.get_path(dataset_name) / filename

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

        return file_path

    # =====================================================
    # PROCESSED GRAPH CACHE
    # =====================================================

    @staticmethod
    def graph_exists(dataset_name: str) -> bool:
        graph_path = CacheManager.get_path(dataset_name) / "graph.pkl"

        return (
            graph_path.exists()
            and graph_path.stat().st_size > 0
        )

    @staticmethod
    def save_graph(dataset_name: str, graph) -> Path:
        """
        Save a processed NetworkX graph safely.
        """

        if graph is None:
            raise ValueError(
                f"Cannot save an empty graph for dataset: {dataset_name}"
            )

        graph_path = CacheManager.get_path(dataset_name) / "graph.pkl"
        temp_path = graph_path.with_suffix(".tmp")

        with open(temp_path, "wb") as file:
            pickle.dump(
                graph,
                file,
                protocol=pickle.HIGHEST_PROTOCOL
            )

        temp_path.replace(graph_path)

        return graph_path


    @staticmethod
    def load_graph(dataset_name: str):
        """
        Load a processed NetworkX graph from disk.
        """

        graph_path = CacheManager.get_path(dataset_name) / "graph.pkl"

        if (
            not graph_path.exists()
            or graph_path.stat().st_size == 0
        ):
            return None

        try:
            with open(graph_path, "rb") as file:
                return pickle.load(file)

        except (
            EOFError,
            pickle.UnpicklingError,
            AttributeError,
            OSError
        ):
            graph_path.unlink(missing_ok=True)
            return None

    # =====================================================
    # DATASET COMPARISON CACHE
    # =====================================================

    @staticmethod
    def comparison_exists() -> bool:
        file_path = (
            CacheManager.BASE_DIR
            / "comparison"
            / "comparison.csv"
        )

        return file_path.exists() and file_path.stat().st_size > 0

    @staticmethod
    def save_comparison(dataframe: pd.DataFrame) -> Path:
        directory = CacheManager.BASE_DIR / "comparison"
        directory.mkdir(parents=True, exist_ok=True)

        file_path = directory / "comparison.csv"
        dataframe.to_csv(file_path, index=False)

        return file_path

    @staticmethod
    def load_comparison() -> pd.DataFrame:
        file_path = (
            CacheManager.BASE_DIR
            / "comparison"
            / "comparison.csv"
        )

        return pd.read_csv(file_path)