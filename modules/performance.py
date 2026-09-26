"""
=========================================================
Performance Evaluation Module
=========================================================

Description:
Records stage-level execution times for graph analysis
and persistent cache loading.
=========================================================
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.config import Config


class PerformanceManager:
    """
    Stores and analyses application performance results.
    """

    PERFORMANCE_DIR = Config.RESULTS_DIR / "performance"

    PERFORMANCE_FILE = (
        PERFORMANCE_DIR / "benchmark.csv"
    )

    COLUMNS = [
        "Timestamp",
        "Dataset",
        "Run Type",
        "Raw Graph Load Time",
        "Graph Cache Load Time",
        "Preprocessing Time",
        "Statistics Time",
        "Centrality Time",
        "Correlation Time",
        "Analysis Cache Load Time",
        "Total Time",
        "Nodes",
        "Edges",
    ]

    # =====================================================
    # DIRECTORY
    # =====================================================

    @staticmethod
    def ensure_directory() -> Path:

        PerformanceManager.PERFORMANCE_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        return PerformanceManager.PERFORMANCE_DIR

    # =====================================================
    # RECORD RESULT
    # =====================================================

    @staticmethod
    def record(
        dataset: str,
        run_type: str,
        total_time: float,
        raw_graph_load_time: float = 0.0,
        graph_cache_load_time: float = 0.0,
        preprocessing_time: float = 0.0,
        statistics_time: float = 0.0,
        centrality_time: float = 0.0,
        correlation_time: float = 0.0,
        analysis_cache_load_time: float = 0.0,
        nodes: Optional[int] = None,
        edges: Optional[int] = None,
    ) -> Path:
        """
        Append one benchmark record.
        """

        PerformanceManager.ensure_directory()

        record = {
            "Timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),
            "Dataset": dataset,
            "Run Type": run_type,
            "Raw Graph Load Time": round(
                float(raw_graph_load_time),
                6
            ),
            "Graph Cache Load Time": round(
                float(graph_cache_load_time),
                6
            ),
            "Preprocessing Time": round(
                float(preprocessing_time),
                6
            ),
            "Statistics Time": round(
                float(statistics_time),
                6
            ),
            "Centrality Time": round(
                float(centrality_time),
                6
            ),
            "Correlation Time": round(
                float(correlation_time),
                6
            ),
            "Analysis Cache Load Time": round(
                float(analysis_cache_load_time),
                6
            ),
            "Total Time": round(
                float(total_time),
                6
            ),
            "Nodes": nodes,
            "Edges": edges,
        }

        new_row = pd.DataFrame(
            [record],
            columns=PerformanceManager.COLUMNS
        )

        if (
            PerformanceManager.PERFORMANCE_FILE.exists()
            and
            PerformanceManager.PERFORMANCE_FILE.stat().st_size > 0
        ):

            try:

                existing_df = pd.read_csv(
                    PerformanceManager.PERFORMANCE_FILE
                )

                benchmark_df = pd.concat(
                    [existing_df, new_row],
                    ignore_index=True
                )

            except (
                pd.errors.EmptyDataError,
                pd.errors.ParserError
            ):

                benchmark_df = new_row

        else:

            benchmark_df = new_row

        temporary_file = (
            PerformanceManager.PERFORMANCE_FILE
            .with_suffix(".tmp")
        )

        benchmark_df.to_csv(
            temporary_file,
            index=False
        )

        temporary_file.replace(
            PerformanceManager.PERFORMANCE_FILE
        )

        return PerformanceManager.PERFORMANCE_FILE

    # =====================================================
    # LOAD RESULTS
    # =====================================================

    @staticmethod
    def load() -> pd.DataFrame:

        if (
            not PerformanceManager.PERFORMANCE_FILE.exists()
            or
            PerformanceManager.PERFORMANCE_FILE.stat().st_size == 0
        ):

            return pd.DataFrame(
                columns=PerformanceManager.COLUMNS
            )

        try:

            dataframe = pd.read_csv(
                PerformanceManager.PERFORMANCE_FILE
            )

        except (
            pd.errors.EmptyDataError,
            pd.errors.ParserError
        ):

            return pd.DataFrame(
                columns=PerformanceManager.COLUMNS
            )

        numeric_columns = [
            "Raw Graph Load Time",
            "Graph Cache Load Time",
            "Preprocessing Time",
            "Statistics Time",
            "Centrality Time",
            "Correlation Time",
            "Analysis Cache Load Time",
            "Total Time",
            "Nodes",
            "Edges",
        ]

        for column in numeric_columns:

            if column not in dataframe.columns:
                dataframe[column] = 0.0

            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce"
            ).fillna(0.0)

        return dataframe

    # =====================================================
    # DATASET FILTER
    # =====================================================

    @staticmethod
    def filter_dataset(
        dataframe: pd.DataFrame,
        dataset: str
    ) -> pd.DataFrame:

        if dataframe.empty:
            return dataframe.copy()

        return dataframe[
            dataframe["Dataset"] == dataset
        ].copy()

    # =====================================================
    # AVERAGE TIMES
    # =====================================================

    @staticmethod
    def average_by_run_type(
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:

        if dataframe.empty:
            return pd.DataFrame()

        timing_columns = [
            "Raw Graph Load Time",
            "Graph Cache Load Time",
            "Preprocessing Time",
            "Statistics Time",
            "Centrality Time",
            "Correlation Time",
            "Analysis Cache Load Time",
            "Total Time",
        ]

        available_columns = [
            column
            for column in timing_columns
            if column in dataframe.columns
        ]

        summary = (
            dataframe
            .groupby(
                ["Dataset", "Run Type"],
                as_index=False
            )[available_columns]
            .mean()
        )

        for column in available_columns:

            summary[column] = summary[
                column
            ].round(4)

        return summary

    # =====================================================
    # CACHE IMPROVEMENT
    # =====================================================

    @staticmethod
    def cache_improvement(
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:

        if dataframe.empty:
            return pd.DataFrame()

        rows = []

        for dataset in dataframe[
            "Dataset"
        ].dropna().unique():

            dataset_df = dataframe[
                dataframe["Dataset"] == dataset
            ]

            first_runs = dataset_df[
                dataset_df["Run Type"]
                == "First Run"
            ]

            cached_runs = dataset_df[
                dataset_df["Run Type"]
                == "Cached Run"
            ]

            if (
                first_runs.empty
                or
                cached_runs.empty
            ):
                continue

            first_average = float(
                first_runs["Total Time"].mean()
            )

            cached_average = float(
                cached_runs["Total Time"].mean()
            )

            saved_seconds = (
                first_average - cached_average
            )

            improvement = (
                (saved_seconds / first_average) * 100
                if first_average > 0
                else 0.0
            )

            speedup = (
                first_average / cached_average
                if cached_average > 0
                else 0.0
            )

            rows.append({
                "Dataset": dataset,
                "First Run Average (s)": round(
                    first_average,
                    4
                ),
                "Cached Run Average (s)": round(
                    cached_average,
                    4
                ),
                "Time Saved (s)": round(
                    saved_seconds,
                    4
                ),
                "Improvement (%)": round(
                    improvement,
                    2
                ),
                "Speed-up": round(
                    speedup,
                    2
                ),
            })

        return pd.DataFrame(rows)

    # =====================================================
    # LATEST RESULT
    # =====================================================

    @staticmethod
    def latest_run(
        dataframe: pd.DataFrame,
        dataset: str
    ):

        dataset_df = (
            PerformanceManager.filter_dataset(
                dataframe,
                dataset
            )
        )

        if dataset_df.empty:
            return None

        return dataset_df.iloc[-1]

    # =====================================================
    # CLEAR HISTORY
    # =====================================================

    @staticmethod
    def clear() -> bool:

        if PerformanceManager.PERFORMANCE_FILE.exists():

            PerformanceManager.PERFORMANCE_FILE.unlink()

            return True

        return False