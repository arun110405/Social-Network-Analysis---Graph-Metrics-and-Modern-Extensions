"""
=========================================================
Configuration Module
=========================================================

Project:
    Social Network Analysis – Graph Metrics and Modern Extensions

Author:
    Arun Prasath Velkutty

University:
    University of Liverpool
    MSc Computer Science Dissertation (2025/26)

Description:
    This module centralises all configurable settings used
    throughout the application.

    Keeping configuration in one location improves
    maintainability, readability and scalability.

=========================================================
"""

from pathlib import Path


class Config:
    """
    Global configuration settings for the project.
    """

    # =====================================================
    # Project Root Directory
    # =====================================================

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # =====================================================
    # Dataset Directories
    # =====================================================

    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

    # =====================================================
    # Result Directories
    # =====================================================

    RESULTS_DIR = PROJECT_ROOT / "results"

    CSV_RESULTS_DIR = RESULTS_DIR / "csv"
    IMAGE_RESULTS_DIR = RESULTS_DIR / "images"
    REPORT_RESULTS_DIR = RESULTS_DIR / "reports"
    JSON_RESULTS_DIR = RESULTS_DIR / "json"

    # =====================================================
    # Dashboard Settings
    # =====================================================

    APP_TITLE = "Social Network Analysis Dashboard"

    APP_LAYOUT = "wide"

    SIDEBAR_STATE = "expanded"

    # =====================================================
    # Graph Visualisation
    # =====================================================

    DEFAULT_NODE_SIZE = 20

    DEFAULT_EDGE_WIDTH = 1

    DEFAULT_NODE_COLOUR = "skyblue"

    DEFAULT_EDGE_COLOUR = "gray"

    FIGURE_WIDTH = 1000

    FIGURE_HEIGHT = 700

    # =====================================================
    # PageRank Configuration
    # =====================================================

    PAGERANK_DAMPING_FACTOR = 0.85

    PAGERANK_MAX_ITERATIONS = 100

    PAGERANK_TOLERANCE = 1e-6

    # =====================================================
    # Community Detection
    # =====================================================

    COMMUNITY_RANDOM_STATE = 42

    # =====================================================
    # Explainability Module
    # =====================================================

    MAX_EXPLANATION_LENGTH = 500

    DEFAULT_MODEL = "gpt-4.1"

    # =====================================================
    # Export Settings
    # =====================================================

    CSV_ENCODING = "utf-8"

    JSON_INDENT = 4

    IMAGE_DPI = 300

    # =====================================================
    # Random Seed
    # =====================================================

    RANDOM_SEED = 42