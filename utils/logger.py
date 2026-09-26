"""
=========================================================
Logger Module
=========================================================

Description:
    This module configures the application's logging system.

    Logging provides a structured way to record application
    events, warnings, debugging information, and errors.

=========================================================
"""

import logging
from pathlib import Path


def setup_logger(
    logger_name: str = "SNA_Project",
    log_level: int = logging.INFO
) -> logging.Logger:
    """
    Configure and return a logger instance.

    Parameters
    ----------
    logger_name : str
        Name of the logger.

    log_level : int
        Logging level.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    logger = logging.getLogger(logger_name)

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(log_level)

    # --------------------------------------------------
    # Create logs directory
    # --------------------------------------------------

    log_directory = Path("logs")
    log_directory.mkdir(exist_ok=True)

    log_file = log_directory / "project.log"

    # --------------------------------------------------
    # Log format
    # --------------------------------------------------

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(module)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # --------------------------------------------------
    # Console handler
    # --------------------------------------------------

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # --------------------------------------------------
    # File handler
    # --------------------------------------------------

    file_handler = logging.FileHandler(
        filename=log_file,
        encoding="utf-8"
    )

    file_handler.setFormatter(formatter)

    # --------------------------------------------------
    # Attach handlers
    # --------------------------------------------------

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# ----------------------------------------------------------
# Default project logger
# ----------------------------------------------------------

logger = setup_logger()