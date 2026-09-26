import pandas as pd
import json
from pathlib import Path

from utils.logger import logger


class ExportManager:

    @staticmethod
    def export_csv(df: pd.DataFrame, filename: str, output_dir: str = "results/csv") -> str:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        file_path = Path(output_dir) / f"{filename}.csv"
        df.to_csv(file_path, index=False)
        logger.info("CSV exported: %s", file_path)
        return str(file_path)

    @staticmethod
    def export_json(data: dict, filename: str, output_dir: str = "results/json") -> str:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        file_path = Path(output_dir) / f"{filename}.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        logger.info("JSON exported: %s", file_path)
        return str(file_path)
    @staticmethod
    def export_plotly(fig, filename: str, output_dir: str = "results/plots") -> str:
        from pathlib import Path

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        file_path = Path(output_dir) / f"{filename}.html"

        fig.write_html(file_path)

        return str(file_path)


    @staticmethod
    def plot_exists(filename: str, output_dir: str = "results/plots"):

        from pathlib import Path

        file_path = Path(output_dir) / f"{filename}.html"

        return file_path.exists()