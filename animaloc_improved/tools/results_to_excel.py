from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from animaloc_improved.tools.massage_results import conf_matrix_to_pct, massage_metrics

app = typer.Typer(pretty_exceptions_show_locals=False, no_args_is_help=True)


@app.command()
def to_excel(
    results_dir: Path = typer.Argument(
        help="directory where evaluation results where stored in csv format"
    ),
    excel_out_dir: Path = typer.Option(
        ..., "-o", "--out-dir", help="directory where excel formatted outputs will be stored"
    ),
    decimals: int = typer.Option(1, help="decimal points to use in excel outputs"),
) -> None:
    conf_matrix = pd.read_csv(results_dir / "confusion_matrix.csv")
    conf_matrix_pct = conf_matrix_to_pct(conf_matrix)
    print(f"Confusion matrix [%]\n{conf_matrix_pct.to_markdown()}")

    metrics = pd.read_csv("data/test_results/frcnn-resnet50-full-imgs/metrics_results.csv")
    massaged_metrics = massage_metrics(metrics)
    print(f"Metrics:\n{massaged_metrics.to_markdown()}")

    excel_out_dir.mkdir(parents=True, exist_ok=True)
    conf_matrix_out_fp = excel_out_dir / "conf_matrix_pct.xlsx"
    logger.info(f"Writing confusion matrix to: {conf_matrix_out_fp}")
    conf_matrix_pct.to_excel(conf_matrix_out_fp, index=False)

    metrics_out_fp = excel_out_dir / "metrics_results.xlsx"
    logger.info(f"Writing metrics to: {metrics_out_fp}")
    massaged_metrics.to_excel(conf_matrix_out_fp, index=False)


if __name__ == "__main__":
    app()
