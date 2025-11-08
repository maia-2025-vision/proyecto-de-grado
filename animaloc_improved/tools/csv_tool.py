"""Manipulate csvs in various ways"""

from pathlib import Path

import pandas as pd
import typer
from loguru import logger
from typer import Typer

app = Typer()


@app.command(
    "concat",
    help="Concatenate two csvs. Only columns in first csv will be output. "
    "If second csv does not have all columns  ",
)
def concat(
    csv1: Path,
    csv2: Path,
    out: Path = typer.Option(..., "-o", "--out", help="Output file"),
):
    df1 = pd.read_csv(csv1)
    log_info("csv1", df1)
    df2 = pd.read_csv(csv2)
    log_info("csv2", df2)

    cols1 = list(df1.columns)
    for col1 in cols1:
        assert col1 in df2.columns, f"{col1} not in {df2.columns}!"

    df2_cols1 = df2.loc[:, cols1]
    out_df = pd.concat([df1, df2_cols1])
    log_info("out_df", out_df)
    out_df.to_csv(out, index=False)
    logger.info(f"output file: {out!s}")


@app.command("dtypes", help="Show columns and pandas dtypes for a csv")
def dtypes(csv: Path) -> None:
    df = pd.read_csv(csv)
    log_info("csv", df)

    df_dtypes = pd.DataFrame(df.dtypes).reset_index()
    df_dtypes.columns = ["column", "dtype"]
    print(df_dtypes.to_markdown())


@app.command("describe", help="Show columns and pandas dtypes for a csv")
def describe(csv: Path) -> None:
    df = pd.read_csv(csv)

    df_desc = df.describe(include="all")
    print(df_desc.to_markdown())


def log_info(name: str, df: pd.DataFrame) -> None:
    logger.info(f"{name}, {len(df)} lines, columns= {', '.join(df.columns)}")


if __name__ == "__main__":
    app()
