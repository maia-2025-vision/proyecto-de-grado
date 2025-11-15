from pathlib import Path

import pandas as pd


def summarize_points_gt(fp: Path) -> None:
    print(f"archivo: {fp!s}")
    df = pd.read_csv(fp)
    cols = list(df.columns)
    print(f"\n{len(df)} filas - {len(cols)} columnas: [{', '.join(cols)}]")

    print("\n## Número de valores únicos por columna:")
    print(pd.Series({col: df[col].nunique() for col in cols}).to_markdown())

    print("\n## Conteos de nulos por columna:")
    print(df.isnull().sum().reset_index().to_markdown())

    print("\n## Estadísticas de coordenadas de puntos (x, y)")
    print(df[["x", "y"]].describe().to_markdown())

    print("\n Frecuencias de labels:")
    freqs = df[["labels"]].value_counts().reset_index()
    freqs["pct"] = freqs["count"] / freqs["count"].sum() * 100
    print(freqs.sort_values("labels").to_markdown())
