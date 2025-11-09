import numpy as np
import pandas as pd


def conf_matrix_to_pct(conf_matrix: pd.DataFrame, decimals: int = 1) -> pd.DataFrame:
    conf_matrix = conf_matrix.copy()
    conf_matrix.columns.values[0] = "Especie"
    conf_matrix = conf_matrix.set_index("Especie")
    vals = conf_matrix.values
    row_sums = vals.sum(axis=1, keepdims=True)
    vals_pct = np.round(vals / row_sums * 100, decimals)
    ret = pd.DataFrame(vals_pct, index=conf_matrix.index, columns=conf_matrix.columns)
    # ret['Conteo Total'] = row_sums.astype(int)
    # cols_out = ['Conteo Total'] + list(ret.columns.values[:-1])

    return ret


def massage_metrics(
    metrics: pd.DataFrame, decimals: int = 1, drop_cols: list[str] | None = None
) -> pd.DataFrame:
    ret = metrics.copy()
    for col in ["recall", "precision", "confusion", "f1_score", "ap"]:
        ret[col] = (ret[col] * 100.0).round(decimals)

    for col in ["mae", "mse", "rmse"]:
        ret[col] = ret[col].round(decimals)

    drop_cols = drop_cols if drop_cols is not None else ["class", "mse", "rmse"]

    renames = {
        "species": "Especie",
        "n": "N",
        "recall": "Rec.[%]",
        "precision": "Prec.[%]",
        "confusion": "Conf.[%]",
        "f1_score": "F1[%]",
        "mae": "MAE",
        "mse": "MSE",
        "rmse": "RMSE",
        "ap": "AP [%]",
    }

    renames_subset = {k: v for k, v in renames.items() if k not in drop_cols}
    ret = ret.drop(columns=drop_cols).rename(columns=renames_subset)

    return ret
