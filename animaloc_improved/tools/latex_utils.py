from typing import Literal

import pandas as pd

# Model latex code
r"""
\begin{table}[h!]
%% increase table row spacing, adjust to taste
\renewcommand{\arraystretch}{1.3}
% if using array.sty, it might be a good idea to tweak the value of
% \extrarowheight as needed to properly center the text within the cells
\label{table_example}
\centering
%% Some packages, such as MDW tools, offer better commands for making tables
%% than the plain LaTeX2e tabular which is used here.
\begin{tabular}{c c}
\hline
\textbf{One} & \textbf{Two}\\   % col headers
\hline
Three & Four\\  % row
\hline
\end{tabular}
\caption{An Example of a Table}

\end{table}
"""


def df_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str | None = None,
    placement: str = "h!",  # Force here, other options are t, b ....
    bold_col_headers: bool = True,
    row_index: bool = False,
    bold_row_headers: bool = True,
    array_stretch: float = 1.3,
    centering: bool = True,
    col_sep: Literal[" ", "|"] = " ",
) -> str:
    parts = []
    parts.append(f"\\begin{{table}}[{placement}]")

    if array_stretch is not None:
        parts.append(f"\\renewcommand{{\\arraystretch}}{{{array_stretch}}}")

    if label is not None:
        parts.append(f"\\label{{{label}}}")
    if centering:
        parts.append("\\centering")

    n_cols = len(df.columns)
    col_spec = col_sep.join(["c"] * n_cols)

    parts.append(f"\\begin{{tabular}}{{{col_spec}}}")
    parts.append(r"\hline")  # top border
    col_headers = [_make_column_header(col, bold_col_headers) for col in df.columns]

    parts.append(" & ".join(col_headers) + r"\\")
    parts.append(r"\hline")  # headers to rows separator

    for row_idx, row in df.iterrows():
        row_line_parts = []
        if row_index:
            if bold_row_headers:
                row_line_parts.append(f"\\textbf{{{row_idx}}}")
            else:
                row_line_parts.append(row_idx)
        row_line_parts.extend([str(val) for val in row.values])

        parts.append(" & ".join(row_line_parts) + r"\\")  # each row

    parts.append(r"\hline")  # bottom border
    parts.append(r"\end{tabular}")
    parts.append(f"\\caption{{{caption}}}")
    parts.append(r"\end{table}")

    return "\n".join(parts)


def _make_column_header(col_name: str, bold_col_headers: bool) -> str:
    lines = col_name.split("\n")

    def _one_line(line: str) -> str:
        if bold_col_headers:
            return f"\\textbf{{{line}}}"
        else:
            return col_name

    if len(lines) == 1:  # single line most common case
        return _one_line(col_name)
    else:
        latex_lines = r"\\".join(_one_line(line) for line in lines)
        return f"\\makecell{{{latex_lines}}}"


# %%
def _interactive_test():
    df = pd.DataFrame({"ColA": [1, 2, 3], "ColB": [4, 5, 6]})

    print(df_to_latex(df, "tabla ejemplo"))


# %%
