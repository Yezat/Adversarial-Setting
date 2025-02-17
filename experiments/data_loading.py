"""This module provides code to deserialize experiment results"""

import polars as pl
import io


def read_polars_dataframe(experiment_name: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    with open(f"results/{experiment_name}/df_erm.ser", "rb") as f:
        df_erm = pl.DataFrame.deserialize(io.BytesIO(f.read()))
    with open(f"results/{experiment_name}/df_se.ser", "rb") as f:
        df_se = pl.DataFrame.deserialize(io.BytesIO(f.read()))
    return df_erm, df_se
