"""This module provides code to deserialize experiment results"""

import polars as pl
import io


LIST_COLUMNS_ERM = [
    "adversarial_generalization_errors",
    "adversarial_generalization_errors_teacher",
    "adversarial_generalization_errors_overlap",
    "fair_adversarial_errors",
    "boundary_loss_test_es",
    "boundary_errors",
    "test_losses",
]

LIST_COLUMNS_SE = [
    "adversarial_generalization_errors",
    "test_losses",
    "data_model_adversarial_test_errors",
    "gamma_robustness_es",
    "boundary_errors",
]

KEYS = ["alpha", "epsilon", "tau", "lam", "epsilon_g", "data_model_name"]


def read_result_dataframe(experiment_name: str) -> pl.DataFrame:
    with open(f"results/{experiment_name}/df_erm.ser", "rb") as f:
        df_erm = pl.DataFrame.deserialize(io.BytesIO(f.read()))
        if df_erm.shape != (0, 0):
            df_erm = explode_generalisation_metrics(df_erm, LIST_COLUMNS_ERM)
            df_erm = compute_std(df_erm)
    with open(f"results/{experiment_name}/df_se.ser", "rb") as f:
        df_se = pl.DataFrame.deserialize(io.BytesIO(f.read()))
        df_se = explode_generalisation_metrics(df_se, LIST_COLUMNS_SE)
        df_se = compute_std(df_se)

    if df_erm.shape == (0, 0):
        return df_se
    else:
        return df_se.join(df_erm, on=KEYS, how="inner", validate="1:1", suffix="_erm")


def explode_generalisation_metrics(
    df: pl.DataFrame, column_list: list[str]
) -> pl.DataFrame:
    df = df.explode(columns=column_list)

    for i in range(len(column_list)):
        df = (
            df.with_columns(pl.col(column_list[i]).list.to_struct())
            .unnest(columns=[column_list[i]])
            .rename({"field_0": f"epsilon_g_{i}", "field_1": column_list[i]})
        )
        if i == 0:
            df = df.rename({"epsilon_g_0": "epsilon_g"})
        else:
            df = df.drop("epsilon_g_" + str(i))

    return df


def compute_std(df: pl.DataFrame) -> pl.DataFrame:
    gb = df.group_by(KEYS)
    df_mean = gb.agg(pl.all().mean())
    df_std = gb.agg(pl.all().std())
    return df_mean.join(df_std, on=KEYS, how="inner", validate="1:1", suffix="_std")
