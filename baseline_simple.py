import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.metrics  # use existing libraries!


# score function: binary cross entropy loss
def score_yp(y, p):  # y, p are numpy arrays
    return sklearn.metrics.log_loss(y, p)


# INPUT
training_week = 88   # for model training
validation_week = 89  # for model and baseline validation
test_week = 90  # for the final prediction (one week in the future, beyond our data)
target_customers = list(range(2000))
target_products = list(range(250))

baskets = pd.read_parquet("baskets-s.parquet")
prediction_index = pd.read_parquet("prediction_index.parquet")


def build_frequency_feature(baskets, week_start, week_end, feature_name):
    # subset baskets
    baskets_subset = baskets[
        (baskets["week"] >= week_start) & (baskets["week"] <= week_end)
    ]

    purchase_frequency_ij = (
        (
            baskets_subset.groupby(["customer", "product"])[["week"]].count()
            / baskets_subset.week.nunique()
        )
        .rename(columns={"week": feature_name})
        .reset_index()
    )

    return purchase_frequency_ij


def build_base_table(baskets, week):
    # target variable (product purchase)
    # consider using multiple weeks for training! more data might lead to better results.
    # also, different weeks might have different information.
    y = build_target(baskets, week)
    # features
    # note how features are computed on data BEFORE the target week
    x_1 = build_frequency_feature(baskets, -1, week - 1, "frequency_full")
    x_2 = build_frequency_feature(baskets, week - 30, week - 1, "frequency_l30")
    x_3 = build_frequency_feature(baskets, week - 5, week - 1, "frequency_l5")
    base_table_yx = (
        y.merge(x_1, on=["customer", "product"], how="left")
        .merge(x_2, on=["customer", "product"], how="left")
        .merge(x_3, on=["customer", "product"], how="left")
        .fillna(0)
    )
    return base_table_yx


def build_target(baskets, week):

    baskets_week = baskets[baskets["week"] == week][
        ["week", "customer", "product"]
    ].reset_index(drop=True)
    baskets_week["y"] = 1

    df = pd.DataFrame(
        {
            "week": week,
            "customer": np.repeat(target_customers, len(target_products), axis=0),
            "product": target_products * len(target_customers),
        }
    )

    df = df.merge(baskets_week, on=["week", "customer", "product"], how="left")
    df["y"] = df["y"].fillna(0).astype(int)

    return df


# TRAINING
base_table_train = build_base_table(baskets, training_week)

y = base_table_train["y"].values  # 1s and 0s
X = base_table_train[["frequency_full", "frequency_l30", "frequency_l5"]].values  # purchase frequencies

log_reg = sklearn.linear_model.LogisticRegression().fit(X, y)

base_table_train["probability"] = log_reg.predict_proba(X)[:, 1]

score_train = score_yp(
    base_table_train["y"].values,
    base_table_train["probability"].values,
)

print(f"Training score: {score_train}")


# VALIDATION
base_table_validation = build_base_table(baskets, validation_week)

X_validation = base_table_validation[
    ["frequency_full", "frequency_l30", "frequency_l5"]
].values

base_table_validation["probability"] = log_reg.predict_proba(X_validation)[:, 1]

score_validation = score_yp(
    base_table_validation["y"].values,
    base_table_validation["probability"].values,
)

print(f"Validation score: {score_validation}")
