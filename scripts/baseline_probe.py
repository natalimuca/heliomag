import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from fetch_data import fetch_gfz

LEAD_TIMES = [3, 5, 7, 10, 14]
TRAIN_END = "2021-12-31"
TEST_START = "2022-01-01"


def make_features(df):
    feat = pd.DataFrame(index=df.index)
    feat["SN"] = df["SN"]
    feat["F107"] = df["F107obs"]
    feat["Ap_t"] = df["Ap"]
    feat["Ap_3d_mean"] = df["Ap"].rolling(3).mean()
    feat["Ap_7d_mean"] = df["Ap"].rolling(7).mean()
    return feat


def eval_model(name, model, X_train, y_train, X_test, y_test, scale=False):
    if scale:
        sc = StandardScaler().fit(X_train)
        X_train, X_test = sc.transform(X_train), sc.transform(X_test)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    return rmse


def main():
    df = fetch_gfz().loc["2010-05-01":]
    feat = make_features(df)

    print(f"SDO-era data: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")
    print(f"Train: <= {TRAIN_END} | Test: >= {TEST_START}\n")

    print(f"{'lead':>5} | {'persistence':>12} | {'SN+F107':>10} | {'SN+F107+pers':>14} | {'MLP (all feat)':>15}")
    print("-" * 70)

    for lead in LEAD_TIMES:
        target = df["Ap"].shift(-lead)
        joined = pd.concat([feat, target.rename("target")], axis=1).dropna()

        train = joined.loc[:TRAIN_END]
        test = joined.loc[TEST_START:]
        if len(train) < 100 or len(test) < 50:
            print(f"{lead:>5} | insufficient data")
            continue

        y_train, y_test = train["target"], test["target"]

        # persistence: predict Ap(t+lead) = Ap(t)
        pers_rmse = mean_squared_error(y_test, test["Ap_t"]) ** 0.5

        # classical scalar indices only
        cols_classical = ["SN", "F107"]
        classical_rmse = eval_model(
            "classical", LinearRegression(),
            train[cols_classical], y_train, test[cols_classical], y_test
        )

        # classical + persistence + rolling means (strongest classical baseline)
        cols_full = ["SN", "F107", "Ap_t", "Ap_3d_mean", "Ap_7d_mean"]
        combined_rmse = eval_model(
            "combined", LinearRegression(),
            train[cols_full], y_train, test[cols_full], y_test
        )

        mlp_rmse = eval_model(
            "mlp", MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=0),
            train[cols_full], y_train, test[cols_full], y_test, scale=True
        )

        print(f"{lead:>5} | {pers_rmse:>12.3f} | {classical_rmse:>10.3f} | {combined_rmse:>14.3f} | {mlp_rmse:>15.3f}")


if __name__ == "__main__":
    main()
