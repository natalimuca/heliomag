import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from fetch_data import fetch_gfz

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LEAD_TIMES = [3, 5, 7, 10, 14]
TRAIN_END = "2021-12-31"
TEST_START = "2022-01-01"
TARGETS = ["Ap", "Kp"]

PCA_GRID = [5, 10, 20, 30, 50, 75, 100]
ALPHA_GRID = np.logspace(-2, 4, 13)
N_SPLITS = 5


def load_embeddings():
    d = np.load(DATA_DIR / "embeddings_merged.npz", allow_pickle=True)
    keys = sorted(d.keys())
    dates = pd.to_datetime(keys, format="%Y%m%d")
    mat = np.stack([d[k] for k in keys]).astype(np.float64)
    cols = [f"e{i}" for i in range(mat.shape[1])]
    return pd.DataFrame(mat, index=dates, columns=cols)


def make_classical_features(gfz, index, target_col):
    feat = pd.DataFrame(index=index)
    feat["SN"] = gfz["SN"].reindex(index, method="ffill")
    feat["F107"] = gfz["F107obs"].reindex(index, method="ffill")
    feat[f"{target_col}_t"] = gfz[target_col].reindex(index, method="ffill")
    feat[f"{target_col}_3d_mean"] = gfz[target_col].rolling(3).mean().reindex(index, method="ffill")
    feat[f"{target_col}_7d_mean"] = gfz[target_col].rolling(7).mean().reindex(index, method="ffill")
    return feat


def fit_rmse(estimator, X_train, y_train, X_test, y_test):
    estimator.fit(X_train, y_train)
    pred = estimator.predict(X_test)
    return mean_squared_error(y_test, pred) ** 0.5, pred


def cv_select(pipeline, param_grid, X_train, y_train):
    cv = TimeSeriesSplit(n_splits=N_SPLITS)
    search = GridSearchCV(
        pipeline, param_grid, cv=cv,
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def run_probe(target_col, emb, gfz):
    classical = make_classical_features(gfz, emb.index, target_col)
    emb_cols = emb.columns.tolist()
    cls_cols = classical.columns.tolist()
    t_col = f"{target_col}_t"

    print(f"\n=== target: {target_col} ===")
    header = (
        f"{'lead':>5} | {'persistence':>11} | {'classical':>9} | {'embed':>9} (n_pca,alpha) "
        f"| {'embed+classical':>16} (n_pca,alpha) | {'embed(MLP)':>11} | {'residual-corr':>13}"
    )
    print(header)
    print("-" * 145)

    for lead in LEAD_TIMES:
        target_dates = emb.index + pd.Timedelta(days=lead)
        target = gfz[target_col].reindex(target_dates)
        target.index = emb.index
        target = target.rename("target")

        joined = pd.concat([classical, emb, target], axis=1).dropna()
        if len(joined) < 100:
            print(f"{lead:>5} | insufficient data")
            continue

        train = joined.loc[:TRAIN_END]
        test = joined.loc[TEST_START:]
        y_train, y_test = train["target"], test["target"]

        pers_rmse = mean_squared_error(y_test, test[t_col]) ** 0.5

        classical_model = LinearRegression()
        classical_rmse, classical_pred_test = fit_rmse(
            classical_model, train[cls_cols], y_train, test[cls_cols], y_test,
        )
        classical_pred_train = classical_model.predict(train[cls_cols])

        embed_pipe = Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(random_state=0)),
            ("ridge", Ridge()),
        ])
        embed_grid = {"pca__n_components": PCA_GRID, "ridge__alpha": ALPHA_GRID}
        embed_best, embed_params = cv_select(embed_pipe, embed_grid, train[emb_cols], y_train)
        embed_rmse, _ = fit_rmse(embed_best, train[emb_cols], y_train, test[emb_cols], y_test)

        combined_pre = ColumnTransformer([
            ("emb", Pipeline([("scale", StandardScaler()), ("pca", PCA(random_state=0))]), emb_cols),
            ("cls", StandardScaler(), cls_cols),
        ])
        combined_pipe = Pipeline([("pre", combined_pre), ("ridge", Ridge())])
        combined_grid = {"pre__emb__pca__n_components": PCA_GRID, "ridge__alpha": ALPHA_GRID}
        X_train_combined = pd.concat([train[emb_cols], train[cls_cols]], axis=1)
        X_test_combined = pd.concat([test[emb_cols], test[cls_cols]], axis=1)
        combined_best, combined_params = cv_select(combined_pipe, combined_grid, X_train_combined, y_train)
        combined_rmse, _ = fit_rmse(combined_best, X_train_combined, y_train, X_test_combined, y_test)

        mlp_pipe = Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(random_state=0)),
            ("mlp", MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=0)),
        ])
        mlp_grid = {"pca__n_components": PCA_GRID}
        mlp_best, mlp_params = cv_select(mlp_pipe, mlp_grid, train[emb_cols], y_train)
        mlp_rmse, _ = fit_rmse(mlp_best, train[emb_cols], y_train, test[emb_cols], y_test)

        # residual correction: embeddings predict what the classical linear model got wrong
        residual_train = y_train - classical_pred_train
        resid_pipe = Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(random_state=0)),
            ("ridge", Ridge()),
        ])
        resid_grid = {"pca__n_components": PCA_GRID, "ridge__alpha": ALPHA_GRID}
        resid_best, resid_params = cv_select(resid_pipe, resid_grid, train[emb_cols], residual_train)
        resid_pred_test = resid_best.predict(test[emb_cols])
        residual_corrected_pred = classical_pred_test + resid_pred_test
        residual_rmse = mean_squared_error(y_test, residual_corrected_pred) ** 0.5

        e_tag = f"({embed_params['pca__n_components']},{embed_params['ridge__alpha']:.2g})"
        c_tag = f"({combined_params['pre__emb__pca__n_components']},{combined_params['ridge__alpha']:.2g})"
        m_tag = f"(n_pca={mlp_params['pca__n_components']})"
        print(
            f"{lead:>5} | {pers_rmse:>11.3f} | {classical_rmse:>9.3f} | {embed_rmse:>9.3f} {e_tag:>12} "
            f"| {combined_rmse:>16.3f} {c_tag:>12} | {mlp_rmse:>11.3f} {m_tag} | {residual_rmse:>13.3f}"
        )


def main():
    emb = load_embeddings()
    gfz = fetch_gfz()

    print(f"Weekly embedding samples: {len(emb)}, {emb.index.min().date()} to {emb.index.max().date()}")
    print(f"Train: <= {TRAIN_END} | Test: >= {TEST_START}")
    print(f"PCA dims searched: {PCA_GRID} | Ridge alpha searched: {N_SPLITS}-fold TimeSeriesSplit CV")

    for target_col in TARGETS:
        run_probe(target_col, emb, gfz)


if __name__ == "__main__":
    main()
