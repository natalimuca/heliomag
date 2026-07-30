import numpy as np
import pandas as pd
from scipy.signal import periodogram
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from fetch_data import fetch_gfz
from baseline_probe import make_features, TRAIN_END, TEST_START

LEAD = 14  # longest tested lead time -- most room for recurrent structure to diverge from persistence


def band_power_fraction(signal, period_lo=25, period_hi=29):
    signal = signal - np.mean(signal)
    freqs, power = periodogram(signal, fs=1.0)  # fs=1 sample/day
    with np.errstate(divide="ignore"):
        periods = 1.0 / freqs
    mask = (periods >= period_lo) & (periods <= period_hi)
    total = power.sum()
    band = power[mask].sum()
    return band / total, freqs, power, periods


def main():
    df = fetch_gfz().loc["2010-05-01":]
    feat = make_features(df)
    target = df["Ap"].shift(-LEAD)
    joined = pd.concat([feat, target.rename("target")], axis=1).dropna()

    train = joined.loc[:TRAIN_END]
    test = joined.loc[TEST_START:]

    cols_full = ["SN", "F107", "Ap_t", "Ap_3d_mean", "Ap_7d_mean"]
    model = LinearRegression().fit(train[cols_full], train["target"])
    pred = model.predict(test[cols_full])
    residual = test["target"].values - pred

    rmse = mean_squared_error(test["target"], pred) ** 0.5
    print(f"Lead time: {LEAD} days | test n={len(test)} | classical-baseline RMSE={rmse:.2f}\n")

    frac_raw, _, _, _ = band_power_fraction(test["target"].values)
    frac_resid, _, _, _ = band_power_fraction(residual)

    print(f"Fraction of spectral power in 25-29 day band (solar-rotation recurrence signature):")
    print(f"  Raw Ap (actual, test period):       {frac_raw*100:.2f}%")
    print(f"  Residual (actual - classical pred):  {frac_resid*100:.2f}%")
    print()
    if frac_resid > 0.5 * frac_raw:
        print("-> Residual retains most of the 27-day recurrent power: classical indices are")
        print("   NOT explaining the coronal-hole-driven recurrent component. Supports the")
        print("   hypothesis that a spatial embedding (Surya) could recover this structure.")
    else:
        print("-> Residual has much less 27-day power than raw Ap: classical indices already")
        print("   explain most of the recurrent structure. Weakens the coronal-hole rationale;")
        print("   embedding gains would have to come from elsewhere (e.g. flare/CME timing).")


if __name__ == "__main__":
    main()
