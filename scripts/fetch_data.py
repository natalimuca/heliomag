import requests
import pandas as pd
import numpy as np
from pathlib import Path
from io import StringIO

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

GFZ_URL = "https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt"

COLS = ["year", "month", "day", "days", "days_m", "bsr", "dB",
        "Kp1", "Kp2", "Kp3", "Kp4", "Kp5", "Kp6", "Kp7", "Kp8",
        "ap1", "ap2", "ap3", "ap4", "ap5", "ap6", "ap7", "ap8",
        "Ap", "SN", "F107obs", "F107adj", "D"]


def fetch_gfz(force=False):
    out_path = DATA_DIR / "gfz_kp_ap_sn_f107.csv"
    if out_path.exists() and not force:
        return pd.read_csv(out_path, parse_dates=["date"], index_col="date")

    text = requests.get(GFZ_URL, timeout=30).text
    lines = [l for l in text.splitlines() if l and not l.startswith("#")]
    df = pd.read_csv(StringIO("\n".join(lines)), sep=r"\s+", header=None, names=COLS)
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df.set_index("date")
    df["Ap"] = df["Ap"].replace(-1, np.nan)
    df["SN"] = df["SN"].replace(-1, np.nan)
    df["F107obs"] = df["F107obs"].replace(-1.0, np.nan)
    kp_cols = [f"Kp{i}" for i in range(1, 9)]
    df[kp_cols] = df[kp_cols].replace(-1.0, np.nan)
    df["Kp"] = df[kp_cols].mean(axis=1)
    df = df[["Ap", "Kp", "SN", "F107obs"]]

    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(out_path)
    return df


if __name__ == "__main__":
    df = fetch_gfz(force=True)
    print(f"Cached {len(df)} rows to {DATA_DIR / 'gfz_kp_ap_sn_f107.csv'}")
    print(f"Range: {df.index.min().date()} to {df.index.max().date()}")
