import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed")
LOOKBACK = 28
STORE_IDS = [
    "CA_1", "CA_2", "CA_3", "CA_4",
    "TX_1", "TX_2", "TX_3",
    "WI_1", "WI_2", "WI_3",
]

# Module-level caches to avoid reloading large files on every client call
_sales_cache = None
_calendar_cache = None
_prices_cache = None


def _load_sales() -> pd.DataFrame:
    global _sales_cache
    if _sales_cache is None:
        _sales_cache = pd.read_csv(os.path.join(DATA_DIR, "sales_train_evaluation.csv"))
    return _sales_cache


def _load_calendar() -> pd.DataFrame:
    global _calendar_cache
    if _calendar_cache is None:
        cal = pd.read_csv(os.path.join(DATA_DIR, "calendar.csv"))
        cal["date"] = pd.to_datetime(cal["date"])
        cal["day_of_month"] = cal["date"].dt.day
        cal["day_of_year"] = cal["date"].dt.dayofyear
        cal["is_event"] = cal["event_name_1"].notna().astype(np.float32)
        _calendar_cache = cal
    return _calendar_cache


def _load_prices() -> pd.DataFrame:
    global _prices_cache
    if _prices_cache is None:
        _prices_cache = pd.read_csv(os.path.join(DATA_DIR, "sell_prices.csv"))
    return _prices_cache


def _add_cyclical(df: pd.DataFrame, col: str, period: int) -> None:
    """Add sin/cos cyclical encoding in-place."""
    df[f"sin_{col}"] = np.sin(2 * np.pi * df[col] / period).astype(np.float32)
    df[f"cos_{col}"] = np.cos(2 * np.pi * df[col] / period).astype(np.float32)


def load_data(partition_id: int, num_partitions: int = 10):
    """
    Build train/test DataLoaders for one store partition.

    Temporal features use sin/cos encoding:
      - wday (period=7): day of week
      - day_of_month (period=31)
      - month (period=12)
      - day_of_year (period=365)

    Sliding window: LOOKBACK=28 days → predict next day's total store sales.
    Train/test split is temporal (first 80% / last 20%) to avoid leakage.
    """
    store_id = STORE_IDS[partition_id]
    state = store_id.split("_")[0]  # "CA", "TX", or "WI"

    sales_raw = _load_sales()
    calendar = _load_calendar()
    prices = _load_prices()

    # Sum all items in this store per day (wide → one row per day)
    store_rows = sales_raw[sales_raw["store_id"] == store_id]
    day_cols = [c for c in sales_raw.columns if c.startswith("d_")]
    daily = store_rows[day_cols].sum().reset_index()
    daily.columns = ["d", "sales"]

    # --- NaN treatment on raw sales ---
    daily["sales"] = daily["sales"].ffill().bfill().fillna(0.0)

    # --- Outlier removal: IQR clipping on daily sales ---
    # Clip instead of drop to preserve temporal continuity for the sliding window.
    q1, q3 = daily["sales"].quantile(0.25), daily["sales"].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    daily["sales"] = daily["sales"].clip(lower=lower, upper=upper)

    # Attach temporal features from calendar
    cal_cols = [
        "d", "wm_yr_wk", "wday", "month",
        "day_of_month", "day_of_year",
        "is_event", f"snap_{state}",
    ]
    df = daily.merge(calendar[cal_cols], on="d", how="left")
    df.rename(columns={f"snap_{state}": "snap"}, inplace=True)

    # NaN treatment for calendar features
    for col in ["wday", "month", "day_of_month", "day_of_year"]:
        df[col] = df[col].ffill().bfill()
    df["is_event"] = df["is_event"].fillna(0.0).astype(np.float32)
    df["snap"] = df["snap"].fillna(0.0).astype(np.float32)

    # Average weekly sell price for this store
    avg_price = (
        prices[prices["store_id"] == store_id]
        .groupby("wm_yr_wk")["sell_price"]
        .mean()
        .reset_index()
        .rename(columns={"sell_price": "avg_price"})
    )
    df = df.merge(avg_price, on="wm_yr_wk", how="left")
    df["avg_price"] = df["avg_price"].ffill().bfill().fillna(df["avg_price"].median()).astype(np.float32)

    # Cyclical encoding for all temporal cycles
    _add_cyclical(df, "wday", 7)
    _add_cyclical(df, "day_of_month", 31)
    _add_cyclical(df, "month", 12)
    _add_cyclical(df, "day_of_year", 365)

    # Normalize sales and price: fit scaler only on train portion to avoid leakage
    n_samples = len(df) - LOOKBACK
    train_end_idx = int(0.8 * n_samples) + LOOKBACK  # last row touched by train windows

    sales_scaler = StandardScaler()
    price_scaler = StandardScaler()

    sales_all = df[["sales"]].values.astype(np.float32)
    price_all = df[["avg_price"]].values.astype(np.float32)

    sales_norm = np.empty(len(df), dtype=np.float32)
    price_norm = np.empty(len(df), dtype=np.float32)
    sales_norm[:train_end_idx] = sales_scaler.fit_transform(sales_all[:train_end_idx]).flatten()
    sales_norm[train_end_idx:] = sales_scaler.transform(sales_all[train_end_idx:]).flatten()
    price_norm[:train_end_idx] = price_scaler.fit_transform(price_all[:train_end_idx]).flatten()
    price_norm[train_end_idx:] = price_scaler.transform(price_all[train_end_idx:]).flatten()

    df["sales_norm"] = sales_norm
    df["price_norm"] = price_norm

    # --- Save preprocessed features to disk ---
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    save_cols = [
        "d", "sales", "avg_price",
        "wday", "day_of_month", "month", "day_of_year",
        "is_event", "snap",
        "sin_wday", "cos_wday",
        "sin_day_of_month", "cos_day_of_month",
        "sin_month", "cos_month",
        "sin_day_of_year", "cos_day_of_year",
        "sales_norm", "price_norm",
    ]
    df[save_cols].to_csv(
        os.path.join(PREPROCESSED_DIR, f"{store_id}.csv"),
        index=False,
    )

    feature_cols = [
        "sin_wday", "cos_wday",
        "sin_day_of_month", "cos_day_of_month",
        "sin_month", "cos_month",
        "sin_day_of_year", "cos_day_of_year",
        "snap", "is_event",
        "price_norm",
        "sales_norm",  # lagged sales values inside the window
    ]

    data = df[feature_cols].values.astype(np.float32)   # [T, F]
    targets = df["sales_norm"].values.astype(np.float32)  # [T]

    # Build sliding-window samples
    X_list, y_list = [], []
    for i in range(LOOKBACK, len(data)):
        X_list.append(data[i - LOOKBACK:i].flatten())  # [LOOKBACK * F]
        y_list.append(targets[i])

    X = np.array(X_list, dtype=np.float32)  # [N, LOOKBACK*F]
    y = np.array(y_list, dtype=np.float32)  # [N]

    # Temporal split — never shuffle before splitting to avoid leakage
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train).unsqueeze(1),
        ),
        batch_size=32,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_test),
            torch.from_numpy(y_test).unsqueeze(1),
        ),
        batch_size=32,
        shuffle=False,
    )
    return train_loader, test_loader
