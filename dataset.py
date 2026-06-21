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


def load_all_timeseries() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Carrega as séries temporais brutas de TODAS as lojas sem o conceito de
    partições federadas. Uso exclusivo do baseline centralizado do Croston.

    Returns
    -------
    y_trains : list[np.ndarray]  — série de treino por loja (ordem STORE_IDS)
    y_tests  : list[np.ndarray]  — série de teste por loja
    """
    y_trains, y_tests = [], []
    for i in range(len(STORE_IDS)):
        y_tr, y_te = load_timeseries(i)
        y_trains.append(y_tr)
        y_tests.append(y_te)
    return y_trains, y_tests


def load_timeseries(partition_id: int, num_partitions: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Retorna a série temporal de vendas diárias (após clipping IQR) para uma loja.

    Aplica o mesmo tratamento de NaN e remoção de outliers que load_data().
    A divisão treino/teste é temporal: primeiros 80% / últimos 20%.

    Usada pelo experimento federado com o método de Croston, que opera
    diretamente sobre a série sem engenharia de features ou janela deslizante.
    """
    store_id = STORE_IDS[partition_id]
    sales_raw = _load_sales()

    store_rows = sales_raw[sales_raw["store_id"] == store_id]
    day_cols = [c for c in sales_raw.columns if c.startswith("d_")]
    sales = store_rows[day_cols].sum().values.astype(np.float32)

    # Tratamento de NaN (mesma lógica do load_data)
    sales = pd.Series(sales).ffill().bfill().fillna(0.0).values.astype(np.float32)

    # Clipping IQR apenas sobre o conjunto de treino para evitar vazamento
    split = int(0.8 * len(sales))
    train_sales = sales[:split]
    q1 = float(np.quantile(train_sales, 0.25))
    q3 = float(np.quantile(train_sales, 0.75))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    sales = np.clip(sales, lower, upper).astype(np.float32)

    return sales[:split], sales[split:]


def load_all_data() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    list[int], list[int], list[StandardScaler],
]:
    """
    Carrega e concatena os dados de features de TODAS as lojas sem o conceito
    de partições federadas. Uso exclusivo do baseline centralizado do XGBoost.

    Cada loja mantém seu próprio StandardScaler (ajustado apenas no treino
    local), que é retornado para permitir inverse_transform das previsões de
    volta à escala original de vendas — tornando as métricas comparáveis com
    o Croston, que opera em unidades brutas.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
        Arrays concatenados de todas as lojas (normalized).
    train_sizes : list[int]
        Número de amostras de treino por loja.
    test_sizes : list[int]
        Número de amostras de teste por loja (para desagregar métricas).
    sales_scalers : list[StandardScaler]
        Um scaler por loja (mesma ordem de STORE_IDS) para inverse_transform.
    """
    X_trains, y_trains, X_tests, y_tests, scalers = [], [], [], [], []
    for i in range(len(STORE_IDS)):
        X_tr, y_tr, X_te, y_te, scaler = load_data(i, return_scaler=True)
        X_trains.append(X_tr)
        y_trains.append(y_tr)
        X_tests.append(X_te)
        y_tests.append(y_te)
        scalers.append(scaler)
    train_sizes = [len(y) for y in y_trains]
    test_sizes  = [len(y) for y in y_tests]
    return (
        np.concatenate(X_trains, axis=0),
        np.concatenate(y_trains, axis=0),
        np.concatenate(X_tests,  axis=0),
        np.concatenate(y_tests,  axis=0),
        train_sizes,
        test_sizes,
        scalers,
    )


def load_data(partition_id: int, num_partitions: int = 10, return_scaler: bool = False):
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

    # --- Outlier removal: IQR clipping apenas sobre o treino para evitar vazamento ---
    # Clip instead of drop to preserve temporal continuity for the sliding window.
    n_raw = len(daily)
    train_end_raw = int(0.8 * (n_raw - LOOKBACK)) + LOOKBACK  # alinhado com train_end_idx
    train_sales_raw = daily["sales"].iloc[:train_end_raw]
    q1, q3 = train_sales_raw.quantile(0.25), train_sales_raw.quantile(0.75)
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

    if return_scaler:
        return X_train, y_train, X_test, y_test, sales_scaler
    return X_train, y_train, X_test, y_test


def load_data_torch(partition_id: int, num_partitions: int = 10):
    """
    Wrapper que chama load_data() e empacota os arrays em DataLoaders PyTorch.
    """
    X_train, y_train, X_test, y_test = load_data(partition_id, num_partitions)

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
