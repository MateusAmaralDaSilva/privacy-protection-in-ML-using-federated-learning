"""
Passo a passo: treinar e validar XGBoost e Croston em um único produto do M5.

Por que dois modelos?
  • XGBoost  — usa janela deslizante de 28 dias com 12 features; captura sazonalidade e tendência.
  • Croston   — opera sobre a série 1-D bruta; projetado para demanda intermitente (muitos zeros).
    A documentação (seção 7.3) destaca que Croston é INADEQUADO para demanda agregada de loja
    (soma de centenas de itens raramente é zero), mas ADEQUADO para item individual.

Diferença do main.py / croston_main.py:
  Esses arquivos usam `store_rows[day_cols].sum()` → demanda agregada de toda a loja.
  Este script usa a série do item diretamente → série bruta com zeros reais.

Execução:
  python single_product.py

Dependências: xgboost, pandas, numpy, scikit-learn, matplotlib (opcional)
"""

import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from croston_model import CrostonForecaster

# ─── PASSO 1: Configuração ───────────────────────────────────────────────────
# Altere ITEM_ID e STORE_ID para o produto que deseja analisar.
# Formato item_id: CATEGORIA_GRUPO_NÚMERO  (ex: FOODS_3_090, HOBBIES_1_001)
# Lojas disponíveis: CA_1, CA_2, CA_3, CA_4, TX_1, TX_2, TX_3, WI_1, WI_2, WI_3

ITEM_ID  = "FOODS_3_090"
STORE_ID = "CA_1"
LOOKBACK = 28  # dias de histórico para o XGBoost (Croston não usa janela)

# Hiperparâmetros Croston
CROSTON_ALPHA   = 0.1      # suavização do nível de demanda (igual ao croston_main.py)
CROSTON_BETA    = 0.1      # suavização do intervalo entre demandas
CROSTON_VARIANT = "sba"    # "sba" (Syntetos-Boylan, menos viesado) ou "original"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
STATE    = STORE_ID.split("_")[0]  # "CA", "TX" ou "WI" — para flag SNAP correta


# ─── PASSO 2: Carregar CSVs ─────────────────────────────────────────────────
print(f"[1/7] Carregando dados para '{ITEM_ID}' em '{STORE_ID}'...")

sales_raw = pd.read_csv(os.path.join(DATA_DIR, "sales_train_evaluation.csv"))
calendar  = pd.read_csv(os.path.join(DATA_DIR, "calendar.csv"))
prices    = pd.read_csv(os.path.join(DATA_DIR, "sell_prices.csv"))

# Filtrar a linha exata do produto — NÃO soma com outros itens
item_row = sales_raw[
    (sales_raw["store_id"] == STORE_ID) &
    (sales_raw["item_id"]  == ITEM_ID)
]
if item_row.empty:
    available = sorted(sales_raw["item_id"].unique())[:10]
    raise ValueError(
        f"'{ITEM_ID}' não encontrado em '{STORE_ID}'.\n"
        f"Exemplos de item_ids disponíveis: {available}"
    )

day_cols = [c for c in sales_raw.columns if c.startswith("d_")]
daily = item_row[day_cols].T.reset_index()
daily.columns = ["d", "sales"]
daily["sales"] = daily["sales"].astype(np.float32)

# Proporção de zeros — confirma se a demanda é intermitente (esperado para item individual)
zero_pct = (daily["sales"] == 0).mean() * 100
print(f"    Dias sem venda: {zero_pct:.1f}% — {'demanda intermitente ✓' if zero_pct > 10 else 'demanda contínua'}")


# ─── PASSO 3: Limpeza e remoção de outliers ─────────────────────────────────
# Mesma lógica de load_timeseries() e load_data() em dataset.py:
# IQR clipping fit apenas nos primeiros 80% para evitar data leakage.
print("[2/7] Pré-processando...")

daily["sales"] = daily["sales"].ffill().bfill().fillna(0.0)

n_raw         = len(daily)
train_end_raw = int(0.8 * (n_raw - LOOKBACK)) + LOOKBACK
train_sales   = daily["sales"].iloc[:train_end_raw]
q1, q3        = train_sales.quantile(0.25), train_sales.quantile(0.75)
iqr           = q3 - q1
daily["sales"] = daily["sales"].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)

# Série bruta para o Croston (sem normalização — igual ao load_timeseries)
sales_array = daily["sales"].values.astype(np.float32)
split_raw   = int(0.8 * len(sales_array))
y_train_raw = sales_array[:split_raw]
y_test_raw  = sales_array[split_raw:]


# ─── PASSO 4: Features de calendário e preço (para XGBoost) ─────────────────
# Croston não usa estas features. XGBoost usa as mesmas 12 colunas do main.py,
# com uma diferença: o preço é do item específico (não média de todos os itens).

calendar["date"]         = pd.to_datetime(calendar["date"])
calendar["day_of_month"] = calendar["date"].dt.day
calendar["day_of_year"]  = calendar["date"].dt.dayofyear
calendar["is_event"]     = calendar["event_name_1"].notna().astype(np.float32)

cal_cols = [
    "d", "wm_yr_wk", "wday", "month",
    "day_of_month", "day_of_year",
    "is_event", f"snap_{STATE}",
]
df = daily.merge(calendar[cal_cols], on="d", how="left")
df.rename(columns={f"snap_{STATE}": "snap"}, inplace=True)

for col in ["wday", "month", "day_of_month", "day_of_year"]:
    df[col] = df[col].ffill().bfill()
df["is_event"] = df["is_event"].fillna(0.0).astype(np.float32)
df["snap"]     = df["snap"].fillna(0.0).astype(np.float32)

# Preço do item específico (não a média da loja inteira como em load_data)
item_prices = (
    prices[(prices["store_id"] == STORE_ID) & (prices["item_id"] == ITEM_ID)]
    [["wm_yr_wk", "sell_price"]]
    .rename(columns={"sell_price": "avg_price"})
)
df = df.merge(item_prices, on="wm_yr_wk", how="left")
df["avg_price"] = (
    df["avg_price"].ffill().bfill().fillna(df["avg_price"].median()).astype(np.float32)
)


# ─── PASSO 5: Codificação cíclica e normalização (para XGBoost) ─────────────
# Sin/cos evita descontinuidade em variáveis periódicas.
# StandardScaler fit apenas no treino — nunca no teste.

def _add_cyclical(df: pd.DataFrame, col: str, period: int) -> None:
    df[f"sin_{col}"] = np.sin(2 * np.pi * df[col] / period).astype(np.float32)
    df[f"cos_{col}"] = np.cos(2 * np.pi * df[col] / period).astype(np.float32)

_add_cyclical(df, "wday",         7)
_add_cyclical(df, "day_of_month", 31)
_add_cyclical(df, "month",        12)
_add_cyclical(df, "day_of_year",  365)

n_samples     = len(df) - LOOKBACK
train_end_idx = int(0.8 * n_samples) + LOOKBACK

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


# ─── PASSO 6: Janela deslizante + split temporal (XGBoost) ──────────────────
# 28 dias × 12 features = 336 dims → prever o dia 29. Split 80/20 cronológico.

feature_cols = [
    "sin_wday",         "cos_wday",
    "sin_day_of_month", "cos_day_of_month",
    "sin_month",        "cos_month",
    "sin_day_of_year",  "cos_day_of_year",
    "snap", "is_event",
    "price_norm",
    "sales_norm",
]

data    = df[feature_cols].values.astype(np.float32)   # [T, 12]
targets = df["sales_norm"].values.astype(np.float32)   # [T]

X_list, y_list = [], []
for i in range(LOOKBACK, len(data)):
    X_list.append(data[i - LOOKBACK:i].flatten())  # [336]
    y_list.append(targets[i])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)

split_xgb = int(0.8 * len(X))
X_train, X_test = X[:split_xgb], X[split_xgb:]
y_train_xgb, y_test_xgb = y[:split_xgb], y[split_xgb:]

print(f"    XGBoost  → treino: {len(X_train)} amostras  |  teste: {len(X_test)} amostras")
print(f"    Croston  → treino: {len(y_train_raw)} dias       |  teste: {len(y_test_raw)} dias")


# ─── PASSO 7a: Treinar XGBoost ───────────────────────────────────────────────
print("[3/7] Treinando XGBoost (50 rounds)...")

# Idêntico ao XGB_PARAMS de main.py — garante comparabilidade com o experimento federado.
# num_boost_round=50 equivale a 10 rodadas federadas × 5 rounds locais (total do main.py).
# Não há warm-start nem ensemble: um único produto = um único "cliente", sem agregação.
XGB_PARAMS = {
    "objective":        "reg:squarederror",
    "eval_metric":      "rmse",
    "max_depth":        6,
    "learning_rate":    0.1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
}

dtrain = xgb.DMatrix(X_train, label=y_train_xgb)
dtest  = xgb.DMatrix(X_test,  label=y_test_xgb)

bst = xgb.train(
    XGB_PARAMS,
    dtrain,
    num_boost_round=50,
    evals=[(dtrain, "train"), (dtest, "test")],
    verbose_eval=10,
)


# ─── PASSO 7b: Treinar Croston ───────────────────────────────────────────────
# Croston não tem "rounds" — é ajuste por suavização exponencial em uma passagem.
# Opera diretamente sobre a série bruta, igual ao FlowerCrostonClient.fit().
print("[4/7] Ajustando Croston (SBA)...")

croston = CrostonForecaster(
    alpha=CROSTON_ALPHA, beta=CROSTON_BETA, variant=CROSTON_VARIANT
)
croston.fit(y_train_raw)

non_zeros = int(np.sum(y_train_raw > 0))
print(f"    Estado após fit → demand_level={croston.demand_level:.4f}, "
      f"interval={croston.interval:.4f}, não-zeros no treino: {non_zeros}")


# ─── PASSO 8: Avaliar ambos na escala bruta de vendas ───────────────────────
print("[5/7] Avaliando...")

# XGBoost: inverse_transform para comparar na mesma escala do Croston
preds_norm_xgb = bst.predict(dtest)
preds_raw_xgb  = sales_scaler.inverse_transform(preds_norm_xgb.reshape(-1, 1)).flatten()
actual_raw_xgb = sales_scaler.inverse_transform(y_test_xgb.reshape(-1, 1)).flatten()

errors_xgb = actual_raw_xgb - preds_raw_xgb
mse_xgb    = float(np.mean(errors_xgb ** 2))
mae_xgb    = float(np.mean(np.abs(errors_xgb)))
rmse_xgb   = float(np.sqrt(mse_xgb))
ss_res_xgb = float(np.sum(errors_xgb ** 2))
ss_tot_xgb = float(np.sum((actual_raw_xgb - actual_raw_xgb.mean()) ** 2))
r2_xgb     = 1.0 - ss_res_xgb / ss_tot_xgb if ss_tot_xgb > 0 else 0.0

# Croston: evaluate() retorna (mse, mae, rmse, r2) diretamente em escala bruta
mse_cro, mae_cro, rmse_cro, r2_cro = croston.evaluate(y_test_raw)
forecast_cro = croston.predict()  # constante ao longo de todo o horizonte

print(f"\n{'═'*60}")
print(f"  Produto : {ITEM_ID} @ {STORE_ID}  ({zero_pct:.1f}% zeros no conjunto completo)")
print(f"{'─'*60}")
print(f"  {'Métrica':<10}  {'XGBoost':>12}  {'Croston SBA':>12}")
print(f"{'─'*60}")
print(f"  {'RMSE':<10}  {rmse_xgb:>12.4f}  {rmse_cro:>12.4f}  unidades/dia")
print(f"  {'MAE':<10}  {mae_xgb:>12.4f}  {mae_cro:>12.4f}  unidades/dia")
print(f"  {'MSE':<10}  {mse_xgb:>12.4f}  {mse_cro:>12.4f}")
print(f"  {'R²':<10}  {r2_xgb:>12.4f}  {r2_cro:>12.4f}")
print(f"{'─'*60}")
print(f"  Previsão Croston (constante): {forecast_cro:.4f} unidades/dia")
print(f"{'═'*60}\n")


# ─── PASSO 9: (Opcional) Importância das features — XGBoost ─────────────────
print("[6/7] Top features por ganho (XGBoost)...")

importance = bst.get_score(importance_type="gain")
top = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
for feat, score in top:
    print(f"    {feat:<30s} {score:.2f}")


# ─── PASSO 10: (Opcional) Gráfico de previsão vs real ───────────────────────
print("[7/7] Gerando gráfico...")

try:
    import matplotlib.pyplot as plt

    n_plot = min(120, len(actual_raw_xgb))
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f"{ITEM_ID} @ {STORE_ID}  |  zeros: {zero_pct:.1f}%", fontsize=13)

    # Subplot 1 — XGBoost
    axes[0].plot(actual_raw_xgb[-n_plot:], label="Real",     color="steelblue", linewidth=1.5)
    axes[0].plot(preds_raw_xgb[-n_plot:],  label="Previsão", color="tomato",    linewidth=1.5, linestyle="--")
    axes[0].set_title(f"XGBoost  |  RMSE={rmse_xgb:.2f}  R²={r2_xgb:.3f}")
    axes[0].set_ylabel("Vendas (unidades)")
    axes[0].legend()

    # Subplot 2 — Croston (previsão constante)
    actual_plot = y_test_raw[-n_plot:]
    axes[1].plot(actual_plot,                               label="Real",     color="steelblue", linewidth=1.5)
    axes[1].axhline(forecast_cro, label="Previsão Croston", color="darkorange", linewidth=2, linestyle="--")
    axes[1].set_title(f"Croston SBA (previsão constante)  |  RMSE={rmse_cro:.2f}  R²={r2_cro:.3f}")
    axes[1].set_ylabel("Vendas (unidades)")
    axes[1].set_xlabel(f"Dias (últimos {n_plot} do conjunto de teste)")
    axes[1].legend()

    fig.tight_layout()
    out = f"forecast_{ITEM_ID}_{STORE_ID}.png"
    fig.savefig(out, dpi=150)
    print(f"    Gráfico salvo: {out}")

except ImportError:
    print("    matplotlib não instalado — instale com: pip install matplotlib")
