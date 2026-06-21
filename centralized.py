"""
Baseline centralizado — todos os dados ficam em um único lugar e um único
processo de treinamento acessa tudo simultaneamente.

Serve como teto de referência: o melhor resultado teoricamente alcançável
por cada abordagem de modelagem, sem restrições de privacidade ou federação.

Comparação direta com:
  - main.py        → XGBoost federado (bagging, 10 rodadas)
  - croston_main.py → Croston ensemble federado (10 rodadas)
"""

import numpy as np
import xgboost as xgb

from dataset import load_data, load_timeseries, STORE_IDS
from croston_model import CrostonForecaster

NUM_PARTITIONS = 10

XGB_PARAMS = {
    "objective":        "reg:squarederror",
    "eval_metric":      "rmse",
    "max_depth":        6,
    "learning_rate":    0.1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
}

# ---------------------------------------------------------------------------
# XGBoost Centralizado
#
# Todos os dados de treino das 10 lojas são concatenados e um único modelo
# é treinado. O modelo aprende padrões cross-loja (algo impossível no cenário
# federado, onde cada cliente só vê seus próprios dados).
# ---------------------------------------------------------------------------

def run_centralized_xgboost(num_boost_round: int = 50) -> tuple[float, float]:
    print("\n" + "=" * 60)
    print("  Baseline Centralizado — XGBoost")
    print("=" * 60)

    X_trains, y_trains, X_tests, y_tests = [], [], [], []

    for i in range(NUM_PARTITIONS):
        X_tr, y_tr, X_te, y_te = load_data(i, NUM_PARTITIONS)
        X_trains.append(X_tr)
        y_trains.append(y_tr)
        X_tests.append(X_te)
        y_tests.append(y_te)
        print(f"  Loja {STORE_IDS[i]:>5s} → treino={len(X_tr):>5d}, teste={len(X_te):>4d}")

    X_all_train = np.concatenate(X_trains, axis=0)
    y_all_train = np.concatenate(y_trains, axis=0)
    X_all_test  = np.concatenate(X_tests,  axis=0)
    y_all_test  = np.concatenate(y_tests,  axis=0)

    print(f"\n  Total treino: {len(X_all_train):>6d} amostras (todas as lojas)")
    print(f"  Total teste:  {len(X_all_test):>6d} amostras")

    dtrain = xgb.DMatrix(X_all_train, label=y_all_train)
    dtest  = xgb.DMatrix(X_all_test,  label=y_all_test)

    bst = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=num_boost_round,
        verbose_eval=False,
    )

    preds = bst.predict(dtest)
    mse = float(np.mean((y_all_test - preds) ** 2))
    mae = float(np.mean(np.abs(y_all_test - preds)))

    print(f"\n  MSE global: {mse:.6f}")
    print(f"  MAE global: {mae:.6f}")

    # Métricas por loja para comparação direta com o cenário federado
    print("\n  Métricas por loja:")
    offset = 0
    for i in range(NUM_PARTITIONS):
        n = len(y_tests[i])
        store_preds = preds[offset : offset + n]
        store_mse = float(np.mean((y_tests[i] - store_preds) ** 2))
        store_mae = float(np.mean(np.abs(y_tests[i] - store_preds)))
        print(f"    {STORE_IDS[i]:>5s} → MSE={store_mse:.6f}, MAE={store_mae:.6f}")
        offset += n

    return mse, mae


# ---------------------------------------------------------------------------
# Croston Centralizado
#
# Cada loja tem seu próprio modelo Croston treinado com acesso irrestrito
# aos seus dados. O ensemble é formado por TODOS os 10 modelos simultaneamente
# — sem a limitação de fraction_fit do cenário federado.
#
# Este é o teto teórico para o croston_main.py:
#   Após rounds suficientes para todos os clientes participarem, o ensemble
#   federado converge para este resultado (os dados locais nunca mudam).
# ---------------------------------------------------------------------------

def run_centralized_croston(
    alpha: float = 0.1,
    beta: float = 0.1,
    variant: str = "sba",
) -> tuple[float, float]:
    print("\n" + "=" * 60)
    print("  Baseline Centralizado — Croston / SBA (Ensemble Completo)")
    print("=" * 60)

    models: list[CrostonForecaster] = []
    weights: list[float] = []
    y_tests: list[np.ndarray] = []

    for i in range(NUM_PARTITIONS):
        y_train, y_test = load_timeseries(i, NUM_PARTITIONS)

        m = CrostonForecaster(alpha=alpha, beta=beta, variant=variant)
        m.fit(y_train)

        forecast = m.predict()
        models.append(m)
        weights.append(float(len(y_train)))
        y_tests.append(y_test)

        print(
            f"  Loja {STORE_IDS[i]:>5s} → "
            f"demand_level={m.demand_level:.4f}, "
            f"interval={m.interval:.4f}, "
            f"forecast={forecast:.4f}"
        )

    # Ensemble centralizado: média ponderada das previsões de todas as lojas
    total_weight = sum(weights)
    ensemble_forecast = sum(
        (w / total_weight) * m.predict()
        for m, w in zip(models, weights)
    )

    print(f"\n  Previsão ensemble (média ponderada): {ensemble_forecast:.4f}")

    # Avaliação: cada loja avalia o ensemble global nos seus dados de teste
    all_errors_sq  = []
    all_errors_abs = []
    print("\n  Métricas por loja (ensemble aplicado aos dados de teste):")
    for i in range(NUM_PARTITIONS):
        errors = y_tests[i].astype(np.float64) - ensemble_forecast
        store_mse = float(np.mean(errors ** 2))
        store_mae = float(np.mean(np.abs(errors)))
        all_errors_sq.extend(errors ** 2)
        all_errors_abs.extend(np.abs(errors))
        print(f"    {STORE_IDS[i]:>5s} → MSE={store_mse:.6f}, MAE={store_mae:.6f}")

    mse = float(np.mean(all_errors_sq))
    mae = float(np.mean(all_errors_abs))
    print(f"\n  MSE global: {mse:.6f}")
    print(f"  MAE global: {mae:.6f}")

    return mse, mae


# ---------------------------------------------------------------------------
# Ponto de entrada — executa os dois baselines e imprime o resumo comparativo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    xgb_mse, xgb_mae = run_centralized_xgboost()
    cro_mse, cro_mae = run_centralized_croston()

    print("\n" + "=" * 60)
    print("  Resumo — Baselines Centralizados")
    print("=" * 60)
    print(f"  XGBoost  → MSE={xgb_mse:.6f}  MAE={xgb_mae:.6f}")
    print(f"  Croston  → MSE={cro_mse:.6f}  MAE={cro_mae:.6f}")
    print()
    print("  Compare com os resultados federados em:")
    print("    python main.py          (XGBoost federado)")
    print("    python croston_main.py  (Croston ensemble federado)")
