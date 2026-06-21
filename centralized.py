"""
Baseline centralizado — todos os dados ficam em um único lugar e um único
processo de treinamento acessa tudo simultaneamente.

Serve como teto de referência: o melhor resultado teoricamente alcançável
por cada abordagem de modelagem, sem restrições de privacidade ou federação.

Métricas reportadas em unidades BRUTAS de vendas (não normalizadas) para
permitir comparação direta entre XGBoost e Croston.

Comparação direta com:
  - main.py                  → XGBoost federado
  - croston_main.py          → Croston FedAvg federado
  - croston_ensemble_main.py → Croston ensemble federado
"""

import numpy as np
import xgboost as xgb

from dataset import load_all_data, load_all_timeseries, STORE_IDS
from croston_model import CrostonForecaster

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
# Todos os dados de todas as lojas são carregados de uma vez (load_all_data).
# Um único modelo é treinado no conjunto concatenado — aprende padrões
# cross-loja impossíveis no cenário federado.
#
# As previsões são inverse_transformed por scaler de loja para que as métricas
# finais estejam em unidades brutas de vendas, comparáveis com o Croston.
# ---------------------------------------------------------------------------

def run_centralized_xgboost(num_boost_round: int = 50) -> tuple[float, float]:
    print("\n" + "=" * 60)
    print("  Baseline Centralizado — XGBoost")
    print("=" * 60)

    X_all_train, y_all_train, X_all_test, y_all_test, \
        train_sizes, test_sizes, sales_scalers = load_all_data()

    for i, (n_tr, n_te) in enumerate(zip(train_sizes, test_sizes)):
        print(f"  Loja {STORE_IDS[i]:>5s} → treino={n_tr:>5d}, teste={n_te:>4d}")

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

    # Previsões em espaço normalizado
    preds_norm = bst.predict(dtest)

    # Inverse_transform por loja para escala bruta de vendas
    all_errors_sq, all_errors_abs = [], []
    print("\n  Métricas por loja (escala bruta de vendas):")
    offset = 0
    for i, (n, scaler) in enumerate(zip(test_sizes, sales_scalers)):
        store_pred_norm   = preds_norm[offset:offset + n].reshape(-1, 1)
        store_actual_norm = y_all_test[offset:offset + n].reshape(-1, 1)

        store_pred_raw   = scaler.inverse_transform(store_pred_norm).flatten()
        store_actual_raw = scaler.inverse_transform(store_actual_norm).flatten()

        errs = store_actual_raw - store_pred_raw
        store_mse = float(np.mean(errs ** 2))
        store_mae = float(np.mean(np.abs(errs)))
        all_errors_sq.extend(errs ** 2)
        all_errors_abs.extend(np.abs(errs))
        print(f"    {STORE_IDS[i]:>5s} → MSE={store_mse:.4f}, MAE={store_mae:.4f}")
        offset += n

    mse = float(np.mean(all_errors_sq))
    mae = float(np.mean(all_errors_abs))
    print(f"\n  MSE global (bruto): {mse:.4f}")
    print(f"  MAE global (bruto): {mae:.4f}")
    return mse, mae


# ---------------------------------------------------------------------------
# Croston Centralizado
#
# Cada loja tem seu próprio modelo Croston treinado com acesso irrestrito
# aos seus dados. O ensemble é formado por TODOS os modelos simultaneamente
# — sem a limitação de fraction_fit do cenário federado.
#
# Métricas calculadas em unidades brutas de vendas (load_all_timeseries
# retorna séries sem normalização).
# ---------------------------------------------------------------------------

def run_centralized_croston(
    alpha: float = 0.1,
    beta: float = 0.1,
    variant: str = "sba",
) -> tuple[float, float]:
    print("\n" + "=" * 60)
    print("  Baseline Centralizado — Croston / SBA (Ensemble Completo)")
    print("=" * 60)

    y_trains, y_tests = load_all_timeseries()

    models: list[CrostonForecaster] = []
    inv_mae_weights: list[float] = []

    for i, (y_train, y_test) in enumerate(zip(y_trains, y_tests)):
        m = CrostonForecaster(alpha=alpha, beta=beta, variant=variant)
        m.fit(y_train)

        forecast = m.predict()
        _, local_mae = m.evaluate(y_test)

        models.append(m)
        inv_mae_weights.append(1.0 / (local_mae + 1e-9))

        print(
            f"  Loja {STORE_IDS[i]:>5s} → "
            f"demand_level={m.demand_level:.4f}, "
            f"interval={m.interval:.4f}, "
            f"forecast={forecast:.4f}, "
            f"MAE={local_mae:.4f}"
        )

    # Pesos normalizados pelo inverso do MAE local (igual ao ensemble federado)
    total_w = sum(inv_mae_weights)
    weights = [w / total_w for w in inv_mae_weights]

    ensemble_forecast = sum(w * m.predict() for w, m in zip(weights, models))
    print(f"\n  Previsão ensemble (ponderada por 1/MAE): {ensemble_forecast:.4f}")

    # Avaliação: cada loja avalia o ensemble global nos seus dados de teste
    all_errors_sq, all_errors_abs = [], []
    print("\n  Métricas por loja (escala bruta de vendas):")
    for i, y_test in enumerate(y_tests):
        errs = y_test.astype(np.float64) - ensemble_forecast
        store_mse = float(np.mean(errs ** 2))
        store_mae = float(np.mean(np.abs(errs)))
        all_errors_sq.extend(errs ** 2)
        all_errors_abs.extend(np.abs(errs))
        print(f"    {STORE_IDS[i]:>5s} → MSE={store_mse:.4f}, MAE={store_mae:.4f}")

    mse = float(np.mean(all_errors_sq))
    mae = float(np.mean(all_errors_abs))
    print(f"\n  MSE global (bruto): {mse:.4f}")
    print(f"  MAE global (bruto): {mae:.4f}")
    return mse, mae


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    xgb_mse, xgb_mae = run_centralized_xgboost()
    cro_mse, cro_mae = run_centralized_croston()

    print("\n" + "=" * 60)
    print("  Resumo — Baselines Centralizados (escala bruta de vendas)")
    print("=" * 60)
    print(f"  XGBoost  → MSE={xgb_mse:.4f}  MAE={xgb_mae:.4f}")
    print(f"  Croston  → MSE={cro_mse:.4f}  MAE={cro_mae:.4f}")
    print()
    print("  Ambas as métricas estão em unidades brutas de vendas — comparáveis.")
    print()
    print("  Compare com os resultados federados em:")
    print("    python main.py                  (XGBoost federado)")
    print("    python croston_main.py          (Croston FedAvg federado)")
    print("    python croston_ensemble_main.py (Croston ensemble federado)")
