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
        ss_res_s = float(np.sum(errs ** 2))
        ss_tot_s = float(np.sum((store_actual_raw - np.mean(store_actual_raw)) ** 2))
        store_mse  = ss_res_s / len(errs)
        store_mae  = float(np.mean(np.abs(errs)))
        store_rmse = float(np.sqrt(store_mse))
        store_r2   = 1.0 - ss_res_s / ss_tot_s if ss_tot_s > 0 else 0.0
        all_errors_sq.extend(errs ** 2)
        all_errors_abs.extend(np.abs(errs))
        print(f"    {STORE_IDS[i]:>5s} → RMSE={store_rmse:.4f}  MAE={store_mae:.4f}  R²={store_r2:.4f}")
        offset += n

    all_errs = np.array(all_errors_sq)
    mse  = float(np.mean(all_errs))
    mae  = float(np.mean(all_errors_abs))
    rmse = float(np.sqrt(mse))
    # R² global: sobre todas as lojas concatenadas (escala bruta)
    all_actual = np.concatenate([
        sales_scalers[i].inverse_transform(y_all_test[
            sum(test_sizes[:i]):sum(test_sizes[:i+1])
        ].reshape(-1, 1)).flatten()
        for i in range(len(test_sizes))
    ])
    ss_tot_g = float(np.sum((all_actual - np.mean(all_actual)) ** 2))
    r2 = 1.0 - float(np.sum(all_errs)) / ss_tot_g if ss_tot_g > 0 else 0.0
    print(f"\n  RMSE global (bruto): {rmse:.4f}")
    print(f"  MAE  global (bruto): {mae:.4f}")
    print(f"  R²   global:         {r2:.4f}")
    return mse, mae, rmse, r2


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
        # Pesos calculados pelo MAE no TREINO — consistente com o cenário federado,
        # onde fit() reporta erro de treino e y_test é reservado para evaluate().
        _, mae_train, _, _ = m.evaluate(y_train)

        models.append(m)
        inv_mae_weights.append(1.0 / (mae_train + 1e-9))

        print(
            f"  Loja {STORE_IDS[i]:>5s} → "
            f"demand_level={m.demand_level:.4f}, "
            f"interval={m.interval:.4f}, "
            f"forecast={forecast:.4f}, "
            f"MAE_treino={mae_train:.4f}"
        )

    # Pesos normalizados pelo inverso do MAE local (igual ao ensemble federado)
    total_w = sum(inv_mae_weights)
    weights = [w / total_w for w in inv_mae_weights]

    ensemble_forecast = sum(w * m.predict() for w, m in zip(weights, models))
    print(f"\n  Previsão ensemble (ponderada por 1/MAE): {ensemble_forecast:.4f}")

    # Avaliação: cada loja avalia o ensemble global nos seus dados de teste
    all_errors_sq, all_errors_abs, all_actual, all_pred = [], [], [], []
    print("\n  Métricas por loja (escala bruta de vendas):")
    for i, y_test in enumerate(y_tests):
        y_test_f = y_test.astype(np.float64)
        errs = y_test_f - ensemble_forecast
        ss_res_s = float(np.sum(errs ** 2))
        ss_tot_s = float(np.sum((y_test_f - np.mean(y_test_f)) ** 2))
        store_mse  = ss_res_s / len(errs)
        store_mae  = float(np.mean(np.abs(errs)))
        store_rmse = float(np.sqrt(store_mse))
        store_r2   = 1.0 - ss_res_s / ss_tot_s if ss_tot_s > 0 else 0.0
        all_errors_sq.extend(errs ** 2)
        all_errors_abs.extend(np.abs(errs))
        all_actual.extend(y_test_f)
        print(f"    {STORE_IDS[i]:>5s} → RMSE={store_rmse:.4f}  MAE={store_mae:.4f}  R²={store_r2:.4f}")

    all_actual_arr = np.array(all_actual)
    all_errs_sq    = np.array(all_errors_sq)
    mse  = float(np.mean(all_errs_sq))
    mae  = float(np.mean(all_errors_abs))
    rmse = float(np.sqrt(mse))
    ss_tot_g = float(np.sum((all_actual_arr - np.mean(all_actual_arr)) ** 2))
    r2 = 1.0 - float(np.sum(all_errs_sq)) / ss_tot_g if ss_tot_g > 0 else 0.0
    print(f"\n  RMSE global (bruto): {rmse:.4f}")
    print(f"  MAE  global (bruto): {mae:.4f}")
    print(f"  R²   global:         {r2:.4f}")
    return mse, mae, rmse, r2


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    xgb_mse, xgb_mae, xgb_rmse, xgb_r2 = run_centralized_xgboost()
    cro_mse, cro_mae, cro_rmse, cro_r2 = run_centralized_croston()

    print("\n" + "=" * 60)
    print("  Resumo — Baselines Centralizados (escala bruta de vendas)")
    print("=" * 60)
    print(f"  {'Modelo':<10} {'RMSE':>10} {'MAE':>10} {'R²':>8}")
    print(f"  {'-'*40}")
    print(f"  {'XGBoost':<10} {xgb_rmse:>10.4f} {xgb_mae:>10.4f} {xgb_r2:>8.4f}")
    print(f"  {'Croston':<10} {cro_rmse:>10.4f} {cro_mae:>10.4f} {cro_r2:>8.4f}")
    print()
    print("  Ambas as métricas estão em unidades brutas de vendas — comparáveis.")
    print()
    print("  Compare com os resultados federados em:")
    print("    python main.py                  (XGBoost federado)")
    print("    python croston_main.py          (Croston FedAvg federado)")
    print("    python croston_ensemble_main.py (Croston ensemble federado)")
