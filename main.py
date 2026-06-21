import xgboost as xgb
import numpy as np

from flwr.client import ClientApp, NumPyClient
from flwr.common import (
    Context,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation

from dataset import load_data

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

NUM_PARTITIONS = 10
NUM_SERVER_ROUNDS = 10

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

# ---------------------------------------------------------------------------
# Serialização: bytes XGBoost <-> ndarray uint8 (único formato aceito pelo Flower)
#
# O Flower exige que tensors sejam ndarrays (não bytes puros). Para evitar
# corrupção, usamos np.frombuffer com .copy() para garantir array writeable
# e C-contiguous, que é o que o pipeline interno do Flower espera.
# ---------------------------------------------------------------------------

def booster_to_ndarray(bst: xgb.Booster) -> np.ndarray:
    raw: bytes = bst.save_raw(raw_format="ubj")
    arr = np.frombuffer(raw, dtype=np.uint8).copy()   # writeable, C-contiguous
    return arr


def ndarray_to_booster(arr: np.ndarray) -> xgb.Booster:
    bst = xgb.Booster(XGB_PARAMS)
    bst.load_model(bytearray(arr.tobytes()))
    return bst


# ---------------------------------------------------------------------------
# Estratégia: Ensemble Federado + Warm-start
#
# O servidor acumula um booster por cliente (keyed por cid). A cada rodada:
#   • Fit   → clientes recebem parameters[0] como warm-start e adicionam
#             5 árvores sobre ele (mesma mecânica de antes).
#   • Eval  → clientes recebem TODOS os boosters acumulados e calculam a
#             previsão como média simples do ensemble.
#
# Formato de parameters transmitidos:
#   parameters[0]   = booster de warm-start (o do cid de menor índice)
#   parameters[1:]  = demais boosters do ensemble
#
# Assim, fit() usa parameters[0] igual antes, e evaluate() usa todos.
# ---------------------------------------------------------------------------

class XgbEnsembleStrategy(FedAvg):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # cid → booster serializado como ndarray uint8 (uma entrada por cliente)
        self._client_boosters: dict[str, np.ndarray] = {}
        # cid → MSE no conjunto de teste local (reportado pelo cliente no fit)
        self._client_errors: dict[str, float] = {}

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Atualiza o ensemble — dict por cid evita cópias antigas do mesmo cliente
        for client_proxy, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            if arrays and arrays[0].size > 0:
                cid = client_proxy.cid
                self._client_boosters[cid] = arrays[0]
                self._client_errors[cid] = fit_res.metrics.get("mse", float("inf"))

        if not self._client_boosters:
            return None, {}

        # Warm-start: booster com menor MSE de teste acumulado entre todos os clientes
        best_cid   = min(self._client_errors, key=lambda c: self._client_errors[c])
        warm_start = self._client_boosters[best_cid]
        others     = [
            self._client_boosters[cid]
            for cid in sorted(self._client_boosters.keys())
            if cid != best_cid
        ]

        # parameters[0] = warm-start (melhor); parameters[1:] = resto do ensemble
        parameters_aggregated = ndarrays_to_parameters([warm_start] + others)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        print(
            f"[Rodada {server_round:>2d}] Ensemble: {len(self._client_boosters):>2d} boosters | "
            f"warm-start=cid {best_cid} (MSE={self._client_errors[best_cid]:.6f})"
        )
        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)

        total_loss = sum(res.loss * res.num_examples for _, res in results)
        total_examples = sum(res.num_examples for _, res in results)
        return total_loss / total_examples, metrics_aggregated


# ---------------------------------------------------------------------------
# Cliente XGBoost — usa NumPyClient (obrigatório nesta versão do Flower)
# Os parâmetros são listas de ndarrays, nunca bytes puros.
# ---------------------------------------------------------------------------

class FlowerXGBoostClient(NumPyClient):

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self._has_global_model = False

    def fit(self, parameters, config):
        """
        parameters[0] = booster de warm-start (o com menor MSE acumulado no servidor).
        Adiciona 5 árvores sobre esse modelo e avalia no conjunto de teste local.
        O MSE retornado nas métricas é usado pelo servidor para selecionar o
        próximo warm-start.
        """
        xgb_model_param = None

        if parameters and len(parameters) > 0 and parameters[0].size > 0:
            try:
                xgb_model_param = ndarray_to_booster(parameters[0])
                self._has_global_model = True
            except Exception:
                xgb_model_param = None

        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        bst = xgb.train(
            XGB_PARAMS,
            dtrain,
            num_boost_round=5,
            xgb_model=xgb_model_param,
        )

        # Avalia no teste local — informa o servidor para seleção do warm-start
        dtest = xgb.DMatrix(self.X_test, label=self.y_test)
        test_preds = bst.predict(dtest)
        test_mse = float(np.mean((self.y_test - test_preds) ** 2))

        serialized = booster_to_ndarray(bst)
        return [serialized], len(self.X_train), {"mse": test_mse}

    def evaluate(self, parameters, config):
        """
        Avalia o ensemble federado: previsão = média simples de todos os boosters.

        parameters[0]  → warm-start booster (também membro do ensemble)
        parameters[1:] → demais boosters acumulados pelo servidor
        """
        valid = [p for p in parameters if p.size > 0]
        if not valid:
            return float("inf"), len(self.X_test), {"mae": float("inf")}

        try:
            boosters = [ndarray_to_booster(p) for p in valid]
        except Exception:
            return float("inf"), len(self.X_test), {"mae": float("inf")}

        dtest = xgb.DMatrix(self.X_test, label=self.y_test)

        # Média simples entre as previsões de todos os boosters do ensemble
        all_preds = np.stack([bst.predict(dtest) for bst in boosters], axis=0)
        ensemble_preds = all_preds.mean(axis=0)

        mse = float(np.mean((self.y_test - ensemble_preds) ** 2))
        mae = float(np.mean(np.abs(self.y_test - ensemble_preds)))
        return mse, len(self.X_test), {"mae": mae}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    X_train, y_train, X_test, y_test = load_data(
        partition_id=partition_id,
        num_partitions=NUM_PARTITIONS,
    )
    return FlowerXGBoostClient(X_train, y_train, X_test, y_test).to_client()


# ---------------------------------------------------------------------------
# Servidor
# ---------------------------------------------------------------------------

def weighted_average(metrics):
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    result = {}
    for key in ("mae", "mse"):
        if any(key in m for _, m in metrics):
            result[key] = sum(n * m.get(key, 0.0) for n, m in metrics) / total
    return result


def server_fn(context: Context) -> ServerAppComponents:
    # Array vazio com dtype uint8 — passa pelo pipeline do Flower sem erro
    empty_arr = np.array([], dtype=np.uint8)
    initial_params = ndarrays_to_parameters([empty_arr])

    strategy = XgbEnsembleStrategy(
        initial_parameters=initial_params,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        fraction_fit=0.3,
        fraction_evaluate=0.3,
        min_available_clients=2,
    )
    return ServerAppComponents(
        config=ServerConfig(num_rounds=NUM_SERVER_ROUNDS),
        strategy=strategy,
    )


# ---------------------------------------------------------------------------
# Simulação
# ---------------------------------------------------------------------------

server_app = ServerApp(server_fn=server_fn)
client_app = ClientApp(client_fn=client_fn)

hist = run_simulation(
    server_app=server_app,
    client_app=client_app,
    num_supernodes=NUM_PARTITIONS,
    backend_config={"client_resources": {"num_cpus": 2, "num_gpus": 0}},
)

if hist is not None:
    print("\n=== Histórico de Treinamento ===")
    print(f"Losses distribuídas: {hist.losses_distributed}")
    print(f"Métricas distribuídas: {hist.metrics_distributed}")
else:
    print("\n[AVISO] Simulação não retornou histórico.")