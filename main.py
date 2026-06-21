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
# Estratégia customizada para XGBoost (Bagging)
# ---------------------------------------------------------------------------

class XgbBaggingStrategy(FedAvg):

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        client_boosters: list[xgb.Booster] = []
        for _, fit_res in results:
            # parameters_to_ndarrays devolve lista de ndarrays uint8
            arrays = parameters_to_ndarrays(fit_res.parameters)
            if arrays:
                bst = ndarray_to_booster(arrays[0])
                client_boosters.append(bst)

        if not client_boosters:
            return None, {}

        global_booster = client_boosters[0]
        serialized = booster_to_ndarray(global_booster)
        parameters_aggregated = ndarrays_to_parameters([serialized])

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

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
        parameters: List[np.ndarray] — lista de arrays uint8 com o modelo global.
        Retorna: (List[np.ndarray], int, dict) — obrigatório pelo NumPyClient.
        """
        xgb_model_param = None

        # parameters[0] é o ndarray uint8 do modelo global (se existir)
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

        serialized = booster_to_ndarray(bst)
        return [serialized], len(self.X_train), {}

    def evaluate(self, parameters, config):
        """
        parameters: List[np.ndarray] — modelo global atual.
        """
        if not parameters or len(parameters) == 0 or parameters[0].size == 0:
            return float("inf"), len(self.X_test), {"mae": float("inf")}

        try:
            bst = ndarray_to_booster(parameters[0])
        except Exception:
            return float("inf"), len(self.X_test), {"mae": float("inf")}

        dtest = xgb.DMatrix(self.X_test, label=self.y_test)
        preds = bst.predict(dtest)

        mse = float(np.mean((self.y_test - preds) ** 2))
        mae = float(np.mean(np.abs(self.y_test - preds)))
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
    total_samples = sum(n for n, _ in metrics)
    if total_samples == 0:
        return {}
    weighted_mae = sum(n * m.get("mae", 0.0) for n, m in metrics)
    return {"mae": weighted_mae / total_samples}


def server_fn(context: Context) -> ServerAppComponents:
    # Array vazio com dtype uint8 — passa pelo pipeline do Flower sem erro
    empty_arr = np.array([], dtype=np.uint8)
    initial_params = ndarrays_to_parameters([empty_arr])

    strategy = XgbBaggingStrategy(
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