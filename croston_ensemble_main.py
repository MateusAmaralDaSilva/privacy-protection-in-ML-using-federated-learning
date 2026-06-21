import numpy as np

from flwr.client import ClientApp, NumPyClient
from flwr.common import (
    Context,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation

from croston_model import CrostonForecaster
from dataset import load_timeseries

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

NUM_PARTITIONS = 10
NUM_SERVER_ROUNDS = 10

CROSTON_ALPHA = 0.1
CROSTON_BETA = 0.1
CROSTON_VARIANT = "sba"

# ---------------------------------------------------------------------------
# Estratégia: Ensemble Federado para Croston
#
# Diferença fundamental em relação ao CrostonFedAvgStrategy:
#   • FedAvg clássico: média de [l, p] entre lojas → destrói diversidade local
#   • Este ensemble: acumula UM estado por cliente (cid → [l, p])
#
# Transmissão por rodada:
#   parameters[0]   = estado warm-start (cliente com menor MAE acumulado)
#   parameters[1]   = pesos do ensemble (float64, tamanho = nº de estados)
#   parameters[2:]  = todos os estados [l, p] (um por cliente, sorted by cid)
#
# Avaliação: previsão = média ponderada das previsões individuais.
# Peso de cada modelo ∝ 1 / (MAE_local + ε)  → modelos melhores dominam mais.
# ---------------------------------------------------------------------------

class CrostonEnsembleStrategy(FedAvg):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # cid → estado [l, p] (um por cliente, atualizado a cada participação)
        self._client_states: dict[str, np.ndarray] = {}
        # cid → MAE no teste local (reportado pelo cliente em fit)
        self._client_errors: dict[str, float] = {}

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        for client_proxy, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            if arrays and arrays[0].size == 2:
                cid = client_proxy.cid
                self._client_states[cid] = arrays[0]
                self._client_errors[cid] = fit_res.metrics.get("mae", float("inf"))

        if not self._client_states:
            return None, {}

        # Warm-start = estado do cliente com menor MAE acumulado
        best_cid = min(self._client_errors, key=lambda c: self._client_errors[c])
        warm_start = self._client_states[best_cid]

        # Pesos inversamente proporcionais ao MAE (modelos mais precisos pesam mais)
        cids_sorted = sorted(self._client_states.keys())
        errors = np.array(
            [self._client_errors.get(cid, 1e9) for cid in cids_sorted],
            dtype=np.float64,
        )
        errors = np.where(np.isinf(errors), 1e9, errors)
        inv_errors = 1.0 / (errors + 1e-9)
        weights = (inv_errors / inv_errors.sum()).astype(np.float64)

        states = [self._client_states[cid] for cid in cids_sorted]

        # parameters[0] = warm-start, [1] = pesos, [2:] = todos os estados
        params = ndarrays_to_parameters([warm_start, weights] + states)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        print(
            f"[Rodada {server_round:>2d}] Ensemble: {len(self._client_states):>2d} estados | "
            f"warm-start=cid {best_cid} (MAE={self._client_errors[best_cid]:.4f})"
        )
        return params, metrics_aggregated

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
# Cliente Croston com Ensemble
# ---------------------------------------------------------------------------

class FlowerCrostonEnsembleClient(NumPyClient):

    def __init__(self, y_train: np.ndarray, y_test: np.ndarray):
        self.y_train = y_train
        self.y_test = y_test

    def fit(self, parameters, config):
        """
        Recebe o warm-start (estado do modelo com menor MAE global) e ajusta
        o Croston localmente. Devolve o estado atualizado + MAE local.
        O MAE informa o servidor para seleção do próximo warm-start e cálculo
        dos pesos do ensemble.
        """
        model = CrostonForecaster(
            alpha=CROSTON_ALPHA, beta=CROSTON_BETA, variant=CROSTON_VARIANT
        )

        # parameters[0] = warm-start [l, p] (ignoramos pesos e demais estados)
        if parameters and len(parameters) > 0 and parameters[0].size == 2:
            global_state = parameters[0]
            model.fit(
                self.y_train,
                init_demand_level=float(global_state[0]),
                init_interval=float(global_state[1]),
            )
        else:
            model.fit(self.y_train)

        _, mae = model.evaluate(self.y_test)
        state = model.get_state()
        return [state], len(self.y_train), {"mae": mae}

    def evaluate(self, parameters, config):
        """
        Avalia o ensemble federado de modelos Croston.

        Layout esperado de parameters:
            [0]  warm-start state [l, p]  — ignorado na avaliação
            [1]  pesos do ensemble        — float64 array de tamanho N
            [2:] N estados [l, p]         — um por cliente participante

        Previsão final = média ponderada das N previsões individuais.
        """
        # Rodadas iniciais: sem ensemble ainda → fallback para warm-start
        if not parameters or len(parameters) < 3:
            if parameters and parameters[0].size == 2:
                model = CrostonForecaster(
                    alpha=CROSTON_ALPHA, beta=CROSTON_BETA, variant=CROSTON_VARIANT
                )
                model.set_state(parameters[0])
                mse, mae = model.evaluate(self.y_test)
                return mse, len(self.y_test), {"mae": mae}
            return float("inf"), len(self.y_test), {"mae": float("inf")}

        # parameters[1] = pesos (por posição, não por tamanho)
        weights = parameters[1]
        states = [p for p in parameters[2:] if p.size == 2]

        if not states:
            return float("inf"), len(self.y_test), {"mae": float("inf")}

        # Uma previsão por estado do ensemble
        predictions = []
        for state in states:
            model = CrostonForecaster(
                alpha=CROSTON_ALPHA, beta=CROSTON_BETA, variant=CROSTON_VARIANT
            )
            model.set_state(state)
            predictions.append(model.predict())

        if len(weights) == len(predictions):
            ensemble_pred = float(np.average(predictions, weights=weights))
        else:
            ensemble_pred = float(np.mean(predictions))

        y_test = np.asarray(self.y_test, dtype=np.float64)
        errors = y_test - ensemble_pred
        mse = float(np.mean(errors ** 2))
        mae = float(np.mean(np.abs(errors)))
        return mse, len(self.y_test), {"mae": mae}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    y_train, y_test = load_timeseries(
        partition_id=partition_id,
        num_partitions=NUM_PARTITIONS,
    )
    return FlowerCrostonEnsembleClient(y_train, y_test).to_client()


# ---------------------------------------------------------------------------
# Servidor
# ---------------------------------------------------------------------------

def weighted_average(metrics):
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    weighted_mae = sum(n * m.get("mae", 0.0) for n, m in metrics)
    return {"mae": weighted_mae / total}


def server_fn(context: Context) -> ServerAppComponents:
    empty_state = np.array([], dtype=np.float64)
    initial_params = ndarrays_to_parameters([empty_state])

    strategy = CrostonEnsembleStrategy(
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
    backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0}},
)

if hist is not None:
    print("\n=== Histórico de Treinamento (Croston Ensemble Federado) ===")
    print(f"Losses distribuídas (MSE): {hist.losses_distributed}")
    print(f"Métricas distribuídas (MAE): {hist.metrics_distributed}")
else:
    print("\n[AVISO] Simulação não retornou histórico.")
