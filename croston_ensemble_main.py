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
# Estratégia: Ensemble Federado sem Agregação de Parâmetros
#
# Princípio de independência estatística:
#   O estado [l, p] de Croston é um estimador local — específico para a série
#   daquele cliente. Fazer FedAvg entre lojas destrói a interpretação local.
#   Aqui o servidor NUNCA envia estado para guiar o treino. Cada cliente
#   treina do zero (cold start) a cada rodada, preservando a validade
#   estatística dos seus estimadores.
#
# Fluxo por rodada:
#   Fit:      servidor → parâmetros vazios → cliente treina independentemente
#             → cliente devolve [l, p] local + MAE local
#   Evaluate: servidor acumula estados por cid, computa pesos por 1/MAE,
#             envia ensemble para os clientes avaliarem
#
# Transmissão para evaluate:
#   parameters[0]   = pesos do ensemble (float64, tamanho = nº de clientes)
#   parameters[1:]  = estados [l, p] (um por cid, sorted by cid)
# ---------------------------------------------------------------------------

class CrostonEnsembleStrategy(FedAvg):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # cid → estado [l, p] (atualizado a cada participação do cliente)
        self._client_states: dict[str, np.ndarray] = {}
        # cid → MAE no conjunto de teste local (reportado pelo cliente)
        self._client_errors: dict[str, float] = {}

    def configure_fit(self, server_round, parameters, client_manager):
        """Sempre envia parâmetros vazios: clientes treinam de forma independente."""
        empty = ndarrays_to_parameters([np.array([], dtype=np.float64)])
        return super().configure_fit(server_round, empty, client_manager)

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

        # Pesos inversamente proporcionais ao MAE local de cada modelo
        cids_sorted = sorted(self._client_states.keys())
        errors = np.array(
            [self._client_errors.get(cid, 1e9) for cid in cids_sorted],
            dtype=np.float64,
        )
        errors = np.where(np.isinf(errors), 1e9, errors)
        inv_errors = 1.0 / (errors + 1e-9)
        weights = (inv_errors / inv_errors.sum()).astype(np.float64)

        states = [self._client_states[cid] for cid in cids_sorted]

        # parameters[0] = pesos; parameters[1:] = estados do ensemble
        params = ndarrays_to_parameters([weights] + states)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        best_cid = cids_sorted[int(np.argmin(errors))]
        print(
            f"[Rodada {server_round:>2d}] Ensemble: {len(self._client_states):>2d} modelos | "
            f"melhor=cid {best_cid} (MAE={self._client_errors[best_cid]:.4f})"
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
# Cliente Croston — Ensemble sem warm-start
# ---------------------------------------------------------------------------

class FlowerCrostonEnsembleClient(NumPyClient):

    def __init__(self, y_train: np.ndarray, y_test: np.ndarray):
        self.y_train = y_train
        self.y_test = y_test

    def fit(self, parameters, config):
        """
        Treina o Croston exclusivamente nos dados locais (cold start sempre).
        Ignora qualquer estado recebido do servidor para preservar a
        independência estatística do estimador local.
        Devolve o estado local + MAE no teste, usado pelo servidor para
        calcular os pesos do ensemble.
        """
        model = CrostonForecaster(
            alpha=CROSTON_ALPHA, beta=CROSTON_BETA, variant=CROSTON_VARIANT
        )
        model.fit(self.y_train)

        # MAE no TREINO para calcular pesos do ensemble — evita vazar o conjunto
        # de teste na ponderação do servidor. O y_test é usado apenas em evaluate().
        _, mae_train, _, _ = model.evaluate(self.y_train)
        state = model.get_state()
        return [state], len(self.y_train), {"mae": mae_train}

    def evaluate(self, parameters, config):
        """
        Avalia o ensemble federado de modelos Croston.

        Layout esperado:
            parameters[0]   = pesos do ensemble (float64, tamanho N)
            parameters[1:]  = N estados [l, p]

        Previsão = média ponderada das N previsões individuais.
        """
        if not parameters or len(parameters) < 2:
            return float("inf"), len(self.y_test), {"mae": float("inf")}

        weights = parameters[0]
        states = [p for p in parameters[1:] if p.size == 2]

        if not states:
            return float("inf"), len(self.y_test), {"mae": float("inf")}

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
        ss_res = float(np.sum(errors ** 2))
        ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
        mse  = ss_res / len(y_test)
        mae  = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(mse))
        r2   = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return mse, len(self.y_test), {"mae": mae, "rmse": rmse, "r2": r2}


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
    result = {}
    for key in ("mae", "rmse", "r2"):
        if any(key in m for _, m in metrics):
            result[key] = sum(n * m.get(key, 0.0) for n, m in metrics) / total
    return result


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
