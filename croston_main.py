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

CROSTON_ALPHA = 0.1    # suavização do nível de demanda
CROSTON_BETA = 0.1     # suavização do intervalo entre demandas
CROSTON_VARIANT = "sba"  # "original" ou "sba" (Syntetos-Boylan, menos viesado)

# ---------------------------------------------------------------------------
# Estratégia federada — FedAvg sobre [demand_level, interval]
#
# Cada cliente devolve um ndarray float64 de 2 elementos.
# O servidor calcula a média ponderada pelo número de amostras de cada cliente.
# ---------------------------------------------------------------------------

class CrostonFedAvgStrategy(FedAvg):

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        total_samples = sum(res.num_examples for _, res in results)
        if total_samples == 0:
            return None, {}

        # Média ponderada de [demand_level, interval] entre todos os clientes
        aggregated = np.zeros(2, dtype=np.float64)
        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            if arrays and arrays[0].size == 2:
                aggregated += fit_res.num_examples * arrays[0]

        aggregated /= total_samples
        params_aggregated = ndarrays_to_parameters([aggregated])

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        print(
            f"[Rodada {server_round:>2d}] Estado global agregado → "
            f"demand_level={aggregated[0]:.4f}, interval={aggregated[1]:.4f}"
        )
        return params_aggregated, metrics_aggregated

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
# Cliente Croston
#
# parameters (List[np.ndarray]) recebido pelo NumPyClient já está convertido
# pelo Flower — não é necessário chamar parameters_to_ndarrays aqui.
# ---------------------------------------------------------------------------

class FlowerCrostonClient(NumPyClient):

    def __init__(self, y_train: np.ndarray, y_test: np.ndarray):
        self.y_train = y_train
        self.y_test = y_test

    def fit(self, parameters, config):
        """
        Recebe o estado global do servidor, ajusta o Croston localmente
        com warm-start (quando disponível) e devolve o estado atualizado.
        """
        model = CrostonForecaster(
            alpha=CROSTON_ALPHA, beta=CROSTON_BETA, variant=CROSTON_VARIANT
        )

        if parameters and len(parameters) > 0 and parameters[0].size == 2:
            # Warm-start: suavização parte do estado global agregado
            global_state = parameters[0]
            model.fit(
                self.y_train,
                init_demand_level=float(global_state[0]),
                init_interval=float(global_state[1]),
            )
        else:
            # Rodada 1 — inicialização fria a partir dos dados locais
            model.fit(self.y_train)

        state = model.get_state()
        return [state], len(self.y_train), {}

    def evaluate(self, parameters, config):
        """
        Avalia o modelo global (estado recebido do servidor) nos dados de teste locais.
        """
        if not parameters or len(parameters) == 0 or parameters[0].size != 2:
            return float("inf"), len(self.y_test), {"mae": float("inf")}

        model = CrostonForecaster(
            alpha=CROSTON_ALPHA, beta=CROSTON_BETA, variant=CROSTON_VARIANT
        )
        model.set_state(parameters[0])

        mse, mae = model.evaluate(self.y_test)
        return mse, len(self.y_test), {"mae": mae}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    y_train, y_test = load_timeseries(
        partition_id=partition_id,
        num_partitions=NUM_PARTITIONS,
    )
    return FlowerCrostonClient(y_train, y_test).to_client()


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
    # Array vazio sinaliza "sem estado global ainda" (cold start na rodada 1)
    empty_state = np.array([], dtype=np.float64)
    initial_params = ndarrays_to_parameters([empty_state])

    strategy = CrostonFedAvgStrategy(
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
    print("\n=== Histórico de Treinamento (Croston Federado) ===")
    print(f"Losses distribuídas (MSE): {hist.losses_distributed}")
    print(f"Métricas distribuídas (MAE): {hist.metrics_distributed}")
else:
    print("\n[AVISO] Simulação não retornou histórico.")
