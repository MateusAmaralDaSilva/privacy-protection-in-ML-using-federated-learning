from collections import OrderedDict

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation

from dataset import load_data
from experiment import evaluate, train
from model import DemandForecaster

# Federated learning config
NUM_PARTITIONS = 10      # one partition per Walmart store
NUM_SERVER_ROUNDS = 10

# Differential privacy config (DP-SGD)
# Increase NOISE_MULTIPLIER for stronger privacy (at the cost of accuracy).
CLIP_NORM = 1.0
NOISE_MULTIPLIER = 1.0


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train(
            self.net,
            self.trainloader,
            clip_norm=CLIP_NORM,
            noise_multiplier=NOISE_MULTIPLIER,
            use_dp=True,
        )
        return get_weights(self.net), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        mse, mae = evaluate(self.net, self.testloader)
        return float(mse), len(self.testloader.dataset), {"mae": float(mae)}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    train_loader, test_loader = load_data(
        partition_id=partition_id,
        num_partitions=NUM_PARTITIONS,
    )
    net = DemandForecaster()
    return FlowerClient(net, train_loader, test_loader).to_client()


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def weighted_average(metrics):
    total_samples = sum(n for n, _ in metrics)
    weighted_mae = sum(n * m["mae"] for n, m in metrics)
    return {"mae": weighted_mae / total_samples}


def server_fn(context: Context) -> ServerAppComponents:
    net = DemandForecaster()
    params = ndarrays_to_parameters(get_weights(net))
    strategy = FedAvg(
        initial_parameters=params,
        evaluate_metrics_aggregation_fn=weighted_average,
        fraction_fit=0.3,
        fraction_evaluate=0.3,
    )
    return ServerAppComponents(
        config=ServerConfig(num_rounds=NUM_SERVER_ROUNDS),
        strategy=strategy,
    )


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

server_app = ServerApp(server_fn=server_fn)
client_app = ClientApp(client_fn=client_fn)

hist = run_simulation(
    server_app=server_app,
    client_app=client_app,
    num_supernodes=NUM_PARTITIONS,
    backend_config={"client_resources": {"num_cpus": 6, "num_gpus": 0}},
)
