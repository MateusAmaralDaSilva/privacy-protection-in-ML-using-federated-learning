"""fltabular: Flower Example on Adult Census Income Tabular Dataset."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from flwr_datasets import FederatedDataset
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from flwr_datasets.partitioner import IidPartitioner

fds = None  # Cache FederatedDataset

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_weights(net):
    ndarrays = [val.cpu().numpy() for _, val in net.state_dict().items()]
    return ndarrays

"""fltabular: Flower Example on Adult Census Income Tabular Dataset."""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train(self.net, self.trainloader)
        return get_weights(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = evaluate(self.net, self.testloader)
        return loss, len(self.testloader), {"accuracy": accuracy}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]

    train_loader, test_loader = load_data(
        partition_id=partition_id, num_partitions=NUM_PARTITIONS
    )
    net = IncomeClassifier()
    return FlowerClient(net, train_loader, test_loader).to_client()

"""fltabular: Flower Example on Adult Census Income Tabular Dataset."""

from flwr.common import ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.common import Context



def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context) -> ServerAppComponents:
    net = IncomeClassifier()
    params = ndarrays_to_parameters(get_weights(net))

    strategy = FedAvg(
        initial_parameters=params,
        evaluate_metrics_aggregation_fn=weighted_average,
        fraction_fit=0.2,  # 10% clients sampled each round to do fit()
        fraction_evaluate=0.5,  # 50% clients sample each round to do evaluate()
    )
    num_rounds = NUM_SERVER_ROUNDS
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(config=config, strategy=strategy)

from flwr.simulation import run_simulation

NUM_PARTITIONS = 15
NUM_SERVER_ROUNDS = 10

server_app = ServerApp(server_fn=server_fn)
client_app = ClientApp(client_fn=client_fn)


hist = run_simulation(
    server_app=server_app, client_app=client_app, num_supernodes=NUM_PARTITIONS
)

