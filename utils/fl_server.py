"""
Federated Server for simulating federated training.
"""

import copy
import math
import time
from collections import OrderedDict, defaultdict
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import tenseal as ts
import torch.nn
from torch.nn import Module
from torch.utils.data import DataLoader

from utils.fl_aggregator import Aggregator
from utils.fl_sampler import RandomSampler, Sampler
from utils.homomorphic import TrustedAuthority, FHEServer, FHEClient
from utils.train_utils import fit


def compute_mean(dataloader):
    total_sum = 0.0
    total_count = 0

    for batch in dataloader:
        data, _ = batch  # Assuming each batch is a tuple of (data, labels)
        total_sum += data.sum().item()
        total_count += data.numel()

    return total_sum / total_count


def compute_mean_of_squared(dataloader):
    total_sum_of_squares = 0.0
    total_count = 0

    for batch in dataloader:
        data, _ = batch  # Assuming each batch is a tuple of (data, labels)
        total_sum_of_squares += (data ** 2).sum().item()
        total_count += data.numel()

    return total_sum_of_squares / total_count


def compute_std(dataloader):
    mean = compute_mean(dataloader)
    mean_of_squared = compute_mean_of_squared(dataloader)
    return math.sqrt(mean_of_squared - mean ** 2)


def weighted_metric_avg(n_per_client: Union[List[int], None], metrics: List[float]) -> float:
    """Aggregates losses or metrics received from clients."""
    if n_per_client is not None:
        n = sum(n_per_client)
        weighted_metrics = [n_k * metric for n_k, metric in zip(n_per_client, metrics)]
    else:
        n = len(metrics)
        weighted_metrics = [metric for metric in metrics]
    return sum(weighted_metrics) / n


class Server:
    """
    Represents a server in a Federated Learning setup which coordinates the clients,
    aggregates the results, and manages the global model.

    This class is for simulation purposes only.

    Args:
        model (torch.nn.Module): The global model shared across clients.
        clients (Dict[str, Dict[str, DataLoader]]): Dictionary mapping of client IDs with their respective data loaders.
        clients_transformed (Dict[str, Dict[str, np.ndarray]]): Dictionary mapping of client IDs to 2D transformed data.
        aggregation_algo (str): Aggregation algorithm for aggregating client updates.
        sampler (Sampler): Client sampler used to determine which clients should train in each round.
        training_params (Dict[str, Dict[str, Any]]): Dictionary of training parameters for each client.
        aggregation_params (Dict[str, Any], optional): Parameters for the aggregation algorithm. Default is None.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 clients: Dict[str, Dict[str, DataLoader]],
                 clients_transformed: Dict[str, Dict[str, np.ndarray]],
                 aggregation_algo: str,
                 sampler: Sampler,
                 training_params: Dict[str, Dict[str, Any]],
                 aggregation_params: Dict[str, Any] = None,
                 fhe: bool = False
                 ):
        self.global_model = model
        self.clients = clients
        self.clients_transformed = clients_transformed
        client_list = list(self.clients.keys())
        assert client_list == list(clients_transformed.keys())
        self.weighted_metric = weighted_metric_avg
        if aggregation_algo is None:
            aggregation_algo = "fedavg"
        self.aggregator = Aggregator(
            aggregation_alg=aggregation_algo,
            params=aggregation_params
        )
        print(f"Initialized Aggregator: {repr(self.aggregator)}")

        if sampler is None:
            self.sampler = RandomSampler(list(clients.keys()))
        else:
            self.sampler = sampler
        print(f"Sampler: {self.sampler.__repr__()}. Clients: {self.sampler.clients}")

        self.training_params = training_params

        self.fhe = fhe
        self.fhe_server: Union[None, FHEServer] = None
        self.fhe_client: Union[None, FHEClient] = None
        if self.fhe:
            self.make_crypto()

    def make_crypto(self):
        ta = TrustedAuthority()
        ta.create_context()
        context = ta.context
        print("[FHE] Generated Context")
        self.fhe_server = FHEServer(context=context)
        print("[FHE] Initialized FHE SERVER")
        self.fhe_client = FHEClient(context=context)
        print("[FHE] Initialized FHE Client")

    def fit(self,
            num_rounds: int,
            selection_fraction: float,
            return_last_model: bool = False
            ) -> Union[Tuple[Module, Dict[Any, Module], Dict[str, List[float]]], Tuple[Module, Dict[str, List[float]]]]:
        """
        Simulates federated training over a specified number of rounds.

        Args:
            num_rounds (int): Number of federated learning rounds.
            selection_fraction (float): Fraction of clients to sample in each round.

        Returns:
            torch.nn.Module: Updated global model after training.

        Note:
            This is for simulation purposes. In actual deployment, the server communicates weights to clients
            and does not directly access client data or train client models.
        """
        start_time = time.time()

        # federated training loop
        # THIS IS ONLY FOR SIMULATION PURPOSES. In practice, the server should communicate the global weights to
        # the selected clients, which have local fit() and evaluate() methods (i.e., we cannot actually make a
        # loop inside the server).
        global_history: Dict[str, List[float]] = defaultdict(list)
        client_rounds: Dict[Any, List[int]] = dict()  # a list that shows the rounds that each client has been selected
        client_models: Dict[Any, Dict[int, torch.nn.Module]] = dict()  # local models per round

        for fl_round in range(1, num_rounds + 1):
            # STEP 1: Sample available clients
            if self.sampler.__repr__() == "RandomSampler":
                selected_clients = self.sampler.sample(
                    fl_round, selection_fraction, verbose=True
                )
            elif self.sampler.__repr__() == "QuantitySampler":
                clients = self.sampler.all()
                samples_per_client = {}
                for client in clients:
                    samples_per_client[client] = len(self.clients[client]['train'].dataset)
                selected_clients = self.sampler.sample(
                    fl_round, selection_fraction, samples_per_client, verbose=True
                )
            elif self.sampler.__repr__() == "StdSampler":
                clients = self.sampler.all()
                std_per_client = {}
                for client in clients:
                    dataloader = self.clients[client]['train']
                    std = compute_std(dataloader)
                    std_per_client[client] = std
                print(std_per_client)
                selected_clients = self.sampler.sample(
                    fl_round, selection_fraction, std_per_client, verbose=True
                )
            else:
                raise ValueError

            # update the client rounds
            for client in selected_clients:
                if client not in client_rounds:
                    client_rounds[client] = []
                client_rounds[client].append(fl_round)

            num_train_examples: List[int] = []
            num_test_examples: List[int] = []
            round_history: Dict[str, List[float]] = {
                # losses
                "train_losses": [],
                "test_losses": [],
                "test_losses_normal": [],
                "test_losses_abnormal": [],
                "test_losses_unknown": [],
                # metrics
                "sireos": [],
                "p_scores": [],
                "auc_roc": [],
                "ap_scores": []
            }
            local_weights: Union[List[Tuple[List[np.ndarray], int]], List[ts.CKKSVector]] = []
            local_n_train: List[ts.CKKSVector] = []

            # STEP 2: Sampled clients to perform local training
            local_models = dict()
            for client in selected_clients:
                local_model = copy.deepcopy(self.global_model)
                # IGNORE THE FOLLOWING
                # if fl_round > 1:
                #    for k, val in local_model.state_dict().items():
                #        print(k)
                #        if "lpl" in k:
                #            prev_round = max(client_models[client].keys())
                #            prev_model = client_models[client][prev_round]
                #            # set the value from the previous local model to the current local model
                #            local_model.state_dict()[k].data.copy_(prev_model.state_dict()[k].data)'''

                num_train = len(self.clients[client]['train'].dataset)
                num_test = len(self.clients[client]['test'].dataset)
                if self.fhe:
                    enc_num_train = self.fhe_client.encrypt(np.array([num_train]))

                updated_model, client_history, _, _ = fit(
                    model=local_model,
                    train_loader=self.clients[client]['train'],
                    test_loader=self.clients[client]['test'],
                    data_loader=None,
                    criterion=self.training_params[client]['criterion'],
                    optim=self.training_params[client]['optimizer'],
                    optim_args=self.training_params[client]['optim_args'],
                    lr=self.training_params[client]['lr'],
                    epochs=self.training_params[client]['local_epochs'],
                    x_transformed_test=self.clients_transformed[client]['test'],
                    x_transformed_data=None,
                    percentile=self.training_params[client]['percentile'],
                    normalize_scores=self.training_params[client]['normalize'],
                    kappas=self.training_params[client]['kappas'],
                    log_interval=self.training_params[client]['log_interval'],
                    plot_interval=self.training_params[client]['plot_interval'],
                    device=self.training_params[client]['device'],
                    plot_history=False,
                    fl_note=client
                )
                # keep history
                num_train_examples.append(num_train)
                num_test_examples.append(num_test)

                client_history = {key: value[-1] if value else None for key, value in client_history.items()}
                for k in round_history.keys():
                    if client_history[k] is not None:
                        round_history[k].append(client_history[k])

                if fl_round == num_rounds:
                    local_models[client] = copy.deepcopy(updated_model)
                if client not in client_models.keys():
                    client_models[client] = dict()
                client_models[client][fl_round] = copy.deepcopy(updated_model)

                updated_model_np = []
                shapes = {}
                for k, val in updated_model.state_dict().items():
                    if self.fhe:
                        updated_model_np.append(val.cpu().numpy().reshape(-1))
                    else:
                        updated_model_np.append(val.cpu().numpy())
                    shapes[k] = val.shape
                    # IGNORE
                    # if "lpl" not in k:
                    #    updated_model_np.append(val.cpu().numpy())
                    # else:
                    #    updated_model_np.append(self.global_model.state_dict()[k].cpu().numpy())
                if self.fhe:
                    updated_model_np = np.concatenate(updated_model_np)
                    # client encrypts its local model
                    encrypted_model_np = self.fhe_client.encrypt(num_train * updated_model_np)
                    # the server gets the encrypted vectors and the encrypted number of samples
                    local_weights.append(encrypted_model_np)
                    local_n_train.append(enc_num_train)
                    # Note: In a real world scenario this involves additional overhead. We just demonstrate
                    # the efficacy and viability of FHE.

                else:
                    # STEP 3: Collect local models
                    local_weights.append((updated_model_np, num_train))

            # (Step 3.5) Generate metric-based report
            for k, v in round_history.items():
                if len(v) == 0 or isinstance(next(iter(v)), dict):
                    # Flattening the metric dictionaries into a single dictionary
                    tmp_metric = defaultdict(list)
                    for client_metric in v:
                        for metric_k, value in client_metric.items():
                            tmp_metric[metric_k].append(value)

                    # Calculating the weighted metric for each metric key
                    weighted_metric = {key: self.weighted_metric(None, values) for key, values in tmp_metric.items()}
                else:
                    weighted_metric = self.weighted_metric(None, v)

                global_history[k].append(weighted_metric)

            # STEP 4: Aggregate local models
            if self.fhe:
                # The server performs encrypted aggregation
                summed_weights, summed_n, _ = self.fhe_server.fedavg(
                    local_weights, local_n_train
                )
                # the clients decrypt the weight sum and the sum of samples
                weights = np.array(summed_weights.decrypt())
                n = summed_n.decrypt()[0]
                # clients also locally aggregate the model
                aggregated = weights / n
                # transform the vectors back to tensors
                global_model_np = self._vector_to_model(aggregated, shapes)

            else:
                global_model_np = self.aggregator.aggregate(
                    local_weights, self._get_global_weights()
                )

            self._set_global_weights(global_model_np)
            print(f"[INFO] Federated round no. {fl_round} completed.")

        end_time = time.time()
        global_history = dict(global_history)
        print(f"FL finished in {end_time - start_time} seconds.")
        if return_last_model:
            return self.global_model, local_models, global_history
        return self.global_model, global_history

    def _set_global_weights(self, parameters: Union[List[np.ndarray], torch.nn.Module]) -> None:
        """
        Update the global model with given parameters.

        Args:
            parameters (Union[List[np.ndarray], torch.nn.Module]): Parameters to update the global model with.
        """
        if not isinstance(parameters, torch.nn.Module):
            params_dict = zip(self.global_model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            self.global_model.load_state_dict(state_dict, strict=True)
        else:
            self.global_model.load_state_dict(parameters.state_dict(), strict=True)

    def _get_global_weights(self) -> List[np.ndarray]:
        """
        Retrieve the weights of the global model.

        Returns:
            List[np.ndarray]: A list of Numpy arrays representing the weights of the global model.
        """
        return [val.cpu().numpy() for _, val in self.global_model.state_dict().items()]

    def _vector_to_model(self, vector: np.ndarray, shapes: Dict[Any, int]):
        tensors = {}
        idx = 0
        for k, shape in shapes.items():
            # compute number of elements in this tensor
            num_elements = np.prod(shape)

            # extract relevant portion from the numpy vector
            tensor_flat = vector[idx:idx + num_elements]

            # reshape to get the original tensor
            tensor = tensor_flat.reshape(shape)

            tensors[k] = np.array(tensor, dtype='float32')

            idx += num_elements
        tensors = list(tensors.values())
        return tensors
