"""
Federated Sampler representation.
"""

import random
from abc import ABC, abstractmethod
from typing import List, Union, Dict

import numpy as np


class Sampler(ABC):
    """
    Abstract base class for sampling clients.

    Args:
        clients (list): A list of client names.
    """
    clients: List[str] = []

    def __init__(self, clients):
        """Initializes the sampler with a list of clients."""
        self.clients = clients

    def __len__(self) -> int:
        """Returns the number of clients."""
        return len(self.clients)

    def __repr__(self):
        """Returns a string representation of the sampler."""
        return f"{self.__class__.__name__}"

    @abstractmethod
    def num_available(self, verbose: bool = True) -> int:
        """Returns the number of available clients. If verbose, prints the info."""
        if verbose:
            print(f"[INFO] Number of available clients: {len(self)}")
        return len(self)

    @abstractmethod
    def all(self) -> List[str]:
        """Returns all available clients."""
        return self.clients

    @abstractmethod
    def register(self, client_name: str) -> bool:
        """Registers a client by name. If the client is already registered, returns False."""
        if client_name in self.clients:
            return False
        self.clients.append(client_name)
        print(f"[INFO] Registered client with id: {client_name}")
        return True

    @abstractmethod
    def unregister(self, client_name: str) -> None:
        """Unregisters a client by name."""
        if client_name in self.clients:
            self.clients.remove(client_name)
            print(f"[INFO] Unregistered client {client_name}")

    @abstractmethod
    def sample(self, fl_round: int, c: float, verbose: bool = True, *args, **kwargs):
        """Abstract method to sample a number of clients."""


class RandomSampler(Sampler):
    """
    Sampler subclass that selects clients randomly.

    Initializes the random sampler with a list of clients and an optional seed.

    Args:
        clients: A list of clients.
        seed: An optional seed for reproducibility.
    """

    def __init__(self, clients: List[str], seed: Union[None, int] = None):
        if seed is not None:
            random.seed(seed)
        super().__init__(clients)

    def __len__(self) -> int:
        return super().__len__()

    def __repr__(self) -> str:
        return super().__repr__()

    def num_available(self, verbose: bool = True) -> int:
        return super().num_available(verbose)

    def all(self) -> List[str]:
        return super().all()

    def register(self, client_name: str) -> bool:
        return super().register(client_name)

    def unregister(self, client_name: str) -> None:
        return super().unregister(client_name)

    def sample(self, fl_round: int, c: float, verbose: bool = True, **kwargs) -> List[str]:
        """Samples a random subset of clients based on a given fraction c."""
        available_clients = self.all()
        num_available = self.num_available(verbose=False)
        if num_available == 0:
            print(f"[Error][Round={fl_round}] Cannot sample clients. The number of available clients is 0.")
            return []
        num_selection = int(c * num_available)
        if num_selection == 0:
            num_selection = 1
        if num_selection > num_available:
            num_selection = num_available
        sampled_clients = random.sample(available_clients, num_selection)
        if verbose:
            print(f"[INFO][Round={fl_round}] Sampled {num_selection} client(s): {sampled_clients} (c={c})")
        return sampled_clients


class QuantitySampler(Sampler):
    """
    Sampler subclass that selects clients based on their number of samples with respect to the total number of samples.

    For this sampler, we assume that the number of samples do not change or that the server asks for the number
    of samples.

    Initializes the quantity sampler with a list of clients and an optional seed.

    Args:
        clients: A list of clients.
        seed: An optional seed for reproducibility.
    """

    def __init__(self, clients: List[str], seed: Union[None, int] = None):
        if seed is not None:
            random.seed(seed)
        super().__init__(clients)

    def __len__(self) -> int:
        return super().__len__()

    def __repr__(self) -> str:
        return super().__repr__()

    def num_available(self, verbose: bool = True) -> int:
        return super().num_available(verbose)

    def all(self) -> List[str]:
        return super().all()

    def register(self, client_name: str) -> bool:
        return super().register(client_name)

    def unregister(self, client_name: str) -> None:
        return super().unregister(client_name)

    def sample(self, fl_round: int, c: float,
               samples_per_user: Dict[str, int] = None, verbose: bool = True, **kwargs) -> List[str]:
        """Samples a subset of clients based on a given fraction c."""
        if samples_per_user is None:
            samples_per_user = {client: 1 for client in self.clients}
        available_clients = self.all()
        num_available = self.num_available(verbose=False)
        if num_available == 0:
            print(f"[Error][Round={fl_round}] Cannot sample clients. The number of available clients is 0.")
            return []
        num_selection = int(c * num_available)
        if num_selection == 0:
            num_selection = 1
        if num_selection > num_available:
            num_selection = num_available

        probabilities = [val / sum(samples_per_user.values()) for val in samples_per_user.values()]
        sampled_clients = np.random.choice(available_clients, size=num_selection, p=probabilities, replace=False)

        if verbose:
            print(f"[INFO][Round={fl_round}] Sampled {num_selection} client(s): {sampled_clients} (c={c})")
        return sampled_clients


class StdSampler(Sampler):
    """
    Sampler subclass that selects clients based on their number of samples with respect to the total number of samples.

    For this sampler, we assume that the number of samples do not change or that the server asks for the number
    of samples.

    Initializes the quantity sampler with a list of clients and an optional seed.

    Args:
        clients: A list of clients.
        seed: An optional seed for reproducibility.
    """

    def __init__(self, clients: List[str], seed: Union[None, int] = None):
        if seed is not None:
            random.seed(seed)
        super().__init__(clients)

    def __len__(self) -> int:
        return super().__len__()

    def __repr__(self) -> str:
        return super().__repr__()

    def num_available(self, verbose: bool = True) -> int:
        return super().num_available(verbose)

    def all(self) -> List[str]:
        return super().all()

    def register(self, client_name: str) -> bool:
        return super().register(client_name)

    def unregister(self, client_name: str) -> None:
        return super().unregister(client_name)

    def sample(self, fl_round: int, c: float,
               std_per_user: Dict[str, int] = None, verbose: bool = True, **kwargs) -> List[str]:
        """Samples a subset of clients based on a given fraction c."""
        if std_per_user is None:
            std_per_user = {client: 1 for client in self.clients}
        available_clients = self.all()
        num_available = self.num_available(verbose=False)
        if num_available == 0:
            print(f"[Error][Round={fl_round}] Cannot sample clients. The number of available clients is 0.")
            return []
        num_selection = int(c * num_available)
        if num_selection == 0:
            num_selection = 1
        if num_selection > num_available:
            num_selection = num_available

        total_samples = sum(std_per_user.values())
        inverse_probabilities = [1 / (val / total_samples) for val in std_per_user.values()]

        # normalize the inverse probabilities to sum up to 1
        inverse_probabilities /= np.sum(inverse_probabilities)
        sampled_clients = np.random.choice(available_clients, size=num_selection, p=inverse_probabilities,
                                           replace=False)

        if verbose:
            print(f"[INFO][Round={fl_round}] Sampled {num_selection} client(s): {sampled_clients} (c={c})")
        return sampled_clients


def random_sampler_test():
    clients = [f"doctor{i}" for i in range(1, 11)]
    sampler = RandomSampler(clients, seed=0)
    for i in range(5):
        sampler.sample(i, 0.2)
    sampler.unregister("doctor1")
    print(sampler.all())
    sampler.sample(5, 0.5)
    sampler.register("doctor100")
    sampler.unregister("doctor1")
    sampler.register("doctor100")
    print(sampler.all())
    sampler.sample(6, 0.9)
    sampler.sample(7, 0.)
    sampler.sample(8, 2.5)
    print(sampler.__repr__())


def quantity_sampler_test():
    samples_client = {f"doctor{i}": random.randint(20, 150) for i in range(1, 6)}
    clients = list(samples_client.keys())
    sampler = QuantitySampler(clients, seed=0)
    print(sampler.__repr__())

    for i in range(5):
        sampler.sample(i, 0.9, samples_per_user=samples_client)


def std_sampler_test():
    samples_client = {f"doctor{i}": random.uniform(0.1, 2.) for i in range(1, 6)}
    clients = list(samples_client.keys())
    sampler = StdSampler(clients, seed=0)
    print(sampler.__repr__())
    print(samples_client)

    for i in range(5):
        sampler.sample(i, 0.9, std_per_user=samples_client)


if __name__ == "__main__":
    std_sampler_test()
