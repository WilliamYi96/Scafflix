from typing import Callable

from fedjax.core import federated_data
from fedjax.core import client_datasets


def train_preprocess_client(
    train_split: float
) -> Callable[[federated_data.ClientId, client_datasets.Examples],
              client_datasets.Examples]:
    """Creates a function which takes first
    int(train_split * # of examples) examples of client dataset
    for Grid Search train purposes."""
    def train_gd_preprocess_client(
      client_id: federated_data.ClientId,
      examples: client_datasets.Examples) -> client_datasets.Examples:
        return {key: value[:int(train_split * len(value))]
                for (key, value) in examples.items()}
    return train_gd_preprocess_client


def validation_preprocess_client(
    train_split: float
) -> Callable[[federated_data.ClientId, client_datasets.Examples],
              client_datasets.Examples]:
    """Creates a function which takes all examples
    after int(train_split * # of examples) index
    of client dataset for Grid Search validation purposes."""
    def validation_gd_preprocess_client(
      client_id: federated_data.ClientId,
      examples: client_datasets.Examples) -> client_datasets.Examples:
        return {key: value[int(train_split * len(value)):]
                for (key, value) in examples.items()}
    return validation_gd_preprocess_client