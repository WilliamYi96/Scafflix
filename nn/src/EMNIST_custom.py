from typing import Optional, Tuple
from fedjax.core import federated_data
from fedjax.datasets import emnist
from custom_utils import train_preprocess_client, validation_preprocess_client


def emnist_preprocess_train_split(
    fd: federated_data.FederatedData,
    train_val_split: float
) -> federated_data.FederatedData:
    return (fd.preprocess_client(emnist.preprocess_client).preprocess_client(
        train_preprocess_client(train_val_split)).preprocess_batch(
            emnist.preprocess_batch))


def emnist_preprocess_validation_split(
    fd: federated_data.FederatedData,
    train_val_split: float
) -> federated_data.FederatedData:
    return (fd.preprocess_client(emnist.preprocess_client).preprocess_client(
        validation_preprocess_client(train_val_split)).preprocess_batch(
        emnist.preprocess_batch))


def emnist_load_gd_data(
    train_val_split: float,
    only_digits: bool = False,
    mode: str = 'sqlite',
    cache_dir: Optional[str] = None
) -> Tuple[federated_data.FederatedData, federated_data.FederatedData]:
    """Loads processed EMNIST train and validation splits."""
    train = emnist.load_split(
        'train', only_digits=only_digits, mode=mode, cache_dir=cache_dir)
    return emnist_preprocess_train_split(train, train_val_split), \
        emnist_preprocess_validation_split(train, train_val_split)