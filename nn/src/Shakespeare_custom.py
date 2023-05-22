import functools
from typing import Optional, Tuple
from fedjax.core import federated_data
from fedjax.datasets import shakespeare
from custom_utils import train_preprocess_client, validation_preprocess_client


def shakespeare_preprocess_train_split(
    fd: federated_data.FederatedData,
    train_val_split: float,
    sequence_length
) -> federated_data.FederatedData:
    preprocess = functools.partial(
      shakespeare.preprocess_client,
      sequence_length=sequence_length
    )
    return (fd.preprocess_client(preprocess).preprocess_client(
        train_preprocess_client(train_val_split)))


def shakespeare_preprocess_validation_split(
    fd: federated_data.FederatedData,
    train_val_split: float,
    sequence_length
) -> federated_data.FederatedData:
    preprocess = functools.partial(
      shakespeare.preprocess_client,
      sequence_length=sequence_length
    )
    return (fd.preprocess_client(preprocess).preprocess_client(
        validation_preprocess_client(train_val_split)))


def shakespeare_load_gd_data(
    train_val_split: float,
    sequence_length=80,
    mode: str = 'sqlite',
    cache_dir: Optional[str] = None
) -> Tuple[federated_data.FederatedData, federated_data.FederatedData]:
    """Loads preprocessed Shakespeare train and validation splits."""
    train = shakespeare.load_split('train', mode=mode, cache_dir=cache_dir)
    return (shakespeare_preprocess_train_split(
      train, train_val_split, sequence_length),
            shakespeare_preprocess_validation_split(
              train, train_val_split, sequence_length))