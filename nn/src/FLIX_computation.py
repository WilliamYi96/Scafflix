from typing import Callable, Tuple, Iterable, Dict


from fedjax.core import client_datasets
from fedjax.core import dataclasses
from fedjax.core import federated_data
from fedjax.core import metrics
from fedjax.core import models
from fedjax.core.client_samplers import UniformShuffledClientSampler
from fedjax.core.typing import BatchExample
from fedjax.core.typing import Params
from fedjax.core.typing import PRNGKey

import fedjax
import jax
import jax.numpy as jnp

import FLIX

Grads = Params


def evaluate_model(
    model: models.Model,
    params: Params,
    client_data: Iterable[Tuple[float, Params, client_datasets.ClientDataset]],
    client_batch_parameters: client_datasets.BatchHParams
) -> Dict[str, jnp.ndarray]:
    """Evaluates FLIX model for multiple batches and returns final results.
    Args:
        model: Model container.
        params: Pytree of model parameters to be evaluated.
        client_data: Tuple of client's alpha, pure local model and dataset.
        client_batch_parameters: Hyperparameters for batching client dataset
        for evaluation.
    Returns:
        A dictionary of evaluation :class:`~fedjax.metrics.Metric` results.
    """
    stat = {k: metric.zero() for k, metric in model.eval_metrics.items()}
    for alpha, plm, cds in client_data:
        personalized_params = FLIX.convex_combination(
            params, plm, alpha)
        for batch in cds.batch(client_batch_parameters):
            stat = models._evaluate_model_step(
                model, personalized_params, batch, stat)
    return jax.tree_util.tree_map(
        lambda x: x.result(), stat,
        is_leaf=lambda v: isinstance(v, metrics.Stat))


@dataclasses.dataclass
class FLIXHParams:
    """FLIX hyperparameters.
    Attributes:
        server_lr: A learning rate of a server optimizer
        client_lr: A learning rate of a client optimizer
        num_clients_per_round: A number of clients participating
        in training each round
        client_batch_size: Batch size of a client
    """
    server_lr: float
    client_lr: float
    num_clients_per_round: int
    client_batch_size: int


@dataclasses.dataclass
class FLIXComputationParams:
    """FLIX computation parameters.
    Attributes:
        server_optimizer: A server-lever optimizer.
        Allowed options are 'adam' and 'sgd'.
        client_optimizer: A client-level optimizer. 
        Allowed options are 'adam' and 'sgd'.
        init_params: An initialization of a model in the beginning
        of the training process.
        num_rounds: A number of rounds server communicates with clients.
    """
    server_optimizer: str
    client_optimizer: str
    # client_lr: float
    init_params: Params
    num_rounds: int


def flix_computation_with_statistics(
    train_fd: federated_data.FederatedData,
    validation_fd: federated_data.FederatedData,
    grad_fn: Callable[[Params, BatchExample, PRNGKey], Grads],
    grad_fn_eval: Callable[[Params, BatchExample], Grads],
    model: models.Model,
    plms: dict,
    alphas: dict,
    flix_hparams: FLIXHParams,
    flix_comp_params: FLIXComputationParams,
    stat_every: int
) -> Tuple[Params, list]:

    client_optimizer = fedjax.optimizers.sgd(learning_rate=flix_hparams.server_lr)
    # client_optimizer = fedjax.optimizers.sgd(learning_rate=flix_hparams.client_lr)
    server_optimizer = fedjax.optimizers.adam(learning_rate=flix_hparams.client_lr, b1=0.9, b2=0.999, eps=10**(-4))
    # server_optimizer = flix_comp_params.server_optimizer
    # client_optimizer = flix_comp_params.client_optimizer

    train_client_sampler = UniformShuffledClientSampler(
        shuffled_clients_iter=train_fd.shuffled_clients(buffer_size=100),
        num_clients=flix_hparams.num_clients_per_round)
    client_batch_hparams = fedjax.ShuffleRepeatBatchHParams(
        batch_size=flix_hparams.client_batch_size)
    algorithm = FLIX.flix(
        grad_fn, client_optimizer, server_optimizer, client_batch_hparams, plms, alphas)
    server_state = algorithm.init(flix_comp_params.init_params)
    stats = []
    for round_num in range(1, flix_comp_params.num_rounds + 1):
        print('Round {} / {}'.format(
            round_num, flix_comp_params.num_rounds), end='\r')
        clients = train_client_sampler.sample()
        # print(server_state, clients)
        server_state, _ = algorithm.apply(server_state, clients)

        if round_num % stat_every == 0:
            val_client_sampler = UniformShuffledClientSampler(
                shuffled_clients_iter=validation_fd.clients(),
                num_clients=validation_fd.num_clients())
            clients = val_client_sampler.sample()
            client_data_for_evaluation = [(alphas[cid], plms[cid], cds)
                                          for cid, cds, _ in clients]
            client_batch_hparams_eval = fedjax.BatchHParams(batch_size=256)
            grid_search_metrics = evaluate_model(model, server_state.params,
                                                 client_data_for_evaluation,
                                                 client_batch_hparams_eval)
            stats.append(grid_search_metrics)
            # print(f'Round {round_num} / {flix_comp_params.num_rounds}  ',
            #       grid_search_metrics['accuracy'])

    return server_state.params, stats


def flix_computation(
    train_fd: federated_data.FederatedData,
    grad_fn: Callable[[Params, BatchExample, PRNGKey], Grads],
    plms: dict,
    alphas: dict,
    flix_hparams: FLIXHParams,
    flix_comp_params: FLIXComputationParams
) -> Params:
    stat_every = flix_comp_params.num_rounds + 1
    params, stats = flix_computation_with_statistics(
        train_fd, None, grad_fn, None, None, plms, alphas,
        flix_hparams, flix_comp_params, None, stat_every
    )
    if len(stats) > 0:
        raise RuntimeError('stats should not get computed, \
            length is {}'.format(len(stats)))
    return params