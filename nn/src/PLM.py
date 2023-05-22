from typing import Any, Callable, Mapping, Sequence, Tuple

from fedjax.core import client_datasets
from fedjax.core import dataclasses
from fedjax.core import federated_algorithm
from fedjax.core import federated_data
from fedjax.core import for_each_client
from fedjax.core import optimizers
from fedjax.core.typing import BatchExample
from fedjax.core.typing import Params
from fedjax.core.typing import PRNGKey

import jax

Grads = Params


def create_train_for_each_client(grad_fn, client_optimizer):
    """Builds client_init, client_step, client_final for for_each_client."""

    def client_init(server_params, client_rng):
        opt_state = client_optimizer.init(server_params)
        client_step_state = {
            'params': server_params,
            'opt_state': opt_state,
            'rng': client_rng,
        }
        return client_step_state

    def client_step(client_step_state, batch):
        rng, use_rng = jax.random.split(client_step_state['rng'])
        grads = grad_fn(client_step_state['params'], batch, use_rng)
        opt_state, params = client_optimizer.apply(
            grads, client_step_state['opt_state'], client_step_state['params'])
        next_client_step_state = {
          'params': params,
          'opt_state': opt_state,
          'rng': rng,
        }
        return next_client_step_state

    def client_final(server_params, client_step_state):
        del server_params
        return client_step_state['params']

    return for_each_client.for_each_client(
        client_init, client_step, client_final)


@dataclasses.dataclass
class ServerState:
    """State of server passed between rounds.
    Attributes:
        params: A pytree representing the server model parameters.
        PLM: A pytree representing pure local models of each client
    """
    params: Params
    PLM: Params


def PLM(
    grad_fn: Callable[[Params, BatchExample, PRNGKey], Grads],
    client_optimizer: optimizers.Optimizer,
    client_batch_hparams: client_datasets.ShuffleRepeatBatchHParams
) -> federated_algorithm.FederatedAlgorithm:
    """Builds computing pure local models algorithm.
    Args:
        grad_fn: A function from (params, batch_example, rng) to gradients.
        This can be created with :func:`fedjax.core.model.model_grad`.
        client_optimizer: Optimizer for local client training.
        client_batch_hparams: Hyperparameters for batching client dataset.
    Returns:
        federated_algorithm.FederatedAlgorithm:
    """
    train_for_each_client = create_train_for_each_client(
        grad_fn, client_optimizer)

    def init(params: Params) -> ServerState:
        PLM_dict = {}
        return ServerState(params, PLM_dict)

    def apply(
        server_state: ServerState,
        clients: Sequence[Tuple[
            federated_data.ClientId, client_datasets.ClientDataset, PRNGKey]]
    ) -> Tuple[ServerState, Mapping[federated_data.ClientId, Any]]:
        batch_clients = [(cid, cds.shuffle_repeat_batch(client_batch_hparams),
                          crng) for cid, cds, crng in clients]
        client_diagnostics = {}
        for client_id, client_plm in train_for_each_client(server_state.params,
                                                           batch_clients):
            server_state.PLM[client_id] = client_plm
        return server_state, client_diagnostics
    return federated_algorithm.FederatedAlgorithm(init, apply)