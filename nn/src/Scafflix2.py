from typing import Any, Callable, Mapping, Sequence, Tuple

from fedjax.core import client_datasets
from fedjax.core import dataclasses
from fedjax.core import federated_algorithm
from fedjax.core import federated_data
from fedjax.core import for_each_client
from fedjax.core import optimizers
from fedjax.core import tree_util
from fedjax.core.typing import BatchExample
from fedjax.core.typing import Params
from fedjax.core.typing import PRNGKey

import jax
import jax.numpy as jnp
import numpy as np
import fedjax

Grads = Params


def convex_combination(
    x_global: Params,
    x_local: Params,
    alpha: float
) -> Params:
    """Computes alpha * x_global + (1 - alpha) * x_local for PyTrees."""
    return tree_util.tree_add(tree_util.tree_weight(x_global, alpha),
                              tree_util.tree_weight(x_local, 1 - alpha))


def create_train_for_each_client(grad_fn, client_optimizer):
    """Builds client_init, client_step, client_final for for_each_client."""

    def client_init(server_params, client_input):
        opt_state = client_optimizer.init(server_params)
        client_plm = client_input['plm']
        client_alpha = client_input['alpha']
        client_rng = client_input['rng']
        p_rate = client_input['p_rate']
        p = client_input['p']

        client_step_state = {
            'params': server_params,
            'opt_state': opt_state,
            'rng': client_rng,
            'h': jax.tree_map(jnp.zeros_like, server_params),
            'p_rate': p_rate,
            'p': p,
            'plm': client_plm,
            'alpha': client_alpha
            # 'p_choices': np.random.choice([0, 1], p=[0.8, 0.2])# generate by numpy, TODO
        }
        return client_step_state

    def client_step(client_step_state, batch):
        rng, use_rng = jax.random.split(client_step_state['rng'])
        point = convex_combination(client_step_state['params'],
                                   client_step_state['plm'],
                                   client_step_state['alpha'])
        grads = grad_fn(point, batch, use_rng)
        grads = tree_util.tree_weight(grads, client_step_state['alpha'])
        adjusted_grads = jax.tree_map(lambda x, y: x - y, grads, client_step_state['h'])

        # grng = jax.random.PRNGKey(int(client_step_state['params'][0] * 100000))
        # p_choice = jax.random.choice(grng, jax.numpy.array([0, 1]), p=jax.numpy.array([1 - p, p]))

        opt_state, params = client_optimizer.apply(adjusted_grads,
                                                   client_step_state['opt_state'],
                                                   client_step_state['params'])
        # from jax import jit
        # from functools import partial
        # @partial(jit, static_argnums=(0,))
        # def choose_p():
        #     return np.random.choice([0, 1], p=[1-client_step_state['p'], client_step_state['p']])  # generate by numpy, TODO
        # # p_choice = client_step_state['choices'][client_step_state['p_idx']]
        # p_choice = choose_p

        p_choice = np.random.choice([0, 1], p=[0.9, 0.1])  # TODO: this should be the hps tuned later.
        # p_choice = jax.random.choice(use_rng, jax.numpy.array([0, 1]),
        #                              p=jax.numpy.array([1 - client_step_state['p'], client_step_state['p']]))

        if p_choice:  # For simplexity, we use the previous average
            updated_params = client_step_state['params']
        else:
            # updated_params = params
            updated_params = point

        diff = jax.tree_map(lambda x, y: x - y, updated_params, point)
        # weight_diff = fedjax.tree_util.tree_weight(diff, client_step_state['p'] / client_step_state['learning_rate'])
        weight_diff = fedjax.tree_util.tree_weight(diff, client_step_state['p_rate'])
        h = tree_util.tree_add(client_step_state['h'], weight_diff)

        next_client_step_state = {
            'params': params,
            'opt_state': opt_state,
            'rng': rng,
            'h': h,
            'p_rate': client_step_state['p_rate'],
            'p': client_step_state['p'],
            'plm': client_step_state['plm'],
            'alpha': client_step_state['alpha']
        }
        return next_client_step_state

    def client_final(server_params, client_step_state):
        delta_params = jax.tree_util.tree_map(lambda a, b: a - b,
                                              server_params,
                                              client_step_state['params'])
        return delta_params

    return for_each_client.for_each_client(client_init, client_step, client_final)


@dataclasses.dataclass
class ServerState:
    """State of server passed between rounds.
  Attributes:
    params: A pytree representing the server model parameters.
    opt_state: A pytree representing the server optimizer state.
  """
    params: Params
    # old_params: Params
    opt_state: optimizers.OptState


def scafflix(
        grad_fn: Callable[[Params, BatchExample, PRNGKey], Grads],
        client_optimizer: optimizers.Optimizer,
        server_optimizer: optimizers.Optimizer,
        client_batch_hparams: client_datasets.ShuffleRepeatBatchHParams,
        p_rates: dict,
        ps: dict,
        plms: dict,
        alphas: dict
) -> federated_algorithm.FederatedAlgorithm:
    """Builds federated averaging.
  Args:
    grad_fn: A function from (params, batch_example, rng) to gradients.
      This can be created with :func:`fedjax.core.model.model_grad`.
    client_optimizer: Optimizer for local client training.
    server_optimizer: Optimizer for server update.
    client_batch_hparams: Hyperparameters for batching client dataset for train.
  Returns:
    FederatedAlgorithm
    :param lr: client learning rate
    :param gamma:
    :param p:
    :param p_choices:
  """
    train_for_each_client = create_train_for_each_client(grad_fn, client_optimizer)

    def init(params: Params) -> ServerState:
        opt_state = server_optimizer.init(params)
        return ServerState(params, opt_state)

    def apply(
            server_state: ServerState,
            # p_choice: int,
            clients: Sequence[Tuple[federated_data.ClientId,
                                    client_datasets.ClientDataset, PRNGKey]],
            # p_choice: int,
            # p: float,
            # client_lr: float
    ) -> Tuple[ServerState, Mapping[federated_data.ClientId, Any]]:
        client_num_examples = {cid: len(cds) for cid, cds, _ in clients}
        # batch_clients = [(cid, cds.shuffle_repeat_batch(client_batch_hparams), crng)
        #                  for cid, cds, crng in clients]
        batch_clients = [
            (cid, cds.shuffle_repeat_batch(client_batch_hparams),
             {'alpha': alphas[cid], 'plm': plms[cid], 'rng': crng, 'p_rate': p_rates[cid], 'p': ps[cid]})
            for cid, cds, crng in clients
        ]
        client_diagnostics = {}
        # Running weighted mean of client updates. We do this iteratively to avoid
        # loading all the client outputs into memory since they can be prohibitively
        # large depending on the model parameters size.
        delta_params_sum = tree_util.tree_zeros_like(server_state.params)
        num_examples_sum = 0.
        for client_id, delta_params in train_for_each_client(server_state.params,
                                                             batch_clients):
            num_examples = client_num_examples[client_id]
            delta_params_sum = tree_util.tree_add(
                delta_params_sum, tree_util.tree_weight(delta_params, num_examples))
            num_examples_sum += num_examples
            # We record the l2 norm of client updates as an example, but it is not
            # required for the algorithm.
            client_diagnostics[client_id] = {
                'delta_l2_norm': tree_util.tree_l2_norm(delta_params)
            }
        mean_delta_params = tree_util.tree_inverse_weight(delta_params_sum,
                                                          num_examples_sum)
        server_state = server_update(server_state, mean_delta_params)
        # client_diagnostics['old_params'] = old_params
        return server_state, client_diagnostics

    def server_update(server_state, mean_delta_params):
        # old_params = server_state.params
        opt_state, params = server_optimizer.apply(mean_delta_params,
                                                   server_state.opt_state,
                                                   server_state.params)
        # server_state.old_params = old_params
        return ServerState(params, opt_state)

    return federated_algorithm.FederatedAlgorithm(init, apply)
