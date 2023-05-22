from typing import Callable

from fedjax.core import federated_data
from fedjax.core import dataclasses
from fedjax.core import models

from fedjax.core.typing import BatchExample
from fedjax.core.typing import Params
from fedjax.core.typing import PRNGKey

from PLM_computation import PLMComputationProcessParams
from PLM_computation import PLMComputationHParams
from PLM_computation import plm_computation

from FLIX_computation import FLIXHParams
from FLIX_computation import FLIXComputationParams
from FLIX_computation import flix_computation_with_statistics

import jax.numpy as jnp
import fedjax

Grads = Params


@dataclasses.dataclass
class FLIXGrid:
    flix_lrs: jnp.ndarray
    plm_lrs: jnp.ndarray
    client_lrs: jnp.ndarray
    flix_batch_sizes: list
    plm_batch_sizes: list


def grid_search(
    train_fd: federated_data.FederatedData,
    validation_fd: federated_data.FederatedData,
    grad_fn: Callable[[Params, BatchExample, PRNGKey], Grads],
    grad_fn_eval: Callable[[Params, BatchExample], Grads],
    model: models.Model,
    alpha: float,
    plm_comp_params: PLMComputationProcessParams,
    flix_comp_params: FLIXComputationParams,
    grid: FLIXGrid,
    num_epochs_plm: int,
    num_clients_per_flix_round: int,
    save_file: str,
    grid_metrics: str = 'accuracy'
) -> jnp.ndarray:
    GridSearch_table = jnp.zeros(shape=(len(grid.plm_batch_sizes),
                                        len(grid.plm_lrs),
                                        len(grid.flix_batch_sizes),
                                        len(grid.flix_lrs),
                                        len(grid.client_lrs)))
    alphas_dict = {}
    for cid in train_fd.client_ids():
        alphas_dict[cid] = alpha
    for plm_b_id, plm_batch_size in enumerate(grid.plm_batch_sizes):
        for plm_lr_id, plm_lr in enumerate(grid.plm_lrs):
            plm_comp_hparams = PLMComputationHParams(
                num_epochs_plm, plm_lr, plm_batch_size)
            fedjax.set_for_each_client_backend('pmap')
            PLM_dict = plm_computation(
                train_fd, grad_fn, plm_comp_hparams, plm_comp_params)
            for flix_b_id, \
                    flix_batch_size in enumerate(grid.flix_batch_sizes):
                for flix_lr_id, flix_lr in enumerate(grid.flix_lrs):
                    for client_lr_id, client_lr in enumerate(grid.client_lrs):
                        print('{}-{}-{}-{}-{}'.format(
                            plm_batch_size,
                            plm_lr,
                            flix_batch_size,
                            flix_lr,
                            client_lr)
                        )
                        # fedjax.set_for_each_client_backend('debug')
                        flix_hparams = FLIXHParams(
                            flix_lr,
                            client_lr,
                            num_clients_per_flix_round,
                            flix_batch_size
                        )
                        _, stats = flix_computation_with_statistics(
                            train_fd,
                            validation_fd,
                            grad_fn,
                            grad_fn_eval,
                            model,
                            PLM_dict,
                            alphas_dict,
                            flix_hparams,
                            flix_comp_params,
                            flix_comp_params.num_rounds
                        )
                        assert len(stats) == 1
                        GridSearch_table = GridSearch_table.at[
                            plm_b_id,
                            plm_lr_id,
                            flix_b_id,
                            flix_lr_id,
                            client_lr_id].set(stats[0][grid_metrics])
                        jnp.save(save_file, GridSearch_table)
    return GridSearch_table