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

from Scafflix_computation import ScafflixHParams
from Scafflix_computation import ScafflixComputationParams
from Scafflix_computation import scafflix_computation_with_statistics

import jax.numpy as jnp
import fedjax

Grads = Params


@dataclasses.dataclass
class ScafflixGrid:
    scafflix_lrs: jnp.ndarray
    plm_lrs: jnp.ndarray
    client_lrs: jnp.ndarray
    scafflix_batch_sizes: list
    plm_batch_sizes: list


def scafflix_grid_search(
    train_fd: federated_data.FederatedData,
    validation_fd: federated_data.FederatedData,
    grad_fn: Callable[[Params, BatchExample, PRNGKey], Grads],
    grad_fn_eval: Callable[[Params, BatchExample], Grads],
    model: models.Model,
    alpha: float,
    p: float,
    plm_comp_params: PLMComputationProcessParams,
    scafflix_comp_params: ScafflixComputationParams,
    grid: ScafflixGrid,
    num_epochs_plm: int,
    num_clients_per_flix_round: int,
    save_file: str,
    grid_metrics: str = 'accuracy'
) -> jnp.ndarray:
    GridSearch_table = jnp.zeros(shape=(len(grid.plm_batch_sizes),
                                        len(grid.plm_lrs),
                                        len(grid.scafflix_batch_sizes),
                                        len(grid.scafflix_lrs),
                                        len(grid.client_lrs)))
    alphas_dict = {}
    p_dict = {}
    for cid in train_fd.client_ids():
        alphas_dict[cid] = alpha
        p_dict[cid] = p

    for plm_b_id, plm_batch_size in enumerate(grid.plm_batch_sizes):
        for plm_lr_id, plm_lr in enumerate(grid.plm_lrs):
            plm_comp_hparams = PLMComputationHParams(
                num_epochs_plm, plm_lr, plm_batch_size)
            fedjax.set_for_each_client_backend('pmap')
            PLM_dict = plm_computation(
                train_fd, grad_fn, plm_comp_hparams, plm_comp_params)
            for scafflix_b_id, scafflix_batch_size in enumerate(grid.scafflix_batch_sizes):
                for scafflix_lr_id, scafflix_lr in enumerate(grid.scafflix_lrs):
                    for client_lr_id, client_lr in enumerate(grid.client_lrs):
                        print('{}-{}-{}-{}-{}'.format(
                            plm_batch_size,
                            plm_lr,
                            scafflix_batch_size,
                            scafflix_lr,
                            client_lr)
                        )
                        p_rate = p / client_lr
                        p_rate_dict = {}
                        for cid in train_fd.client_ids():
                            p_rate_dict[cid] = p_rate
                        # fedjax.set_for_each_client_backend('debug')
                        flix_hparams = ScafflixHParams(
                            scafflix_lr,
                            client_lr,
                            num_clients_per_flix_round,
                            scafflix_batch_size
                        )
                        _, stats = scafflix_computation_with_statistics(
                            train_fd,
                            validation_fd,
                            grad_fn,
                            grad_fn_eval,
                            model,
                            PLM_dict,
                            alphas_dict,
                            p_dict,
                            p_rate_dict,
                            flix_hparams,
                            scafflix_comp_params,
                            scafflix_comp_params.num_rounds
                        )
                        assert len(stats) == 1
                        GridSearch_table = GridSearch_table.at[
                            plm_b_id,
                            plm_lr_id,
                            scafflix_b_id,
                            scafflix_lr_id,
                            client_lr_id].set(stats[0][grid_metrics])
                        jnp.save(save_file, GridSearch_table)
    return GridSearch_table