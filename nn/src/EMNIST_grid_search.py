import argparse
import fedjax
import jax
import jax.numpy as jnp
import PLM_computation
import FLIX_computation
from grid_search import FLIXGrid, grid_search
from EMNIST_custom import emnist_load_gd_data
import itertools


def loss(params, batch, rng):
    # `rng` used with `apply_for_train` to apply dropout during training.
    preds = model.apply_for_train(params, batch, rng)
    # Per example loss of shape [batch_size].
    example_loss = model.train_loss(batch, preds)
    return jnp.mean(example_loss)


def loss_for_eval(params, batch):
    preds = model.apply_for_eval(params, batch)
    example_loss = model.train_loss(batch, preds)
    return jnp.mean(example_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', dest='alpha', type=float, action='store', help='Set alpha', default=0.2)
    args = parser.parse_args()
    alpha = args.alpha

    model = fedjax.models.emnist.create_conv_model(only_digits=False)
    grad_fn = jax.jit(jax.grad(loss))
    grad_fn_eval = jax.jit(jax.grad(loss_for_eval))

    CACHE_DIR = '../data/'
    NUM_CLIENTS_GRID_SEARCH = 3400
    TRAIN_VALIDATION_SPLIT = 0.8
    NUM_CLIENTS_PER_PLM_ROUND = 5
    NUM_CLIENTS_PER_FEDMIX_ROUND = 10
    FEDMIX_ALGORITHM = 'adam'
    FEDMIX_NUM_ROUNDS = 500
    PLM_NUM_EPOCHS = 100

    train_fd, validation_fd = emnist_load_gd_data(
        train_val_split=TRAIN_VALIDATION_SPLIT,
        only_digits=False,
        cache_dir=CACHE_DIR)
    client_ids = set([cid for cid in itertools.islice(
        train_fd.client_ids(), NUM_CLIENTS_GRID_SEARCH)])
    train_fd = fedjax.SubsetFederatedData(train_fd, client_ids)
    validation_fd = fedjax.SubsetFederatedData(validation_fd, client_ids)
    plm_init_params = model.init(jax.random.PRNGKey(0))
    plm_comp_params = PLM_computation.PLMComputationProcessParams(
        plm_init_params, NUM_CLIENTS_PER_PLM_ROUND)
    flix_init_params = model.init(jax.random.PRNGKey(20))
    flix_comp_params = FLIX_computation.FLIXComputationParams(
        FEDMIX_ALGORITHM, flix_init_params, FEDMIX_NUM_ROUNDS)
    flix_lrs = 10**jnp.arange(-5., 0.5, 1)
    flix_batch_sizes = [20, 50, 100, 200]
    plm_lrs = 10**jnp.arange(-5., 0.5, 1)
    plm_batch_sizes = [10, 20, 50, 100]
    grid = FLIXGrid(flix_lrs, plm_lrs, flix_batch_sizes, plm_batch_sizes)
    SAVE_FILE = '../results/EMNIST_{}_gd.npy'.format(int(10 * alpha))
    table = grid_search(
        train_fd, validation_fd, grad_fn, grad_fn_eval, model, alpha,
        plm_comp_params, flix_comp_params, grid, PLM_NUM_EPOCHS,
        NUM_CLIENTS_PER_FEDMIX_ROUND, SAVE_FILE
    )