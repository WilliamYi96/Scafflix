import fedjax
import jax
import jax.numpy as jnp
import PLM_computation
import FLIX_computation
from grid_search_general import FLIXGrid, grid_search
from Shakespeare_custom import shakespeare_load_gd_data
import itertools

from matplotlib import pyplot as plt
import pickle

model = fedjax.models.shakespeare.create_lstm_model()
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

grad_fn = jax.jit(jax.grad(loss))
grad_fn_eval = jax.jit(jax.grad(loss_for_eval))

CACHE_DIR = '../data/'
NUM_CLIENTS_GRID_SEARCH = 715
TRAIN_VALIDATION_SPLIT = 0.8
NUM_CLIENTS_PER_PLM_ROUND = 5
NUM_CLIENTS_PER_FEDMIX_ROUND = 10
FEDMIX_ALGORITHM = 'sgd'
CLIENT_ALGORITHM = 'sgd'
FEDMIX_NUM_ROUNDS = 500
PLM_NUM_EPOCHS = 25

train_fd, validation_fd = shakespeare_load_gd_data(
    train_val_split=TRAIN_VALIDATION_SPLIT,
    cache_dir=CACHE_DIR
)

client_ids = set([cid for cid in itertools.islice(
    train_fd.client_ids(), NUM_CLIENTS_GRID_SEARCH)])

train_fd = fedjax.SubsetFederatedData(train_fd, client_ids)
validation_fd = fedjax.SubsetFederatedData(validation_fd, client_ids)

plm_init_params = model.init(jax.random.PRNGKey(0))

plm_comp_params = PLM_computation.PLMComputationProcessParams(
    plm_init_params, NUM_CLIENTS_PER_PLM_ROUND)

fedmix_init_params = model.init(jax.random.PRNGKey(20))

fedmix_comp_params = FLIX_computation.FLIXComputationParams(
    FEDMIX_ALGORITHM, CLIENT_ALGORITHM, fedmix_init_params, FEDMIX_NUM_ROUNDS)

alpha = 0.7

fedmix_lrs = 10 ** jnp.arange(-1, 1.1, 0.5)
fedmix_batch_sizes = [1, 4, 10, 20]
plm_lrs = 10 ** jnp.arange(-1, 1.1, 0.5)
plm_batch_sizes = [1, 4, 10, 20]
client_lrs = 10 ** jnp.arange(-1, 1.1, 0.5)

grid = FLIXGrid(fedmix_lrs,
                  plm_lrs, client_lrs,
                  fedmix_batch_sizes,
                  plm_batch_sizes
                 )

SAVE_FILE = '../results/fedavg_fedmix_Shakespeare_debug_{}_gd.npy'.format(
    int(10 * alpha))

table = grid_search(
    train_fd, validation_fd, grad_fn, grad_fn_eval, model, alpha,
    plm_comp_params, fedmix_comp_params, grid, PLM_NUM_EPOCHS,
    NUM_CLIENTS_PER_FEDMIX_ROUND, SAVE_FILE, grid_metrics='accuracy_in_vocab'
)

table = jnp.load(SAVE_FILE)
best_ind = jnp.unravel_index(jnp.argmax(table), table.shape)
plm_batch_size = plm_batch_sizes[best_ind[0]]
plm_lr = plm_lrs[best_ind[1]]
fedmix_batch_size = fedmix_batch_sizes[best_ind[2]]
fedmix_lr = fedmix_lrs[best_ind[3]]
client_lr = client_lrs[best_ind[4]]

num_rounds = 3000
train_fd, test_fd = fedjax.datasets.shakespeare.load_data(cache_dir='../data/')
plm_comp_hparams = PLM_computation.PLMComputationHParams(PLM_NUM_EPOCHS,
                                                         plm_lr,
                                                         plm_batch_size)
PLM_dict = PLM_computation.plm_computation(train_fd,
                                           grad_fn,
                                           plm_comp_hparams,
                                           plm_comp_params)
save_file = '../results/PLM_Shakespeare_{}_{}.pickle'.format(best_ind[0], best_ind[1])
alpha_dict = {}
for cid in train_fd.client_ids():
    alpha_dict[cid] = alpha

fedmix_hparams = FLIX_computation.FLIXHParams(
    fedmix_lr, client_lr, NUM_CLIENTS_PER_FEDMIX_ROUND, fedmix_batch_size)
fedmix_comp_params = FLIX_computation.FLIXComputationParams(
    FEDMIX_ALGORITHM, CLIENT_ALGORITHM, fedmix_init_params, num_rounds)
_, stats = FLIX_computation.flix_computation_with_statistics(
    train_fd, test_fd, grad_fn, grad_fn_eval, model, PLM_dict, alpha_dict,
    fedmix_hparams, fedmix_comp_params, 100)

client_optimizer = fedjax.optimizers.sgd(learning_rate=1)
server_optimizer = fedjax.optimizers.sgd(learning_rate=1)
# Hyperparameters for client local traing dataset preparation.
client_batch_hparams = fedjax.ShuffleRepeatBatchHParams(batch_size=4)
algorithm = fedjax.algorithms.fed_avg.federated_averaging(grad_fn,
                                                          client_optimizer,
                                                          server_optimizer,
                                                          client_batch_hparams)
# Initialize model parameters and algorithm server state.
init_params = model.init(jax.random.PRNGKey(17))
server_state = algorithm.init(init_params)

train_client_sampler = fedjax.client_samplers.UniformGetClientSampler(fd=train_fd, num_clients=10, seed=0)

fedavg_test_acc_progress = []
max_rounds = 1200
fedjax.set_for_each_client_backend('pmap')

for round_num in range(1, max_rounds + 1):
    # Sample 10 clients per round without replacement for training.
    clients = train_client_sampler.sample()
    # Run one round of training on sampled clients.
    server_state, client_diagnostics = algorithm.apply(server_state, clients)
    print(f'[round {round_num}]', end='\r')
    # Optionally print client diagnostics if curious about each client's model
    # update's l2 norm.
    # print(f'[round {round_num}] client_diagnostics={client_diagnostics}')

    if round_num % 100 == 0:
        test_eval_datasets = [cds for _, cds in test_fd.clients()]
        test_eval_batches = fedjax.padded_batch_client_datasets(test_eval_datasets, batch_size=256)
        test_metrics = fedjax.evaluate_model(model, server_state.params, test_eval_batches)
        fedavg_test_acc_progress.append(test_metrics['accuracy_in_vocab'])
        print('Test accuracy = {}'.format(test_metrics['accuracy_in_vocab']))

accs1 = [stat['accuracy_in_vocab'] for stat in stats]
accs = [accs1, fedavg_test_acc_progress]

saved_log_nm = f'../logs/Shakespeare_debug.txt'
with open(f'{saved_log_nm}', 'w+') as output:
    for acc in accs:
        for element in acc:
            output.write(str(element) + ',')
        output.write('\n')

round_nums = jnp.linspace(100, max_rounds, num=12, endpoint=True)
plt.plot(round_nums[:-18], accs[:-18], label='FedMix, alpha={}'.format(alpha))
plt.plot(round_nums[:-18], fedavg_test_acc_progress[:-18], label='FedAvg')
plt.xlim(left=0)
plt.ylabel('accuracy')
plt.xlabel('rounds')
plt.grid()
plt.title('EMNIST')
plt.legend()
plt.tight_layout()
plt.savefig('../results/plots/Shakespeare_debug_{}.pdf'.format(int(10 * alpha)))