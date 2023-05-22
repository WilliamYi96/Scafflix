import sys, os
sys.path.append("../")
from algs import scaffnew, mime
import fedjax
import jax
import jax.numpy as jnp
import PLM_computation
import FLIX_computation
from grid_search import FLIXGrid, grid_search
from EMNIST_custom import emnist_load_gd_data
import itertools
import matplotlib.pyplot as plt
import pickle
import copy
import argparse, time

# Arguments
parser = argparse.ArgumentParser(description='Combination of ProxSkip and FLIX.')
parser.add_argument('--cache_dir', default='../data/', type=str)
parser.add_argument('--fedavg', action='store_true')
parser.add_argument('--local_epochs', default=0, type=int, help='epochs for fedavg local udpates')
parser.add_argument('--flix', action='store_true')
parser.add_argument('--scaffnew', action='store_true')
parser.add_argument('--scaffnew_flix', action='store_true')
parser.add_argument('--mime', action='store_true')
parser.add_argument('--prob', default=0.1, type=float, help='probability of skipping communication')
parser.add_argument('--alpha', default=0.7, type=float)
parser.add_argument('--debug', action='store_true', help='whether in the debug mode')
parser.add_argument('--total_points', default=30, type=int, help='number of total points to print in the figure')
parser.add_argument('--plm_num_epochs', default=100, type=int)
parser.add_argument('--stat_every', default=100, type=int)
parser.add_argument('--server_alg', default='adam', type=str, help='server algorithm')
parser.add_argument('--n_clients_grid_search', default=3400, type=int)
parser.add_argument('--n_clients_per_plm_round', default=5, type=int)
parser.add_argument('--n_clients_per_flix_round', default=10, type=int)
parser.add_argument('--train_val_split', default=0.8, type=float)
parser.add_argument('--max_rounds', default=1000, type=int, help='FedAvg total round')
parser.add_argument('--flix_num_rounds', default=1000, type=int)
parser.add_argument('--bs', default=4096, type=int)
parser.add_argument('--plm_lrs', default=0.01, type=float)

args = parser.parse_args()
args.bs, plm_lrs = [args.bs], [args.plm_lrs]
print(args)

################### Model setup
model = fedjax.models.emnist.create_conv_model(only_digits=False)


def loss(params, batch, rng):
    # `rng` used with `apply_for_train` to apply dropout during training.
    preds = model.apply_for_train(params, batch, rng)
    # Per example loss of shape [batch_size]
    example_loss = model.train_loss(batch, preds)
    return jnp.mean(example_loss)


def loss_for_eval(params, batch):
    preds = model.apply_for_eval(params, batch)
    example_loss = model.train_loss(batch, preds)
    return jnp.mean(example_loss)


grad_fn = jax.jit(jax.grad(loss))
grad_fn_eval = jax.jit(jax.grad(loss_for_eval))

if args.debug:
    flix_lrs = [0.01]
    client_lrs = [0.01]
    args.stat_every = 1
    args.plm_num_epochs = 3
    args.flix_num_rounds = 5
    args.total_points = 3
    args.n_clients_grid_search = 30
    args.max_rounds = 5
else:
    flix_lrs = 10 ** jnp.arange(-5., 0.5, 1)
    client_lrs = 10 ** jnp.arange(-5., 0.5, 1)
    # flix_lrs = [0.01]
    # client_lrs = [0.01]
    flix_batch_sizes = [20, 50, 100, 200]


################### Grid search setup
if args.flix or args.scaffnew_flix:
    print("================ Now start grid search for Scaffnew and Scaffnew-FLIX =================")
    train_fd, validation_fd = emnist_load_gd_data(train_val_split=args.train_val_split,
                                                  only_digits=False, cache_dir=args.cache_dir)
    client_ids = set([cid for cid in itertools.islice(train_fd.client_ids(), args.n_clients_grid_search)])
    train_fd = fedjax.SubsetFederatedData(train_fd, client_ids)
    validation_fd = fedjax.SubsetFederatedData(validation_fd, client_ids)

    plm_init_params = model.init(jax.random.PRNGKey(200))
    plm_comp_params = PLM_computation.PLMComputationProcessParams(plm_init_params, args.n_clients_per_plm_round)

    flix_init_params = model.init(jax.random.PRNGKey(20))
    flix_comp_params = FLIX_computation.FLIXComputationParams(
        args.server_alg, flix_init_params, args.flix_num_rounds)

    grid = FLIXGrid(flix_lrs, plm_lrs, args.bs, args.bs)

    name_list = f"{args.fedavg}_{args.flix}_{args.mime}_{args.scaffnew}_{args.scaffnew_flix}_" \
                f"{args.prob}_{args.alpha}_{args.stat_every}_{args.n_clients_grid_search}_" \
                f"{args.n_clients_per_plm_round}_{args.n_clients_per_flix_round}_{args.train_val_split}_" \
                f"{args.max_rounds}_{args.flix_num_rounds}_{args.bs}"

    SAVE_FILE = '../results/fedavg_flix_EMNIST_gd_{}.npy'.format(name_list)

    table = grid_search(train_fd, validation_fd, grad_fn, grad_fn_eval, model, args.alpha,
                        plm_comp_params, flix_comp_params, grid, args.plm_num_epochs,
                        args.n_clients_per_flix_round, SAVE_FILE)

    table = jnp.load(SAVE_FILE)
    best_ind = jnp.unravel_index(jnp.argmax(table), table.shape)

    plm_batch_size = args.bs[best_ind[0]]
    plm_lr = plm_lrs[best_ind[1]]
    flix_batch_size = args.bs[best_ind[2]]
    flix_lr = flix_lrs[best_ind[3]]

# Load training and testing data
train_fd, test_fd = fedjax.datasets.emnist.load_data(only_digits=False, cache_dir=args.cache_dir)


# accs = [stat['accuracy'] for stat in stats]
# ######################################### FLIX
def run_flix():
    print("================ Now start running FLIX/FLIX algorithm =================")
    # print(args.plm_num_epochs, plm_lr, plm_batch_size)
    plm_comp_hparams = PLM_computation.PLMComputationHParams(args.plm_num_epochs, plm_lr, plm_batch_size)
    PLM_dict = PLM_computation.plm_computation(train_fd, grad_fn, plm_comp_hparams, plm_comp_params)
    alpha_dict = {}
    for cid in train_fd.client_ids():
        alpha_dict[cid] = args.alpha

    flix_hparams = FLIX_computation.FLIXHParams(flix_lr, args.n_clients_per_flix_round, flix_batch_size)
    flix_comp_params = FLIX_computation.FLIXComputationParams(args.server_alg, flix_init_params,
                                                                    args.flix_num_rounds)
    _, stats = FLIX_computation.flix_computation_with_statistics(train_fd, test_fd, grad_fn, grad_fn_eval,
                                                                     model, PLM_dict, alpha_dict,
                                                                     flix_hparams, flix_comp_params,
                                                                     args.stat_every)
    accs = [stat['accuracy'] for stat in stats]
    return accs


######################################### FedAvg
def run_fedavg(local_epochs=0):
    print("================ Now start running FedAvg algorithm =================")
    client_optimizer = fedjax.optimizers.sgd(learning_rate=10 ** (-1.5))
    server_optimizer = fedjax.optimizers.adam(
        learning_rate=10 ** (-2.5), b1=0.9, b2=0.999, eps=10 ** (-4))
    # Hyperparameters for client local traing dataset preparation.
    if local_epochs:
        client_batch_hparams = fedjax.ShuffleRepeatBatchHParams(batch_size=20, local_epochs=local_epochs)
    else:
        client_batch_hparams = fedjax.ShuffleRepeatBatchHParams(batch_size=20)

    algorithm = fedjax.algorithms.fed_avg.federated_averaging(grad_fn, client_optimizer,
                                                              server_optimizer,
                                                              client_batch_hparams)
    # Initialize model parameters and algorithm server state.
    init_params = model.init(jax.random.PRNGKey(17))
    server_state = algorithm.init(init_params)

    train_client_sampler = fedjax.client_samplers.UniformGetClientSampler(fd=train_fd, num_clients=10, seed=0)
    fedavg_test_acc_progress = []
    for round_num in range(1, args.max_rounds + 1):
        # Sample 10 clients per round without replacement for training.
        clients = train_client_sampler.sample()
        # Run one round of training on sampled clients.
        server_state, client_diagnostics = algorithm.apply(server_state, clients)
        print(f'[round {round_num}]', end='\r')
        # Optionally print client diagnostics if curious about each client's model
        # update's l2 norm.
        # print(f'[round {round_num}] client_diagnostics={client_diagnostics}')

        if round_num % args.stat_every == 0:
            test_eval_datasets = [cds for _, cds in test_fd.clients()]
            test_eval_batches = fedjax.padded_batch_client_datasets(test_eval_datasets, batch_size=256)
            test_metrics = fedjax.evaluate_model(model, server_state.params, test_eval_batches)
            fedavg_test_acc_progress.append(test_metrics['accuracy'])
            print('Test accuracy = {}'.format(test_metrics['accuracy']))

    save_file = f'../results/test_acc_fedavg_{name_list}.pickle'

    with open(save_file, 'wb') as handle:
        pickle.dump(fedavg_test_acc_progress, handle)

    with open(save_file, 'rb') as handle:
        fedavg_test_acc_progress = pickle.load(handle)

    fedavg_test_acc_progress = fedavg_test_acc_progress[:args.total_points]

    return fedavg_test_acc_progress


######################################### Scaffnew
def run_scaffnew(alpha, prob):
    print("================ Now start running ProxSkip/Scaffnew algorithm =================")
    num_clients = train_fd.num_clients()
    client_optimizer = fedjax.optimizers.sgd(learning_rate=10 ** (-1.5))
    server_optimizer = fedjax.optimizers.adam(learning_rate=10 ** (-2.5), b1=0.9, b2=0.999, eps=10 ** (-4))
    # Hyperparameters for client local training dataset preparation.
    client_batch_hparams = fedjax.ShuffleRepeatBatchHParams(batch_size=2048)
    grads_batch_hparams = fedjax.PaddedBatchHParams(batch_size=2048)

    algorithm = scaffnew.scaffnew(grad_fn, client_optimizer, server_optimizer,
                                  client_batch_hparams, grads_batch_hparams,
                                  learning_rate=client_lrs, prob=args.prob, num_clients=num_clients)

    # Initialize model parameters and algorithm server state
    init_params = model.init(jax.random.PRNGKey(17))
    server_state = algorithm.init(init_params)

    train_client_sampler = fedjax.client_samplers.UniformGetClientSampler(fd=train_fd, num_clients=10, seed=0)

    scaffnew_test_acc_progress = []
    for round_num in range(1, args.max_rounds + 1):
        # Sample 10 clients per round without replacement for training
        clients = train_client_sampler.sample()
        # Run one round of training on sampled clients.
        server_state, client_diagnostics = algorithm.apply(server_state, clients)
        # print(f'[round {round_num}]', end='\r')
        # Optionally print client diagnostics if curious about each client's model update's l2 norm
        print(f'[round {round_num}] client_diagnostics={client_diagnostics}')

        if round_num % args.stat_every == 0:
            test_eval_datasets = [cds for _, cds in test_fd.clients()]
            test_eval_batches = fedjax.padded_batch_client_datasets(test_eval_datasets, batch_size=4096)
            test_metrics = fedjax.evaluate_model(model, server_state.params, test_eval_batches)
            scaffnew_test_acc_progress.append(test_metrics['accuracy'])
            print('Test accuracy = {}'.format(test_metrics['accuracy']))

    save_file = f'../results/test_acc_scaffnew_{name_list}.pickle'
    with open(save_file, 'wb') as handle:
        pickle.dump(scaffnew_test_acc_progress, handle)
    with open(save_file, 'rb') as handle:
        scaffnew_test_acc_progress = pickle.load(handle)
    scaffnew_test_acc_progress = scaffnew_test_acc_progress[:args.total_points]

    return scaffnew_test_acc_progress


######################################### Mime
def run_mime():
    print("================ Now start running MIME algorithm =================")
    client_optimizer = fedjax.optimizers.sgd(learning_rate=10 ** (-1.5))
    server_optimizer = fedjax.optimizers.adam(learning_rate=10 ** (-2.5), b1=0.9, b2=0.999, eps=10 ** (-4))
    # Hyperparameters for client local training dataset preparation.
    client_batch_hparams = fedjax.ShuffleRepeatBatchHParams(batch_size=2048)
    grads_batch_hparams = fedjax.PaddedBatchHParams(batch_size=2048)

    server_learning_rate = args.plm_lrs  # TODO: change this learning rate later
    algorithm = mime.mime(grad_fn, client_optimizer, client_batch_hparams,
                          grads_batch_hparams, server_learning_rate)
    # Here we do not have any regularizer so we convery grad_fn isntead of per_example_loss
    # algorithm = fedjax.algorithms.mime.mime(grad_fn, client_optimizer, client_batch_hparams,
    #                                         grads_batch_hparams, server_learning_rate)

    # Initialize model parameters and algorithm server state
    init_params = model.init(jax.random.PRNGKey(17))
    server_state = algorithm.init(init_params)

    train_client_sampler = fedjax.client_samplers.UniformGetClientSampler(fd=train_fd, num_clients=10, seed=0)

    scaffnew_test_acc_progress = []
    for round_num in range(1, args.max_rounds + 1):
        # Sample 10 clients per round without replacement for training
        clients = train_client_sampler.sample()
        # Run one round of training on sampled clients.
        server_state, client_diagnostics = algorithm.apply(server_state, clients)
        # print(f'[round {round_num}]', end='\r')
        # Optionally print client diagnostics if curious about each client's model update's l2 norm
        print(f'[round {round_num}] client_diagnostics={client_diagnostics}')

        if round_num % args.stat_every == 0:
            test_eval_datasets = [cds for _, cds in test_fd.clients()]
            test_eval_batches = fedjax.padded_batch_client_datasets(test_eval_datasets, batch_size=4096)
            test_metrics = fedjax.evaluate_model(model, server_state.params, test_eval_batches)
            scaffnew_test_acc_progress.append(test_metrics['accuracy'])
            print('Test accuracy = {}'.format(test_metrics['accuracy']))

    save_file = f'../results/test_acc_mime_{name_list}.pickle'
    with open(save_file, 'wb') as handle:
        pickle.dump(scaffnew_test_acc_progress, handle)
    with open(save_file, 'rb') as handle:
        scaffnew_test_acc_progress = pickle.load(handle)
    scaffnew_test_acc_progress = scaffnew_test_acc_progress[:args.total_points]

    return scaffnew_test_acc_progress


def run_scaffnew_flix(alpha, prob):
    # TODO
    return [0]


# if args.flix:
#     accs = run_flix()
#     round_nums = jnp.linspace(100, 3000, num=len(accs), endpoint=True)
#     plt.plot(round_nums, accs, label='FLIX')
# if args.fedavg:
#     if args.local_epochs:
#         accs = run_fedavg(args.local_epochs)
#     else:
#         accs = run_fedavg()
#     round_nums = jnp.linspace(100, 3000, num=len(accs), endpoint=True)
#     plt.plot(round_nums, accs, label='FedAvg')
# if args.scaffnew:
#     accs = run_scaffnew(args.alpha, args.prob)
#     round_nums = jnp.linspace(100, 3000, num=len(accs), endpoint=True)
#     plt.plot(round_nums, accs, label='Scaffnew')
# if args.mime:
#     accs = run_mime()
#     round_nums = jnp.linspace(100, 3000, num=len(accs), endpoint=True)
#     plt.plot(round_nums, accs, label='MIME')
# if args.scaffnew_flix:
#     accs = run_scaffnew_flix(args.alpha, args.prob)
#     round_nums = jnp.linspace(100, 3000, num=len(accs), endpoint=True)
#     plt.plot(round_nums, accs, label='Scaffnew-FLIX')
# if args.mime:
#     accs = run_mime()
#     plt.plot(round_nums, accs, label='Scaffnew-FLIX')

accs1 = run_flix()
accs2 = run_fedavg()
accs3 = run_mime()
mlen = min(len(accs1), len(accs2), len(accs3))
round_nums = jnp.linspace(int(args.max_rounds / 100), args.max_rounds, num=mlen, endpoint=True)
plt.plot(round_nums, accs1[:mlen], label='FLIX')
plt.plot(round_nums, accs2[:mlen], label='FedAvg')
plt.plot(round_nums, accs3[:mlen], label='MIME')

plt.xlim(left=0)
plt.ylabel('accuracy')
plt.xlabel('rounds')
plt.grid()
plt.title('FEMNIST')
plt.legend()
plt.tight_layout()
plt.savefig(f'../results/plots/FEMNIST_{name_list}.pdf')
