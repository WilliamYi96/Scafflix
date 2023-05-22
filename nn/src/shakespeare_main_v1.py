import sys, os
sys.path.append("../")
from algs import scaffnew, mime, scafflix
import fedjax
import jax
import jax.numpy as jnp
import PLM_computation
import FLIX_computation
import Scafflix_computation
from grid_search_general import FLIXGrid, grid_search
from scafflix_grid_search_general import ScafflixGrid, scafflix_grid_search
from Shakespeare_custom import shakespeare_load_gd_data
import itertools
import matplotlib.pyplot as plt
import pickle 
import copy
import argparse, time

# Arguments
parser = argparse.ArgumentParser(description='Combination of ProxSkip and FLIX.')
parser.add_argument('--cache_dir', default='../data/', type=str)
parser.add_argument('--fedavg', action='store_true', help='FedAvg algorithm')
parser.add_argument('--local_epochs', default=0, type=int, help='epochs for fedavg local udpates')
parser.add_argument('--flix', action='store_true', help='FLIX algorithm')
parser.add_argument('--scaffnew', action='store_true', help='Scaffnew algorithm')
parser.add_argument('--scafflix', action='store_true', help='Scafflix algorithm')
parser.add_argument('--mime', action='store_true', help='MIME algorithm')
parser.add_argument('--prob', default=0.2, type=float, help='probability of skipping communication')
parser.add_argument('--alpha', default=0.7, type=float)
parser.add_argument('--debug', action='store_true', help='whether in the debug mode')
parser.add_argument('--total_points', default=100, type=int, help='number of total points to print in the figure')
parser.add_argument('--plm_num_epochs', default=1000, type=int)
parser.add_argument('--plm_lrs', default=0.005, type=float)
parser.add_argument('--n_clients_per_plm_round', default=5, type=int)
parser.add_argument('--stat_every', default=100, type=int)
parser.add_argument('--server_alg_type', default='sgd', type=str, help='server algorithm type')
parser.add_argument('--client_alg_type', default='sgd', type=str, help='client algorithm type')
parser.add_argument('--bs', default=4096, type=int)

parser.add_argument('--client_lr', default=10 ** (-1.5), type=float)
parser.add_argument('--server_lr', default=10 ** (-2.5), type=float)
# Grid search parameters
parser.add_argument('--n_clients_grid_search', default=715, type=int)
parser.add_argument('--train_val_split', default=0.8, type=float)
# FLIX parameters
parser.add_argument('--n_clients_per_flix_round', default=10, type=int)
parser.add_argument('--flix_num_rounds', default=1000, type=int)
# FedAvg parameters
parser.add_argument('--max_rounds', default=1000, type=int, help='FedAvg total round')
# Scafflix parameters
parser.add_argument('--n_clients_per_scafflix_round', type=int, default=10)
parser.add_argument('--scafflix_num_rounds', default=5000, type=int)

parser.add_argument('--exp_no', type=str, default='0000')

args = parser.parse_args()
args.bs, plm_lrs = [args.bs], [args.plm_lrs]
args.scafflix_num_rounds = args.flix_num_rounds * int(1 / args.prob)
print(args)

# ################## Model setup
model = fedjax.models.shakespeare.create_lstm_model()
client_optimizer = fedjax.optimizers.sgd(learning_rate=args.client_lr)
server_optimizer = fedjax.optimizers.adam(learning_rate=args.server_lr, b1=0.9, b2=0.999, eps=10 ** (-4))


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

# Choose from debug/training mode
if args.debug:
    # server_lrs = [1.0]
    server_lrs = [10 ** (-2.5)]
    client_lrs = [0.01]
    args.stat_every = 1
    args.plm_num_epochs = 2
    args.flix_num_rounds = 5
    args.total_points = 5
    args.n_clients_grid_search = 30
    args.max_rounds = 10
    args.scafflix_num_rounds = 50
else:
    # server_lrs = 10 ** jnp.arange(-5., 0.5, 1)
    # server_lrs = [1.]
    client_lrs = 10 ** jnp.arange(-5., 0.5, 1)
    server_lrs = 10 ** jnp.arange(-5., 0.5, 1)
    # flix_batch_sizes = [20, 50, 100, 200]


# Load training and testing data
train_fd, test_fd = fedjax.datasets.shakespeare.load_data(cache_dir=args.cache_dir)

name_list = f"{args.fedavg}_{args.flix}_{args.mime}_{args.scaffnew}_{args.scafflix}_" \
            f"{args.prob}_{args.alpha}_{args.stat_every}_{args.n_clients_grid_search}_" \
            f"{args.n_clients_per_plm_round}_{args.n_clients_per_flix_round}_{args.train_val_split}_" \
            f"{args.max_rounds}_{args.flix_num_rounds}_{args.bs}_{args.exp_no}"


# ######################################### FLIX
def run_flix():
    if args.debug:
        plm_batch_size = args.bs[0]
        plm_lr = plm_lrs[0]
        flix_batch_size = args.bs[0]
        server_lr = 10 ** (-2.5)
        client_lr = client_lrs[0]
    else:
        ################### Grid search setup
        print("================ Now start grid search for FLIX =================")
        train_fd1, validation_fd1 = shakespeare_load_gd_data(train_val_split=args.train_val_split, cache_dir=args.cache_dir)
        client_ids = set([cid for cid in itertools.islice(train_fd1.client_ids(), args.n_clients_grid_search)])
        train_fd1 = fedjax.SubsetFederatedData(train_fd1, client_ids)
        validation_fd1 = fedjax.SubsetFederatedData(validation_fd1, client_ids)

        plm_init_params = model.init(jax.random.PRNGKey(200))
        plm_comp_params = PLM_computation.PLMComputationProcessParams(plm_init_params, args.n_clients_per_plm_round)

        flix_init_params = model.init(jax.random.PRNGKey(20))
        flix_comp_params = FLIX_computation.FLIXComputationParams(args.server_alg_type, args.client_alg_type,
                                                                  flix_init_params, args.flix_num_rounds)

        # server_lrs = [args.server_lr]
        grid = FLIXGrid(server_lrs, plm_lrs, client_lrs, args.bs, args.bs)

        SAVE_FILE = '../results/fedavg_flix_EMNIST_gd_{}.npy'.format(name_list)

        table = grid_search(train_fd1, validation_fd1, grad_fn, grad_fn_eval, model, args.alpha,
                            plm_comp_params, flix_comp_params, grid, args.plm_num_epochs,
                            args.n_clients_per_flix_round, SAVE_FILE, grid_metrics='accuracy_in_vocab')

        table = jnp.load(SAVE_FILE)
        best_ind = jnp.unravel_index(jnp.argmax(table), table.shape)

        # Obtaining the best hyper-parameters
        plm_batch_size = args.bs[best_ind[0]]
        plm_lr = plm_lrs[best_ind[1]]
        flix_batch_size = args.bs[best_ind[2]]
        server_lr = server_lrs[best_ind[3]]
        client_lr = client_lrs[best_ind[4]]

    print("================ Now start running FLIX algorithm with optimal parameters=================")
    # print(args.plm_num_epochs, plm_lr, plm_batch_size)
    print(plm_batch_size, plm_lr, flix_batch_size, server_lr, client_lr)
    plm_comp_hparams = PLM_computation.PLMComputationHParams(args.plm_num_epochs, plm_lr, plm_batch_size)
    PLM_dict = PLM_computation.plm_computation(train_fd, grad_fn, plm_comp_hparams, plm_comp_params)

    alpha_dict = {}
    for cid in train_fd.client_ids():
        alpha_dict[cid] = args.alpha

    flix_hparams = FLIX_computation.FLIXHParams(server_lr, client_lr, args.n_clients_per_flix_round, flix_batch_size)
    flix_comp_params = FLIX_computation.FLIXComputationParams(args.server_alg_type, args.client_alg_type,
                                                              flix_init_params, args.flix_num_rounds)
    _, stats = FLIX_computation.flix_computation_with_statistics(train_fd, test_fd, grad_fn, grad_fn_eval,
                                                                     model, PLM_dict, alpha_dict,
                                                                     flix_hparams, flix_comp_params,
                                                                     args.stat_every)
    accs = [stat['accuracy_in_vocab'] for stat in stats]
    return accs


######################################### FedAvg
def run_fedavg(local_epochs=0):
    print("================ Now start running FedAvg algorithm =================")
    # server_optimizer = fedjax.optimizers.adam(
    #     learning_rate=1, b1=0.9, b2=0.999, eps=10 ** (-4))
    # Hyperparameters for client local traing dataset preparation.
    if local_epochs:
        client_batch_hparams = fedjax.ShuffleRepeatBatchHParams(batch_size=20, local_epochs=local_epochs)
    else:
        client_batch_hparams = fedjax.ShuffleRepeatBatchHParams(batch_size=20)

    import sys
    sys.path.append("../algs")
    import fed_avg
    algorithm = fed_avg.federated_averaging(grad_fn, client_optimizer,
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
            fedavg_test_acc_progress.append(test_metrics['accuracy_in_vocab'])
            print('Test accuracy = {}'.format(test_metrics['accuracy_in_vocab']))

    save_file = f'../results/test_acc_fedavg_{name_list}.pickle'

    with open(save_file, 'wb') as handle:
        pickle.dump(fedavg_test_acc_progress, handle)

    with open(save_file, 'rb') as handle:
        fedavg_test_acc_progress = pickle.load(handle)

    fedavg_test_acc_progress = fedavg_test_acc_progress[:args.total_points]

    return fedavg_test_acc_progress


######################################### Scaffnew
def run_scaffnew(local_epochs=0):
    print("================ Now start running Scaffnew algorithm =================")
    # server_optimizer = fedjax.optimizers.adam(learning_rate=1, b1=0.9, b2=0.999, eps=10 ** (-4))
    # Hyperparameters for client local traing dataset preparation.
    if local_epochs:
        client_batch_hparams = fedjax.ShuffleRepeatBatchHParams(batch_size=20, local_epochs=local_epochs)
    else:
        client_batch_hparams = fedjax.ShuffleRepeatBatchHParams(batch_size=20)

    lr = client_lrs[0]  # TODO: Should be fine-tuned later
    p_choices = []
    p = args.prob
    for i in range(args.max_rounds * int(1/p)):
        prng = jax.random.PRNGKey(i)
        p_choice = jax.random.choice(prng, jax.numpy.array([0, 1]), p=jax.numpy.array([1-p, p]))
        p_choices.append(p_choice)

    algorithm = scaffnew.scaffnew(grad_fn, client_optimizer, server_optimizer, client_batch_hparams, lr, p, p_choices)

    # Initialize model parameters and algorithm server state.
    init_params = model.init(jax.random.PRNGKey(17))
    server_state = algorithm.init(init_params)

    train_client_sampler = fedjax.client_samplers.UniformGetClientSampler(fd=train_fd, num_clients=10, seed=0)
    fedavg_test_acc_progress = []
    for round_num in range(1, args.max_rounds * int(1/p) + 1):
        # Sample 10 clients per round without replacement for training.
        clients = train_client_sampler.sample()
        # Run one round of training on sampled clients.
        # server_state, client_diagnostics = algorithm.apply(server_state, clients, p_choices[round_num-1],
        #                                                    p, client_lrs[0])
        server_state, client_diagnostics = algorithm.apply(server_state, clients)
        print(f'[round {round_num}]', end='\r')
        # Optionally print client diagnostics if curious about each client's model
        # update's l2 norm.
        # print(f'[round {round_num}] client_diagnostics={client_diagnostics}')

        if round_num % args.stat_every == 0:
            test_eval_datasets = [cds for _, cds in test_fd.clients()]
            test_eval_batches = fedjax.padded_batch_client_datasets(test_eval_datasets, batch_size=256)
            test_metrics = fedjax.evaluate_model(model, server_state.params, test_eval_batches)
            fedavg_test_acc_progress.append(test_metrics['accuracy_in_vocab'])
            print('Test accuracy = {}'.format(test_metrics['accuracy_in_vocab']))

    save_file = f'../results/test_acc_fedavg_{name_list}.pickle'

    with open(save_file, 'wb') as handle:
        pickle.dump(fedavg_test_acc_progress, handle)

    with open(save_file, 'rb') as handle:
        fedavg_test_acc_progress = pickle.load(handle)

    fedavg_test_acc_progress = fedavg_test_acc_progress[:args.total_points]

    return fedavg_test_acc_progress


######################################### Mime
def run_mime():
    print("================ Now start running MIME algorithm =================")
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
            scaffnew_test_acc_progress.append(test_metrics['accuracy_in_vocab'])
            print('Test accuracy = {}'.format(test_metrics['accuracy_in_vocab']))

    save_file = f'../results/test_acc_mime_{name_list}.pickle'
    with open(save_file, 'wb') as handle:
        pickle.dump(scaffnew_test_acc_progress, handle)
    with open(save_file, 'rb') as handle:
        scaffnew_test_acc_progress = pickle.load(handle)
    scaffnew_test_acc_progress = scaffnew_test_acc_progress[:args.total_points]

    return scaffnew_test_acc_progress


######################################### Scafflix
def run_scafflix(local_epochs=0):
    #### Grid search
    print("================ Now start grid search for Scafflix =================")
    train_fd1, validation_fd1 = shakespeare_load_gd_data(train_val_split=args.train_val_split, cache_dir=args.cache_dir)
    client_ids = set([cid for cid in itertools.islice(train_fd1.client_ids(), args.n_clients_grid_search)])
    train_fd1 = fedjax.SubsetFederatedData(train_fd1, client_ids)
    validation_fd1 = fedjax.SubsetFederatedData(validation_fd1, client_ids)

    # p_choices = []
    p = args.prob
    # for i in range(args.max_rounds):
    #     prng = jax.random.PRNGKey(i)
    #     p_choice = jax.random.choice(prng, jax.numpy.array([0, 1]), p=jax.numpy.array([1-p, p]))
    #     p_choices.append(p_choice)

    plm_init_params = model.init(jax.random.PRNGKey(201))
    plm_comp_params = PLM_computation.PLMComputationProcessParams(plm_init_params, args.n_clients_per_plm_round)

    scafflix_init_params = model.init(jax.random.PRNGKey(21))
    scafflix_comp_params = Scafflix_computation.ScafflixComputationParams(args.server_alg_type, args.client_alg_type,
                                                              scafflix_init_params, args.flix_num_rounds)

    # server_lrs = [args.server_lr]
    grid = ScafflixGrid(server_lrs, plm_lrs, client_lrs, args.bs, args.bs)

    SAVE_FILE = '../results/fedavg_flix_EMNIST_gd_{}.npy'.format(name_list)

    table = scafflix_grid_search(train_fd1, validation_fd1, grad_fn, grad_fn_eval, model, args.alpha, p,
                        plm_comp_params, scafflix_comp_params, grid, args.plm_num_epochs,
                        args.n_clients_per_flix_round, SAVE_FILE, grid_metrics='accuracy_in_vocab')

    table = jnp.load(SAVE_FILE)
    best_ind = jnp.unravel_index(jnp.argmax(table), table.shape)

    # Obtaining the best hyper-parameters
    # Using flexible batch sizes here - TODO
    plm_batch_size = args.bs[best_ind[0]]
    plm_lr = plm_lrs[best_ind[1]]
    scafflix_batch_size = args.bs[best_ind[2]]
    server_lr = server_lrs[best_ind[3]]
    client_lr = client_lrs[best_ind[4]]

    print("================ Now start running Scafflix algorithm =================")
    print(plm_batch_size, plm_lr, scafflix_batch_size, server_lr, client_lr)
    # server_optimizer = fedjax.optimizers.adam(learning_rate=1, b1=0.9, b2=0.999, eps=10 ** (-4))
    # Hyperparameters for client local traing dataset preparation.
    # if local_epochs:
    #     client_batch_hparams = fedjax.ShuffleRepeatBatchHParams(batch_size=20, local_epochs=local_epochs)
    # else:
    #     client_batch_hparams = fedjax.ShuffleRepeatBatchHParams(batch_size=20)

    plm_comp_hparams = PLM_computation.PLMComputationHParams(args.plm_num_epochs, plm_lr, plm_batch_size)
    PLM_dict = PLM_computation.plm_computation(train_fd, grad_fn, plm_comp_hparams, plm_comp_params)

    alpha_dict = {}
    p_rate_dict = {}
    p_dict = {}
    for cid in train_fd.client_ids():
        alpha_dict[cid] = args.alpha
        p_rate_dict[cid] = p / server_lr
        p_dict[cid] = p

    scafflix_hparams = Scafflix_computation.ScafflixHParams(server_lr, client_lr,
                                                            args.n_clients_per_scafflix_round, scafflix_batch_size)
    scafflix_comp_params = Scafflix_computation.ScafflixComputationParams(args.server_alg_type, args.client_alg_type,
                                                              scafflix_init_params, args.scafflix_num_rounds)
    _, stats = Scafflix_computation.scafflix_computation_with_statistics(train_fd, test_fd, grad_fn, grad_fn_eval,
                                                                     model, PLM_dict, alpha_dict, p_dict, p_rate_dict,
                                                                     scafflix_hparams, scafflix_comp_params,
                                                                     args.stat_every)

    accs = [stat['accuracy_in_vocab'] for stat in stats]
    return accs


def select_run():
    mlen = 10
    round_nums = jnp.linspace(int(args.max_rounds / 100), args.max_rounds, num=mlen, endpoint=True)
    if args.flix:
        accs = run_flix()
        print(f'FLIX accs: {accs}')
        round_nums = jnp.linspace(100, 3000, num=len(accs), endpoint=True)
        plt.plot(round_nums[:mlen], accs[:mlen], label='FLIX')
    if args.fedavg:
        if args.local_epochs:
            accs = run_fedavg(args.local_epochs)
        else:
            accs = run_fedavg()
        print(f'FedAvg accs: {accs}')
        round_nums = jnp.linspace(100, 3000, num=len(accs), endpoint=True)
        plt.plot(round_nums[:mlen], accs[:mlen], label='FedAvg')
    if args.scaffnew:
        accs = run_scaffnew()
        interval = int(1/args.prob)
        comm_accs = [accs[interval*i] for i in range(int(len(accs)/interval))]
        print(f'Scaffnew accs: {accs}')
        round_nums = jnp.linspace(100, 3000, num=len(accs), endpoint=True)
        plt.plot(round_nums[:mlen], comm_accs[:mlen], label='Scaffnew')
    # if args.mime:
    #     accs = run_mime()
    #     print(f'MIME accs: {accs}')
    #     round_nums = jnp.linspace(100, 3000, num=len(accs), endpoint=True)
    #     plt.plot(round_nums, accs[:mlen], label='MIME')
    if args.scafflix:
        accs = run_scafflix()
        print(f'Scafflix accs: {accs}')
        round_nums = jnp.linspace(100, 3000, num=len(accs), endpoint=True)
        plt.plot(round_nums[:mlen], accs[:mlen], label='Scafflix')
    if args.mime:
        accs = run_mime()
        plt.plot(round_nums[:mlen], accs[:mlen], label='MIME')


if args.debug:
    select_run()
else:
    accs1 = run_flix()
    accs2 = run_fedavg()
    accs3 = run_scafflix()
    accs = [accs1, accs2, accs3]
    interval = int(1 / args.prob)
    comm_accs = [accs3[interval * i] for i in range(int(len(accs3) / interval))]
    mlen = min(len(accs1), len(accs2), len(accs3))
    round_nums = jnp.linspace(int(args.max_rounds / mlen), args.max_rounds, num=mlen, endpoint=True)
    plt.plot(round_nums[:mlen], accs1[:mlen], label='FLIX')
    plt.plot(round_nums[:mlen], accs2[:mlen], label='FedAvg')
    plt.plot(round_nums[:mlen], comm_accs[:mlen], label='Scafflix')

    saved_log_nm = f'../logs/{args.exp_no}.txt'
    with open(f'{saved_log_nm}', 'w+') as output:
        for acc in accs:
            for element in acc:
                output.write(str(element) + ',')
            output.write('\n')


plt.xlim(left=0)
plt.ylabel('accuracy')
plt.xlabel('communication rounds')
plt.grid()
plt.title('Shakespeare')
plt.legend()
plt.tight_layout()
plt.savefig(f'../results/plots/Shakespeare_{name_list}.pdf')
