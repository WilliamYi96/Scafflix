from workers import MasterNode
from models import LinReg, LogReg, LogRegNoncvx, NN_1d_regression
from utils import read_run, get_alg, create_plot_dir, PLOT_PATH
from sklearn.datasets import dump_svmlight_file

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from prep_data import number_of_features
import math
import torch

from numpy.random import default_rng
from numpy import linalg as la
from prep_data import DATASET_PATH
import copy

plt.style.use('fast')
mpl.rcParams['mathtext.fontset'] = 'cm'
# mpl.rcParams['mathtext.fontset'] = 'dejavusans'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['axes.titlesize'] = 'xx-large'
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['axes.labelsize'] = 'xx-large'

markers = ['x', '.', '+', '1', 'p','*', 'D' , '.',  's']

# Gradient descent
alphas = np.array([1.0, 1e-1, 1e-2, 1e-3, 1e-4])
dataset_name = 'mushrooms'  # ijcnn1.bz2, a6a, w6a
n_workers = 8
exp = 'gd'
max_it = 50

alg = LogReg
logreg = True

for alpha in alphas:
    print('------------------- alpha = {} --------------------'.format(alpha))
    alpha = float(alpha)
    model = MasterNode(n_workers, alpha, alg, dataset_name, logreg, True, max_it, regularization=0.1)
    print('Running GD...')
    model.run_gd(max_it)

plt.figure(figsize=(7, 5), constrained_layout=True)
n_iter_shown = 50

alphas_shown = alphas
ind = 0

for alpha in alphas_shown:
    alpha = float(alpha)
    run = read_run(exp, [alpha] * n_workers, dataset_name, logreg)

    # f_values = run['fval'][:n_iter_shown]
    fvals = run['fval'][:n_iter_shown]
    dists = run['dist'][:n_iter_shown]
    # gnorms = run['grad'][:n_iter_shown]

    markevery = int(fvals.size / 10)

    plt.plot(dists, marker=markers[ind], markevery=(markevery + 2 * ind, markevery), markersize=10)
    # axs[1].plot(dists, marker=markers[ind], markevery=(markevery + 2 * ind, markevery), markersize=10)
    # axs[1].plot(gnorms, marker=markers[ind], markevery=(markevery + 2 * ind, markevery), markersize=10)
    ind += 1

plt.legend(
    [r'$\alpha$ = {}'.format(np.format_float_scientific(alpha, trim='-', exp_digits=1)) for alpha in alphas_shown])
plt.yscale('log')
# axs[0].set_ylabel('Squared distance')
plt.ylabel(r'$\|x-x^{\star}\|^2$')
plt.xlabel('Communication rounds')
# axs[1].set_ylabel('Loss')
plt.title(dataset_name)
alg = get_alg(logreg)
name = exp + '_' + alg + '_' + dataset_name
create_plot_dir()
plt.savefig(PLOT_PATH + '/' + name + '_x_' + '.pdf')
plt.show()