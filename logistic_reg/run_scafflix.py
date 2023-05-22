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
p = 0.1
exp = f'scafflix'
max_it = 150

alg = LogReg
logreg = True
ps = np.zeros(shape=(len(alphas)))

i = 1
for alpha in alphas:
    print('------------------- alpha = {} --------------------'.format(alpha))
    alpha = float(alpha)
    model = MasterNode(n_workers, alpha, alg, dataset_name, logreg, True, max_it, regularization=0.1)
    print('Running GD...')
    w, ps[np.where(alphas == alpha)] = model.run_scafflix(max_it, p=p, exp_name=exp)
    i += 1

fig, axs = plt.subplots(2, figsize=(7, 10), constrained_layout=True)
n_iter_shown = 150

alphas_shown = alphas
ind = 0

for alpha in alphas_shown:
    alpha = float(alpha)
    exp = f'{exp}_{ps[np.where(alphas == alpha)]}'
    run = read_run(exp, [alpha] * n_workers, dataset_name, logreg)

    fvals = run['fval'][:n_iter_shown]
    dists = run['dist'][:n_iter_shown]
    gnorms = run['grad'][:n_iter_shown]

    markevery = int(fvals.size / 10)

    axs[0].plot(fvals, marker=markers[ind], markevery=(markevery + 2 * ind, markevery), markersize=10)
    # axs[1].plot(dists, marker=markers[ind], markevery=(markevery + 2 * ind, markevery), markersize=10)
    axs[1].plot(gnorms, marker=markers[ind], markevery=(markevery + 2 * ind, markevery), markersize=10)
    ind += 1

axs[0].legend(
    [r'$\alpha$ = {}'.format(np.format_float_scientific(alpha, trim='-', exp_digits=1)) for alpha in alphas_shown])
axs[0].set_yscale('log')
axs[1].set_yscale('log')
axs[1].set_xlabel('Communication rounds')
# axs[0].set_ylabel('Squared distance')
axs[0].set_ylabel(r'$\|f(x)-f^{\star}\|^2$')
# axs[1].set_ylabel('Loss')
axs[1].set_ylabel(r'$\|\nabla f(x)- \nabla f(x^{\star})\|^2$')
axs[0].set_title(dataset_name)
alg = get_alg(logreg)
name = exp + '_' + alg + '_' + dataset_name
create_plot_dir()
plt.savefig(PLOT_PATH + '/' + name + '_ours1_' + '.pdf')
plt.show()