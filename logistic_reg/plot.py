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

# color_ar_1 = ['coral', 'violet', 'brown', 'darkorange', 'cornflowerblue', 'darkgreen', 'coral', 'lime', 'darkgreen', 'goldenrod', 'maroon', 'black', 'brown', 'yellowgreen', "purple", "violet", "magenta", "green"]
# color_ar_1 = ['coral', 'purple', 'violet', 'brown', 'cornflowerblue', 'darkgreen', 'goldenrod', 'black', 'yellowgreen', "maroon", "magenta", "green"]
color_ar_1 = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']

markers = ['x', '.', '+', '1', 'p','*', 'D' , '.',  's']

p = 0.1
exp = f'compare'
n_workers = 8
alg = LogReg
logreg = True

# Gradient descent
alphas = np.array([1.0, 1e-1, 1e-2, 1e-3, 1e-4])
idxs = [0, 1, 2, 3]
dataset_names = ['mushrooms', 'ijcnn1.bz2', 'a6a', 'w6a']
max_its = np.array([100, 100, 100, 100])
n_iter_showns = np.array([30, 15, 21, 25])
n_iter_showns2 = np.array([45, 30, 35, 25])
ps = [0.16883865479367444, 0.5499520082585111, 0.2344113416108792, 0.3357686362513109]

# for alpha in alphas:
#     print('------------------- alpha = {} --------------------'.format(alpha))
#     alpha = float(alpha)
#     model = MasterNode(n_workers, alpha, alg, dataset_name, logreg, True, max_it, regularization=0.1)
#     print(f'Running {exp}...')
#     # model.run_gd(max_it, exp_name=exp)  # name should be 'gd' here
#     w, ps = model.run_scafflix(max_it, p=p, exp_name=exp, optim_p=True)
#     break

# fig, axs = plt.subplots(4, figsize=(7, 20), constrained_layout=True)

alphas_shown = alphas
ind = 0

for idx in idxs:
    fig, axs = plt.subplots(2, figsize=(7, 10), constrained_layout=True)
    dataset_name = dataset_names[idx]
    max_it = max_its[idx]
    n_iter_shown = n_iter_showns[idx]
    n_iter_shown2 = n_iter_showns2[idx]
    k = 0
    for alpha in alphas_shown:
        alpha = float(alpha)
        exp1 = 'flix_gd'
        exp2 = f'scafflix_{ps[idx]}'

        run1 = read_run(exp1, [alpha] * n_workers, dataset_name, logreg)
        run2 = read_run(exp2, [alpha] * n_workers, dataset_name, logreg)

        fvals1 = run1['fval'][:n_iter_shown]
        gnorms1 = run1['grad'][:n_iter_shown2]
        dists1 = run1['dist'][:n_iter_shown]
        dists21 = run1['dist2'][:n_iter_shown]
        fvals2 = run2['fval'][:n_iter_shown]
        gnorms2 = run2['grad'][:n_iter_shown2]
        dists2 = run2['dist'][:n_iter_shown]
        dists22 = run2['dist2'][:n_iter_shown]

        markevery = int(fvals1.size / 5)

        axs[0].plot(fvals1, '--', marker=markers[ind], markevery=(markevery + 2 * ind, markevery), markersize=10, color=color_ar_1[k])
        axs[0].plot(fvals2, marker=markers[ind], markevery=(markevery + 2 * ind, markevery), markersize=10, color=color_ar_1[k],
                    label=r'$\alpha$ = {}'.format(np.format_float_scientific(alpha, trim='-', exp_digits=1)))
        axs[1].plot(gnorms1, '--', marker=markers[ind], markevery=(markevery + 2 * ind, markevery), markersize=10, color=color_ar_1[k])
        axs[1].plot(gnorms2, marker=markers[ind], markevery=(markevery + 2 * ind, markevery), markersize=10, color=color_ar_1[k],
                    label=r'$\alpha$ = {}'.format(np.format_float_scientific(alpha, trim='-', exp_digits=1)))
        # axs[2].plot(dists, marker=markers[ind], markevery=(markevery + 2 * ind, markevery), markersize=10)
        # axs[3].plot(dists2, marker=markers[ind], markevery=(markevery + 2 * ind, markevery), markersize=10)
        # ind += 1
        k += 1

    # axs[0].legend(
        # [r'$\alpha$ = {}'.format(np.format_float_scientific(alpha, trim='-', exp_digits=1)) for alpha in alphas_shown])
    # axs[0].legend([r'$\alpha$=1', r'$\alpha$=1e-1', r'$\alpha$=1e-2', r'$\alpha$=1e-3', r'$\alpha$=1e-4'])
    axs[0].legend()
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    # axs[2].set_yscale('log')
    # axs[3].set_yscale('log')
    axs[1].set_xlabel('Communication rounds')
    # axs[0].set_ylabel('Squared distance')
    axs[0].set_ylabel(r'$f(x^k)-f^{\star}$')
    # axs[1].set_ylabel('Loss')
    # axs[1].set_ylabel(r'$\|\nabla f(x^k)- \nabla f^{\star}\|^2$')
    axs[1].set_ylabel(r'$\|\nabla f(x^k)\|^2$')
    # axs[2].set_ylabel(r'$\frac{1}{n}\sum_{i=1}^n \|x_i^k - x^{\star} \|^2$')
    # axs[3].set_ylabel(r'$\frac{1}{n}\sum_{i=1}^n \|x_i^k - x_i^{\star} \|^2$')
    axs[0].set_title(dataset_name)
    alg = get_alg(logreg)
    name = exp + '_' + alg + '_' + dataset_name
    create_plot_dir()
    plt.savefig(PLOT_PATH + '/' + name + '_ours1_' + '.pdf')
    plt.show()