from workers import MasterNode
from models import LinReg, LogReg, LogRegNoncvx
from utils import read_run, get_alg, create_plot_dir, PLOT_PATH

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from prep_data import number_of_features

plt.style.use('ggplot')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['legend.fontsize'] = 'x-large'
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['axes.labelsize'] = 'x-large'


alphas = [0.05, 0.1, 0.4, 0.8, 1.0]
dataset_name = 'mushrooms'
n_workers = 8
exp = 'diana'
max_it = 400

alg = LogReg
logreg = True

d = number_of_features(dataset_name)
ks = np.random.randint(d, size=n_workers)
ks += 1
print(ks)


for alpha in alphas:
    print('alpha = {} \n --------------------'.format(alpha))
    model = MasterNode(n_workers, alpha, alg, dataset_name, logreg, True, max_it)
    if (exp == 'gd'):
        print('Running Gradient Descent...')
        model.run_gd(max_it)
    elif (exp == 'cgd'):
        print('Running Compressed Gradient Descent...')
        model.run_cgd(max_it)
    elif (exp == 'diana'):
        print('Running DIANA...')
        model.run_diana_sparsification(ks, max_it)
    else:
        print("Error: experiment name {} is not recognized".format(exp))

fig, axs = plt.subplots(2)
for alpha in alphas:
    run = read_run(exp, alpha, dataset_name, logreg)

    f_values = run['fval']
    dists = run['dist']

    axs[0].plot(dists)
    axs[1].plot(f_values)

axs[0].legend([r'$\alpha$ = {}'.format(alpha) for alpha in alphas])
axs[0].set_yscale('log')
axs[1].set_yscale('log')
axs[1].set_xlabel('Communication rounds')
axs[0].set_ylabel(r'$\|\|x - x^\ast\|\|^2$')
axs[1].set_ylabel(r'$f(x) - f^\star$')

# for alpha in alphas:
#     run = read_run(exp, alpha, dataset_name, logreg)
#     grad_norms = run['grad']
#     plt.plot(grad_norms, label=r'$\alpha=$ {}'.format(alpha))

# plt.tight_layout()
# plt.xlabel('iteration')
# plt.ylabel(r'$\|\nabla f(x^k)\|^2$')
# plt.yscale('log')
alg = get_alg(logreg)
name = exp + '_' + alg + '_' + dataset_name
create_plot_dir()
plt.tight_layout()
plt.savefig(PLOT_PATH + '/' + name + '.pdf')
plt.show()