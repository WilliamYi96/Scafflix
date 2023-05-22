import numpy as np
import os
import glob
import random
from pickle import load, dump
from functools import wraps
from warnings import warn

SAVED_RUNS_PATH = 'saved_exp'
PLOT_PATH = 'plots'


def save_run(exp, alpha, grad, f_val, distance, distance2, dataset_name, logreg, k_sparse=None):
    assert np.std(alpha) < 1e-5
    assert k_sparse is None or np.std(k_sparse) == 0
    if not os.path.isdir(SAVED_RUNS_PATH):
        os.makedirs(SAVED_RUNS_PATH)
    run = create_run(exp, alpha, grad, f_val, distance, distance2)
    alg = get_alg(logreg)
    name = exp + '_' + alg + '_' + dataset_name + '_' + str(alpha[0])
    # name = exp + '_' + alg + '_' + dataset_name + '_' + str(alpha)

    if k_sparse is not None:
        name += '_' + str(k_sparse[0])
    
    file = os.path.join(SAVED_RUNS_PATH, name + '.pickle')
    print(f'Saved file name: {file}')
    with open(file, 'wb') as f:
        dump(run, f)


def read_run(exp, alpha, dataset_name, logreg, k_sparse=None):
    assert np.std(alpha) < 1e-5
    assert k_sparse is None or np.std(k_sparse) == 0
    alg = get_alg(logreg)
    name = exp + '_' + alg + '_' + dataset_name + '_' + str(alpha[0])
    # name = exp + '_' + alg + '_' + dataset_name + '_' + str(alpha)

    if k_sparse is not None:
        name += '_' + str(k_sparse[0])

    file = os.path.join(SAVED_RUNS_PATH, name + '.pickle')
    print(f'Read file name: {file}')
    with open(file, 'rb') as f:
        run = load(f)
    return run


def create_run(exp, alpha, grad, f_val, distance, distance2):
    run = {'name': exp,
           'alpha': alpha,
           'grad': grad,
           'fval': f_val,
           'dist': distance,
           'dist2': distance2}
    return run


def create_plot_dir():
    if not os.path.isdir(PLOT_PATH):
        os.makedirs(PLOT_PATH)


def get_alg(logreg):
    if logreg:
        return 'log_reg'
    return 'linreg'


def seed_everything(seed=42):
    """
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    
def deprecated(func):
    """Raise deprecation warning."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        warn('Function {} is deprecated.'.format(func.__name__), category=DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)
    return wrapper
        