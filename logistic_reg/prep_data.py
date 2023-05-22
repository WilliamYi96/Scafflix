from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
import warnings

DATASET_PATH = 'datasets/'
RANDOM_STATE = 100


def load_data(dataset_name, n_workers, logreg, ordered, multilabel=False, d_y=None, npz=False):
    if not ordered:
        warnings.warn('<ordered> is set to False.')
    if multilabel & (not logreg):
        raise ValueError('If <multilabel> is True, <logreg> must be True, too.')
    if multilabel & (d_y is None):
        raise ValueError('<d_y> must be an integer.')
   
    if npz:
        arrays = np.load(DATASET_PATH + dataset_name)
        x = arrays['X']
        y = arrays['y']
    else:
        x, y = load_svmlight_file(DATASET_PATH + dataset_name, multilabel=multilabel)
        
    n = x.shape[0]
    if (not multilabel) & logreg:
        y = (y != -1.) * 1
    indices = np.arange(n)
    if multilabel and (type(y) == coo_matrix or type(y) == csr_matrix):
        labels = np.zeros(shape=(n, d_y))
        labels[[row_num for row_num in range(len(y)) for col_num in y[row_num]], [int(col_num) for row in y for col_num in row]] = 1
        y = labels
    if not ordered:
        np.random.shuffle(indices)
    indices = np.array_split(indices, n_workers)
    data_workers = [(x[ind_i], y[ind_i]) for ind_i in indices]
    return data_workers

def load_data_npy(prefix, node_id, postfix_X='_X', postfix_y='_y'):
    path_x = DATASET_PATH + prefix + str(node_id) + postfix_X + '.npy'
    path_y = DATASET_PATH + prefix + str(node_id) + postfix_y + '.npy'
    with open(path_x, 'rb') as file:
        x = np.load(file)
    with open(path_y, 'rb') as file:
        y = np.load(file)       
    ones = np.ones((x.shape[0], 1))
    x = np.hstack([ones, x])
    return x, y

def number_of_features(dataset_name):
    x, _ = load_svmlight_file(DATASET_PATH + dataset_name)
    # additional dimension corresponds to the intercept
    return x.shape[1] + 1

def split_data(dataset_name, validation_proportion=0.1):
    x, y = load_svmlight_file(DATASET_PATH + dataset_name)
    x_train, x_validation, y_train, y_validation = train_test_split(x, 
                                                                    y, 
                                                                    test_size=validation_proportion)
    print(x_train.shape)
    print(x_validation.shape)
    dump_svmlight_file(x_train, y_train, DATASET_PATH + dataset_name + 'train')
    dump_svmlight_file(x_validation, y_validation, DATASET_PATH + dataset_name + 'validation')
    return dataset_name + 'train', dataset_name + 'validation'
        
