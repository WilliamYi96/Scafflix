import numpy as np
from numpy.random import default_rng
from typing import Callable

from prep_data import load_data, load_data_npy
from models import Node
import models
from utils import save_run, deprecated
from scipy.sparse import csr_matrix, coo_matrix
from tensorflow.keras.metrics import Recall
import warnings
import time
import copy

K_CONST_SPARSIFICATION = 5
EXTRA_IT = 200


class MasterNode:
    def __init__(self, n_workers: int, alpha: [float, int, np.ndarray, list], worker: Node, dataset_name: str,
                 logreg: bool = True, ordered: bool = True, max_it: int = 100, compute_smoothness_min: bool = True,
                 **kwargs):
        """Generate Master Node.

        Each computing edge (or client) must belong to Node class. Datasets (train and, if provided, validation) are distributed among computing edges from MasterNode class.
        If alpha is float or int, it is assumed that each client has got the same alpha parameter.

        Arguments:
        n_workers -- number of computing edges
        alpha -- alpha parameter of Explicit Mixture method
        worker -- type of each computing edge
        dataset_name -- name of the dataset from folder 'datasets/' from where the data is divided equally between computing edges
        logreg -- boolean representing whether the problem solved is Logistic Regression or not
        ordered -- boolean representing whether the data for train and validation (if provided) is shuffled (False) or not (True)
        max_it -- number of gradient steps for estimating the global model for Explicit Mixture method
        compute_smoothness_min -- boolean representing whether to compute smoothness constant and pure local model of each computing edge or not

        Keyword arguments:
        features_data_type (str) -- data type of feature matrix; options np.ndarray, coo_matrix, csr_matrix (default np.ndarray)
        labels_data_type (str) -- data type of label matrix; options np.ndarray, coo_matrix, csr_matrix (default np.ndarray)
        multilabel (bool) -- boolean representing whether the problem is multiclass (True) or one-class (False) logistic regression
        d_y (int) -- number of classes in case of multiclass logistic regression
        npz (bool) -- whether dataset is saved in numpy npz (True) or svmlight (False) format (default False)
        validation (bool) -- boolean representing whether the validation dataset is presented or not
        val_rat (float) -- float between 0.0 and 1.0 representing in what proportion (1 - val_rat : val_rat) the dataset should be split between train and validation
        validation_dataset_name (str) -- name of the validation dataset in folder 'datasets/'
        tolerance (float) -- tolerance in terms of gradient norm to which each pure local model is computed
        gd_stepsize (float) -- stepsize for stochastic gradient descent used to compute pure local model for Node of type 'NN_1d_regression' from models.py
        regularization (float) -- l_2 regularization parameter
        interm_dim1 (int) -- interm_dim1 parameter for Node of type 'NN_1d_regression' (default 40)
        interm_dim2 (int) -- interm_dim2 parameter for Node of type 'NN_1d_regression' (default 40)
        fomaml_inner_loop_lr (float) -- inner loop stepsize for FOMAML method (default 0.01)
        fomaml_outer_loop_lr (float) -- outer loop stepsize for FOMAML method (default 0.001)
        fomaml_number_of_inner_steps (int) -- number of inner loops for FOMAML method (default 5)
        reptile_inner_loop_lr (float) -- inner loop stepsize for Reptile method (default 0.01)
        reptile_outer_loop_lr (float) -- outer loop stepsize for Reptile method (default 0.001)
        reptile_number_of_inner_steps (int) -- number of inner loops for Reptile method (default 5)
        sgd_mixed_outer_loop (int) -- number of (outer) loops for Explicit Mixture method (default 0.001)
        modexpmix_inner_loop_lr (float) -- inner loop stepsize for Modified Explicit Mixture method (default 0.01)
        modexpmix_outer_loop_lr (float) -- outer loop stepsize for Modified Explicit Mixture method (default 0.001)
        modexpmix_number_of_inner_steps (int) -- number of inner loops for Modified Explicit Mixture method (default 5)
        """
        if type(alpha) == float or type(alpha) == int:
            if alpha < 0.0 or alpha > 1.0:
                raise ValueError('alpha must be between 0 and 1, current value {}'.format(alpha))
        elif type(alpha) == np.ndarray or type(alpha) == list:
            for alpha_ in alpha:
                if alpha_ < 0.0 or alpha_ > 1.0:
                    raise ValueError('all elements of alpha must be between 0 and 1, current state {}'.format(alpha))
        else:
            raise TypeError(
                'Unknown type ({}) for alpha. float, int, np.ndarray or list are allowed'.format(type(alpha)))

        features_data_type = kwargs.get('features_data_type', 'np.ndarray')
        labels_data_type = kwargs.get('labels_data_type', 'np.ndarray')

        if not (features_data_type in {'np.ndarray', 'coo_matrix', 'csr_matrix'}):
            raise TypeError('Unknown type: {}'.format(features_data_type))

        if not (labels_data_type in {'np.ndarray', 'coo_matrix', 'csr_matrix'}):
            raise TypeError('Unknown type: {}'.format(labels_data_type))

        self.n_workers = n_workers
        if isinstance(alpha, (float, int)):
            alpha = alpha * np.ones(n_workers)

        self.validation = kwargs.get('validation', False)
        if self.validation:
            assert kwargs.get('val_rat') is not None or kwargs.get('validation_dataset_name') is not None

        multilabel = kwargs.get('multilabel', False)
        d_y = kwargs.get('d_y', None)
        npz = kwargs.get('npz', False)
        assert type(npz) == bool

        data = load_data(dataset_name, n_workers, logreg, ordered, multilabel=multilabel, d_y=d_y, npz=npz)

        self.validation_ratio = kwargs.get('val_rat', None)

        if self.validation is True and self.validation_ratio is None:
            val_data = load_data(kwargs.get('validation_dataset_name'), n_workers, logreg, ordered,
                                 multilabel=multilabel, d_y=d_y)

        self.workers = dict()
        self.alpha = alpha

        for i in range(n_workers):
            x_train, y_train = data[i]

            if features_data_type == 'np.ndarray' and not npz:
                x_train = x_train.toarray()
            elif features_data_type == 'csr_matrix':
                x_train = csr_matrix(x_train)
            elif features_data_type == 'coo_matrix' and npz:
                x_train = coo_matrix(x_train)

            if labels_data_type == 'coo_matrix':
                y_train = coo_matrix(y_train)
            elif labels_data_type == 'csr_matrix':
                y_train = csr_matrix(y_train)
            else:
                assert labels_data_type == 'np.ndarray'
                assert type(y_train) == np.ndarray

            worker_kwargs = {}
            worker_kwargs['compute_smoothness_min'] = compute_smoothness_min

            if kwargs.get('tolerance') is not None:
                worker_kwargs['tolerance'] = kwargs.get('tolerance')

            if kwargs.get('gd_stepsize') is not None:
                worker_kwargs['stepsize'] = kwargs.get('gd_stepsize')

            if kwargs.get('regularization') is not None:
                worker_kwargs['regularization'] = kwargs.get('regularization')

            if self.validation:
                worker_kwargs['validation'] = True
                if self.validation_ratio is None:
                    x_val, y_val = val_data[i]

                    if features_data_type == 'np.ndarray':
                        x_val = x_val.toarray()
                    elif features_data_type == 'csr_matrix':
                        x_val = csr_matrix(x_val)
                    else:
                        assert features_data_type == 'coo_matrix'
                        assert type(x_val) == coo_matrix

                    if labels_data_type == 'coo_matrix':
                        y_val = coo_matrix(y_val)
                    elif labels_data_type == 'csr_matrix':
                        y_val = csr_matrix(y_val)
                    else:
                        assert labels_data_type == 'np.ndarray'
                        assert type(y_val) == np.ndarray

                    worker_kwargs['x_validation'] = x_val
                    worker_kwargs['y_validation'] = y_val
                else:
                    worker_kwargs['val_rat'] = self.validation_ratio

            if worker == models.NN_1d_regression:
                worker_kwargs['interm_dim1'] = kwargs.get('interm_dim1', 40)
                worker_kwargs['interm_dim2'] = kwargs.get('interm_dim2', 40)

            # print(x_train.shape, y_train.shape)  # (6249, 22) (6249, )
            self.workers[i] = worker(i, alpha[i], x_train, y_train, **worker_kwargs)

        self.l_s = np.array([self.workers[i].smoothness for i in self.workers])
        self.dataset_name = dataset_name
        # FOMAML parameters
        self.fomaml_inner_loop_lr = kwargs.get('fomaml_inner_loop_lr', 0.01)
        self.fomaml_outer_loop_lr = kwargs.get('fomaml_outer_loop_lr', 0.001)
        self.fomaml_number_of_inner_steps = kwargs.get('fomaml_number_of_inner_steps', 5)
        # Reptile parameters
        self.reptile_inner_loop_lr = kwargs.get('reptile_inner_loop_lr', 0.01)
        self.reptile_outer_loop_lr = kwargs.get('reptile_outer_loop_lr', 0.001)
        self.reptile_number_of_inner_steps = kwargs.get('reptile_number_of_inner_steps', 5)
        # SGD for explicit mixture parameters
        self.sgd_mixed_outer_loop = kwargs.get('sgd_mixed_outer_loop', 0.001)
        # SGD for Modified Explicit Mixture Model
        self.modexpmix_inner_loop_lr = kwargs.get('modexpmix_inner_loop_lr', 0.01)
        self.modexpmix_outer_loop_lr = kwargs.get('modexpmix_outer_loop_lr', 0.001)
        self.modexpmix_number_of_inner_steps = kwargs.get('modexpmix_number_of_inner_steps', 5)

        self.logreg = logreg
        self.d = self.workers[0].iterate_size()
        self.d_y = self.workers[0].d_y
        self.max_it = max_it
        self.compute_smoothness_min = compute_smoothness_min
        if compute_smoothness_min and worker != models.NN_1d_regression:
            self.smoothness = self._smoothness()
            self.w_opt_global = self.find_min()
        else:
            self.smoothness = None
            self.w_opt_global = None

    def update_workers(self):
        for i in range(self.n_workers):
            worker = self.workers[i]
            worker.w_opt = worker.find_min()
            worker.smoothness = worker._smoothness()
        self.l_s = np.array([self.workers[i].smoothness for i in self.workers])

    def change_alpha(self, alpha, max_it=None):
        if isinstance(alpha, (float, int)):
            alpha = alpha * np.ones(self.n_workers)
        for i in range(self.n_workers):
            self.workers[i].change_alpha(alpha[i])
        self.alpha = alpha
        if self.compute_smoothness_min:
            self.smoothness = self._smoothness()
            if max_it is None:
                max_it = self.max_it

            self.recompute_global_min(max_it)

    def _smoothness(self):
        return np.mean(self.alpha ** 2 * self.l_s)

    def grad(self, w):
        grad_vec = np.array([self.workers[i].grad_shift(self.workers[i].x_train, self.workers[i].y_train, w)
                             for i in self.workers])
        grad = np.mean(grad_vec, axis=0)
        return grad

    def grad_local(self, x, y, w, i):
        grad_local, w_local = self.workers[i].grad_local2(x, y, w)
        return grad_local, w_local

    def fun_value(self, w):
        f_val_vec = np.array([self.workers[i].fun_value_shift(w) for i in self.workers])
        return np.mean(f_val_vec)

    def model_grad(self, model: str, w: np.ndarray, task_batch: [int, 'full'] = 'full',
                   data_batch: [int, 'full'] = 'full', chosen_workers: [np.array, list] = None, **kwargs) -> np.ndarray:
        """Compute a stochastic update for a model.

        Arguments:
        model -- model for which the update is computed; available values are 'expmix' for Explicit Mixture, 'modexpmix' for Modified Explicit Mixture, 'fomaml' for FOMAML, and 'reptile' for Reptile
        w -- point to estimate the gradient at
        task_batch -- number of clients involved in estimation; if chosen_workers array is given, task_batch is ignored; if set to 'full', all clients are used for estimation of the gradient (default 'full')
        data_batch -- number of data points at each client used for estimation of gradient; if set to 'full', all the data of a client is used for estimation (default 'full')
        chosen_workers -- array of workers involved in estimation of gradient; if given, task_batch is ignored (default=None)

        Keyword arguments:
        joint_dataset -- boolean representing whether to concatenate train and validation dataset at each machine or not; valid only when model = 'reptile'
        """
        if chosen_workers is None:
            if task_batch == 'full':
                task_batch = len(self.workers)
            generator = default_rng()
            chosen_workers = generator.choice(len(self.workers), size=task_batch, replace=False)

        f_kwargs = {}

        if model == 'expmix':
            f_kwargs = {'w': w, 'batch': data_batch, **kwargs}
        elif model == 'modexpmix':
            f_kwargs = {'w': w, 'k': self.modexpmix_number_of_inner_steps,
                        'inner_loop_lr': self.modexpmix_inner_loop_lr, 'batch': data_batch, **kwargs}
        elif model == 'fomaml':
            f_kwargs = {'w': w, 'k': self.fomaml_number_of_inner_steps, 'inner_loop_lr': self.fomaml_inner_loop_lr,
                        'batch': data_batch, **kwargs}
        elif model == 'reptile':
            f_kwargs = {'w': w, 'k': self.reptile_number_of_inner_steps, 'inner_loop_lr': self.reptile_inner_loop_lr,
                        'batch': data_batch, **kwargs}
        else:
            raise NameError(
                'Unknown model {}. Available models are expmix, modexpmix, fomaml and reptile.'.format(model))
        return np.array(
            [self.workers[i].model_stochastic_update(model=model, **f_kwargs) for i in chosen_workers]).mean(axis=0)

    def learning(self, model: str = 'expmix', w_0: np.ndarray = None, epochs: int = None, task_batch: [int, 'full'] = 5,
                 data_batch: [int, 'full'] = 'full', n_iter: int = 100, save_history: bool = False,
                 **kwargs) -> np.ndarray:
        """Run learning algorithm depending on a model.

        Arguments:
        model -- model for which the algorithm is run; available values are 'expmix' for Explicit Mixture, 'modexpmix' for Modified Explicit Mixture, 'fomaml' for FOMAML, and 'reptile' for Reptile (default 'expmix')
        w_0 -- starting point of the algorithm; if None, the starting point is sampled from standard disribution (default None)
        epochs -- number of client epochs (default None)
        task_batch -- number of clients participating in each round; if set to 'full', all clients participate in each round (default 5)
        data_batch -- number of data points at each client used for estimation of gradient; if set to 'full', all the data of a client is used for estimation (default 'full')
        n_iter -- number of rounds; if epochs is given, then n_iter is ignored; either epochs or n_iter must be provided (default None)
        save_history -- whether to save the history of learning or not; if True, the function returns np.ndarray with all points through which the learning algorithm has passed, otherwise, only the last point (deault False)

        Keyword arguments:
        joint_dataset -- boolean representing whether to concatenate train and validation dataset at each machine or not; valid only when model is 'reptile'
        """
        if w_0 is None:
            generator = default_rng()
            w_0 = generator.normal(size=self.d)

        w = copy.deepcopy(w_0)

        if epochs is None and n_iter is None:
            raise ValueError('Either epochs or n_iter must be set; both are None values.')

        history = [w]

        params = {}
        if model == 'expmix':
            params = {'full name': 'Explicit Mixture', 'outer loop': self.sgd_mixed_outer_loop, 'update sign': -1}
        elif model == 'modexpmix':
            params = {'full name': 'Modified Explicit Mixture', 'outer loop': self.modexpmix_outer_loop_lr,
                      'update sign': -1}
        elif model == 'fomaml':
            params = {'full name': 'FOMAML', 'outer loop': self.fomaml_outer_loop_lr, 'update sign': -1}
        elif model == 'reptile':
            params = {'full name': 'Reptile', 'outer loop': self.reptile_outer_loop_lr, 'update sign': +1}
        else:
            raise NameError(
                'Unknown model {}. Available models are expmix, modexpmix, fomaml and reptile.'.format(model))

        print('Running {}.'.format(params['full name']))

        if epochs is None:
            for it in range(n_iter):
                print('Iteration {} / {}'.format(it, n_iter))
                curr_grad = self.model_grad(model, w, task_batch, data_batch, None, **kwargs)
                if (curr_grad.shape != w.shape):
                    warnings.warn(
                        'Gradient {} and w {} shapes mismatch. Shapes will be equalled.'.format(curr_grad.shape,
                                                                                                w.shape))
                    curr_grad = np.reshape(curr_grad, w.shape)
                w += params['update sign'] * params['outer loop'] * curr_grad
                if save_history:
                    history.append(w)
        else:
            if task_batch == 'full':
                task_batch = self.n_workers

            epoch_size = int(self.n_workers / task_batch)
            generator = default_rng()
            for epoch in range(epochs):
                print('Epoch # {}'.format(epoch))
                seq = generator.choice(self.n_workers, self.n_workers)
                for i in range(epoch_size):
                    print('#' * (i + 1) + '-' * (epoch_size - 1 - i), end='\r')
                    chosen_workers = seq[i * task_batch: (i + 1) * task_batch]
                    curr_grad = self.model_grad(model, w, task_batch, data_batch, chosen_workers, **kwargs)
                    if (curr_grad.shape != w.shape):
                        warnings.warn(
                            'Gradient {} and w {} shapes mismatch. Shapes will be equalled.'.format(curr_grad.shape,
                                                                                                    w.shape))
                        curr_grad = np.reshape(curr_grad, w.shape)
                    w += params['update sign'] * params['outer loop'] * curr_grad
                    if save_history:
                        history.append(w)
                print('')

        if save_history:
            return history
        return w

    def compute_recall_at_k(self, model: str, w: np.ndarray, counter_to_id: list, prefix: str, postfix_X: str = '_X',
                            postfix_y: str = '_y', top_k=5, **kwargs) -> float:
        """Compute Recall @ k for given model.
        Arguments:
        model -- string representing the model; available values are 'expmix' for Explicit Mixture and 'local' for Modified Explicit Mixture, FOMAML, and Reptile
        w -- global model for Explicit Mixture or initial point for local methods
        counter_to_id -- list connecting sequence number computing edge with external dataset id; e.g. with Stackoveflow dataset 'client_id' parameter
        prefix -- prefix of external dataset for testing; full dataset address for features (labels) is DATASET_PATH + prefix + external_dataset_id + postfix_X (+ postfix_y)
        postfix_X -- postfix of external features dataset for testing (default '_X')
        postfix_y -- postfix of external labels dataset for testing (default '_y')
        top_k -- value specifying top-k predictations used in calculation recall
        Keyword arguments:
        stepsize -- stepsize of local model
        k -- number of inner steps of local model
        """
        if not (model == 'expmix' or model == 'local'):
            raise NameError('Unknown model {}. Available models are \'expmix\' and \'local\'.'.format(model))
        if not (type(self.workers[0]) == models.LogReg or type(self.workers[0] == models.MulticlassLogReg) or type(
                self.workers[0]) == models.LogRegNoncvx or type(self.workers[0]) == models.LinReg):
            raise TypeError(
                'Function \'compute_recall_at_k\' can be applied only on classification task problems. Task of type {} is encountered.'.format(
                    type(self.workers[0])))

        metrics = Recall(top_k=top_k)
        metrics.reset_state()
        predicts = np.empty(shape=(0, self.d_y))
        true_labels = np.empty(shape=(0, self.d_y))
        for worker_counter, worker in self.workers.items():
            client_id = counter_to_id[worker_counter]
            start_time = time.time()
            x, y = load_data_npy(prefix=prefix, node_id=client_id, postfix_X=postfix_X, postfix_y=postfix_y)

            assert y.shape[1] == self.d_y
            assert x.shape[0] == y.shape[0]

            true_labels = np.vstack((true_labels, y))
            start_time = time.time()
            if model == 'expmix':
                w_local = worker.compute_local(w)
            else:
                w_local = worker.local_steps(w, **kwargs)
            curr_predicts = worker.get_h(x, w_local)
            predicts = np.vstack((predicts, curr_predicts))
        metrics.update_state(true_labels, predicts)
        return metrics.result().numpy(), true_labels, predicts

    def compute_recall_at_k_mixture(self, w: np.ndarray, counter_to_id: list, prefix: str, postfix_X: str = '_X',
                                    postfix_y: str = '_y') -> float:
        """Compute Recall @ k value for Explicit Mixture.
        Arguments:
        model -- workers.MasterNode instance containing all workers - computing edges; alpha must be set by MasterNode class methods prior to function call
        w -- global model for Explicit Mixture
        counter_to_id -- list connecting sequence number computing edge with external dataset id; e.g. with Stackoveflow dataset 'client_id' parameter
        prefix -- prefix of external dataset for testing; full dataset address for features (labels) is DATASET_PATH + prefix + external_dataset_id + postfix_X (+ postfix_y)
        postfix_X -- postfix of external features dataset for testing (default '_X')
        postfix_y -- postfix of external labels dataset for testing (default '_y')
        """
        return self.compute_recall_at_k(model='expmix', w=w, counter_to_id=counter_to_id, prefix=prefix,
                                        postfix_X=postfix_X, postfix_y=postfix_y)

    def compute_recall_at_k_local(self, w: np.ndarray, counter_to_id: list, stepsize: float, k: int, prefix: str,
                                  postfix_X: str = '_X', postfix_y: str = '_y') -> float:
        """Compute Recall @ k value for local models (FOMAML, Reptile, and Modified Explicit Mixture).
        Arguments:
        model -- workers.MasterNode instance containing all workers - computing edges
        w -- initial point of the local model
        counter_to_id -- list connecting sequence number computing edge with external dataset id; e.g. with Stackoveflow dataset 'client_id' parameter
        prefix -- prefix of external dataset for testing; full dataset address for features (labels) is DATASET_PATH + prefix + external_dataset_id + postfix_X (+ postfix_y)
        postfix_X -- postfix of external features dataset for testing (default '_X')
        postfix_y -- postfix of external labels dataset for testing (default '_y')
        stepsize -- stepsize of the local model
        k -- number of inner steps of local model
        """
        return self.compute_recall_at_k(model='local', w=w, counter_to_id=counter_to_id, prefix=prefix,
                                        postfix_X=postfix_X, postfix_y=postfix_y, stepsize=stepsize, k=k)

    def compute_loss_on_external_dataset(self, model: str, w: np.ndarray, counter_to_id: list, prefix: str,
                                         postfix_X: str = '_X', postfix_y: str = '_y', **kwargs) -> np.float32:
        """Compute the loss value on external dataset.

        Arguments:
        model -- string representing the model; available values are 'expmix' for Explicit Mixture and 'local' for Modified Explicit Mixture, FOMAML, and Reptile
        w -- global model for Explicit Mixture or initial point for local methods
        counter_to_id -- list connecting sequence number computing edge with external dataset id; e.g. with Stackoveflow dataset 'client_id' parameter
        prefix -- prefix of external dataset for testing; full dataset address for features (labels) is DATASET_PATH + prefix + external_dataset_id + postfix_X (+ postfix_y)
        postfix_X -- postfix of external features dataset for testing (default '_X')
        postfix_y -- postfix of external labels dataset for testing (default '_y')
        Keyword arguments:
        stepsize -- stepsize of local model
        k -- number of inner steps of local model
        """
        if not (model == 'expmix' or model == 'local'):
            raise NameError('Unknown model {}. Available models are \'expmix\' and \'local\'.'.format(model))

        loss_array = []
        loss_weights = []

        for worker_counter, worker in self.workers.items():
            client_id = counter_to_id[worker_counter]
            start_time = time.time()
            x, y = load_data_npy(prefix=prefix, node_id=client_id, postfix_X=postfix_X, postfix_y=postfix_y)

            assert y.shape[1] == self.d_y
            assert x.shape[0] == y.shape[0]

            start_time = time.time()
            n, _ = x.shape
            if model == 'expmix':
                w_local = worker.compute_local(w)
            else:
                w_local = worker.local_steps(w, **kwargs)
            loss_array.append(worker.fun_value_general(x, y, w_local))
            loss_weights.append(n)
        return np.average(a=loss_array, weights=loss_weights)

    def compute_loss_on_external_dataset_mixture(self, w: np.ndarray, counter_to_id: list, prefix: str,
                                                 postfix_X: str = '_X', postfix_y: str = '_y') -> np.float32:
        return self.compute_loss_on_external_dataset(model='expmix', w=w, counter_to_id=counter_to_id, prefix=prefix,
                                                     postfix_X=postfix_X, postfix_y=postfix_y)

    def compute_loss_on_external_dataset_local(self, w: np.ndarray, counter_to_id: list, stepsize: float, k: int,
                                               prefix: str, postfix_X: str = '_X', postfix_y: str = '_y') -> np.float32:
        return self.compute_loss_on_external_dataset(model='local', w=w, counter_to_id=counter_to_id, prefix=prefix,
                                                     postfix_X=postfix_X, postfix_y=postfix_y, stepsize=stepsize, k=k)

    @deprecated
    def stochastic_grad(self, w: np.ndarray, task_batch='full', data_batch='full', chosen_workers=None):
        """Compute stochastic gradient for Explicit Mixture Algorithm."""
        if chosen_workers is None:
            if task_batch == 'full':
                task_batch = len(self.workers)
            generator = default_rng()
            chosen_workers = generator.choice(len(self.workers), size=task_batch, replace=False)

        stoch_grad_vec = np.array([self.workers[i].explicit_mixture_stochastic_grad(w, batch=data_batch)
                                   for i in chosen_workers])
        stoch_grad = np.mean(stoch_grad_vec, axis=0)
        return stoch_grad

    @deprecated
    def modified_stochastic_grad(self, w: np.ndarray, task_batch='full', data_batch='full', chosen_workers=None):
        """Compute stochastic gradient for Modified Explicit Mixture Algorithm."""
        if chosen_workers is None:
            if task_batch == 'full':
                task_batch = len(self.workers)
            generator = default_rng()
            chosen_workers = generator.choice(len(self.workers), size=task_batch, replace=False)

        grad_vec = np.array([
            self.workers[i].modified_explicit_mixture_stochastic_gradient(w, self.modexpmix_number_of_inner_steps,
                                                                          self.modexpmix_inner_loop_lr, data_batch)
            for i in chosen_workers
        ])
        grad = np.mean(grad_vec, axis=0)
        return grad

    @deprecated
    def fomaml_grad(self, w: np.ndarray, task_batch='full', data_batch='full', chosen_workers=None):
        """Compute FOMAML gradient.

        Arguments:
        w -- point to compute the gradient at
        batch -- number of data points used for estimating the gradient at each machine (default 'full')
        """
        if chosen_workers is None:
            if task_batch == 'full':
                task_batch = len(self.workers)
            generator = default_rng()
            chosen_workers = generator.choice(len(self.workers), size=task_batch, replace=False)

        grad_vec = np.array([
            self.workers[i].fomaml_stochastic_grad(w, self.fomaml_number_of_inner_steps, self.fomaml_inner_loop_lr,
                                                   data_batch)
            for i in chosen_workers
        ])
        #         print("FOMAML's table of gradients: {}".format(grad_vec))
        grad = np.mean(grad_vec, axis=0)
        return grad

    @deprecated
    def reptile_update(self, w: np.ndarray, task_batch: [int, 'full'] = 'full', data_batch: [int, 'full'] = 'full',
                       chosen_workers: [list, np.ndarray] = None, joint_dataset: bool = True) -> np.ndarray:
        """Compute Reptile update.

        Arguments:
        w -- point to compute the gradient at
        task_batch -- number of computing edges involved in gradient estimation (default 'full')
        data_batch -- number of data points used for estimating the gradient at each machine (default 'full')
        chosen_workers -- array of workers involved in computation; if given task_batch parameter is ignored
        joint_dataset -- boolean representing whether to concatenate train and validation dataset at each machine or not
        """
        if chosen_workers is None:
            if task_batch == 'full':
                task_batch = len(self.workers)
            generator = default_rng()
            chosen_workers = generator.choice(len(self.workers), size=task_batch, replace=False)

        update_vec = np.array([self.workers[i].reptile_stochastic_update(w, self.reptile_number_of_inner_steps,
                                                                         self.reptile_inner_loop_lr, data_batch,
                                                                         joint_dataset=joint_dataset)
                               for i in chosen_workers])
        update = np.mean(update_vec, axis=0)
        return update

    @deprecated
    def sgd_mixed(self, w: np.ndarray = None, epochs=None, task_batch=5, data_batch='full', n_iter=100,
                  save_history=False):
        print_every = 100
        if w is None:
            generator = default_rng()
            w = generator.normal(size=self.d)

        history = [w]

        if epochs is None:
            return NotImplementedError
        else:
            generator = default_rng()
            epoch_size = int(self.n_workers / task_batch)
            curr_it = None
            for epoch in range(epochs):
                print('Epoch # {}'.format(epoch))
                seq = generator.choice(self.n_workers, self.n_workers)
                for i in range(epoch_size):
                    print('{} / {} epoch progress'.format(i, epoch_size))
                    chosen_workers = seq[i * task_batch: (i + 1) * task_batch]
                    curr_grad = self.stochastic_grad(w, data_batch=data_batch, chosen_workers=chosen_workers)
                    if (curr_grad.shape != w.shape):
                        warnings.warn(
                            'Gradient {} and w {} shapes mismatch. Shapes will be equalled.'.format(curr_grad.shape,
                                                                                                    w.shape))
                        curr_grad = np.reshape(curr_grad, w.shape)
                    w -= self.sgd_mixed_outer_loop * curr_grad
                    if save_history:
                        history.append(w)
                    curr_it = epoch * epoch_size + i
        #                     if curr_it % print_every == 0:
        #                         print('Iteration {}'.format(curr_it))
        if save_history:
            return history
        else:
            return w

    @deprecated
    def sgd_mixed_modified(self, w: np.ndarray = None, epochs=None, task_batch=5, data_batch='full', n_iter=100):
        print_every = 100
        if w is None:
            generator = default_rng()
            w = generator.normal(size=self.d)

        if epochs is None:
            return NotImplementedError
        else:
            generator = default_rng()
            for epoch in range(epochs):
                seq = generator.choice(self.n_workers, self.n_workers)
                epoch_size = int(self.n_workers / task_batch)
                for i in range(epoch_size):
                    chosen_workers = seq[i * task_batch: (i + 1) * task_batch]
                    curr_grad = self.modified_stochastic_grad(w, data_batch=data_batch, chosen_workers=chosen_workers)
                    w -= self.modexpmix_outer_loop_lr * curr_grad
                    curr_it = epoch * epoch_size + i
        #                     if curr_it % print_every == 0:
        #                         print('Iteration {}'.format(curr_it))
        return w

    @deprecated
    def fomaml(self, w: np.ndarray = None, epochs=None, task_batch=5, data_batch='full', n_iter=100,
               save_history=False):
        """Run FOMAML. See https://arxiv.org/pdf/1803.02999.pdf

        Arguments:
        w -- initial guess (default None)
        batch -- number of data points used for estimating the gradient at each machine (default 'full')
        n_iter -- number of iterations taken for FOMAML to run (default 100)
        """
        print_every = 100
        if w is None:
            generator = default_rng()
            w = generator.normal(size=self.d)

        history = [w]

        print("Running FOMAML.")
        if epochs is None:
            for it in range(n_iter):
                print('Iteration {}'.format(it))
                curr_grad = self.fomaml_grad(w, task_batch, data_batch)
                w -= self.fomaml_outer_loop_lr * curr_grad
                if save_history:
                    history.append(w)
        #                 if it % print_every == 0:
        #                     print('Iteration {}'.format(it))
        #             if it % 10 == 0:
        #                 print("Iteration {}: current gradient: {}".format(it, curr_grad))
        #                 print("Iteration {}: value of <<w>> {}".format(it, w))
        else:
            epoch_size = int(self.n_workers / task_batch)
            curr_it = None
            for epoch in range(epochs):
                print('Epoch #{}'.format(epoch))
                print('Epoch size is {}'.format(epoch_size))
                generator = default_rng()
                seq = generator.choice(self.n_workers, self.n_workers)
                for i in range(epoch_size):
                    print('{} / {} epoch progress'.format(i, epoch_size))
                    chosen_workers = seq[i * task_batch: (i + 1) * task_batch]
                    curr_grad = self.fomaml_grad(w, data_batch=data_batch, chosen_workers=chosen_workers)
                    w -= self.fomaml_outer_loop_lr * curr_grad
                    if save_history:
                        history.append(w)
                    curr_it = epoch * epoch_size + i
        #                     if curr_it % print_every == 0:
        #                         print('Iteration {}'.format(curr_it))

        if save_history:
            return history
        else:
            return w

    @deprecated
    def reptile(self, w: np.ndarray = None, epochs=None, task_batch=5, data_batch='full', n_iter=100,
                joint_dataset=True, save_history=False):
        """Run Reptile."""
        print_every = 100
        if w is None:
            generator = default_rng()
            w = generator.normal(size=self.d)

        print("Running Reptile.")

        history = [w]

        if epochs is None:
            for it in range(n_iter):
                curr_update = self.reptile_update(w, task_batch, data_batch)
                w += self.reptile_outer_loop_lr * curr_update
                if save_history:
                    history.append(w)
        #                 if it % print_every == 0:
        #                     print('Iteration {}, current_update = {}, current w = {}'.format(it, curr_update, w))
        else:
            for epoch in range(epochs):
                generator = default_rng()
                seq = generator.choice(self.n_workers, self.n_workers)
                epoch_size = int(self.n_workers / task_batch)
                curr_it = None
                for i in range(epoch_size):
                    chosen_workers = seq[i * task_batch: (i + 1) * task_batch]
                    curr_update = self.reptile_update(w, data_batch=data_batch, chosen_workers=chosen_workers,
                                                      joint_dataset=joint_dataset)
                    w += self.reptile_outer_loop_lr * curr_update
                    if save_history:
                        history.append(w)
                    curr_it = epoch * epoch_size + i
        #                     if curr_it % print_every == 0:
        #                         print('Iteration {}, current_update = {}, current w = {}'.format(curr_it, curr_update, w))

        if save_history:
            return history
        else:
            return w

    def run_gd(self, n_iter=100, save_memory=False, exp_name='gd', simple=True):
        """Run Gradient Descent.

        Arguments:
        n_iter -- maximum number of iterations taken for GD to converge (default 100)
        save_memory -- set to False to save the history of convergence
        """
        lr = 1 / self.smoothness
        w = np.ones_like(self.workers[0].w_opt)
        f_values = []
        grad_norms = []
        l2_distances = []
        l2_distances2 = []
        # exp_name = 'gd' + exp_name
        ws, ws_tmp = np.array([w] * self.n_workers), np.array([w] * self.n_workers)

        # fun_value = self.fun_value(w)
        # grad = self.grad(w)
        # # optim_local = np.array([self.workers[i].w_opt for i in self.workers])
        # f_values.append(fun_value)
        # grad_norms.append(np.sum(grad ** 2))
        # l2_distances.append(np.sum((ws - self.w_opt_global) ** 2))
        # l2_distances2.append(np.sum((ws - optim_local) ** 2))

        for it in range(n_iter):
            if simple:
                grad = self.grad(w)
                fun_value = self.fun_value(w)
            else:
                grad_vec = np.array(
                    [self.workers[i].grad_shift(self.workers[i].x_train, self.workers[i].y_train, ws[i])
                     for i in self.workers])
                # print(np.mean(grad), np.mean(np.mean(grad_vec, axis=0)), np.mean(grad-np.mean(grad_vec, axis=0)))
                ws_tmp = ws - lr * grad_vec
                for i in self.workers:
                    ws[i] = np.mean(ws_tmp, axis=0)
                wo = np.mean(np.array(ws), axis=0)
                # print(ws.shape, wo.shape)
                # assert 0
                fun_value = self.fun_value(wo)
                grad = np.mean(grad_vec, axis=0)
                w = copy.deepcopy(wo)

            ws = np.array([w] * self.n_workers)
            optim_local = np.array([self.workers[i].w_opt for i in self.workers])
            # ws = self.alpha * ws + (1 - self.alpha) * optim_local
            ws = np.multiply(self.alpha, ws.T).T + np.multiply(1 - self.alpha, optim_local.T).T

            if not save_memory:
                f_values.append(fun_value)
                grad_norms.append(np.sum(grad ** 2))
                l2_distances.append(np.sum((ws - self.w_opt_global) ** 2))
                l2_distances2.append(np.sum((ws - optim_local) ** 2))

            w -= lr * grad
            print('{:5d}/{:5d} Iterations: fun_value {:f}'
                  .format(it + 1, n_iter, fun_value), end='\r')
        print('')

        if not save_memory:
            f_values = np.array(f_values)
            f_values -= self.fun_value(self.w_opt_global)
            grad_norms = np.array(grad_norms)

            print(self.alpha, f_values[0], l2_distances[0])
            save_run(exp_name, self.alpha, grad_norms, f_values, l2_distances, l2_distances, self.dataset_name, self.logreg)

        return w

    def find_min(self):
        """Return the minimum of the explicit mixture."""
        print("Finding minimum of the global function with Gradient Descent...")
        return self.run_gd(max(2 * self.max_it, self.max_it + 100), True)

    def recompute_global_min(self, max_it):
        print("Recomputing the minimum of the global function with Gradient Descent...")
        self.w_opt_global = self.run_gd(max_it, True)
        return

    def sparsification(self, vec, k):
        """Return a random sparsification for the given vector.

        Arguments:
        vec -- the vector to be sparsified
        k -- 'k' in random-k compression operator, the cardinality of the set of indices returned
        """
        generator = default_rng()
        d = vec.size
        inds = generator.choice(a=np.arange(d), size=k, replace=False)

        positions = np.zeros(d)
        positions[inds] = float(d) / k
        # omega = d/k - 1
        return positions * vec

    def compression_level_sparsification(self, d, k):
        return float(d) / k - 1.0

    def compressed_grad(self, w, ks):
        compressed_grad_vec = np.array([self.sparsification(self.workers[i].grad_shift(
            self.workers[i].x_train, self.workers[i].y_train, w), ks[i])
            for i in self.workers])
        compressed_grad = np.mean(compressed_grad_vec, axis=0)
        return compressed_grad

    def run_cgd(self, ks, n_iter=100, model='str_cvx', exp_name='cgd'):
        """Run Compressed Gradient Descent.

        Arguments:
        ks -- list or numpy.array of integer numbers between 1 and self.d (both inclusively), size equal to the number of workers (self.n_workers); contains 'k's for random-k operator of each worker
        n_iter -- maximum number of iterations taken for CGD to converge (default 100)
        model -- type of functions being optimized; possible options - ['str_cvx', 'cvx'] (default str_cvx)
        """
        compression_levels = self.compression_level_sparsification(self.d, ks)
        max_L_omega_alpha_sq = max(self.l_s * self.alpha ** 2 * compression_levels)
        if model == 'str_cvx':
            lr = 1.0 / (self.smoothness + 2 * max_L_omega_alpha_sq / self.n_workers)
        elif model == 'cvx':
            lr = 0.25 / (self.smoothness + 2 * max_L_omega_alpha_sq / self.n_workers)
        else:
            raise ValueError('Parameter <model> unrecognized')
        #         w = np.ones_like(self.workers[0].w_opt) * 0.1
        w = np.ones_like(self.workers[0].w_opt)
        grad_norms = []
        f_values = []
        l2_distances = []
        # exp_name = 'cgd' + exp_name

        for it in range(n_iter):
            grad = self.grad(w)
            compressed_grad = self.compressed_grad(w, ks)
            fun_value = self.fun_value(w)
            grad_norms.append(np.sum(grad ** 2))
            f_values.append(fun_value)
            l2_distances.append(np.sum((w - self.w_opt_global) ** 2))
            w -= lr * compressed_grad
            print('{:5d}/{:5d} Iterations: fun_value {:f}'
                  .format(it + 1, n_iter, fun_value), end='\r')
        print('')

        grad_norms = np.array(grad_norms)
        f_values -= self.fun_value(self.w_opt_global)
        save_run(exp_name, self.alpha, grad_norms, f_values, l2_distances, self.dataset_name, self.logreg, ks)

        return w

    def diana_update_sparsification(self, w, h, ks, betas):
        """Return the unbiased estimate of the gradient and the table of updated memory [h_1, ..., h_n] for DIANA
        algorithm. """
        # TODO
        grad_vec = np.array([self.sparsification(
            self.workers[i].grad_shift(self.workers[i].x_train, self.workers[i].y_train, w) - h[i], ks[i])
            for i in self.workers])
        grad = np.mean(grad_vec, axis=0) + np.mean(h, axis=0)
        betas = betas[:, np.newaxis]
        h += grad_vec * betas
        return grad, h

    def run_diana_sparsification(self, ks, n_iter=100, model='str_cvx', exp_name='diana'):
        """Run DIANA algorithm.

        Arguments:
        ks -- list or numpy.array of integer numbers between 1 and self.d (both inclusively), size equal to the number of workers (self.n_workers); contains 'k's for random-k operator of each worker
        n_iter -- maximum number of iterations taken for DIANA to converge (default 100)
        model -- type of functions being optimized; possible options - ['str_cvx', 'cvx'] (default str_cvx)
        """
        compression_levels = self.compression_level_sparsification(self.d, ks)

        betas = 1.0 / (compression_levels + 1)
        max_L_omega_alpha_sq = max(self.l_s * self.alpha ** 2 * compression_levels)
        max_beta_L_omega_alpha_sq = max(self.l_s * self.alpha ** 2 * compression_levels * betas)
        min_beta = min(betas)
        # min_beta = betas
        exp_name = exp_name

        if model == 'str_cvx':
            lr = 1.0 / (self.smoothness + 2 * max_L_omega_alpha_sq / self.n_workers + 4 * max_beta_L_omega_alpha_sq / (
                    self.n_workers * min_beta))
        elif model == 'cvx':
            lr = 0.25 / (self.smoothness + 2 * max_L_omega_alpha_sq / self.n_workers + 4 * max_beta_L_omega_alpha_sq / (
                    self.n_workers * min_beta))
        else:
            raise ValueError('Parameter <model> unrecognized')
        w = np.ones_like(self.workers[0].w_opt)
        h = np.zeros(shape=(self.n_workers, w.size))
        f_values = []
        grad_norms = []
        l2_distances = []

        for it in range(n_iter):
            diana_grad, h = self.diana_update_sparsification(w, h, ks, betas)
            fun_value = self.fun_value(w)

            f_values.append(fun_value)
            grad_norms.append(np.sum(self.grad(w) ** 2))
            l2_distances.append(np.sum((w - self.w_opt_global) ** 2))

            w -= lr * diana_grad
            print('{:5d}/{:5d} Iterations: fun_value {:f}'
                  .format(it + 1, n_iter, fun_value), end='\r')
        print('')

        grad_norms = np.array(grad_norms)
        f_values -= self.fun_value(self.w_opt_global)
        save_run(exp_name, self.alpha, grad_norms, f_values, l2_distances, self.dataset_name, self.logreg, ks)

        return w

    def run_scafflix(self, n_iter=100, exp_name='scafflix', p=1, optim_p=True, mu=0.1):
        """
        Run scafflix algorithm.

        Arguments:
        n_iter -- maximum number of iterations taken for scafflix to converge (default 100)
        p: communication probability
        """
        from termcolor import colored, cprint
        if optim_p:
            kappa = np.mean(self.l_s) / mu
            p = 1. / np.sqrt(kappa)
            cprint(p, 'red')

        exp_name = f'{exp_name}_{p}'
        lr = 1 / self.smoothness
        w = np.ones_like(self.workers[0].w_opt)
        h = np.zeros(shape=(self.n_workers, w.size))
        f_values = []
        grad_norms = []
        l2_distances = []
        l2_distances2 = []
        ws, ws_tmp = np.array([w] * self.n_workers), np.array([w] * self.n_workers)

        fun_value = self.fun_value(w)
        grad = self.grad(w)
        # # optim_local = np.array([self.workers[i].w_opt for i in self.workers])
        f_values.append(fun_value)
        grad_norms.append(np.sum(grad ** 2))
        # l2_distances.append(np.sum((ws - self.w_opt_global) ** 2))
        # l2_distances2.append(np.sum((ws - optim_local) ** 2))

        for it in range(n_iter):
            grad_vec = np.array([self.workers[i].grad_shift(self.workers[i].x_train, self.workers[i].y_train, ws[i]) - h[i]
                                 for i in self.workers])
            ws_tmp = ws - lr * grad_vec

            p_gen = np.random.random_sample()
            if p_gen < p:
                for i in self.workers:
                    ws[i] = np.mean(ws_tmp, axis=0)
            else:
                for i in self.workers:
                    ws[i] = ws_tmp[i]  # skip communication
            h += p / lr * (ws - ws_tmp)

            wo = np.mean(ws, axis=0)
            fun_value = self.fun_value(wo)
            grad = np.mean(grad_vec, axis=0)

            # obtain local opt
            optim_local = np.array([self.workers[i].w_opt for i in self.workers])
            # print(optim_local.shape, ws.shape)

            if p_gen < p:
                f_values.append(fun_value)
                grad_norms.append(np.sum(grad ** 2))
                l2_distances.append(np.sum((ws - self.w_opt_global) ** 2))
                l2_distances2.append(np.sum((ws-optim_local) ** 2))

            print('{:5d}/{:5d} Iterations: fun_value {:f}'
                  .format(it + 1, n_iter, fun_value), end='\r')
        print('')

        grad_norms = np.array(grad_norms)
        f_values -= self.fun_value(self.w_opt_global)

        save_run(exp_name, self.alpha, grad_norms, f_values, l2_distances, l2_distances2, self.dataset_name, self.logreg)

        cprint(f"{p} --> after", 'red')
        return w, p
