import numpy as np
from numpy.random import default_rng
from scipy.special import expit
import numpy.linalg as la
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack
from prep_data import number_of_features
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import warnings
import copy

MAX_VALIDATION_NOT_DECREASING = 200



class Node:
    def __init__(self, id_node, alpha, x_train, y_train, regularization, nn_flag=False, compute_smoothness_min=True, **args):
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError('number of rows in x_train ({}) and y_train ({}) must be equal'.format(x_train.shape[0], y_train.shape[0]))
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError('parameter alpha must be between 0 and 1')
        
        self.id = id_node
        self.alpha = alpha
        
        if not nn_flag:
            self.x_train = x_train
            self.y_train = y_train
            self.tolerance = args.get('tolerance', 1e-6)
            self.validation = args.get('validation')
            self.val_rat = args.get('val_rat')
            self.n = self.x_train.shape[0]                
            if self.validation is True:
                if self.val_rat is None:
                    self.x_validation = args.get('x_validation')
                    self.y_validation = args.get('y_validation')
                    if self.x_validation is None:
                        raise ValueError('Validation dataset is not provided. Either provide it, or set val_rat to some value')
                else:
                    assert self.val_rat > 0.0
                    assert self.val_rat < 1.0
                    train_size = int((1 - self.val_rat) * self.n)
                    self.train_indices = np.arange(train_size)
                    self.validation_indices = np.arange(train_size, self.n)
            self._add_intercept()
            self.d = self.x_train.shape[1]
            if len(self.y_train.shape) == 1:
                self.d_y = 1
            else:
                self.d_y = self.y_train.shape[1]
            
        self.reg = regularization

        
        if compute_smoothness_min:
            if not nn_flag:
                self.smoothness = self._smoothness()
                self.w_opt = self.find_min()
            else:
                self.smoothness = None
                self.w_opt = self.find_min()
        else:
            self.smoothness = None
            self.w_opt = None
            
    def change_alpha(self, alpha):
        self.alpha = alpha

    def _add_intercept(self):
        ones = np.ones((self.x_train.shape[0], 1))
        if type(self.x_train) == np.ndarray:
            self.x_train = np.hstack([ones, self.x_train])
        else: 
            if type(self.x_train) == coo_matrix:
                fmt = 'coo'
            elif type(self.x_train) == csr_matrix:
                fmt = 'csr'
            self.x_train = hstack([ones, self.x_train], format=fmt)
            
        if self.validation is True and self.val_rat is None:
            # if validation dataset is provided
            ones = np.ones((self.x_validation.shape[0], 1))
            if type(self.x_validation) == np.ndarray:
                self.x_validation = np.hstack([ones, self.x_validation])
            else:     
                if type(self.x_train) == coo_matrix:
                    fmt = 'coo'
                elif type(self.x_train) == csr_matrix:
                    fmt = 'csr'
                self.x_validation = hstack([ones, self.x_validation], format=fmt)        

    def f_value(self, w):
        raise NotImplementedError

    def fun_value(self, w):
        return NotImplementedError
    
    def f_value_general(self, x, y, w):
        return NotImplementedError
    
    def fun_value_general(self, x, y, w):
        return NotImplementedError        

    def g(self, x, y, w):
        raise NotImplementedError

    def grad(self, x, y, w):
        return NotImplementedError

    def grad_shift(self, x, y, w):
        grad_shift = self.alpha * self.grad(x, y, self.compute_local(w))
        return grad_shift

    def grad_local2(self, x, y, w):
        grad_local = self.grad(x, y, self.compute_local(w))
        return grad_local, self.compute_local(w)

    def fun_value_shift(self, w):
        f_val_shift = self.fun_value(self.compute_local(w))
        return f_val_shift

    def _smoothness(self, **args):
        raise NotImplementedError

    def find_min(self, **args):
        raise NotImplementedError

    def compute_local(self, w):
        return self.alpha * w + (1 - self.alpha) * self.w_opt
    
    def compute_local_modified(self, local_w: np.ndarray, global_w: np.ndarray):
        """Compute linear combination of weights for Modified Explicit Mixture Algorithm."""
        return self.alpha * global_w + (1 - self.alpha) * local_w
    
    def iterate_size(self):
        return NotImplementedError
    
    def local_steps(self, w: np.ndarray, stepsize: float, k: int, x=None, y=None) -> np.ndarray:
        """Perform local steps. 
        
        Arguments:
        w -- initial vector
        stepsize -- stepsize for Gradient Descent
        k -- number of gradient steps
        x -- features dataset; if None, self.x_train is used (default None)
        y -- labels; if None, self.y_train is used (default None)
        """
        assert stepsize > 0
        if x is None:
            x = self.x_train
        if y is None:
            y = self.y_train

        weights = copy.deepcopy(w)
        for _ in range(k):
            weights -= stepsize * self.grad(x, y, weights)
        return weights    

    def fomaml_grad(self, x, y, w, k, inner_loop_lr):
        """Return FOMAML with k SGD steps gradient. Deprecated."""
        weights = w
        for i in range(k):
            weights -= inner_loop_lr * self.grad(x, y, weights)
        return self.grad(x, y, weights)
    
    def reptile_update(self, x, y, w, k, inner_loop_lr):
        """Return Reptile update after k inner steps. Deprecated."""
        weights = copy.deepcopy(w)
        initial_weights = copy.deepcopy(w)
        for i in range(k):
             weights -= inner_loop_lr * self.grad(x, y, weights)
        print((((weights - initial_weights) / inner_loop_lr / k) ** 2).sum())
        return (weights - initial_weights) / inner_loop_lr / k
    
    def model_stochastic_update(self, model: str, *args, **kwargs):
        """Return a stochastic update depending on model. 
        
        E.g., if model is 'expmix', the function returns explicit_mixture_stochastic_grad(*args, **kwargs).
        Arguments:
        model -- model for which the update is computed; available values are 'expmix' for Explicit Mixture, 'modexpmix' for Modified Explicit Mixture, 'fomaml' for FOMAML, and 'reptile' for Reptile
        """
        if model == 'expmix':
            return self.explicit_mixture_stochastic_grad(*args, **kwargs)
        if model == 'modexpmix':
            return self.modified_explicit_mixture_stochastic_gradient(*args, **kwargs)
        if model == 'fomaml':
            return self.fomaml_stochastic_grad(*args, **kwargs)
        if model == 'reptile':
            return self.reptile_stochastic_update(*args, **kwargs)
        
        raise NameError('Unknown model {}. Available models are expmix, modexpmix, fomaml and reptile.'.format(model))

    def explicit_mixture_stochastic_grad(self, w: np.ndarray, batch='full'):
        if batch == 'full':
            return self.grad_shift(self.x_train, self.y_train, w)
        else:
            generator = default_rng()
            choices = generator.choice(self.n, size=batch, replace=False)
            return self.grad_shift(self.x_train[choices], self.y_train[choices], w)
        
    def fomaml_stochastic_grad(self, w, k, inner_loop_lr, batch='full'):
        assert self.validation 
        assert self.val_rat is not None or (self.x_validation is not None & self.y_validation is not None)
        weights = copy.deepcopy(w)
        if batch == 'full':
            if self.val_rat is None:
                x = self.x_train
                y = self.y_train
            else:
                x = self.x_train[self.train_indices]
                y = self.y_train[self.train_indices]
            for i in range(k):
                weights -= inner_loop_lr * self.grad(x, y, weights)        
        else:
            generator = default_rng()
            for i in range(k):
                if self.val_rat is None:
                    inds = self.n
                else:
                    inds = self.train_indices
                choices = generator.choice(inds, size=batch, replace=False)
                x = self.x_train[choices]
                y = self.y_train[choices]
                weights -= inner_loop_lr * self.grad(x, y, weights)
#         if (torch.isnan(self.x_train).any() is None or torch.isnan(self.y_train).any() is None):
#             warnings.warn("fomaml_stochastic_grad; None value is encountered.")
#         print("FOMAML stoch gradient: {}".format(self.fomaml_grad(x, y, w, k, inner_loop_lr)))
        if self.val_rat is None:
            return self.grad(self.x_validation, self.y_validation, weights)
        else:
            return self.grad(self.x_train[self.validation_indices], self.y_train[self.validation_indices], weights)


    def reptile_stochastic_update(self, w: np.ndarray, k: int, inner_loop_lr: float, batch: [int, 'full']='full', joint_dataset: bool=False):
        """Compute a stochastic Reptile update. 
        
        Stochasticity comes from random data selection of size 'batch'. Arguments:
        w -- point to estimate the gradient at
        k -- number of inner steps
        inner_loop_lr -- inner loop learning step size
        batch -- batch size
        joint_dataset -- if True, data is selected from joint train and validation dataset, otherwise, only from train dataset
        """
        if joint_dataset and (self.val_rat is None):
            warnings.warn('Parameter <joint_dataset> is set to True, although validation dataset is provided as a separate object. Copyings slow down the computation.')
        weights = copy.deepcopy(w)
        initial_weights = copy.deepcopy(w)
        
        assert joint_dataset is not None
        
        if joint_dataset:
            assert self.validation is True

        if joint_dataset and self.val_rat is None:
            joint_x = copy.deepcopy(self.x_train)
            joint_y = copy.deepcopy(self.y_train)
            if joint_dataset:
                assert self.validation
                if type(self.x_train) == np.ndarray:
                    joint_x = np.concatenate((joint_x, self.x_validation))
                    joint_y = np.concatenate((joint_y, self.y_validation))
                elif type(self.x_train) == torch.Tensor:
                    joint_x = torch.cat((joint_x, self.x_validation))
                    joint_y = torch.cat((joint_y, self.y_validation))
                elif type(self.x_train) == coo_matrix:
                    joint_x = vstack([joint_x, self.x_validation])
                    joint_y = np.concatenate((joint_y, self.y_validation))
                else:
                    raise TypeError('{}: unexpected type of the dataset'.format(reptile_stochastic_update.__name__))
        else:
            # depending on self.val_rat the function chooses points either from all rows of joint_x and joint_y or from self.train_indices
            joint_x = self.x_train
            joint_y = self.y_train
        generator = default_rng()
        
        for i in range(k):
            if batch == 'full':
                x = joint_x
                y = joint_y
            else:
                if joint_dataset is False and self.val_rat is not None:
                    assert self.validation is True
                    arr = self.train_indices
                else:
                    arr = joint_x.shape[0]
                choices = generator.choice(arr, size=batch, replace=False)
                if type(joint_x) == coo_matrix:
                    warnings.warn('Features data type <coo_matrix> does not support indexing. Computations slow donw.')
                    x = joint_x.toarray()[choices]
                else:
                    x = joint_x[choices]
                y = joint_y[choices]
            weights -= inner_loop_lr * self.grad(x, y, weights)
        return (weights - initial_weights) / inner_loop_lr / k 

    def modified_explicit_mixture_stochastic_gradient(self, w: np.ndarray, k: int, inner_loop_lr: float, batch='full'):
        assert self.validation # output gradient is computed on validation dataset
        weights = copy.deepcopy(w)
        initial_weights = copy.deepcopy(w) # initial weights imitate global mixture
        def modified_grad(x, y, local_weights):
            return (1 - self.alpha) * self.grad(x, y, self.compute_local_modified(local_weights, initial_weights))
        generator = default_rng()
        for i in range(k):
            if batch == 'full':
                if self.val_rat is None:
                    x = self.x_train
                    y = self.y_train
                else:
                    x = self.x_train[self.train_indices]
                    y = self.y_train[self.train_indices]
            else:
                if self.val_rat is None:
                    inds = self.n
                else:
                    inds = self.train_indices                
                choices = generator.choice(inds, size=batch, replace=False)
                if type(self.x_train) == coo_matrix:
                    warnings.warn('Features data type <coo_matrix> does not support indexing. Computations slow donw.')
                    x = self.x_train.toarray()[choices]
                else:
                    x = self.x_train[choices]
                y = self.y_train[choices]
            weights -= inner_loop_lr * modified_grad(x, y, weights)
        if self.val_rat is None:
            return self.grad(self.x_validation, self.y_validation, self.compute_local_modified(weights, initial_weights))
        else:
            return self.grad(self.x_train[self.validation_indices], self.y_train[self.validation_indices], self.compute_local_modified(weights, initial_weights))
        

class LogReg(Node):
    def f_value(self, w):
        h = self.get_h(self.x_train, w)
        y = self.y_train
        zeros = np.zeros_like(h)
        return -(np.where(y == 1, np.log(h), zeros) + np.where(y == 0, np.log(1-h), zeros)).mean()

    def g(self, x, y, w):
        h = self.get_h(x, w)
        return x.T.dot(h - y) / y.shape[0]

    def fun_value(self, w):
        return self.f_value(w) + self.reg * np.sum(w ** 2)/2
    
    def fun_value_general(self, x, y, w):
        return self.f_value_general(x, y, w) + self.reg * np.sum(w ** 2)/2
    
    def grad(self, x, y, w):
        return self.g(x, y, w) + self.reg * w

    def _smoothness(self):
        xtx = self.x_train.T.dot(self.x_train)
        if type(xtx) == coo_matrix or type(xtx) == csr_matrix:
            xtx = xtx.toarray()
        return np.max(la.eigvalsh(xtx + self.reg * np.eye(self.d))) / (4 * self.n)

    def find_min(self):
        max_it = 100000
        print("Worker(logreg) {} tolerance {}".format(self.id, self.tolerance))
        w = np.zeros(self.x_train.shape[1])
        lr = 1/self.smoothness
        print("Learning rate is {}".format(lr))
        for it in range(max_it):
            grad = self.grad(self.x_train, self.y_train, w)
            w -= lr * grad
            print('{:5d}/{:5d} Iterations: fun_value {:f} norm of the gradient {}'
                  .format(it+1, max_it, self.fun_value(w), la.norm(grad)), end='\r')
            if la.norm(grad) < self.tolerance:
                break
        print('')
        return w

    def get_h(self, x, w):
        z = x.dot(w)
        h = self._sigmoid(z)
        return h

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1 + np.exp(-x))
    
    def iterate_size(self):
        return self.d
    
    
class MulticlassLogReg(LogReg):
    def f_value(self, w):
        return self.f_value_general(self.x_train, self.y_train, w)
        
    def f_value_general(self, x, y, w):
        h = self.get_h(x, w).reshape((-1))
        y = y.reshape((-1))
        zeros = np.zeros_like(h)
        return -(np.where(y == 1, np.log(h), zeros) + np.where(y == 0, np.log(1-h), zeros)).mean()
        
    def g(self, x, y, w):
        h = self.get_h(x, w)
        assert y.ndim == 2
        assert y.shape == h.shape
        return (x.T.dot(h - y)).reshape((-1)) / y.size            
        
    def get_h(self, x, w):
        """Return probability of belonging to each class."""
        z = x @ w.reshape((self.d, -1))
        h = self._sigmoid(z)
        return h
    
    def find_min(self):
        max_it = 100000
        print("Worker(logreg) {} tolerance {}".format(self.id, self.tolerance))
        w = np.zeros(self.d * self.y_train.shape[1])
        lr = 1 / self.smoothness
        print("Learning rate is {}".format(lr))
        for it in range(max_it):
            grad = self.grad(self.x_train, self.y_train, w)
            w -= lr * grad
            print('{:5d}/{:5d} Iterations: fun_value {:f} norm of the gradient {}'
                  .format(it+1, max_it, self.fun_value(w), la.norm(grad)), end='\r')
            if la.norm(grad) < self.tolerance:
                break
        print('')
        return w
    
    def iterate_size(self):
        return self.d * self.y_train.shape[1]



class LogRegNoncvx(Node):   
    """
    Implement logistic regression function with a nonconvex regularizer.
    
    See Tran-Dinh et al. "Hybrid Stochastic Gradient Descent Algorithms for Stochastic Nonconvex Optimization"
    """
    
    def f_value(self, w):
        h = self.get_h(self.x_train, w)
        y = self.y_train
        zeros = np.zeros_like(h)
        return -(np.where(y == 1, np.log(h), zeros) + np.where(y == 0, np.log(1-h), zeros)).mean() 
    
    def fun_value(self, w):
        return self.f_value(w) - self.reg * ( 1 / (1 + w ** 2) ).sum() + self.d * self.reg

    def g(self, x, y, w):
        h = self.get_h(x, w)
        return x.T.dot(h - y) / y.shape[0]
    
    def grad(self, x, y, w):
        return self.g(x, y, w) + 2 * self.reg * w * (1 / (1 + w ** 2) ** 2)

    def _smoothness(self):
        xtx = self.x_train.T.dot(self.x_train)
        # the hessian of the regularizer is bounded above by 2 * self.reg * I
        return np.max(la.eigvalsh(xtx.toarray())) / (4 * self.n) + 2 * self.reg

    def find_min(self):
        max_it = 100000
        w = np.zeros(self.x_train.shape[1])
        lr = 1/self.smoothness
        for it in range(max_it):
            grad = self.grad(self.x_train, self.y_train, w)
            w -= lr * grad
#             print('{:5d}/{:5d} Iterations: fun_value {:f}'
#                   .format(it+1, max_it, self.fun_value(w)), end='\r')
            if la.norm(grad) < self.tolerance:
                break
#         print('')
        return w

    def get_h(self, x, w):
        z = x.dot(w)
        h = self._sigmoid(z)
        return h

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def iterate_size(self):
        return self.d
    

class LinReg(Node):
    def f_value(self, w):
        z = self.x_train.dot(w) - self.y_train
        return 1/2 * np.mean(z**2)

    def g(self, x, y, w):
        xtx = x.T.dot(x)
        xty = x.T.dot(y)
        return (xtx.dot(w) - xty) / y.shape[0]
    
    def grad(self, x, y, w):
        return self.g(x, y, w) + self.reg * w
    
    def fun_value(self, w):
        return self.f_value(w) + self.reg * np.sum(w ** 2)/2

    def _smoothness(self):
        xtx = self.x_train.T.dot(self.x_train)
        return np.max(la.eigvalsh(xtx.toarray() + self.reg * np.eye(self.d))) / self.n

    def find_min(self):
        xtx = self.x_train.T.dot(self.x_train)
        xty = self.x_train.T.dot(self.y_train)
        w = la.solve(xtx.toarray() + self.reg * np.eye(self.d), xty)
#         print('The exact formula for minimum computed. Fun. value: {:f}'
#               .format(self.fun_value(w)))
        return w
    
    def iterate_size(self):
        return self.d

class NN_2layer_regression(nn.Module):
    def __init__(self, input_dim, interm_dim1, interm_dim2):
        super().__init__()
        
        self.d = input_dim
        self.interm_dim1 = interm_dim1
        self.interm_dim2 = interm_dim2
        
        self.fc1 = nn.Linear(input_dim, interm_dim1)
        self.fc2 = nn.Linear(interm_dim1, interm_dim2)
        self.fc3 = nn.Linear(interm_dim2, 1)
        
        self.modules_sizes = [self.d * self.interm_dim1, 
                              self.interm_dim1, 
                              self.interm_dim1 * self.interm_dim2, 
                              self.interm_dim2, 
                              self.interm_dim2, 
                              1]
                
        self.w_opt = None
        self.alpha = None
        self.mixed_linead_weights = None
        
    def set_mixed_linear_weights(self, w):
        self.mixed_linead_weights = self.alpha * w + (1 - self.alpha) * self.w_opt
        self.mixed_linead_weights.retain_grad()
        fc_parameters = torch.split(self.mixed_linead_weights, self.modules_sizes)
        ind = 0
        for module in self.modules():
            if type(module) == nn.Linear:
                    module.weight = torch.nn.Parameter(fc_parameters[ind].view(module.weight.shape))
                    ind += 1
                    module.bias = torch.nn.Parameter(fc_parameters[ind].view(module.bias.shape))
                    ind += 1
        
        
    def forward(self, x, w=None):
        if w is not None:
            assert w.requires_grad
            assert self.alpha is not None
            assert self.w_opt is not None
            self.set_mixed_linear_weights(w)
            
        
        out = torch.tanh(self.fc1(x))
        out = torch.tanh(self.fc2(out))
        out = self.fc3(out)   
        return out


class NN_1d_regression(Node):
    def __init__(self, id_node, alpha, x_train, y_train, tolerance, compute_smoothness_min=True, validation=False, x_validation=None, y_validation=None, gpu=False, stepsize=1e-2, interm_dim1=40, interm_dim2=40):
        if gpu == True:
            assert torch.cuda.is_available() == True 
        self.interm_dim1 = interm_dim1
        self.interm_dim2 = interm_dim2
        self.stepsize = stepsize
        
        self.tolerance = tolerance
                
        self.n, self.d = x_train.shape
        
        self.separators = np.array([self.d * self.interm_dim1, # fc1.weight
                   (self.d + 1) * self.interm_dim1, # fc1.bias
                   (self.d + 1) * self.interm_dim1 + self.interm_dim1 * self.interm_dim2, # fc2.weight
                   (self.d + 1) * self.interm_dim1 + (self.interm_dim1 + 1) * self.interm_dim2, # fc2.bias
                   (self.d + 1) * self.interm_dim1 + (self.interm_dim1 + 2) * self.interm_dim2], dtype=np.int) # fc3.weight


        
        self.model = NN_2layer_regression(self.d, self.interm_dim1, self.interm_dim2)
        self.x_train = torch.from_numpy(x_train).float()
        self.y_train = torch.from_numpy(np.array(y_train)).view(-1, 1).float()
        self.memory = None # self._smoothness() for this class computes optimal point which is preserved in self.memory and passed in self.find_min()
        
        self.validation = validation
        self.x_validation = torch.from_numpy(x_validation).float() if validation else None
        self.y_validation = torch.from_numpy(np.array(y_validation)).view(-1, 1).float() if validation else None
        
        if gpu:
            self.model = self.model.to('cuda')
            self.x_train = self.x_train.to('cuda')
            self.y_train = self.y_train.to('cuda')
            self.x_validation = self.x_validation.to('cuda')
            self.y_validation = self.y_validation.to('cuda') 
            
        self.gpu = gpu
        
        super().__init__(id_node, alpha, x_train, y_train, 0, True, compute_smoothness_min)           
        
        
        
    def set_weights(self, w):
        fc_parameters = np.split(w, self.separators)
        
        device = torch.device('cuda') if self.gpu else torch.device('cpu')
        
        ind = 0
        with torch.no_grad():
            for module in self.model.modules():
                if type(module) == nn.Linear:
#                     print("fc_parameters[{}] shape = {}".format(ind, fc_parameters[ind].shape))
#                     print("weight_shape_to_transform = {}".format(module.weight.shape))
                    module.weight = torch.nn.Parameter(torch.from_numpy(fc_parameters[ind]).float().view(module.weight.shape).to(device))
                    ind += 1
#                     print("fc_parameters[{}] shape = {}".format(ind, fc_parameters[ind].shape))
#                     print("bias_shape_to_transform = {}".format(module.bias.shape))
                    module.bias = torch.nn.Parameter(torch.from_numpy(fc_parameters[ind]).float().view(module.bias.shape).to(device))
                    ind += 1
                    
    def get_weights(self):
        fc_parameters = []
        with torch.no_grad():
            for module in self.model.modules():
                if type(module) == nn.Linear:
                    fc_parameters.append(module.weight.data.clone().detach().view(-1).cpu().numpy())
                    fc_parameters.append(module.bias.data.clone().detach().view(-1).cpu().numpy())                
        return np.concatenate(fc_parameters)
        
        
    def iterate_size(self):
        return (self.d + 1) * self.interm_dim1 + (self.interm_dim1 + 1) * self.interm_dim2 + self.interm_dim2 + 1
        
    def f_value(self, w):
        self.set_weights(w)
        criterion = nn.MSELoss()
        return criterion(self.model(self.x_train), self.y_train).detach().numpy()
    
    def fun_value(self, w):
        return self.f_value(w)
    
    def g(self, x, y, w):
        self.set_weights(w)
        self.model.zero_grad()
        criterion = nn.MSELoss()
        mse = criterion(self.model(x), y)
        mse.backward()
        g = []
        ind = 0
        
        for module in self.model.modules():
            if type(module) == nn.Linear:
                g.append(module.weight.grad.clone().detach().numpy().flatten()) # append weight gradient
                g.append(module.bias.grad.clone().detach().numpy().flatten()) # append bias gradient
                
        g = np.concatenate(g)  
        return g
    
    def grad(self, x, y, w):
        return self.g(x, y, w)
    
#     def _smoothness(self):
#         if not self.gpu:
#             return self._smoothness_non_gpu()
#         return None
     
#     def _smoothness_non_gpu(self):
#         print('Computing Lipschitz smoothness constant...')
#         L = 0.1
#         max_L = 0.1
#         max_it = 60000
#         w = np.random.randn(self.iterate_size())
#         tol = self.tolerance
#         max_L_constant = 0.1 * 2 ** 40
#         grad_norm = None
#         min_f_value = float('Inf')
#         min_f_value_validation = float('Inf')
#         validation_not_decreasing_counter = 0
#         data_batch = int(0.7 * self.n)
#         generator = default_rng()
        
#         for it in range(max_it):
#             grad = self.grad(self.x_train, self.y_train, w)
#             L = 0.1
#             curr_fun_value = self.fun_value(w)
            
#             while True:
#                 if L > max_L_constant: # if L becomes too large, jump to another random point w
#                     w = np.random.randn(self.iterate_size())
#                     grad = self.grad(self.x_train, self.y_train, w)
#                     L = 0.1
#                     curr_fun_value = self.fun_value(w)

#                 print('Current L = {:f}'.format(L), end='\r')
                    
#                 f_value_ = self.fun_value(w - grad / L)
#                 if curr_fun_value - f_value_ > 0:
#                     break
#                 L *= 2.0
                
#             w -= grad / L
#             grad_norm = la.norm(grad)
            
#             if not self.validation:

#                 if f_value_ < min_f_value:
#                     min_f_value = f_value_
#                     self.memory = w

#                 if max_L < L:
#                     max_L = L
#                 print('                               {:5d}/{:5d} Iterations: fun_value {:f} grad_norm {:f}'.format(it+1, max_it, f_value_, grad_norm), end='\r')                
#                 if grad_norm < tol and f_value_ < tol ** 2:
#                       break
            
#             else:
                
#                 self.set_weights(w)
#                 criterion = nn.MSELoss()
#                 f_value_validation = criterion(self.model(self.x_validation), self.y_validation).detach().numpy()
#                 if f_value_validation < min_f_value_validation:
#                     min_f_value_validation = f_value_validation
#                     self.memory = w
#                 else:
#                     validation_not_decreasing_counter += 1
                    
#                 if validation_not_decreasing_counter >= MAX_VALIDATION_NOT_DECREASING:
#                     break
                
#                 print('                               {:5d}/{:5d} Iterations: fun_value {:f} grad_norm {:f} fun_value_on_validation {:f}'.format(it+1, max_it, f_value_, grad_norm, f_value_validation), end='\r')                
#         print('')
#         print('Worker {} smoothness constant: {}'.format(self.id, max_L))   
#         return max_L
        
    
    def find_min(self):
        print('Finding minimum of the local model. Worker id is {}'.format(self.id))
        max_it = 60000
        w = np.random.randn(self.iterate_size())
        best_w = copy.deepcopy(w)
        grad_norm = None
        fun_value = None
        min_f_value_validation = float('Inf')
        validation_not_decreasing_counter = 0
        criterion = nn.MSELoss()
        generator = default_rng()
        data_batch = int(0.7 * self.n) # hardcoded
        
        for it in range(max_it):
            fun_value = self.fun_value(w)
            choices = generator.choice(self.n, size=data_batch, replace=False)
            x = self.x_train[choices]
            y = self.y_train[choices]
            grad = self.grad(x, y, w)
            w -= self.stepsize * grad
            
            if self.validation:
                self.set_weights(w)
                out_val = self.model(self.x_validation)
                loss_val = criterion(out_val, self.y_validation).to('cpu').item()
                if loss_val < min_f_value_validation:
                    min_f_value_validation = loss_val
                    validation_not_decreasing_counter = 0
                    best_w = copy.deepcopy(w)
                else:
                    validation_not_decreasing_counter += 1
                    
                print('{:5d}/{:5d} Iterations: fun_value {:f} fun_value_on_validation {:f}'.format(
                    it + 1, max_it, fun_value, loss_val), end='\r', flush=True)
                    
                if validation_not_decreasing_counter >= MAX_VALIDATION_NOT_DECREASING:
                    break
                    
            else:
                if la.norm(self.grad(self.x_train, self.y_train, w)) < self.tolerance and self.fun_value(w) < self.tolerance ** 2:
                    return w
                print('{:5d}/{:5d} Iterations: fun_value {:f}'.format(it+1, max_it, fun_value, end='\r', flush=True))
        return best_w