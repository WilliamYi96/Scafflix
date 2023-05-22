from workers import *
import models
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from numpy.linalg import norm
from prep_data import DATASET_PATH
import os
import unittest

class TestMasterNodeBasic(unittest.TestCase):
    
    def test_basic_init(self):
        n_workers = 2
        alpha = 0.5
        worker = Node
        x_data = np.array([[0, 0, 0], 
                           [0, 0, 1], 
                           [0, 1, 1],
                           [1, 0, 0]])
        y_data = np.array([0, 1, 1, 0])
        dataset_name = 'toy_dataset'
        dump_svmlight_file(x_data, y_data, DATASET_PATH + dataset_name)
        logreg = False
        ordered = True
        max_it = 10
        regularization = 0.0
        compute_smoothness_min = False
        with self.assertRaises(ValueError):
            test_master = MasterNode(n_workers, -0.1, worker, dataset_name, logreg, ordered, max_it, compute_smoothness_min, regularization=regularization)
        with self.assertRaises(ValueError):
            test_master = MasterNode(n_workers, 1.1, worker, dataset_name, logreg, ordered, max_it, compute_smoothness_min, regularization=regularization)
        with self.assertRaises(ValueError):
            test_master = MasterNode(n_workers, [0.5, -0.2], worker, dataset_name, logreg, ordered, max_it, compute_smoothness_min, regularization=regularization)
        with self.assertRaises(ValueError):
            test_master = MasterNode(n_workers, [1.4, 0.3], worker, dataset_name, logreg, ordered, max_it, compute_smoothness_min, regularization=regularization)
        with self.assertRaises(ValueError):
            test_master = MasterNode(n_workers, [1.4, -0.5], worker, dataset_name, logreg, ordered, max_it, compute_smoothness_min, regularization=regularization)
        test_master = MasterNode(n_workers, alpha, worker, dataset_name, logreg, ordered, max_it, compute_smoothness_min, regularization=regularization)
        test_master = MasterNode(n_workers, alpha, worker, dataset_name, logreg, ordered, max_it, compute_smoothness_min, validation=True,
                                val_rat = 0.5, 
                                regularization=regularization)
        self.assertEqual(len(test_master.workers[0].train_indices), 1)
        self.assertEqual(len(test_master.workers[1].train_indices), 1)        
        if os.path.exists(DATASET_PATH + dataset_name):
            os.remove(DATASET_PATH + dataset_name)
        else:
            raise RuntimeError('Dataset does not exist')

class TestMasterNodeUnifiedFunctions(unittest.TestCase):
    
    def setUp(self):
        n_workers = 8
        data_per_worker = 20
        sampling_size = 100
        alpha = 0.5
        worker = models.LogReg
        rng = default_rng()
        x_data = rng.random(size=(n_workers * data_per_worker, 10))
        y_data = rng.integers(low=0, high=1, size=n_workers * data_per_worker, endpoint=True)
        dataset_name = 'toy_dataset'
        dump_svmlight_file(x_data, y_data, DATASET_PATH + dataset_name)
        logreg = True
        ordered = True
        max_it = 10
        regularization = 0.0
        compute_smoothness_min = False
        validation = True
        val_rat = 0.8
        self.test_master = MasterNode(n_workers, 
                                      alpha, 
                                      worker, 
                                      dataset_name, 
                                      logreg, 
                                      ordered, 
                                      max_it, 
                                      compute_smoothness_min, 
                                      regularization=regularization, 
                                      validation=validation, 
                                      val_rat=val_rat)
        if os.path.exists(DATASET_PATH + dataset_name):
            os.remove(DATASET_PATH + dataset_name)
        else:
            raise RuntimeError('Dataset does not exist') 
        
    def tearDown(self):
        del self.test_master
                
    def test_model_grad(self):
        self.assertIsNotNone(self.test_master.validation_ratio)
        self.assertIsNotNone(self.test_master.workers[0].val_rat)
        for i in range(self.test_master.n_workers):
            self.test_master.workers[i].w_opt = np.zeros(self.test_master.d)
        with self.assertRaises(NameError):
            self.test_master.model_grad(model='explicit_mixture', w=None)
        weights = w=np.ones(self.test_master.d)
        self.test_master.model_grad(model='fomaml', w=weights)
        self.test_master.model_grad(model='reptile', w=weights, joint_dataset=True)
        self.test_master.model_grad(model='expmix', w=weights)
        with self.assertWarns(DeprecationWarning):
            self.test_master.stochastic_grad(weights)
        with self.assertWarns(DeprecationWarning):
            self.test_master.modified_stochastic_grad(weights)
        with self.assertWarns(DeprecationWarning):
            self.test_master.fomaml_grad(weights)
        with self.assertWarns(DeprecationWarning):
            self.test_master.reptile_update(weights)        

        
    def test_learning_errors_warnings(self):
        with self.assertRaises(ValueError):
            self.test_master.learning(epochs=None, n_iter=None)
        with self.assertRaises(NameError):
            self.test_master.learning(model='explicit mixture')  
        with self.assertWarns(DeprecationWarning):
            self.test_master.sgd_mixed()
        with self.assertWarns(DeprecationWarning):
            self.test_master.sgd_mixed_modified()
        with self.assertWarns(DeprecationWarning):
            self.test_master.fomaml()
        with self.assertWarns(DeprecationWarning):
            self.test_master.reptile()
            
    def test_learning(self):
        for i in range(self.test_master.n_workers):
            self.test_master.workers[i].w_opt = np.zeros(self.test_master.d)        
        for model in ['expmix', 'fomaml', 'reptile']:
            self.test_master.learning(model=model)
        
            
if __name__ == '__main__':
    unittest.main()