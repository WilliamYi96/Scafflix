from models import *
import unittest

class TestBaseClassMethods(unittest.TestCase):
    
    def test_validation_ratio_initialization(self):
        id_node = 'test_node'
        alpha = 0.5
        x_train = np.random.random(size = (5, 5))
        y_train = np.random.random(size = (5,))
        x_validation = np.random.random(size = (5, 5))
        y_validation = np.random.random(size = (5,))
        regularization = 0.1
        nn_flag = False
        compute_smoothness_min = False
        validation = True
        with self.assertRaises(ValueError):
            test_node = Node(id_node, alpha, x_train, y_train, regularization, nn_flag, compute_smoothness_min, validation=validation)
        val_rat = 0.2
        test_node = Node(id_node, alpha, x_train, y_train, regularization, nn_flag, compute_smoothness_min, 
                         validation=validation, 
                         val_rat=val_rat)
        self.assertEqual(len(test_node.train_indices), 4)
        self.assertEqual(len(test_node.validation_indices), 1)
        self.assertEqual(test_node.validation_indices[0], 4)

class TestMulticlassLogRegMethods(unittest.TestCase):

    def test_MLG_init_fail(self):
        d = 10
        n = 100
        x_train = np.random.random((n + 1, d))
        y_train = np.random.random(n)
        regularization = 0    
        with self.assertRaises(ValueError):
            test_class = MulticlassLogReg('000', 0.5, x_train, y_train, regularization)
            
    def test_MLG_flow(self):
        x_train = np.array([[0, 0, 1], 
                            [0, 1, 0], 
                            [1, 0, 0], 
                            [1, 1, 0]])
        y_train = np.random.random((4, 2))
        regularization = 0
        w = np.array([0, 0, 100, 0, 0, 100, 0, 0]) 
        tolerance = 1e-1
        test_class = MulticlassLogReg('000', 0.5, x_train, y_train, regularization, tolerance=tolerance)
        probs = test_class.get_h(test_class.x_train, w)
        probs_true = np.array([[0.5, 0.5], [0.5, 1], [1, 0.5], [1, 1]])
        self.assertEqual(test_class.d, 4)
        self.assertEqual(test_class.get_h(test_class.x_train, w).size, 4 * 2)
        self.assertAlmostEqual(((probs - probs_true) ** 2).sum(), 0)
        self.assertEqual(test_class.f_value(w), test_class.fun_value(w))
    
    def test_MLG_grad_1(self):
        x_train = np.array([[2, 3, 4]])
        y_train = np.array([[0, 0.5]])
        regularization = 0
        w = np.array([10, 0, 10, 0, 10, 0, 10, 0])
        tolerance = 1e-1
        test_class = MulticlassLogReg('test', 0.5, x_train, y_train, regularization, tolerance=tolerance)
        self.assertAlmostEqual(((test_class.g(test_class.x_train, test_class.y_train, w) - np.array([0.5, 0, 1, 0, 1.5, 0, 2, 0])) ** 2).sum(), 0)
        
            
if __name__ == '__main__':
    unittest.main()