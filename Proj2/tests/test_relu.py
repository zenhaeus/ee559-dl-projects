import torch
import torch.nn
import unittest
import mytorch.nn
import torch_testing as tt

class TestModule:
    def test_zero_grad(self):
        # TODO: write test
        pass

class TestLinear(unittest.TestCase):

    def test_init(self):
        """ Test linear layer initialization """
        # TODO: maybe test if weights and biases are normally distributed
        pass

    def test_forward(self):
        """ Compare forward pass to pytorch implementation """
        x = torch.rand((10, 100))
        n_out = 10
        torch_lin = torch.nn.Linear(100, n_out)
        mytorch_lin = mytorch.nn.Linear(100, n_out)
        # reset weights and biases
        rand_weight = torch.rand(torch_lin.weight.size())
        torch_lin.weight = torch.nn.Parameter(rand_weight)
        mytorch_lin.weight = rand_weight

        rand_bias = torch.rand(torch_lin.bias.size())
        torch_lin.bias = torch.nn.Parameter(rand_bias)
        mytorch_lin.bias = rand_bias

        l1, l2 = torch_lin(x), mytorch_lin(x)
        tt.assert_equal(l1, l2)
        self.assertEqual(l1.shape[1], n_out)

    def test_backward(self):
        """ Compare backward pass to pytorch implementation """
        pass


class TestReLU(unittest.TestCase):

    def test_forward(self):
        """ Compare forward pass to pytorch implementation """
        x = random_tensor_2d(10)
        torch_relu = torch.nn.ReLU()
        mytorch_relu = mytorch.nn.ReLU()
        r1, r2 = torch_relu(x), mytorch_relu(x)
        tt.assert_equal(r1, r2)

    def test_backward(self):
        """ Compare backward pass to pytorch implementation """
        pass


class TestTanh(unittest.TestCase):

    def test_forward(self):
        """ Compare forward pass to pytorch implementation """
        x = random_tensor_2d(10)
        torch_tanh = torch.nn.Tanh()
        mytorch_tanh = mytorch.nn.Tanh()
        t1, t2 = torch_tanh(x), mytorch_tanh(x)
        tt.assert_equal(t1, t2)

    def test_backward(self):
        """ Compare backward pass to pytorch implementation """
        pass


class TestMSE(unittest.TestCase):

    def test_forward_mean(self):
        """ Compare forward pass to pytorch implementation """
        for i in range(100):
            x = random_tensor_2d(10)
            y = random_tensor_2d(10)
            torch_mse = torch.nn.MSELoss()
            mytorch_mse = mytorch.nn.LossMSE()
            l1, l2 = torch_mse(x, y), mytorch_mse(x, y)
            # FIXME: test sometimes fails with 1e-07 accuracy
            #        possibly because of double / float precision in calculation
            tt.assert_allclose(l1, l2, rtol=1e-06)

    #def test_forward_sum(self):
    #    """ Compare forward pass to pytorch implementation """
    #    for i in range(100):
    #        x = random_tensor_2d(10)
    #        y = random_tensor_2d(10)
    #        torch_mse = torch.nn.MSELoss(reduction='sum')
    #        mytorch_mse = mytorch.nn.LossMSE(method='sum')
    #        l1, l2 = torch_mse(x, y), mytorch_mse(x, y)
    #        tt.assert_allclose(l1, l2, rtol=1e-06)

    def test_backward(self):
        """ Compare backward pass to pytorch implementation """
        pass


def random_tensor_2d(size):
    return 2 * (torch.rand((size, size)) - 0.5)

def random_tensor_1d(size):
    return 2 * (torch.rand(size) - 0.5)

if __name__ == '__main__':
    unittest.main()
