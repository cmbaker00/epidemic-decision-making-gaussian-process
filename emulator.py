import models.epi_models_basic as epi
import random
import scipy.stats as stats
import numpy.random as np_rand
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, RationalQuadratic
import matplotlib.pyplot as plt

class Emulator:
    def __init__(self, model, parameters_range, name):
        self.model = model
        self.parameters_range = parameters_range
        self.name = name
        # kernel = ConstantKernel(.10, (1e-3, 1e3))\
        #          * RBF(10, (1e-5, 1e5))
        kernel = RBF()
        kernel = 1.0 * RBF(length_scale=.35,
                           length_scale_bounds=(1e-1, 10.0))
        # kernel = 1.0 * RationalQuadratic(length_scale=10, alpha=1)
        self.gp = GaussianProcessRegressor(kernel=kernel)


    def gen_parameters(self):
        parameters = {}
        for key, attribute in self.parameters_range.items():
            distribution_type = attribute['type']
            if distribution_type == 'point':
                value = attribute['value']
            elif distribution_type == 'uniform':
                value_min, value_max = attribute['value']
                value = random.uniform(value_min, value_max)
            elif distribution_type == 'gamma':
                # k shape, theta scale
                # mean = k*theta
                # var = k theta**2
                # theta = var/mean
                # k = mean/theta
                value_mean, value_var = attribute['value']
                theta = value_var/value_mean
                k = value_mean/theta
                value = np_rand.gamma(shape=k, scale=theta)
            elif distribution_type == 'normal':
                value_mean, value_var = attribute['value']
                value_std = np.sqrt(value_var)
                value = np_rand.normal(loc=value_mean,
                                       scale=value_std)
            else:
                raise ValueError("Parameter input type {} is not valid"
                                 .format(distribution_type))
            parameters[key] = value
        return parameters

    def run_model(self, parameters):
        print(self.model(parameters))


    def fit_gp(self, x, y):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        self.gp.fit(x, y)

    def predict_gp(self, x, return_std=True):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        return self.gp.predict(x, return_std=return_std)


    def plot_1d(self, x, y_pred, y_std, show_plot = True):
        plt.figure()
        plt.plot(x, y_pred, 'b-', label='Prediction')
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([y_pred - 1.9600 * y_std,
                                 (y_pred + 1.9600 * y_std)[::-1]]),
                 alpha=.5, fc='b', ec='None', label='95% confidence interval')
        if show_plot:
            plt.show()


if __name__ == "__main__":
    # emulator_test = Emulator(
    #     model=epi.run_sir_model,
    #     parameters_range={
    #         'beta': {'value': [0.005, 0.00002], 'type': 'gamma'},
    #         'gamma': {'value': [1, .05], 'type': 'normal'},
    #         'initial_condition': {'value': [999, 1, 0], 'type': 'point'}
    #                       },
    #     name='epi_SIR_test'
    # )
    # for i in range(3):
    #     params = emulator_test.gen_parameters()
    #     print(params)
    #     emulator_test.run_model(params)
    x_data_test = np.array([1,1.5,2,3,4,5])
    y_data_test = np.array([0,4  ,5,0,3,2])
    em = Emulator(None, None, None)
    em.fit_gp(x_data_test, y_data_test)
    xv = np.arange(1, 5, .01)
    yv, ystd = em.predict_gp(xv)
    em.plot_1d(xv, yv, ystd)
    print(em.gp.get_params())