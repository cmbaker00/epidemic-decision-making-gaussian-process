import models.epi_models_basic as epi
import random
import scipy.stats as stats
from scipy.optimize import minimize, LinearConstraint
import numpy.random as np_rand
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, RationalQuadratic
import matplotlib.pyplot as plt
import os
import pandas as pd
import itertools
from functools import lru_cache


class Emulator:
    def __init__(self, model, parameters_range, name):
        self.model = model
        self.parameters_range = parameters_range
        self.name = name
        # kernel = ConstantKernel(.10, (1e-3, 1e3))\
        #          * RBF(.5, (1e-5, 1e5))
        # kernel = RBF()
        kernel = 1.0 * RBF(length_scale=.35,
                            length_scale_bounds=(1e-5, 1e5))
        # kernel = 1.0 * RationalQuadratic(length_scale=.5, alpha=.1)
        #kernel = None
        self.gp = GaussianProcessRegressor(kernel=kernel)

        self.data = self.load_previous_data()

    def data_file(self):
        return 'data/{}.csv'.format(self.name)

    def load_previous_data(self):
        data_file_name = self.data_file()
        file_exists = os.path.isfile(data_file_name)
        if file_exists:
            return pd.read_csv(data_file_name)
        else:
            print('Data file does is not found: {}'
                  .format(data_file_name))
            print('Creating new data strucutre')
            return None
            # raise FileNotFoundError('Data file not found: {}'.format(
            #     data_file_name
            # ))

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
        return self.model(parameters)

    def flatten_values(self, parameters):
        flat_list = []
        col_names = []
        for name, entry in parameters.items():
            try:
                entry_list_length = len(entry)
            except TypeError:
                flat_list.append(entry)
                col_names.append(name)
                entry_list_length = 0
            if entry_list_length > 0:
                flat_list += entry
                for i in range(entry_list_length):
                    col_names.append(name + '{}'.format(i))
        return col_names, flat_list

    def add_results_to_data_frame(self, parameters, result):
        names, values = self.flatten_values(parameters=parameters)
        data_dict = {name: value for name, value in zip(names, values)}
        data_dict[result[1]] = result[0]
        df = pd.DataFrame([data_dict])
        if self.data is None:
            self.data = df
        else:
            self.data = self.data.append(df)

    def run_save_simulation(self, parameters):
        result = self.run_model(parameters)
        self.add_results_to_data_frame(parameters, result)


    def save_results(self):
        if type(self.data) is pd.DataFrame:
            self.data.to_csv(self.data_file(), index=False)
        elif self.data is None:
            raise ValueError('There is no data to store')
#todo - need some code to go between the data frame and the GP code

    def optimise_gp(self, x, y):
        initial_guess = np.array([.5, .5])
        constraints = LinearConstraint(np.eye(2), 0, 10)
        min_sol = minimize(lambda z: -self.gp_likelihood(x, y, z[0], z[1]),
                           initial_guess, constraints=constraints)

    def gp_likelihood(self, x, y, h, alpha):
        self.set_gp(h=h, alpha=alpha)
        self.fit_gp(x, y)
        like_value = self.gp.log_marginal_likelihood_value_
        # yv, y_std = self.predict_gp(self.generate_test_x_mesh(x))
        # print(np.mean(y_std))
        return like_value

    def generate_test_x_mesh(self, x):
        return np.linspace(min(x), max(x), 100)

    def set_gp(self, h=.3, alpha=1.):
        kernel = RBF(length_scale=h, length_scale_bounds=(1e-5, 1e5)) + \
                 RBF(length_scale=h*5, length_scale_bounds=(1e-5, 1e5)) + \
                 ConstantKernel()
        gp = GaussianProcessRegressor(kernel=kernel,
                                      alpha=alpha, random_state=0)
        self.gp = gp

    def fit_gp(self, x, y):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        self.gp.fit(x, y)

    def predict_gp(self, x, return_std=True):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        return self.gp.predict(x, return_std=return_std)

    def run_random_simulation_overwrite_data(self, num_simulations=5):
        for i in range(num_simulations):
            params = self.gen_parameters()
            self.run_save_simulation(params)
        self.save_results()

    @staticmethod
    def plot_1d(x, y_pred, y_std, x_data=None, y_data=None, show_plot=True):
        plt.figure()
        plt.plot(x, y_pred, 'b-', label='Prediction')
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([y_pred - 3 * y_std,
                                 (y_pred + 3 * y_std)[::-1]]),
                 alpha=.5, fc='b', ec='None', label='95% confidence interval')
        if x_data is not None and y_data is not None:
            plt.plot(x_data, y_data, 'o')
        if show_plot:
            plt.show()


if __name__ == "__main__":
    # em = Emulator(
    #     model=epi.run_sir_model,
    #     parameters_range={
    #         'beta': {'value': [0.005, 0.00002], 'type': 'gamma'},
    #         'gamma': {'value': [1, .05], 'type': 'normal'},
    #         'initial_condition': {'value': [999, 1, 0], 'type': 'point'}
    #                       },
    #     name='epi_SIR_test'
    # )
    # for i in range(1):
    #     params = em.gen_parameters()
    #     print(params)
    #     em.run_model(params)
    #
    # params = em.gen_parameters()
    # em.run_save_simulation(params)
    # params = em.gen_parameters()
    # em.run_save_simulation(params)
    # em.save_results()
    # print(em.data)

    # em.run_random_simulation_overwrite_data()

    basic_emulation_plot_1d_test = True
    if basic_emulation_plot_1d_test:
        x_data_test = np.array([0,1, 1.5, 2, 3, 3.5, 6.5])
        y_data_test = np.array([1,3, 4, 5, -2, 3, 2])
        em = Emulator(None, None, None)
        em.set_gp(h=.25, alpha=.05)
        em.fit_gp(x_data_test, y_data_test)
        xv = np.arange(-1, 7, .01)
        yv, ystd = em.predict_gp(xv)
        em.plot_1d(xv, yv, ystd, x_data=x_data_test, y_data=y_data_test)
        print(em.gp.get_params())
        print(em.gp.log_marginal_likelihood_value_)
        em.optimise_gp(x_data_test, y_data_test)
        print(em.gp.log_marginal_likelihood_value_)

        print(em.gp.get_params())
        # em.plot_1d(xv, yv, ystd, x_data=x_data_test, y_data=y_data_test)