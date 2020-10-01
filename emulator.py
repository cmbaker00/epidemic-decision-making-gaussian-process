import models.epi_models_basic as epi
import random
import numpy.random as np_rand
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import itertools
from functools import lru_cache

import time
import pyDOE

import gpflow
import tensorflow as tf
from gpflow.utilities import print_summary

class DynamicEmulator:
    def __init__(self, model, parameters_range, name):
        self.model = model
        self.parameters_range = parameters_range
        self.name = name
        self.data = self.load_previous_data()

        self.gp = None
        self.kernal = None
        self.meanf = None

    def data_file(self):
        return 'data/{}.csv'.format(self.name)

    def delete_existing_data(self):
        print(f'This will delete the file: {self.data_file()}')
        x = input('Proceed (Y/N)')
        if x.lower() == 'y':
            os.remove(self.data_file())
            self.load_previous_data()
            print('Deleted')
        else:
            print('Cancelled')

    def load_previous_data(self):
        data_file_name = self.data_file()
        file_exists = os.path.isfile(data_file_name)
        if file_exists:
            return pd.read_csv(data_file_name)
        else:
            print('Data file does is not found: {}'
                  .format(data_file_name))
            print('Creating new data structure')
            return None

    def gen_parameters_from_priors(self):
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
                raise ValueError(f"Parameter input type {distribution_type} is not valid")
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

    def run_model_add_results_to_data_frame(self, parameters):
        result = self.run_model(parameters)
        self.add_results_to_data_frame(parameters, result)

    def save_data_to_csv(self):
        if type(self.data) is pd.DataFrame:
            self.data.to_csv(self.data_file(), index=False)
        elif self.data is None:
            raise ValueError('There is no data to store')

    def set_gp_parameters(self, dimension=1):
        ker = gpflow.kernels.Matern52()
        meanf = None#gpflow.mean_functions.Constant(0)
        self.kernal = ker
        self.meanf = meanf

    def fit_gp(self, x, y):
        m = gpflow.models.GPR(data=(x, y), kernel=self.kernal, mean_function=self.meanf)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
        self.gp = m

    def predict_gp(self, x, return_std=True):
        # if len(x.shape) == 1:
        #     x = x.reshape(-1, 1)
        vals, vars = self.gp.predict_f(x)
        std = np.sqrt(vars)
        return vals, std

    def run_random_simulation_save_data(self, num_simulations=5):
        for i in range(num_simulations):
            params = self.gen_parameters_from_priors()
            self.run_model_add_results_to_data_frame(params)
        self.save_data_to_csv()

    @staticmethod
    def plot_1d(x, y_pred, y_std, x_data=None, y_data=None, show_plot=True):
        plt.figure()
        plt.plot(x, y_pred, 'b-', label='Prediction')
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([y_pred - 1.96*y_std,
                                 (y_pred + 1.96*y_std)[::-1]]),
                 alpha=.15, fc='b', ec='None', label='95% confidence interval')
        if x_data is not None and y_data is not None:
            plt.plot(x_data, y_data, 'o')
        if show_plot:
            plt.show()


if __name__ == "__main__":
    em = DynamicEmulator(
        model=epi.run_sir_model,
        parameters_range={
            'beta': {'value': [0.005, 0.00002], 'type': 'gamma'},
            'gamma': {'value': 1, 'type': 'point'},
            'initial_condition': {'value': [999, 1, 0], 'type': 'point'}
                          },
        name='epi_SIR_test'
    )
    for i in range(6):
        params = em.gen_parameters_from_priors()
        print(params)
        em.run_model(params)
        em.run_model_add_results_to_data_frame(params)
        em.save_data_to_csv()
    #
    # params = em.gen_parameters()
    # em.run_save_simulation(params)
    # params = em.gen_parameters()
    # em.run_save_simulation(params)
    #
    # print(em.data)
    em.set_gp_parameters(dimension=1)
    # em.fit_gp(np.array(em.data[['beta','gamma']]), em.data['AR10'])
    x_data = np.array([em.data['beta']]).transpose()
    y_data = np.array([em.data['AR10']]).transpose()
    em.fit_gp(x_data, y_data)

    xv = np.arange(min(x_data)*.95, max(x_data)*1.05, .0001)
    yv, ystd = em.predict_gp(np.reshape(xv, (-1, 1)))
    em.plot_1d(xv, yv, ystd, x_data=x_data, y_data=y_data)

    # em.run_random_simulation_overwrite_data()

    basic_emulation_plot_1d_test = False
    if basic_emulation_plot_1d_test:

        X1 = np.random.uniform(-3., 3., (20, 1))
        X2 = np.random.uniform(8., 15., (20, 1))
        Y1 = np.sin(X1 - 5) + np.random.randn(20, 1) * .25 + 1
        Y2 = np.sin(X2) + np.random.randn(20, 1) * 0.05

        X = np.r_[X1, X2]
        Y = np.r_[Y1, Y2]

        x_data_test = X
        y_data_test = Y


        x_data_test = np.array([[1,2,3,4,5,6,7,8,9,10,18]]).transpose() - 5
        y_data_test = np.array([[1,2,4,2,3,4,4,5,7,8,1]]).transpose()

        # x_data_test = np.array([[1]]).transpose() - 5
        # y_data_test = np.array([[1]]).transpose()

        em = DynamicEmulator(None, None, None)
        em.set_gp_parameters()
        em.fit_gp(x_data_test, y_data_test)
        xv = np.arange(-10, 17, .01)
        yv, ystd = em.predict_gp(np.reshape(xv,(-1,1)))
        em.plot_1d(xv, yv, ystd, x_data=x_data_test, y_data=y_data_test)


        # em.plot_1d(xv, yv, ystd, x_data=x_data_test, y_data=y_data_test)