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
import GPy
import time

class Emulator:
    def __init__(self, model, parameters_range, name):
        self.model = model
        self.parameters_range = parameters_range
        self.name = name
        self.data = self.load_previous_data()

        self.gp = None
        self.kernal = None


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

    def set_gp(self, dimension=1):
        # ker = GPy.kern.Matern52(2,ARD=True) + GPy.kern.White(2)
        ker = GPy.kern.RBF(input_dim=dimension) \
              + GPy.kern.White(input_dim=dimension) \
            + GPy.kern.Matern52(input_dim=dimension)
        # ker = GPy.kern.src.spline.Spline(input_dim=1)
        self.kernal = ker

    def fit_gp(self, x, y):
        time1 = time.time()
        m = GPy.models.GPRegression(x, y, self.kernal)
        time2 = time.time()
        m.optimize()
        time3 = time.time()
        print(f"Regression time = {time2-time1}")
        print(f"Optimisation time = {time3-time2}")
        self.gp = m


    def predict_gp(self, x, return_std=True):
        # if len(x.shape) == 1:
        #     x = x.reshape(-1, 1)
        return self.gp.predict(x)

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
                 np.concatenate([y_pred - y_std,
                                 (y_pred + y_std)[::-1]]),
                 alpha=.15, fc='b', ec='None', label='95% confidence interval')
        if x_data is not None and y_data is not None:
            plt.plot(x_data, y_data, 'o')
        if show_plot:
            plt.show()


if __name__ == "__main__":
    em = Emulator(
        model=epi.run_sir_model,
        parameters_range={
            'beta': {'value': [0.005, 0.00002], 'type': 'gamma'},
            'gamma': {'value': 1, 'type': 'point'},
            'initial_condition': {'value': [999, 1, 0], 'type': 'point'}
                          },
        name='epi_SIR_test'
    )
    for i in range(100):
        params = em.gen_parameters()
        print(params)
        em.run_model(params)
        em.run_save_simulation(params)
        em.save_results()
    #
    # params = em.gen_parameters()
    # em.run_save_simulation(params)
    # params = em.gen_parameters()
    # em.run_save_simulation(params)
    #
    # print(em.data)
    em.set_gp(dimension=1)
    # em.fit_gp(np.array(em.data[['beta','gamma']]), em.data['AR10'])
    x_data = np.array([em.data['beta']]).transpose()
    y_data = np.array([em.data['AR10']]).transpose()
    em.fit_gp(x_data, y_data)

    xv = np.arange(0, 0.015, .0001)
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

        em = Emulator(None, None, None)
        em.set_gp(dimension=1)
        em.fit_gp(x_data_test, y_data_test)
        xv = np.arange(-10, 17, .01)
        yv, ystd = em.predict_gp(np.reshape(xv,(-1,1)))
        em.plot_1d(xv, yv, ystd, x_data=x_data_test, y_data=y_data_test)


        # em.plot_1d(xv, yv, ystd, x_data=x_data_test, y_data=y_data_test)