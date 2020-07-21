import models.epi_models_basic as epi
import random
import scipy.stats as stats
import numpy.random as np_rand


class Emulator:
    def __init__(self, model, parameters_range, name):
        self.model = model
        self.parameters_range = parameters_range
        self.name = name

    def gen_parameters(self):
        parameters = {}
        for key, attribute in self.parameters_range.items():
            type = attribute['type']
            if type == 'point':
                value = attribute['value']
            elif type == 'uniform':
                value_min, value_max = attribute['value']
                value = random.uniform(value_min, value_max)
            elif type == 'gamma':
                # k shape, theta scale
                # mean = k*theta
                # var = k theta**2
                # theta = var/mean
                # k = mean/theta
                value_mean, value_var = attribute['value']
                theta = value_var/value_mean
                k = value_mean/theta
                value = np_rand.gamma(shape=k, scale=theta)
            else:
                raise ValueError("Parameter input type {} is not valid"
                                 .format(type))
            parameters[key] = value
        return parameters

    def run_model(self, parameters):
        print(self.model(parameters))


if __name__ == "__main__":
    emulator_test = Emulator(
        model=epi.run_sir_model,
        parameters_range={
            'beta': {'value': [0.005, 0.00001], 'type': 'gamma'},
            'gamma': {'value': [0, 1], 'type': 'uniform'},
            'initial_condition': {'value': [999, 1, 0], 'type': 'point'}
                          },
        name='epi_SIR_test'
    )
    for i in range(3):
        params = emulator_test.gen_parameters()
        print(params)
        emulator_test.run_model(params)

