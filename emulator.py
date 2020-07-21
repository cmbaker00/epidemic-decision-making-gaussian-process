import models.epi_models_basic as epi
import random

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
            if type == 'uniform':
                value_min, value_max = attribute['value']
                value = random.uniform(value_min, value_max)
            parameters[key] = value
        return parameters

    def run_model(self, parameters):
        print(self.model(parameters))


if __name__ == "__main__":
    emulator_test = Emulator(
        model=epi.run_sir_model,
        parameters_range={
            'beta': {'value': [0, 1], 'type': 'uniform'},
            'gamma': {'value': [0, 1], 'type': 'uniform'},
            'initial_condition': {'value': [999, 1, 0], 'type': 'point'}
                          },
        name='epi_SIR_test'
    )
    params = emulator_test.gen_parameters()
    print(params)
    emulator_test.run_model(params)

