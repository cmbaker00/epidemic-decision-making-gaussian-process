import models.epi_models_basic as epi

class Emulator:
    def __init__(self, model, parameters, name):
        pass


if __name__ == "__main__":
    emulator_test = Emulator(epi.BasicSIR,
                             {'beta':[0, 1],
                              'gamma':[0, 1],
                              'initial_condition': [999, 1, 0]},
                             'epi_SIR_test'
    )

