import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import emulator
import models.epi_models_basic as epi


create_search_space_plot = True

if create_search_space_plot:
    rerun_simulations = True

    emulator_names = {'epi_SIR_test_random_search_example': 'random',
                      'epi_SIR_test_uncertainty_search_example': 'uncertainty'}

    emulators = {}
    for em_name in emulator_names:
        em = emulator.DynamicEmulator(
            model=epi.run_sir_model,
            parameters_range={
                'beta': {'value': [0.005, 0.00002], 'type': 'gamma'},
                'gamma': {'value': 1, 'type': 'point'},
                'initial_condition': {'value': [999, 1, 0], 'type': 'point'}
            },
            name=em_name
        )
        emulators[em_name] = em
    if rerun_simulations:
        for em_name, emulator in emulators.items():
            emulator.explore_parameter_space_save_to_csv(number_model_runs=50,
                                                         mode=emulator_names[em_name],
                                                         num_per_batch=1)


