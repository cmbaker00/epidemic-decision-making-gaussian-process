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
    emulator_names = {'epi_SIR_test_uncertainty_search_example': 'uncertainty'}

    # emulator_names = {'epi_SIR_test_random_search_example': 'random'}

    emulators = {}
    for em_name in emulator_names:
        em = emulator.DynamicEmulator(
            model=epi.run_sir_model,
            parameters_range={
                'beta': {'value': [0.5, 0.2], 'type': 'gamma'},
                'gamma': {'value': 1, 'type': 'point'},
                'initial_condition': {'value': [999, 1, 0], 'type': 'point'}
            },
            name=em_name
        )
        emulators[em_name] = em
    if rerun_simulations:
        for em_name, current_emulator in emulators.items():
            current_emulator.explore_parameter_space_save_to_csv(number_model_runs=50,
                                                         mode=emulator_names[em_name],
                                                         num_per_batch=1)

#TODO code to loop over different amoutns of data, show GP prediction change.
    xv = np.arange(0, 5, .0001)

    for em_name in emulator_names:
        em = emulator.DynamicEmulator(
            model=epi.run_sir_model,
            parameters_range={
                'beta': {'value': [0.5, 0.002], 'type': 'gamma'},
                'gamma': {'value': 1, 'type': 'point'},
                'initial_condition': {'value': [999, 1, 0], 'type': 'point'}
            },
            name=em_name
        )
        max_num_rows = 50
        for rows_to_use in range(5, max_num_rows):
            em.set_gp_parameters(dimension=1)
            em.optimise_gp_using_df_data(num_rows=rows_to_use)
            yv, ystd = em.predict_gp(np.reshape(xv, (-1, 1)))
            x_data = em.data['beta'][:rows_to_use]
            y_data = em.data['AR10'][:rows_to_use]
            em.plot_1d(xv, yv, ystd, x_data_plot=x_data, y_data_plot=y_data, show_plot=False)
            search_type = emulator_names[em_name]
            plt.title(f'{search_type}: num data points = {rows_to_use}')
            plt.savefig(f'figures/explore_plots/{search_type}{rows_to_use}.png')
            plt.close()
