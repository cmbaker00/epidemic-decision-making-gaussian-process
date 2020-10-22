import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from numpy.random import random
from functools import lru_cache
import emulator
from models import epi_model_example

test_emulator = True
if test_emulator:
    em = emulator.DynamicEmulator(
        model=epi_model_example.run_example_seir_model,
        name='SEIR_example',
        parameters_range={
            'pop_size': {'value': 1000, 'type': 'point'},
            'init_infected': {'value': 50, 'type': 'point'},
            'beta': {'value': .1, 'type': 'point'},
            'gamma': {'value': .05, 'type': 'point'},
            'incubation_pr': {'value': .2, 'type': 'point'},
            'hosp_rate': {'value': [.05, .2], 'type': 'uniform'},
            'test_percentage': {'value': [0, .2], 'type': 'uniform'}
        }
    )
    for i in range(0):
        print(i)
        params = em.gen_parameters_from_priors()
        print(params)
        em.run_model(params)
        em.run_model_add_results_to_data_frame(params)
        em.save_current_data_frame_to_csv()

    em.explore_parameter_space_save_to_csv(number_model_runs=0,
                                           mode='uncertainty',
                                           num_candidates=None,
                                           num_per_batch=1)

    em.set_gp_parameters(dimension=2)
    x_data = np.array(em.data[['test_percentage', 'hosp_rate']])
    y_data = np.array([em.data['max_hospital']], dtype='float64').transpose()
    em.change_data_optimise_gp(x_data, y_data)

    # xv = np.arange(0, 0.45, .01)
    # yv, ystd = em.predict_gp(np.reshape(xv, (-1, 1)))
    # em.plot_1d(xv, yv, ystd, x_data_plot=x_data, y_data_plot=y_data)

    test_options = np.array([0, .1, .2])
    hosp_prob = np.linspace(0.05, 0.2,50)
    optimal_decision_list = []
    decision_certainty_list = []
    for h in hosp_prob:
        predictor_values = np.array([[test_options[0], h],
                                     [test_options[1], h],
                                     [test_options[2], h]])
        current_predictions = em.predict_samples(predictor_values, 100000)
        cost_array = []
        for i in range(current_predictions.shape[1]):
            single_cost = epi_model_example.estimate_cost(test_options, current_predictions[:, i])
            cost_array.append(single_cost)
        best_option_list = []
        for costs in cost_array:
            best_option = np.where(min(costs) == costs)[0][0]
            best_option_list.append(best_option)
        best_option_array = np.array(best_option_list)
        option_confidence = [sum(best_option_array == i) for i in range(len(test_options))]
        optimal_decision = np.where(np.array(option_confidence) == max(option_confidence))[0][0]
        decision_certainty = 100*(max(option_confidence) - min(option_confidence))/sum(option_confidence)

        optimal_decision_list.append(optimal_decision)
        decision_certainty_list.append(decision_certainty)
    plt.plot(decision_certainty_list)
    plt.plot([i*50 for i in optimal_decision_list])
    plt.show()
    plt.close()



    # trials = 100
    # cost_array = []
    # for i in range(trials):
    #     ave_hosp = em.predict_samples(test_options, 1)
    #     single_cost = epi_model_example.estimate_cost(test_options, ave_hosp)
    #     cost_array.append(single_cost)
    # best_option_list = []
    # for costs in cost_array:
    #     best_option = np.where(min(costs) == costs)[0][0]
    #     best_option_list.append(best_option)
    # best_option_array = np.array(best_option_list)
    # option_confidence = [sum(best_option_array == i) for i in range(1+max(best_option_array))]
    # decision_certainty = max(option_confidence) - min(option_confidence)

    # y_sample = em.predict_samples(xv, num_samples=10)
    # plt.plot(xv, y_sample)
    # plt.show()