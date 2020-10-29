import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from numpy.random import random
from functools import lru_cache
import emulator
from models import epi_model_deterministic
import copy

def define_dict_for_test_options_from_param_dict(dict, test_perc_opt=(0, 10, 20)):
    return tuple(
        {key: value if key != 'test_percentage' else current_test
         for key, value in dict.items()}
        for current_test in test_perc_opt)

def sample_and_est_utility(param_dict):
    h_est = em.predict_samples(
        np.array(em.dict_to_data_for_predict(param_dict)), 1001)
    n_tests = param_dict['test_percentage']*param_dict['pop_size']/100
    return epi_model_deterministic.calc_utility(hospital=h_est, num_tests=n_tests)

def sample_utility_across_options(param_set_dict, test_perc_opt=(0, 10, 20)):
    action_param_sets = define_dict_for_test_options_from_param_dict(param_set_dict, test_perc_opt=test_perc_opt)
    utility_samples = np.array([sample_and_est_utility(d) for d in action_param_sets]).transpose()
    return utility_samples

def estimate_probability_of_best_action(param_set_dict, test_perc_opt=(0, 10, 20)):
    utility_samples = sample_utility_across_options(param_set_dict, test_perc_opt)
    best_option_list = []
    for row in utility_samples:
        best_option_list.append(np.where(row == min(row))[0][0])
    best_option_array = np.array(best_option_list)
    perc_each_option = [np.mean(best_option_array==i) for i in range(len(test_perc_opt))]
    return np.array(perc_each_option)

def estimate_best_action(param_set_dict, test_perc_opt=(0, 10, 20)):
    action_probs = estimate_probability_of_best_action(param_set_dict, test_perc_opt)
    return np.where(action_probs == max(action_probs))[0][0]

def estimate_certainty(option_probabilities):
    options = [i for i in option_probabilities]
    options.sort(reverse=True)
    ideal_options = [0]*len(options)
    ideal_options[0] = 1

    options = np.array(options)
    options = options/sum(options)

    ideal_options = np.array(ideal_options)

    worst = np.sum(np.abs(ideal_options - np.array([1]*len(options))/len(options)))

    certainty = 1 - np.sum(np.abs(options - ideal_options))/worst
    return certainty

if __name__ == "__main__":
    em = emulator.DynamicEmulator(
        model=epi_model_deterministic.get_max_hospital,
        parameters_range = {'pop_size': {'value': 1000, 'type': 'point'},
                            'init_infected': {'value': 25, 'type': 'point'},
                            'r0': {'value': [1,3], 'type': 'uniform'},
                            'expected_recovery_time': {'value': [1,14], 'type': 'uniform'},
                            'expected_incubation_time': {'value': [1,5], 'type': 'uniform'},
                            'expected_time_to_hospital': {'value': [1,14], 'type': 'uniform'},
                            'test_percentage': {'value': [0,20], 'type': 'uniform'}
                            },
        name='deterministic_SIR_random_sample'
    )

    for i in range(0):
        params = em.gen_parameters_from_priors()
        print(params)
        em.run_model(params)
        em.run_model_add_results_to_data_frame(params)
        em.save_current_data_frame_to_csv()

    em.optimise_gp_using_df_data()
    data_test = {'pop_size': 1000, 'init_infected': 25, 'r0': 1.5744237738927043,
                 'expected_recovery_time': 9.838055272628877,
                 'expected_incubation_time': 4.1433165448651845,
                 'expected_time_to_hospital': 11.350750481733268,
                 'test_percentage': 13.362669656050768}
    data_input = np.array(em.dict_to_data_for_predict(data_test))
    em.predict_samples(data_input,10)
    hosp_est = np.array(em.predict_gp(data_input)[0])
    epi_model_deterministic.calc_utility(hospital=hosp_est,
                                         num_tests=data_test['test_percentage']*data_test['pop_size']/100)
    define_dict_for_test_options_from_param_dict(data_test)
    np.array(em.dict_to_data_for_predict(define_dict_for_test_options_from_param_dict(data_test)[0]))
    epi_model_deterministic.calc_utility(hospital=em.predict_samples(np.array(em.dict_to_data_for_predict(define_dict_for_test_options_from_param_dict(data_test)[0])),100),
                                         num_tests=define_dict_for_test_options_from_param_dict(data_test)[0]['test_percentage']*define_dict_for_test_options_from_param_dict(data_test)[0]['pop_size']/100)

    action_probs = estimate_probability_of_best_action(data_test)
    estimate_best_action(data_test)
    print(estimate_certainty(action_probs))

    param_optios = tuple(em.gen_parameters_from_priors() for i in range(100))
    param_action_probs = tuple(estimate_probability_of_best_action(data) for data in param_optios)
    param_certainty = tuple(estimate_certainty(p) for p in param_action_probs)
    best_param_choice = param_optios[param_certainty.index(min(param_certainty))]