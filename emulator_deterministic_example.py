import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from numpy.random import random
from functools import lru_cache
import emulator
from models import epi_model_deterministic
import copy
import time


def define_dict_for_test_options_from_param_dict(input_dict, test_perc_opt=(0, 10, 20)):
    return tuple(
        {key: value if key != 'test_percentage' else current_test
         for key, value in input_dict.items()}
        for current_test in test_perc_opt)


def sample_and_est_utility(param_dict, n_samples=1001):
    h_est = em.predict_samples(
        np.array(em.dict_to_data_for_predict(param_dict)), num_samples=n_samples)
    n_tests = param_dict['test_percentage'] * param_dict['pop_size'] / 100
    return epi_model_deterministic.calc_utility(hospital=h_est, num_tests=n_tests)


def sample_utility_across_options(param_set_dict, test_perc_opt=(0, 10, 20), n_samples=1001):
    action_param_sets = define_dict_for_test_options_from_param_dict(param_set_dict, test_perc_opt=test_perc_opt)
    utility_samples = np.array([sample_and_est_utility(d, n_samples=n_samples) for d in action_param_sets]).transpose()
    return utility_samples


def estimate_probability_of_best_action(param_set_dict, test_perc_opt=(0, 10, 20), n_samples=1001):
    utility_samples = sample_utility_across_options(param_set_dict, test_perc_opt, n_samples=n_samples)
    best_option_list = []
    for row in utility_samples:
        best_option_list.append(np.where(row == min(row))[0][0])
    best_option_array = np.array(best_option_list)
    perc_each_option = [np.mean(best_option_array == i) for i in range(len(test_perc_opt))]
    return np.array(perc_each_option)


def estimate_best_action(param_set_dict, test_perc_opt=(0, 10, 20), n_samples=1001):
    action_probs = estimate_probability_of_best_action(param_set_dict, test_perc_opt, n_samples=n_samples)
    return np.where(action_probs == max(action_probs))[0][0]


def estimate_certainty(option_probabilities):
    options = [i for i in option_probabilities]
    options.sort(reverse=True)
    ideal_options = [0] * len(options)
    ideal_options[0] = 1

    options = np.array(options)
    options = options / sum(options)

    ideal_options = np.array(ideal_options)

    worst = np.sum(np.abs(ideal_options - np.array([1] * len(options)) / len(options)))

    certainty = 1 - np.sum(np.abs(options - ideal_options)) / worst
    return certainty


def determine_best_param_set_by_action_certainty(current_emulator, num_param_sets=100, num_gp_draws=1000):
    param_options = tuple(current_emulator.gen_parameters_from_priors() for _ in range(num_param_sets))
    param_action_probs = tuple(estimate_probability_of_best_action(data, n_samples=num_gp_draws) for data in param_options)
    param_certainty = tuple(estimate_certainty(p) for p in param_action_probs)
    best_param_choice = param_options[param_certainty.index(min(param_certainty))]
    return best_param_choice

def create_emulator_object(params, gp_save_name):
    em = emulator.DynamicEmulator(
        model=epi_model_deterministic.get_max_hospital,
        parameters_range=params,
        name=gp_save_name
    )
    return em

def get_gp_save_names(search_type):
    if search_type == 'random':
        return 'deterministic_example_random_gp_data'
    if search_type == 'gp_uncertainty':
        return 'deterministic_example_uncertainty_gp_data'
    if search_type == 'gp_action_certainty':
        return 'deterministic_example_action_certainty_gp_data'
    if search_type == 'test_data':
        return 'deterministic_example_test_data'
    raise NameError(f'Search type "{search_type}" not recognised')

def generate_gp_test_data(params, num_samples):
    em = create_emulator_object(params, gp_save_name=get_gp_save_names('test_data'))
    em.delete_existing_data(force_delete=True)
    for i in range(num_samples):
        print(i)
        params = em.gen_parameters_from_priors()
        em.run_model_add_results_to_data_frame(params)
        em.save_current_data_frame_to_csv()


def generate_gp_data_random_search(params, num_samples):
    em=create_emulator_object(params, gp_save_name=get_gp_save_names('random'))
    em.delete_existing_data(force_delete=True)
    for i in range(num_samples):
        print(i)
        params = em.gen_parameters_from_priors()
        # em.run_model(params)
        em.run_model_add_results_to_data_frame(params)
        em.save_current_data_frame_to_csv()


def generate_gp_data_uncertainty_search(params, num_samples):
    em=create_emulator_object(params, gp_save_name=get_gp_save_names('gp_uncertainty'))
    em.delete_existing_data(force_delete=True)
    em.explore_parameter_space_save_to_csv(number_model_runs=num_samples, mode='uncertainty')

def generate_gp_data_action_certainty_search(params, num_samples):
    em=create_emulator_object(params, gp_save_name=get_gp_save_names('gp_action_certainty'))
    em.delete_existing_data(force_delete=True)
    em.run_random_simulation_save_data(10)
    samples_to_run = num_samples - 10
    if samples_to_run <= 0:
        raise ValueError(f'Number of samples must be greater than 10.')
    for i in range(samples_to_run):
        print(i)
        best_params = determine_best_param_set_by_action_certainty(em)
        em.run_model_add_results_to_data_frame(best_params)
        em.save_current_data_frame_to_csv()


if __name__ == "__main__":
    parameter_range = {'pop_size': {'value': 1000, 'type': 'point'},
                          'init_infected': {'value': 25, 'type': 'point'},
                          'r0': {'value': [1, 3], 'type': 'uniform'},
                          'expected_recovery_time': {'value': [1, 14], 'type': 'uniform'},
                          'expected_incubation_time': {'value': [1, 5], 'type': 'uniform'},
                          'expected_time_to_hospital': {'value': [1, 14], 'type': 'uniform'},
                          'test_percentage': {'value': [0, 20], 'type': 'uniform'}
                          }
    em = emulator.DynamicEmulator(
        model=epi_model_deterministic.get_max_hospital,
        parameters_range=parameter_range,
        name='deterministic_SIR_random_sample'
    )

    for i in range(0):
        params = em.gen_parameters_from_priors()
        print(params)
        em.run_model(params)
        em.run_model_add_results_to_data_frame(params)
        em.save_current_data_frame_to_csv()

    em.optimise_gp_using_df_data()
    # data_test = {'pop_size': 1000, 'init_infected': 25, 'r0': 1.5744237738927043,
    #              'expected_recovery_time': 9.838055272628877,
    #              'expected_incubation_time': 4.1433165448651845,
    #              'expected_time_to_hospital': 11.350750481733268,
    #              'test_percentage': 13.362669656050768}
    # data_input = np.array(em.dict_to_data_for_predict(data_test))
    # em.predict_samples(data_input, 10)
    # hosp_est = np.array(em.predict_gp(data_input)[0])
    # epi_model_deterministic.calc_utility(hospital=hosp_est,
    #                                      num_tests=data_test['test_percentage'] * data_test['pop_size'] / 100)
    # define_dict_for_test_options_from_param_dict(data_test)
    # np.array(em.dict_to_data_for_predict(define_dict_for_test_options_from_param_dict(data_test)[0]))
    # epi_model_deterministic.calc_utility(
    #     hospital=em.predict_samples(
    #         np.array(em.dict_to_data_for_predict(define_dict_for_test_options_from_param_dict(data_test)[0])), 100),
    #     num_tests=define_dict_for_test_options_from_param_dict(data_test)[0]['test_percentage'] *
    #               define_dict_for_test_options_from_param_dict(data_test)[0]['pop_size'] / 100)
    #
    # action_probs = estimate_probability_of_best_action(data_test)
    # estimate_best_action(data_test)
    # print(estimate_certainty(action_probs))
    # ts = time.time()
    # param_options = tuple(em.gen_parameters_from_priors() for i in range(100))
    # param_action_probs = tuple(estimate_probability_of_best_action(data, n_samples=1000) for data in param_options)
    # param_certainty = tuple(estimate_certainty(p) for p in param_action_probs)
    # best_param_choice = param_options[param_certainty.index(min(param_certainty))]
    # te = time.time()
    # print(te - ts)


    # print(determine_best_param_set_by_action_certainty())
    data_samples = 500

    generate_random_data = False
    generate_uncertain_data = False
    generate_action_certainty_data = False
    generate_test_data = False

    if generate_random_data:
        generate_gp_data_random_search(params=parameter_range, num_samples=data_samples)
    if generate_uncertain_data:
        generate_gp_data_uncertainty_search(params=parameter_range, num_samples=data_samples)
    if generate_action_certainty_data:
        generate_gp_data_action_certainty_search(params=parameter_range, num_samples=data_samples)
    if generate_test_data:
        generate_gp_test_data(params=parameter_range, num_samples=10000)