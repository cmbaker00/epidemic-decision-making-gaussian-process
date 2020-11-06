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


def sample_and_est_utility(emulator_object, param_dict, n_samples=1001):
    h_est = emulator_object.predict_samples(
        np.array(emulator_object.dict_to_data_for_predict(param_dict)), num_samples=n_samples)
    n_tests = param_dict['test_percentage'] * param_dict['pop_size'] / 100
    return epi_model_deterministic.calc_utility(hospital=h_est, num_tests=n_tests)


def sample_utility_across_options(emulator_object, param_set_dict, test_perc_opt=(0, 10, 20), n_samples=1001):
    action_param_sets = define_dict_for_test_options_from_param_dict(param_set_dict, test_perc_opt=test_perc_opt)
    utility_samples = np.array([sample_and_est_utility(emulator_object, d, n_samples=n_samples)
                                for d in action_param_sets]).transpose()
    return utility_samples


def estimate_probability_of_best_action(emulator_object, param_set_dict, test_perc_opt=(0, 10, 20), n_samples=1001):
    utility_samples = sample_utility_across_options(emulator_object,
                                                    param_set_dict, test_perc_opt, n_samples=n_samples)
    best_option_list = []
    for row in utility_samples:
        best_option_list.append(np.where(row == min(row))[0][0])
    best_option_array = np.array(best_option_list)
    perc_each_option = [np.mean(best_option_array == i) for i in range(len(test_perc_opt))]
    return np.array(perc_each_option)


def estimate_best_action(emulator_object, param_set_dict, test_perc_opt=(0, 10, 20), n_samples=1001):
    action_probs = estimate_probability_of_best_action(emulator_object, param_set_dict,
                                                       test_perc_opt, n_samples=n_samples)
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
    param_action_probs = tuple(estimate_probability_of_best_action(emulator_object=current_emulator,
                                                                   param_set_dict=data,
                                                                   n_samples=num_gp_draws) for data in param_options)
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
    em = create_emulator_object(params, gp_save_name=get_gp_save_names('random'))
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
    em = create_emulator_object(params, gp_save_name=get_gp_save_names('gp_action_certainty'))
    em.delete_existing_data(force_delete=True)
    em.run_random_simulation_save_data(10)
    samples_to_run = num_samples - 10
    if samples_to_run <= 0:
        raise ValueError(f'Number of samples must be greater than 10.')
    for i in range(samples_to_run):
        em.optimise_gp_using_df_data()
        print(i)
        best_params = determine_best_param_set_by_action_certainty(current_emulator=em,
                                                                   num_param_sets=100,
                                                                   num_gp_draws=1001)
        em.run_model_add_results_to_data_frame(best_params)
        em.save_current_data_frame_to_csv()

def get_true_optimal_action(parameter_dict, test_options=(0, 10, 20)):

    utility_options = [epi_model_deterministic.get_utility_from_simulation_dict_input(parameter_dict,
                                                                                      test_percentage)
                       for test_percentage in test_options]
    return np.where(utility_options == min(utility_options))[0][0]


def prob_optimal_choice(em_object_list, test_param_set):
    true_optimal = get_true_optimal_action(test_param_set)
    pr_correct = []
    for test_emulator in em_object_list:
        pr_opts = estimate_probability_of_best_action(test_emulator, test_param_set)
        pr_correct.append(pr_opts[true_optimal])
    return pr_correct

def test_emulator_accuracy(em_object_list, test_data_object, num_test_points_max):
    data = test_data_object.data
    accuracy_list = []
    for i in range(len(data)):
        accuracy_list.append(prob_optimal_choice(em_object_list, data.iloc[i]))
        if i == num_test_points_max:
            break
    return np.mean(accuracy_list, axis=0)

def test_all_emulator_accuracy(param_range, save_name_keys, test_data_key,
                               num_training_data, num_test_data):
    em_object_list = []
    for em_name in save_name_keys:
        current_em = create_emulator_object(param_range, get_gp_save_names(em_name))
        current_em.data = current_em.data[:num_training_data]
        current_em.optimise_gp_using_df_data(num_rows=num_training_data)
        em_object_list.append(current_em)
    data_object = create_emulator_object(parameter_range, get_gp_save_names(test_data_key))
    return test_emulator_accuracy(em_object_list=em_object_list, test_data_object=data_object,
                           num_test_points_max=num_test_data)


if __name__ == "__main__":
    parameter_range = {'pop_size': {'value': 1000, 'type': 'point'},
                          'init_infected': {'value': 25, 'type': 'point'},
                          'r0': {'value': [1, 3], 'type': 'uniform'},
                          'expected_recovery_time': {'value': [1, 14], 'type': 'uniform'},
                          'expected_incubation_time': {'value': [1, 5], 'type': 'uniform'},
                          'expected_time_to_hospital': {'value': [1, 14], 'type': 'uniform'},
                          'test_percentage': {'value': [0, 20], 'type': 'uniform'}
                          }
    em_test = emulator.DynamicEmulator(
        model=epi_model_deterministic.get_max_hospital,
        parameters_range=parameter_range,
        name='deterministic_SIR_random_sample'
    )

    for i in range(0):
        params = em_test.gen_parameters_from_priors()
        print(params)
        em_test.run_model(params)
        em_test.run_model_add_results_to_data_frame(params)
        em_test.save_current_data_frame_to_csv()

    em_test.optimise_gp_using_df_data()
    # data_test = {'pop_size': 1000, 'init_infected': 25, 'r0': 1.5744237738927043,
    #              'expected_recovery_time': 9.838055272628877,
    #              'expected_incubation_time': 4.1433165448651845,
    #              'expected_time_to_hospital': 11.350750481733268,
    #              'test_percentage': 13.362669656050768}
    # data_input = np.array(em.dict_to_data_for_predict(data_test))
    # em_test.predict_samples(data_input, 10)
    # hosp_est = np.array(em_test.predict_gp(data_input)[0])
    # epi_model_deterministic.calc_utility(hospital=hosp_est,
    #                                      num_tests=data_test['test_percentage'] * data_test['pop_size'] / 100)
    # define_dict_for_test_options_from_param_dict(data_test)
    # np.array(em.dict_to_data_for_predict(define_dict_for_test_options_from_param_dict(data_test)[0]))
    # epi_model_deterministic.calc_utility(
    #     hospital=em_test.predict_samples(
    #         np.array(em_test.dict_to_data_for_predict(define_dict_for_test_options_from_param_dict(data_test)[0])), 100),
    #     num_tests=define_dict_for_test_options_from_param_dict(data_test)[0]['test_percentage'] *
    #               define_dict_for_test_options_from_param_dict(data_test)[0]['pop_size'] / 100)
    #
    # action_probs = estimate_probability_of_best_action(data_test)
    # estimate_best_action(data_test)
    # print(estimate_certainty(action_probs))
    # ts = time.time()
    # param_options = tuple(em_test.gen_parameters_from_priors() for i in range(100))
    # param_action_probs = tuple(estimate_probability_of_best_action(data, n_samples=1000) for data in param_options)
    # param_certainty = tuple(estimate_certainty(p) for p in param_action_probs)
    # best_param_choice = param_options[param_certainty.index(min(param_certainty))]
    # te = time.time()
    # print(te - ts)


    # print(determine_best_param_set_by_action_certainty())
    data_samples = 1000

    generate_random_data = False
    generate_uncertain_data = False
    generate_action_certainty_data = False
    generate_test_data = False

    run_initial_testing = False
    run_testing_by_emulator = True

    if generate_random_data:
        generate_gp_data_random_search(params=parameter_range, num_samples=data_samples)
    if generate_uncertain_data:
        generate_gp_data_uncertainty_search(params=parameter_range, num_samples=data_samples)
    if generate_action_certainty_data:
        generate_gp_data_action_certainty_search(params=parameter_range, num_samples=data_samples)
    if generate_test_data:
        generate_gp_test_data(params=parameter_range, num_samples=10000)

    if run_initial_testing:
        data_test_size_list = list(range(15,300,5))
        accuracy_list = []
        for test_size in data_test_size_list:
            print(f'Running test data size {test_size}')
            try:
                accuracy = test_all_emulator_accuracy(parameter_range,
                                           ['random', 'gp_uncertainty', 'gp_action_certainty'],
                                           'test_data',
                                           num_training_data=test_size,
                                           num_test_data=2000)
            except:
                accuracy = np.array([np.nan]*3)
            accuracy_list.append(accuracy)
        accuracy_array = np.array(accuracy_list)

        plt.plot(data_test_size_list, accuracy_array)
        plt.legend(['Random', 'Uncertain', 'Action certainty'])
        plt.xlabel('Training data')
        plt.ylabel('Test accuracy')
        plt.title('500 test data sets')
        plt.savefig('figures/test_set_accuracy.png')
        plt.show()

        new_data_num_list = []
        new_acc_list = []
        for num_data, acc in zip(data_test_size_list, accuracy_list):
            if any(np.isnan(acc) == True):
                pass
            else:
                new_data_num_list.append(num_data)
                new_acc_list.append(acc)
        new_acc_array = np.array(new_acc_list)

        plt.plot(new_data_num_list, new_acc_array)
        plt.legend(['Random', 'Uncertain', 'Action certainty'])
        plt.xlabel('Training data')
        plt.ylabel('Test accuracy')
        plt.title('500 test data sets')
        plt.savefig('figures/test_set_accuracy_removed_nan.png')
        plt.show()

    if run_testing_by_emulator:
        max_data = 300
        amount_of_training_data_to_test = tuple(range(15,max_data))
        gp_name_list = ['random', 'gp_uncertainty', 'gp_action_certainty']
        gp_accuracy_list_list = []
        gp_data_list_list = []
        for gp_name in gp_name_list:
            gp_training_data_list = []
            gp_accuracy_list = []
            for test_size in amount_of_training_data_to_test:
                print(f'{gp_name}, {test_size} data')
                try:
                    accuracy = test_all_emulator_accuracy(parameter_range,
                                                          [gp_name],
                                                          'test_data',
                                                          num_training_data=test_size,
                                                          num_test_data=50)
                except:
                    accuracy = np.array([0])
                if len(gp_accuracy_list) == 0:
                    gp_accuracy_list.append(accuracy)
                    gp_training_data_list.append(test_size)
                else:
                    previous_best_accuracy = gp_accuracy_list[-1][0]
                    if accuracy[0] >= previous_best_accuracy:
                        gp_accuracy_list.append(accuracy)
                        gp_training_data_list.append(test_size)
            gp_accuracy_list.append(gp_accuracy_list[-1])
            gp_training_data_list.append(max_data)
            gp_accuracy_list_list.append(gp_accuracy_list)
            gp_data_list_list.append(gp_training_data_list)

        for gp_name, acc_list, data_list in zip(gp_name_list,
                                                gp_accuracy_list_list,
                                                gp_data_list_list):
            plt.plot(data_list, acc_list)
        plt.xlabel('Training data')
        plt.ylabel('Test accuracy')
        plt.legend(gp_name_list)
        plt.savefig('figures/test_set_accuracy_removed_all_vals.png')

        plt.show()