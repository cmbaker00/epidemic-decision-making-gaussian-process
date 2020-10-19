import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from numpy.random import random
from functools import lru_cache

class DiseaseDynamics:
    def __init__(self, pop_size, init_infected,
                 beta, gamma, incubation_pr, hosp_rate,
                 test_percentage):

        self.total_pop = pop_size
        self.initial_infected = init_infected
        self.beta = beta
        self.gamma = gamma
        self.incubation_pr = incubation_pr
        self.hosp_rate = hosp_rate
        self.test_percentage = test_percentage

        self.current_time = 0
        self.time_list = [self.current_time]

        states = ('susceptible', 'exposed',
                  'exposed_q', 'infected',
                  'infected_q', 'infected_h',
                  'recovered')

        self.state_names = states

        self.states_index_dict = {
            state: i for state, i in zip(states, count())
        }

        self.state_short_list = ('s', 'e',
                            'eq', 'i',
                            'iq', 'ih',
                            'r')

        population_state = [0]*self.total_pop
        for i in range(self.initial_infected):
            population_state[i] = self.states_index_dict['infected']
        self.population_state = np.array(population_state)

        self.state_totals = self.calculate_state_totals(as_list = True)

    def run_one_time_step(self):
        self.current_time += 1

        pop = self.population_state
        state_updates = np.zeros(pop.shape)

        s_index = self.states_index_dict['susceptible']

        # perc_s = sum(pop == self.states_index_dict['susceptible'])/self.total_pop
        num_i = sum(pop == self.states_index_dict['infected'])

        # new exposures
        pr_infected = self.beta*num_i/self.total_pop
        self.draw_and_update_state_transitions(
            pr_infected, 's', 'e', state_updates, stochastic=True)

        # exposed to infected
        self.draw_and_update_state_transitions(
            self.incubation_pr, 'e', 'i', state_updates)

        # exposed quarantine to infected quarantine
        self.draw_and_update_state_transitions(
            self.incubation_pr, 'eq', 'iq', state_updates)

        # infected to hospital
        self.draw_and_update_state_transitions(
            self.hosp_rate, 'i', 'ih', state_updates
        )

        # infected quarantine to hospital
        self.draw_and_update_state_transitions(
            self.hosp_rate, 'iq', 'ih', state_updates
        )

        # infected to recovered
        self.draw_and_update_state_transitions(
            self.gamma, 'i', 'r', state_updates
        )

        # infected quarantine to recovered
        self.draw_and_update_state_transitions(
            self.gamma, 'iq', 'r', state_updates
        )

        # infected hospital to recovered
        self.draw_and_update_state_transitions(
            self.gamma/2, 'ih', 'r', state_updates
        )

        # tested
        self.draw_and_update_state_transitions(
            self.test_percentage, 'e', 'eq', state_updates, stochastic=True
        )
        self.draw_and_update_state_transitions(
            self.test_percentage, 'i', 'iq', state_updates, stochastic=True
        )

        pop += state_updates.astype(int)
        self.population_state = pop

        self.append_population_state()

    def draw_and_update_state_transitions(self, pr_transition, current_state, new_state, current_state_update, stochastic=True):
        pop = self.population_state
        pop_in_state_flag = pop == self.state_index(current_state)
        # num_in_current_state = sum(pop_in_state_flag)
        state_full_name = self.short_to_long_state(current_state) if len(current_state) < 3 else current_state
        num_in_current_state = self.state_totals[state_full_name][-1]
        expected_transmissions = pr_transition*num_in_current_state
        if expected_transmissions < 5 and stochastic:
            state_transitions = np.array(random(num_in_current_state) < pr_transition)
        else:
            deterministic_transmissions = int(np.round(expected_transmissions))
            state_transitions = np.zeros(num_in_current_state, dtype=bool)
            state_transitions[0:deterministic_transmissions] = True

        transitions = np.zeros(self.total_pop, dtype=bool)
        transitions[pop_in_state_flag] = state_transitions
        current_state_update[transitions] = self.state_increment(current_state, new_state)
        return current_state_update

    def calculate_state_totals(self, as_list=False):
        state_totals = [sum(self.population_state == self.state_index(state_name)) for state_name in self.state_names]
        if as_list:
            state_totals = [[total] for total in state_totals]
        total_dict = {state_name: total for state_name, total in zip(self.state_names, state_totals)}
        total_dict['time'] = [self.current_time] if as_list else self.current_time
        return total_dict

    def append_population_state(self):
        old_totals = self.state_totals
        new_totals = self.calculate_state_totals()
        for state_name in new_totals.keys():
            old_totals[state_name].append(new_totals[state_name])

    def state_increment(self, from_state, to_state):
        init_index = self.state_index(from_state)
        final_index = self.state_index(to_state)
        return final_index - init_index

    def state_index(self, current_state):
        if len(current_state) < 4:
            current_state = self.short_to_long_state(current_state)
        return self.states_index_dict[current_state]

    @lru_cache()
    def short_to_long_state(self, state_name):
        short_to_long_dict = {short_name: long_name for short_name, long_name in
                              zip(self.state_short_list, self.states_index_dict.keys())}
        return short_to_long_dict[state_name]

def estimate_max_hopsital_rep(pop_size=1000,
                              init_infected=25,
                              beta=.1,
                              gamma=.05,
                              incubation_pr=.2,
                              hosp_rate=.01,
                              test_percentage=.001,
                              reps=10):
    max_hospital = [get_max_hospital_single_simulation(pop_size=pop_size,
                                init_infected=init_infected,
                                beta=beta,
                                gamma=gamma,
                                incubation_pr=incubation_pr,
                                hosp_rate=hosp_rate,
                                test_percentage=test_percentage)
                    for i in range(reps)]
    return np.mean(max_hospital)

def get_max_hospital_single_simulation(pop_size=1000,
                          init_infected=25,
                          beta=.1,
                          gamma=.05,
                          incubation_pr=.2,
                          hosp_rate=.01,
                          test_percentage=.001):

    epi_model = DiseaseDynamics(pop_size=pop_size,
                                init_infected=init_infected,
                                beta=beta,
                                gamma=gamma,
                                incubation_pr=incubation_pr,
                                hosp_rate=hosp_rate,
                                test_percentage=test_percentage)

    flag_passed_max_infected = False
    flag_passed_max_hospital = False
    while flag_passed_max_hospital == False:
        epi_model.run_one_time_step()
        infected = np.array(epi_model.state_totals['infected']) + \
                           np.array(epi_model.state_totals['infected_q']) + \
                           np.array(epi_model.state_totals['infected_h'])
        if infected[-1] <= max(infected)/2:
            flag_passed_max_infected = True
        if flag_passed_max_infected:
            hospital = np.array(epi_model.state_totals['infected_h'])
            if hospital[-1] <= max(hospital)/2:
                return max(hospital)



if __name__ == "__main__":
    make_simple_example_plot = False
    test_estimate_max_hospital = True

    if test_estimate_max_hospital:
        print(estimate_max_hopsital_rep(reps=3, test_percentage=0))

    if make_simple_example_plot:

        epi = DiseaseDynamics(pop_size=1000,
                              init_infected=50,
                              beta=.1,
                              gamma=.05,
                              incubation_pr=.2,
                              hosp_rate=.01,
                              test_percentage=.008)
        for i in range(500):
            print(i)
            epi.run_one_time_step()

        print(epi.state_totals)
        print(epi.calculate_state_totals())
        plt.plot(epi.state_totals['infected_h'])
        plt.plot(epi.state_totals['infected'])
        plt.plot(epi.state_totals['infected_q'])
        plt.legend(['Hospital active cases',
                    'Mild active cases',
                    'Quarantine active cases'])
        plt.show()