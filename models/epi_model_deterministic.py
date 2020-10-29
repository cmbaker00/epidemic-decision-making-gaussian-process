import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from numpy.random import random
from functools import lru_cache
from scipy.integrate import solve_ivp


class DiseaseDynamicsDeterministic:
    def __init__(self, pop_size, init_infected,
                 r0, expected_recovery_time, expected_incubation_time,
                 expected_time_to_hospital, test_percentage):
        gamma = 1 / expected_recovery_time
        beta = r0 * gamma
        incubation_pr = 1 / expected_incubation_time
        hosp_rate = 1 / expected_time_to_hospital

        self.total_pop = pop_size
        self.initial_infected = init_infected
        self.beta = beta
        self.gamma = gamma
        self.incubation_pr = incubation_pr
        self.hosp_rate = hosp_rate
        self.test_probability = test_percentage / 100

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

        population_state = [0] * self.total_pop
        for i in range(self.initial_infected):
            population_state[i] = self.states_index_dict['infected']
        self.population_state = np.array(population_state)

        self.population_initial_condition = np.array(
            [np.sum(self.population_state == i) for i in range(len(states))]) / self.total_pop

        # self.state_totals = self.calculate_state_totals(as_list = True)

    def ode_rhs(self, t, y):
        s, e, eq, i, iq, ih, r = y

        total_infected_for_transmission = i + .25 * (iq + ih)

        de_s = -self.beta * total_infected_for_transmission * s
        de_e = self.beta * total_infected_for_transmission * s \
               - self.test_probability * e - self.incubation_pr * e
        de_eq = self.test_probability * e - self.incubation_pr * eq
        de_i = self.incubation_pr * e - \
               self.test_probability * i - self.hosp_rate * i \
               - self.gamma * i
        de_iq = self.incubation_pr * eq + self.test_probability * i \
                - self.hosp_rate * iq - self.gamma * iq
        de_ih = self.hosp_rate * i + self.hosp_rate * iq \
                - self.gamma * ih / 2
        de_r = self.gamma * (i + iq + ih / 2)

        return de_s, de_e, de_eq, de_i, de_iq, de_ih, de_r

    def run_ode(self, timespan):
        sol = solve_ivp(lambda t, y: self.ode_rhs(t, y), timespan, self.population_initial_condition)
        return sol

def get_max_hospital_default(pop_size=1000,
                     init_infected=25,
                     r0=2,
                     expected_recovery_time=20,
                     expected_incubation_time=5,
                     expected_time_to_hospital=10,
                             test_percentage=.1):
    get_max_hospital(pop_size, init_infected,
                     r0, expected_recovery_time,
                     expected_incubation_time,
                     expected_time_to_hospital,
                     test_percentage)


def get_max_hospital(kwargs, stat='max_hospital'):
    epi_model = DiseaseDynamicsDeterministic(**kwargs)
    y = epi_model.run_ode([0, 365])
    return max(y.y[-2]*kwargs['pop_size']), stat


def get_utility_from_simulation(pop_size=1000,
                                init_infected=25,
                                r0=2,
                                expected_recovery_time=20,
                                expected_incubation_time=5,
                                expected_time_to_hospital=10,
                                test_percentage=.1):
    max_hospital = get_max_hospital(pop_size=pop_size,
                                    init_infected=init_infected,
                                    r0=r0,
                                    expected_recovery_time=expected_recovery_time,
                                    expected_incubation_time=expected_incubation_time,
                                    expected_time_to_hospital=expected_time_to_hospital,
                                    test_percentage=test_percentage)
    return calc_utility(max_hospital, pop_size*test_percentage/100)

def calc_utility(hospital, num_tests):
    number_of_tests = np.array(num_tests)
    max_hospital = np.array(hospital)
    if number_of_tests.shape == max_hospital.shape:
        pass
    else:
        if number_of_tests.ndim == 1 and max_hospital.ndim == 1:
            pass
        else:
            if number_of_tests.ndim == 2:
                number_of_tests = number_of_tests.transpose()
            if max_hospital.ndim == 2:
                number_of_tests = number_of_tests.transpose()
    return np.array(3500 * max_hospital + 100 * number_of_tests)

if __name__ == "__main__":

    make_simple_example_plot = True

    print(get_utility_from_simulation(test_percentage=0))
    print(get_utility_from_simulation(test_percentage=.05))
    print(get_utility_from_simulation(test_percentage=.1))

    if make_simple_example_plot:
        pop_size = 1000
        init_infected = 1
        r0 = 2
        expected_recovery_time = 5
        expected_incubation_time = 5
        expected_time_to_hospital = 15
        test_percentage = 100
        epi_model = DiseaseDynamicsDeterministic(pop_size=pop_size,
                                                 init_infected=init_infected,
                                                 r0=r0,
                                                 expected_recovery_time=expected_recovery_time,
                                                 expected_incubation_time=expected_incubation_time,
                                                 expected_time_to_hospital=expected_time_to_hospital,
                                                 test_percentage=test_percentage)
        y = epi_model.run_ode([0, 500])
        plt.plot(y.t, y.y[1:-1].transpose()*pop_size)
        plt.legend(['Exposed', 'Exposed quarantine',
                    'Infected', 'Infected quarantine',
                    'Infected hospital'])
        plt.show()
