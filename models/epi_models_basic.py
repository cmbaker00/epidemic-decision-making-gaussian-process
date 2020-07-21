import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class BasicSIR:
    """
    A class that defines the standard SIR ODE system.
    s' = -beta*s*i
    i' = beta*s*i - gamma*i
    r' = gamma*i
    """


    def __init__(self, beta, gamma, initial_condition):
        """
        :param beta: transmission rate, float > 0
        :param gamma: recovery rate, float > 0
        :param initial_condition: a list of length three, with the initial
        number of [susceptible, infected, recovered].
        """
        self.beta = beta
        self.gamma = gamma
        self.initial_condition = initial_condition

    def ode_system(self, y, t):
        susceptible, infected, recovered = y
        dydt = [-self.beta*susceptible*infected,
                self.beta*susceptible*infected - self.gamma*infected,
                self.gamma*infected]
        return dydt

    def solve_system(self, t_end, t_steps=10, return_full_timeseries=False):
        t = np.linspace(0, t_end, t_steps)
        sol = odeint(self.ode_system, self.initial_condition, t)
        if return_full_timeseries:
            return t, sol
        else:
            return sol[-1]

    def generate_epidemic_stats(self, stat='AR10'):
        if stat == 'AR10':
            s, i, r = self.solve_system(10)
            return r/(s + i + r)


def run_sir_model(params, stat='AR10'):
    model = BasicSIR(**params)
    res = model.generate_epidemic_stats(stat=stat)
    return res, stat


if __name__ == "__main__":
    ode = BasicSIR(beta=.005, gamma=.2, initial_condition=[1000, 1, 0])
    print(ode.solve_system(10))
    tvals, solution = ode.solve_system(t_end=10, t_steps=1000, return_full_timeseries=True)
    plt.plot(tvals, solution)
    plt.show()

    print(ode.generate_epidemic_stats(stat='AR10'))

    sir_params = {'beta': .005,
                  'gamma': .2,
                  'initial_condition': [1000, 1, 0]}
    print(run_sir_model(sir_params))