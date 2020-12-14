import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import gpflow
import tensorflow as tf
from gpflow.utilities import print_summary
#
# X = np.array([0,1,2,3,4,5])
# Y = np.array([5,4,3,0,0,-2])
# data = np.genfromtxt("data/epi_SIR_test_random_search_example.csv", delimiter=",")
data = pd.read_csv("../data/epi_SIR_test_random_search_example.csv")

rows = 10
X = np.array(data['beta']).reshape(-1,1)[0:rows]*50
X[5] = X[3]
Y = np.array(data['AR10']).reshape(-1,1)[0:rows]

k = gpflow.kernels.SquaredExponential() #+ gpflow.kernels.Linear()#
meanf = gpflow.mean_functions.Constant()# + gpflow.mean_functions.Product(gpflow.mean_functions.Linear(), gpflow.mean_functions.Identity())
# meanf = gpflow.mean_functions.Identity()
meanf = None
m = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=meanf)


opt = gpflow.optimizers.Scipy()

opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
print_summary(m)

xmin = 0
xmax = 0.03*50
## generate test points for prediction
xx = np.linspace(xmin, xmax, 1000).reshape(1000, 1)  # test points must be of shape (N, D)

## predict mean and variance of latent GP at test points
mean, var = m.predict_f(xx)

## generate 10 samples from posterior
tf.random.set_seed(1)  # for reproducibility
samples = m.predict_f_samples(xx, 10)  # shape (10, 100, 1)

## plot
plt.figure(figsize=(12, 6))
plt.plot(X, Y, "kx", mew=2)
plt.plot(xx, mean, "C0", lw=2)
plt.fill_between(
    xx[:, 0],
    mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
    mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
    color="C0",
    alpha=0.2,
)

plt.plot(xx, samples[:, :, 0].numpy().T, "C0", linewidth=0.5)
plt.xlim(xmin-.5, xmax)
plt.show()