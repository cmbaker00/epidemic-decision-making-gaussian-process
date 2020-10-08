import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import gpflow
import tensorflow as tf
from gpflow.utilities import print_summary
#
# X = np.array([0,1,2,3,4,5])
# Y = np.array([5,4,3,0,0,-2])
data = np.genfromtxt("data.csv", delimiter=",")
X = data[0:10, 0].reshape(-1, 1)
Y = data[0:10, 1].reshape(-1, 1)
# plt.plot(X,Y,"kx", mew=2)
# plt.show()


k = gpflow.kernels.Matern52() #+ gpflow.kernels.Linear()#
k = gpflow.kernels.SquaredExponential() #+ gpflow.kernels.Linear()#
meanf = gpflow.mean_functions.Constant()# + gpflow.mean_functions.Product(gpflow.mean_functions.Linear(), gpflow.mean_functions.Identity())
# meanf = gpflow.mean_functions.Identity()
m = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=meanf)

opt = gpflow.optimizers.Scipy()

opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
print_summary(m)

## generate test points for prediction
xx = np.linspace(-5.1, 3.5, 1000).reshape(1000, 1)  # test points must be of shape (N, D)

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
plt.xlim(-0.1, 3.5)
plt.show()