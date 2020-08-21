import GPy
# GPy.plotting.change_plotting_library()
import numpy as np
X = np.random.uniform(-3.,3.,(20,1))
Y = np.sin(X) + np.random.randn(20,1)*0.05

kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
m = GPy.models.GPRegression(X,Y,kernel)

from IPython.display import display
display(m)

fig = m.plot()
GPy.plotting.show(fig)